from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..image_io import load_rgb, resize_for_inference
from ..model_registry import package_for_model_row
from .models import build_model


@dataclass
class Prediction:
    detection_index: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # coordinates in resized inference image
    mask: np.ndarray  # uint8 mask at resized inference dimensions


@dataclass
class DeviceInfo:
    label: str
    device: Any
    backend: str
    note: str


def select_device(preference: str = "auto") -> DeviceInfo:
    import torch
    pref = preference.lower()
    if pref == "cpu":
        return DeviceInfo("CPU", torch.device("cpu"), "cpu", "CPU inference selected")
    if torch.cuda.is_available():
        # torch.cuda includes ROCm builds; the label is deliberately generic.
        name = torch.cuda.get_device_name(0)
        return DeviceInfo(name, torch.device("cuda:0"), "torch_cuda_or_rocm", "PyTorch accelerator detected")
    if pref in {"auto", "amd", "directml"} and platform.system() == "Windows":
        try:
            import torch_directml
            dml = torch_directml.device()
            return DeviceInfo("DirectML device 0", dml, "directml", "DirectML detected; first inference validates operator support")
        except Exception as exc:
            if pref in {"amd", "directml"}:
                return DeviceInfo("CPU", torch.device("cpu"), "cpu", f"DirectML unavailable ({exc}); CPU fallback")
    return DeviceInfo("CPU", torch.device("cpu"), "cpu", "No supported PyTorch GPU runtime detected; CPU fallback")


class InferenceEngine:
    def __init__(self, model_row: dict[str, Any], inference_side: int = 768,
                 device_preference: str = "auto", score_threshold: float | None = None,
                 mask_threshold: float = 0.5):
        self.model_row = model_row
        self.package = package_for_model_row(model_row)
        self.inference_side = int(inference_side)
        self.device_info = select_device(device_preference)
        # Keep candidate-level detections. The final application threshold is applied
        # after inference so low-confidence instances remain available for human review.
        self.score_threshold = float(self.package.config.get(
            "inference_score_floor", self.package.config.get("model_score_floor", 0.01)))
        self.mask_threshold = float(mask_threshold)
        self.model = None
        self.load_error: str | None = None
        self.load_model()

    def load_model(self) -> None:
        import torch
        try:
            self.model = build_model(self.package, self.inference_side).to(self.device_info.device).eval()
        except Exception as exc:
            if self.device_info.backend in {"cpu", "cpu_fallback"}:
                self.load_error = f"Could not load {self.package.architecture}: {type(exc).__name__}: {exc}"
                raise RuntimeError(self.load_error) from exc
            # Some TorchVision detection operators may not load on a given GPU backend.
            # Rebuild on CPU from the same checkpoint rather than leaving the model unusable.
            self.device_info = DeviceInfo("CPU fallback", torch.device("cpu"), "cpu_fallback",
                                          f"GPU model load failed ({type(exc).__name__}: {exc}); loaded on CPU")
            try:
                self.model = build_model(self.package, self.inference_side).to(torch.device("cpu")).eval()
            except Exception as cpu_exc:
                self.load_error = f"GPU load failed ({exc}); CPU load failed ({type(cpu_exc).__name__}: {cpu_exc})"
                raise RuntimeError(self.load_error) from cpu_exc
        if self.device_info.backend in {"cpu", "cpu_fallback"}:
            # Keep CPU threads bounded on the target 12 GB / 4-core-class desktop.
            torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) - 2)))

    def _predict_on_current_device(self, image: Image.Image) -> tuple[list[Prediction], tuple[int, int], float]:
        import torch
        from torchvision.transforms.functional import pil_to_tensor
        resized, scale_x, scale_y = resize_for_inference(image, self.inference_side)
        tensor = pil_to_tensor(resized).to(dtype=torch.float32, device=self.device_info.device).div_(255.0)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model([tensor])[0]
        elapsed = time.perf_counter() - started
        boxes = output["boxes"].detach().to("cpu").numpy()
        labels = output["labels"].detach().to("cpu").numpy()
        scores = output["scores"].detach().to("cpu").numpy()
        masks = output["masks"].detach().to("cpu").numpy()
        class_map = self.package.classes
        predictions: list[Prediction] = []
        for idx, (box, label, score, mask_prob) in enumerate(zip(boxes, labels, scores, masks)):
            label_i = int(label)
            score_f = float(score)
            if label_i not in class_map or score_f < self.score_threshold:
                continue
            mask = (mask_prob[0] >= self.mask_threshold).astype(np.uint8)
            if int(mask.sum()) == 0:
                continue
            predictions.append(Prediction(
                detection_index=idx,
                class_id=label_i,
                class_name=class_map[label_i],
                confidence=score_f,
                bbox=tuple(float(x) for x in box.tolist()),
                mask=mask,
            ))
        # Free per-image tensors before returning to the service layer.
        del tensor, output, boxes, labels, scores, masks
        if self.device_info.backend in {"torch_cuda_or_rocm"}:
            torch.cuda.empty_cache()
        return predictions, resized.size, elapsed

    def predict_path(self, image_path: Path) -> tuple[list[Prediction], Image.Image, dict[str, Any]]:
        image = load_rgb(image_path)
        return self.predict_image(image)

    def predict_image(self, image: Image.Image) -> tuple[list[Prediction], Image.Image, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        # DirectML may not implement every TorchVision custom operator for a given
        # build. Validate on actual inference and automatically retry on CPU once.
        try:
            predictions, size, seconds = self._predict_on_current_device(image)
        except Exception as exc:
            if self.device_info.backend in {"cpu", "cpu_fallback"}:
                raise
            import torch
            # GPU OOM and unsupported detection ops both fall back to CPU.
            self.model = self.model.to(torch.device("cpu")).eval()
            self.device_info = DeviceInfo("CPU fallback", torch.device("cpu"), "cpu_fallback",
                                          f"{self.device_info.label} inference failed ({type(exc).__name__}: {exc}); retried on CPU")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) - 2)))
            predictions, size, seconds = self._predict_on_current_device(image)
        original_size = image.size
        info = {
            "device": self.device_info.label,
            "backend": self.device_info.backend,
            "device_note": self.device_info.note,
            "inference_seconds": round(seconds, 4),
            "original_width": original_size[0],
            "original_height": original_size[1],
            "inference_width": size[0],
            "inference_height": size[1],
            "scale_x": original_size[0] / size[0],
            "scale_y": original_size[1] / size[1],
        }
        return predictions, image, info
