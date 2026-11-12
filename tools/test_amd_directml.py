#!/usr/bin/env python
"""Run an actual YOLOv8-seg checkpoint through AMD DirectML end to end.

This is a full-model smoke test, not an accuracy evaluation. It has no CPU fallback.
Use a target oyster checkpoint for production acceptance; M18K WB weights only test
basic YOLO runtime compatibility.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cv2
import psutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="YOLOv8-seg .pt checkpoint")
    parser.add_argument("--image", required=True, help="Representative image with visible target objects")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--out", default="results/directml")
    args = parser.parse_args()

    if platform.system() != "Windows":
        raise SystemExit("AMD DirectML requires native Windows; use CPU on this operating system.")
    weights = Path(args.weights).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    import torch_directml
    from ultralytics import YOLO

    device = torch_directml.device()
    model = YOLO(str(weights), task="segment")
    model.model.to(device)
    model.model.eval()
    parameter_device = str(next(model.model.parameters()).device)
    if not parameter_device.startswith("privateuseone"):
        raise RuntimeError(f"Model parameters are not on DirectML: {parameter_device}")

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 * 1024)
    started = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        retina_masks=True,
        device=device,
        verbose=False,
    )[0]
    elapsed = time.perf_counter() - started
    rss_after = process.memory_info().rss / (1024 * 1024)

    if result.boxes is None or result.masks is None:
        raise RuntimeError(
            "The YOLO model executed but returned no instance masks for this image. "
            "Choose a representative image with clearly visible target objects and retry."
        )
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    masks = result.masks.data.detach().cpu().numpy()
    if len(boxes) == 0 or len(masks) == 0:
        raise RuntimeError("No instance masks were produced; full segmentation path was not verified.")

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    overlay = result.plot()
    cv2.imwrite(str(output / "overlay.jpg"), overlay)
    summary = {
        "weights": str(weights),
        "image": str(image_path),
        "device": str(device),
        "model_parameter_device": parameter_device,
        "imgsz": args.imgsz,
        "confidence_floor": args.conf,
        "nms_iou": args.iou,
        "detections": int(len(boxes)),
        "mask_count": int(len(masks)),
        "class_ids": classes.tolist(),
        "confidences": [float(value) for value in scores],
        "elapsed_seconds": elapsed,
        "ms_per_image": elapsed * 1000,
        "process_rss_before_mb": rss_before,
        "process_rss_after_mb": rss_after,
        "status": "full_yolo_seg_directml_inference_passed",
        "note": "Smoke/compatibility test only; it is not a labelled accuracy evaluation.",
    }
    (output / "directml_test.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Overlay: {output / 'overlay.jpg'}")


if __name__ == "__main__":
    main()
