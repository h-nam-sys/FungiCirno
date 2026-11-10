from __future__ import annotations

from collections import OrderedDict
from typing import Any


def _torch_modules():
    import torch
    import torchvision
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import FeaturePyramidNetwork
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
    return torch, torchvision, MaskRCNN, FastRCNNPredictor, MaskRCNNPredictor, FeaturePyramidNetwork, LastLevelMaxPool


def build_charisis_model(num_classes: int, side: int, config: dict[str, Any]):
    torch, torchvision, MaskRCNN, _, _, FeaturePyramidNetwork, LastLevelMaxPool = _torch_modules()

    class ConvNeXtTinyFPN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            features = torchvision.models.convnext_tiny(weights=None).features
            self.stem = features[0]
            self.stage1 = features[1]
            self.down2 = features[2]
            self.stage2 = features[3]
            self.down3 = features[4]
            self.stage3 = features[5]
            self.down4 = features[6]
            self.stage4 = features[7]
            self.fpn = FeaturePyramidNetwork([96, 192, 384, 768], 256, extra_blocks=LastLevelMaxPool())
            self.out_channels = 256

        def forward(self, x):
            x = self.stem(x)
            c2 = self.stage1(x)
            c3 = self.stage2(self.down2(c2))
            c4 = self.stage3(self.down3(c3))
            c5 = self.stage4(self.down4(c4))
            return self.fpn(OrderedDict([("0", c2), ("1", c3), ("2", c4), ("3", c5)]))

    model = MaskRCNN(
        ConvNeXtTinyFPN(),
        num_classes=num_classes,
        min_size=side,
        max_size=side,
        box_detections_per_img=int(config.get("box_detections_per_img", 100)),
        box_score_thresh=float(config.get("inference_score_floor", config.get("model_score_floor", 0.01))),
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return model


def build_mushr_model(num_classes: int, side: int, config: dict[str, Any]):
    torch, torchvision, _, FastRCNNPredictor, MaskRCNNPredictor, _, _ = _torch_modules()
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=side,
        max_size=side,
        box_detections_per_img=int(config.get("box_detections_per_img", 100)),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.roi_heads.score_thresh = float(config.get("model_score_floor", 0.01))
    return model


def build_model(package, side: int):
    architecture = package.architecture.lower()
    class_count = len(package.classes) + 1  # background + foreground labels
    if "convnext" in architecture:
        model = build_charisis_model(class_count, side, package.config)
    elif "resnet-50" in architecture or "resnet50" in architecture:
        model = build_mushr_model(class_count, side, package.config)
    else:
        raise ValueError(f"Unsupported model architecture: {package.architecture}")

    import torch
    payload = torch.load(package.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a dictionary exported by the training notebook")
    state = payload.get("model") or payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a 'model' state dictionary")
    # Handle DataParallel wrappers without masking genuine architecture mismatches.
    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    # The application supplies an already resized image; fix the internal transform
    # to that same side so the 20 MP source image is never expanded back to full size.
    model.transform.min_size = (int(side),)
    model.transform.max_size = int(side)
    model.eval()
    return model
