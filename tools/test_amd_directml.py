"""Strict AMD DirectML smoke test for this app's Mask R-CNN operator path.

Run in the app virtual environment: python tools/test_amd_directml.py
The script never moves tensors to CPU to make an operator pass.
"""
import platform
import sys

if platform.system() != "Windows":
    raise SystemExit("This check targets native Windows DirectML.")

import torch
import torchvision
import torch_directml
from torchvision.ops import nms, roi_align

DML = torch_directml.device()
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("DirectML device:", DML)

# Verify DirectML tensor compute.
x = torch.randn((1, 8, 32, 32), device=DML)
conv = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1).to(DML).eval()
with torch.no_grad():
    y = conv(x)
print("Conv2d on DirectML:", tuple(y.shape), "device=", y.device)
assert str(y.device).startswith("privateuseone"), f"Unexpected device: {y.device}"

# Mask R-CNN requires ROIAlign and NMS. Test these explicitly before loading checkpoints.
rois = torch.tensor([[0, 2, 2, 24, 24]], dtype=torch.float32, device=DML)
roi = roi_align(y, rois, output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)
print("torchvision ROIAlign on DirectML:", tuple(roi.shape), "device=", roi.device)
assert str(roi.device).startswith("privateuseone"), f"ROIAlign left DirectML: {roi.device}"

boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32, device=DML)
scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device=DML)
keep = nms(boxes, scores, 0.5)
print("torchvision NMS on DirectML:", keep.tolist(), "device=", keep.device)

print("PASS: DirectML tensor, ROIAlign and NMS checks completed.")
print("Next: run the app with Compute backend = AMD and test one image per model.")
