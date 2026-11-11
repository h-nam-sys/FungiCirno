FungiCirno TorchVision NMS registration hotfix

Replace:
  src/mushroom_monitor/inference/runtime.py

Then fully stop and restart Streamlit. Re-open Run inference and retry a capture.

Reason:
The MushR TorchScript graph references torchvision::nms. TorchVision's C++ extension
must be imported before torch.jit.load() so the operator is registered in the process.
This patch does not enable CPU fallback and does not alter model weights or preprocessing.
