Optional legacy runtime patch — MushR TorchScript / TorchVision NMS

This patch note applies only to the optional older MushR TorchScript path, if that
extension is reintroduced. It is not required by the primary Charisis model.

The current core Charisis checkpoint is a regular TorchVision Mask R-CNN state
dictionary loaded by `src/mushroom_monitor/inference/runtime.py` and
`src/mushroom_monitor/inference/models.py`. Do not replace the current runtime
with an old patch unless a reproducible traceback demonstrates that the active
code needs it.

Historical issue:
A MushR TorchScript graph referenced `torchvision::nms`; TorchVision's C++
extension had to be imported before `torch.jit.load()` so the operator was
registered. That workaround did not change weights or enable CPU fallback.
