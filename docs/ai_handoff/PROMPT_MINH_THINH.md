# Coding-AI prompt — Minh Thinh: Charisis model/runtime owner

You are validating the primary FungiCirno cluster-growth model. Inspect the real repository and run artifacts before editing. Do not redesign the model system or convert the Charisis model to YOLO.

## Fixed facts

- Primary model key: `charisis_cluster_tracking`.
- Architecture: TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN.
- Runtime: `torchvision_state_dict`.
- Foreground classes and IDs: 1=`immature_cluster`, 2=`mature_cluster`, 3=`overstayed_cluster`; 0 is background. Preserve exact names and IDs.
- Model output: per-image whole-cluster boxes, masks, stage labels and confidence.
- FungiCirno's application tracker assigns persistent identities after inference. The model itself does not output persistent track IDs.
- The training notebook's own tracker is separate from the application tracker.
- The optional MushR model is a distinct individual-fruitbody task, only if lead-approved.
- M18K is a historical one-class White Button YOLO experiment. It is not a Charisis initialization because the architectures/checkpoints are incompatible.

## Existing evidence

Read `docs/MVP_VALIDATION.md` and inspect:
- `notebooks/Charisis_Full_Training.ipynb`
- `src/mushroom_monitor/inference/models.py`
- `src/mushroom_monitor/inference/runtime.py`
- `src/mushroom_monitor/model_registry.py`
- `src/mushroom_monitor/service.py`

The attached notebook completed 10 epochs and reports held-out mask AP@[.50:.95]=0.650, AP50=0.782 and AP75=0.759 on 1,055 images from sequences 1 and 3. The supplied SQLite snapshot contains 15 successful AMD DirectML Charisis inference runs. The `.pth` binary is not inside the source ZIP; obtain `model_bundle.zip` from Kaggle output.

## Your tasks

1. Verify the actual Kaggle `model_bundle.zip`: checkpoint hash, config, class mapping, state-dict keys and evaluation metrics. Do not assume notebook output filenames are present without inspecting the bundle.
2. Install the bundle into a clean checkout through Models. Confirm it canonicalizes to `charisis_cluster_tracking`.
3. Reproduce CPU inference with the actual checkpoint. Verify mask/box dimensions, class IDs, confidence and persisted source-image area.
4. Reproduce AMD DirectML inference on the target RX 5500 XT when available. Record exact package versions, backend, latency, memory and overlays; CPU remains mandatory.
5. Compare the notebook builder with the app builder. Strict state-dict loading must pass without silently dropping/mapping parameters.
6. Fix only demonstrated defects in your owned model/runtime files. Add focused regression tests.
7. Inspect human-reviewed COCO export compatibility with the Charisis notebook. Do not use the YOLO exporter or `tools/train_candidate.py` for this TorchVision checkpoint.
8. Do not train optional MushR unless Hai Nam confirms individual harvest readiness is a required deliverable.

## Ownership

- `src/mushroom_monitor/inference/`
- `src/mushroom_monitor/model_registry.py`
- `src/mushroom_monitor/learning.py`
- Charisis notebook and model docs
- Optional Ultralytics YOLO tooling only when explicitly requested

## Return evidence

Checkpoint bundle/hash, config/class map, package versions, clean-install logs, CPU/AMD logs, sample overlays, timing/RSS, commands, tests and their exact results, commit hash, and known limitations. Never claim tracking accuracy from the notebook's 213 predicted tracks.
