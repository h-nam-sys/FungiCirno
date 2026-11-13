# Workstream A — Minh Thinh: Charisis model lifecycle and runtime

## Ownership

Own the primary Charisis model's checkpoint, packaging, model loading, inference validation, and offline model lifecycle. The trained member checkpoint is TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN. Do not rewrite it as YOLO or create another inference stack.

## Fixed model decisions

- Primary family: `charisis_cluster_tracking`.
- Architecture: TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN.
- Runtime: `torchvision_state_dict`.
- Detection unit: whole *Pleurotus ostreatus* cluster.
- Preserve exact foreground mapping: `1=immature_cluster`, `2=mature_cluster`, `3=overstayed_cluster`; class 0 is background.
- Capabilities: `track_objects=true`, `spatial_grouping=false`.
- Legacy installed DB alias `legacy_charisis_maskrcnn` remains recognized.
- `mushr_maturity` is an optional separate individual-fruitbody task, only if the lead approves it.
- M18K WB-only YOLO is a historical White Button benchmark. It is not a Charisis initializer because YOLO and TorchVision Mask R-CNN checkpoints are architecturally incompatible.

## Evidence already available

- The executed Charisis notebook completed 10 epochs on a Kaggle Tesla T4.
- Dataset: 6,382 images and 37,004 annotations; train/validation sequences 2 and 4, test sequences 1 and 3; one-hour sampling.
- Split sizes: 433 train, 109 validation, 1,055 held-out test.
- Mask AP@[.50:.95]=0.650, AP50=0.782, AP75=0.759; final train loss 0.1449, validation loss 0.1649.
- Notebook test inference: 9,483 detections and 213 notebook-generated predicted tracks. No GT identity metric is available.
- Supplied DB snapshot: 15 successful Charisis AMD DirectML runs. The checkpoint bytes are not included in the project ZIP; DB path points to the original Windows checkout.

## Required work

1. Inspect `notebooks/Charisis_Full_Training.ipynb`, `src/mushroom_monitor/inference/models.py`, `runtime.py`, `model_registry.py`, and the exact bundle metadata.
2. Obtain `model_bundle.zip` from the Kaggle output. Verify its checkpoint SHA-256, state-dictionary key layout, class IDs/names, architecture config and any evaluation metrics.
3. Install the bundle into a clean project checkout through **Models**. Confirm it canonicalizes to `charisis_cluster_tracking` and has `runtime_type=torchvision_state_dict`.
4. Run a representative capture on CPU. Confirm class IDs 1/2/3, boxes, mask dimensions, confidence, source-image scaling and persisted pixel area.
5. Run on AMD DirectML on the target RX 5500 XT. Record runtime, peak process memory, backend/device and actual overlay. CPU remains mandatory. Do not infer full-model compatibility from an operator-only smoke test.
6. Inspect the loader's strict state-dictionary compatibility against the executed notebook. Fix only demonstrated discrepancies and add regression tests.
7. Keep the Charisis notebook's COCO training/evaluation split by complete sequences. Do not randomly split adjacent time-lapse frames.
8. The application's reviewed-sample export is COCO-oriented. Verify its class map and masks before proposing a Charisis fine-tuning path. `tools/train_candidate.py` and `tools/package_yolo_model.py` are YOLO-only; do not use them to train/package the primary `.pth` model.
9. Only work on MushR if the lead confirms per-fruitbody harvest readiness. It is a different task and must have its own checkpoint and class map.

## Owned paths

- `src/mushroom_monitor/inference/`
- `src/mushroom_monitor/model_registry.py`
- `src/mushroom_monitor/learning.py`
- `notebooks/Charisis_Full_Training.ipynb`
- model bundle/packaging documentation
- Optional YOLO tools only when explicitly scoped

## Do not

- Replace Charisis with M18K or claim M18K predicts oyster maturity.
- Rename `mature_cluster` to `well-defined` without a documented taxonomy migration and new evaluation.
- Force `.pth` into the Ultralytics `.pt` loader.
- Use notebook-generated tracks as ground-truth tracking metrics.
- Change shared DB schema or app entrypoint without Hai Nam's approval.
- Claim a workflow passed because code or a notebook cell exists.

## Return package

Provide the model bundle outside Git, checkpoint SHA-256, exact package versions, run manifest, held-out metrics, overlay examples, CPU/AMD logs, inference timing/RSS, tests run, commit hash and known limitations.
