# Workstream A — Minh Thinh: model/data/runtime verification

## Ownership

Validate and repair the existing YOLO-seg model lifecycle. Do not design a new model registry, new inference backend, or separate application.

## Fixed tasks and taxonomies

- `charisis_cluster_tracking`: whole oyster cluster segmentation, classes in this exact order: `immature`, `well-defined`, `overstayed`. This is the primary model for growth-stage history and persistent tracking.
- `mushr_maturity`: individual oyster fruitbody segmentation, classes: `not-ready`, `ready`, `overdue`. This is a separate maturity task.
- `m18k_source_wb`: WB-only source model, class `WB`. It is not a target oyster classifier.

Both production models use the same `ultralytics_yolo_seg` runtime and normalized Prediction record. Do not merge class maps or compare metrics across task taxonomies.

## What to verify and deliver

1. Download/inspect the public Charisis oyster cluster dataset (DOI `10.5281/zenodo.15041243`) and the annotated MushR dataset linked from `https://github.com/ETCE-LAB/MushR`. Record licences/terms, annotation format, class names, sequence/camera metadata, counts and hashes. Do not redistribute raw data without permission.
2. Convert Charisis COCO masks using `tools/prepare_coco_yolo_dataset.py` if the source annotations match the converter's supported COCO format. Inspect random exported labels, all three classes and any fragmented masks. If MushR is not COCO-compatible, make a small dataset-specific converter and document the annotation semantics; do not force an incompatible format through the generic converter.
3. Establish train/validation/test splits by complete trial/sequence/farm group. Never split adjacent time-lapse frames randomly. Preserve the final test set untouched.
4. Train Model A twice on identical splits: once initialized from the team's M18K WB `best.pt`, once from COCO-pretrained `yolov8n-seg.pt`. The M18K head must adapt to the three target classes; inspect training logs for transferred/reinitialized layers. Evaluate both on the same Charisis holdout and select by target-domain results, not M18K metrics.
5. Train Model B as a separate MushR maturity checkpoint. Do not use the Charisis model as its baseline.
6. Package each candidate with `tools/train_candidate.py` or `tools/package_yolo_model.py`. Verify `model_key`, `prediction_role`, `taxonomy`, zero-based class mapping, checkpoint hash and evaluation metadata.
7. Verify reviewed-data export is separated by model family/taxonomy. Run one complete review -> export -> Colab fine-tune -> holdout evaluate -> candidate install -> activation gate -> rollback cycle when target-labelled data are available.
8. Run the actual installed checkpoints through the local app on CPU. Then test AMD DirectML on the RX 5500 XT. Record unsupported operators/errors precisely; never call a smoke test full inference.
9. Record peak process RSS, GPU memory if available, per-image latency, masks visually inspected, package versions and checkpoint hashes.

## Files you own

- `src/mushroom_monitor/inference/`
- `src/mushroom_monitor/model_registry.py`
- `src/mushroom_monitor/learning.py`
- `tools/train_candidate.py`
- `tools/prepare_reviewed_dataset.py`
- `tools/prepare_coco_yolo_dataset.py` (dataset conversion fixes)
- `tools/package_yolo_model.py`
- model/training documentation and Colab notebook updates

## Definition of done

- Both target-domain model bundles are reproducibly packaged with the correct independent taxonomies, or a specific public-data/annotation blocker is returned.
- CPU end-to-end inference succeeds for each available real checkpoint.
- AMD DirectML end-to-end inference succeeds with evidence or has a reproducible documented blocker.
- The M18K-vs-COCO initialization comparison for Model A is completed on the same untouched target holdout.
- The model-specific human-review and candidate promotion/rollback workflow is exercised.

## Guardrails

Do not change SQLite schema, app navigation or shared prediction contracts without Hai Nam's approval. Never use WB source metrics as oyster evidence. Do not use adjacent time-lapse frames across splits. Do not promote without an independent target holdout.

## Return package

Branch `member/minh-thinh-ml`; changed files and commit hash; source/dataset URLs and terms; split manifest; class maps; checkpoint SHA-256; training commands/configs; holdout metrics; sample overlays; CPU/AMD logs; package versions; known issues.
