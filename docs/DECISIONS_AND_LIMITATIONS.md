# Architecture decisions and claim limits

## Current decisions

1. **Primary detector:** the team's M18K V2 YOLOv8n-seg checkpoint is the starting point. It is a trained team checkpoint, not an official pretrained M18K model.
2. **Target domain:** the intended application is fixed-camera side-view monitoring of mushroom growth. M18K's White Button training results are source-domain results only. Transfer to side-view oyster imagery requires labelled target-domain adaptation and independent evaluation.
3. **Tracking reference:** Charisis et al. inform mask-based association, missed-detection handling, post-processing and growth monitoring. The local implementation uses its own motion-assisted Hungarian assignment and is not an exact reproduction.
4. **Maturity:** the WB-only model recognizes WB, not developmental maturity. Maturity labels are only available when a separate model is trained with appropriate stage labels.
5. **Compute:** model training is performed outside the local app (e.g. Colab). The local app supports explicit AMD DirectML and CPU execution. A failed DirectML operation must not be silently reported as GPU inference; users can select CPU.
6. **Persistence:** SQLite is the record of truth for captures, model versions, inference runs, predictions, tracks, observations, reviews and environmental readings.
7. **Learning lifecycle:** human-reviewed data are exported for offline fine-tuning. New checkpoints enter as candidates and require evaluation/explicit activation. The active model remains installed for rollback.

## Measurement limits

- `area_px2` is projected two-dimensional mask area in source-image pixels.
- `area_mm2` is computed only using an entered camera-specific `mm_per_px`. A single scale is valid only for a fixed camera geometry/reference plane; perspective and depth changes can invalidate it.
- A relative area change is not biomass, volume, mass, or a full 3D growth estimate.
- Tracking similarity scores are heuristics, not probabilities.
- DBSCAN-equivalent connected grouping with `min_samples=1` groups all detections, including isolated objects. It does not estimate biological flush identity by itself.
- Nearest environmental readings are time-aligned descriptors; correlations are not evidence of causation.

## Evaluation limits

- M18K WB-only metrics cannot be reported as side-view oyster-mushroom accuracy.
- A validation split used during fine-tuning is not an untouched test set.
- Adjacent time-lapse images must not be randomly split between training and test. Prefer sequence-/farm-level separation.
- High-confidence model predictions are not ground truth. Human review should inspect both image and mask.
- A candidate should be evaluated against the active model on the same independent holdout set. A candidate failing the gate must not silently replace the active model.
- Unit tests and mocked inference do not establish full-model AMD DirectML support or biological validity.

## Still requiring real-world/member verification

- Ultralytics execution on the RX 5500 XT through DirectML (operator coverage varies).
- Full-model CPU inference and RAM use under the 12 GB constraint.
- Side-view M18K transfer-learning metrics and sequence/farm-level generalization.
- Tracking identity accuracy on annotated multi-capture sequences.
- Physical calibration validity for the actual camera and cultivation geometry.
- Maturity estimation once suitable labels/checkpoints exist.
- Actual incremental retraining and regression/rollback run end-to-end.
