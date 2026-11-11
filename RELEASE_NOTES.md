# FungiCirno integrated implementation handoff

## Included

- Streamlit cultivation and capture-management application.
- SQLite schema creation and additive migration for model-specific grouping records.
- YOLOv8-seg model bundle import and inference adapter.
- Legacy TorchVision bundle compatibility.
- Explicit CPU and AMD DirectML execution selection.
- Per-image predictions, mask persistence, review queue and corrected-mask support.
- Reviewed YOLO-seg dataset export for offline fine-tuning.
- Candidate fine-tuning/evaluation/package script.
- Candidate promotion gates, model versioning and rollback path.
- Adaptive per-image spatial grouping.
- Mask/centroid/area-based temporal tracking with Hungarian assignment and missed-frame tolerance.
- Growth history and environmental CSV alignment.
- Member handoffs, integration contracts and end-to-end acceptance checklist.

## Not included

- Trained M18K `best.pt` weights or candidate weights.
- Raw M18K, side-view or other datasets.
- Previous developer's SQLite database, local capture files or mask files.
- A validated side-view oyster maturity classifier.
- Live sensor driver/hardware integration.

## Validation status

This package is an integrated implementation handoff for member testing and correction. It has not been run end-to-end on the target AMD PC in this session. DirectML operator coverage for Ultralytics YOLO, CPU RAM behavior under the team's real checkpoint, sequence-level tracking accuracy and candidate promotion/rollback must be verified by the team. Do not report those as completed until evidence is collected.
