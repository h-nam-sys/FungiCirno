# FungiCirno implementation release notes

## Architecture finalized

- One production inference implementation: Ultralytics YOLOv8-seg.
- Primary Model A: `charisis_cluster_tracking`, oyster cluster stage segmentation (`immature`, `well-defined`, `overstayed`) plus persistent temporal tracking and stage history.
- Model B: `mushr_maturity`, individual oyster fruitbody maturity segmentation (`not-ready`, `ready`, `overdue`) plus spatial/flush aggregation.
- M18K WB-only checkpoint is retained as a source initialization candidate and integration model, not the deployed oyster maturity model.
- Legacy TorchVision model loader remains for old bundle compatibility only.

## Application and workflow changes

- Added a split-preserving COCO instance-segmentation to YOLO-seg converter, including mask decoding and dataset manifest output.
- Candidate training now accepts model family, role, species and taxonomy; it rejects mismatched target class order and same-taxonomy baseline comparisons.
- Candidate evaluation uses an untouched supplied holdout; initial M18K-to-oyster transfer does not compare against WB source AP.
- Model packaging now requires explicit model key/species/taxonomy/role and validates the production class order.
- Separated the low inference `score_floor` from the final automatic-analysis confidence threshold so uncertain predictions can remain available to human review.
- Growth UI displays per-capture stage label and confidence from the linked prediction.
- Inference UI uses each model's `track_objects` metadata rather than assuming every segmentation model should be tracked.
- Acceptance checklist and team handoffs now follow the finalized two-task architecture.

## Evidence status

The code and training workflows are included. No Charisis/MushR trained target checkpoints are bundled. Full target-domain model metrics, CPU/AMD execution on actual target checkpoints, tracking identity metrics, calibration evidence and a complete reviewed-learning/rollback cycle remain to be produced from real datasets and target hardware. No such results are fabricated in this release.
