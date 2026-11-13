# Coding-AI prompt — Le Thien Khang: Charisis tracking and growth analytics

Validate the existing application tracker and growth analysis using the trained Charisis cluster model. Do not create a second tracker or change the model taxonomy.

## Fixed facts

- Primary input: `charisis_cluster_tracking` whole-cluster masks.
- Class IDs/names: 1=`immature_cluster`, 2=`mature_cluster`, 3=`overstayed_cluster`; 0 is background.
- Charisis model predicts per-image instances/classes. FungiCirno's application tracker assigns persistent IDs.
- Application tracking is distinct from the notebook's own predicted tracks (213 in the full notebook).
- Charisis whole-cluster detections must not be passed through the individual-fruitbody spatial grouping path.
- Individual maturity (`not-ready`, `ready`, `overdue`) is optional and deferred unless the lead approves it.

## Your tasks

1. Inspect `src/mushroom_monitor/tracking.py`, `service.py`, database persistence and Growth tracking UI.
2. Validate fixed-camera series grouping, chronological order, missing timestamps, empty captures and model-version scoping.
3. Verify mask IoU, predicted centroid proximity, area-ratio score, one-to-one Hungarian assignment and bounded missed-capture gaps.
4. Test dense clusters, overlaps, new pins, partial occlusions, disappearance/reappearance, cluster merge/split and partial harvesting.
5. Confirm every track observation links to its source prediction and preserves timestamp, class/stage, confidence, source-image mask area and growth vs first observation.
6. Build a manually checked tracking identity subset if the source dataset does not provide usable track IDs. Report identity switches, fragmentation, IDF1/HOTA/MOTA only with actual ground-truth persistent IDs and a documented implementation.
7. Compare image-plane growth measurements against manually reviewed masks. Do not call projected 2D mask-area increase biological mass/volume growth.
8. Keep physical units disabled unless camera-specific calibration is valid. Validate environmental CSV alignment and describe associations as descriptive, not causal.
9. Validate grouping only if a separate individual-fruitbody model is approved and installed.

## Ownership

- `src/mushroom_monitor/tracking.py`
- `src/mushroom_monitor/maturity.py` (optional individual grouping only)
- `src/mushroom_monitor/environment.py`
- analytics tests/docs

Do not change shared DB schema, app entrypoint, checkpoint class IDs or inference contracts without Hai Nam's approval.

## Return evidence

Sequence manifest, visual overlays, per-track observation export, manually verified reference IDs, identity metrics if valid, growth-error comparison, calibration evidence, environment tests, exact commands/results, commit hash and remaining risks.
