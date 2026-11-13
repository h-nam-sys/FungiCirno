# Workstream B — Le Thien Khang: tracking and growth analytics

## Ownership

Validate and repair FungiCirno's sequence handling, temporal cluster tracking, growth measurements, calibration and environmental association. The primary input is the trained Charisis cluster model. Do not build a second tracker.

## Fixed inputs and concepts

- Primary input: `charisis_cluster_tracking` predictions.
- Whole-cluster masks and exact labels: `immature_cluster`, `mature_cluster`, `overstayed_cluster` with class IDs 1, 2, 3 (0 is background).
- Charisis neural network predicts a cluster instance and its stage; it does **not** assign persistent IDs.
- FungiCirno's application tracker associates predictions across captures using mask overlap, predicted centroid proximity and area similarity with one-to-one Hungarian assignment.
- Notebook's own tracker is a separate implementation. Its 213 predicted tracks are not the app's 22 tracks and are not ground-truth IDs.
- Per-image spatial grouping is for individual fruitbody predictions only. It must not run on Charisis whole-cluster masks.
- Individual-fruitbody maturity (`not-ready`, `ready`, `overdue`) is optional and deferred unless the lead confirms it is required.

## Required work

1. Verify that each capture series uses one growing unit and fixed camera/view, and is sorted by capture timestamp. Test missing timestamps, duplicate timestamps, empty captures and sequences imported out of order.
2. Inspect the application tracker in `src/mushroom_monitor/tracking.py`. Validate predicted centroids, mask IoU, area-ratio scoring, one-to-one Hungarian assignment, bounded missed-frame gaps, stable track keys and rebuild idempotency.
3. Test crowded clusters, overlaps, new pins, occlusions, disappearance/reappearance, partial harvest and cluster split/merge cases.
4. Confirm each track observation links to its source prediction and retains per-capture stage, confidence, timestamp, source-image area and growth-vs-first value.
5. Build a manually checked identity-reference subset if the source dataset has no usable persistent track IDs. Report ID switches, IDF1/HOTA/MOTA only with actual ground-truth identities and a documented metric implementation.
6. Compare selected image-plane growth curves with manual mask-area measurements. Distinguish actual biological growth from viewpoint, occlusion and segmentation changes.
7. Keep `area_mm2` disabled unless camera-specific calibration and geometry are valid. Pixel area is not volume, biomass or physical surface area.
8. Validate environmental CSV timestamp matching and missing/malformed readings. Interpret relationships descriptively, not causally.
9. Validate optional individual-maturity grouping only if that task is approved and an individual-fruitbody checkpoint exists.

## Owned paths

- `src/mushroom_monitor/tracking.py`
- `src/mushroom_monitor/maturity.py` (individual-fruitbody grouping only)
- `src/mushroom_monitor/environment.py`
- analytics tests and documentation

Do not change the Charisis model class map, inference contract, DB schema or app entry point without Hai Nam's approval.

## Return evidence

Return sequence manifest, test cases, overlays, per-track observation CSV, manually verified reference IDs and identity metrics (if available), growth-measurement comparison, calibration method, environment CSV checks, exact test commands/results, commit hash and known limitations.
