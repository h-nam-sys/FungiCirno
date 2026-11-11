# Workstream B — Le Thien Khang

## Ownership

**Temporal growth analytics: grouping, mask-based tracking, measurements, camera calibration and environmental alignment.** You are the subsystem validator and bug-fix owner. The integrated implementation is already in this release; do not replace the application or build another tracker in a separate project.

## Files you own

- `src/mushroom_monitor/tracking.py`
- `src/mushroom_monitor/maturity.py`
- `src/mushroom_monitor/environment.py`
- tracking/analytics docs and focused visualizations

## Required work

1. Use a fixed-camera capture series with multiple timestamped images and predictions from the same active model version.
2. Verify the Hungarian association uses mask IoU, predicted centroid proximity and area ratio, with one-to-one assignment and a bounded missed-frame gap.
3. Verify track rebuilding is scoped by `(unit_id, camera_id, series_key, model_version_id)` and does not delete another model's histories.
4. Inspect masks, centroids and matches on crowded/overlapping frames. Record identity switches, new tracks, missed matches and fragmented tracks.
5. Check growth curves using source-image pixel area. Confirm `mm^2` remains blank unless a valid camera calibration has been entered.
6. Validate grouping outputs are scoped to capture + model version and are not confused with persistent track identities.
7. Import environmental CSV files; verify timestamp parsing, ranges, duplicate handling and nearest-reading matching within 12 hours.
8. Keep environmental comparisons descriptive. Do not claim that temperature/humidity/CO2 caused growth changes.
9. Fix bugs within your modules. Ask Hai Nam before changing shared schema or app contracts.

## Definition of done

- Multi-capture sequence produces explainable track IDs and growth history.
- Missed captures/temporary occlusion do not instantly destroy a track.
- Distinct model versions retain separate tracking/group records.
- Camera calibration is explicit and physical measurement is only emitted when justified.
- Environmental import and timestamp alignment work with real CSV examples.
- Failure examples and quantitative sequence evidence are saved.

## Return package

Provide changes on `member/khang-growth-analytics`, a small anonymized test sequence/reference manifest, before/after overlays, track observation CSV, measurement notes, calibration evidence, environmental CSV example and known issues. Do not submit an unlabelled sequence as tracking ground truth.

## Do not

- change model weights or class mappings.
- treat `association_score` as a probability.
- treat a per-image spatial group as a persistent temporal ID.
- report physical dimensions using an assumed pixel scale.
- rewrite `app.py` or SQLite migrations independently.
