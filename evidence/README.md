# Research evidence index

This directory preserves the machine-readable outputs, source prediction records, figures, run manifests, and integrity reports used in the project. The paper's final claims should be traced to the final inference metrics and the frozen tracking evaluation described below.

## Canonical evidence for the paper

### Segmentation and matched-instance measurement

The fresh inference run is recorded in:

- `run_manifest.json` — source, checkpoint, split, runtime, and execution provenance.
- `integrity_report.json` — source-frame, prediction, and artifact checks.
- `fresh_segmentation_metrics.csv` — segmentation metrics for the two evaluated checkpoints.
- `fresh_per_instance_geometry_metrics.csv` and `fresh_matched_instance_details.csv` — matched-instance geometric measurements.
- `fresh_prediction_records/` — compressed prediction records for YOLO26-L-seg and Mask R-CNN.

### Final tracking and growth-history results

Use [`growth_tracking/`](growth_tracking/) as the canonical evidence for the paper's final frozen-tracker results. In particular:

- `test_overall_metrics.csv` and `test_per_sequence_metrics.csv` — aggregate and sequence-specific tracking results.
- `selected_tracker_config_v5.json` and its SHA-256 sidecar — the frozen tracker configuration.
- `test_track_purity.csv`, `test_source_identity_coverage.csv`, and `test_mixed_identity_track_failures.csv` — track purity and strict history screening.
- `test_strict_growth_observations.csv`, `test_growth_metrics_by_identity.csv`, and `test_adjacent_area_change_transitions.csv` — source-consistent geometry and temporal growth metrics.
- `figures/`, `FIGURE_CAPTIONS.md`, and `METRIC_DEFINITIONS.md` — publication figures and definitions.
- `tracking_evaluation_run_manifest.json` and `integrity_report.json` — run provenance and checks.

## Historical exploratory tracker snapshots

Files prefixed `A_iou_only`, `B_spatial_only`, `C_iou_plus_lowconf`, and `D_spatial_plus_lowconf` are preserved as historical tracker-comparison outputs from development. They include validation and retrospective test records. They are retained for transparency and auditability, but they are **not** the final frozen-tracker evaluation reported in the paper. The test sequences were inspected during tracker development; these historical comparisons must not be presented as results on an untouched independent test set.

For the final paper's tracking claims, use `growth_tracking/` and its frozen configuration rather than selecting the best-looking row from the historical A/B/C/D files.

## Data and use restrictions

The original images and annotations are not redistributed in this repository. Obtain them from the [Zenodo record for “Oyster mushroom clusters growth monitoring timelapses”](https://doi.org/10.5281/zenodo.15041243) and follow its reuse terms. Prediction records and metrics are derived research artifacts; check the applicable source-dataset terms before redistributing them separately.

## Naming and provenance notes

The repository's current public name is **Oyster Mushroom Cluster Monitoring**. Some archived input filenames, the frozen tracker's embedded `artifact` label, and earlier run manifests retain the former project label. These values are preserved because they identify the original recorded artifacts and, for the frozen configuration, are covered by its SHA-256 hash. New documentation and newly generated output paths use the current project name.
