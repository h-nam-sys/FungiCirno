# Integration contracts

These interfaces are the shared boundary between the three workstreams. Keep them stable. A change to a shared field requires a coordinated database migration and an update to all consumers.

## Capture identity

A capture is identified by its database `capture_id` and belongs to one `unit_id`, `camera_id`, and `series_key`. Captures in one series must come from the same fixed camera/view and be ordered by `captured_at` when available, otherwise `imported_at`.

## Model prediction record

The database `predictions` row is the canonical normalized detection contract:

```python
{
    "id": int,                    # prediction_id assigned by SQLite
    "capture_id": int,
    "model_version_id": int,
    "detection_index": int,
    "class_id": int,              # model-local ID; preserve the model mapping
    "class_name": str,
    "confidence": float,          # [0, 1]
    "x1": float, "y1": float, "x2": float, "y2": float,  # source-image pixels
    "area_px2": float,            # source-image projected mask area
    "area_mm2": float | None,     # only with camera calibration
    "centroid_x": float, "centroid_y": float,
    "mask_path": str,             # binary mask file, resolvable locally
    "mask_width": int, "mask_height": int,
    "review_status": "pending|accepted|corrected|rejected",
    "final_for_analysis": bool,
}
```

Model adapters return a list of `Prediction` objects from `inference/runtime.py`. They must return masks at the resized inference-image resolution, and boxes in the same coordinate system. `ModelService` scales boxes, area and centroid to the original capture dimensions before saving.

M18K YOLO models use zero-based class IDs. The old TorchVision adapters may use one-based labels. Do not renumber labels without rewriting `class_mapping.json` and the checkpoint's metadata together.

## Tracking output

`rebuild_sequence_tracks()` consumes only the latest inference run for each capture/model version and returns:

```python
{
    "captures": int,
    "observations": int,
    "tracks": int,
    "active_tracks": int,
}
```

Tracks are scoped by `(unit_id, camera_id, series_key, model_version_id)`. A track's stable key is deterministic for the same ordered sequence and model version. A rebuilt sequence may reassign numeric SQLite IDs; consumers should use `track_key` as the stable display/export key.

Each `track_observations` row stores `frame_index`, `captured_at`, `area_px2`, optional `area_mm2`, `growth_vs_first_pct`, and `association_score`. The association score is a heuristic similarity, not a probability.

## Spatial grouping output

`rebuild_flush_groups()` consumes the latest inference run for one capture and one model version. `flush_groups.model_version_id` is required to keep different models' group outputs separate. Groups are per-image connected components, not persistent temporal identities.

## Environmental readings

CSV fields:

```text
measured_at,temperature_c,humidity_pct,co2_ppm,notes
```

`timestamp` is accepted as an alias for `measured_at`. Blank sensor values are allowed. Duplicate `(unit_id, measured_at, source)` imports update the existing row. Time-aligned comparisons use the nearest reading within 12 hours and are descriptive only.

## Reviewed learning samples

Only `review_items.status IN ('accepted','corrected')` are exported for candidate training. A human mask correction replaces the stored binary mask at source-image resolution. An accepted prediction means the reviewer accepted the existing mask/class; it is not automatically an independent annotation unless the reviewer actually inspected it.

## Model bundle contract (YOLO-seg)

```text
best.pt
model_config.json
class_mapping.json
evaluation_metrics.json   # recommended/required for promotion
README.txt
```

`model_config.json` includes `model_key`, `version`, `architecture`, `runtime_type=ultralytics_yolo_seg`, `task=segment`, and inference parameters. `class_mapping.json` contains `label_to_name` with zero-based IDs. `evaluation_metrics.json` should include `promotion_ready`, baseline/candidate metrics, holdout identity and gate values.

## Ownership boundaries

| Files | Owner |
|---|---|
| `src/mushroom_monitor/inference/`, `model_registry.py`, `learning.py`, `tools/train_candidate.py`, `tools/package_yolo_model.py` | Minh Thinh — model/lifecycle workstream |
| `tracking.py`, `maturity.py`, `environment.py` | Le Thien Khang — analytics workstream |
| `app.py`, `db.py`, `config.py`, shared interfaces, migrations, `README.md`, release packaging | Nguyen Hai Nam — integration lead |

Members may submit bug fixes outside their ownership, but must not independently change shared interfaces or database schema.

## Production model-family contract

The production runtime is one Ultralytics YOLO-seg adapter. Model family is metadata, not a different downstream prediction schema.

| `model_key` | `prediction_role` | Required zero-based class order | `track_objects` |
|---|---|---|---|
| `charisis_cluster_tracking` | `cluster_stage` | `immature`, `well-defined`, `overstayed` | `true` |
| `mushr_maturity` | `fruitbody_maturity` | `not-ready`, `ready`, `overdue` | `false` |
| `m18k_source_wb` | `generic_instance` | `WB` | `false` by default; enable only for explicit integration tests |

The package and training tools validate the required target class order for the two production roles. Do not rename labels or mix classes between model families. The class name and model version on each prediction are retained all the way into the track observation/review history.

`confidence_threshold` is the minimum score for automatic downstream analysis (`final_for_analysis`). `score_floor` is the lower inference threshold. Detections between the two remain stored and reviewable, but are excluded from automatic grouping/tracking until accepted/corrected according to the application's review policy.

## Dataset and model lineage

- Model A is trained with the Charisis oyster cluster time-lapse annotations. Compare M18K-initialized and COCO-initialized runs on the same target split. WB source metrics are not a baseline for oyster class performance.
- Model B is trained with the annotated MushR oyster fruitbody dataset. Its object unit and labels differ from Model A.
- M18K source weights, model A, and model B use separate `model_key` entries and independent checkpoints.
- Each model bundle must retain source species, taxonomy, role, class mapping, training source, inference settings, checkpoint hash, holdout identity and metrics where available.
