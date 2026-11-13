# FungiCirno integration contracts

These interfaces are the shared boundary between inference, persistence, review, tracking and the UI. Keep them stable. A shared-field change requires coordinated tests and, where relevant, a database migration.

## Capture identity

A capture belongs to one `unit_id`, `camera_id` and `series_key`. A tracking series must use one fixed camera/view and be ordered by `captured_at` where available; import time is only a fallback.

## Normalized prediction contract

All model loaders return `Prediction` objects from `src/mushroom_monitor/inference/runtime.py`, regardless of backend. The database prediction row is the downstream contract:

```python
{
    "id": int,
    "capture_id": int,
    "model_version_id": int,
    "detection_index": int,
    "class_id": int,              # model-local ID; preserve mapping
    "class_name": str,
    "confidence": float,          # [0, 1]
    "x1": float, "y1": float, "x2": float, "y2": float,  # source-image pixels
    "area_px2": float,            # projected source-image mask area
    "area_mm2": float | None,     # only with camera calibration
    "centroid_x": float, "centroid_y": float,
    "mask_path": str,
    "mask_width": int, "mask_height": int,
    "review_status": "pending|accepted|corrected|rejected",
    "final_for_analysis": bool,
}
```

The runtime adapter returns masks and boxes in inference-image coordinates. `ModelService` maps boxes, centroids and area back to source-image dimensions before persistence.

## Supported model bundles and class-ID conventions

### Primary Charisis TorchVision bundle

```text
model_checkpoint.pth
model_config.json
class_mapping.json
evaluation_metrics.json       # when available
MODEL_IMPLEMENTATION_NOTE.txt
```

- `runtime_type`: `torchvision_state_dict`
- `architecture`: `TorchVision Mask R-CNN with ConvNeXt-Tiny-FPN`
- `model_key`: `charisis_cluster_tracking`
- `prediction_role`: `cluster_stage`
- `track_objects`: `true`
- `spatial_grouping`: `false`
- `source_species`: `Pleurotus ostreatus`
- `taxonomy`: `charisis_oyster_cluster_stage_v1`
- `class_mapping.json`: preserve TorchVision foreground IDs 1, 2, 3 for `immature_cluster`, `mature_cluster`, `overstayed_cluster`; ID 0 is background.
- Checkpoint payload contains the trained `model` state dictionary. Load with strict state-dictionary matching into the matching ConvNeXt-Tiny-FPN Mask R-CNN builder.

Older model rows/bundles named `legacy_charisis_maskrcnn` remain supported and are treated as Charisis cluster models by the capability resolver.

### Optional MushR TorchVision bundle

- `runtime_type`: `torchvision_state_dict`
- Architecture: Mask R-CNN ResNet-50-FPN
- Task: individual oyster fruitbody maturity
- Class IDs: preserve the checkpoint's one-based foreground mapping for `not-ready`, `ready`, `overdue`.
- `track_objects=false`, `spatial_grouping=true`.
- This model is optional; it is not required to validate cluster-level growth monitoring.

### Historical M18K YOLO bundle

- Checkpoint: `best.pt`
- `runtime_type`: `ultralytics_yolo_seg`
- Class IDs are zero-based (`0: WB`).
- Task: White Button instance segmentation only.
- It is a historical benchmark/reference model, not the Charisis production model.

Do not renumber class IDs or reuse a taxonomy across model families.

## Downstream capabilities

`model_capabilities(model_key, architecture, config)` determines whether to run downstream tasks:

- Charisis cluster model: temporal tracking enabled; per-image spatial flush grouping disabled.
- MushR individual maturity model: temporal tracking disabled by default; per-image spatial grouping enabled.
- M18K generic instance model: neither capability enabled unless explicitly configured.

Spatial groups are within-image aggregates, not persistent IDs. Track records are linked to source predictions and scoped by unit, camera, series and model version.

## Tracking output

`rebuild_sequence_tracks()` consumes the latest inference run for each capture/model version and returns capture, observation, track and active-track counts. `track_key` is the stable display/export key for a fixed ordered series and model version; SQLite numeric IDs may change on a rebuild.

The FungiCirno app tracker uses mask IoU, predicted-centroid proximity, area ratio and Hungarian assignment. Its score is a heuristic, not a probability.

## Reviewed samples and training

Only human accepted/corrected samples are learning candidates. `export_reviewed_samples()` produces a COCO-style reviewed-mask archive with the original model taxonomy and sequence-group metadata. `export_reviewed_yolo_dataset()` is specifically a YOLO-seg export and must not be passed to the Charisis TorchVision loader/training workflow.

The current Charisis notebook trains from the public Charisis dataset. Direct ingestion of reviewed COCO exports into its resume/fine-tuning flow is not yet automated. Preserve sequence identity and a separate untouched holdout when adding that capability.

## Environmental readings

CSV fields:

```text
measured_at,temperature_c,humidity_pct,co2_ppm,notes
```

`timestamp` is accepted as an alias. The nearest-reading join is descriptive and limited to the configured time window; it does not establish causality.
