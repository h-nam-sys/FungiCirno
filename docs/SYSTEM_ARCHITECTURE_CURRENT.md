# FungiCirno — current MVP architecture

## Runtime

- Streamlit local dashboard (`app.py`).
- SQLite persistence (`src/mushroom_monitor/db.py`).
- Local image and mask storage under `data/`.
- Model registry with candidate/active/archived versions.
- Model inference service with a hardware-specific runtime adapter.
- Human review queue and reviewed-sample export for Colab.
- Descriptive environment-to-growth time alignment.

## Two independent model paths

### Charisis-inspired cluster path

`image -> cluster segmentation -> confidence/review filter -> temporal mask association -> track observations -> area/growth series`

- Tracks cluster masks across captures from the same camera and series.
- Uses mask IoU, predicted centroid, area ratio and Hungarian assignment.
- Supports short missed-frame gaps.
- Tracks are isolated by model version.
- Does not currently implement camera-motion compensation, homography calibration, or learned re-identification.

### MushR maturity path

`image -> fruitbody segmentation/maturity classes -> confidence/review filter -> within-image spatial grouping -> flush summary`

- DBSCAN-equivalent connected components with `min_samples=1` group nearby fruitbody predictions.
- A flush is a spatial grouping in one image, not a temporal identity.
- MushR and Charisis class taxonomies must remain separate.

## Model contract status

The current `InferenceEngine` is built around TorchVision Mask R-CNN models reconstructed from `.pth` state dictionaries. The official MushR export being tested separately is Detectron2 TorchScript. A dedicated adapter is required before that artifact can be registered and run through this app. Input preprocessing, output tuple structure and class mapping must be verified from the actual exported artifact.

## Storage / data flow

- Captures are immutable source images identified by SHA-256.
- Each inference run records device, duration, status, detection count and diagnostics.
- Prediction masks are stored as image files; database rows retain paths and source-image geometry.
- Human decisions are stored separately from model confidence.
- Only accepted/corrected review records are eligible for COCO export.
- Pixel area is not physical area. mm² requires a camera-specific mm-per-pixel calibration.

## Current deployment discrepancy

The earlier architecture specification says CPU inference must always work. The latest runtime in this repository is AMD DirectML-only and explicitly disables CPU fallback. This document records the code's current behavior; CPU support remains an unimplemented architecture requirement.

## MVP exit criteria

1. Import the official MushR TorchScript port using a dedicated, tested runtime adapter.
2. Verify official class-ID/name mapping from release metadata.
3. Run the complete model on the RX 5500 XT at a conservative image size; record latency, memory behavior and failures.
4. Run Charisis and MushR on real source images and preserve overlays.
5. Capture at least two or three time points from one fixed camera for temporal tracking demonstration.
6. Manually review a sample of masks and track associations.
7. Report software tests separately from model quality and biological claims.
