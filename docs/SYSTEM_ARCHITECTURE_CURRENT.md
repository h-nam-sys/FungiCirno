# FungiCirno — current MVP architecture

## Core objective

Monitor oyster mushroom cluster development from fixed-camera side-view captures. The core path uses the Charisis-trained cluster-stage model, the application's temporal tracker and image-plane growth measurements.

## Primary model path — Charisis

`timestamped capture -> TorchVision Mask R-CNN/ConvNeXt-Tiny-FPN -> cluster masks + stage labels -> Hungarian temporal association -> track observations -> area/growth history`

- Three trained labels: `immature_cluster`, `mature_cluster`, `overstayed_cluster` (TorchVision IDs 1, 2, 3; background is ID 0).
- New model bundles register as `charisis_cluster_tracking`; older rows under `legacy_charisis_maskrcnn` remain supported.
- The existing local app has a successful AMD DirectML integration sample: 15 inference runs, 96 candidate predictions, 43 passed the 0.20 analysis threshold, and 22 application tracks / 43 observations.
- Charisis detections are clusters. Do not spatially regroup them as if they were individual fruitbodies.

## Optional individual maturity path — MushR

`image -> individual fruitbody masks + maturity labels -> per-image spatial grouping -> flush summary`

- Optional scope extension only when individual harvest readiness is required.
- Separate taxonomy: `not-ready`, `ready`, `overdue`.
- Uses per-image spatial grouping; not temporal identity tracking.
- Do not require this model to demonstrate the core cluster-growth system.

## Historical model path — M18K

- WB-only YOLOv8-seg benchmark/source-domain experiment.
- Retained for historical results and optional generic segmentation testing.
- Not an oyster cluster model and not needed to load or initialize the trained Charisis checkpoint.

## Inference and storage contract

The runtime is selected by bundle metadata/architecture. Charisis uses the TorchVision Mask R-CNN loader; historical M18K uses the Ultralytics YOLO loader. Both normalize outputs into the same `Prediction` contract consumed by `ModelService`, SQLite, review, tracking and the dashboard.

- Source-image mask area is in px².
- Physical area requires valid camera-specific calibration.
- Track association scores are heuristics, not probabilities.
- Track end due to a missed-frame gap does not prove harvest/removal.

## Training and learning status

- Charisis full training notebook completed 10 epochs on a Kaggle T4; held-out mask AP@[.50:.95] 0.650, AP50 0.782 and AP75 0.759.
- The notebook generated 9,483 accepted test detections and 213 tracks using its own tracker implementation; no ground-truth tracking metric is available.
- FungiCirno uses its own Hungarian tracker. The notebook tracker and app tracker are separate implementations.
- Generic COCO reviewed-sample export exists. Direct reviewed-data fine-tuning in the Charisis notebook is not yet automated.
- The project archive excludes the trained `.pth` bytes. Download the notebook's `model_bundle.zip` separately for a clean installation.
