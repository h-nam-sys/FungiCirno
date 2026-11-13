# Model storage and taxonomy

- `incoming/`: uploaded model ZIPs staged before installation.
- `registry/<model_key>/<version>/`: installed immutable model versions.

## Primary model

`charisis_cluster_tracking` is the production cluster-growth model:

- TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN
- Runtime: `torchvision_state_dict`
- Class IDs: 1=`immature_cluster`, 2=`mature_cluster`, 3=`overstayed_cluster` (ID 0 is background)
- `track_objects=true`, `spatial_grouping=false`
- Bundle files: `model_checkpoint.pth`, `model_config.json`, `class_mapping.json`, optional `evaluation_metrics.json`, and `MODEL_IMPLEMENTATION_NOTE.txt`.

Old installed rows named `legacy_charisis_maskrcnn` remain supported as an alias. New bundles should use `charisis_cluster_tracking`.

## Optional extension

`mushr_maturity` is an optional individual-fruitbody harvest-readiness model (`not-ready`, `ready`, `overdue`). It uses a separate checkpoint/taxonomy and per-image spatial grouping. It is not required for the core cluster-growth pipeline.

## Historical benchmark

`m18k_source_wb` is a White Button YOLOv8-seg benchmark/reference model. It is not an oyster cluster model or maturity classifier.

## Other supported backend

The runtime still supports Ultralytics YOLO-seg bundles (`best.pt`) for historical/experimental models. TorchVision `.pth` and Ultralytics `.pt` checkpoints are not interchangeable. The registry chooses the loader from bundle metadata/architecture.

Weights are intentionally excluded from source archives. Download the trained Charisis `model_bundle.zip` from the executed notebook output and install it through the Models page. Never commit raw datasets, private captures or model weights unless their redistribution terms allow it.
