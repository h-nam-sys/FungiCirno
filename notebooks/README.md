# FungiCirno training notebooks

## Primary training workflow

`Charisis_Full_Training.ipynb` trains the primary oyster cluster-growth model: TorchVision Mask R-CNN with a ConvNeXt-Tiny-FPN backbone. It preserves the Charisis source classes and sequence-based train/test split, evaluates held-out masks, exports tracking examples and creates a TorchVision `model_bundle.zip`.

After changing the export metadata, re-run the final export cell to regenerate a package that includes:

- `model_checkpoint.pth`
- `model_config.json` with `model_key=charisis_cluster_tracking` and `runtime_type=torchvision_state_dict`
- one-based foreground `class_mapping.json`
- `evaluation_metrics.json` when the full COCO evaluation artifact exists
- `MODEL_IMPLEMENTATION_NOTE.txt`

The notebook's full training outputs are retained as evidence from the member run. The project archive does not include the binary model checkpoint; download the generated model bundle from Kaggle separately.

## Historical experiment

`FungiCirno_Oyster_Model_Training.ipynb` is retained as a historical M18K/YOLO-seg experiment. It is not the production Charisis training workflow. M18K is a White Button benchmark, not the oyster cluster-stage model.
