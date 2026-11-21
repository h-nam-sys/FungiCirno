# FungiCirno model card

## Summary

FungiCirno provides two selected instance-segmentation checkpoints for detecting oyster mushroom clusters in the Charisis time-lapse dataset. The models are intended for research reproduction and comparative evaluation, not as a validated commercial cultivation or harvest-decision system.

## Available checkpoints

| Model | Architecture | Foreground classes | Expected SHA-256 |
|---|---|---:|---|
| YOLO26-L-seg | Ultralytics YOLO26-L instance segmentation | 1 (`mushroom_cluster`) | `7964816abe44fd635725fa0997aaf161ce8d7429c7f265116bf8e53c966cbb8b` |
| Mask R-CNN | TorchVision Mask R-CNN with ConvNeXt-Tiny-FPN | 3 foreground maturity classes + background | `3aabb8c4a9ed6016c29b80008afe6dc1d4a1c454392bebaac3fc110c0c79f2d8` |

Download both checkpoints and the canonical split manifest from the [FungiCirno Kaggle model dataset](https://www.kaggle.com/datasets/hnamedu/fungicirno-models).

## Inference configuration

The reproducibility notebook records and verifies the following settings:

| Setting | YOLO26-L-seg | Mask R-CNN |
|---|---:|---:|
| Input | 1,024 px | Source image / model transform |
| Candidate score floor | 0.001 | 0.001 |
| NMS IoU | 0.70 | TorchVision model implementation |
| Maximum detections | 1,000 | 100 |
| Mask handling | Retina masks | Full-resolution masks pasted per instance on CPU |
| Binary mask threshold | Project export path | 0.50 |

The YOLO validation-selected fixed-threshold evaluation confidence is 0.871056. The Mask R-CNN threshold is 0.757535. These operating thresholds are distinct from the lower candidate-generation score floors above.

The Mask R-CNN model uses a custom ConvNeXt-Tiny-FPN backbone with feature channels [96, 192, 384, 768] and 256-channel FPN outputs. Its three foreground maturity classes are collapsed into one cluster category for the common class-agnostic evaluation.

## Intended use

- Reproduce the experiments described in the FungiCirno report.
- Compare instance masks and image-space geometry on the source dataset.
- Explore temporal association and cluster-level growth histories using the supplied notebook.

## Out-of-scope uses and limitations

- Do not interpret pixel area or bounding-box dimensions as physical dimensions without camera calibration.
- Do not interpret the training-derived relative-size index as a maturity probability or harvest-readiness decision.
- The YOLO model predicts one cluster class; it does not classify maturity.
- Tracking errors can merge or fragment source identities. The paper reports these failures and the retrospective nature of test-set tracking evaluation.
- The checkpoints were not trained under a matched-budget protocol, and the Mask R-CNN checkpoint is not the original Charisis implementation.
- Complete Mask R-CNN training hyperparameters were not retained in the available experiment artifacts.

## Data and licensing

The models were developed and evaluated using the Charisis oyster-mushroom dataset. The images and annotations are not included in the Kaggle model-artifact dataset or this repository. Obtain the dataset from its [original Zenodo record](https://doi.org/10.5281/zenodo.15041243) and follow its licence and citation conditions.

The code licence and model-weight redistribution terms must be specified by the project maintainers before reuse or redistribution is invited. Public hosting alone does not grant such permission.
