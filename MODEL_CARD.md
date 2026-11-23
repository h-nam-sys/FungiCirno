# Model card — Oyster Mushroom Cluster Monitoring

## Summary

This project distributes two selected instance-segmentation checkpoints for detecting oyster mushroom clusters in the Charisis time-lapse dataset. They support research reproduction and comparative evaluation; they are not a validated commercial cultivation, physical-measurement, or harvest-decision system.

The segmentation model is **YOLO26-L-seg**. The complete research workflow also includes a mask-IoU tracker and image-space measurement stage. The project/repository name refers to that broader workflow, not to a new neural-network architecture.

## Available checkpoints

| Model | Architecture | Foreground classes | Expected SHA-256 |
|---|---|---:|---|
| YOLO26-L-seg | Ultralytics YOLO26-L instance segmentation | 1 (`mushroom_cluster`) | `7964816abe44fd635725fa0997aaf161ce8d7429c7f265116bf8e53c966cbb8b` |
| Mask R-CNN | TorchVision Mask R-CNN with ConvNeXt-Tiny-FPN | 3 foreground maturity classes + background | `3aabb8c4a9ed6016c29b80008afe6dc1d4a1c454392bebaac3fc110c0c79f2d8` |

Download both checkpoints and the canonical split manifest from [Oyster Mushroom Cluster Monitoring Models — Kaggle](https://www.kaggle.com/datasets/hnamedu/fungicirno-models). The existing Kaggle dataset URL is unchanged.

## Selected evaluation results

| Result | Value |
|---|---:|
| YOLO26-L-seg mask AP50:95 | 89.02% |
| Mask R-CNN baseline mask AP50:95 | 57.03% |
| Frozen-tracker MOTA / IDF1 | 89.19% / 85.56% |
| Strict-history coverage (pooled / macro) | 81.45% / 82.45% |
| Strict-history area / width / height RAE | 4.36% / 1.54% / 1.65% |
| Macro growth-direction agreement | 89.75% |

The tracking and growth-history results are retrospective because the test sequences were inspected during development. Tracking uses the separate `tracking-and-growth-history.ipynb` notebook and a frozen validation-selected configuration. Refer to the paper and evidence files for full protocol details and per-sequence results.

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

- Reproduce the segmentation and geometric-measurement experiments described in the research report.
- Compare instance masks and image-space geometry on the source dataset.
- Explore temporal association and cluster-level growth histories using the two supplied notebooks.

## Out-of-scope uses and limitations

- Do not interpret pixel area or bounding-box dimensions as physical dimensions without camera calibration.
- Do not interpret the training-derived relative-size index as a maturity probability or harvest-readiness decision.
- YOLO26-L-seg predicts one cluster class; it does not classify maturity.
- Tracking errors can merge or fragment source identities. The paper reports these failures and the retrospective nature of test-set tracking evaluation.
- The checkpoints were not trained under a matched-budget protocol, and the Mask R-CNN checkpoint is not the original Charisis implementation.
- Complete Mask R-CNN training hyperparameters were not retained in the available experiment artifacts.

## Data and licensing

The models were developed and evaluated using the dataset **Oyster mushroom clusters growth monitoring timelapses**, associated with Charisis et al. The images and annotations are not included in the Kaggle model-artifact dataset or this repository. Obtain the dataset from its [original Zenodo record](https://doi.org/10.5281/zenodo.15041243) and follow its licence and citation conditions.

The repository's MIT license covers source code and documentation only. It does not determine the redistribution rights for the separately hosted model weights or source dataset. Check the applicable terms before redistributing those artifacts.
