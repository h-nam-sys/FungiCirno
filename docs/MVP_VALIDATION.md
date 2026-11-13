# MVP validation status — Charisis-first build

## What has been demonstrated

### Charisis model training

The attached executed `charisis-full-retraining.ipynb` completed its full 10-epoch run on a Kaggle Tesla T4. It merged the four public Charisis sequences (6,382 images, 37,004 cluster annotations), sampled each fourth frame, trained/validated on sequences 2 and 4, and evaluated on sequences 1 and 3.

The held-out set contained 1,055 images. Overall COCO mask results were AP@[.50:.95]=0.650, AP50=0.782, AP75=0.759 and AR100=0.765. The notebook generated 9,483 accepted detections and exported 213 predicted tracks. Ground-truth track IDs/identity metrics are not present in the supplied evidence.

### Local application integration

The supplied SQLite application snapshot records 15 successful runs of the Charisis TorchVision Mask R-CNN on AMD DirectML. It contains 96 candidate predictions, 43 passing the 0.20 automatic-analysis threshold and 22 application tracks with 43 observations. Across these 15 runs, median model inference time was approximately 0.56 s and median end-to-end run time approximately 2.15 s at a 768-pixel inference long side. These are small-sample observations, not a device benchmark.

The source code includes the matching ConvNeXt-Tiny-FPN model builder, strict `.pth` checkpoint loading, normalized predictions, SQLite persistence and downstream tracking. The saved DB model row is still named `legacy_charisis_maskrcnn` for compatibility and records SHA-256 `3aabb8c4a9ed6016c29b80008afe6dc1d4a1c454392bebaac3fc110c0c79f2d8`. The binary checkpoint is absent from the source ZIP, so its identity cannot be compared against the member notebook's exported checkpoint from the supplied files alone.

### Code-level regression checks in this updated workspace

`python -m unittest discover -s tests -v` completed with 15 passing tests. New coverage verifies TorchVision Charisis bundle canonicalization without class-ID remapping, downstream tracking/grouping capabilities, activation across the legacy/canonical Charisis aliases, and ModelService dispatch that tracks Charisis clusters without creating individual-fruitbody flush groups. This validates code paths with test doubles; it does not replace running the actual checkpoint in a clean target environment.

## What is not yet demonstrated

- CPU inference using the actual Charisis checkpoint in the clean target environment.
- A repeatable clean-install AMD test and sustained multi-capture runtime benchmark.
- Tracking accuracy against manually verified ground-truth identities.
- Growth-measurement error against independent manual measurements.
- Valid camera calibration for physical area/size.
- Automated ingestion of reviewed COCO samples into the Charisis fine-tuning notebook.
- Optional individual-fruitbody harvest maturity, unless confirmed as required by the lead.

## Packaging note

The project ZIP does not contain the binary `model_checkpoint.pth`; the notebook reports creating `model_bundle.zip` in Kaggle's output directory. Download and retain that bundle separately. The SQLite snapshot stores a checkpoint path from the original Windows checkout, so a fresh installation must import the actual bundle rather than rely on that absolute path. On import, new bundles are registered as `charisis_cluster_tracking`; the older DB alias remains recognized and is deactivated when the new canonical version is activated.
