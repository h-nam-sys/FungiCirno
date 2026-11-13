# FungiCirno release notes — Charisis-primary update

## Primary model decision

- Make the member-trained Charisis TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN the primary oyster cluster-growth model.
- Preserve its exact foreground mapping: `1=immature_cluster`, `2=mature_cluster`, `3=overstayed_cluster`; TorchVision background is ID `0`.
- Use `charisis_cluster_tracking` as the canonical model-family key. Existing `legacy_charisis_maskrcnn` rows remain supported during migration.
- Do not convert the trained Charisis `.pth` checkpoint to YOLO. The project already has a matching ConvNeXt-Tiny-FPN builder and strict TorchVision state-dictionary loader.
- M18K remains a historical White Button YOLO segmentation benchmark/reference, not a required oyster model or compatible Charisis initializer.
- MushR remains an optional, separately trained individual-fruitbody harvest-readiness extension. It is not needed for the core cluster-growth pipeline.

## Application changes

- Added model capability resolution so the Charisis family enables temporal tracking and disables per-image fruitbody/flush grouping.
- Updated inference-service dispatch to rebuild the application tracker for Charisis captures without creating flush groups from whole-cluster detections.
- Updated the maturity/grouping page to exclude Charisis predictions and describe its separate optional scope.
- Updated review, overview, model registry and learning-lab descriptions to distinguish the primary Charisis `.pth` workflow from optional YOLO `.pt` tools.
- Canonical Charisis activation now treats `charisis_cluster_tracking` and `legacy_charisis_maskrcnn` as the same family, preventing two active versions of the same task.
- Added visible checkpoint-availability checks and an instruction to import the model bundle when a stored absolute path is unavailable on a clean machine.
- Replaced the duplicate top-level service implementation with a compatibility shim to the maintained `src/mushroom_monitor.service` implementation.
- Added regression tests for class-ID preservation, model capabilities, legacy/canonical activation and Charisis service dispatch.

## Member training evidence

The attached executed Charisis notebook completed a full 10-epoch training run on a Kaggle Tesla T4. It used the Charisis oyster cluster data (6,382 images and 37,004 annotations), sampled at one-hour intervals, with sequences 2/4 used for training/validation and sequences 1/3 held out for testing.

| Held-out evidence | Result |
|---|---:|
| Test images | 1,055 |
| Mask AP@[0.50:0.95] | 0.650 |
| Mask AP@0.50 | 0.782 |
| Mask AP@0.75 | 0.759 |
| AR@100 | 0.765 |
| Test detections | 9,483 |
| Notebook-generated predicted tracks | 213 |

The 213 tracks are outputs from the notebook's own tracker, not ground-truth identities and not a tracking-accuracy score. The notebook did not supply IDF1, HOTA or MOTA against ground-truth track IDs.

## Existing application evidence

The supplied project SQLite snapshot records 15 successful Charisis AMD DirectML inference runs, 96 candidate predictions, 43 predictions eligible at the configured 0.20 analysis threshold, and 22 application tracks containing 43 observations. The observed median model inference time is approximately 0.56 seconds and median end-to-end run time approximately 2.15 seconds at the recorded 768-pixel inference side. This is a small historical sample, not a controlled benchmark.

The database's registered legacy Charisis row records checkpoint SHA-256 `3aabb8c4a9ed6016c29b80008afe6dc1d4a1c454392bebaac3fc110c0c79f2d8`. The checkpoint binary is not present in the uploaded project archive, so its identity cannot be independently matched to the attached notebook's final exported checkpoint from these files alone.

## Verification performed on the updated source

- `python -m unittest discover -s tests -v` — 15 tests passed.
- Changed application, service, registry, database and test modules passed Python syntax compilation.
- The full project source is being packaged without checkpoint weights, raw cultivation data, private captures or credentials.

## Remaining acceptance work

- Download `model_bundle.zip` from the member's Kaggle notebook output and import it on a clean checkout; the project archive intentionally does not contain the `.pth` weights.
- Verify the imported bundle hash/configuration and run a representative CPU inference using that exact checkpoint.
- Repeat AMD DirectML inference after clean installation and record a controlled latency/memory run.
- Validate application tracking against a manually verified identity subset before reporting IDF1/HOTA/MOTA.
- Compare image-plane growth curves with independent manual measurements; enable physical area only after camera calibration is validated.
- Integrate reviewed Charisis COCO samples into a dedicated fine-tuning workflow before claiming an automated human-review-to-retraining loop.
- Add the optional individual-maturity model only if the lead confirms that per-fruitbody harvest readiness is a required deliverable.
