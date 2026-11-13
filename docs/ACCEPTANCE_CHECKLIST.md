# FungiCirno core acceptance checklist

Record actual run IDs, checkpoint hashes, dataset/split IDs, commands, logs and sample artifacts. Do not tick an item because code exists.

## A. Primary Charisis model

- [x] Charisis notebook full training completed for 10 epochs on a Kaggle Tesla T4.
- [x] Train/validation uses sequences 2 and 4; held-out test uses sequences 1 and 3.
- [x] Held-out inference completed on 1,055 sampled images; overall mask AP@[.50:.95]=0.650, AP50=0.782, AP75=0.759.
- [x] Notebook exported `model_bundle.zip` from the full run.
- [x] Project has a ConvNeXt-Tiny-FPN Mask R-CNN builder and strict TorchVision `.pth` state-dictionary loader.
- [x] Bundle import preserves Charisis foreground IDs/names and canonicalizes the family to `charisis_cluster_tracking`.
- [x] Unit tests verify Charisis tracking is enabled, individual-fruitbody grouping is disabled, and legacy/canonical Charisis activation is mutually exclusive.
- [x] Supplied app database records 15 successful Charisis AMD DirectML inference runs.
- [ ] Install the actual bundle into a clean checkout and verify checkpoint hash/config/class map.
- [ ] Run a representative Charisis capture on CPU and record inference time/memory.
- [ ] Repeat AMD DirectML inference after clean installation and retain overlays/logs.
- [ ] Confirm the exact checkpoint binary and bundle are retained outside the source archive.

## B. Capture, inference and persistence

- [ ] Create a site, growing unit, fixed side-view camera and capture series.
- [ ] Import multiple timestamped side-view images; duplicate image import is skipped by SHA-256.
- [ ] Confirm capture order uses capture timestamp, not import order.
- [ ] Run the Charisis model and visually inspect masks, boxes, labels, confidence and source-image scaling.
- [ ] Verify class IDs 1, 2, 3 map to the checkpoint's names; ID 0 remains background.
- [ ] Verify predictions/masks survive restart and failed runs retain full error details.

## C. Temporal tracking and growth

- [x] Project app has a separate Hungarian mask/centroid/area tracker and persistent track records.
- [x] Model-service regression test confirms Charisis inference dispatches to tracking without creating fruitbody flush groups.
- [x] Supplied DB snapshot records 22 Charisis track records containing 43 observations.
- [ ] Manually annotate a reference subset with persistent cluster IDs.
- [ ] Measure identity switches, track fragmentation and IDF1/HOTA/MOTA only if ground-truth IDs are available.
- [ ] Check empty captures, missed-frame gaps, reappearance, cluster splits/merges and new pins.
- [ ] Reconcile selected track observations against their source images.
- [ ] Compare image-plane growth curves with manual measurements.
- [ ] Show physical units only when camera-specific calibration is validated.

## D. Review and model lifecycle

- [ ] Accept, reject, correct class labels and replace masks.
- [x] Generic human-reviewed COCO export exists.
- [ ] Confirm reviewed Charisis COCO exports preserve class IDs and sequence metadata.
- [ ] Integrate reviewed COCO samples into a Charisis-specific fine-tuning workflow; this is not yet automated.
- [ ] Candidate must be evaluated on an untouched sequence-/farm-separated holdout.
- [ ] Candidate activation and rollback are exercised.

## E. Optional individual maturity extension

- [ ] Confirm with the lead whether per-individual harvest readiness is required.
- [ ] Only if required: train and validate the optional MushR/individual-fruitbody model with its separate `not-ready`, `ready`, `overdue` taxonomy.
- [ ] Keep its per-image spatial grouping separate from Charisis temporal cluster tracking.

## F. Runtime and deployment

- [ ] Fresh Windows/Python environment setup succeeds.
- [ ] CPU inference succeeds with the actual Charisis checkpoint.
- [x] AMD DirectML succeeded for 15 sample Charisis app runs in the supplied DB snapshot.
- [ ] Record a repeatable clean-install AMD test, latency and memory; existing sample evidence is not a controlled benchmark.
- [ ] Confirm the app remains within the 12 GB system RAM constraint under a multi-capture sequence.

## G. Environment and reporting

- [ ] Import CSV with timestamps, temperature, humidity and CO2.
- [ ] Invalid timestamps/ranges are rejected; duplicate imports update.
- [ ] Nearest environmental reading is aligned within the configured window.
- [ ] Correlation outputs remain exploratory/descriptive, not causal.
- [ ] Paper/presentation claims distinguish notebook mask AP, generated tracks and app integration evidence. Do not report tracking accuracy without ground truth.

## H. Automated regression checks

- [x] `python -m unittest discover -s tests -v` — 15 tests passed in the updated project workspace.
- [x] Changed Python sources passed in-memory syntax compilation.
