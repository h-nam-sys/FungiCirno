# FungiCirno whole-system acceptance

Record actual run IDs, model versions, checkpoint hashes, dataset/split IDs, commands, logs and sample artifacts. Do not tick an item because code exists. The system has two production model families with separate taxonomies; complete acceptance requires both target-domain bundles unless a documented scope decision defers Model B.

## A. Environment and application
- [ ] Fresh setup succeeds on Windows/Python 3.12.
- [ ] Streamlit starts and all pages load.
- [ ] CPU PyTorch is installed and CPU inference succeeds with each real target checkpoint.
- [ ] AMD DirectML enumerates the RX 5500 XT and full YOLO inference succeeds for each target checkpoint, or a reproducible backend blocker is recorded.
- [ ] The app remains within the 12 GB system RAM constraint; record peak process RSS and GPU memory observations.
- [ ] App restart preserves model versions, captures, inference runs, predictions, masks, tracks, reviews and environment records.

## B. Model and dataset preparation
- [ ] Charisis public oyster cluster dataset source, terms, annotation format and sequence metadata are recorded.
- [ ] Charisis train/validation/test splits are independent by complete trial/sequence/farm group; split manifest and hashes are saved.
- [ ] Charisis model class order is exactly `immature`, `well-defined`, `overstayed`.
- [ ] M18K initialization and COCO-pretrained initialization are trained on identical Charisis splits and compared on the same untouched target test set.
- [ ] MushR annotated oyster fruitbody dataset source, terms, annotation format and split IDs are recorded.
- [ ] MushR class order is exactly `not-ready`, `ready`, `overdue`.
- [ ] Candidate bundles contain `best.pt`, `model_config.json`, `class_mapping.json`, `evaluation_metrics.json`; hashes and metadata agree.
- [ ] Class IDs are zero-based and are not remapped after training without changing metadata consistently.

## C. Cultivation and capture
- [ ] Create a site, growing unit, fixed camera and capture series.
- [ ] Import multiple timestamped side-view images; EXIF/manual/import fallback is visible.
- [ ] Duplicate image import is skipped by SHA-256.
- [ ] Capture ordering uses capture timestamp, not import order when capture timestamps exist.
- [ ] Images in one track series use the same camera, view and framing.

## D. Detection and persistence
- [ ] Install/activate a real `charisis_cluster_tracking` bundle.
- [ ] Run cluster inference on CPU and AMD; visually inspect masks, boxes, labels, scores and original-image scaling.
- [ ] Verify per-capture `immature` / `well-defined` / `overstayed` predictions are persisted with their model version.
- [ ] Install/activate a real `mushr_maturity` bundle.
- [ ] Run individual maturity inference and visually inspect `not-ready` / `ready` / `overdue` masks and scores.
- [ ] Confirm low-confidence detections remain in the review queue while final-analysis inclusion follows the configured threshold.
- [ ] Verify predictions/masks survive restart.
- [ ] Verify failed inference stores complete error details and does not erase earlier successful runs.

## E. Spatial and temporal analytics
- [ ] Spatial groups are generated per capture and model version.
- [ ] Model B ready-area fraction is derived from `ready` mask area and shown as a descriptive flush summary.
- [ ] Model A's ordered time series produces stable `track_key` values across adjacent captures.
- [ ] Track observations preserve the linked prediction's per-capture stage and confidence.
- [ ] Empty/missed captures advance the gap counter; reappearance within the configured gap is handled as expected.
- [ ] Crowded/overlapping cases, newly appearing pins and partial harvesting are inspected.
- [ ] Multiple model versions do not delete or overwrite each other's tracks/groups.
- [ ] Growth charts show source-image mask area and relative area change; physical units appear only after calibration.
- [ ] Tracking metrics are reported only against explicit reference object IDs with a stated metric definition.
- [ ] Export a track observation CSV and manually reconcile sample rows against source images.

## F. Environment
- [ ] Import CSV with timestamps, temperature, humidity and CO2.
- [ ] Invalid timestamp/range is rejected with a clear message.
- [ ] Duplicate timestamp/source imports update instead of duplicating records.
- [ ] Nearest environmental join respects the configured 12-hour limit.
- [ ] Correlations are labelled exploratory/descriptive, not causal.

## G. Human review and incremental learning
- [ ] Accept, reject and correct class labels for each model family independently.
- [ ] Replace a mask with a corrected binary mask.
- [ ] Export reviewed YOLO data and inspect images/labels manually.
- [ ] Confirm the exported class mapping/taxonomy matches the selected model family.
- [ ] Fine-tune from the active model checkpoint in Colab.
- [ ] Compare active baseline and candidate on the same untouched holdout for same-taxonomy updates.
- [ ] First M18K-to-oyster transfer is evaluated against target-domain absolute gates; WB source metrics are not used as a regression baseline.
- [ ] Candidate with missing/failed holdout is not promoted without an explicit override.
- [ ] Successful candidate activation and rollback to the previous model are both demonstrated.

## H. Reproducibility and deliverables
- [ ] Store dataset source/terms, annotation converter version, split manifests, checkpoint SHA-256 and training configuration.
- [ ] Save test outputs, overlays and failure cases under `results/`.
- [ ] Record software versions, ms/image, process RSS and available GPU memory.
- [ ] Complete the course-required baseline/ablation/self-implemented-algorithm evidence.
- [ ] Record a demo showing import -> Model A inference -> tracking/stage history/growth -> review -> export; show Model B maturity summary separately.
- [ ] Build the PDF report and presentation from actual measured results only.
