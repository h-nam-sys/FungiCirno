# FungiCirno whole-system acceptance

Record actual run IDs, model versions, dataset identifiers and logs. Do not tick an item merely because the code exists.

## A. Environment and application
- [ ] Fresh setup succeeds on Windows/Python 3.12.
- [ ] Streamlit starts and the dashboard loads.
- [ ] CPU PyTorch is installed and CPU inference succeeds.
- [ ] AMD DirectML enumerates the RX 5500 XT and full YOLO inference succeeds (record failures if not).
- [ ] Application works within available system memory; record peak process RSS and any GPU memory observations.

## B. Cultivation and capture
- [ ] Create a site, growing unit, camera and capture series.
- [ ] Import multiple timestamped images; EXIF/manual/import fallback is visible.
- [ ] Duplicate image import is skipped by SHA-256.
- [ ] Capture records survive an application restart.

## C. Detection and persistence
- [ ] Install a real M18K YOLO-seg bundle.
- [ ] Activate a chosen model version.
- [ ] Run one image on CPU and one on AMD; record backend, model version, time and process RSS.
- [ ] Verify masks, boxes, class IDs, confidence and source-image scaling visually.
- [ ] Verify predictions and masks persist after restart.
- [ ] Verify failed inference stores the complete error and does not clear previous runs.

## D. Spatial and temporal analytics
- [ ] Spatial groups are generated per capture and model version.
- [ ] A multi-capture series produces stable track keys across consecutive observations.
- [ ] An empty/missed capture advances the track gap counter.
- [ ] Multiple model versions do not delete or overwrite each other's tracks/groups.
- [ ] Growth charts show area and relative area change; physical units appear only after calibration.
- [ ] Export a track observation CSV and manually reconcile sample rows.

## E. Environment
- [ ] Import a CSV with timestamps, temperature, humidity and CO2.
- [ ] Invalid range/timestamp is rejected with a clear message.
- [ ] Duplicate timestamp/source imports update instead of duplicating records.
- [ ] Nearest environmental join respects the 12-hour limit.
- [ ] Correlation output is labelled exploratory/descriptive.

## F. Human review and incremental learning
- [ ] Accept, reject and correct a class.
- [ ] Replace a mask with a corrected binary mask.
- [ ] Export reviewed YOLO data and inspect several exported images/labels manually.
- [ ] Fine-tune from the previous checkpoint in Colab.
- [ ] Compare baseline and candidate on the same untouched holdout.
- [ ] Candidate with no/failed holdout is not promoted without explicit override.
- [ ] Successful candidate activates; previous model remains available and can be restored.

## G. Reproducibility and deliverables
- [ ] Store dataset links, split manifests, checkpoint SHA-256 and training configuration.
- [ ] Save test/validation outputs and failure cases under `results/`.
- [ ] Record software versions, ms/image and memory usage.
- [ ] Complete course-required baseline/ablation/self-implemented-algorithm evidence.
- [ ] Record a demo video showing import -> inference -> tracking/history -> review.
- [ ] Build PDF report and presentation from actual measured results only.
