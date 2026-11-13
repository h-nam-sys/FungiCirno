# FungiCirno — team verification handoff

This release uses the already-trained Charisis model as the primary oyster cluster-growth model. The model is TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN, not YOLO. The application already contains a matching builder/loader and its included SQLite snapshot records successful AMD DirectML runs.

## Fixed architecture

- **Core model:** `charisis_cluster_tracking`, TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN.
- **Object and labels:** whole oyster mushroom clusters; preserve foreground class IDs exactly: `1=immature_cluster`, `2=mature_cluster`, `3=overstayed_cluster`; ID 0 is background.
- **Core downstream:** application temporal tracker, persistent cluster IDs, stage history and visible mask-area growth curves.
- **Tracker:** FungiCirno's own mask/centroid/area association with Hungarian assignment. It is separate from the tracker included in the training notebook.
- **Optional extension:** `mushr_maturity`, individual-fruitbody harvest readiness (`not-ready`, `ready`, `overdue`). Defer unless the lead requires individual harvest-readiness output.
- **M18K:** historical White Button YOLO segmentation benchmark only. It is not the oyster production model and is not a compatible initialization for the Charisis TorchVision checkpoint.
- All inference backends return the shared normalized `Prediction` record. Do not force Charisis through the YOLO loader.
- Training is offline in Kaggle/Colab. Local app performs inference, persistence, review, tracking and model management.

Read `docs/FINAL_SYSTEM_ARCHITECTURE.md` and `docs/INTEGRATION_CONTRACTS.md` before making shared changes.

## Evidence and current blockers

- Charisis full training: 10 epochs; 433 train, 109 validation, 1,055 held-out test images; sequences 2/4 for training/validation and 1/3 for test.
- Held-out mask metrics: AP@[.50:.95] 0.650, AP50 0.782, AP75 0.759.
- Notebook generated 9,483 detections and 213 predicted tracks. These are not tracking accuracy metrics; no ground-truth persistent IDs were supplied.
- Supplied app database: 15 successful Charisis AMD DirectML runs, 96 candidate predictions, 43 passing the 0.20 analysis threshold, and 22 app tracks containing 43 observations. Small-sample evidence, not a controlled benchmark.
- The project archive does not include `model_checkpoint.pth`. The SQLite snapshot records an absolute path from the original Windows checkout. A clean checkout must import the Charisis `model_bundle.zip` downloaded from Kaggle.
- CPU inference with the actual checkpoint and clean-install acceptance are still required.

## Team ownership

- **Nguyen Hai Nam (lead):** main branch, app, DB/schema, shared contracts, bundle integration, release and system acceptance.
- **Nguyen Minh Thinh:** Charisis checkpoint packaging/loading, model runtime and training/evaluation lifecycle. Optional YOLO/M18K work only when explicitly scoped.
- **Le Thien Khang:** sequence handling, application tracking, growth measurements, calibration and environment integration. Do not treat the notebook's 213 predicted tracks as verified ground truth.

## Rules

- Never work directly on `main`; use assigned member branches.
- Do not change DB schema/migrations, app entry point or shared inference contracts without Hai Nam's approval.
- Preserve Charisis class IDs and names exactly. Do not re-label `mature_cluster` as `well-defined`.
- Do not run per-image fruitbody grouping on Charisis whole-cluster detections.
- Do not mix cluster-stage and individual-maturity taxonomies.
- Do not claim IDF1/HOTA/MOTA without ground-truth persistent identities.
- CPU inference is mandatory. AMD DirectML must be verified with the actual checkpoint on the target PC; no silent fallback.
- Keep weights, raw datasets, private captures, credentials and local databases out of Git.
- Do not claim a test passed unless it was executed; include the command and result.

## Return to Hai Nam

Send branch/commit, exact commands and results, checkpoint SHA-256, package versions, logs, sample overlays, dataset/split IDs, failures and remaining risks. See `docs/ACCEPTANCE_CHECKLIST.md` for system acceptance.
