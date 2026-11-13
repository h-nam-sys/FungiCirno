# Workstream C — Nguyen Hai Nam (lead)

## Ownership

Core application, integration, shared database/contracts, hardware execution policy, release control, research claims and final system acceptance.

## Current architecture

- Primary model: Charisis-trained TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN.
- Task: side-view *Pleurotus ostreatus* whole-cluster segmentation and three-stage classification.
- Preserve class IDs: 1=`immature_cluster`, 2=`mature_cluster`, 3=`overstayed_cluster`; ID 0 is background.
- Application tracking: its own mask/centroid/area association with Hungarian assignment.
- Core outputs: persistent cluster IDs, stage history and projected visible-mask-area growth.
- MushR individual-fruitbody harvest readiness is optional; do not require it for core growth monitoring until the lead confirms scope.
- M18K is historical White Button YOLO segmentation research; it is neither the oyster model nor a compatible Charisis initialization.
- TorchVision and optional YOLO loaders share a normalized prediction contract; do not force all checkpoints through one backend.

## Responsibilities

1. Keep `main` protected. Members work on their dedicated branches and submit focused pull requests.
2. Own `app.py`, DB/schema, shared interfaces, project packaging and final documentation.
3. Integrate Minh Thinh's Charisis model/runtime changes and Khang's tracking/analytics changes independently.
4. Download and preserve the actual Charisis `model_bundle.zip` from the member's Kaggle output; verify SHA-256. The submitted source ZIP omits the binary `.pth` weights.
5. On a clean checkout, import the bundle through **Models**, activate the verified version, and confirm the legacy DB row's absolute Windows checkpoint path is no longer relied upon.
6. Run end-to-end CPU acceptance and the actual AMD DirectML check on the RX 5500 XT. Record device, runtime, latency, memory and sample overlays.
7. Require a manually verified temporal identity subset before claiming tracking accuracy. Require manual measurement/calibration before physical sizing claims.
8. Keep the optional MushR maturity task out of the critical path unless individual fruitbody harvest-readiness is explicitly required.
9. Update the research paper with Charisis training and application evidence, while distinguishing segmentation metrics, app integration evidence and true tracking accuracy.

## Release acceptance

A clean checkout must install from README, import and run the Charisis bundle, ingest timestamped side-view captures, persist masks/predictions, maintain cluster tracks, show growth and stage history, support review/export, import environmental readings and run on CPU. AMD DirectML is accepted only after real full-model inference on target hardware. Candidate promotion and rollback require explicit evidence.

## Branch policy

- `main`: release baseline; only Hai Nam merges.
- `member/minh-thinh-ml`: Charisis model/runtime fixes.
- `member/khang-growth-analytics`: tracking and analytics fixes.
- Pull requests must include files changed, test commands/results, screenshots or logs, and known risks.
