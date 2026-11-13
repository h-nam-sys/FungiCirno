FungiCirno — side-view oyster mushroom growth monitoring

Primary model: Charisis TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN.
Task: segment whole oyster clusters, classify developmental stage, track clusters across captures, and measure visible mask-area change.

The core cluster-growth pipeline does NOT require a separate MushR model.
MushR is an optional individual-fruitbody harvest-readiness extension.
M18K is a historical White Button segmentation benchmark, not the oyster production model.

The Charisis full training run and prior AMD DirectML app inference evidence are documented in:
- docs/MVP_VALIDATION.md
- docs/CHARISIS_TRACKING_METHOD.md
- docs/ACCEPTANCE_CHECKLIST.md

The source archive intentionally excludes model weights and raw datasets. Download the executed Charisis notebook's model_bundle.zip from Kaggle and import it through the app's Models page. The SQLite snapshot contains a historic Windows checkpoint path; re-import the bundle in a clean checkout.

See README.md for setup, installation, operation, limitations and team documentation.
