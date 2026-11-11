# FungiCirno MVP validation record

Validation date: 2026-11-11

## Confirmed in this repository snapshot

- Python source compilation: all 18 Python files compiled successfully.
- Core unit tests: 9/9 passed.
- SQLite persistence and default records.
- Capture hashing and duplicate detection.
- Model bundle metadata parsing and registry installation using a stub checkpoint.
- MushR-style spatial grouping mechanics.
- Charisis-inspired mask-based sequence tracking across frames.
- Empty captures count as missed frames.
- Recently observed tracks remain marked active.
- Track records are isolated by model version.

## Not established yet

- No trained model checkpoint is present in the supplied project ZIP.
- No real-world segmentation quality metric is established by the unit tests.
- Full Mask R-CNN inference on the RX 5500 XT has not been demonstrated.
- Passing isolated DirectML convolution, ROIAlign and NMS tests does not prove that the full model graph is supported.
- No camera-calibrated physical growth result has been validated.
- No controlled environmental-growth experiment or causal relationship is established.

## Model integration blocker

The current application model loader rebuilds TorchVision Mask R-CNN architectures from `.pth` state dictionaries. The separate official MushR port workflow exports a Detectron2 `TracingAdapter` TorchScript artifact (`.ts`) and metadata. Those are different runtime contracts. The TorchScript artifact must be loaded and its exact input/output tensor contract normalized into FungiCirno's `Prediction` interface before it can be installed as a runnable model. Do not rename the `.ts` file to `.pth` or pass it through the existing TorchVision state-dictionary loader.

The official MushR class-ID-to-name mapping must also be confirmed from the release's dataset metadata; it must not be guessed.

## Research reporting

Report the 9/9 tests as software/mechanics validation only. Keep model performance, hardware acceleration, tracking accuracy and biological interpretation as pending until measured on real images and the target AMD system.
