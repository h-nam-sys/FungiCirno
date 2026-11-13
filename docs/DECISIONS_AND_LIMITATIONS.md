# Architecture decisions and claim limits

## Current decisions

1. **Primary production model:** Charisis-trained TorchVision Mask R-CNN with ConvNeXt-Tiny-FPN, operating on whole oyster mushroom clusters.
2. **Primary classes:** preserve the notebook's exact labels and IDs: `1=immature_cluster`, `2=mature_cluster`, `3=overstayed_cluster`; TorchVision reserves 0 for background.
3. **Primary temporal logic:** the FungiCirno application performs its own mask/centroid/area association with Hungarian assignment. This is not the notebook's tracker or an exact reproduction of Charisis et al.
4. **Core output:** persistent cluster IDs, per-capture predicted stage, visible mask area and relative growth history.
5. **Optional second task:** individual fruitbody harvest maturity using MushR classes only if the lead explicitly requires per-mushroom readiness. It is not required for the core cluster-growth pipeline.
6. **M18K:** historical White Button YOLO benchmark/reference. It is not the oyster production model and is not a required Charisis initialization.
7. **Shared application interface:** backend-specific loaders feed one normalized prediction record and the same persistence/review/tracking services. Do not force the Charisis checkpoint into the YOLO loader.
8. **Training:** offline notebook. The local app runs inference and manages data/review/model versions; it does not train.

## Evidence currently available

- Charisis full run: 10 epochs; 433 train images, 109 validation images, 1,055 held-out test images; test sequences are disjoint from training/validation sequences.
- Overall mask AP@[0.50:0.95]=0.650, AP50=0.782, AP75=0.759.
- Notebook test inference: 9,483 detections and 213 predicted tracks. No ground-truth track IDs/identity metric is provided.
- Supplied app database: 15 successful Charisis AMD DirectML runs, 96 candidate predictions, 43 analysis-eligible predictions at threshold 0.20, 22 application tracks and 43 observations. The active legacy row reports checkpoint SHA-256 `3aabb8c4a9ed6016c29b80008afe6dc1d4a1c454392bebaac3fc110c0c79f2d8`.
- Local app median model inference ≈0.56 s and median end-to-end run ≈2.15 s over that small sample. Not a benchmark.

## Taxonomy boundaries

- Charisis cluster task: `immature_cluster`, `mature_cluster`, `overstayed_cluster`.
- MushR individual-maturity task: `not-ready`, `ready`, `overdue`.
- M18K source task: `WB` only.

Do not mix labels or class IDs across model families. A cluster-stage class does not establish the maturity of every individual fruitbody inside that cluster. Mask area is not a maturity label.

## Measurement and evaluation limits

- `area_px2` is projected visible 2D mask area.
- `area_mm2` requires valid camera-specific calibration and stable reference geometry.
- Relative area change is not biomass, volume, mass or full 3D growth.
- Association score is a heuristic, not a probability.
- The notebook's 213 tracks and the app's 22 tracks are outputs of different tracking runs/implementations; neither number is a tracking-accuracy metric.
- No IDF1, HOTA or MOTA claim is valid without verified ground-truth identity annotations.
- The 15 app runs in the DB demonstrate that a Charisis TorchVision checkpoint was loaded through AMD DirectML in the recorded local environment. The checkpoint file is absent from the source archive, so the DB hash cannot be independently compared with the attached notebook's Kaggle output bundle. This is functional integration evidence, not an independent deployment benchmark.
- The notebook's training/validation loss and COCO mask AP do not alone prove correct biological maturity estimation in a new farm/camera setting.

## Remaining evidence dependencies

- CPU inference run with the actual Charisis checkpoint and current environment.
- Reproducible AMD DirectML test on the actual checkpoint after a clean installation; existing database records show successful sample runs.
- Manually verified track-ID subset and tracking metrics, or explicit report that public ground-truth IDs are unavailable.
- Manual growth measurement comparison and camera calibration evidence.
- Full sequence run in the application, not only isolated images.
- A decision from the lead on whether individual fruitbody harvest-readiness is required. If not, defer MushR.
- Charisis-aware reviewed-sample fine-tuning integration if incremental learning is required; the current generic COCO export is not yet wired directly into the training notebook.

## Packaging limitation

The supplied project ZIP and notebook do not contain the binary `model_checkpoint.pth`. The notebook indicates that it writes `model_bundle.zip` to Kaggle's output directory. Keep the downloaded bundle/checkpoint separately and install it through the Models page. Do not claim a clean archive includes model weights when it does not.
