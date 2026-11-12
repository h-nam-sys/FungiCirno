# FungiCirno — oyster mushroom growth monitoring

FungiCirno is a local computer-vision application for fixed-camera, side-view monitoring of oyster mushrooms (*Pleurotus ostreatus*). It combines instance segmentation, per-capture stage labels, persistent cluster tracking, visible-area growth measurement, individual maturity estimation, human review, environmental context, and offline model improvement.

**Important status:** the application and training/integration workflows are implemented in this release. The trained M18K White Button checkpoint is a source-domain starting point, not the production oyster model. Trained Charisis-taxonomy and MushR-taxonomy oyster checkpoints are not included. CPU and AMD DirectML end-to-end execution, target-domain metrics, sequence tracking metrics, and a full learning/rollback run still require real-data and target-hardware evidence. See `docs/FINAL_SYSTEM_ARCHITECTURE.md` and `docs/ACCEPTANCE_CHECKLIST.md`.

## 1. Final system design

One Ultralytics YOLOv8-seg inference implementation serves two separately trained model families. Both produce the same normalized `Prediction` records, so the database, review workflow, tracking, grouping, and dashboard do not depend on the model architecture.

| Model family | Detection unit | Class taxonomy | Downstream role |
|---|---|---|---|
| `charisis_cluster_tracking` | Whole oyster cluster | `immature`, `well-defined`, `overstayed` | Primary cluster tracking, stage history, growth curves |
| `mushr_maturity` | Individual oyster fruitbody | `not-ready`, `ready`, `overdue` | Individual harvest maturity and spatial flush summaries |
| `m18k_source_wb` | White Button mushroom instance | `WB` | Source-domain checkpoint and integration/debug baseline only |

Do not merge these taxonomies. The Charisis model predicts a cluster-level development stage. The MushR model predicts individual fruitbody harvest readiness. M18K predicts only White Button instances. A detection's area must never be used as a substitute for a maturity classifier.

### Architecture

```text
Site / growing unit / fixed side-view camera
                    |
                    v
 Timestamped capture ingestion + SHA-256 deduplication
                    |
                    v
             Model registry
                    |
          +---------+----------+
          |                    |
          v                    v
 Charisis cluster model    MushR fruitbody model
 YOLOv8-seg                YOLOv8-seg
 3 cluster stages          3 maturity classes
          |                    |
          v                    v
     Normalized Prediction contract
                    |
          +---------+----------+
          |                    |
          v                    v
 Cluster tracking        Spatial grouping
 Stage history           Ready-area fraction
 Growth measurements    Flush maturity summary
          |                    |
          +---------+----------+
                    |
                    v
       SQLite persistence + Streamlit UI
                    |
                    v
 Human review / class and mask correction
                    |
                    v
 Reviewed YOLO dataset export
                    |
                    v
 Colab fine-tuning + independent holdout evaluation
                    |
                    v
 Candidate bundle -> registry -> approval / rollback
```

The full component decisions, data contracts, failure boundaries and research limitations are in `docs/FINAL_SYSTEM_ARCHITECTURE.md`.

## 2. Model and dataset plan

### A. Primary cluster-stage model

Use the public Charisis et al. oyster cluster time-lapse dataset (DOI `10.5281/zenodo.15041243`). It contains side-view cluster masks and the three cluster-level labels used by the paper. Preserve complete time-lapse trials/sequences when splitting data; adjacent frames from one trial must not leak across train, validation and test.

Train two initializations on the same target dataset and split:

1. Team M18K `best.pt` as an initialization candidate. Its one-class WB head is replaced/adapted to the three oyster cluster classes; compatible feature weights may transfer, but this does not establish that the M18K model is better.
2. COCO-pretrained `yolov8n-seg.pt` as the comparison initialization.

Compare both on the same untouched oyster test sequences. Select based on target-domain segmentation quality and qualitative mask/stage errors, not source-domain M18K AP.

### B. Individual maturity model

Use the public MushR annotated oyster mushroom dataset and its three labels: `not-ready`, `ready`, and `overdue`. This model is a separate task and checkpoint. It reuses the same YOLO-seg runtime and prediction schema. Spatial grouping estimates per-flush readiness by aggregating predicted mask area by class; that summary is not a separately trained classifier.

MushR data are linked from `https://github.com/ETCE-LAB/MushR`. Inspect the supplied annotation format and its dataset terms before conversion. `tools/prepare_coco_yolo_dataset.py` converts COCO instance-segmentation annotations into YOLO-seg format when the downloaded annotations are COCO-compatible. Do not assume an annotation format without inspecting it.

### C. M18K's correct role

The current M18K checkpoint is team-trained YOLOv8n-seg on M18K V2, WB-only class ID `0`. Its reported metrics apply only to the White Button held-out test split. Keep its weights and evaluation record separate. It can seed a target transfer experiment, but must not be deployed as the oyster model or assigned oyster stage names without target-domain training.

Wang's seven-stage *Pleurotus pulmonarius* taxonomy is a separate research reference. It is not the production label map for this *P. ostreatus* system.

## 3. Install and run

Use Python 3.12. Training runs offline in Google Colab; the local application performs inference and management.

### Windows CPU

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

### Windows AMD DirectML

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
python -m pip install -e .
python -m streamlit run app.py
```

Alternatively run `setup_windows_cpu.ps1` or `setup_windows_amd.ps1`, then `run_windows.bat`.

CPU is the guaranteed fallback. AMD DirectML is the primary optional accelerator, but full Ultralytics YOLO execution on the RX 5500 XT must be tested with the actual bundle. Run `tools/test_amd_directml.py --weights <model.pt> --image <capture.jpg>` to attempt a real YOLO segmentation forward pass on DirectML; this smoke test does not establish accuracy or sustained stability. Never label CPU execution as AMD execution. NVIDIA is future work.

## 4. Convert a public COCO dataset

When the source provides separate COCO JSON files and a common image root:

```bash
python tools/prepare_coco_yolo_dataset.py \
  --images /content/source_images \
  --train-json /content/annotations/train.json \
  --val-json /content/annotations/val.json \
  --test-json /content/annotations/test.json \
  --class-order immature,well-defined,overstayed \
  --out /content/charisis_yolo
```

The converter produces `images/`, `labels/`, `data.yaml`, and `dataset_manifest.json`. Inspect converted masks/labels. It reduces each instance mask to its largest external contour because a YOLO-seg label row represents one polygon. If the source annotations are not COCO-compatible or use multiple disconnected components/holes that matter, write and validate a dataset-specific conversion rather than silently accepting that loss.

For MushR, use the correct class order:

```bash
--class-order not-ready,ready,overdue
```

Only use a `test` split that is genuinely held out by full sequence/camera/growing-scene or equivalent independent grouping. The converter does not create a leakage-free split for you; it preserves the three source splits supplied to it.

## 5. Train the primary oyster cluster model in Colab

Run both initializations independently on the **same** target dataset, training split, validation split, and untouched test split.

M18K initialization:

```bash
python tools/train_candidate.py \
  --weights /content/best.pt \
  --data /content/charisis_yolo/data.yaml \
  --holdout-data /content/charisis_yolo/data.yaml --holdout-split test \
  --holdout-id charisis_trial_holdout_v1 \
  --model-key charisis_cluster_tracking \
  --species 'Pleurotus ostreatus' \
  --taxonomy charisis_oyster_cluster_stage_v1 \
  --prediction-role cluster_stage \
  --pretrained-source 'team M18K V2 WB-only YOLOv8n-seg checkpoint' \
  --epochs 50 --imgsz 640 --batch 8 --device 0 \
  --out /content/charisis_m18k_init_candidate.zip
```

COCO-pretrained comparison:

```bash
python tools/train_candidate.py \
  --weights yolov8n-seg.pt \
  --data /content/charisis_yolo/data.yaml \
  --holdout-data /content/charisis_yolo/data.yaml --holdout-split test \
  --holdout-id charisis_trial_holdout_v1 \
  --model-key charisis_cluster_tracking \
  --species 'Pleurotus ostreatus' \
  --taxonomy charisis_oyster_cluster_stage_v1 \
  --prediction-role cluster_stage \
  --pretrained-source 'Ultralytics COCO-pretrained YOLOv8n-seg' \
  --epochs 50 --imgsz 640 --batch 8 --device 0 \
  --out /content/charisis_coco_init_candidate.zip
```

For the first transfer, omit `--baseline-weights`: the one-class WB model and three-class oyster-stage model have different taxonomies, so their AP numbers are not comparable. Compare the two resulting candidates on the same target test set outside the training loop and choose one target initialization. For later updates to an already trained Charisis model, pass the active same-taxonomy model as both `--weights` and `--baseline-weights` so baseline and candidate are measured on the same untouched holdout.

## 6. Train the individual MushR maturity model

Prepare its annotated data using the original dataset split and class mapping. Then train as a separate model family, for example:

```bash
python tools/train_candidate.py \
  --weights yolov8n-seg.pt \
  --data /content/mushr_yolo/data.yaml \
  --holdout-data /content/mushr_yolo/data.yaml --holdout-split test \
  --holdout-id mushr_test_v1 \
  --model-key mushr_maturity \
  --species 'Oyster mushroom (MushR dataset)' \
  --taxonomy mushr_oyster_fruitbody_maturity_v1 \
  --prediction-role fruitbody_maturity \
  --pretrained-source 'Ultralytics COCO-pretrained YOLOv8n-seg' \
  --epochs 50 --imgsz 640 --batch 8 --device 0 \
  --out /content/mushr_candidate.zip
```

Do not use the Charisis model as a baseline for MushR or vice versa. Their objects and class meanings differ.

## 7. Package an existing checkpoint

```bash
python tools/package_yolo_model.py \
  --weights runs/segment/train/weights/best.pt \
  --out charisis_cluster_bundle.zip \
  --model-key charisis_cluster_tracking \
  --species 'Pleurotus ostreatus' \
  --taxonomy charisis_oyster_cluster_stage_v1 \
  --prediction-role cluster_stage \
  --pretrained-source 'M18K transfer or COCO initialization'
```

For Model B, use `--model-key mushr_maturity`, `--prediction-role fruitbody_maturity`, and `--taxonomy mushr_oyster_fruitbody_maturity_v1`. The candidate training script already emits a bundle with model metadata and evaluation metrics; do not re-package unless metadata needs correction.

Upload the resulting ZIP in **Models**. Install as a candidate, inspect class mapping and `evaluation_metrics.json`, then explicitly activate it. A prior active version remains available for rollback. Never promote without an untouched target-domain evaluation.

## 8. Normal operation

1. Create the site, growing unit and fixed camera in **Cultivation**.
2. Enter camera-specific `mm_per_px` only when measured from a reliable reference with unchanged camera geometry.
3. Import timestamped images in **Capture images**. Every sequence must share one `series_key` and camera/view.
4. Activate `charisis_cluster_tracking` and run inference across the ordered capture series. Review stage predictions, masks, and confidence.
5. Use **Growth tracking** for stable cluster IDs, stage progression, visible area, and relative area change.
6. Run `mushr_maturity` separately when individual fruitbody readiness/flush summaries are needed; inspect **Maturity & grouping**.
7. Import environmental CSV (`measured_at`, `temperature_c`, `humidity_pct`, `co2_ppm`, optional `notes`).
8. Review and correct predictions in **Review & learning**. Use **Learning lab** to export reviewed data for the matching model family.

## 9. Human-reviewed learning and rollback

1. The human inspects the source image, class and mask. Only genuinely inspected accepted items and corrected items are exported.
2. Export the reviewed dataset for that model family. Do not mix class taxonomies.
3. Keep a final target holdout out of the review/training export. Split by full sequence/farm/camera group, not adjacent frames.
4. Fine-tune in Colab from the currently active checkpoint. Evaluate active baseline and candidate on the same untouched holdout only when taxonomy and test data are identical.
5. Import the candidate. Require target holdout metrics and visual checks. Keep the prior active model installed.
6. Activate explicitly; verify that rollback restores the prior model and that old results remain version-scoped.

## 10. What is measured and what is not

- `area_px2` is visible 2D mask area in source-image pixels.
- `area_mm2` is available only with valid camera calibration; neither is 3D volume, mass, or biomass.
- A stage change is the model's observed class change at capture times, not the exact biological transition time.
- Track association score is a heuristic, not a probability.
- Spatial grouping is per-capture and does not create persistent flush identities.
- Flush readiness is a descriptive aggregate of individual predictions, not a separately trained classifier.
- Environmental alignment/correlation is descriptive, not causal. Live sensor drivers are not included.

## 11. Project structure

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit UI and workflows |
| `src/mushroom_monitor/db.py` | SQLite schema and persistence |
| `src/mushroom_monitor/image_io.py` | Capture ingestion, timestamping, hashing, preprocessing |
| `src/mushroom_monitor/model_registry.py` | Model bundle install, metadata, versions and activation |
| `src/mushroom_monitor/inference/runtime.py` | Shared YOLO-seg inference and legacy TorchVision compatibility |
| `src/mushroom_monitor/service.py` | Normalization, inference orchestration, persistence and derived outputs |
| `src/mushroom_monitor/tracking.py` | Mask/centroid/area Hungarian association and persistent tracks |
| `src/mushroom_monitor/maturity.py` | Per-capture spatial grouping and class-area summaries |
| `src/mushroom_monitor/environment.py` | Environmental CSV parsing and time alignment |
| `src/mushroom_monitor/review.py`, `learning.py` | Human review, mask correction and reviewed-data export |
| `tools/prepare_coco_yolo_dataset.py` | COCO instance-mask to YOLO-seg conversion |
| `tools/train_candidate.py` | Offline training, holdout evaluation and candidate bundle |
| `tools/package_yolo_model.py` | YOLO model bundle creation |

## 12. Team validation

- `START_HERE_TEAM.md`: short team entry point.
- `docs/FINAL_SYSTEM_ARCHITECTURE.md`: architecture contract.
- `docs/INTEGRATION_CONTRACTS.md`: shared schemas and ownership boundaries.
- `docs/ACCEPTANCE_CHECKLIST.md`: executable system acceptance criteria.
- `docs/handoff/`: member validation responsibilities.
- `docs/ai_handoff/`: ready-to-paste coding-AI prompts.

Do not mark features complete from code presence alone. Record commands, model hashes, data/split IDs, backend, inference time, RSS, example masks, tracking errors and failure logs. Do not commit weights, raw datasets, local captures, masks or databases.
