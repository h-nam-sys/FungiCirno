# FungiCirno — local mushroom monitoring

FungiCirno is a modular local application for organizing cultivation images, running instance segmentation, grouping detections, associating clusters across captures, measuring image-plane growth, reviewing predictions, and maintaining versioned models. The M18K-trained Ultralytics YOLO-seg family is the primary detection path. Charisis et al. are used as a methodological reference for mask-based temporal tracking; their Mask R-CNN checkpoint is not a runtime requirement.

This release is an **integrated implementation handoff** for the team. It includes the application modules, YOLO model adapter, local CPU and AMD DirectML device selection, tracking and measurement, environmental data, human mask/class review, reviewed-data export, candidate fine-tuning/evaluation scripts, model versioning/rollback controls, and member runbooks. It does not include the trained M18K weights or a validated side-view oyster-mushroom checkpoint. The team must supply those artifacts and run the acceptance workflow on the target PC.

## 1. System architecture

```text
Cultivation site / unit / camera
             |
             v
     Capture ingestion
             |
             v
  Image validation + SHA-256 dedupe
             |
             v
    Active model registry
             |
             v
  YOLOv8-seg inference (CPU / AMD)
             |
             v
  Prediction records + binary masks
             |
       +-----+------+
       |            |
       v            v
 Spatial groups   Temporal tracking
       |            |
       +-----+------+
             |
             v
   Growth / maturity records
             |
             +---- Environmental readings (CSV)
             |
             v
       SQLite + Streamlit
             |
             v
 Human review (accept / reject / class / mask correction)
             |
             v
 Reviewed YOLO-seg dataset export
             |
             v
 Colab fine-tuning + independent holdout evaluation
             |
             v
 Candidate model ZIP -> registry -> activation / rollback
```

### Model scope

- **M18K YOLOv8n-seg:** trained by the team using M18K V2. The existing WB-only checkpoint recognizes the White Button class; it is not a mushroom maturity classifier and has not been proven to generalize to side-view oyster mushrooms.
- **Side-view transfer learning:** fine-tune from the existing checkpoint using reviewed/public side-view labels. Report target-domain results only after evaluating on a separate sequence/farm-level holdout.
- **Charisis-inspired tracking:** the app associates masks using IoU, predicted centroid proximity, area similarity and Hungarian assignment. This is a project implementation inspired by the paper, not an exact reproduction.
- **Maturity estimation:** the app displays stage-labelled detections when the active model contains stage labels. A WB-only class does not provide maturity labels; do not infer readiness from area alone.

## 2. Install on Windows — AMD DirectML

Use native Windows and Python 3.12 for the pinned DirectML environment.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
python -m pip install -e .
```

If PowerShell blocks activation, use `cmd.exe`:

```bat
.venv\Scripts\activate.bat
```

Run the app:

```powershell
python -m streamlit run app.py
```

Select **AMD Radeon RX 5500 XT (DirectML)** under Run inference. If the complete YOLO graph or an operation is not supported by the installed DirectML stack, choose CPU explicitly. DirectML availability and isolated operator tests do not prove full-model execution.

## 3. Install CPU-only

CPU inference is part of the application contract and must remain available. In a fresh environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

On Linux, use the same CPU PyTorch wheel instruction with the platform-appropriate Python environment. DirectML is a native Windows backend.
Optional COCO RLE tooling for research dataset conversion: `pip install -r requirements-research.txt`. It is not required by the normal application.

## 4. Install an M18K YOLO model

The registry accepts a ZIP bundle containing:

```text
model_bundle.zip
  best.pt
  model_config.json
  class_mapping.json
  evaluation_metrics.json   # optional but strongly recommended
  README.txt                # optional
```

Create a bundle from an existing checkpoint:

```powershell
python tools/package_yolo_model.py --weights path\to\best.pt --out m18k_model_bundle.zip
```

For a target-domain model, specify the class taxonomy and species:

```powershell
python tools/package_yolo_model.py --weights runs\segment\train\weights\best.pt `
  --model-key m18k_yolo_seg --species "Pleurotus pulmonarius" `
  --taxonomy sideview_mushroom_instance --out sideview_candidate.zip
```

In FungiCirno, open **Models**, upload the ZIP, install it as a candidate, inspect its metrics, then activate it. Candidate activation can be blocked when the bundle explicitly records a failed/missing promotion gate. The UI exposes an explicit override for prototype use; do not use that to claim a validated model. The previous active version remains installed and can be reactivated for rollback.

## 5. Normal operation

1. Add a site, growing unit and camera in **Cultivation**.
2. Enter a camera-specific `mm_per_px` calibration only when a reliable known reference and fixed camera geometry are available. Otherwise, measurements remain in pixels.
3. Import captures in **Capture images**. Assign the same series key to images from the same camera sequence and provide capture times when EXIF is absent.
4. Activate a model in **Models**.
5. Run selected captures in **Run inference**. Choose AMD or CPU, image size and confidence threshold.
6. Inspect outputs in **Growth tracking** and **Maturity & grouping**.
7. Import sensor CSV files in **Environment**. Required timestamp column: `measured_at` or `timestamp`; optional columns: `temperature_c`, `humidity_pct`, `co2_ppm`, `notes`.
8. Review predictions in **Review & learning**. Accept, reject, correct class, or upload a corrected binary mask (white foreground on black background).
9. Use **Learning lab** to export reviewed data for Colab fine-tuning.

## 6. Incremental learning workflow

The local app does not train on the user's machine. It exports reviewed samples; training and evaluation happen in Colab or another GPU environment.

1. Review predictions; do not treat model output as ground truth without human inspection.
2. Export an Ultralytics YOLO-seg dataset from **Learning lab**. The export includes `images/`, `labels/`, `data.yaml` and a manifest.
3. Keep an independent holdout dataset separated by capture sequence, growing scene or farm. Do not reuse the training export as the final test set.
4. Run `tools/train_candidate.py` using the existing `best.pt` as initialization and the independent holdout YAML. The script measures baseline and candidate on the same holdout and writes `evaluation_metrics.json` into the candidate bundle.
5. Import the candidate bundle into **Models**. Activate only after reviewing the holdout metrics, sample overlays, class map and device execution. Keep the previous model for rollback.

Example Colab commands after transferring the files:

```bash
python tools/train_candidate.py \
  --weights m18k_wb_best.pt \
  --data reviewed_dataset/data.yaml \
  --holdout-data sideview_holdout.yaml \
  --source-holdout-data m18k_source_holdout.yaml \
  --epochs 50 --imgsz 640 --batch 8 --device 0 \
  --species 'Pleurotus pulmonarius' \
  --taxonomy sideview_mushroom_instance \
  --out sideview_candidate_bundle.zip
```

A candidate is promotion-ready only when it passes a target-domain minimum AP gate and retains source-domain performance within the configured regression tolerance. Both holdouts must be independent of training data. A candidate without either holdout is packaged as **not promotion-ready**.

## 7. Database and files

- SQLite database: `data/mushroom_monitor.sqlite3`
- Original captures: `data/captures/`
- Prediction masks: `data/masks/`
- Exports and reviewed datasets: `data/exports/`
- Installed models: `models/registry/<model_key>/<version>/`
- Incoming model ZIPs: `models/incoming/`

Back up the database and `data/` before changing releases. Database initialization performs additive schema creation and adds `flush_groups.model_version_id` to earlier databases when needed. Do not delete the database during an upgrade.

## 8. Main modules

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit application and workflows |
| `src/mushroom_monitor/db.py` | SQLite schema, transactions, migrations, model status |
| `src/mushroom_monitor/image_io.py` | Image validation, EXIF timestamps, deduplication, resizing |
| `src/mushroom_monitor/model_registry.py` | Bundle inspection, safe installation, version registry |
| `src/mushroom_monitor/inference/runtime.py` | YOLO-seg and TorchVision inference, CPU/DirectML device selection |
| `src/mushroom_monitor/service.py` | Inference orchestration, persistence, derived outputs |
| `src/mushroom_monitor/tracking.py` | Motion-assisted mask association and persistent cluster tracks |
| `src/mushroom_monitor/maturity.py` | Adaptive connected-component spatial grouping |
| `src/mushroom_monitor/environment.py` | CSV parsing and nearest-time growth/environment joins |
| `src/mushroom_monitor/review.py` | Human review, mask correction, COCO export |
| `src/mushroom_monitor/learning.py` | Reviewed YOLO dataset export |
| `tools/train_candidate.py` | Fine-tuning and same-holdout candidate comparison |
| `tools/prepare_reviewed_dataset.py` | Convert reviewed COCO ZIP to YOLO-seg format |
| `tools/package_yolo_model.py` | Package YOLO checkpoint for registry installation |

## 9. Runbook and ownership

- `docs/handoff/MINH_THINH.md`: M18K model lifecycle, transfer learning, local backend and incremental learning.
- `docs/handoff/THIEN_KHANG.md`: tracking, grouping, growth measurements and environmental analytics.
- `docs/handoff/HAI_NAM.md`: integration, database, app, system acceptance and release control.
- `docs/INTEGRATION_CONTRACTS.md`: shared data structures and file ownership.
- `docs/ACCEPTANCE_CHECKLIST.md`: whole-product validation.
- `docs/DECISIONS_AND_LIMITATIONS.md`: fixed decisions and claims the system must not make without evidence.

## 10. Data and research limitations

- M18K V2 WB-only metrics apply to the M18K held-out test split, not side-view oyster mushroom production images.
- A single WB class does not estimate maturity.
- Mask area is an image-plane measurement. Physical units require calibrated camera geometry and a stable reference plane.
- A correlation between environmental readings and growth is descriptive, not causal.
- Track association scores are heuristic similarities, not calibrated probabilities.
- The Charisis-inspired tracker is a local implementation; it must be evaluated on annotated sequences before reporting tracking accuracy.
- Human acceptance of a model-generated mask does not automatically make it ground truth; inspect it.

## 11. Licence and attribution

Cite M18K, Ultralytics, PyTorch, OpenCV, SciPy, Streamlit, the relevant datasets and all reused source implementations. Review the licences of datasets and model weights before redistribution. See `LICENSE` for this repository's code licence.
