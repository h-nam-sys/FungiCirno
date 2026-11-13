# FungiCirno — side-view oyster mushroom growth monitoring

FungiCirno is a local computer-vision application for monitoring *Pleurotus ostreatus* from timestamped, fixed-camera side-view images. Its core job is to turn repeated captures into **persistent mushroom-cluster growth histories**.

The primary production model is the member-trained **Charisis Mask R-CNN with ConvNeXt-Tiny-FPN**. It segments whole oyster clusters and classifies their developmental stage. FungiCirno then tracks those cluster detections across captures and measures changes in visible mask area.

## 1. What the core system does

1. Import timestamped images and associate them with a growing unit, camera and capture series.
2. Run Charisis instance segmentation to obtain one mask, box, class and confidence per cluster.
3. Persist predictions and masks in SQLite.
4. Associate clusters across captures with the application's temporal tracker.
5. Record each track's stage history and visible-area growth curve.
6. Associate environmental CSV readings with the growing unit and capture timeline.
7. Allow a human to review predictions and correct classes or masks.

**Core scope is cluster-level growth monitoring.** It does not require a separate individual-maturity model.

## 2. Primary model: Charisis

| Property | Value |
|---|---|
| Model family | `charisis_cluster_tracking` |
| Architecture | TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN |
| Runtime | `torchvision_state_dict` |
| Species | *Pleurotus ostreatus* |
| Detection unit | Whole mushroom cluster |
| Foreground class IDs | 1=`immature_cluster`, 2=`mature_cluster`, 3=`overstayed_cluster` |
| Background | Class ID 0, reserved by TorchVision |
| Downstream | Persistent cluster IDs, per-capture stage history and growth curves |

Preserve the class names and IDs from the trained notebook. In particular, its middle class is `mature_cluster`; do not silently rename it to `well-defined`. The source research describes the corresponding developmental phase using visible fruitbody characteristics, including stems.

### Model versus tracker

The Charisis neural network does **not** create persistent IDs. It predicts cluster instances independently in each image.

FungiCirno's application tracker consumes those masks and performs temporal association using mask overlap, predicted centroid proximity and area similarity, followed by one-to-one Hungarian assignment. Its track IDs and growth history are application outputs, not model predictions.

The member notebook also exported its own predicted tracks using a separate notebook-level tracker. Those 213 notebook tracks are not ground-truth identities and are not the same as the application's 22 tracks in the supplied database snapshot. Do not combine the two result sets.

## 3. Training and integration evidence

The executed member notebook completed a full 10-epoch training run on a Kaggle Tesla T4.

| Evidence | Result |
|---|---:|
| Source images | 6,382 |
| Source cluster annotations | 37,004 |
| Training images | 433 |
| Validation images | 109 |
| Held-out test images | 1,055 |
| Training/validation sequences | 2 and 4 |
| Held-out test sequences | 1 and 3 |
| Final training loss | 0.1449 |
| Final validation loss | 0.1649 |
| Mask AP@[0.50:0.95] | 0.650 |
| Mask AP@0.50 | 0.782 |
| Mask AP@0.75 | 0.759 |
| Test detections | 9,483 |
| Notebook-generated predicted tracks | 213 |

Images were sampled at one-hour cadence from the source time-lapse sequences. Training and test sets use disjoint sequences, reducing the risk of near-duplicate adjacent frames leaking across the split.

The supplied SQLite application snapshot additionally records 15 successful Charisis inference runs on AMD DirectML, with 96 candidate predictions, 43 passing the 0.20 analysis threshold, and 22 application tracks containing 43 observations. The small-sample median model inference time was approximately 0.56 seconds and median end-to-end run time approximately 2.15 seconds at a 768-pixel inference side. These observations show that the model was loaded and exercised in the supplied environment; they are **not** a controlled performance benchmark or a tracking-accuracy measurement.

No ground-truth persistent identity metric (IDF1, HOTA or MOTA) is present in the supplied training evidence. The 213 predicted notebook tracks and the 22 application tracks must not be presented as tracking accuracy.

See:
- `docs/MVP_VALIDATION.md` for the evidence and limitations.
- `docs/CHARISIS_TRACKING_METHOD.md` for the distinction between notebook tracking and the application tracker.
- `docs/ACCEPTANCE_CHECKLIST.md` for work still required.

## 4. Optional individual-maturity extension

A separate model is needed **only if the lead requires harvest-readiness estimates for each individual fruitbody**.

That is a different task from cluster-stage monitoring:

| Task | Object | Labels |
|---|---|---|
| Charisis core model | Whole cluster | `immature_cluster`, `mature_cluster`, `overstayed_cluster` |
| Optional individual maturity | Individual fruitbody | `not-ready`, `ready`, `overdue` |

The public MushR annotations can support that optional task because they provide individual-fruitbody maturity labels. The model would be trained and evaluated separately, with its own class mapping and per-image spatial grouping.

M18K does **not** supply those maturity labels. Its existing checkpoint is a one-class White Button (`WB`) YOLO segmentation model. It cannot determine oyster mushroom maturity without training on appropriately labelled oyster data. It is not required for the core Charisis pipeline.

Do not add or train the optional maturity model until individual harvest readiness is confirmed as a project requirement.

## 5. M18K's status

M18K V2 is retained as a historical White Button segmentation experiment and generic pipeline reference.

- Architecture: Ultralytics YOLOv8n-seg.
- Class: `WB` only.
- Its reported AP and area metrics apply only to its own M18K White Button test split.
- It is not the oyster production model.
- Its YOLO checkpoint cannot initialize the TorchVision Mask R-CNN Charisis model directly.

The application still supports Ultralytics `.pt` bundles for optional or historical experiments, but the primary Charisis checkpoint uses the TorchVision `.pth` loader.

## 6. Install and run

Use Python 3.12 where possible. Model training is offline in Kaggle/Colab; the local application performs inference, persistence, review and model management.

### Windows CPU

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

### Windows AMD DirectML

Use the supplied AMD dependency file and setup script:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
python -m pip install -e .
python -m streamlit run app.py
```

CPU inference is the required fallback. AMD DirectML must be tested using the actual installed Charisis bundle on the target PC; an operator smoke test alone is not enough. The supplied database contains prior AMD DirectML run evidence, but a clean-install retest is still required.

## 7. Install the trained Charisis model

**The project ZIP does not contain the binary `model_checkpoint.pth`.** The executed notebook generated `model_bundle.zip` in the Kaggle notebook output directory. Download that bundle separately; do not assume the SQLite snapshot's old absolute Windows checkpoint path will exist on a clean checkout.

Expected bundle:

```text
model_bundle.zip
├── model_checkpoint.pth
├── model_config.json
├── class_mapping.json
├── evaluation_metrics.json       # included when exported by the updated final notebook cell
└── MODEL_IMPLEMENTATION_NOTE.txt
```

The model registry recognizes the ConvNeXt-Tiny-FPN TorchVision bundle and canonicalizes it to `charisis_cluster_tracking`. The original `legacy_charisis_maskrcnn` database alias remains recognized for backward compatibility.

1. Open the local application and navigate to **Models**.
2. Upload the downloaded `model_bundle.zip`.
3. Confirm the class map preserves foreground IDs 1, 2 and 3.
4. Install it as a candidate.
5. Run a representative capture and inspect masks, boxes, classes and confidence.
6. Activate the candidate only after checking the output.

The updated notebook is at `notebooks/Charisis_Full_Training.ipynb`. Its final export cell writes canonical FungiCirno metadata and includes the full evaluation metrics file when available. The source training run's outputs are preserved in the notebook. Re-run the final export cell in the original live Kaggle session if you need to regenerate the ZIP with the updated metadata. Do not rerun the ten-epoch training solely to change package metadata if the saved checkpoint and Kaggle session artifacts are still available.

## 8. Normal operation

1. Create the site, growing unit and fixed side-view camera under **Cultivation**.
2. Assign captures to one `series_key` per camera/view and growing unit.
3. Import images with capture timestamps.
4. Activate `charisis_cluster_tracking` and process captures in chronological order.
5. Use **Growth tracking** to inspect cluster IDs, stage progression and visible-area curves.
6. Import environmental readings from CSV (`measured_at`, `temperature_c`, `humidity_pct`, `co2_ppm`, optional `notes`).
7. Review or correct model predictions under **Review & learning**.

The **Maturity & grouping** page is for a separately trained individual-fruitbody model, if that optional scope is approved. Charisis cluster detections must not be grouped again as individual fruitbodies.

## 9. Measurement and claim limits

- `area_px2` is projected visible 2D mask area in source-image pixels.
- Physical area requires valid camera-specific calibration and stable camera geometry.
- Pixel area is not 3D surface area, volume, mass or biomass.
- A class transition is the model's observed label change between captures, not the exact biological transition time.
- A missed track or track termination does not prove that a cluster was harvested.
- Association scores are heuristic similarities, not probabilities.
- Environmental alignment is descriptive; it does not establish causality.
- Do not report tracking accuracy without ground-truth persistent IDs.

## 10. Project structure

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit UI and workflows |
| `src/mushroom_monitor/db.py` | SQLite schema and persistence |
| `src/mushroom_monitor/image_io.py` | Capture ingestion, timestamping, hashing and preprocessing |
| `src/mushroom_monitor/model_registry.py` | Model bundle installation, metadata, versions and activation |
| `src/mushroom_monitor/inference/models.py` | TorchVision Mask R-CNN model builders |
| `src/mushroom_monitor/inference/runtime.py` | TorchVision and optional Ultralytics runtime adapters with normalized predictions |
| `src/mushroom_monitor/service.py` | Inference orchestration, source-image scaling, persistence and downstream dispatch |
| `src/mushroom_monitor/tracking.py` | Cluster mask/centroid/area Hungarian association |
| `src/mushroom_monitor/maturity.py` | Optional per-capture individual-fruitbody spatial grouping |
| `src/mushroom_monitor/environment.py` | Environmental CSV parsing and time alignment |
| `src/mushroom_monitor/review.py`, `learning.py` | Human review, mask correction and reviewed-data export |
| `notebooks/Charisis_Full_Training.ipynb` | Primary Charisis training/evaluation workflow |
| `notebooks/FungiCirno_Oyster_Model_Training.ipynb` | Historical M18K/YOLO experiment |

## 11. Team documentation

- `START_HERE_TEAM.md`: team entry point.
- `docs/FINAL_SYSTEM_ARCHITECTURE.md`: architecture contract.
- `docs/INTEGRATION_CONTRACTS.md`: shared schemas and ownership boundaries.
- `docs/ACCEPTANCE_CHECKLIST.md`: outstanding acceptance criteria.
- `docs/handoff/`: member responsibilities.
- `docs/ai_handoff/`: coding-AI prompts.

Keep model weights, raw datasets, private captures, credentials and local runtime artifacts out of Git. Record commands, checkpoint hashes, dataset/split IDs, backend, inference time, memory, sample overlays and failures. Do not mark work complete from code presence alone.
