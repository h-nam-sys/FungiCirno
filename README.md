# Mushroom Growth Monitor

Local Python application for image capture records, oyster-mushroom segmentation, cluster tracking, growth measurements, maturity grouping, environmental readings, human review and model version management.

## Run locally

1. Install Python 3.11 (recommended) and create a virtual environment.
2. Install a matching PyTorch and TorchVision pair from the [official selector](https://pytorch.org/get-started/locally/).
3. Run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

The application opens in a browser at the local Streamlit address. The database, uploaded images, masks, model bundles and exports stay under this directory by default. Set `MUSHROOM_APP_HOME` to move them elsewhere.

## Install the two Colab models

On each member's Google Drive, retrieve the `model_bundle.zip` created by that notebook and use **Models → Model bundle ZIP** in the app. The bundle must contain:

- `model_checkpoint.pth`
- `model_config.json`
- `class_mapping.json`

The app rebuilds the exact TorchVision architecture used in the notebooks:

- `charisis_cluster_tracking`: Mask R-CNN + ConvNeXt-Tiny-FPN; labels `immature_cluster`, `mature_cluster`, `overstayed_cluster`.
- `mushr_maturity`: Mask R-CNN ResNet-50 FPN; labels `Not-Ready`, `Ready`, `Overdue`.

The two taxonomies are kept separate. Both are oyster-mushroom models; recording another species does not make these models valid for that species.

**The result ZIPs supplied so far do not contain `.pth` checkpoints.** Thinh's notebook points to a model bundle in Drive. Khang's latest uploaded notebook skipped its bundle-export cell. The app therefore starts without trained models until their actual model bundles are imported. It does not convert smoke predictions into fake model weights.

## Hardware

- CPU inference is always available and is the fallback.
- On Windows, `auto` tries DirectML if `torch-directml` is installed; the first inference checks whether the required TorchVision operations work. If not, it retries on CPU.
- A ROCm-enabled PyTorch build is used automatically if `torch.cuda.is_available()` reports an accelerator.
- CUDA/NVIDIA can use the same model interface later.

DirectML is optional and experimental for these Mask R-CNN models. Keep a working CPU environment. Install `torch-directml` only in a separate test environment first; its PyTorch/Python compatibility must match the installed TorchVision build. The RX 5500 XT is DirectX 12 capable, but hardware detection alone does not prove that every Mask R-CNN operator runs on it.

The default inference long side is 768 pixels, batch size is one, and masks are stored at inference resolution. Each image is processed separately; tracking loads downsampled masks only for active tracks. Areas are scaled back to source-image pixel units. Physical mm² is only calculated after entering a camera-specific mm-per-pixel calibration.

## Workflow

1. Add a site, growing unit and camera under **Cultivation**.
2. Import side-view images under **Capture images**. Give images a series/batch key. Timestamps use the manual value first, then EXIF; missing capture times stay unknown.
3. Install and activate a model bundle under **Models**.
4. Run inference for the appropriate model under **Run inference**. Charisis predictions feed persistent per-sequence tracking; MushR predictions feed per-image DBSCAN flush grouping.
5. Review predictions under **Review & learning**. Confidence thresholds control automatic tracking/flush inclusion; only explicitly accepted or corrected samples can be exported as a COCO ZIP for a later Colab training run.
6. Enter environmental CSV readings under **Environment** and inspect time-aligned growth/environment records.

## Data formats

Environmental CSV headers: `measured_at` (or `timestamp`), `temperature_c`, `humidity_pct`, `co2_ppm`, optional `notes`.

Images: JPG/JPEG, PNG, TIFF, WebP and BMP. Images are de-duplicated by SHA-256.

## Important interpretation limits

- Charisis tracks mushroom clusters through time. MushR groups fruitbodies spatially within an image. A MushR flush ID is not a persistent temporal track.
- `area_px2` is an image-derived area estimate. It is not a physical size unless camera calibration is valid.
- The system's environment join is descriptive, not causal.
- Accepted detections and growth curves are model outputs, not ground truth. Human review records remain separate.
- This application is designed around the supplied oyster datasets. A broader species claim requires additional validated models and taxonomies.

## Diagnostics and tests

```bash
python tools/system_check.py
python -m unittest discover -s tests -v
```

The local tests cover database operations, capture deduplication, model-bundle ZIP installation, maturity grouping and multi-frame tracking. Both model builders also passed CPU forward tests with random weights. The Streamlit interface and AMD acceleration still need a run on the target desktop. No trained model-quality result is claimed by these software tests.
