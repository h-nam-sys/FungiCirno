# Local runtime setup — Charisis TorchVision Mask R-CNN

FungiCirno's primary model is the Charisis TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN checkpoint. The application uses the same runtime adapter for CPU and AMD DirectML, selected explicitly on the **Run inference** page.

## AMD DirectML (native Windows, CPython 3.12)

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
python -m pip install -e .
python -m streamlit run app.py
```

`torch-directml==0.2.5.dev240914` pins the project's tested PyTorch/TorchVision compatibility family. Do not independently upgrade PyTorch or TorchVision without rerunning full-model tests.

## CPU fallback

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

CPU inference is mandatory. AMD DirectML is an optional accelerator and must never silently fall back while being reported as GPU inference.

## Verify the actual Charisis checkpoint

1. Download `model_bundle.zip` from the executed Charisis notebook's Kaggle output.
2. Import it through **Models**.
3. Select AMD DirectML on **Run inference** and process a representative side-view capture.
4. Confirm the run diagnostics explicitly report `AMD DirectML`, the expected `torchvision_state_dict` runtime, masks/classes and no CPU fallback.
5. Save the inference-run ID, output overlay, timing and process memory. Repeat on CPU and compare.
6. Test a multi-capture sequence as well as one image.

The supplied SQLite snapshot already records 15 successful Charisis AMD DirectML application runs. Their median inference/run times are small-sample observations, not a controlled benchmark. A clean-install retest is still required.

**Important:** `tools/test_amd_directml.py` is specifically a YOLO `.pt` smoke test. It does not validate the primary Charisis TorchVision `.pth` model. Use the real FungiCirno application path for Charisis acceptance.

If DirectML fails, preserve the full error and use CPU explicitly. Never label CPU execution as AMD execution.
