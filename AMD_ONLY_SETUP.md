# Local runtime setup — AMD DirectML and CPU

FungiCirno supports two explicit local backends:

- **AMD DirectML** on native Windows, targeting the RX 5500 XT.
- **CPU** as the always-available compatibility backend.

The app does not silently claim a GPU run when a CPU backend was used. Select the backend on the Run inference page; the selected backend is recorded with each inference run. Ultralytics/DirectML operator coverage must be checked against the full model, not inferred from a small tensor smoke test.

## AMD (Windows / CPython 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
python -m pip install -e .
python -m streamlit run app.py
```

`torch-directml==0.2.5.dev240914` pins the supported PyTorch/TorchVision pair for this project. Do not independently upgrade PyTorch after installing DirectML without revalidating the backend.

## CPU (Windows or Linux)

Install CPU PyTorch first, then the application requirements:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

## Diagnostics

```powershell
python tools/system_check.py
python tools/test_amd_directml.py
```

The operator diagnostic checks selected operations only. Full trained-model inference is the acceptance criterion. If DirectML fails, select CPU explicitly and retain the exception in the inference run history.
