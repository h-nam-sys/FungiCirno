$ErrorActionPreference = 'Stop'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
python -m pip install -e .
Write-Host "AMD environment installed. Run: python -m streamlit run app.py"
