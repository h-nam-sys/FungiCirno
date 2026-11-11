$ErrorActionPreference = 'Stop'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
Write-Host "CPU environment installed. Run: python -m streamlit run app.py"
