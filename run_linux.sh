#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Create a CPU PyTorch environment first; see README.md."
  exit 1
fi
source .venv/bin/activate
python -m streamlit run app.py
