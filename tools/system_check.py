from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mushroom_monitor.config import AppPaths
from mushroom_monitor.db import Database

paths = AppPaths.from_env()
db = Database(paths.database)
db.ensure_defaults()
print("OS:", platform.platform())
print("Python:", sys.version.split()[0])
print("App home:", paths.root)
print("SQLite:", paths.database)
try:
    import torch
    import torchvision
    print("PyTorch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("CUDA/ROCm available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Torch accelerator:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch/TorchVision not installed or not importable:", repr(exc))
try:
    import torch_directml
    print("DirectML device:", torch_directml.device())
except Exception as exc:
    print("DirectML unavailable:", str(exc))
print("Registered models:")
for row in db.query("SELECT id,model_key,version,status,checkpoint_path FROM model_versions ORDER BY id"):
    exists = Path(row["checkpoint_path"]).exists()
    print(f"  #{row['id']} {row['model_key']} {row['version']} [{row['status']}] checkpoint_exists={exists}")
print("Captures:", db.one("SELECT COUNT(*) AS n FROM captures")["n"])
print("System check complete. Hardware inference is only verified after a real model run.")
