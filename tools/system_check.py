from __future__ import annotations
import platform,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from mushroom_monitor.config import AppPaths
from mushroom_monitor.db import Database
paths=AppPaths.from_env(); db=Database(paths.database); db.ensure_defaults()
print('OS:',platform.platform()); print('Python:',sys.version.split()[0]); print('App home:',paths.root); print('SQLite:',paths.database)
try:
 import psutil
 print('RAM total GB:',round(psutil.virtual_memory().total/1024**3,2)); print('RAM available GB:',round(psutil.virtual_memory().available/1024**3,2)); print('Process RSS MB:',round(psutil.Process(os.getpid()).memory_info().rss/1024**2,1))
except Exception as exc: print('psutil unavailable:',exc)
try:
 import torch,torchvision
 print('PyTorch:',torch.__version__,'TorchVision:',torchvision.__version__,'CUDA:',torch.cuda.is_available())
except Exception as exc: print('PyTorch/TorchVision unavailable:',repr(exc))
try:
 import ultralytics
 print('Ultralytics:',ultralytics.__version__)
except Exception as exc: print('Ultralytics unavailable:',repr(exc))
try:
 import torch_directml
 print('DirectML device:',torch_directml.device())
except Exception as exc: print('DirectML unavailable:',repr(exc))
print('Registered models:')
for row in db.query('SELECT id,model_key,version,status,architecture,checkpoint_path FROM model_versions ORDER BY id'):
 cp=Path(row['checkpoint_path']); print(f" #{row['id']} {row['model_key']} {row['version']} [{row['status']}] checkpoint_exists={cp.exists()}")
print('Captures:',db.one('SELECT COUNT(*) n FROM captures')['n'])
print('Predictions:',db.one('SELECT COUNT(*) n FROM predictions')['n'])
print('Tracks:',db.one('SELECT COUNT(*) n FROM tracks')['n'])
print('Environment readings:',db.one('SELECT COUNT(*) n FROM environmental_readings')['n'])
print('System check complete. End-to-end inference must be run in the app; this report does not claim accelerator verification.')
