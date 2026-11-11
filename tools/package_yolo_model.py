#!/usr/bin/env python
"""Package an Ultralytics segmentation checkpoint for the FungiCirno registry."""
from __future__ import annotations
import argparse,json,zipfile,hashlib
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',required=True); ap.add_argument('--out',default='m18k_model_bundle.zip'); ap.add_argument('--model-key',default='m18k_yolo_seg'); ap.add_argument('--species',default='Agaricus bisporus (M18K White Button)'); ap.add_argument('--taxonomy',default='m18k_mushroom_instance'); ap.add_argument('--version',default=''); ap.add_argument('--confidence',type=float,default=.25); ap.add_argument('--nms-iou',type=float,default=.70); ap.add_argument('--metrics',default='')
    a=ap.parse_args(); w=Path(a.weights).resolve()
    if not w.exists(): raise FileNotFoundError(w)
    model=YOLO(str(w),task='segment'); names={int(k):str(v) for k,v in model.names.items()}; digest=hashlib.sha256(w.read_bytes()).hexdigest()
    config={'model_key':a.model_key,'version':a.version or f'm18k-{digest[:12]}','architecture':'Ultralytics YOLOv8n-seg','runtime_type':'ultralytics_yolo_seg','task':'segment','source_species':a.species,'taxonomy':a.taxonomy,'names':{str(k):v for k,v in names.items()},'confidence_threshold':a.confidence,'nms_iou':a.nms_iou,'max_det':1000}
    out=Path(a.out).resolve(); out.parent.mkdir(parents=True,exist_ok=True); metrics=Path(a.metrics) if a.metrics else None
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.write(w,'best.pt'); z.writestr('model_config.json',json.dumps(config,indent=2)); z.writestr('class_mapping.json',json.dumps({'label_to_name':{str(k):v for k,v in names.items()}},indent=2))
        if metrics and metrics.exists(): z.write(metrics,'evaluation_metrics.json')
        z.writestr('README.txt',f"FungiCirno YOLO-seg model bundle\nVersion: {config['version']}\nCheckpoint SHA256: {digest}\nCreated: {datetime.now().isoformat()}\nValidate on an independent holdout and target hardware before deployment.\n")
    print(out)
if __name__=='__main__': main()
