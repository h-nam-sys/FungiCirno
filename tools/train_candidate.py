#!/usr/bin/env python
"""Fine-tune an existing YOLO segmentation checkpoint and package a candidate.

Promotion is never automatic. A separate untouched holdout dataset is required to
write promotion_ready=true. This script compares base and candidate on that same holdout.
"""
from __future__ import annotations
import argparse, json, shutil, zipfile
from pathlib import Path
from datetime import datetime, timezone
import yaml
from ultralytics import YOLO

def metric_values(model:YOLO,data:str,imgsz:int,device:str)->dict:
    result=model.val(data=data,imgsz=imgsz,device=device,plots=True,verbose=False)
    seg=getattr(result,'seg',None); box=getattr(result,'box',None)
    return {'seg_map50':float(getattr(seg,'map50',0.0)),'seg_map50_95':float(getattr(seg,'map',0.0)),
            'box_map50':float(getattr(box,'map50',0.0)),'box_map50_95':float(getattr(box,'map',0.0))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',required=True,help='Existing trained M18K YOLOv8-seg .pt checkpoint')
    ap.add_argument('--data',required=True,help='Reviewed YOLO data.yaml for fine-tuning train/val')
    ap.add_argument('--holdout-data',default='',help='Independent side-view target-domain holdout data.yaml')
    ap.add_argument('--source-holdout-data',default='',help='Independent M18K/source-domain regression holdout data.yaml')
    ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--imgsz',type=int,default=640)
    ap.add_argument('--batch',type=int,default=8); ap.add_argument('--device',default='0',help='Colab GPU: 0; CPU: cpu')
    ap.add_argument('--project',default='runs/fungicirno'); ap.add_argument('--name',default='sideview_candidate')
    ap.add_argument('--min-seg-map50',type=float,default=0.50); ap.add_argument('--allowed-drop',type=float,default=0.01)
    ap.add_argument('--species',default='M18K source domain; verify target species in evaluation'); ap.add_argument('--taxonomy',default='m18k_mushroom_instance')
    ap.add_argument('--out',default='artifacts/candidate_bundle.zip')
    args=ap.parse_args(); weights=Path(args.weights).resolve(); data=Path(args.data).resolve()
    if not weights.exists(): raise FileNotFoundError(weights)
    if not data.exists(): raise FileNotFoundError(data)
    base=YOLO(str(weights))
    base_target=metric_values(base,str(Path(args.holdout_data).resolve()),args.imgsz,args.device) if args.holdout_data else None
    base_source=metric_values(base,str(Path(args.source_holdout_data).resolve()),args.imgsz,args.device) if args.source_holdout_data else None
    train_result=base.train(data=str(data),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device=args.device,project=args.project,name=args.name,patience=12,save=True,plots=True,close_mosaic=10)
    save_dir=Path(getattr(getattr(base,'trainer',None),'save_dir',Path(args.project)/args.name))
    best=save_dir/'weights'/'best.pt'
    if not best.exists(): raise FileNotFoundError(f'Training did not produce {best}')
    trained=YOLO(str(best)); names={int(k):str(v) for k,v in trained.names.items()}
    metrics={'created_at':datetime.now(timezone.utc).isoformat(),'base_checkpoint':str(weights),'candidate_checkpoint':str(best),'training':{'epochs':args.epochs,'imgsz':args.imgsz,'batch':args.batch,'device':args.device},'promotion_ready':False,'promotion_reason':'Both target-domain and source-domain regression holdouts are required'}
    target_metrics=None; source_metrics=None
    if args.holdout_data:
        hold=Path(args.holdout_data).resolve()
        if not hold.exists(): raise FileNotFoundError(hold)
        target_metrics=metric_values(trained,str(hold),args.imgsz,args.device)
    if args.source_holdout_data:
        source_hold=Path(args.source_holdout_data).resolve()
        if not source_hold.exists(): raise FileNotFoundError(source_hold)
        source_metrics=metric_values(trained,str(source_hold),args.imgsz,args.device)
    if target_metrics is not None and source_metrics is not None:
        target_pass=target_metrics['seg_map50']>=args.min_seg_map50
        source_pass=(source_metrics['seg_map50']>=base_source['seg_map50']-args.allowed_drop and source_metrics['seg_map50_95']>=base_source['seg_map50_95']-args.allowed_drop)
        passed=target_pass and source_pass
        metrics.update({'target_holdout_data':str(Path(args.holdout_data).resolve()),'base_target_metrics':base_target,'candidate_target_metrics':target_metrics,
                        'source_holdout_data':str(Path(args.source_holdout_data).resolve()),'base_source_metrics':base_source,'candidate_source_metrics':source_metrics,
                        'promotion_ready':bool(passed),'promotion_reason':'Passed target-domain AP and source-domain retention gates' if passed else 'Candidate failed target minimum AP or source-domain regression tolerance',
                        'promotion_gates':{'min_target_seg_map50':args.min_seg_map50,'max_source_drop':args.allowed_drop}})
    config={'model_key':'m18k_yolo_seg','version':f"m18k-{datetime.now().strftime('%Y%m%d-%H%M%S')}",'architecture':'Ultralytics YOLOv8n-seg','runtime_type':'ultralytics_yolo_seg','task':'segment','source_species':args.species,'taxonomy':args.taxonomy,'names':{str(k):v for k,v in names.items()},'confidence_threshold':0.25,'nms_iou':0.70,'max_det':1000,'imgsz':args.imgsz}
    out=Path(args.out).resolve(); out.parent.mkdir(parents=True,exist_ok=True); staging=out.parent/(out.stem+'_staging'); shutil.rmtree(staging,ignore_errors=True); staging.mkdir()
    shutil.copy2(best,staging/'best.pt')
    (staging/'model_config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    (staging/'class_mapping.json').write_text(json.dumps({'label_to_name':{str(k):v for k,v in names.items()}},indent=2),encoding='utf8')
    (staging/'evaluation_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    (staging/'README.txt').write_text('Candidate model only. Do not activate until its evaluation_metrics.json shows promotion_ready=true and a reviewer approves it in FungiCirno. Training/validation and independent holdout must be separated by capture sequence or growing scene.\n',encoding='utf8')
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in staging.iterdir(): z.write(p,p.name)
    shutil.rmtree(staging,ignore_errors=True)
    print(json.dumps({'candidate_bundle':str(out),'metrics':metrics},indent=2))
if __name__=='__main__': main()
