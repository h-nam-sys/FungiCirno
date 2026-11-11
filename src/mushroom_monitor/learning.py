from __future__ import annotations
import json, zipfile, shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from .db import Database, utc_now

def export_reviewed_yolo_dataset(db:Database,destination:Path,model_key:str,reviewer_note:str='') -> tuple[Path,int]:
    """Export only human-accepted/corrected predictions to Ultralytics segmentation format.

    Review acceptance confirms the predicted mask; class corrections change the category.
    This exporter does not claim that an accepted prediction is independently ground-truth verified.
    """
    rows=db.query('''SELECT p.*,c.file_path,c.original_name,c.width,c.height,c.captured_at,c.series_key,c.unit_id,c.camera_id,
                    r.status AS human_status FROM predictions p JOIN captures c ON c.id=p.capture_id
                    JOIN model_versions m ON m.id=p.model_version_id JOIN review_items r ON r.prediction_id=p.id
                    WHERE m.model_key=? AND r.status IN ('accepted','corrected') AND p.run_id=(SELECT MAX(r2.id) FROM inference_runs r2 WHERE r2.capture_id=p.capture_id AND r2.model_version_id=p.model_version_id) ORDER BY p.capture_id,p.id''',(model_key,))
    if not rows: raise ValueError('No accepted/corrected predictions for this model family')
    class_ids=sorted({int(x['class_id']) for x in rows}); map_id={old:new for new,old in enumerate(class_ids)}
    class_names={map_id[int(r['class_id'])]:str(r['class_name']) for r in rows}
    # Use capture-level deterministic split. For multiple series, split whole series to reduce temporal leakage.
    series=sorted({f"{r['unit_id']}|{r['camera_id']}|{r['series_key']}" for r in rows})
    if len(series)>=3:
        train_series=set(series[:-max(1,round(len(series)*.2))]); split=lambda r:'train' if f"{r['unit_id']}|{r['camera_id']}|{r['series_key']}" in train_series else 'val'
        split_note='Grouped by growing-unit/camera/series. Keep an independent farm/sequence test set for promotion.'
    else:
        capture_ids=sorted({int(r['capture_id']) for r in rows})
        if len(capture_ids)<2: raise ValueError('At least two reviewed captures are required to create train/validation splits')
        nval=max(1,round(len(capture_ids)*.2)); val_ids=set(capture_ids[-nval:]); split=lambda r:'val' if int(r['capture_id']) in val_ids else 'train'
        split_note='Fallback capture-level split because fewer than three distinct series were available; adjacent-frame leakage may remain.'
    destination.parent.mkdir(parents=True,exist_ok=True); temp=destination.with_suffix('.tmp')
    images={}; labels=defaultdict(list); copied=set()
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for row in rows:
                capture_id=int(row['capture_id']); part=split(row)
                if capture_id not in images:
                    ext=Path(row['original_name']).suffix.lower() or '.jpg'; rel=f'images/{part}/capture_{capture_id}{ext}'
                    z.write(row['file_path'],rel); images[capture_id]=rel
                mask=cv2.imread(str(row['mask_path']),cv2.IMREAD_GRAYSCALE)
                if mask is None: continue
                h,w=mask.shape
                if (w,h)!=(int(row['width']),int(row['height'])): mask=cv2.resize(mask,(int(row['width']),int(row['height'])),interpolation=cv2.INTER_NEAREST)
                contours,_=cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                # Ultralytics TXT represents one polygon per instance. If a binary mask
                # contains disconnected fragments, retain the largest external contour rather
                # than accidentally turning fragments into additional mushrooms.
                if not contours: continue
                contour=max(contours,key=cv2.contourArea)
                if len(contour)<3: continue
                approx=cv2.approxPolyDP(contour,1.0,True).reshape(-1,2)
                if len(approx)<3: continue
                vals=[]
                for x,y in approx:
                    vals.extend([min(1,max(0,float(x)/max(1,row['width']))),min(1,max(0,float(y)/max(1,row['height'])))])
                if len(vals)<6: continue
                # Ultralytics segmentation TXT: class followed by normalized polygon points.
                label_name=f"labels/{part}/capture_{capture_id}.txt"
                labels[label_name].append(str(map_id[int(row['class_id'])])+' '+' '.join(f'{v:.6f}' for v in vals))
            for name,lines in labels.items(): z.writestr(name,'\n'.join(lines)+'\n')
            yaml='path: .\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {json.dumps(name)}\n' for i,name in class_names.items())
            z.writestr('data.yaml',yaml)
            z.writestr('dataset_manifest.json',json.dumps({'model_key':model_key,'created_at':utc_now(),'images':len(images),'annotations':sum(map(len,labels.values())), 'class_mapping':class_names,'split_note':split_note,'human_review_note':reviewer_note},indent=2))
            z.writestr('README.txt','This dataset contains accepted/corrected model masks. Human acceptance is not independent ground truth unless reviewed against the original image. Inspect labels before training. For final performance reporting, use a separate untouched holdout dataset.\n'+split_note)
        temp.replace(destination)
    except Exception:
        temp.unlink(missing_ok=True); raise
    count=sum(map(len,labels.values()))
    db.execute('INSERT INTO learning_exports(created_at,path,sample_count,model_key,notes) VALUES(?,?,?,?,?)',(utc_now(),str(destination),count,model_key,'YOLO segmentation export: '+split_note))
    return destination,count
