#!/usr/bin/env python
"""Run one YOLO-seg checkpoint on a single image and export predictions/timing.

Run once with --device cpu and once with --device amd on the same image to compare.
This is an inference smoke/benchmark utility, not an accuracy evaluation.
"""
from __future__ import annotations
import argparse,csv,json,time,os
from pathlib import Path
import numpy as np
import cv2
import psutil
from ultralytics import YOLO

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',required=True); ap.add_argument('--image',required=True); ap.add_argument('--device',choices=['cpu','amd'],default='cpu'); ap.add_argument('--imgsz',type=int,default=768); ap.add_argument('--conf',type=float,default=.25); ap.add_argument('--iou',type=float,default=.70); ap.add_argument('--out',default='results/benchmark'); a=ap.parse_args()
    import torch
    device='cpu'
    if a.device=='amd':
        if os.name!='nt': raise RuntimeError('AMD DirectML benchmark requires native Windows')
        import torch_directml
        device=torch_directml.device()
    model=YOLO(a.weights,task='segment')
    if a.device=='amd': model.model.to(device)
    image=cv2.imread(a.image,cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(a.image)
    proc=psutil.Process(os.getpid()); rss0=proc.memory_info().rss/(1024*1024)
    started=time.perf_counter()
    result=model.predict(source=image,imgsz=a.imgsz,conf=a.conf,iou=a.iou,retina_masks=True,device=device,verbose=False)[0]
    elapsed=time.perf_counter()-started; rss1=proc.memory_info().rss/(1024*1024)
    out=Path(a.out)/a.device; out.mkdir(parents=True,exist_ok=True)
    overlay=result.plot(); cv2.imwrite(str(out/'overlay.jpg'),overlay)
    rows=[]
    if result.boxes is not None:
        boxes=result.boxes.xyxy.detach().cpu().numpy(); classes=result.boxes.cls.detach().cpu().numpy().astype(int); scores=result.boxes.conf.detach().cpu().numpy()
        masks=result.masks.data.detach().cpu().numpy() if result.masks is not None else []
        names=model.names
        for i,(box,cid,score) in enumerate(zip(boxes,classes,scores)):
            row={'detection_index':i,'class_id':int(cid),'class_name':str(names[int(cid)]),'confidence':float(score),'x1':float(box[0]),'y1':float(box[1]),'x2':float(box[2]),'y2':float(box[3])}
            if len(masks)>i:
                m=(masks[i]>.5).astype(np.uint8); row['area_px2']=int(m.sum()); cv2.imwrite(str(out/f'mask_{i:04d}.png'),m*255)
            rows.append(row)
    with (out/'predictions.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r})); writer.writeheader(); writer.writerows(rows)
    summary={'weights':str(Path(a.weights).resolve()),'image':str(Path(a.image).resolve()),'device':a.device,'device_label':str(device),'imgsz':a.imgsz,'confidence':a.conf,'nms_iou':a.iou,'detections':len(rows),'elapsed_seconds':elapsed,'ms_per_image':elapsed*1000,'process_rss_before_mb':rss0,'process_rss_after_mb':rss1,'note':'Inference smoke/benchmark only; not a labelled accuracy evaluation.'}
    (out/'benchmark.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2)); print('Outputs:',out)
if __name__=='__main__': main()
