from __future__ import annotations
import platform, time, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from ..image_io import load_rgb, resize_for_inference
from ..model_registry import package_for_model_row
from .models import build_model

@dataclass
class Prediction:
    detection_index:int; class_id:int; class_name:str; confidence:float
    bbox:tuple[float,float,float,float]; mask:np.ndarray
@dataclass
class DeviceInfo:
    label:str; device:Any; backend:str; note:str

def select_device(preference:str='auto')->DeviceInfo:
    pref=preference.lower()
    if pref in {'cpu','auto'}: return DeviceInfo('CPU','cpu','cpu','CPU execution selected')
    if pref in {'amd','directml'}:
        if platform.system()!='Windows': raise RuntimeError('DirectML requires native Windows')
        try:
            import torch_directml
            return DeviceInfo('AMD DirectML',torch_directml.device(),'directml','AMD DirectML explicitly selected; no silent fallback')
        except Exception as exc: raise RuntimeError(f'AMD DirectML unavailable: {exc}. Select CPU explicitly.') from exc
    raise ValueError('Backend must be amd, cpu, or auto')

class InferenceEngine:
    def __init__(self,model_row:dict[str,Any],inference_side:int=768,device_preference:str='auto',score_threshold:float|None=None,mask_threshold:float=.5):
        self.model_row=model_row; self.package=package_for_model_row(model_row); self.inference_side=int(inference_side); self.device_info=select_device(device_preference)
        self.score_threshold=float(score_threshold if score_threshold is not None else self.package.config.get('confidence_threshold',.25)); self.mask_threshold=float(mask_threshold); self.model=None; self.load_model()
    def load_model(self):
        rt=self.package.config.get('runtime_type','torchvision_state_dict')
        try:
            if rt=='ultralytics_yolo_seg':
                from ultralytics import YOLO
                self.model=YOLO(str(self.package.checkpoint_path),task='segment')
                if self.device_info.backend=='directml': self.model.model.to(self.device_info.device)
                else: self.model.model.to('cpu')
                self.model.model.eval()
            else:
                self.model=build_model(self.package,self.inference_side).to(self.device_info.device).eval()
        except Exception as exc:
            raise RuntimeError(f"Could not load {self.package.architecture} on {self.device_info.label}: {type(exc).__name__}: {exc}") from exc
    def _predict_yolo(self,image:Image.Image):
        arr=np.asarray(image.convert('RGB'))
        results=self.model.predict(source=arr,imgsz=self.inference_side,conf=self.score_threshold,iou=float(self.package.config.get('nms_iou',.7)),max_det=int(self.package.config.get('max_det',1000)),retina_masks=True,device=self.device_info.device if self.device_info.backend=='directml' else 'cpu',verbose=False)
        r=results[0]; preds=[]
        if r.boxes is None or r.masks is None: return preds
        boxes=r.boxes.xyxy.detach().cpu().numpy(); ids=r.boxes.cls.detach().cpu().numpy().astype(int); scores=r.boxes.conf.detach().cpu().numpy(); masks=r.masks.data.detach().cpu().numpy()
        names=self.package.classes
        for i,(box,cid,score,mask) in enumerate(zip(boxes,ids,scores,masks)):
            if cid not in names: continue
            m=(mask>.5).astype(np.uint8)
            if not m.any(): continue
            preds.append(Prediction(i,int(cid),names[int(cid)],float(score),tuple(float(x) for x in box),m))
        return preds
    def _predict_torchvision(self,image:Image.Image):
        import torch
        from torchvision.transforms.functional import pil_to_tensor
        resized,_,_=resize_for_inference(image,self.inference_side)
        tensor=pil_to_tensor(resized).to(dtype=torch.float32,device=self.device_info.device).div_(255.)
        with torch.no_grad(): out=self.model([tensor])[0]
        boxes=out['boxes'].detach().cpu().numpy(); labels=out['labels'].detach().cpu().numpy(); scores=out['scores'].detach().cpu().numpy(); masks=out['masks'].detach().cpu().numpy(); preds=[]
        for i,(b,c,s,m) in enumerate(zip(boxes,labels,scores,masks)):
            if int(c) not in self.package.classes or float(s)<self.score_threshold: continue
            binary=(m[0]>=self.mask_threshold).astype(np.uint8)
            if binary.any(): preds.append(Prediction(i,int(c),self.package.classes[int(c)],float(s),tuple(float(x) for x in b),binary))
        return preds
    def predict_path(self,path:Path): return self.predict_image(load_rgb(path))
    def predict_image(self,image:Image.Image):
        if self.model is None: raise RuntimeError('Model is not loaded')
        try:
            import psutil
            rss_before=psutil.Process(os.getpid()).memory_info().rss/(1024*1024)
        except Exception:
            rss_before=None
        started=time.perf_counter(); resized,_,_=resize_for_inference(image,self.inference_side)
        rt=self.package.config.get('runtime_type','torchvision_state_dict')
        preds=self._predict_yolo(resized) if rt=='ultralytics_yolo_seg' else self._predict_torchvision(resized)
        elapsed=time.perf_counter()-started
        try:
            import psutil
            rss_after=psutil.Process(os.getpid()).memory_info().rss/(1024*1024)
        except Exception:
            rss_after=None
        info={'process_rss_before_mb':round(rss_before,1) if rss_before is not None else None,'process_rss_after_mb':round(rss_after,1) if rss_after is not None else None,'device':self.device_info.label,'backend':self.device_info.backend,'device_note':self.device_info.note,'runtime_type':rt,'inference_seconds':round(elapsed,4),'original_width':image.width,'original_height':image.height,'inference_width':resized.width,'inference_height':resized.height,'scale_x':image.width/resized.width,'scale_y':image.height/resized.height}
        return preds,image,info
