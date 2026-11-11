from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from .config import TRACK_MASK_SIDE, TRACK_MAX_GAP_FRAMES
from .db import Database

def _small_mask(path:str,side:int=TRACK_MASK_SIDE)->np.ndarray:
    with Image.open(path) as im:
        m=im.convert('L'); m.thumbnail((side,side),Image.Resampling.NEAREST); a=np.asarray(m)>0
    canvas=np.zeros((side,side),dtype=bool); h,w=a.shape; y=(side-h)//2; x=(side-w)//2; canvas[y:y+h,x:x+w]=a
    return canvas

def _iou(a:np.ndarray,b:np.ndarray)->float:
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum(); return float(inter/union) if union else 0.0

def _predict_center(state:dict[str,Any],frame:int)->tuple[float,float]:
    x,y=state['center']; prev=state.get('previous_center'); pf=state.get('previous_frame'); lf=state['last_frame']
    if prev is None or pf is None or lf<=pf: return x,y
    dt=max(1,lf-pf); horizon=min(max(1,frame-lf),3)
    return x+(x-prev[0])/dt*horizon,y+(y-prev[1])/dt*horizon

def _score(state:dict[str,Any],det:dict[str,Any],frame:int)->float:
    iou=_iou(state['mask'],det['mask']); pred=_predict_center(state,frame); dx=pred[0]-det['center'][0]; dy=pred[1]-det['center'][1]; dist=math.hypot(dx,dy)
    ratio=min(state['area'],det['area'])/max(state['area'],det['area'],1.0)
    center=math.exp(-dist/.10); score=.55*iou+.30*center+.15*ratio
    if dist>.25*math.sqrt(2) or ratio<.02 or score<.20: return -1.0
    return score

def rebuild_sequence_tracks(db:Database,unit_id:int,camera_id:int,series_key:str,model_version_id:int,max_missed_frames:int=TRACK_MAX_GAP_FRAMES)->dict[str,int]:
    """Rebuild persistent cluster IDs with motion-assisted Hungarian mask association.

    IDs are deterministic for a fixed ordered sequence and model version. A full rebuild
    replaces only that model-version's tracks; other model versions are never touched.
    """
    if max_missed_frames<0: raise ValueError('max_missed_frames must be >= 0')
    captures=db.query('''SELECT * FROM captures WHERE unit_id=? AND camera_id=? AND series_key=? ORDER BY COALESCE(captured_at,imported_at),id''',(unit_id,camera_id,series_key))
    with db.transaction() as conn:
        conn.execute('''DELETE FROM tracks WHERE unit_id=? AND camera_id=? AND series_key=? AND model_version_id=?''',(unit_id,camera_id,series_key,model_version_id))
    active:dict[str,dict[str,Any]]={}; next_id=1; observations=0
    camera=db.one('SELECT mm_per_px FROM cameras WHERE id=?',(camera_id,)); mm=float(camera['mm_per_px']) if camera and camera['mm_per_px'] else None
    for frame,capture in enumerate(captures):
        rows=db.query('''SELECT * FROM predictions p WHERE capture_id=? AND model_version_id=? AND final_for_analysis=1 AND run_id=(SELECT MAX(r.id) FROM inference_runs r WHERE r.capture_id=p.capture_id AND r.model_version_id=p.model_version_id) ORDER BY confidence DESC''',(capture['id'],model_version_id))
        detections=[]; w=max(1,int(capture['width'])); h=max(1,int(capture['height']))
        for row in rows:
            try: mask=_small_mask(row['mask_path'])
            except (OSError,ValueError): continue
            if not mask.any(): continue
            cx=float(row['centroid_x'] if row['centroid_x'] is not None else (row['x1']+row['x2'])/2)/w
            cy=float(row['centroid_y'] if row['centroid_y'] is not None else (row['y1']+row['y2'])/2)/h
            detections.append({'row':row,'mask':mask,'center':(cx,cy),'area':max(float(row['area_px2']),1.0)})
        keys=list(active); matched={}
        if keys and detections:
            scores=np.full((len(keys),len(detections)),-1.0,dtype=np.float64); costs=np.full_like(scores,1e6)
            for ti,key in enumerate(keys):
                for di,det in enumerate(detections):
                    score=_score(active[key],det,frame); scores[ti,di]=score
                    if score>=0: costs[ti,di]=1-score
            ri,ci=linear_sum_assignment(costs)
            for ti,di in zip(ri.tolist(),ci.tolist()):
                if scores[ti,di]>=0: matched[di]=(keys[ti],float(scores[ti,di]))
        used={v[0] for v in matched.values()}
        for key in list(active):
            if key not in used:
                active[key]['missed']+=1
                if active[key]['missed']>max_missed_frames: active.pop(key,None)
        for di,det in enumerate(detections):
            row=det['row']
            if di in matched:
                key,score=matched[di]; state=active[key]; first_area=state['first_area']; first_capture=state['first_capture']; count=state['observations']+1; track_id=state['track_id']; prev_center=state['center']; prev_frame=state['last_frame']
                growth=100*(det['area']/max(first_area,1)-1)
                db.execute('''UPDATE tracks SET class_name=?,last_capture_id=?,last_area_px2=?,observations=?,growth_pct=?,status='active' WHERE id=?''',(row['class_name'],capture['id'],det['area'],count,growth,track_id))
            else:
                key=f'{series_key}-M{model_version_id}-C{next_id:05d}'; next_id+=1; score=None; first_area=det['area']; first_capture=int(capture['id']); count=1; prev_center=None; prev_frame=None; growth=0.0
                track_id=db.execute('''INSERT INTO tracks(unit_id,camera_id,series_key,model_version_id,track_key,class_name,first_capture_id,last_capture_id,first_area_px2,last_area_px2,observations,growth_pct,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(unit_id,camera_id,series_key,model_version_id,key,row['class_name'],first_capture,capture['id'],first_area,det['area'],count,growth,'active'))
            area_mm=det['area']*mm*mm if mm is not None else None
            db.execute('''INSERT INTO track_observations(track_id,prediction_id,capture_id,frame_index,captured_at,area_px2,area_mm2,growth_vs_first_pct,association_score) VALUES(?,?,?,?,?,?,?,?,?)''',(track_id,row['id'],capture['id'],frame,capture['captured_at'],det['area'],area_mm,growth,score))
            active[key]={'track_id':track_id,'mask':det['mask'],'center':det['center'],'previous_center':prev_center,'previous_frame':prev_frame,'last_frame':frame,'area':det['area'],'missed':0,'first_area':first_area,'first_capture':first_capture,'observations':count}
            observations+=1
    active_ids={int(v['track_id']) for v in active.values()}
    with db.transaction() as conn:
        conn.execute('''UPDATE tracks SET status='closed' WHERE unit_id=? AND camera_id=? AND series_key=? AND model_version_id=?''',(unit_id,camera_id,series_key,model_version_id))
        if active_ids:
            q=','.join('?' for _ in active_ids)
            conn.execute(f"UPDATE tracks SET status='active' WHERE id IN ({q})",tuple(active_ids))
    return {'captures':len(captures),'observations':observations,'tracks':next_id-1,'active_tracks':len(active)}
