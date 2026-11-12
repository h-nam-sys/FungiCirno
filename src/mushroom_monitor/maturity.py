from __future__ import annotations
import json, math
from collections import defaultdict
import numpy as np
from .db import Database

def _connected_groups(points:np.ndarray,eps:float)->np.ndarray:
    """Self-contained connected-components grouping equivalent to DBSCAN min_samples=1."""
    n=len(points); parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for i in range(n):
        for j in range(i+1,n):
            if float(np.linalg.norm(points[i]-points[j]))<=eps:
                a,b=find(i),find(j)
                if a!=b: parent[b]=a
    roots={}; labels=np.empty(n,dtype=np.int32)
    for i in range(n):
        r=find(i); roots.setdefault(r,len(roots)); labels[i]=roots[r]
    return labels

def rebuild_flush_groups(db:Database,capture_id:int,multiplier:float=5.0,model_version_id:int|None=None)->int:
    capture=db.one('SELECT width,height FROM captures WHERE id=?',(capture_id,))
    if not capture: raise ValueError('Capture not found')
    
    # Chỉ lấy các prediction có diện tích lớn hơn ngưỡng tối thiểu (ví dụ > 50 pixels) để loại bỏ nhiễu mảnh vỡ
    min_area_threshold = 50.0
    
    if model_version_id is None:
        rows=db.query('''SELECT * FROM predictions p WHERE capture_id=? AND final_for_analysis=1 AND area_px2 >= ? AND run_id=(SELECT MAX(r.id) FROM inference_runs r WHERE r.capture_id=p.capture_id AND r.model_version_id=p.model_version_id) ORDER BY confidence DESC''',(capture_id, min_area_threshold))
        db.execute('DELETE FROM flush_groups WHERE capture_id=? AND model_version_id IS NULL',(capture_id,))
    else:
        rows=db.query('''SELECT * FROM predictions p WHERE capture_id=? AND model_version_id=? AND final_for_analysis=1 AND area_px2 >= ? AND run_id=(SELECT MAX(r.id) FROM inference_runs r WHERE r.capture_id=p.capture_id AND r.model_version_id=p.model_version_id) ORDER BY confidence DESC''',(capture_id, model_version_id, min_area_threshold))
        db.execute('DELETE FROM flush_groups WHERE capture_id=? AND model_version_id=?',(capture_id,model_version_id))
        
    if not rows: return 0
    centers=[]; radii=[]
    for r in rows:
        centers.append([float(r['centroid_x'] if r['centroid_x'] is not None else (r['x1']+r['x2'])/2),float(r['centroid_y'] if r['centroid_y'] is not None else (r['y1']+r['y2'])/2)])
        radii.append(math.sqrt(max(float(r['area_px2']),1.0)/math.pi))
        
    eps=max(1.0,float(multiplier)*float(np.median(radii) if radii else 1.0))
    labels=_connected_groups(np.asarray(centers,dtype=np.float32),eps)
    
    for label in sorted(set(labels.tolist())):
        members=[r for r,l in zip(rows,labels) if int(l)==int(label)]; areas=defaultdict(float)
        for r in members: areas[str(r['class_name'])]+=float(r['area_px2'])
        total=sum(areas.values()); dominant=max(areas,key=areas.get)
        ready=next((k for k in areas if k.strip().lower()=='ready'),None); ready_frac=areas[ready]/total if ready and total else None
        
        db.execute('''INSERT INTO flush_groups(capture_id,model_version_id,group_key,fruitbody_count,total_area_px2,dominant_class,ready_area_fraction,class_area_json) VALUES(?,?,?,?,?,?,?,?)''',(capture_id,model_version_id,f'{capture_id}-flush-{label+1}',len(members),total,dominant,ready_frac,json.dumps(areas)))
    return len(set(labels.tolist()))
