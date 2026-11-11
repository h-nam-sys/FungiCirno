#!/usr/bin/env python
"""Convert a FungiCirno reviewed COCO export ZIP into an Ultralytics YOLO-seg dataset."""
from __future__ import annotations
import argparse,json,zipfile,random,shutil
from pathlib import Path
import cv2,yaml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input_zip'); ap.add_argument('--out',default='reviewed_yolo_dataset'); ap.add_argument('--val-fraction',type=float,default=.2); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
    out=Path(a.out).resolve(); shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    src=Path(a.input_zip)
    with zipfile.ZipFile(src) as z: z.extractall(out/'source')
    cands=list((out/'source').rglob('annotations.json'))
    if not cands: raise FileNotFoundError('annotations.json not found in review export')
    root=cands[0].parent; coco=json.loads(cands[0].read_text()); categories=sorted(coco['categories'],key=lambda c:int(c['id'])); remap={int(c['id']):i for i,c in enumerate(categories)}
    images={int(i['id']):i for i in coco['images']}; anns={k:[] for k in images}
    for an in coco['annotations']: anns.setdefault(int(an['image_id']),[]).append(an)
    ids=sorted(images); random.Random(a.seed).shuffle(ids); nval=max(1,round(len(ids)*a.val_fraction)) if len(ids)>1 else 0; val=set(ids[:nval])
    for split in ('train','val'):
        (out/f'images/{split}').mkdir(parents=True); (out/f'labels/{split}').mkdir(parents=True)
    for iid,rec in images.items():
        split='val' if iid in val else 'train'; srcimg=root/rec['file_name']
        if not srcimg.exists():
            matches=list(root.rglob(Path(rec['file_name']).name))
            if len(matches)!=1: raise FileNotFoundError(f'Image resolve ambiguous/missing: {rec["file_name"]}')
            srcimg=matches[0]
        dst=out/f'images/{split}/{iid}{srcimg.suffix.lower()}'; shutil.copy2(srcimg,dst)
        lines=[]; w=int(rec['width']); h=int(rec['height'])
        for an in anns.get(iid,[]):
            seg=an.get('segmentation',[])
            if isinstance(seg,dict):
                from pycocotools import mask as mu
                m=mu.decode(seg); contours,_=cv2.findContours(m.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); polys=[c.reshape(-1,2).flatten().tolist() for c in contours if len(c)>=3]
            else: polys=seg
            if polys:
                poly=max(polys,key=lambda q:len(q))
                pts=list(map(float,poly))
                if len(pts)<6: continue
                norm=[]
                for j in range(0,len(pts),2): norm.extend([min(1,max(0,pts[j]/w)),min(1,max(0,pts[j+1]/h))])
                lines.append(str(remap[int(an['category_id'])])+' '+' '.join(f'{x:.6f}' for x in norm))
        (out/f'labels/{split}/{iid}.txt').write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf8')
    names={i:c['name'] for i,c in enumerate(categories)}
    data={'path':str(out),'train':'images/train','val':'images/val','names':names}
    (out/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf8')
    (out/'dataset_info.json').write_text(json.dumps({'images':len(images),'annotations':len(coco['annotations']),'classes':names,'split':'deterministic image split; use only as a temporary validation split, not final sequence-isolated test'},indent=2))
    print(f'Prepared {len(images)} images and {len(coco["annotations"])} annotations at {out}')
if __name__=='__main__': main()
