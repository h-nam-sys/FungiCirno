from __future__ import annotations
import hashlib, json, shutil, tempfile, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .db import Database, utc_now

@dataclass(frozen=True)
class ModelPackage:
    model_key: str
    version: str
    architecture: str
    source_species: str
    taxonomy: str
    classes: dict[int, str]
    config: dict[str, Any]
    bundle_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    metrics: dict[str, Any]

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def _safe_extract(zip_path: Path, destination: Path) -> None:
    dest=destination.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            target=(destination/info.filename).resolve()
            if dest not in target.parents and target != dest: raise ValueError(f'Unsafe archive path: {info.filename}')
        z.extractall(destination)

def _read_root(root: Path) -> ModelPackage:
    cfgp=root/'model_config.json'
    if not cfgp.exists(): raise FileNotFoundError('model_config.json is required')
    config=json.loads(cfgp.read_text(encoding='utf8'))
    metrics={}
    metricp=root/'evaluation_metrics.json'
    if metricp.exists(): metrics=json.loads(metricp.read_text(encoding='utf8'))
    # M18K / Ultralytics YOLO segmentation bundle.
    pt_candidates=[root/'best.pt',root/'model_checkpoint.pt',root/'model.pt']
    pt=next((p for p in pt_candidates if p.exists()),None)
    if pt:
        mappingp=root/'class_mapping.json'
        if mappingp.exists(): raw=json.loads(mappingp.read_text(encoding='utf8')); raw=raw.get('label_to_name',raw.get('classes',raw))
        else: raw=config.get('names',config.get('classes',{0:'mushroom'}))
        classes={int(k):str(v) for k,v in raw.items()} if isinstance(raw,dict) else {i:str(v) for i,v in enumerate(raw)}
        if not classes: raise ValueError('YOLO bundle contains no class mapping')
        model_key=str(config.get('model_key','m18k_yolo_seg'))
        architecture=str(config.get('architecture','Ultralytics YOLOv8n-seg'))
        version=str(config.get('version') or f"m18k-{sha256_file(pt)[:12]}")
        species=str(config.get('source_species','Agaricus bisporus (M18K White Button)'))
        taxonomy=str(config.get('taxonomy','m18k_mushroom_instance'))
        config={**config,'runtime_type':'ultralytics_yolo_seg','task':'segment'}
        return ModelPackage(model_key,version,architecture,species,taxonomy,classes,config,root,pt,sha256_file(pt),metrics)
    # Existing TorchVision Mask R-CNN member format.
    mappingp=root/'class_mapping.json'; cp=root/'model_checkpoint.pth'
    if not mappingp.exists() or not cp.exists(): raise FileNotFoundError('Bundle needs best.pt or model_checkpoint.pth plus model_config.json/class_mapping.json')
    raw=json.loads(mappingp.read_text(encoding='utf8')); raw=raw.get('label_to_name',raw.get('classes',{}))
    classes={int(k):str(v) for k,v in raw.items()}; architecture=str(config.get('architecture',''))
    low=architecture.lower()
    if 'convnext' in low: key,tax,species='charisis_cluster_tracking','charisis_oyster_cluster_stage','Pleurotus ostreatus'
    elif 'resnet' in low: key,tax,species='mushr_maturity','mushr_oyster_maturity','Pleurotus ostreatus'
    else: key,tax,species=str(config.get('model_key','custom_torchvision')),str(config.get('taxonomy','custom')),str(config.get('source_species','unknown'))
    config={**config,'runtime_type':config.get('runtime_type','torchvision_state_dict')}
    version=str(config.get('version') or f'checkpoint-{sha256_file(cp)[:12]}')
    return ModelPackage(key,version,architecture,species,tax,classes,config,root,cp,sha256_file(cp),metrics)

def read_model_package(bundle: Path) -> ModelPackage:
    bundle=bundle.resolve()
    if bundle.is_dir(): return _read_root(bundle)
    if bundle.suffix.lower() not in {'.zip'}: raise ValueError('Upload a model ZIP bundle')
    temp=Path(tempfile.mkdtemp(prefix='fungicirno-model-read-'))
    try:
        _safe_extract(bundle,temp)
        configs=list(temp.rglob('model_config.json'))
        if len(configs)!=1: raise ValueError(f'Expected one model_config.json, found {len(configs)}')
        return _read_root(configs[0].parent)
    finally: shutil.rmtree(temp,ignore_errors=True)

def install_model_bundle(bundle: Path, registry_root: Path, db: Database) -> int:
    package=read_model_package(bundle); target=registry_root/package.model_key/package.version
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise FileExistsError(f'Model version already installed: {target}')
    extract=None; stage=target.with_name(target.name+'.installing')
    try:
        if bundle.is_file():
            extract=Path(tempfile.mkdtemp(prefix='.extract-',dir=target.parent)); _safe_extract(bundle,extract)
            cp=list(extract.rglob('model_config.json'))
            if len(cp)!=1: raise ValueError('Bundle must have exactly one model_config.json')
            shutil.copytree(cp[0].parent,stage)
        else: shutil.copytree(bundle,stage)
        cfg=json.loads((stage/'model_config.json').read_text(encoding='utf8'))
        artifact=next((stage/n for n in ('best.pt','model_checkpoint.pt','model.pt','model_checkpoint.pth') if (stage/n).exists()),None)
        if artifact is None: raise FileNotFoundError('No supported checkpoint file found')
        if not (stage/'class_mapping.json').exists():
            (stage/'class_mapping.json').write_text(json.dumps({'label_to_name':{str(k):v for k,v in package.classes.items()}},indent=2),encoding='utf8')
        stage.rename(target); artifact=target/artifact.name
        mid=db.execute('''INSERT INTO model_versions(model_key,version,architecture,source_species,taxonomy,classes_json,config_json,bundle_path,checkpoint_path,checkpoint_sha256,status,installed_at,metrics_json)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(package.model_key,package.version,package.architecture,package.source_species,package.taxonomy,json.dumps({str(k):v for k,v in package.classes.items()}),json.dumps({**cfg,'runtime_type':package.config.get('runtime_type')}),str(target),str(artifact),sha256_file(artifact),'candidate',utc_now(),json.dumps(package.metrics)))
        return mid
    except Exception:
        shutil.rmtree(stage,ignore_errors=True); shutil.rmtree(target,ignore_errors=True); raise
    finally:
        if extract: shutil.rmtree(extract,ignore_errors=True)

def package_for_model_row(row: dict[str,Any]) -> ModelPackage:
    root=Path(row['bundle_path']); classes={int(k):str(v) for k,v in json.loads(row['classes_json']).items()}; config=json.loads(row['config_json']); cp=Path(row['checkpoint_path'])
    return ModelPackage(row['model_key'],row['version'],row['architecture'],row['source_species'],row['taxonomy'],classes,config,root,cp,row['checkpoint_sha256'],json.loads(row.get('metrics_json') or '{}'))
