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
        if sorted(classes) != list(range(len(classes))):
            raise ValueError('YOLO class IDs must be contiguous and zero-based')
        model_key=str(config.get('model_key','m18k_yolo_seg'))
        architecture=str(config.get('architecture','Ultralytics YOLOv8n-seg'))
        version=str(config.get('version') or f"yolo-{sha256_file(pt)[:12]}")
        species=str(config.get('source_species','unknown'))
        taxonomy=str(config.get('taxonomy','custom'))
        role=str(config.get('prediction_role','generic_instance'))
        expected = {
            'charisis_cluster_tracking': ('cluster_stage', ['immature_cluster', 'mature_cluster', 'overstayed_cluster'], True),
            'mushr_maturity': ('fruitbody_maturity', ['not-ready', 'ready', 'overdue'], False),
            'm18k_source_wb': ('generic_instance', ['wb'], False),
        }
        if model_key in expected:
            expected_role, expected_names, expected_tracking = expected[model_key]
            actual_names=[classes[i].strip().casefold() for i in range(len(classes))]
            if role != expected_role or actual_names != expected_names:
                raise ValueError(
                    f'Model {model_key!r} requires role={expected_role!r} and class order {expected_names}; '
                    f'got role={role!r}, classes={actual_names}'
                )
            if bool(config.get('track_objects', expected_tracking)) != expected_tracking:
                raise ValueError(f'Model {model_key!r} has invalid track_objects configuration')
        configured_names=config.get('names')
        if configured_names is not None:
            configured={int(k):str(v) for k,v in configured_names.items()} if isinstance(configured_names,dict) else {i:str(v) for i,v in enumerate(configured_names)}
            if configured != classes:
                raise ValueError('model_config.json names do not match class_mapping.json')
        config={**config,'runtime_type':'ultralytics_yolo_seg','task':'segment'}
        return ModelPackage(model_key,version,architecture,species,taxonomy,classes,config,root,pt,sha256_file(pt),metrics)
    # TorchVision Mask R-CNN bundles. Charisis is the primary oyster model;
    # MushR remains an optional individual-fruitbody maturity extension.
    mappingp = root / 'class_mapping.json'
    cp = root / 'model_checkpoint.pth'
    if not mappingp.exists() or not cp.exists():
        raise FileNotFoundError('Bundle needs best.pt or model_checkpoint.pth plus model_config.json/class_mapping.json')
    raw = json.loads(mappingp.read_text(encoding='utf8'))
    raw = raw.get('label_to_name', raw.get('classes', {}))
    classes = {int(k): str(v) for k, v in raw.items()}
    architecture = str(config.get('architecture', ''))
    low = architecture.lower()

    if 'convnext' in low:
        # Older project bundles used this legacy key. New imports use the
        # stable production family key while retaining the old directory/path.
        key = 'charisis_cluster_tracking'
        expected_classes = {
            1: 'immature_cluster',
            2: 'mature_cluster',
            3: 'overstayed_cluster',
        }
        if classes != expected_classes:
            raise ValueError(
                'Charisis Mask R-CNN class_mapping.json must preserve '
                '1=immature_cluster, 2=mature_cluster, 3=overstayed_cluster; '
                'class ID 0 is reserved for background'
            )
        taxonomy = str(config.get('taxonomy') or 'charisis_oyster_cluster_stage_v1')
        species = str(config.get('source_species') or 'Pleurotus ostreatus')
        capabilities = {'prediction_role': 'cluster_stage', 'track_objects': True, 'spatial_grouping': False}
    elif 'resnet-50' in low or 'resnet50' in low:
        key = 'mushr_maturity'
        expected_classes = {
            1: 'not-ready',
            2: 'ready',
            3: 'overdue',
        }
        if classes != expected_classes:
            raise ValueError(
                'MushR Mask R-CNN class_mapping.json must preserve '
                '1=not-ready, 2=ready, 3=overdue; class ID 0 is reserved for background'
            )
        taxonomy = str(config.get('taxonomy') or 'mushr_oyster_fruitbody_maturity_v1')
        species = str(config.get('source_species') or 'Pleurotus ostreatus')
        capabilities = {'prediction_role': 'fruitbody_maturity', 'track_objects': False, 'spatial_grouping': True}
    else:
        key = str(config.get('model_key', 'custom_torchvision'))
        taxonomy = str(config.get('taxonomy', 'custom'))
        species = str(config.get('source_species', 'unknown'))
        capabilities = {'track_objects': False, 'spatial_grouping': False}

    # Keep the source checkpoint's class IDs and names unchanged. TorchVision
    # detection models reserve class ID 0 for background; Charisis labels are 1..3.
    config = {
        **config,
        'model_key': key,
        'runtime_type': 'torchvision_state_dict',
        'source_species': species,
        'taxonomy': taxonomy,
        **capabilities,
    }
    version = str(config.get('version') or f'checkpoint-{sha256_file(cp)[:12]}')
    return ModelPackage(key, version, architecture, species, taxonomy, classes, config,
                        root, cp, sha256_file(cp), metrics)


def model_capabilities(model_key: str, architecture: str, config: dict[str, Any] | None = None) -> dict[str, bool]:
    """Return downstream capabilities without coupling them to the model backend."""
    cfg = config or {}
    key = str(model_key).casefold()
    arch = str(architecture).casefold()
    is_charisis = key in {'charisis_cluster_tracking', 'legacy_charisis_maskrcnn'} or 'convnext' in arch
    is_mushr = key in {'mushr_maturity', 'legacy_mushr_maskrcnn'} or ('resnet' in arch and 'mask' in arch)
    return {
        'track_objects': bool(cfg.get('track_objects', is_charisis)),
        'spatial_grouping': bool(cfg.get('spatial_grouping', is_mushr)),
    }

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
