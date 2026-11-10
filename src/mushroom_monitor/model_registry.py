from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe path in model bundle: {info.filename}")
        archive.extractall(destination)


def read_model_package(bundle: Path) -> ModelPackage:
    bundle = bundle.resolve()
    if bundle.is_file() and bundle.suffix.lower() == ".zip":
        temp = Path(tempfile.mkdtemp(prefix="mushroom-model-check-"))
        _safe_extract(bundle, temp)
        dirs = [p for p in temp.rglob("model_config.json")]
        if len(dirs) != 1:
            raise ValueError("Model ZIP must contain exactly one model_config.json")
        root = dirs[0].parent
        config_path = root / "model_config.json"
        mapping_path = root / "class_mapping.json"
        checkpoint_path = root / "model_checkpoint.pth"
    else:
        root = bundle
        config_path = root / "model_config.json"
        mapping_path = root / "class_mapping.json"
        checkpoint_path = root / "model_checkpoint.pth"
    for required in (config_path, mapping_path, checkpoint_path):
        if not required.exists():
            raise FileNotFoundError(f"Model bundle is missing {required.name}: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mapping_raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    label_map = mapping_raw.get("label_to_name") or mapping_raw.get("classes")
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("class_mapping.json must define label_to_name")
    classes = {int(k): str(v) for k, v in label_map.items()}
    if sorted(classes) != list(range(1, len(classes) + 1)):
        raise ValueError("Model classes must use contiguous labels starting at 1; background is label 0")
    architecture = str(config.get("architecture", ""))
    lower = architecture.lower()
    if "convnext" in lower:
        model_key = "charisis_cluster_tracking"
        species = "Pleurotus ostreatus"
        taxonomy = "charisis_oyster_cluster_stage"
    elif "resnet-50" in lower or "resnet50" in lower:
        model_key = "mushr_maturity"
        species = "Pleurotus ostreatus"
        taxonomy = "mushr_oyster_maturity"
    else:
        raise ValueError(f"Unsupported model architecture in bundle: {architecture!r}")
    extra = {}
    # Bundle files are small except for the checkpoint. Metadata can optionally be added by the notebook.
    bundle_meta = root / "bundle_manifest.json"
    if bundle_meta.exists():
        extra = json.loads(bundle_meta.read_text(encoding="utf-8"))
    version = str(extra.get("version") or config.get("version") or f"colab-{int(checkpoint_path.stat().st_mtime)}")
    metrics = extra.get("metrics", {}) or {}
    return ModelPackage(model_key, version, architecture, species, taxonomy, classes, config,
                        root, checkpoint_path, sha256_file(checkpoint_path), metrics)


def install_model_bundle(bundle: Path, registry_root: Path, db: Database) -> int:
    package = read_model_package(bundle)
    target = registry_root / package.model_key / package.version
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Model version already installed: {target}")

    if Path(bundle).is_file():
        # Extract and stage beside the final target so the final rename is atomic.
        extract_dir = Path(tempfile.mkdtemp(prefix=f".{package.version}.extract-", dir=target.parent))
        stage_dir = target.with_name(target.name + ".installing")
        try:
            _safe_extract(Path(bundle), extract_dir)
            config_files = list(extract_dir.rglob("model_config.json"))
            if len(config_files) != 1:
                raise ValueError("Bundle archive has ambiguous model_config.json")
            source_root = config_files[0].parent
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            shutil.copytree(source_root, stage_dir)
            if not (stage_dir / "model_checkpoint.pth").exists():
                raise FileNotFoundError("Bundle has no model_checkpoint.pth")
            stage_dir.rename(target)
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
    else:
        shutil.copytree(Path(bundle), target)

    checkpoint_path = target / "model_checkpoint.pth"
    if not checkpoint_path.exists():
        shutil.rmtree(target, ignore_errors=True)
        raise FileNotFoundError("Installed model has no model_checkpoint.pth")
    config = json.loads((target / "model_config.json").read_text(encoding="utf-8"))
    mapping = json.loads((target / "class_mapping.json").read_text(encoding="utf-8"))
    metrics = {}
    metric_file = target / "evaluation_metrics.json"
    if metric_file.exists():
        metrics = json.loads(metric_file.read_text(encoding="utf-8"))
    try:
        return db.execute(
            """INSERT INTO model_versions(model_key,version,architecture,source_species,taxonomy,classes_json,
               config_json,bundle_path,checkpoint_path,checkpoint_sha256,status,installed_at,metrics_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (package.model_key, package.version, package.architecture, package.source_species, package.taxonomy,
             json.dumps(mapping.get("label_to_name", mapping.get("classes", {}))), json.dumps(config),
             str(target), str(checkpoint_path), sha256_file(checkpoint_path), "candidate", utc_now(), json.dumps(metrics)),
        )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def package_for_model_row(row: dict[str, Any]) -> ModelPackage:
    root = Path(row["bundle_path"])
    classes_raw = json.loads(row["classes_json"])
    classes = {int(k): str(v) for k, v in classes_raw.items()}
    config = json.loads(row["config_json"])
    return ModelPackage(row["model_key"], row["version"], row["architecture"],
                        row["source_species"], row["taxonomy"], classes, config, root,
                        Path(row["checkpoint_path"]), row["checkpoint_sha256"],
                        json.loads(row.get("metrics_json") or "{}"))
