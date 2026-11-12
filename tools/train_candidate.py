#!/usr/bin/env python
"""Fine-tune a YOLO-seg model for one FungiCirno task and package it as a candidate.

The training checkpoint and the optional comparison baseline are separate arguments:
- --weights: initialization used for fine-tuning (e.g. M18K WB-only best.pt).
- --baseline-weights: current model for the SAME target taxonomy, evaluated on the
  exact same untouched target holdout before candidate promotion.

For the first oyster cluster-stage model, omit --baseline-weights because the M18K
WB-only class taxonomy is not comparable to the oyster cluster-stage taxonomy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(model: YOLO, data: str, split: str, imgsz: int, device: str) -> dict[str, float]:
    result = model.val(data=data, split=split, imgsz=imgsz, device=device, plots=True, verbose=False)
    seg = getattr(result, "seg", None)
    box = getattr(result, "box", None)
    return {
        "seg_map50": float(getattr(seg, "map50", 0.0)),
        "seg_map50_95": float(getattr(seg, "map", 0.0)),
        "box_map50": float(getattr(box, "map50", 0.0)),
        "box_map50_95": float(getattr(box, "map", 0.0)),
    }


def names_dict(model: YOLO) -> dict[int, str]:
    raw = model.names
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    return {index: str(value) for index, value in enumerate(raw)}


def _split_image_paths(config: dict[str, Any], yaml_path: Path, split: str) -> set[Path]:
    """Resolve a local Ultralytics split into image paths for leakage checks."""
    import glob

    value = config.get(split)
    if value is None:
        raise ValueError(f"Dataset YAML is missing split {split!r}")
    root_value = config.get("path", yaml_path.parent)
    root = Path(root_value)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    values = value if isinstance(value, list) else [value]
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    found: set[Path] = set()
    for item in values:
        path = Path(str(item))
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_dir():
            found.update(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                entry = Path(line.strip())
                if not line.strip():
                    continue
                if not entry.is_absolute():
                    entry = root / entry
                found.add(entry.resolve())
        elif path.is_file() and path.suffix.lower() in allowed:
            found.add(path)
        else:
            matches = [Path(p).resolve() for p in glob.glob(str(path), recursive=True)
                       if Path(p).is_file() and Path(p).suffix.lower() in allowed]
            found.update(matches)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Initialization checkpoint, e.g. M18K best.pt")
    parser.add_argument("--data", required=True, help="YOLO data.yaml with train and val splits")
    parser.add_argument("--holdout-data", default="", help="Separate data.yaml containing an untouched test split")
    parser.add_argument("--holdout-split", choices=("val", "test"), default="test")
    parser.add_argument("--holdout-id", default="", help="Human-readable dataset/sequence identifier saved in model metadata")
    parser.add_argument("--baseline-weights", default="", help="Existing same-taxonomy target model for regression comparison")
    parser.add_argument("--model-key", default="charisis_cluster_tracking")
    parser.add_argument("--species", default="Pleurotus ostreatus")
    parser.add_argument("--taxonomy", default="charisis_oyster_cluster_stage_v1")
    parser.add_argument("--prediction-role", choices=("cluster_stage", "fruitbody_maturity"), default="cluster_stage")
    parser.add_argument("--pretrained-source", default="M18K WB-only checkpoint; target-domain transfer must be evaluated")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Colab GPU index (usually 0) or cpu")
    parser.add_argument("--project", default="runs/fungicirno")
    parser.add_argument("--name", default="oyster_candidate")
    parser.add_argument("--min-seg-map50", type=float, default=0.50)
    parser.add_argument("--min-seg-map50-95", type=float, default=0.30)
    parser.add_argument("--allowed-ap-drop", type=float, default=0.01)
    parser.add_argument("--out", default="artifacts/oyster_candidate_bundle.zip")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not data.is_file():
        raise FileNotFoundError(data)
    data_config = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    if not all(key in data_config for key in ("train", "val", "names")):
        raise ValueError("Training data.yaml must define train, val and names")
    target_class_names = (
        [str(data_config["names"][key]) for key in sorted(data_config["names"], key=lambda x: int(x))]
        if isinstance(data_config["names"], dict)
        else [str(value) for value in data_config["names"]]
    )
    expected_taxonomies = {
        "cluster_stage": ["immature", "well-defined", "overstayed"],
        "fruitbody_maturity": ["not-ready", "ready", "overdue"],
    }
    expected_names = expected_taxonomies[args.prediction_role]
    expected_model_keys = {"cluster_stage": "charisis_cluster_tracking", "fruitbody_maturity": "mushr_maturity"}
    if args.model_key != expected_model_keys[args.prediction_role]:
        raise ValueError(f"{args.prediction_role} must use model_key={expected_model_keys[args.prediction_role]!r}")
    if [name.strip().lower() for name in target_class_names] != expected_names:
        raise ValueError(
            f"{args.prediction_role} requires class order {expected_names}; "
            f"data.yaml contains {target_class_names}"
        )
    if args.holdout_data:
        holdout_yaml = Path(args.holdout_data).resolve()
        if not holdout_yaml.is_file():
            raise FileNotFoundError(holdout_yaml)
        holdout_config = yaml.safe_load(holdout_yaml.read_text(encoding="utf-8")) or {}
        if args.holdout_split not in holdout_config:
            raise ValueError(f"Holdout data.yaml must define `{args.holdout_split}`")
        holdout_names = holdout_config.get("names")
        if holdout_names is None:
            raise ValueError("Holdout data.yaml must define the same `names` class mapping")
        holdout_class_names = (
            [str(holdout_names[key]) for key in sorted(holdout_names, key=lambda x: int(x))]
            if isinstance(holdout_names, dict)
            else [str(value) for value in holdout_names]
        )
        if [name.strip().lower() for name in holdout_class_names] != expected_names:
            raise ValueError("Holdout class mapping/order does not match the training taxonomy")
        train_paths = _split_image_paths(data_config, data, "train")
        val_paths = _split_image_paths(data_config, data, "val")
        holdout_paths = _split_image_paths(holdout_config, holdout_yaml, args.holdout_split)
        overlap = (train_paths | val_paths) & holdout_paths
        if overlap:
            sample = sorted(str(path) for path in overlap)[:5]
            raise ValueError(
                f"Holdout leakage: {len(overlap)} image(s) also appear in train/validation; examples: {sample}"
            )
        if not holdout_paths:
            raise ValueError("No local image files were resolved from the requested holdout split")
        if args.holdout_id:
            holdout_id = args.holdout_id
        else:
            holdout_id = f"{holdout_yaml.name}:{sha256(holdout_yaml)[:12]}"
    else:
        holdout_yaml = None
        holdout_id = None
    if args.baseline_weights and not args.holdout_data:
        raise ValueError("--baseline-weights requires --holdout-data to compare both models on the same holdout")
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        raise ValueError("epochs, batch and imgsz must be positive")

    baseline_metrics = None
    baseline_names = None
    if args.baseline_weights:
        baseline_path = Path(args.baseline_weights).resolve()
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        baseline = YOLO(str(baseline_path), task="segment")
        baseline_names = names_dict(baseline)
        baseline_class_names = [baseline_names[k] for k in sorted(baseline_names)]
        if baseline_class_names != target_class_names:
            raise ValueError(
                "Baseline class names/order do not match the training taxonomy. "
                "Do not compare M18K WB metrics with oyster stage metrics; omit --baseline-weights for initial transfer."
            )
        baseline_metrics = metric_values(baseline, str(holdout_yaml), args.holdout_split, args.imgsz, args.device)

    model = YOLO(str(weights), task="segment")
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name=args.name,
        patience=12, save=True, plots=True, close_mosaic=10,
    )
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce {best}")
    candidate = YOLO(str(best), task="segment")
    candidate_names = names_dict(candidate)
    candidate_class_names = [candidate_names[key].strip().lower() for key in sorted(candidate_names)]
    if candidate_class_names != expected_names:
        raise ValueError(
            f"Trained checkpoint class order {candidate_class_names} does not match required {expected_names}; "
            "refusing to package a potentially mislabeled model."
        )

    target_metrics = None
    promotion_ready = False
    promotion_reason = "No untouched target holdout was supplied"
    if holdout_yaml is not None:
        target_metrics = metric_values(candidate, str(holdout_yaml), args.holdout_split, args.imgsz, args.device)
        absolute_pass = (
            target_metrics["seg_map50"] >= args.min_seg_map50
            and target_metrics["seg_map50_95"] >= args.min_seg_map50_95
        )
        regression_pass = True
        if baseline_metrics is not None:
            regression_pass = (
                target_metrics["seg_map50"] >= baseline_metrics["seg_map50"] - args.allowed_ap_drop
                and target_metrics["seg_map50_95"] >= baseline_metrics["seg_map50_95"] - args.allowed_ap_drop
            )
        promotion_ready = bool(absolute_pass and regression_pass)
        if not absolute_pass:
            promotion_reason = "Candidate failed the configured absolute target-domain holdout gates"
        elif not regression_pass:
            promotion_reason = "Candidate regressed beyond the configured tolerance on the same target holdout"
        elif baseline_metrics is None:
            promotion_reason = "Passed absolute target-domain holdout gates; no same-taxonomy baseline was supplied"
        else:
            promotion_reason = "Passed target-domain holdout gates and same-taxonomy regression comparison"

    metrics: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "initialization_checkpoint": str(weights),
        "initialization_sha256": sha256(weights),
        "candidate_checkpoint_sha256": sha256(best),
        "training": {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "device": args.device},
        "target_holdout_id": holdout_id,
        "target_holdout_split": args.holdout_split if holdout_yaml else None,
        "target_holdout_metrics": target_metrics,
        "baseline_holdout_metrics": baseline_metrics,
        "baseline_checkpoint_sha256": sha256(Path(args.baseline_weights).resolve()) if args.baseline_weights else None,
        "promotion_ready": promotion_ready,
        "promotion_reason": promotion_reason,
        "promotion_gates": {
            "min_seg_map50": args.min_seg_map50,
            "min_seg_map50_95": args.min_seg_map50_95,
            "allowed_ap_drop": args.allowed_ap_drop if args.baseline_weights else None,
        },
    }
    config = {
        "model_key": args.model_key,
        "version": f"{args.model_key}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "architecture": "Ultralytics YOLOv8n-seg",
        "runtime_type": "ultralytics_yolo_seg",
        "task": "segment",
        "prediction_role": args.prediction_role,
        "track_objects": args.prediction_role == "cluster_stage",
        "source_species": args.species,
        "taxonomy": args.taxonomy,
        "pretrained_source": args.pretrained_source,
        "names": {str(k): v for k, v in candidate_names.items()},
        "confidence_threshold": 0.25,
        "score_floor": 0.01,
        "nms_iou": 0.70,
        "max_det": 1000,
        "imgsz": args.imgsz,
    }

    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fungicirno_candidate_") as temp_dir:
        staging = Path(temp_dir)
        shutil.copy2(best, staging / "best.pt")
        (staging / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (staging / "class_mapping.json").write_text(
            json.dumps({"label_to_name": {str(k): v for k, v in candidate_names.items()}}, indent=2),
            encoding="utf-8",
        )
        (staging / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (staging / "README.txt").write_text(
            "FungiCirno YOLO-seg candidate. This is not an active model. Install as a candidate and inspect "
            "evaluation_metrics.json before activation. Initial M18K-to-oyster transfer has a different taxonomy "
            "and must not be compared against WB source-domain metrics.\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for path in sorted(staging.iterdir()):
                archive.write(path, path.name)

    print(json.dumps({"candidate_bundle": str(output), "model_config": config, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
