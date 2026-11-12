#!/usr/bin/env python
"""Package an Ultralytics YOLO-seg checkpoint as a FungiCirno model bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Ultralytics YOLO segmentation checkpoint")
    parser.add_argument("--out", default="model_bundle.zip")
    parser.add_argument("--model-key", required=True, help="e.g. charisis_cluster_tracking or mushr_maturity")
    parser.add_argument("--species", required=True, help="Species/domain represented by this checkpoint")
    parser.add_argument("--taxonomy", required=True, help="Stable taxonomy identifier")
    parser.add_argument(
        "--prediction-role", required=True,
        choices=("cluster_stage", "fruitbody_maturity", "generic_instance"),
    )
    parser.add_argument("--pretrained-source", default="")
    parser.add_argument("--version", default="")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Threshold for final automatic analysis; lower predictions remain reviewable")
    parser.add_argument("--score-floor", type=float, default=0.01,
                        help="Low inference floor used to retain uncertain predictions for human review")
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--metrics", default="", help="Optional evaluation_metrics.json")
    args = parser.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    for name, value in (("confidence", args.confidence), ("score-floor", args.score_floor),
                        ("nms-iou", args.nms_iou)):
        if not 0 <= value <= 1:
            raise ValueError(f"--{name} must be within [0, 1]")
    if args.score_floor > args.confidence:
        raise ValueError("--score-floor must be <= --confidence")
    if args.max_det < 1 or args.imgsz < 32:
        raise ValueError("--max-det must be positive and --imgsz at least 32")

    model = YOLO(str(weights), task="segment")
    raw_names: Any = model.names
    names = ({int(key): str(value) for key, value in raw_names.items()}
             if isinstance(raw_names, dict)
             else {index: str(value) for index, value in enumerate(raw_names)})
    if not names:
        raise ValueError("Checkpoint has no class names; refusing to create an incomplete mapping")
    ordered_names = [names[key].strip().lower() for key in sorted(names)]
    expected_taxonomies = {
        "cluster_stage": ["immature", "well-defined", "overstayed"],
        "fruitbody_maturity": ["not-ready", "ready", "overdue"],
    }
    expected = expected_taxonomies.get(args.prediction_role)
    expected_keys = {"cluster_stage": "charisis_cluster_tracking", "fruitbody_maturity": "mushr_maturity"}
    if args.prediction_role in expected_keys and args.model_key != expected_keys[args.prediction_role]:
        raise ValueError(f"{args.prediction_role} must use model_key={expected_keys[args.prediction_role]!r}")
    if args.model_key == "m18k_source_wb" and (args.prediction_role != "generic_instance" or ordered_names != ["wb"]):
        raise ValueError("m18k_source_wb must be generic_instance with exactly one WB class")
    if expected is not None and ordered_names != expected:
        raise ValueError(
            f"Prediction role {args.prediction_role!r} requires class order {expected}; "
            f"checkpoint contains {ordered_names}. Refusing to package a mismatched taxonomy."
        )
    if sorted(names) != list(range(len(names))):
        raise ValueError("YOLO model class IDs must be contiguous and zero-based")

    digest = sha256(weights)
    version = args.version or f"{args.model_key}-{digest[:12]}"
    track_objects = args.prediction_role == "cluster_stage"
    config = {
        "model_key": args.model_key,
        "version": version,
        "architecture": "Ultralytics YOLOv8-seg",
        "runtime_type": "ultralytics_yolo_seg",
        "task": "segment",
        "prediction_role": args.prediction_role,
        "track_objects": track_objects,
        "source_species": args.species,
        "taxonomy": args.taxonomy,
        "pretrained_source": args.pretrained_source,
        "names": {str(key): value for key, value in names.items()},
        "confidence_threshold": args.confidence,
        "score_floor": args.score_floor,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "imgsz": args.imgsz,
        "checkpoint_sha256": digest,
    }

    metrics_path = Path(args.metrics).expanduser().resolve() if args.metrics else None
    if metrics_path is not None and not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        archive.write(weights, "best.pt")
        archive.writestr("model_config.json", json.dumps(config, indent=2))
        archive.writestr(
            "class_mapping.json",
            json.dumps({"label_to_name": {str(key): value for key, value in names.items()}}, indent=2),
        )
        if metrics_path is not None:
            archive.write(metrics_path, "evaluation_metrics.json")
        archive.writestr(
            "README.txt",
            "FungiCirno YOLO-seg model bundle\n"
            f"Model key: {args.model_key}\nVersion: {version}\n"
            f"Prediction role: {args.prediction_role}\nTaxonomy: {args.taxonomy}\n"
            f"Checkpoint SHA256: {digest}\n"
            f"Created: {datetime.now(timezone.utc).isoformat()}\n"
            "Validate on an independent target-domain holdout and the intended local hardware before promotion.\n",
        )

    print(json.dumps({"bundle": str(output), "model_config": config}, indent=2))


if __name__ == "__main__":
    main()
