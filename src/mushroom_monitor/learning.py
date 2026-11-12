from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .db import Database, utc_now


def _decode_class_map(value: str) -> dict[int, str]:
    raw = json.loads(value or "{}")
    result = {int(key): str(name) for key, name in raw.items()}
    if not result:
        raise ValueError("Model family has no registered class mapping")
    return result


def _validated_family_metadata(rows: list[dict[str, Any]]) -> tuple[dict[int, str], str]:
    class_map = _decode_class_map(rows[0]["classes_json"])
    taxonomy = str(rows[0]["taxonomy"])
    for row in rows[1:]:
        if _decode_class_map(row["classes_json"]) != class_map:
            raise ValueError("Model versions under this model_key use different class mappings; export them separately")
        if str(row["taxonomy"]) != taxonomy:
            raise ValueError("Model versions under this model_key use different taxonomies; export them separately")
    return class_map, taxonomy


def export_reviewed_yolo_dataset(
    db: Database, destination: Path, model_key: str, reviewer_note: str = ""
) -> tuple[Path, int]:
    """Export human-reviewed predictions as a sequence-grouped YOLO-seg dataset.

    The full model taxonomy is retained even when some classes have zero reviewed examples.
    A final test holdout is never created from these review exports.
    """
    rows = db.query(
        """SELECT p.*,c.file_path,c.original_name,c.width,c.height,c.captured_at,
                  c.series_key,c.unit_id,c.camera_id,m.classes_json,m.taxonomy,
                  r.status AS human_status
           FROM predictions p
           JOIN captures c ON c.id=p.capture_id
           JOIN model_versions m ON m.id=p.model_version_id
           JOIN review_items r ON r.prediction_id=p.id
           WHERE m.model_key=? AND r.status IN ('accepted','corrected')
             AND p.run_id=(SELECT MAX(r2.id) FROM inference_runs r2
                           WHERE r2.capture_id=p.capture_id AND r2.model_version_id=p.model_version_id)
           ORDER BY p.capture_id,p.id""",
        (model_key,),
    )
    if not rows:
        raise ValueError("No accepted/corrected predictions for this model family")
    class_map, taxonomy = _validated_family_metadata(rows)
    ordered_old_ids = sorted(class_map)
    old_to_yolo_id = {old_id: new_id for new_id, old_id in enumerate(ordered_old_ids)}
    yolo_names = {old_to_yolo_id[old_id]: class_map[old_id] for old_id in ordered_old_ids}

    series_groups = sorted({
        f"{row['unit_id']}|{row['camera_id']}|{row['series_key']}" for row in rows
    })
    if len(series_groups) < 2:
        raise ValueError(
            "At least two distinct unit/camera/series groups are required for a sequence-disjoint "
            "train/validation export. Add another reviewed series or use an existing public training dataset."
        )
    val_count = max(1, round(len(series_groups) * 0.2))
    val_groups = set(series_groups[-val_count:])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    labels: dict[str, list[str]] = defaultdict(list)
    image_records: dict[int, dict[str, Any]] = {}
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for row in rows:
                capture_id = int(row["capture_id"])
                group_key = f"{row['unit_id']}|{row['camera_id']}|{row['series_key']}"
                split = "val" if group_key in val_groups else "train"
                if capture_id not in image_records:
                    ext = Path(row["original_name"]).suffix.lower() or ".jpg"
                    rel = f"images/{split}/capture_{capture_id}{ext}"
                    archive.write(row["file_path"], rel)
                    image_records[capture_id] = {"path": rel, "split": split, "group": group_key}
                elif image_records[capture_id]["split"] != split:
                    raise ValueError(f"Capture {capture_id} was assigned to multiple series splits")

                mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Reviewed mask is missing/unreadable: {row['mask_path']}")
                if mask.shape != (int(row["height"]), int(row["width"])):
                    mask = cv2.resize(mask, (int(row["width"]), int(row["height"])), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    raise ValueError(f"Reviewed mask for prediction {row['id']} has no foreground")
                contour = max(contours, key=cv2.contourArea)
                if len(contour) < 3:
                    raise ValueError(f"Reviewed mask for prediction {row['id']} has a degenerate contour")
                approx = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
                if len(approx) < 3:
                    raise ValueError(f"Reviewed mask for prediction {row['id']} has a degenerate polygon")
                coords: list[float] = []
                for x, y in approx:
                    coords.extend([
                        min(1.0, max(0.0, float(x) / max(1, int(row["width"])))),
                        min(1.0, max(0.0, float(y) / max(1, int(row["height"])))),
                    ])
                label_path = f"labels/{split}/capture_{capture_id}.txt"
                class_id = int(row["class_id"])
                if class_id not in old_to_yolo_id:
                    raise ValueError(f"Prediction {row['id']} uses class ID {class_id} outside the registered taxonomy")
                labels[label_path].append(
                    str(old_to_yolo_id[class_id]) + " " + " ".join(f"{value:.6f}" for value in coords)
                )

            for name, lines in labels.items():
                archive.writestr(name, "\n".join(lines) + "\n")
            yaml_text = "path: .\ntrain: images/train\nval: images/val\nnames:\n" + "".join(
                f"  {index}: {json.dumps(class_name)}\n" for index, class_name in yolo_names.items()
            )
            archive.writestr("data.yaml", yaml_text)
            manifest = {
                "model_key": model_key,
                "taxonomy": taxonomy,
                "created_at": utc_now(),
                "images": len(image_records),
                "annotations": sum(len(lines) for lines in labels.values()),
                "class_mapping": {str(key): value for key, value in yolo_names.items()},
                "split_groups": {"train": sorted(set(series_groups) - val_groups), "val": sorted(val_groups)},
                "split_note": "Grouped by complete unit/camera/series; no final test split is included.",
                "human_review_note": reviewer_note,
            }
            archive.writestr("dataset_manifest.json", json.dumps(manifest, indent=2))
            archive.writestr(
                "README.txt",
                "Human-reviewed YOLO segmentation export. Full model taxonomy is preserved even when some "
                "classes have zero reviewed examples. Train/validation are grouped by complete capture series. "
                "This export is not a final test holdout. Inspect labels before training.\n",
            )
        temp.replace(destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    count = sum(len(lines) for lines in labels.values())
    db.execute(
        "INSERT INTO learning_exports(created_at,path,sample_count,model_key,notes) VALUES(?,?,?,?,?)",
        (utc_now(), str(destination), count, model_key, f"Sequence-grouped YOLO export: {taxonomy}"),
    )
    return destination, count
