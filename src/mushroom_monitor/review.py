from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .db import Database, utc_now


def review_prediction(db: Database, prediction_id: int, decision: str, *,
                      reviewer: str = "local_user", corrected_class: str | None = None,
                      notes: str = "") -> None:
    if decision not in {"accepted", "rejected", "corrected"}:
        raise ValueError("Decision must be accepted, rejected or corrected")
    pred = db.one(
        """SELECT p.*,m.classes_json,m.model_key,c.width,c.height FROM predictions p
           JOIN model_versions m ON m.id=p.model_version_id
           JOIN captures c ON c.id=p.capture_id WHERE p.id=?""", (prediction_id,)
    )
    if not pred:
        raise ValueError("Prediction not found")
    if decision == "corrected":
        if not corrected_class:
            raise ValueError("Select the corrected class")
        classes = json.loads(pred["classes_json"])
        reverse = {str(name): int(label) for label, name in ((int(k), v) for k, v in classes.items())}
        if corrected_class not in reverse:
            raise ValueError("Corrected class is not part of this model's taxonomy")
        db.execute("UPDATE predictions SET class_id=?,class_name=?,review_status='corrected',final_for_analysis=1 WHERE id=?",
                   (reverse[corrected_class], corrected_class, prediction_id))
    elif decision == "accepted":
        db.execute("UPDATE predictions SET review_status='accepted',final_for_analysis=1 WHERE id=?", (prediction_id,))
    else:
        db.execute("UPDATE predictions SET review_status='rejected',final_for_analysis=0 WHERE id=?", (prediction_id,))
    existing = db.one("SELECT id FROM review_items WHERE prediction_id=?", (prediction_id,))
    if existing:
        db.execute(
            """UPDATE review_items SET status=?,corrected_class=?,notes=?,reviewer=?,updated_at=? WHERE id=?""",
            (decision, corrected_class, notes, reviewer, utc_now(), existing["id"]),
        )
    else:
        db.execute(
            """INSERT INTO review_items(prediction_id,status,corrected_class,notes,reviewer,updated_at)
               VALUES(?,?,?,?,?,?)""",
            (prediction_id, decision, corrected_class, notes, reviewer, utc_now()),
        )


def export_reviewed_samples(db: Database, destination: Path, model_key: str) -> tuple[Path, int]:
    """Export accepted/relabelled masks as a small COCO polygon dataset for Colab review/retraining."""
    rows = db.query(
        """SELECT p.*,c.file_path,c.original_name,c.width,c.height,c.captured_at,m.model_key
           FROM predictions p JOIN captures c ON c.id=p.capture_id
           JOIN model_versions m ON m.id=p.model_version_id
           JOIN review_items r ON r.prediction_id=p.id
           WHERE m.model_key=? AND r.status IN ('accepted','corrected')
           ORDER BY p.capture_id,p.id""", (model_key,)
    )
    if not rows:
        raise ValueError("No human-accepted samples are available for export")
    destination.parent.mkdir(parents=True, exist_ok=True)
    images: dict[int, dict[str, Any]] = {}
    annotations: list[dict[str, Any]] = []
    next_ann = 1
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for row in rows:
            capture_id = int(row["capture_id"])
            if capture_id not in images:
                image_rel = f"images/capture_{capture_id}{Path(row['original_name']).suffix.lower()}"
                zf.write(row["file_path"], image_rel)
                images[capture_id] = {"id": capture_id, "file_name": image_rel,
                                      "width": int(row["width"]), "height": int(row["height"]),
                                      "date_captured": row["captured_at"] or ""}
            with Image.open(row["mask_path"]) as im:
                mask = (np.asarray(im.convert("L")) > 0).astype(np.uint8)
            if mask.shape != (int(row["height"]), int(row["width"])):
                mask = cv2.resize(mask, (int(row["width"]), int(row["height"])), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for contour in contours:
                if len(contour) < 3:
                    continue
                approx = cv2.approxPolyDP(contour, epsilon=1.0, closed=True).reshape(-1, 2)
                if len(approx) >= 3:
                    polygons.append(approx.astype(float).flatten().tolist())
            if not polygons:
                continue
            bbox = [float(row["x1"]), float(row["y1"]),
                    float(row["x2"] - row["x1"]), float(row["y2"] - row["y1"])]
            annotations.append({
                "id": next_ann, "image_id": capture_id, "category_id": int(row["class_id"]),
                "segmentation": polygons, "area": float(row["area_px2"]), "bbox": bbox,
                "iscrowd": 0, "source_prediction_id": int(row["id"]),
            })
            next_ann += 1
        categories = db.query(
            """SELECT DISTINCT p.class_id,p.class_name FROM predictions p
               JOIN model_versions m ON m.id=p.model_version_id
               WHERE m.model_key=? ORDER BY p.class_id""", (model_key,)
        )
        coco = {"images": list(images.values()), "annotations": annotations,
                "categories": [{"id": int(r["class_id"]), "name": r["class_name"]} for r in categories],
                "info": {"description": "Human-reviewed local mushroom monitoring samples",
                         "created_utc": datetime.now(timezone.utc).isoformat(),
                         "model_key": model_key}}
        zf.writestr("annotations.json", json.dumps(coco, indent=2))
        zf.writestr("README.txt", "These are human-reviewed prediction masks exported for candidate retraining. Review polygon simplification and image IDs before training. Original model predictions are not ground truth until reviewed.")
    db.execute("INSERT INTO learning_exports(created_at,path,sample_count,model_key,notes) VALUES(?,?,?,?,?)",
               (utc_now(), str(destination), len(annotations), model_key, "Human-reviewed COCO polygon export"))
    return destination, len(annotations)
