from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import TRACK_MASK_SIDE, TRACK_MAX_GAP_FRAMES
from .db import Database


def _small_mask(path: str, side: int = TRACK_MASK_SIDE) -> np.ndarray:
    with Image.open(path) as im:
        mask = im.convert("L")
        mask.thumbnail((side, side), Image.Resampling.NEAREST)
        array = np.asarray(mask, dtype=np.uint8) > 0
    return array


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        with Image.fromarray(a.astype(np.uint8) * 255) as im:
            im = im.resize((side := max(a.shape[1], b.shape[1]), side), Image.Resampling.NEAREST)
            a = np.asarray(im) > 0
        with Image.fromarray(b.astype(np.uint8) * 255) as im:
            im = im.resize((side, side), Image.Resampling.NEAREST)
            b = np.asarray(im) > 0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _similarity(track: dict[str, Any], det: dict[str, Any]) -> float:
    iou = _mask_iou(track["mask"], det["mask"])
    cx1, cy1 = track["center"]
    cx2, cy2 = det["center"]
    dist = math.hypot(cx1 - cx2, cy1 - cy2)
    diag = math.sqrt(2.0)
    area_ratio = min(track["area"], det["area"]) / max(track["area"], det["area"], 1.0)
    center_affinity = math.exp(-dist / 0.12)
    score = 0.65 * iou + 0.20 * center_affinity + 0.15 * area_ratio
    return score if dist <= 0.20 * diag and area_ratio >= 0.05 and score >= 0.30 else -1.0


def rebuild_sequence_tracks(db: Database, unit_id: int, camera_id: int, series_key: str,
                            model_version_id: int, max_missed_frames: int = TRACK_MAX_GAP_FRAMES) -> dict[str, int]:
    """Rebuild tracks in capture order, retaining only active downsampled masks in RAM."""
    # Keep captures with zero detections in the sequence: they are real missed frames
    # and must advance the occlusion/missed-frame counter.
    captures = db.query(
        """SELECT * FROM captures WHERE unit_id=? AND camera_id=? AND series_key=?
           ORDER BY COALESCE(captured_at,imported_at), id""",
        (unit_id, camera_id, series_key),
    )
    with db.transaction() as conn:
        conn.execute("DELETE FROM tracks WHERE unit_id=? AND camera_id=? AND series_key=?",
                     (unit_id, camera_id, series_key))
    active: dict[str, dict[str, Any]] = {}
    next_id = 1
    observed = 0
    for frame_index, capture in enumerate(captures):
        predictions = db.query(
            """SELECT * FROM predictions WHERE capture_id=? AND model_version_id=?
               AND final_for_analysis=1 ORDER BY confidence DESC""",
            (capture["id"], model_version_id),
        )
        detections: list[dict[str, Any]] = []
        width, height = float(capture["width"]), float(capture["height"])
        diagonal = math.hypot(width, height)
        for row in predictions:
            try:
                mask = _small_mask(row["mask_path"])
            except (OSError, FileNotFoundError):
                continue
            if not mask.any():
                continue
            center = (((float(row["x1"]) + float(row["x2"])) / 2) / max(width, 1),
                      ((float(row["y1"]) + float(row["y2"])) / 2) / max(height, 1))
            detections.append({"row": row, "mask": mask, "center": center, "area": float(row["area_px2"])})

        candidate_pairs = []
        for track_key, state in active.items():
            for di, det in enumerate(detections):
                score = _similarity(state, det)
                if score >= 0:
                    candidate_pairs.append((score, track_key, di))
        candidate_pairs.sort(reverse=True)
        used_tracks: set[str] = set()
        used_detections: set[int] = set()
        matches: dict[int, tuple[str, float]] = {}
        for score, track_key, di in candidate_pairs:
            if track_key not in used_tracks and di not in used_detections:
                used_tracks.add(track_key)
                used_detections.add(di)
                matches[di] = (track_key, score)

        for track_key in list(active):
            if track_key not in used_tracks:
                active[track_key]["missed"] += 1
                if active[track_key]["missed"] > max_missed_frames:
                    active.pop(track_key, None)

        for di, det in enumerate(detections):
            row = det["row"]
            if di in matches:
                track_key, score = matches[di]
                state = active[track_key]
                first_area = state["first_area"]
                first_capture = state["first_capture_id"]
                observations = state["observations"] + 1
            else:
                track_key = f"{series_key}-C{next_id:05d}"
                next_id += 1
                score = None
                first_area = det["area"]
                first_capture = int(capture["id"])
                observations = 1
            growth = 100.0 * (det["area"] / max(first_area, 1.0) - 1.0)
            track = db.one(
                """SELECT id FROM tracks WHERE unit_id=? AND camera_id=? AND series_key=? AND track_key=?""",
                (unit_id, camera_id, series_key, track_key),
            )
            if track is None:
                track_id = db.execute(
                    """INSERT INTO tracks(unit_id,camera_id,series_key,model_version_id,track_key,class_name,
                       first_capture_id,last_capture_id,first_area_px2,last_area_px2,observations,growth_pct,status)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (unit_id, camera_id, series_key, model_version_id, track_key, row["class_name"],
                     first_capture, capture["id"], first_area, det["area"], observations, growth, "active"),
                )
            else:
                track_id = int(track["id"])
                db.execute(
                    """UPDATE tracks SET class_name=?,last_capture_id=?,last_area_px2=?,observations=?,growth_pct=?,status='active'
                       WHERE id=?""",
                    (row["class_name"], capture["id"], det["area"], observations, growth, track_id),
                )
            mm_per_px = db.one("SELECT mm_per_px FROM cameras WHERE id=?", (camera_id,))
            area_mm2 = det["area"] * float(mm_per_px["mm_per_px"]) ** 2 if mm_per_px and mm_per_px["mm_per_px"] else None
            db.execute(
                """INSERT INTO track_observations(track_id,prediction_id,capture_id,frame_index,captured_at,
                   area_px2,area_mm2,growth_vs_first_pct,association_score) VALUES(?,?,?,?,?,?,?,?,?)""",
                (track_id, row["id"], capture["id"], frame_index, capture["captured_at"],
                 det["area"], area_mm2, growth, score),
            )
            active[track_key] = {
                "mask": det["mask"], "center": det["center"], "area": det["area"],
                "missed": 0, "first_area": first_area, "first_capture_id": first_capture,
                "observations": observations,
            }
            observed += 1
    db.execute("UPDATE tracks SET status='closed' WHERE unit_id=? AND camera_id=? AND series_key=?",
               (unit_id, camera_id, series_key))
    # A track's last state remains closed at the end of this batch; it can be rebuilt when a new capture arrives.
    return {"captures": len(captures), "observations": observed, "tracks": next_id - 1}
