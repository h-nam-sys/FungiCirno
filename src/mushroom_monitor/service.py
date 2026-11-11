from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import AppPaths, DEFAULT_INFERENCE_SIDE, DEFAULT_SCORE_THRESHOLD
from .db import Database, json_dumps, utc_now
from .inference.runtime import InferenceEngine
from .maturity import rebuild_flush_groups
from .tracking import rebuild_sequence_tracks


class ModelService:
    def __init__(self, db: Database, paths: AppPaths):
        self.db = db
        self.paths = paths
        self._engines: dict[tuple[int, int, str, float | None], InferenceEngine] = {}

    def engine_for(self, model_id: int, inference_side: int = DEFAULT_INFERENCE_SIDE,
                   device_preference: str = "cpu", score_threshold: float | None = None) -> InferenceEngine:
        row = self.db.one("SELECT * FROM model_versions WHERE id=?", (model_id,))
        if not row:
            raise ValueError("Model version is not registered")
        key = (model_id, inference_side, device_preference, score_threshold)
        if key not in self._engines:
            self._engines[key] = InferenceEngine(row, inference_side, device_preference, score_threshold)
        return self._engines[key]

    def clear_model_cache(self) -> None:
        self._engines.clear()

    def run_capture(self, capture_id: int, model_id: int, *, inference_side: int = DEFAULT_INFERENCE_SIDE,
                    device_preference: str = "cpu", final_threshold: float | None = None,
                    mask_threshold: float = 0.5, rebuild_derived: bool = True) -> dict[str, Any]:
        capture = self.db.one("SELECT * FROM captures WHERE id=?", (capture_id,))
        model_row = self.db.one("SELECT * FROM model_versions WHERE id=?", (model_id,))
        if not capture or not model_row:
            raise ValueError("Capture or model version not found")
        if not Path(capture["file_path"]).exists():
            raise FileNotFoundError(f"Capture file is missing: {capture['file_path']}")
        config = json.loads(model_row["config_json"])
        threshold = float(final_threshold if final_threshold is not None else
                          config.get("confidence_threshold", DEFAULT_SCORE_THRESHOLD))
        started = time.perf_counter()
        run_id = self.db.execute(
            """INSERT INTO inference_runs(capture_id,model_version_id,started_at,status)
               VALUES(?,?,?,'running')""",
            (capture_id, model_id, utc_now()),
        )
        try:
            engine = self.engine_for(model_id, inference_side, device_preference, threshold)
            predictions, image, info = engine.predict_path(Path(capture["file_path"]))
            original_w, original_h = image.size
            scale_x = original_w / info["inference_width"]
            scale_y = original_h / info["inference_height"]
            camera = self.db.one("SELECT mm_per_px FROM cameras WHERE id=?", (capture["camera_id"],))
            mm_per_px = float(camera["mm_per_px"]) if camera and camera["mm_per_px"] else None
            mask_folder = self.paths.masks / f"capture_{capture_id}" / f"run_{run_id}"
            mask_folder.mkdir(parents=True, exist_ok=True)
            created = 0
            for pred in predictions:
                x1, y1, x2, y2 = pred.bbox
                x1, x2 = max(0.0, x1 * scale_x), min(float(original_w), x2 * scale_x)
                y1, y2 = max(0.0, y1 * scale_y), min(float(original_h), y2 * scale_y)
                mask = np.asarray(pred.mask, dtype=np.uint8)
                area_small = int(mask.sum())
                area_px2 = float(area_small) * scale_x * scale_y
                ys, xs = np.where(mask > 0)
                centroid_x = float(xs.mean() * scale_x) if len(xs) else (x1 + x2) / 2
                centroid_y = float(ys.mean() * scale_y) if len(ys) else (y1 + y2) / 2
                mask_path = mask_folder / f"detection_{pred.detection_index:04d}.png"
                Image.fromarray(mask * 255, mode="L").save(mask_path, optimize=True)
                is_final = int(pred.confidence >= threshold)
                # Confidence controls automatic analysis inclusion, not human approval.
                # Every detection remains reviewable; only explicit human decisions enter learning exports.
                review_status = "pending"
                self.db.add_prediction(
                    run_id=run_id, capture_id=capture_id, model_version_id=model_id,
                    detection_index=pred.detection_index, class_id=pred.class_id, class_name=pred.class_name,
                    confidence=pred.confidence, x1=x1, y1=y1, x2=x2, y2=y2,
                    area_px2=area_px2, area_mm2=(area_px2 * mm_per_px * mm_per_px if mm_per_px else None),
                    centroid_x=centroid_x, centroid_y=centroid_y, mask_path=str(mask_path),
                    mask_width=int(mask.shape[1]), mask_height=int(mask.shape[0]),
                    review_status=review_status, final_for_analysis=is_final,
                    metadata_json=json_dumps({"scale_x": scale_x, "scale_y": scale_y,
                                              "inference_width": info["inference_width"],
                                              "inference_height": info["inference_height"]}),
                )
                created += 1
            model_key = str(model_row["model_key"])
            downstream: dict[str, Any] = {}
            if rebuild_derived:
                downstream["spatial_groups"] = rebuild_flush_groups(self.db, capture_id, model_version_id=int(model_id))
            if rebuild_derived and model_key in {"charisis_cluster_tracking", "m18k_yolo_seg"}:
                downstream["tracking"] = rebuild_sequence_tracks(
                    self.db, int(capture["unit_id"]), int(capture["camera_id"]),
                    str(capture["series_key"]), int(model_id),
                )
            elapsed = time.perf_counter() - started
            diagnostics = {**info, **downstream, "candidate_count": created,
                           "accepted_count": sum(1 for p in predictions if p.confidence >= threshold),
                           "final_threshold": threshold, "model_key": model_key}
            self.db.execute(
                """UPDATE inference_runs SET finished_at=?,device=?,duration_seconds=?,status='completed',
                   detection_count=?,diagnostics_json=? WHERE id=?""",
                (utc_now(), info["device"], elapsed, created, json_dumps(diagnostics), run_id),
            )
            self.db.execute("UPDATE captures SET status='processed' WHERE id=?", (capture_id,))
            return {"run_id": run_id, "detection_count": created, "accepted_count": diagnostics["accepted_count"],
                    "duration_seconds": round(elapsed, 2), "diagnostics": diagnostics}
        except Exception as exc:
            self.db.execute(
                "UPDATE inference_runs SET finished_at=?,status='failed',error=? WHERE id=?",
                (utc_now(), f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", run_id),
            )
            self.db.execute("UPDATE captures SET status='failed' WHERE id=?", (capture_id,))
            raise


def add_environment_rows(db: Database, unit_id: int, rows: list[dict[str, Any]], source: str = "csv") -> int:
    inserted = 0
    for row in rows:
        measured_at = str(row.get("measured_at") or row.get("timestamp") or "").strip()
        if not measured_at:
            continue
        from datetime import datetime
        try:
            datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid environmental timestamp: {measured_at}") from exc
        temp = row.get("temperature_c")
        humidity = row.get("humidity_pct")
        co2 = row.get("co2_ppm")
        if temp is not None and not -10 <= float(temp) <= 60:
            raise ValueError(f"Temperature outside accepted range (-10..60 C): {temp}")
        if humidity is not None and not 0 <= float(humidity) <= 100:
            raise ValueError(f"Humidity outside accepted range (0..100%): {humidity}")
        if co2 is not None and not 0 <= float(co2) <= 100000:
            raise ValueError(f"CO2 outside accepted range (0..100000 ppm): {co2}")
        existing = db.one("SELECT id FROM environmental_readings WHERE unit_id=? AND measured_at=? AND source=?", (unit_id, measured_at, source))
        if existing:
            db.execute("""UPDATE environmental_readings SET temperature_c=?,humidity_pct=?,co2_ppm=?,notes=? WHERE id=?""",
                       (temp, humidity, co2, row.get("notes"), existing["id"]))
        else:
            db.execute(
                """INSERT INTO environmental_readings(unit_id,measured_at,temperature_c,humidity_pct,co2_ppm,notes,source)
                   VALUES(?,?,?,?,?,?,?)""",
                (unit_id, measured_at, temp, humidity, co2, row.get("notes"), source),
            )
            inserted += 1
    return inserted
