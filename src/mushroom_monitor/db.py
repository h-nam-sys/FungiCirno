from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = r"""
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sites (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, location TEXT, notes TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grow_units (
  id INTEGER PRIMARY KEY, site_id INTEGER NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
  code TEXT NOT NULL, species TEXT NOT NULL DEFAULT 'Pleurotus ostreatus',
  cultivar TEXT, substrate TEXT, batch_code TEXT, notes TEXT,
  UNIQUE(site_id, code)
);
CREATE TABLE IF NOT EXISTS cameras (
  id INTEGER PRIMARY KEY, unit_id INTEGER NOT NULL REFERENCES grow_units(id) ON DELETE CASCADE,
  code TEXT NOT NULL, view_name TEXT NOT NULL DEFAULT 'side',
  mm_per_px REAL, reference_width_mm REAL, reference_height_mm REAL, notes TEXT,
  UNIQUE(unit_id, code)
);
CREATE TABLE IF NOT EXISTS captures (
  id INTEGER PRIMARY KEY, unit_id INTEGER NOT NULL REFERENCES grow_units(id),
  camera_id INTEGER NOT NULL REFERENCES cameras(id), series_key TEXT NOT NULL DEFAULT 'default',
  original_name TEXT NOT NULL, file_path TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL UNIQUE,
  captured_at TEXT, time_source TEXT NOT NULL DEFAULT 'unknown', imported_at TEXT NOT NULL,
  width INTEGER NOT NULL, height INTEGER NOT NULL, source TEXT NOT NULL DEFAULT 'upload',
  metadata_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_captures_series_time ON captures(unit_id,camera_id,series_key,captured_at,imported_at);
CREATE TABLE IF NOT EXISTS model_versions (
  id INTEGER PRIMARY KEY, model_key TEXT NOT NULL, version TEXT NOT NULL,
  architecture TEXT NOT NULL, source_species TEXT NOT NULL, taxonomy TEXT NOT NULL,
  classes_json TEXT NOT NULL, config_json TEXT NOT NULL, bundle_path TEXT NOT NULL,
  checkpoint_path TEXT NOT NULL, checkpoint_sha256 TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'candidate', installed_at TEXT NOT NULL,
  metrics_json TEXT NOT NULL DEFAULT '{}', notes TEXT,
  UNIQUE(model_key, version)
);
CREATE INDEX IF NOT EXISTS idx_model_versions_key_status ON model_versions(model_key,status);
CREATE TABLE IF NOT EXISTS inference_runs (
  id INTEGER PRIMARY KEY, capture_id INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
  model_version_id INTEGER NOT NULL REFERENCES model_versions(id), started_at TEXT NOT NULL,
  finished_at TEXT, device TEXT, duration_seconds REAL, status TEXT NOT NULL,
  error TEXT, detection_count INTEGER DEFAULT 0, diagnostics_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES inference_runs(id) ON DELETE CASCADE,
  capture_id INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
  model_version_id INTEGER NOT NULL REFERENCES model_versions(id), detection_index INTEGER NOT NULL,
  class_id INTEGER NOT NULL, class_name TEXT NOT NULL, confidence REAL NOT NULL,
  x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
  area_px2 REAL NOT NULL, area_mm2 REAL, centroid_x REAL, centroid_y REAL,
  mask_path TEXT NOT NULL, mask_width INTEGER NOT NULL, mask_height INTEGER NOT NULL,
  review_status TEXT NOT NULL DEFAULT 'pending', final_for_analysis INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_predictions_capture ON predictions(capture_id);
CREATE TABLE IF NOT EXISTS flush_groups (
  id INTEGER PRIMARY KEY, capture_id INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
  model_version_id INTEGER REFERENCES model_versions(id),
  group_key TEXT NOT NULL, fruitbody_count INTEGER NOT NULL, total_area_px2 REAL NOT NULL,
  dominant_class TEXT, ready_area_fraction REAL, class_area_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS tracks (
  id INTEGER PRIMARY KEY, unit_id INTEGER NOT NULL REFERENCES grow_units(id),
  camera_id INTEGER NOT NULL REFERENCES cameras(id), series_key TEXT NOT NULL,
  model_version_id INTEGER NOT NULL REFERENCES model_versions(id), track_key TEXT NOT NULL,
  class_name TEXT, first_capture_id INTEGER REFERENCES captures(id),
  last_capture_id INTEGER REFERENCES captures(id), first_area_px2 REAL, last_area_px2 REAL,
  observations INTEGER NOT NULL DEFAULT 0, growth_pct REAL, status TEXT NOT NULL DEFAULT 'active',
  UNIQUE(unit_id,camera_id,series_key,track_key)
);
CREATE TABLE IF NOT EXISTS track_observations (
  id INTEGER PRIMARY KEY, track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
  capture_id INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
  frame_index INTEGER NOT NULL, captured_at TEXT, area_px2 REAL NOT NULL, area_mm2 REAL,
  growth_vs_first_pct REAL NOT NULL, association_score REAL
);
CREATE INDEX IF NOT EXISTS idx_track_obs_track_frame ON track_observations(track_id,frame_index);
CREATE TABLE IF NOT EXISTS environmental_readings (
  id INTEGER PRIMARY KEY, unit_id INTEGER NOT NULL REFERENCES grow_units(id) ON DELETE CASCADE,
  measured_at TEXT NOT NULL, temperature_c REAL, humidity_pct REAL, co2_ppm REAL,
  notes TEXT, source TEXT NOT NULL DEFAULT 'manual'
);
CREATE INDEX IF NOT EXISTS idx_env_unit_time ON environmental_readings(unit_id,measured_at);
CREATE TABLE IF NOT EXISTS learning_jobs (
  id INTEGER PRIMARY KEY, model_key TEXT NOT NULL, base_model_version_id INTEGER REFERENCES model_versions(id),
  created_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL DEFAULT 'prepared',
  dataset_path TEXT, candidate_path TEXT, config_json TEXT NOT NULL DEFAULT '{}',
  metrics_json TEXT NOT NULL DEFAULT '{}', error TEXT
);
CREATE TABLE IF NOT EXISTS model_evaluations (
  id INTEGER PRIMARY KEY, model_version_id INTEGER NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
  dataset_name TEXT NOT NULL, created_at TEXT NOT NULL, metrics_json TEXT NOT NULL,
  approved INTEGER NOT NULL DEFAULT 0, reviewer TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS review_items (
  id INTEGER PRIMARY KEY, prediction_id INTEGER NOT NULL UNIQUE REFERENCES predictions(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending', corrected_class TEXT, notes TEXT,
  reviewer TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_exports (
  id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, path TEXT NOT NULL,
  sample_count INTEGER NOT NULL, model_key TEXT NOT NULL, notes TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def initialize(self) -> None:
        conn = self.connect()
        try:
            conn.executescript(SCHEMA)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(flush_groups)").fetchall()}
            if "model_version_id" not in columns:
                conn.execute("ALTER TABLE flush_groups ADD COLUMN model_version_id INTEGER REFERENCES model_versions(id)")
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self.transaction() as conn:
            cur = conn.execute(sql, tuple(params))
            return int(cur.lastrowid or cur.rowcount)

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        conn = self.connect()
        try:
            return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]
        finally:
            conn.close()

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def ensure_defaults(self) -> None:
        if self.one("SELECT id FROM sites LIMIT 1"):
            return
        site_id = self.execute(
            "INSERT INTO sites(name,location,notes,created_at) VALUES(?,?,?,?)",
            ("Default Site", "", "Edit this site in Cultivation setup.", utc_now()),
        )
        unit_id = self.execute(
            "INSERT INTO grow_units(site_id,code,species,substrate,batch_code) VALUES(?,?,?,?,?)",
            (site_id, "UNIT-01", "Pleurotus ostreatus", "unspecified", "batch-01"),
        )
        self.execute(
            "INSERT INTO cameras(unit_id,code,view_name) VALUES(?,?,?)",
            (unit_id, "CAM-01", "side"),
        )

    def create_capture(self, **row: Any) -> int:
        cols = ["unit_id", "camera_id", "series_key", "original_name", "file_path", "sha256",
                "captured_at", "time_source", "imported_at", "width", "height", "source", "metadata_json", "status"]
        values = [row.get(c) for c in cols]
        return self.execute(
            f"INSERT INTO captures({','.join(cols)}) VALUES({','.join('?' for _ in cols)})", values
        )

    def add_prediction(self, **row: Any) -> int:
        cols = ["run_id", "capture_id", "model_version_id", "detection_index", "class_id", "class_name",
                "confidence", "x1", "y1", "x2", "y2", "area_px2", "area_mm2", "centroid_x", "centroid_y",
                "mask_path", "mask_width", "mask_height", "review_status", "final_for_analysis", "metadata_json"]
        return self.execute(
            f"INSERT INTO predictions({','.join(cols)}) VALUES({','.join('?' for _ in cols)})",
            [row.get(c) for c in cols],
        )

    def set_model_status(self, model_id: int, status: str, force: bool = False) -> None:
        model = self.one("SELECT model_key,metrics_json FROM model_versions WHERE id=?", (model_id,))
        if not model:
            raise ValueError("Unknown model version")
        if status == "active":
            metrics = json.loads(model.get("metrics_json") or "{}")
            if metrics and metrics.get("promotion_ready") is False and not force:
                raise ValueError("Candidate has not passed independent holdout promotion gates. Explicit override is required.")
        with self.transaction() as conn:
            if status == "active":
                conn.execute("UPDATE model_versions SET status='candidate' WHERE model_key=? AND status='active'", (model["model_key"],))
            conn.execute("UPDATE model_versions SET status=? WHERE id=?", (status, model_id))


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
