from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import Database


def parse_environment_csv(path: Path) -> list[dict[str, Any]]:
    accepted = {"measured_at", "timestamp", "temperature_c", "humidity_pct", "co2_ppm", "notes"}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        normalized = {name.strip().lower() for name in reader.fieldnames}
        if not ("measured_at" in normalized or "timestamp" in normalized):
            raise ValueError("CSV needs a measured_at or timestamp column")
        rows = []
        for row in reader:
            clean = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
            item: dict[str, Any] = {"measured_at": clean.get("measured_at") or clean.get("timestamp"),
                                    "notes": clean.get("notes") or ""}
            for key in ("temperature_c", "humidity_pct", "co2_ppm"):
                raw = clean.get(key)
                item[key] = float(raw) if raw not in (None, "") else None
            if item["measured_at"]:
                rows.append(item)
        return rows


def growth_environment_join(db: Database, unit_id: int) -> list[dict[str, Any]]:
    observations = db.query(
        """SELECT o.captured_at,o.area_px2,o.growth_vs_first_pct,t.track_key,t.camera_id
           FROM track_observations o JOIN tracks t ON t.id=o.track_id
           WHERE t.unit_id=? AND o.captured_at IS NOT NULL ORDER BY o.captured_at""", (unit_id,)
    )
    readings = db.query(
        "SELECT measured_at,temperature_c,humidity_pct,co2_ppm FROM environmental_readings WHERE unit_id=? ORDER BY measured_at",
        (unit_id,),
    )
    if not observations or not readings:
        return []
    from datetime import datetime
    parsed_env = []
    for row in readings:
        try:
            when = datetime.fromisoformat(row["measured_at"].replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.astimezone()
            else:
                when = when.astimezone()
            parsed_env.append((when, row))
        except ValueError:
            continue
    result = []
    for obs in observations:
        try:
            ts = datetime.fromisoformat(str(obs["captured_at"]).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.astimezone()
            else:
                ts = ts.astimezone()
        except ValueError:
            continue
        candidates = [(abs((when - ts).total_seconds()), row) for when, row in parsed_env]
        if not candidates:
            continue
        seconds, nearest = min(candidates, key=lambda item: item[0])
        # Avoid attaching environmental readings from a different day to an observation.
        if seconds > 12 * 3600:
            continue
        result.append({**obs, **nearest, "environment_offset_hours": seconds / 3600})
    return result
