from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .db import Database


def _connected_groups(points: np.ndarray, eps: float) -> np.ndarray:
    """DBSCAN-equivalent connected components for min_samples=1."""
    n = len(points)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(points[i] - points[j])) <= eps:
                union(i, j)
    roots: dict[int, int] = {}
    labels = np.empty(n, dtype=np.int32)
    for i in range(n):
        root = find(i)
        roots.setdefault(root, len(roots))
        labels[i] = roots[root]
    return labels


def rebuild_flush_groups(db: Database, capture_id: int, multiplier: float = 5.0) -> int:
    capture = db.one("SELECT width,height FROM captures WHERE id=?", (capture_id,))
    if not capture:
        raise ValueError("Capture not found")
    rows = db.query(
        """SELECT * FROM predictions WHERE capture_id=? AND final_for_analysis=1
           ORDER BY confidence DESC""", (capture_id,)
    )
    db.execute("DELETE FROM flush_groups WHERE capture_id=?", (capture_id,))
    if not rows:
        return 0
    centers, radii = [], []
    for row in rows:
        centers.append([float(row["centroid_x"] if row["centroid_x"] is not None else (row["x1"] + row["x2"]) / 2),
                        float(row["centroid_y"] if row["centroid_y"] is not None else (row["y1"] + row["y2"]) / 2)])
        radii.append(math.sqrt(max(float(row["area_px2"]), 1.0) / math.pi))
    eps = max(1.0, float(multiplier) * float(np.mean(radii)))
    labels = _connected_groups(np.asarray(centers, dtype=np.float32), eps)
    count = 0
    for label in sorted(set(labels.tolist())):
        members = [r for r, lab in zip(rows, labels) if int(lab) == int(label)]
        class_area: dict[str, float] = defaultdict(float)
        for row in members:
            class_area[str(row["class_name"])] += float(row["area_px2"])
        total = float(sum(class_area.values()))
        dominant = max(class_area, key=class_area.get)
        ready_name = next((name for name in class_area if name.strip().lower() == "ready"), None)
        ready_fraction = class_area[ready_name] / total if ready_name and total else None
        db.execute(
            """INSERT INTO flush_groups(capture_id,group_key,fruitbody_count,total_area_px2,
               dominant_class,ready_area_fraction,class_area_json) VALUES(?,?,?,?,?,?,?)""",
            (capture_id, f"{capture_id}-flush-{label+1}", len(members), total, dominant,
             ready_fraction, json.dumps(class_area)),
        )
        count += 1
    return count
