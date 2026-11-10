from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path
    data: Path
    captures: Path
    masks: Path
    exports: Path
    models: Path
    model_registry: Path
    incoming_models: Path
    database: Path

    @classmethod
    def from_env(cls) -> "AppPaths":
        root = Path(os.getenv("MUSHROOM_APP_HOME", Path(__file__).resolve().parents[2])).expanduser().resolve()
        data = root / "data"
        models = root / "models"
        paths = cls(
            root=root,
            data=data,
            captures=data / "captures",
            masks=data / "masks",
            exports=data / "exports",
            models=models,
            model_registry=models / "registry",
            incoming_models=models / "incoming",
            database=data / "mushroom_monitor.sqlite3",
        )
        for path in (paths.data, paths.captures, paths.masks, paths.exports,
                     paths.models, paths.model_registry, paths.incoming_models):
            path.mkdir(parents=True, exist_ok=True)
        return paths


DEFAULT_INFERENCE_SIDE = int(os.getenv("MUSHROOM_INFERENCE_SIDE", "768"))
DEFAULT_SCORE_THRESHOLD = float(os.getenv("MUSHROOM_SCORE_THRESHOLD", "0.30"))
TRACK_MAX_GAP_FRAMES = int(os.getenv("MUSHROOM_TRACK_MAX_GAP", "3"))
TRACK_MASK_SIDE = int(os.getenv("MUSHROOM_TRACK_MASK_SIDE", "192"))
