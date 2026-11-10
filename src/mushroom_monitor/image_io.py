from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ExifTags, UnidentifiedImageError

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
EXIF_DATETIME_TAGS = {36867: "DateTimeOriginal", 306: "DateTime"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_exif_datetime(image: Image.Image) -> datetime | None:
    try:
        exif = image.getexif()
        raw = exif.get(36867) or exif.get(306)
        if not raw:
            return None
        parsed = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
        return parsed
    except (ValueError, TypeError, OSError):
        return None


def safe_name(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base or "capture.jpg"


def inspect_image(data: bytes) -> tuple[int, int, datetime | None, str]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            captured = parse_exif_datetime(image)
            fmt = image.format or Path("capture.jpg").suffix.lstrip(".").upper()
        if width < 2 or height < 2:
            raise ValueError("Image dimensions are invalid")
        return width, height, captured, fmt
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unsupported or corrupt image: {exc}") from exc


def store_capture(data: bytes, original_name: str, destination_root: Path) -> tuple[Path, str, int, int, datetime | None]:
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported image extension: {suffix}")
    width, height, exif_time, _ = inspect_image(data)
    digest = sha256_bytes(data)
    folder = destination_root / digest[:2]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{digest}{suffix}"
    if not target.exists():
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)
    return target, digest, width, height, exif_time


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def resize_for_inference(image: Image.Image, max_side: int) -> tuple[Image.Image, float, float]:
    if max_side < 128:
        raise ValueError("Inference side must be at least 128 pixels")
    w, h = image.size
    scale = min(1.0, float(max_side) / max(w, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if (nw, nh) == (w, h):
        return image, 1.0, 1.0
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    return resized, w / nw, h / nh
