#!/usr/bin/env python
"""Convert split COCO instance-segmentation annotations into Ultralytics YOLO-seg.

The caller supplies train/val/test COCO JSONs separately. This is deliberate:
sequence-level split decisions must be made before conversion to avoid time-lapse
leakage between training and the final test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _polygon_mask(polygons: list[Any], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = []
    for polygon in polygons:
        values = np.asarray(polygon, dtype=np.float32).reshape(-1)
        if values.size < 6 or values.size % 2:
            continue
        xy = np.rint(values.reshape(-1, 2)).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
        if len(xy) >= 3:
            points.append(xy)
    if points:
        cv2.fillPoly(mask, points, 1)
    return mask


def _rle_mask(segmentation: dict[str, Any], height: int, width: int) -> np.ndarray:
    counts = segmentation.get("counts")
    if isinstance(counts, list):
        flat = np.zeros(height * width, dtype=np.uint8)
        offset = 0
        value = 0
        for run in counts:
            end = offset + int(run)
            if end > flat.size or int(run) < 0:
                raise ValueError("Invalid uncompressed COCO RLE run")
            if value:
                flat[offset:end] = 1
            offset = end
            value = 1 - value
        if offset != flat.size:
            raise ValueError("COCO RLE does not cover the declared image size")
        return flat.reshape((height, width), order="F")
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "Compressed COCO RLE requires pycocotools. Install it with "
            "`pip install -r requirements-research.txt`."
        ) from exc
    rle = dict(segmentation)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return (decoded > 0).astype(np.uint8)


def segmentation_mask(segmentation: Any, width: int, height: int) -> np.ndarray:
    if isinstance(segmentation, list):
        return _polygon_mask(segmentation, width, height)
    if isinstance(segmentation, dict):
        return _rle_mask(segmentation, height, width)
    return np.zeros((height, width), dtype=np.uint8)


def mask_to_yolo_polygon(mask: np.ndarray) -> list[float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # YOLO-seg stores one polygon per instance. Keep the largest connected outer
    # contour if an annotation contains disconnected fragments.
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3 or cv2.contourArea(contour) <= 0:
        return None
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(0.5, perimeter * 0.001), True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    height, width = mask.shape
    out: list[float] = []
    for x, y in approx:
        out.extend((min(1.0, max(0.0, float(x) / width)),
                    min(1.0, max(0.0, float(y) / height))))
    return out if len(out) >= 6 else None


def read_coco(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"{path} is missing a COCO `{key}` array")
    return payload


def convert_split(
    split: str,
    annotation_path: Path,
    image_root: Path,
    output_root: Path,
    category_to_label: dict[int, int],
) -> dict[str, Any]:
    coco = read_coco(annotation_path)
    images = {int(row["id"]): row for row in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if int(ann.get("iscrowd", 0)):
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    image_out = output_root / "images" / split
    label_out = output_root / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    counts: Counter[int] = Counter()
    written_images = 0
    written_instances = 0
    for image_id, image_info in images.items():
        rel = Path(str(image_info["file_name"]))
        source = (image_root / rel).resolve()
        root = image_root.resolve()
        if root not in source.parents and source != root:
            raise ValueError(f"Image path escapes image root: {rel}")
        if not source.is_file():
            raise FileNotFoundError(f"Image listed in {annotation_path} is missing: {source}")
        width, height = int(image_info["width"]), int(image_info["height"])
        image_name = f"{image_id:08d}_{rel.name}"
        shutil.copy2(source, image_out / image_name)
        lines: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            category_id = int(ann["category_id"])
            if category_id not in category_to_label:
                raise ValueError(f"Unknown category_id={category_id} in {annotation_path}")
            mask = segmentation_mask(ann.get("segmentation"), width, height)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            polygon = mask_to_yolo_polygon(mask)
            if polygon is None:
                continue
            label_id = category_to_label[category_id]
            lines.append(str(label_id) + " " + " ".join(f"{value:.6f}" for value in polygon))
            counts[label_id] += 1
            written_instances += 1
        label_name = Path(image_name).with_suffix(".txt")
        (label_out / label_name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written_images += 1

    return {
        "annotation_file": str(annotation_path),
        "annotation_sha256": sha256(annotation_path),
        "images": written_images,
        "instances": written_instances,
        "instances_by_label": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Root directory containing COCO file_name paths")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--test-json", required=True, help="Untouched, sequence-disjoint final test annotations")
    parser.add_argument("--out", required=True, help="Output YOLO dataset directory")
    parser.add_argument("--class-order", default="", help="Optional comma-separated category names in desired class-ID order")
    args = parser.parse_args()

    image_root = Path(args.images).resolve()
    annotation_paths = {name: Path(getattr(args, f"{name}_json")).resolve() for name in ("train", "val", "test")}
    for path in [image_root, *annotation_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    coco_by_split = {name: read_coco(path) for name, path in annotation_paths.items()}

    categories_by_split = {}
    for split, coco in coco_by_split.items():
        categories = {int(x["id"]): str(x["name"]).strip() for x in coco["categories"]}
        if not categories or any(not name for name in categories.values()):
            raise ValueError(f"{split} split has no valid category definitions")
        categories_by_split[split] = categories
    reference_names = {name.strip().casefold() for name in categories_by_split["train"].values()}
    for split, categories in categories_by_split.items():
        if {name.strip().casefold() for name in categories.values()} != reference_names:
            raise ValueError(f"COCO category names differ between train and {split} splits")

    requested_order = [x.strip() for x in args.class_order.split(",") if x.strip()]
    if requested_order:
        class_names = requested_order
        if {name.casefold() for name in class_names} != reference_names or len(class_names) != len(reference_names):
            raise ValueError("--class-order must contain each COCO category name exactly once (case-insensitive)")
    else:
        class_names = [name for _, name in sorted(categories_by_split["train"].items())]

    output_root = Path(args.out).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    split_results = {}
    category_maps = {}
    for split, categories in categories_by_split.items():
        by_name = {name.strip().casefold(): category_id for category_id, name in categories.items()}
        category_maps[split] = {by_name[name.casefold()]: index for index, name in enumerate(class_names)}
        split_results[split] = convert_split(
            split, annotation_paths[split], image_root, output_root, category_maps[split]
        )

    yaml_payload = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
    manifest = {
        "format": "ultralytics_yolo_instance_segmentation",
        "classes": {str(index): name for index, name in enumerate(class_names)},
        "splits": split_results,
        "split_policy": "Train, validation and untouched test COCO files are supplied separately by the user. Verify they are split by complete timelapse/farm sequences, not random adjacent frames.",
        "conversion_note": "Each COCO instance is exported as its largest connected outer contour because a YOLO-seg label row stores one polygon per instance. Inspect exported labels for fragmented/holed masks.",
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "README.txt").write_text(
        "Converted COCO instance masks to Ultralytics YOLO-seg. The train/val/test annotation files were kept separate.\n"
        "Before training, inspect random labels and confirm no time-lapse sequence appears in more than one split.\n"
        + manifest["conversion_note"] + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset_yaml": str(output_root / "data.yaml"), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
