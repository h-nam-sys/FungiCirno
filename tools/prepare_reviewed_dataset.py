#!/usr/bin/env python
"""Convert a FungiCirno reviewed COCO ZIP into sequence-grouped YOLO-seg data."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zip")
    parser.add_argument("--out", default="reviewed_yolo_dataset")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-image-split", action="store_true",
        help="Explicitly allow a random image split when sequence group metadata is unavailable; this can leak adjacent frames.",
    )
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be strictly between 0 and 1")

    output = Path(args.out).resolve()
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    with zipfile.ZipFile(args.input_zip) as archive:
        archive.extractall(output / "source")
    annotation_files = list((output / "source").rglob("annotations.json"))
    if not annotation_files:
        raise FileNotFoundError("annotations.json not found in review export")
    root = annotation_files[0].parent
    coco = json.loads(annotation_files[0].read_text(encoding="utf-8"))

    categories = sorted(coco["categories"], key=lambda item: int(item["id"]))
    if not categories:
        raise ValueError("COCO export has no categories")
    old_to_new = {int(category["id"]): index for index, category in enumerate(categories)}
    names = {index: str(category["name"]) for index, category in enumerate(categories)}
    images = {int(image["id"]): image for image in coco["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)

    group_by_image = {
        image_id: str(record.get("split_group", "")).strip()
        for image_id, record in images.items()
    }
    distinct_groups = sorted({group for group in group_by_image.values() if group})
    split_note = ""
    if len(distinct_groups) >= 2 and all(group_by_image.values()):
        shuffled = distinct_groups[:]
        random.Random(args.seed).shuffle(shuffled)
        val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * args.val_fraction)))
        val_groups = set(shuffled[:val_count])
        split_by_image = {
            image_id: ("val" if group_by_image[image_id] in val_groups else "train")
            for image_id in images
        }
        split_note = "Train/validation split grouped by complete unit/camera/series metadata."
    elif args.allow_image_split and len(images) >= 2:
        ids = sorted(images)
        random.Random(args.seed).shuffle(ids)
        val_count = max(1, round(len(ids) * args.val_fraction))
        val_ids = set(ids[:val_count])
        split_by_image = {image_id: ("val" if image_id in val_ids else "train") for image_id in images}
        split_note = "WARNING: random image-level split used because sequence group metadata was missing; adjacent-frame leakage may occur."
    else:
        raise ValueError(
            "Reviewed export lacks at least two complete `split_group` values. Export samples from multiple "
            "unit/camera/series groups. Use --allow-image-split only for a clearly marked temporary experiment."
        )

    for split in ("train", "val"):
        (output / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (output / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    written_annotations = 0
    for image_id, record in images.items():
        split = split_by_image[image_id]
        source_image = root / str(record["file_name"])
        if not source_image.is_file():
            matches = list(root.rglob(Path(str(record["file_name"])).name))
            if len(matches) != 1:
                raise FileNotFoundError(f"Image path is missing or ambiguous: {record['file_name']}")
            source_image = matches[0]
        destination_image = output / f"images/{split}/{image_id}{source_image.suffix.lower()}"
        shutil.copy2(source_image, destination_image)
        width, height = int(record["width"]), int(record["height"])
        lines: list[str] = []
        for annotation in annotations.get(image_id, []):
            old_class = int(annotation["category_id"])
            if old_class not in old_to_new:
                raise ValueError(f"Annotation references unknown category_id={old_class}")
            segmentation = annotation.get("segmentation", [])
            if isinstance(segmentation, dict):
                from pycocotools import mask as mask_utils
                binary = mask_utils.decode(segmentation).astype("uint8")
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygons = [contour.reshape(-1, 2).flatten().tolist() for contour in contours if len(contour) >= 3]
            else:
                polygons = segmentation
            valid_polygons = [polygon for polygon in polygons if len(polygon) >= 6]
            if not valid_polygons:
                continue
            polygon = max(valid_polygons, key=len)
            normalized: list[float] = []
            for index in range(0, len(polygon), 2):
                normalized.extend([
                    min(1.0, max(0.0, float(polygon[index]) / max(1, width))),
                    min(1.0, max(0.0, float(polygon[index + 1]) / max(1, height))),
                ])
            lines.append(str(old_to_new[old_class]) + " " + " ".join(f"{value:.6f}" for value in normalized))
            written_annotations += 1
        (output / f"labels/{split}/{image_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    manifest = {
        "source_model_key": coco.get("info", {}).get("model_key"),
        "taxonomy": coco.get("info", {}).get("taxonomy"),
        "images": len(images),
        "annotations_written": written_annotations,
        "classes": names,
        "split_note": split_note,
        "groups": sorted(distinct_groups),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "final_test_included": False,
    }
    (output / "dataset_info.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_yaml": str(output / "data.yaml"), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
