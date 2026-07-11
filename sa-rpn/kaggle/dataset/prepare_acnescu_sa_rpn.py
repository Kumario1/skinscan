"""Prepare the AcneSCU Pascal VOC mirror for the paper's SA-RPN training.

Implements Zhang et al.'s masked-crop preprocessing: split each high-resolution
face into 1024x1024 equal-overlap tiles, retain lesions fully contained in a
tile, and paint partial lesions black so they cannot become false negatives.
Outputs COCO instance JSON and images in the layout expected by the authors'
MMDetection 2.15 repository.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = (
    "closed_comedo", "open_comedo", "papule", "pustule", "nodule",
    "atrophic_scar", "hypertrophic_scar", "melasma", "nevus", "other",
)


@dataclass(frozen=True)
class Lesion:
    label: str
    bbox: tuple[float, float, float, float]
    polygon: tuple[tuple[float, float], ...]


def axis_starts(length: int, tile_size: int = 1024) -> list[int]:
    """Paper grid: ceil(length/tile) tiles, distributed with equal overlap."""
    count = max(1, math.ceil(length / tile_size))
    if count == 1:
        return [0]
    return [round(i * (length - tile_size) / (count - 1)) for i in range(count)]


def _point_number(tag: str) -> int:
    match = re.search(r"\d+", tag)
    return int(match.group()) if match else 0


def parse_voc(path: Path) -> tuple[str, int, int, list[Lesion]]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    filename = root.findtext("filename")
    lesions = []
    for obj in root.findall("object"):
        label = obj.findtext("name")
        if label not in CLASSES:
            raise ValueError(f"unknown AcneSCU class {label!r} in {path}")
        box = obj.find("bndbox")
        bbox = tuple(float(box.findtext(name)) for name in ("xmin", "ymin", "xmax", "ymax"))
        polygon_node = obj.find("polygon")
        xs, ys = {}, {}
        if polygon_node is not None:
            for child in polygon_node:
                number = _point_number(child.tag)
                (xs if child.tag.startswith("x") else ys)[number] = float(child.text)
        polygon = tuple((xs[i], ys[i]) for i in sorted(xs.keys() & ys.keys()))
        if len(polygon) < 3:
            x0, y0, x1, y1 = bbox
            polygon = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        lesions.append(Lesion(label, bbox, polygon))
    return filename, width, height, lesions


def polygon_area(points: list[float]) -> float:
    pairs = list(zip(points[::2], points[1::2]))
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pairs, pairs[1:] + pairs[:1]))) / 2


def fully_inside(bbox, left, top, right, bottom) -> bool:
    x0, y0, x1, y1 = bbox
    return left <= x0 and top <= y0 and x1 <= right and y1 <= bottom


def intersects(bbox, left, top, right, bottom) -> bool:
    x0, y0, x1, y1 = bbox
    return max(x0, left) < min(x1, right) and max(y0, top) < min(y1, bottom)


def split_sources(xml_paths: list[Path], test_count=28, seed=42):
    paths = sorted(xml_paths)
    if not 0 < test_count < len(paths):
        raise ValueError("test_count must leave at least one train image")
    shuffled = paths.copy()
    random.Random(seed).shuffle(shuffled)
    test = set(shuffled[:test_count])
    return [p for p in paths if p not in test], [p for p in paths if p in test]


def prepare_split(xml_paths: list[Path], source_dir: Path, image_out: Path,
                  annotation_out: Path, *, tile_size=1024):
    image_out.mkdir(parents=True, exist_ok=True)
    categories = [{"id": i + 1, "name": name, "supercategory": "acne"} for i, name in enumerate(CLASSES)]
    category_ids = {row["name"]: row["id"] for row in categories}
    images, annotations = [], []
    image_id = annotation_id = 1

    for xml_path in xml_paths:
        filename, width, height, lesions = parse_voc(xml_path)
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        for row, top in enumerate(axis_starts(height, tile_size)):
            for column, left in enumerate(axis_starts(width, tile_size)):
                right, bottom = min(left + tile_size, width), min(top + tile_size, height)
                tile = source.crop((left, top, right, bottom))
                kept, partial = [], []
                for lesion in lesions:
                    if fully_inside(lesion.bbox, left, top, right, bottom):
                        kept.append(lesion)
                    elif intersects(lesion.bbox, left, top, right, bottom):
                        partial.append(lesion)
                if not kept:
                    continue

                draw = ImageDraw.Draw(tile)
                for lesion in partial:
                    shifted = [(x - left, y - top) for x, y in lesion.polygon]
                    draw.polygon(shifted, fill="black")

                output_name = f"{Path(filename).stem}_crop_r{row}_c{column}.jpg"
                tile.save(image_out / output_name, quality=95)
                images.append({
                    "id": image_id, "file_name": output_name,
                    "width": tile.width, "height": tile.height,
                    "source_image": filename, "position": {
                        "x_index": column, "x_len": len(axis_starts(width, tile_size)),
                        "y_index": row, "y_len": len(axis_starts(height, tile_size)),
                    },
                })
                for lesion in kept:
                    x0, y0, x1, y1 = lesion.bbox
                    segmentation = [value for x, y in lesion.polygon for value in (x - left, y - top)]
                    annotations.append({
                        "id": annotation_id, "image_id": image_id,
                        "category_id": category_ids[lesion.label],
                        "bbox": [x0 - left, y0 - top, x1 - x0, y1 - y0],
                        "area": polygon_area(segmentation),
                        "segmentation": [segmentation], "iscrowd": 0,
                    })
                    annotation_id += 1
                image_id += 1

    annotation_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"images": images, "annotations": annotations, "categories": categories}
    annotation_out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return {"source_images": len(xml_paths), "tiles": len(images), "annotations": len(annotations)}


def prepare_dataset(voc_root: Path, out: Path, *, test_count=28, seed=42, tile_size=1024):
    source_dir = voc_root / "train" if (voc_root / "train").is_dir() else voc_root
    xml_paths = sorted(source_dir.glob("*.xml"))
    if not xml_paths:
        raise SystemExit(f"no Pascal VOC XML files under {source_dir}")
    train, valid = split_sources(xml_paths, test_count=test_count, seed=seed)
    reports = {
        "train": prepare_split(
            train, source_dir, out / "images/train_crop_1024",
            out / "annotations/train_crop_1024.json", tile_size=tile_size,
        ),
        "valid": prepare_split(
            valid, source_dir, out / "images/val_crop_1024",
            out / "annotations/val_crop_1024.json", tile_size=tile_size,
        ),
    }
    metadata = {
        "paper": "Zhang et al., Learning High-quality Proposals for Acne Detection",
        "paper_url": "https://arxiv.org/abs/2207.03674",
        "classes": list(CLASSES), "seed": seed, "test_count": test_count,
        "tile_size": tile_size, "reports": reports,
        "limitation": "Roboflow mirror has 275 source images; paper reports 276. Patient IDs are unavailable.",
    }
    (out / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-count", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = prepare_dataset(
        args.voc_root, args.out, test_count=args.test_count,
        seed=args.seed, tile_size=args.tile_size,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
