"""ACNE04 -> YOLO format converter.

Pipeline stage 1 data prep (DECISIONS.md D-010). Converts bounding boxes into
the YOLO txt format Ultralytics expects: one .txt per image, each line
`class_id x_center y_center width height`, all coordinates NORMALIZED to [0,1].

IMPORTANT — inspect before you parse (the GGC HAR-teardown instinct):
ACNE04's raw detection annotations are redistributed in *several* formats. The
official LDL release is converted through a VOC-XML intermediate; other copies
ship flat-text or COCO JSON. Run `inspect_raw()` on your actual download FIRST,
confirm which format you have, then point the converter at the right parser.
The geometry (`voc_box_to_yolo`) is format-independent and is the part that
actually matters for correctness — it's unit-tested (tests/test_voc_to_yolo.py).
"""
from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class Box:
    """A bounding box in VOC corner format (absolute pixels)."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float


# --- the part that must be correct: geometry (unit-tested) -----------------
def voc_box_to_yolo(box: Box, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Corner pixels -> normalized (x_center, y_center, width, height).

    YOLO wants the box CENTER (not top-left) and everything divided by image
    size so it's resolution-independent. This is where the classic bugs live:
    forgetting to use center instead of corner, or dividing x by height.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"bad image size: {img_w}x{img_h}")
    # clamp to image bounds — ACNE04 boxes occasionally spill past the edge
    xmin = max(0.0, min(box.xmin, img_w))
    ymin = max(0.0, min(box.ymin, img_h))
    xmax = max(0.0, min(box.xmax, img_w))
    ymax = max(0.0, min(box.ymax, img_h))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"degenerate box after clamp: {box}")
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def yolo_line(class_id: int, yolo_box: tuple[float, float, float, float]) -> str:
    xc, yc, w, h = yolo_box
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# --- parser A: VOC XML (the standard intermediate) -------------------------
def parse_voc_xml(xml_path: str) -> tuple[int, int, list[Box]]:
    """Returns (width, height, boxes) from a PASCAL VOC .xml annotation."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append(Box(
            float(b.find("xmin").text), float(b.find("ymin").text),
            float(b.find("xmax").text), float(b.find("ymax").text),
        ))
    return w, h, boxes


# --- parser B: flat text (some ACNE04 copies ship this) --------------------
def parse_flat_line(line: str) -> tuple[str, list[Box]]:
    """Parse a line like:  `img.jpg  x1,y1,x2,y2  x1,y1,x2,y2 ...`
    Adjust the split/delimiter here once you've seen your real file. This is
    the ONE function to touch if your raw format differs — geometry downstream
    is unchanged."""
    parts = line.split()
    name = parts[0]
    boxes = []
    for tok in parts[1:]:
        nums = [float(v) for v in tok.replace(",", " ").split()]
        if len(nums) >= 4:
            boxes.append(Box(*nums[:4]))
    return name, boxes


# --- inspection: run this on your download before converting ---------------
def inspect_raw(root_dir: str, n: int = 3) -> None:
    """Print the directory tree (2 levels) and a sample annotation so you can
    confirm the real format before committing to a parser."""
    print(f"== tree of {root_dir} ==")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath[len(root_dir):].count(os.sep)
        if depth > 1:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/")
        for f in sorted(filenames)[:5]:
            print(f"{indent}  {f}")
        if len(filenames) > 5:
            print(f"{indent}  ... (+{len(filenames)-5} more)")
    print("\n== inspect the first annotation file by hand before trusting a parser ==")


def convert_voc_dir(xml_dir: str, out_label_dir: str, class_id: int = 0) -> dict:
    """Convert a directory of VOC XMLs into YOLO label txts. Returns a small
    report (the import-log habit from GGC): counts, skips, and reasons."""
    os.makedirs(out_label_dir, exist_ok=True)
    report = {"images": 0, "boxes": 0, "skipped_boxes": 0, "empty_images": 0}
    for fn in os.listdir(xml_dir):
        if not fn.endswith(".xml"):
            continue
        w, h, boxes = parse_voc_xml(os.path.join(xml_dir, fn))
        lines = []
        for b in boxes:
            try:
                lines.append(yolo_line(class_id, voc_box_to_yolo(b, w, h)))
                report["boxes"] += 1
            except ValueError:
                report["skipped_boxes"] += 1
        stem = os.path.splitext(fn)[0]
        with open(os.path.join(out_label_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        report["images"] += 1
        if not lines:
            report["empty_images"] += 1
    return report
