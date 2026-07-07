"""Run Stage 1 detector outputs through the acne type classifier."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .classifier import AcneTypeClassifier, crop_with_context, read_model_metadata
from ..config import load_config
from ..utils import require


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--detector", type=Path, default=Path(cfg["detection"]["weights"]))
    p.add_argument("--classifier", type=Path, default=Path(cfg["classification"]["weights"]))
    p.add_argument("--image", type=Path, help="single uploaded/full-face image to run end-to-end")
    p.add_argument("--images", type=Path, default=Path("data/raw/acne04/Classification/JPEGImages"))
    p.add_argument("--out", type=Path, default=Path("runs/acne04_pipeline_check"))
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--max-boxes", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=0, help="classifier input size; default reads metadata or uses 224")
    p.add_argument("--crop-pad", type=float, default=cfg["classification"]["crop_pad"])
    p.add_argument("--collage-tiles", type=int, default=9)
    p.add_argument("--crops-only", action="store_true", help="skip classifier and only write detector crop inputs")
    p.add_argument("--conf", type=float, default=cfg["detection"]["conf_threshold"])
    p.add_argument("--iou", type=float, default=cfg["detection"]["iou_threshold"])
    p.add_argument("--imgsz", type=int, default=cfg["detection"]["img_size"])
    return p.parse_args()


def image_files(root):
    if root.is_file():
        return [root]
    return sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def selected_images(image, images, limit):
    paths = image_files(image) if image else image_files(images)
    return paths if image else paths[:limit]


def classifier_image_size(model_path, fallback=224):
    return int(read_model_metadata(model_path).get("image_size", fallback))


def draw_input_collage(crops, out_path, size, max_tiles=9):
    sheet = Image.new("RGB", (size, size), "white")
    if crops:
        crops = crops[:max_tiles]
        cols = math.ceil(math.sqrt(len(crops)))
        rows = math.ceil(len(crops) / cols)
        for i, crop in enumerate(crops):
            col, row = i % cols, i // cols
            x0, x1 = col * size // cols, (col + 1) * size // cols
            y0, y1 = row * size // rows, (row + 1) * size // rows
            sheet.paste(Image.fromarray(crop).resize((x1 - x0, y1 - y0)), (x0, y0))
    sheet.save(out_path)


def draw_sheet(items, out_path, cell=None):
    if not items:
        return
    cell = cell or items[0][0].shape[0]
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 36)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (crop, label) in enumerate(items):
        x, y = (i % cols) * cell, (i // cols) * (cell + 36)
        im = Image.fromarray(crop).resize((cell, cell))
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + cell + 4), label[:28], fill=(0, 0, 0))
    sheet.save(out_path)


def acne_type_counts(detections, classes=None):
    counts = Counter(d["prediction"] for d in detections if "prediction" in d)
    if classes:
        return {name: counts[name] for name in classes if counts[name]}
    return dict(sorted(counts.items()))


def load_rgb(path):
    """EXIF-corrected pixels — must match the orientation YOLO sees (cv2 applies EXIF)."""
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def analyze_image(img_path, model, clf, out_dir, *, crop_size, crop_pad, max_boxes, conf, iou, imgsz, collage_tiles):
    image = load_rgb(img_path)
    result = model.predict(
        str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )[0]
    crops = []
    detections = []
    for box in result.boxes[:max_boxes]:
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        crop = crop_with_context(image, (x0, y0, x1 - x0, y1 - y0), pad=crop_pad, size=crop_size)
        crop_path = out_dir / f"{img_path.stem}_crop_{len(detections) + 1:02d}.jpg"
        Image.fromarray(crop).save(crop_path, quality=92)
        crops.append(crop)
        detections.append({
            "box": [x0, y0, x1, y1],
            "detector_conf": float(box.conf),
            "input_crop": str(crop_path),
        })

    sheet_items = []
    if clf:
        for crop, record, probs in zip(crops, detections, clf.predict_batch(crops)):
            label, prob = max(probs.items(), key=lambda kv: kv[1])
            sheet_items.append((crop, f"{label} {prob:.2f}"))
            record.update({"prediction": label, "probability": prob, "probs": probs})
    else:
        for crop, record in zip(crops, detections):
            sheet_items.append((crop, f"conf {record['detector_conf']:.2f}"))

    collage_path = out_dir / f"{img_path.stem}_input_collage.jpg"
    draw_input_collage([crop for crop, _ in sheet_items], collage_path, crop_size, collage_tiles)
    draw_sheet(sheet_items, out_dir / f"{img_path.stem}_crops.jpg", cell=crop_size)
    type_counts = acne_type_counts(detections, clf.classes if clf else None)
    return {
        "image": str(img_path),
        "crop_size": crop_size,
        "collage_max_tiles": collage_tiles,
        "input_collage": str(collage_path),
        "detection_count": len(detections),
        "acne_types": list(type_counts),
        "acne_type_counts": type_counts,
        "detections": detections,
    }


def main():
    args = parse_args()
    require(args.detector, "YOLO detector weights")
    if not args.crops_only:
        require(args.classifier, "Keras acne type classifier")
    require(args.image if args.image else args.images, "input image path")

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.detector))
    clf = None if args.crops_only else AcneTypeClassifier(args.classifier)
    crop_size = args.crop_size or (clf.image_size if clf else classifier_image_size(args.classifier))
    if crop_size <= 0:
        raise SystemExit("--crop-size must be positive")
    if args.collage_tiles <= 0:
        raise SystemExit("--collage-tiles must be positive")
    records = []

    for img_path in selected_images(args.image, args.images, args.limit):
        record = analyze_image(
            img_path,
            model,
            clf,
            args.out,
            crop_size=crop_size,
            crop_pad=args.crop_pad,
            max_boxes=args.max_boxes,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            collage_tiles=args.collage_tiles,
        )
        records.append(record)
        types = ", ".join(record["acne_types"]) or "none"
        print(img_path.name, record["detection_count"], "detections", "types:", types)

    (args.out / "predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
