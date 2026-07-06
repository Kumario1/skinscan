"""Run Stage 1 detector outputs through the acne type classifier."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .classifier import AcneTypeClassifier, crop_with_context


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", type=Path, default=Path("models/detection/acne04_yolov8m_best.pt"))
    p.add_argument("--classifier", type=Path, default=Path("models/classification/acne_model.keras"))
    p.add_argument("--images", type=Path, default=Path("data/raw/acne04/Classification/JPEGImages"))
    p.add_argument("--out", type=Path, default=Path("runs/acne04_pipeline_check"))
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--max-boxes", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=0, help="classifier input size; default reads metadata or uses 224")
    p.add_argument("--crop-pad", type=float, default=1.5)
    p.add_argument("--collage-tiles", type=int, default=9)
    p.add_argument("--crops-only", action="store_true", help="skip classifier and only write detector crop inputs")
    p.add_argument("--conf", type=float, default=0.07)
    p.add_argument("--iou", type=float, default=0.2)
    p.add_argument("--imgsz", type=int, default=1024)
    return p.parse_args()


def require(path, label):
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")


def image_files(root):
    return sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def classifier_image_size(model_path, fallback=224):
    meta = Path(model_path).with_suffix(Path(model_path).suffix + ".labels.json")
    if not meta.exists():
        return fallback
    return int(json.loads(meta.read_text()).get("image_size", fallback))


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


def main():
    args = parse_args()
    require(args.detector, "YOLO detector weights")
    if not args.crops_only:
        require(args.classifier, "Keras acne type classifier")
    require(args.images, "ACNE04 image directory")

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

    for img_path in image_files(args.images)[:args.limit]:
        image = np.asarray(Image.open(img_path).convert("RGB"))
        result = model.predict(
            str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        sheet_items = []
        image_records = []
        for box in result.boxes[:args.max_boxes]:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            crop = crop_with_context(image, (x0, y0, x1 - x0, y1 - y0), pad=args.crop_pad, size=crop_size)
            crop_path = args.out / f"{img_path.stem}_crop_{len(image_records) + 1:02d}.jpg"
            Image.fromarray(crop).save(crop_path, quality=92)
            record = {
                "box": [x0, y0, x1, y1],
                "detector_conf": float(box.conf),
                "input_crop": str(crop_path),
            }
            if clf:
                probs = clf.predict(crop)
                label, prob = max(probs.items(), key=lambda kv: kv[1])
                sheet_items.append((crop, f"{label} {prob:.2f}"))
                record.update({"prediction": label, "probability": prob, "probs": probs})
            else:
                sheet_items.append((crop, f"conf {float(box.conf):.2f}"))
            image_records.append(record)
        collage_path = args.out / f"{img_path.stem}_input_collage.jpg"
        draw_input_collage([crop for crop, _ in sheet_items], collage_path, crop_size, args.collage_tiles)
        draw_sheet(sheet_items, args.out / f"{img_path.stem}_crops.jpg", cell=crop_size)
        records.append({
            "image": str(img_path),
            "crop_size": crop_size,
            "collage_max_tiles": args.collage_tiles,
            "input_collage": str(collage_path),
            "detections": image_records,
        })
        print(img_path.name, len(image_records), "detections")

    (args.out / "predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
