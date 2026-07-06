"""Run Stage 1 detector outputs through the acne type classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .classifier import AcneTypeClassifier, crop_with_context


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", type=Path, default=Path("models/detection/acne04_yolov8m_best.pt"))
    p.add_argument("--classifier", type=Path, default=Path("models/classification/acne_type_efficientnetb0.keras"))
    p.add_argument("--images", type=Path, default=Path("data/raw/acne04/Classification/JPEGImages"))
    p.add_argument("--out", type=Path, default=Path("runs/acne04_pipeline_check"))
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--max-boxes", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.07)
    p.add_argument("--iou", type=float, default=0.2)
    p.add_argument("--imgsz", type=int, default=1024)
    return p.parse_args()


def require(path, label):
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")


def image_files(root):
    return sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def draw_sheet(items, out_path, cell=180):
    if not items:
        return
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
    require(args.classifier, "Keras acne type classifier")
    require(args.images, "ACNE04 image directory")

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.detector))
    clf = AcneTypeClassifier(args.classifier)
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
            crop = crop_with_context(image, (x0, y0, x1 - x0, y1 - y0))
            probs = clf.predict(crop)
            label, prob = max(probs.items(), key=lambda kv: kv[1])
            sheet_items.append((crop, f"{label} {prob:.2f}"))
            image_records.append({
                "box": [x0, y0, x1, y1],
                "detector_conf": float(box.conf),
                "prediction": label,
                "probability": prob,
                "probs": probs,
            })
        draw_sheet(sheet_items, args.out / f"{img_path.stem}_crops.jpg")
        records.append({"image": str(img_path), "detections": image_records})
        print(img_path.name, len(image_records), "detections")

    (args.out / "predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
