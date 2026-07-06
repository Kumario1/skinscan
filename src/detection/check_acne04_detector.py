"""Render and score the ACNE04 detector."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

from ..config import load_config


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path(cfg["detection"]["weights"]))
    p.add_argument("--images", type=Path, default=Path("data/raw/acne04/Classification/JPEGImages"))
    p.add_argument("--annotations", type=Path, default=Path("data/raw/acne04/Detection/VOC2007/Annotations"))
    p.add_argument("--split", type=Path, default=Path("data/raw/acne04/Detection/VOC2007/ImageSets/Main/NNEW_test_0.txt"))
    p.add_argument("--out", type=Path, default=Path("runs/detection_check"))
    p.add_argument("--conf", type=float, default=cfg["detection"]["conf_threshold"])
    p.add_argument("--iou", type=float, default=cfg["detection"]["iou_threshold"])
    p.add_argument("--imgsz", type=int, default=cfg["detection"]["img_size"])
    p.add_argument("--render-limit", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="debug only: limit split images")
    return p.parse_args()


def require(path, label):
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")


def split_ids(path):
    return [Path(line.split()[0]).stem for line in path.read_text().splitlines() if line.strip()]


def gt_boxes(root, stem):
    xml = ET.parse(root / f"{stem}.xml").getroot()
    boxes = []
    for obj in xml.findall("object"):
        b = obj.find("bndbox")
        boxes.append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    return boxes


def box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter else 0


def match_count(pred, gt, threshold):
    pairs = sorted(
        ((box_iou(p, g), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(gt)),
        reverse=True,
    )
    used_p, used_g, matches = set(), set(), 0
    for score, pi, gi in pairs:
        if score < threshold:
            break
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matches += 1
    return matches


def render_sheet(items, out_path, cols=2, cell_w=520, cell_h=430):
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(items):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        image.thumbnail((360, 360))
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + 36))
        draw.text((x + 8, y + 8), label[:82], fill=(0, 0, 0))
    sheet.save(out_path, quality=92)


def main():
    args = parse_args()
    for path, label in [
        (args.weights, "YOLO weights"),
        (args.images, "ACNE04 images"),
        (args.annotations, "ACNE04 annotations"),
        (args.split, "ACNE04 split"),
    ]:
        require(path, label)

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    ids = split_ids(args.split)
    if args.limit:
        ids = ids[:args.limit]
    thresholds = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
    stats = {t: {"pred": 0, "gt": 0, "m20": 0, "m30": 0} for t in thresholds}
    renders = []

    for i, stem in enumerate(ids, 1):
        img_path = args.images / f"{stem}.jpg"
        if not img_path.exists():
            continue
        result = model.predict(str(img_path), conf=0.03, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        raw = [(*b.xyxy[0].tolist(), float(b.conf)) for b in result.boxes]
        gt = gt_boxes(args.annotations, stem)
        for t in thresholds:
            pred = [list(b[:4]) for b in raw if b[4] >= t]
            stats[t]["pred"] += len(pred)
            stats[t]["gt"] += len(gt)
            stats[t]["m20"] += match_count(pred, gt, 0.2)
            stats[t]["m30"] += match_count(pred, gt, 0.3)

        if len(renders) < args.render_limit:
            pred = [b for b in raw if b[4] >= args.conf]
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            for box in gt:
                draw.rectangle(box, outline=(0, 190, 0), width=3)
            for x0, y0, x1, y1, conf in pred:
                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                draw.text((x0, max(0, y0 - 12)), f"{conf:.2f}", fill=(255, 0, 0))
            renders.append((f"{stem} | gt {len(gt)} pred {len(pred)}", image))
        if i % 50 == 0:
            print(i, "done")

    rows = []
    for t, s in stats.items():
        for key, iou in [("m20", 0.2), ("m30", 0.3)]:
            precision = s[key] / s["pred"] if s["pred"] else 0
            recall = s[key] / s["gt"] if s["gt"] else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            rows.append({
                "conf": t,
                "iou": iou,
                "pred": s["pred"],
                "gt": s["gt"],
                "matches": s[key],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

    (args.out / "threshold_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
    render_sheet(renders, args.out / "gt_green_pred_red_sheet.jpg")
    print("\nconf  iou pred  P     R     F1")
    for row in rows:
        print(f"{row['conf']:.2f} {row['iou']:.2f} {row['pred']:5d} {row['precision']:.3f} {row['recall']:.3f} {row['f1']:.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
