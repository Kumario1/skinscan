"""Harvest Not_acne negative crops for the Stage 2 classifier.

Executes docs/STAGE2_NEGATIVES_DESIGN.md. Every negative is a real detector box
run through the *same* crop_with_context transform used at inference, so the
model learns lesion-presence, not crop style (the design doc's central
constraint). Two sources:

  --mode ffhq    clear-skin faces (D-013): every detector box is a false positive.
  --mode acne04  dermatologist-boxed images (D-010): keep detector boxes with
                 zero IoU against all GT boxes (off-lesion false positives).

Both run YOLO at the locked operating point (conf 0.07 / IoU 0.2 / imgsz 1024,
D-018 / configs/default.yaml). Crops split 60/20/20 into
<out>/{train,valid,test}/Not_acne/ — the class-per-directory layout
train_type_classifier.py already consumes.

Size the negative set from live counts FIRST:
  python -m src.classification.train_type_classifier --inspect
then cap with --limit so Not_acne stays <= the largest real class (design doc).
Hold out an FFHQ sheet the detector never sees here for the reject-rate eval.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image

from .classifier import crop_with_context
from .run_acne04_pipeline import image_files, load_rgb
from ..config import load_config
from ..detection.check_acne04_detector import box_iou, gt_boxes, split_ids
from ..utils import require

NEG_CLASS = "Not_acne"


def split_for(key, ratios=(0.6, 0.2, 0.2)):
    """Deterministic train/valid/test bucket for a crop id — stable across runs.

    Uses md5 (not the salted built-in hash) so a re-harvest reproduces the split.
    """
    frac = int(hashlib.md5(key.encode()).hexdigest(), 16) / 16 ** 32
    train, valid, _ = ratios
    return "train" if frac < train else "valid" if frac < train + valid else "test"


def offlesion_boxes(pred, gt):
    """Predicted boxes that miss every GT box (IoU 0) — off-lesion false positives.

    Empty gt keeps all boxes (an unannotated image is all off-lesion).
    """
    return [b for b in pred if all(box_iou(b, g) == 0 for g in gt)]


def harvest(model, img_paths, out, args, gt_lookup=None):
    """Run the detector over img_paths, crop kept boxes into <out>/<split>/Not_acne."""
    counts = {"train": 0, "valid": 0, "test": 0}
    total = 0
    for img_path in img_paths:
        if args.limit and total >= args.limit:
            break
        result = model.predict(str(img_path), conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        pred = [b.xyxy[0].tolist() for b in result.boxes]
        if gt_lookup is not None:
            pred = offlesion_boxes(pred, gt_lookup(img_path))
        if not pred:
            continue
        image = load_rgb(img_path)
        for i, (x0, y0, x1, y1) in enumerate(pred):
            if args.limit and total >= args.limit:
                break
            key = f"{img_path.stem}_{i}"
            split = split_for(img_path.stem)  # split by image -> all crops of one face share a split (no leakage)
            dst = out / split / NEG_CLASS
            dst.mkdir(parents=True, exist_ok=True)
            crop = crop_with_context(image, (x0, y0, x1 - x0, y1 - y0), pad=args.pad, size=args.size)
            Image.fromarray(crop).save(dst / f"{key}.jpg", quality=92)
            counts[split] += 1
            total += 1
    return total, counts


def parse_args():
    cfg = load_config()
    d = cfg["detection"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["ffhq", "acne04"], required=True)
    p.add_argument("--weights", type=Path, default=Path(d["weights"]))
    p.add_argument("--images", type=Path, help="ffhq: face dir; acne04: JPEGImages dir")
    p.add_argument("--annotations", type=Path, default=Path("data/raw/acne04/Detection/VOC2007/Annotations"))
    p.add_argument("--split", type=Path, default=Path("data/raw/acne04/Detection/VOC2007/ImageSets/Main/NNEW_trainval_0.txt"))
    p.add_argument("--out", type=Path, default=Path(cfg["classification"]["local_data"]),
                   help="dataset root; crops land in <out>/{train,valid,test}/Not_acne")
    p.add_argument("--conf", type=float, default=d["conf_threshold"])
    p.add_argument("--iou", type=float, default=d["iou_threshold"])
    p.add_argument("--imgsz", type=int, default=d["img_size"])
    p.add_argument("--pad", type=float, default=cfg["classification"]["crop_pad"])
    p.add_argument("--size", type=int, default=cfg["classification"]["crop_size"])
    p.add_argument("--limit", type=int, default=0, help="max crops (0 = all); size from --inspect")
    return p.parse_args()


def main():
    args = parse_args()
    require(args.weights, "YOLO detector weights")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    if args.mode == "ffhq":
        if args.images is None:
            raise SystemExit("--mode ffhq requires --images <FFHQ face dir>")
        require(args.images, "FFHQ image dir")
        total, counts = harvest(model, image_files(args.images), args.out, args)
    else:
        images = args.images or Path("data/raw/acne04/Classification/JPEGImages")
        require(images, "ACNE04 images")
        require(args.annotations, "ACNE04 annotations")
        require(args.split, "ACNE04 split")
        img_paths = [images / f"{s}.jpg" for s in split_ids(args.split)]
        img_paths = [p for p in img_paths if p.exists()]
        total, counts = harvest(model, img_paths, args.out, args,
                                gt_lookup=lambda p: gt_boxes(args.annotations, p.stem))

    print(f"harvested {total} {NEG_CLASS} crops -> {args.out}")
    for split in ("train", "valid", "test"):
        print(f"  {split:6s} {counts[split]}")


if __name__ == "__main__":
    main()
