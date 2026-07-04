"""Draw YOLO-format labels back onto their images.

Step 3 of the build sequence and the single most important sanity check: it
catches denormalization bugs, center-vs-corner mistakes, and x/y swaps that
metrics will NOT tell you about (a flipped-coordinate model can still post a
plausible-looking loss curve while boxing the wrong spots). Look before you
train. (Lesson-2 principle: pictures first, numbers second.)
"""
from __future__ import annotations
import os
from PIL import Image, ImageDraw


def yolo_to_corners(xc: float, yc: float, w: float, h: float,
                    img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Inverse of the converter: normalized center -> absolute corner pixels."""
    bw, bh = w * img_w, h * img_h
    cx, cy = xc * img_w, yc * img_h
    return int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)


def draw_labels(image_path: str, label_path: str, out_path: str,
                color: str = "#E24B4A", width: int = 2) -> int:
    """Draw every box in a YOLO label file onto its image. Returns box count."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    n = 0
    if os.path.exists(label_path):
        for line in open(label_path):
            line = line.strip()
            if not line:
                continue
            _cls, xc, yc, w, h = (float(v) for v in line.split())
            draw.rectangle(yolo_to_corners(xc, yc, w, h, W, H),
                           outline=color, width=width)
            n += 1
    img.save(out_path)
    return n


def contact_sheet(image_dir: str, label_dir: str, out_dir: str, n: int = 20) -> None:
    """Render the first n images with their boxes so you can flip through them.
    Expect to SEE the known ACNE04 issues here: loose boxes, a few junk/far
    images. Finding them now is the point."""
    os.makedirs(out_dir, exist_ok=True)
    imgs = sorted(f for f in os.listdir(image_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))[:n]
    for f in imgs:
        stem = os.path.splitext(f)[0]
        cnt = draw_labels(os.path.join(image_dir, f),
                          os.path.join(label_dir, stem + ".txt"),
                          os.path.join(out_dir, f))
        print(f"{f}: {cnt} boxes")
