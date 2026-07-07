# Notebook 01 — ACNE04 → YOLO detector (Stage 1)

Cell-by-cell Colab walkthrough. Runtime → change runtime type → **GPU (T4)**.
The two `src/detection/*.py` files are tested; the cells below are the parts
that need a GPU and the real data, so they run in Colab, not locally.

Build order (each cell is one step of the sequence):
1. environment + get the code
2. get ACNE04
3. **inspect the raw format before parsing** (don't skip this)
4. convert → YOLO labels
5. eyeball 20 images with boxes (look before you train)
6. write data.yaml + split
7. fine-tune yolov8n
8. look at predictions before metrics

---

### Cell 1 — environment

```python
!pip install ultralytics gdown -q
import os, glob, random
from pathlib import Path
# pull our tested converter/visualizer into the Colab session
# (upload the skinscan/ folder to the session, or git clone your repo)
import sys; sys.path.insert(0, "/content/skinscan")
```

### Cell 2 — get ACNE04

The dataset lives on Google Drive via the official repo `xpwu95/LDL`
(also Baidu, pw `fbrm`). You need the Google Drive file id from that repo's
README. It unpacks into `Classification.tar` (severity + counts) and
`Detection.tar` (boxes).

```python
# replace FILE_ID with the id from the xpwu95/LDL README's Google link
!gdown --id FILE_ID -O /content/acne04.zip
!unzip -q /content/acne04.zip -d /content/acne04_raw
# the tars, if present:
!cd /content/acne04_raw && (tar -xf Detection.tar 2>/dev/null; tar -xf Classification.tar 2>/dev/null; true)
```

> Fallback if the Drive download is painful: a Roboflow copy of acne04 ships
> pre-converted to YOLO format. Faster, but it's single-class only (no severity
> /count metadata) and the labels are a third party's — trust but verify in
> cell 5. Prefer the official set so we keep the count labels for deriving
> severity later (DECISIONS.md).

### Cell 3 — INSPECT before you parse

```python
from src.detection.voc_to_yolo import inspect_raw
inspect_raw("/content/acne04_raw")
```

Look at the tree and open the first annotation file **by hand**. Decide which
parser you need:
- annotation files end in `.xml` → VOC path (cell 4a), `parse_voc_xml`
- one big `.txt` with `img.jpg x1,y1,x2,y2 ...` lines → flat path,
  `parse_flat_line` (adjust the delimiter to match what you actually see)

This is the same move as reading a HAR payload before writing the scraper: the
format claim in a README is a hint, the bytes on disk are the truth.

### Cell 4a — convert (VOC-XML case)

```python
from src.detection.voc_to_yolo import convert_voc_dir
report = convert_voc_dir(
    xml_dir="/content/acne04_raw/Detection/VOC2007/Annotations",  # adjust path
    out_label_dir="/content/acne04_yolo/labels",
    class_id=0,   # single class 'lesion' (DECISIONS.md D-004)
)
print(report)   # sanity: images≈1457, boxes in the thousands, few skips
```

If `empty_images` is high or `boxes` is implausibly low, the parser/path is
wrong — go back to cell 3. Don't proceed on a bad report.

### Cell 5 — EYEBALL (the important one)

```python
# copy/point images next to labels, then render 20 with boxes drawn
from src.detection.visualize_labels import contact_sheet
contact_sheet(
    image_dir="/content/acne04_raw/Detection/VOC2007/JPEGImages",  # adjust
    label_dir="/content/acne04_yolo/labels",
    out_dir="/content/checked",
    n=20,
)
# then display them
from IPython.display import Image as IPyImage, display
for f in sorted(glob.glob("/content/checked/*.jpg"))[:20]:
    display(IPyImage(f, width=300))
```

**What to look for (expect to find problems — that's success, not failure):**
- boxes sitting ON lesions, not shifted/flipped → confirms geometry
- loose boxes enclosing normal skin → the known "boxes are loose" issue;
  a research group literally shrank these. Note it; it caps achievable mAP.
- a few far-away / low-quality images → the known contamination (~1513 raw vs
  1457 usable). Candidates to filter later.

Write down what you see. This is the residual-plot equivalent for detection:
the aggregate label file looks fine; only looking at individual images reveals
how the labels actually behave.

### Cell 6 — data.yaml + train/val split

```python
import shutil, yaml
root = Path("/content/acne04_yolo")
# expected layout: images/{train,val}/*.jpg, labels/{train,val}/*.txt
# (do the 80/20 split here — move files or write train.txt/val.txt)

data_yaml = {
    "path": str(root),
    "train": "images/train",
    "val":   "images/val",
    "names": {0: "lesion"},
}
yaml.safe_dump(data_yaml, open(root/"data.yaml","w"))
print(open(root/"data.yaml").read())
```

### Cell 7 — fine-tune yolov8n

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")           # COCO-pretrained (the transfer step)
results = model.train(
    data=str(root/"data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.001,        # low LR: shift pretrained features gently, don't torch them
    name="acne_y8n_v0",
    patience=15,      # early stop if val stalls
)
```

Watch `box_loss` and `cls_loss` fall. `cls_loss` is nearly trivial here (one
class), so `box_loss` and mAP are the real signals.

### Cell 8 — predictions BEFORE metrics

```python
best = YOLO("runs/detect/acne_y8n_v0/weights/best.pt")
val_imgs = glob.glob(str(root/"images/val/*.jpg"))
for f in random.sample(val_imgs, 8):
    r = best.predict(f, conf=0.25, imgsz=640, verbose=False)[0]
    display(IPyImage(r.plot(), width=320))   # r.plot() draws predicted boxes
```

Look first: is it finding lesions? Missing the small ones? Firing on clear
skin? Doubling up (NMS)? *Then* read the numbers:

```python
metrics = best.val()
print("mAP@0.5     :", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)
print("precision   :", metrics.box.mp)
print("recall      :", metrics.box.mr)
```

Set expectations from the literature: single-class acne detection lands at
modest mAP (small dense objects + loose labels). A low number here is the
*terrain*, not a bug. The learning is in reading WHERE it fails — which
lesion sizes, which face regions, which skin tones (that last one is Notebook
02's disaggregated eval, DECISIONS.md D-016).
```
