# Notebook 02 — lesion-crop type classifier (Stage 2, spec §4.2)

Cell-by-cell Colab walkthrough. Runtime → **GPU (T4)**. The inference wrapper,
crop logic, and class list live in `src/classification/classifier.py` (tested,
runs locally); the cells below need the GPU, the Stage 1 weights, and your
labeling time — so they run in Colab.

The real deliverable of this notebook is not the weights, it's the
**self-labeled crop set**: a few hundred detector crops hand-sorted into 5
classes. That set is the only train data matched to the crop domain the model
sees at inference (spec §4.2). Budget most of your time on cell 3, not cell 8.

Build order:
1. environment + code + Drive
2. harvest crops: Stage 1 detector over ACNE04 → unsorted crops
3. **hand-sort into 5 class folders** (the actual work)
4. negatives: on-face random patches + FFHQ clear skin (D-013)
5. count classes → thin-class fallback decision
6. (optional) Kaggle pretrain — verify the dataset before trusting it
7. datasets + weighted loss
8. fine-tune MobileNetV3-small
9. look at predictions before metrics
10. confusion matrix + per-class precision/recall (+ Fitzpatrick note)
11. export weights + round-trip check

---

### Cell 1 — environment

Crops must survive across sessions (labeling takes days) → everything lands in
Drive, not `/content`.

```python
from google.colab import drive
drive.mount("/content/drive")

import os, sys
REPO = "/content/skinscan"
if os.path.isdir(REPO):
    !git -C {REPO} pull -q
else:
    !git clone -q https://github.com/Kumario1/skinscan.git {REPO}
sys.path.insert(0, REPO)
import importlib; importlib.invalidate_caches()   # in case an earlier bad clone poisoned the finder cache
from src.classification.classifier import CLASSES, crop_with_context, build_net
print(CLASSES)   # ['comedonal', 'cystic', 'inflammatory', 'not_acne', 'post_acne_mark']

from pathlib import Path
WORK = Path("/content/drive/MyDrive/skinscan_stage2")
UNSORTED = WORK / "crops_unsorted"
LABELED  = WORK / "crops_labeled"        # <- ImageFolder root, one dir per class
for c in CLASSES: (LABELED / c).mkdir(parents=True, exist_ok=True)
UNSORTED.mkdir(parents=True, exist_ok=True)
```

### Cell 2 — harvest crops from Stage 1 detections

Run the notebook-01 detector over ACNE04 images and save one padded crop per
box. Filename carries **source image stem** (needed for a leak-free split in
cell 7) and detector confidence (useful when sorting: low-conf boxes are where
`not_acne` and `post_acne_mark` hide).

Prefer harvesting the detector's **val** images first — boxes on images the
detector trained on are unrealistically clean. Top up from train images if
counts run thin.

Assumes `Detection.tar` / `Classification.tar` sit in your Drive root and the
stage-1 weights at `MyDrive/skinscan_best_y8m.pt` — where notebook 01 left
them. The cell restages ACNE04 itself, so it works in a fresh session.

```python
!pip install ultralytics -q
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# restage ACNE04 from the Drive tars (as in notebook 01)
!mkdir -p /content/acne04_raw
!cp "/content/drive/MyDrive/Detection.tar" "/content/drive/MyDrive/Classification.tar" /content/acne04_raw/
!cd /content/acne04_raw && tar -xf Detection.tar && tar -xf Classification.tar

SPLIT  = "/content/acne04_raw/Detection/VOC2007/ImageSets/Main/NNEW_test_0.txt"   # detector-val ids
IMGDIR = "/content/acne04_raw/Classification/JPEGImages"
val_ids = [os.path.splitext(ln.split()[0])[0] for ln in open(SPLIT) if ln.strip()]
imgs = [f"{IMGDIR}/{s}.jpg" for s in val_ids if os.path.exists(f"{IMGDIR}/{s}.jpg")]
print(len(imgs), "val images")

model = YOLO("/content/drive/MyDrive/skinscan_best_y8m.pt")   # stage 1 weights (Drive)

n = 0
for f in imgs[:120]:                       # a slice is plenty; widen if counts run thin
    im = np.asarray(Image.open(f).convert("RGB"))
    # conf/iou/imgsz = stage 1's LOCKED operating point — harvest what inference will see
    r = model.predict(f, conf=0.07, iou=0.2, imgsz=1024, verbose=False)[0]
    for k, b in enumerate(r.boxes[:15]):   # top-15 by conf per face keeps one face from flooding the set
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        crop = crop_with_context(im, (x0, y0, x1 - x0, y1 - y0))   # pad/size from config defaults
        Image.fromarray(crop).save(UNSORTED / f"{Path(f).stem}_{k}_{float(b.conf):.2f}.png")
        n += 1
print(n, "crops")   # aim for a few hundred+; widen the imgs slice if short
```

Already stocked? Cell 2 is skippable once `crops_unsorted/` has supply. To top
up the rare classes, harvest the remaining **severe** faces — cystic lives on
`levle2`/`levle3` images:

```python
severe = [f for f in imgs[120:] if "levle3" in f or "levle2" in f]
print(len(severe), "severe faces to harvest")
n = 0
for f in severe:
    im = np.asarray(Image.open(f).convert("RGB"))
    r = model.predict(f, conf=0.07, iou=0.2, imgsz=1024, verbose=False)[0]
    for k, b in enumerate(r.boxes[:15]):
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        crop = crop_with_context(im, (x0, y0, x1 - x0, y1 - y0))
        Image.fromarray(crop).save(UNSORTED / f"{Path(f).stem}_{k}_{float(b.conf):.2f}.png")
        n += 1
print(n, "new crops")
```

Comedonal starved? It lives on **mild** faces. The detector's train split
(~1165 faces) is fair game for classifier training — spec's "top up from
train" clause. Standalone cell (restages + loads the detector itself):

```python
!pip install ultralytics -q
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

!mkdir -p /content/acne04_raw
!cp "/content/drive/MyDrive/Detection.tar" "/content/drive/MyDrive/Classification.tar" /content/acne04_raw/
!cd /content/acne04_raw && tar -xf Detection.tar && tar -xf Classification.tar

IMGDIR  = "/content/acne04_raw/Classification/JPEGImages"
TRSPLIT = "/content/acne04_raw/Detection/VOC2007/ImageSets/Main/NNEW_trainval_0.txt"
tr_ids = [os.path.splitext(ln.split()[0])[0] for ln in open(TRSPLIT) if ln.strip()]
timgs = [f"{IMGDIR}/{s}.jpg" for s in tr_ids if os.path.exists(f"{IMGDIR}/{s}.jpg")]
mild   = [f for f in timgs if "levle0" in f or "levle1" in f][:80]   # comedonal supply
severe = [f for f in timgs if "levle3" in f][:40]                    # cystic supply
print(len(mild), "mild,", len(severe), "severe")

model = YOLO("/content/drive/MyDrive/skinscan_best_y8m.pt")
n = 0
for f in mild + severe:
    im = np.asarray(Image.open(f).convert("RGB"))
    r = model.predict(f, conf=0.07, iou=0.2, imgsz=1024, verbose=False)[0]
    for k, b in enumerate(r.boxes[:15]):
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        crop = crop_with_context(im, (x0, y0, x1 - x0, y1 - y0))
        Image.fromarray(crop).save(UNSORTED / f"{Path(f).stem}_{k}_{float(b.conf):.2f}.png")
        n += 1
print(n, "new crops")
```

Caveat: the detector trained on these faces, so its boxes here are cleaner
than at deployment — fine for classifier train data, and the face-level split
in cell 7 keeps eval honest.

### Cell 2d — one external image → crops (check, then stage)

Feed any image (yours, web, a specific acne type ACNE04 is thin on) through
OUR detector so the crops share the training domain. **Same move as the cell 6c
transfusion — which backfired for cystic (domain gap).** Safe only with the
guardrails: `ext_` prefix keeps them train-only (val stays ACNE04, cell 7), and
you must re-check the target class's val recall after — if it doesn't rise, the
external images didn't transfer; pull them like the kaggle cysts. Modest counts;
use images you have the right to use.

Preview first — did the detector actually find the acne?

```python
IMG = "/content/example.jpg"      # <- upload via Colab's file panel, or a Drive path

try: model
except NameError:
    from ultralytics import YOLO
    model = YOLO("/content/drive/MyDrive/skinscan_best_y8m.pt")

import numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

im = np.asarray(Image.open(IMG).convert("RGB"))
r = model.predict(IMG, conf=0.07, iou=0.2, imgsz=1024, verbose=False)[0]
boxes = [b.xyxy[0].tolist() for b in r.boxes]
crops = [crop_with_context(im, (x0, y0, x1 - x0, y1 - y0)) for (x0, y0, x1, y1) in boxes]
print(len(boxes), "detections — check the boxes before saving")

fig, ax = plt.subplots(figsize=(6, 6)); ax.imshow(im); ax.axis("off")
for (x0, y0, x1, y1) in boxes:
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=2))
plt.show()

fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for a in axes.flat: a.axis("off")
for a, c in zip(axes.flat, crops[:24]): a.imshow(c)
plt.show()
```

Boxes look right? Stage them (then sort in cell 3):

```python
from pathlib import Path
stem = Path(IMG).stem                       # give each source image a unique name
for k, c in enumerate(crops):
    Image.fromarray(c).save(UNSORTED / f"ext_{stem}_{k}_0.00.png")
print(len(crops), "crops staged as ext_ (train-only) -> sort in cell 3")
```

If the detector missed the acne or boxed the wrong things, don't save — a
detector that can't find comedones on this photo can't harvest them, and a
hand-drawn box is a different tool (not built here).

**A whole folder?** Same thing in a loop. The folder-level "is this good"
check is the **hit ratio** it prints — if few images produced any detections,
the detector doesn't transfer to this folder's style and the crops it did make
are probably junk. `ext_` keeps it all train-only.

```python
from pathlib import Path
import numpy as np
from PIL import Image

EXT_DIR = "/content/comedonal_folder"    # <- your folder of images
files = [p for p in sorted(Path(EXT_DIR).iterdir())
         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
print(len(files), "images")

try: model
except NameError:
    from ultralytics import YOLO
    model = YOLO("/content/drive/MyDrive/skinscan_best_y8m.pt")

n, hits, sample = 0, 0, []
for f in files:
    im = np.asarray(Image.open(f).convert("RGB"))
    r = model.predict(str(f), conf=0.07, iou=0.2, imgsz=1024, verbose=False)[0]
    if len(r.boxes): hits += 1
    for k, b in enumerate(r.boxes[:15]):
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        crop = crop_with_context(im, (x0, y0, x1 - x0, y1 - y0))
        Image.fromarray(crop).save(UNSORTED / f"ext_{f.stem}_{k}_{float(b.conf):.2f}.png")
        if k == 0 and len(sample) < 24: sample.append(crop)   # one per image, spread the preview
        n += 1
print(f"{hits}/{len(files)} images had detections -> {n} crops staged as ext_ (train-only)")
print("keep this comparable to your ACNE04 count for the class — dominating it is what broke cystic")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for a in axes.flat: a.axis("off")
for a, c in zip(axes.flat, sample): a.imshow(c)
plt.show()
```

Low hit ratio, or the sample grid is garbage? Pull it all back out — it's one
prefix: `import shutil; [shutil.move(str(p), str(WORK/"ext_holdout"/p.name)) for p in UNSORTED.glob("ext_*.png")]` (make `ext_holdout` first).

### Cell 3 — hand-sort (the real work)

One click per crop: the buttons move the file and show the next. Files move
as you label, so you can stop anytime and re-run the cell to resume with
whatever is still unsorted. Sorting guide:

| put in | when you see |
|--------|--------------|
| `comedonal` | blackhead / whitehead — small, no redness |
| `inflammatory` | papule / pustule — red bump, maybe a white head |
| `cystic` | large, deep, angry nodule |
| `post_acne_mark` | flat brown/red mark or pitted scar, no active lesion |
| `not_acne` | mole, shadow, hair, clear skin — detector was wrong |

```python
import shutil
import ipywidgets as W

crops = sorted(UNSORTED.glob("*.png"))
print(len(crops), "to sort")

img, label, state = W.Image(format="png", width=300), W.Label(), {"i": 0}

def show():
    if state["i"] >= len(crops):
        label.value = "done — all sorted"
        return
    p = crops[state["i"]]
    img.value = p.read_bytes()
    label.value = f"{state['i'] + 1}/{len(crops)}   {p.name}"

def mover(cls):
    def cb(_):
        if state["i"] < len(crops):
            p = crops[state["i"]]
            if cls != "skip":
                shutil.move(str(p), str(LABELED / cls / p.name))
            state["i"] += 1
            show()
    return cb

names = CLASSES + ["skip"]
buttons = [W.Button(description=n) for n in names]
for b, n in zip(buttons, names):
    b.on_click(mover(n))
display(label, img, W.HBox(buttons))
show()
```

Rules of the game:
- **Unsure → skip.** A smaller clean set beats a bigger
  noisy one; the detector's loose boxes (D-010) are noise enough.
- `post_acne_mark` examples are detector *false positives* that landed on a
  mark — hunt for them among low-confidence crops specifically (spec §4.2).
- Self-collected phone photos are **test-only (D-014)** — never sort them into
  these training folders. They get their own dir in cell 10.

### Cell 3b — relabel pass (audit an existing class for contamination)

When a confusion audit (cell 10) shows a folder is contaminated — e.g. flat
marks sorted as `cystic` — walk that folder and re-file. Point `REVIEW` at the
class to clean; buttons move each crop to the right class, keep it, or delete
junk. Same resume-on-rerun behaviour as the labeler.

```python
import shutil
import ipywidgets as W
from IPython.display import display

REVIEW = "cystic"   # <- folder to audit; change to review another class
src = sorted((LABELED / REVIEW).glob("*.png"))
print(len(src), f"crops in {REVIEW} — reclassify, keep, or delete")

img, label, state = W.Image(format="png", width=300), W.Label(), {"i": 0}

def show():
    if state["i"] >= len(src):
        label.value = "done"; return
    p = src[state["i"]]
    img.value = p.read_bytes()
    label.value = f"{state['i']+1}/{len(src)}   {p.name}"

def act(target):
    def cb(_):
        if state["i"] < len(src):
            p = src[state["i"]]
            if target == "delete":
                p.unlink()
            elif target != REVIEW:            # "keep" == REVIEW -> leave in place
                shutil.move(str(p), str(LABELED / target / p.name))
            state["i"] += 1; show()
    return cb

keys = [REVIEW] + [c for c in CLASSES if c != REVIEW] + ["delete"]
names = [f"keep ({REVIEW})"] + [c for c in CLASSES if c != REVIEW] + ["delete"]
buttons = [W.Button(description=n) for n in names]
for b, k in zip(buttons, keys):
    b.on_click(act(k))
display(label, img, W.HBox(buttons))
show()
```

### Cell 4 — cheap negatives for `not_acne` (D-013)

Random on-face patches from ACNE04 images at locations the detector did NOT
fire on, plus a handful of FFHQ clear-skin crops. Keep it modest — `not_acne`
must not dwarf the real classes.

```python
import random
import matplotlib.pyplot as plt
random.seed(0)   # same seed -> same filenames -> re-running overwrites, not duplicates
for f in random.sample(imgs, 40):
    im = np.asarray(Image.open(f).convert("RGB"))
    H, W = im.shape[:2]
    # central face-ish region, random small box
    x, y = random.randint(W//4, 3*W//4), random.randint(H//4, 3*H//4)
    crop = crop_with_context(im, (x, y, 30, 30))
    Image.fromarray(crop).save(LABELED / "not_acne" / f"neg_{Path(f).stem}_{x}_{y}.png")

negs = sorted((LABELED / "not_acne").glob("neg_*.png"))
print(len(negs), "negatives — delete lesion-hits by index, e.g. negs[3].unlink()")
fig, axes = plt.subplots(5, 8, figsize=(16, 10))
for ax in axes.flat: ax.axis("off")
for j, (ax, p) in enumerate(zip(axes.flat, negs)):
    ax.imshow(Image.open(p)); ax.set_title(str(j), fontsize=8)
```

A "negative" that landed on a lesion poisons the class — delete offenders
before moving on. Add FFHQ crops the same way if you have FFHQ downloaded.

### Cell 5 — counts → fallback decision

```python
for c in CLASSES:
    print(f"{c:16s} {len(list((LABELED/c).glob('*.png')))}")
```

Decision point (spec §4.2, resolved decision 3): if `post_acne_mark` is too
thin to learn (rule of thumb: under ~30 crops), move its crops into
`not_acne` and train 4 classes — accepting that we lose the whole
hyperpigmentation recommendation for v1, not just a note. `cystic` will be
small; that's what the weighted loss in cell 7 is for. Don't delete the class
for being merely small — only for being unlearnable.

Collapse mechanics: move the crops into `not_acne/` and **leave the empty
`post_acne_mark/` dir in place** — `ImageFolder` keeps the 5-class order, the
head stays 5-class, and the class simply never gets predicted (the intended
v1 behavior). No code or schema changes.

### Cell 6 — (optional) Kaggle pretrain — dataset verified 2026-07-05

The dataset behind the `zulqarnain11` notebook is `tiswan14/acne-dataset-image`:
2778 train images, 5 lesion-type classes — Blackheads 735, Cyst 645, Papules
621, Pustules 584, Whiteheads 193. The notebook's own model (from-scratch
Keras CNN) is **not** used — weaker than our ImageNet start and the wrong
framework. The dataset is the value, above all **Cyst 645** vs our ~15 crops.

Run cells 7–10 WITHOUT this first; that's your baseline. Come back when
per-class recall needs the help (v0 verdict: it does — cystic .11,
inflammatory .14). Still: **look before you train.**

```python
!pip install kagglehub -q
import random, kagglehub
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

kpath = Path(kagglehub.dataset_download("tiswan14/acne-dataset-image")) / "AcneDataset"
kfull = datasets.ImageFolder(kpath / "train")
print(kfull.classes, len(kfull), "images")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, i in zip(axes.flat, random.sample(range(len(kfull)), 10)):
    p, y = kfull.samples[i]
    ax.imshow(Image.open(p)); ax.set_title(kfull.classes[y], fontsize=8); ax.axis("off")
```

Are these lesion-scale close-ups (≈ our crop domain — great) or whole faces
(features still transfer, expect less)? Note what you see, then pretrain:

```python
import torch
from torchvision import transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # must match cell 7 / LesionClassifier
dev = "cuda" if torch.cuda.is_available() else "cpu"

KMAP = {"Blackheads": "comedonal", "Whiteheads": "comedonal",
        "Papules": "inflammatory", "Pustules": "inflammatory",
        "Cyst": "cystic"}
kclasses = datasets.ImageFolder(kpath / "train").classes
kidx = {i: CLASSES.index(KMAP[c]) for i, c in enumerate(kclasses)}

ktf = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
ktrain = datasets.ImageFolder(kpath / "train", ktf, target_transform=lambda y: kidx[y])
kdl = torch.utils.data.DataLoader(ktrain, batch_size=64, shuffle=True, num_workers=2)

pre = build_net(pretrained=True).to(dev)
kopt = torch.optim.AdamW(pre.parameters(), lr=3e-4)
klossf = torch.nn.CrossEntropyLoss()
for epoch in range(3):                     # features, not a final model — 3 epochs is the point
    pre.train(); tot = 0.0
    for x, y in kdl:
        kopt.zero_grad()
        loss = klossf(pre(x.to(dev)), y.to(dev))
        loss.backward(); kopt.step(); tot += loss.item()
    print(f"pretrain epoch {epoch}: loss {tot / len(kdl):.3f}")
torch.save(pre.state_dict(), WORK / "kaggle_pretrain.pt")
```

Then in cell 8, right after `build_net(...)`:
`net.load_state_dict(torch.load(WORK / "kaggle_pretrain.pt"))` — the fine-tune
on our crops re-trains everything; the pretrain just moves the starting point
from "ImageNet objects" to "acne lesions".

### Cell 6c — cystic transfusion (external images, in-domain crops)

Raw external photos must NOT go straight into the train folders — the model
learns the photo *style*, not the lesion (train/serve mismatch, spec §4.2).
Instead run them through OUR detector + cropper so they arrive in the same
geometry as every other crop. `kg_` prefix → cell 7 forces them into TRAIN
only; val stays pure ACNE04. Sort the output with the labeler — third-party
labels, trust but verify. Needs `kpath` (cell 6a) and the stage-1 `model`
(cell 2 — or load it here).

```python
cyst_dir = kpath / "train" / "Cyst"
files = sorted(cyst_dir.glob("*"))[:300]
n = 0
for f in files:
    im = np.asarray(Image.open(f).convert("RGB"))
    r = model.predict(str(f), conf=0.07, iou=0.2, imgsz=640, verbose=False)[0]
    if len(r.boxes):
        for k, b in enumerate(r.boxes[:3]):
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            crop = crop_with_context(im, (x0, y0, x1 - x0, y1 - y0))
            Image.fromarray(crop).save(UNSORTED / f"kg_{f.stem}_{k}_{float(b.conf):.2f}.png")
            n += 1
    else:                       # detector found nothing -> center square, same output geometry
        H, W = im.shape[:2]
        s = min(H, W) // 2
        crop = crop_with_context(im, ((W - s) / 2, (H - s) / 2, s, s))
        Image.fromarray(crop).save(UNSORTED / f"kg_{f.stem}_0_0.00.png")
        n += 1
print(n, "kaggle cystic candidates -> confirm via the labeler (most are one 'cystic' click)")
```

### Cell 7 — datasets, leak-free split, weighted loss

Split by **source image**, not by crop — two crops from the same face must
not straddle train/val, or val is contaminated. Crop filenames carry the stem
(cell 2); negatives carry theirs too.

```python
import torch, hashlib
from torchvision import datasets, transforms
from collections import Counter

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # matches LesionClassifier
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),  # NO hue jitter:
    # red-vs-brown is the inflammatory-vs-post_acne_mark signal — don't augment it away
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

full = datasets.ImageFolder(LABELED)
assert full.classes == CLASSES, f"class order drift: {full.classes}"   # load-bearing

def src_stem(path):        # "<imgstem>_<k>_<conf>.png" / "neg_<imgstem>_<x>_<y>.png" -> imgstem
    s = Path(path).stem
    if s.startswith("neg_"):
        s = s[4:]          # negatives group with their source face, not apart from it
    return "_".join(s.split("_")[:-2])
def is_val(path):          # deterministic ~20% of source images
    if Path(path).stem.startswith(("kg_", "ext_")):
        return False       # external crops (cell 2d / 6c) train only — val stays real ACNE04
    return int(hashlib.md5(src_stem(path).encode()).hexdigest(), 16) % 5 == 0

val_idx   = [i for i, (p, _) in enumerate(full.samples) if is_val(p)]
vs        = set(val_idx)
train_idx = [i for i in range(len(full)) if i not in vs]

train_ds = torch.utils.data.Subset(datasets.ImageFolder(LABELED, train_tf), train_idx)
val_ds   = torch.utils.data.Subset(datasets.ImageFolder(LABELED, val_tf),   val_idx)

counts = Counter(full.targets[i] for i in train_idx)
print({CLASSES[k]: v for k, v in sorted(counts.items())}, "| val:", len(val_idx))

# oversample rare classes (augmentation makes the repeats non-identical);
# CE stays unweighted in cell 8 — one imbalance correction, not two
sw = [1.0 / max(counts[full.targets[i]], 1) for i in train_idx]
sampler = torch.utils.data.WeightedRandomSampler(sw, num_samples=len(train_idx), replacement=True)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=2)
val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=64, shuffle=False,   num_workers=2)
```

### Cell 8 — fine-tune

Same move as notebook 01 cell 7: pretrained backbone, low LR, shift the
features gently. Model selection on **macro recall**, not accuracy — accuracy
is a liar under class imbalance.

```python
from sklearn.metrics import recall_score
dev = "cuda" if torch.cuda.is_available() else "cpu"   # CPU is fine at this dataset size (~15 min)
net = build_net(pretrained=True).to(dev)
kpt = WORK / "kaggle_pretrain.pt"
if kpt.exists():                              # cell 6 ran -> start from lesion features
    net.load_state_dict(torch.load(kpt, map_location=dev))
print("init:", "kaggle pretrain" if kpt.exists() else "imagenet")
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
lossf = torch.nn.CrossEntropyLoss()           # sampler in cell 7 handles imbalance

best = 0.0
for epoch in range(40):
    net.train()
    for x, y in train_dl:
        opt.zero_grad()
        loss = lossf(net(x.to(dev)), y.to(dev))
        loss.backward(); opt.step()
    sched.step()
    net.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in val_dl:
            P += net(x.to(dev)).argmax(1).cpu().tolist(); Y += y.tolist()
    mr = recall_score(Y, P, average="macro", zero_division=0)
    if mr > best:
        best = mr; torch.save(net.state_dict(), WORK / "type_classifier_v0.pt")
    print(f"epoch {epoch:2d}  val macro-recall {mr:.3f}  {'*' if mr == best else ''}")
```

### Cell 9 — predictions BEFORE metrics

```python
import random
import matplotlib.pyplot as plt
net.load_state_dict(torch.load(WORK / "type_classifier_v0.pt")); net.eval()
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for ax in axes.flat: ax.axis("off")          # blanks stay blank if val < 32
sample = random.sample(val_idx, min(32, len(val_idx)))
with torch.no_grad():
    for ax, i in zip(axes.flat, sample):
        p, y = full.samples[i]
        x = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
        pred = net(x).argmax(1).item()
        ax.imshow(Image.open(p)); ax.axis("off")
        ax.set_title(f"{CLASSES[pred][:9]}\n({CLASSES[y][:9]})",
                     color="green" if pred == y else "red", fontsize=8)
```

Read the reds before the numbers: is it confusing `inflammatory` with
`cystic` (a severity error — annoying) or `post_acne_mark` with
`inflammatory` (a *treatment* error — the recommender prescribes the wrong
actives)? Not all confusions cost the same.

### Cell 10 — metrics

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
# score the BEST checkpoint, not whatever epoch cell 8 ended on
net.load_state_dict(torch.load(WORK / "type_classifier_v0.pt", map_location=dev)); net.eval()
P, Y = [], []
with torch.no_grad():
    for x, y in val_dl:
        P += net(x.to(dev)).argmax(1).cpu().tolist(); Y += y.tolist()
print(classification_report(Y, P, target_names=CLASSES, zero_division=0))
ConfusionMatrixDisplay.from_predictions(Y, P, display_labels=CLASSES,
                                        xticks_rotation=45)
```

When one confusion pair dominates the matrix, look at those exact errors
before changing anything — label drift and information ceiling need different
responses (relabel pass vs accept-and-document):

```python
import matplotlib.pyplot as plt
A, B = CLASSES.index("inflammatory"), CLASSES.index("post_acne_mark")   # the pair under audit
errs = [i for i, (yy, pp) in enumerate(zip(Y, P)) if {yy, pp} == {A, B}]
print(len(errs), "confusions between", CLASSES[A], "and", CLASSES[B])
fig, axes = plt.subplots(4, 8, figsize=(16, 9))
for ax in axes.flat: ax.axis("off")
for ax, e in zip(axes.flat, errs[:32]):
    p, yy = full.samples[val_idx[e]]
    ax.imshow(Image.open(p))
    ax.set_title(f"true {CLASSES[yy][:9]}\npred {CLASSES[P[e]][:9]}", fontsize=7)
```

If you can call most of them at a glance, the labels drifted — do a relabel
pass on the offending folder. If half are genuinely ambiguous, that's the
information ceiling (spec §3) — document it and stop paying for it.

Watch specifically (spec §6): `cystic` recall (rare — a missed cyst is a
missed see-a-professional) and `post_acne_mark` recall (thinnest class). If
you have Fitzpatrick tone labels for the source images, disaggregate the
report by tone (D-016); if the set is too small to slice meaningfully, write
that down honestly — that's a finding, not a gap to hide.

**Phone-photo test set (D-014):** when you have self-collected crops, they
live in a separate `crops_phone_test/` dir and only ever run through this
cell's report — never cells 7–8. The gap between ACNE04-val numbers and
phone-photo numbers IS the domain-gap measurement.

### Cell 11 — export + round-trip check

```python
from src.classification.classifier import LesionClassifier
clf = LesionClassifier(WORK / "type_classifier_v0.pt", device="cpu")
p, y = full.samples[val_idx[0]]
probs = clf.predict(np.asarray(Image.open(p).convert("RGB")))
print(max(probs, key=probs.get), "| true:", CLASSES[y], "|", probs)
```

If that runs, the weights and the inference wrapper agree. Copy
`type_classifier_v0.pt` into the repo's `models/` dir — the config already
points at `models/type_classifier_v0.pt`.
