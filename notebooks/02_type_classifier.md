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

Fresh session? Re-stage the ACNE04 images first (notebook 01 cells 2–4), or
point `imgs` at a Drive copy of them.

```python
!pip install ultralytics -q
from ultralytics import YOLO
import numpy as np
from PIL import Image
import glob

model = YOLO("/content/drive/MyDrive/skinscan/acne_y8n_v0_best.pt")  # adjust path
imgs = glob.glob("/content/acne04_yolo/images/val/*.jpg")            # adjust path

n = 0
for f in imgs:
    im = np.asarray(Image.open(f).convert("RGB"))
    r = model.predict(f, conf=0.25, imgsz=640, verbose=False)[0]
    for k, b in enumerate(r.boxes):
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        crop = crop_with_context(im, (x0, y0, x1 - x0, y1 - y0))   # pad/size from config defaults
        Image.fromarray(crop).save(UNSORTED / f"{Path(f).stem}_{k}_{float(b.conf):.2f}.png")
        n += 1
print(n, "crops")   # aim for a few hundred+; harvest more images if short
```

### Cell 3 — hand-sort (the real work)

No code. Open `crops_unsorted/` in Drive and drag each crop into the right
`crops_labeled/<class>/` folder. Sorting guide:

| put in | when you see |
|--------|--------------|
| `comedonal` | blackhead / whitehead — small, no redness |
| `inflammatory` | papule / pustule — red bump, maybe a white head |
| `cystic` | large, deep, angry nodule |
| `post_acne_mark` | flat brown/red mark or pitted scar, no active lesion |
| `not_acne` | mole, shadow, hair, clear skin — detector was wrong |

Rules of the game:
- **Unsure → skip it** (leave in unsorted). A smaller clean set beats a bigger
  noisy one; the detector's loose boxes (D-010) are noise enough.
- `post_acne_mark` examples are detector *false positives* that landed on a
  mark — hunt for them among low-confidence crops specifically (spec §4.2).
- Self-collected phone photos are **test-only (D-014)** — never sort them into
  these training folders. They get their own dir in cell 10.

### Cell 4 — cheap negatives for `not_acne` (D-013)

Random on-face patches from ACNE04 images at locations the detector did NOT
fire on, plus a handful of FFHQ clear-skin crops. Keep it modest — `not_acne`
must not dwarf the real classes.

```python
import random
random.seed(0)
for f in random.sample(imgs, 40):
    im = np.asarray(Image.open(f).convert("RGB"))
    H, W = im.shape[:2]
    # central face-ish region, random small box
    x, y = random.randint(W//4, 3*W//4), random.randint(H//4, 3*H//4)
    crop = crop_with_context(im, (x, y, 30, 30))
    Image.fromarray(crop).save(LABELED / "not_acne" / f"neg_{Path(f).stem}_{x}_{y}.png")
```

Eyeball these after — a "negative" that landed on a lesion poisons the class.
Delete offenders. Add FFHQ crops the same way if you have FFHQ downloaded.

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

### Cell 6 — (optional) Kaggle pretrain

⚠ The candidate dataset (`zulqarnain11/acne-classification-using-cnn`) is
**unverified** — class list and quality unknown (spec §4.2). Run cells 7–10
WITHOUT this first; that's your baseline. Come back only if per-class recall
needs the help.

If you do: download it, **inspect it like cell 3 of notebook 01** (open
files, read the folder names), map its classes onto ours (or fewer), train
`build_net(pretrained=True)` on it for a few epochs at whole-image scale, and
use those weights as the starting point in cell 8 instead of plain ImageNet.
Its value is features, not plug-and-play weights — the head gets retrained on
crops regardless.

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

def src_stem(path):        # "<imgstem>_<k>_<conf>.png" -> imgstem
    return "_".join(Path(path).stem.split("_")[:-2])
def is_val(path):          # deterministic ~20% of source images
    return int(hashlib.md5(src_stem(path).encode()).hexdigest(), 16) % 5 == 0

val_idx   = [i for i, (p, _) in enumerate(full.samples) if is_val(p)]
vs        = set(val_idx)
train_idx = [i for i in range(len(full)) if i not in vs]

train_ds = torch.utils.data.Subset(datasets.ImageFolder(LABELED, train_tf), train_idx)
val_ds   = torch.utils.data.Subset(datasets.ImageFolder(LABELED, val_tf),   val_idx)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)

counts = Counter(full.targets[i] for i in train_idx)
print({CLASSES[k]: v for k, v in sorted(counts.items())}, "| val:", len(val_idx))
weight = torch.tensor([len(train_idx) / (len(CLASSES) * max(counts[i], 1))
                       for i in range(len(CLASSES))])   # max(...,1): survives an empty class
```

### Cell 8 — fine-tune

Same move as notebook 01 cell 7: pretrained backbone, low LR, shift the
features gently. Model selection on **macro recall**, not accuracy — accuracy
is a liar under class imbalance.

```python
from sklearn.metrics import recall_score
dev = "cuda"
net = build_net(pretrained=True).to(dev)      # or load cell-6 pretrain weights here
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
lossf = torch.nn.CrossEntropyLoss(weight=weight.to(dev))

best = 0.0
for epoch in range(20):
    net.train()
    for x, y in train_dl:
        opt.zero_grad()
        loss = lossf(net(x.to(dev)), y.to(dev))
        loss.backward(); opt.step()
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
print(classification_report(Y, P, target_names=CLASSES, zero_division=0))
ConfusionMatrixDisplay.from_predictions(Y, P, display_labels=CLASSES,
                                        xticks_rotation=45)
```

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
