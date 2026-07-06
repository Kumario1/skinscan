# Stage 2 Analysis Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage 2 — `analyze(image, boxes) -> ConcernReport` (region assignment via MediaPipe, 5-class lesion classifier interface, count→severity derivation) plus the recommender touch-ups and the Colab training notebook, per `docs/superpowers/specs/2026-07-05-stage2-analysis-design.md`.

**Architecture:** Three modules in `src/classification/` — `regions.py` (pure geometry + a thin MediaPipe wrapper), `classifier.py` (crop extraction + classifier interface with a weights-free stub), `assemble.py` (orchestration into the locked `ConcernReport` contract). Everything except real inference runs with no GPU and no weights (D-007 discipline). Training happens in Colab via `notebooks/02_type_classifier.md`.

**Tech Stack:** Python 3.9 venv at `.venv/`, numpy, Pillow, matplotlib (already deps), pyyaml, mediapipe (new), pytest (new), torch+torchvision (prod inference path only; lazy-imported so tests never need them).

## Global Constraints

- The `ConcernReport` contract is **LOCKED** (D-008): do NOT change any field of `src/recommendation/schema.py`. Closed concern vocabulary: `acne_comedonal, acne_inflammatory, acne_cystic, hyperpigmentation, dryness`. Closed region vocabulary: `forehead, nose, left_cheek, right_cheek, chin_jaw, perioral`.
- Classifier classes are exactly `["comedonal", "cystic", "inflammatory", "not_acne", "post_acne_mark"]` — **alphabetical**, because that is torchvision `ImageFolder`'s index order at training time; using it everywhere makes a train/serve index swap impossible (spec §3 defines the 5 classes; the order is ours to lock).
- Stage 1 boxes are `(x, y, w, h, det_confidence)` in **pixel** coords with `x, y` = **TOP-LEFT** corner. Images are RGB uint8 numpy arrays of shape `(H, W, 3)` everywhere.
- Config values live in `configs/default.yaml`, never hard-coded: `crop_pad: 1.5`, `crop_size: 112`, severity thresholds default `[1, 3, 6, 10]` (RULES.md §5 discipline).
- Self-collected phone photos are **TEST-ONLY**, never train (D-014). All user-facing wording is cosmetic, never diagnostic (D-002) — "appearance-based", "concerns", not "conditions".
- Run everything from repo root `/Users/princekumar/Documents/skinscan` using the venv: `.venv/bin/python`, `.venv/bin/pytest`.
- The working tree already contains unrelated user changes (deleted `notebooks/01_acne04_detector.md`, untracked `notebooks/acne_model (1).ipynb`). Commit **only the explicit paths** listed in each commit step. Never `git add -A` / `git add .`.
- `# ponytail:` comments mark deliberate, ceiling-known simplifications. Keep them.

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `conftest.py` | create | empty; puts repo root on sys.path for pytest |
| `src/classification/__init__.py` | create | package marker; re-exports `analyze` (added in Task 5) |
| `src/classification/regions.py` | create | landmarks → region + off-face filter (pure geometry) + MediaPipe wrapper + overlay demo |
| `src/classification/classifier.py` | create | crop extraction; `StubClassifier` / `TorchClassifier` behind one interface |
| `src/classification/assemble.py` | create | `analyze(image, boxes) -> ConcernReport`; `derive_severity` lives here |
| `src/recommendation/engine.py` | modify | no-face guard; texture advisory flag |
| `configs/default.yaml` | modify | `classification:` section (crop + severity thresholds + weights path) |
| `requirements.txt` | modify | add mediapipe, pytest; uncomment torch/torchvision |
| `docs/RULES.md` | modify | texture-advisory rule line (§3) |
| `docs/DECISIONS.md` | modify | log D-019 (region method), D-020 (taxonomy) |
| `notebooks/02_type_classifier.md` | create | Colab training curriculum (mirrors notebook 01) |
| `tests/test_regions.py` | create | synthetic-anchor region tests + no-face sentinel |
| `tests/test_classifier.py` | create | crop geometry + stub behaviour |
| `tests/test_engine.py` | create | no-face guard + texture advisory |
| `tests/test_assemble.py` | create | severity boundaries + end-to-end contract test |

---

### Task 1: Environment + regions.py pure geometry

The pure part of region assignment: an `Anchors` bag of landmark-derived
reference lines and `assign_region(cx, cy, anchors)` implementing the spec §4.1
rules. No MediaPipe anywhere in this task — synthetic anchors in tests.

**Files:**
- Create: `.venv/` (not committed; gitignored), `conftest.py`, `src/classification/__init__.py`, `src/classification/regions.py`, `tests/test_regions.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing (first task).
- Produces (Task 2 and tests rely on these exact names):
  - `Anchors` dataclass with fields `brow_y: float, eye_inner_left_x: float, eye_inner_right_x: float, nose_x: float, nose_bottom_y: float, mouth_cx: float, mouth_cy: float, mouth_half_width: float, lower_lip_y: float, oval_xy: list` (ordered `(x, y)` ring); `__post_init__` builds `self.oval_path` (matplotlib Path).
  - `assign_region(cx: float, cy: float, a: Anchors) -> str | None` — region name from the closed vocabulary, or `None` = off-face.
  - `PERIORAL_RADIUS_FACTOR = 1.8` module constant.

- [ ] **Step 1: Create venv and install dependencies**

```bash
cd /Users/princekumar/Documents/skinscan
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy pandas pillow matplotlib scikit-learn pyyaml pytest mediapipe
```

Expected: exits 0. Verify: `.venv/bin/python -c "import mediapipe, matplotlib, PIL, yaml; print('deps ok')"` prints `deps ok`.
(torch/torchvision are NOT installed locally — the torch path is lazy-imported and only needed once real weights exist. Install later with `.venv/bin/pip install torch torchvision` if running real inference locally.)

- [ ] **Step 2: Update requirements.txt**

Replace the full contents of `requirements.txt` with:

```text
# pinned loosely for a learning project; tighten if repro matters
numpy
pandas
pillow
matplotlib
scikit-learn
pyyaml
pytest             # tests (repo has runnable checks per non-trivial unit)
mediapipe          # stage 2 region assignment (face mesh, CPU, no weights to manage)
# classifier inference (TorchClassifier) — training itself happens in Colab:
torch
torchvision
# detection (Stage 1, trains in Colab):
# ultralytics        # YOLO
```

- [ ] **Step 3: Create package skeleton**

`conftest.py` (repo root):

```python
# empty on purpose: a root conftest makes pytest put the repo root on
# sys.path, so tests can `from src.... import ...` without packaging.
```

`src/classification/__init__.py`:

```python
```

(empty file for now; Task 5 adds the `analyze` re-export.)

- [ ] **Step 4: Write the failing test**

Create `tests/test_regions.py`:

```python
"""Region assignment: pure geometry against hand-built synthetic anchors.

The face is a 200x300 octagon. Reference lines (see diagram):
brows at y=80, nose band x in [80,120] down to y=180, mouth center (100,220)
half-width 25 (perioral radius = 1.8*25 = 45), lower lip at y=235.
"""
import numpy as np
from src.classification.regions import Anchors, assign_region


def make_anchors():
    return Anchors(
        brow_y=80.0,
        eye_inner_left_x=80.0, eye_inner_right_x=120.0,
        nose_x=100.0, nose_bottom_y=180.0,
        mouth_cx=100.0, mouth_cy=220.0, mouth_half_width=25.0,
        lower_lip_y=235.0,
        oval_xy=[(20, 0), (180, 0), (200, 40), (200, 260),
                 (160, 300), (40, 300), (0, 260), (0, 40)],
    )


def test_forehead_above_brows():
    assert assign_region(100, 40, make_anchors()) == "forehead"


def test_nose_band_between_eyes():
    assert assign_region(100, 150, make_anchors()) == "nose"


def test_perioral_within_mouth_radius():
    # dist to mouth center = 5 <= 45
    assert assign_region(100, 225, make_anchors()) == "perioral"


def test_chin_jaw_below_lower_lip_outside_perioral():
    # dist to mouth center = 60 > 45, y=280 > lower_lip_y
    assert assign_region(100, 280, make_anchors()) == "chin_jaw"


def test_cheeks_split_by_midline():
    a = make_anchors()
    # (40,150): not forehead, outside nose band, dist to mouth ~92 > 45, above lip
    assert assign_region(40, 150, a) == "left_cheek"
    assert assign_region(160, 150, a) == "right_cheek"


def test_off_face_is_none():
    assert assign_region(300, 150, make_anchors()) is None


def test_nose_bottom_boundary_inclusive():
    assert assign_region(100, 180, make_anchors()) == "nose"


def test_regions_are_schema_vocabulary():
    from src.recommendation.schema import REGIONS
    a = make_anchors()
    pts = [(100, 40), (100, 150), (100, 225), (100, 280), (40, 150), (160, 150)]
    assert {assign_region(x, y, a) for x, y in pts} == REGIONS
```

- [ ] **Step 5: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_regions.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.classification.regions'`

- [ ] **Step 6: Write the implementation**

Create `src/classification/regions.py`:

```python
"""Landmarks -> face region + off-face filter (Stage 2, spec §4.1).

Pure geometry: explicit landmark-anchored rules, auditable, no ML. The face
oval doubles as the off-face false-positive filter (stubble, hairline, neck
moles dropped for free). MediaPipe only appears in the thin wrapper added in
`assign()` — everything here tests with synthetic anchors, no model, no GPU.
Region vocabulary is the locked closed set in CONCERN_SCHEMA.md.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from matplotlib.path import Path as MplPath

# how far from the mouth center counts as perioral, in mouth-half-widths
PERIORAL_RADIUS_FACTOR = 1.8


@dataclass
class Anchors:
    """Pixel-space reference lines derived from face landmarks."""
    brow_y: float               # forehead line (mean of the two brow tops)
    eye_inner_left_x: float     # nose band left edge (inner eye corner)
    eye_inner_right_x: float    # nose band right edge
    nose_x: float               # face midline (nose tip x) — cheek split
    nose_bottom_y: float        # nose band floor (subnasale)
    mouth_cx: float
    mouth_cy: float
    mouth_half_width: float
    lower_lip_y: float          # chin line (lower lip bottom)
    oval_xy: list               # ordered (x, y) ring of the face oval

    def __post_init__(self):
        self.oval_path = MplPath(self.oval_xy)


def assign_region(cx: float, cy: float, a: Anchors) -> str | None:
    """Box CENTER -> region name, or None = off-face. Priority order matters:
    perioral is checked before chin_jaw (a lesion just under the lip is
    perioral, not chin)."""
    if not a.oval_path.contains_point((cx, cy)):
        return None
    if cy < a.brow_y:
        return "forehead"
    if a.eye_inner_left_x <= cx <= a.eye_inner_right_x and cy <= a.nose_bottom_y:
        return "nose"
    if math.dist((cx, cy), (a.mouth_cx, a.mouth_cy)) <= PERIORAL_RADIUS_FACTOR * a.mouth_half_width:
        return "perioral"
    if cy > a.lower_lip_y:
        return "chin_jaw"
    # ponytail: box-center + midline split; a lesion exactly on a region
    # boundary can misplace — upgrade to polygon containment only if boundary
    # errors show up in eval. "left" = image-left (viewer's left), consistent
    # everywhere; the recommender treats cheeks symmetrically anyway.
    return "left_cheek" if cx < a.nose_x else "right_cheek"
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_regions.py -v`
Expected: 8 passed

- [ ] **Step 8: Commit**

```bash
git add conftest.py requirements.txt src/classification/__init__.py src/classification/regions.py tests/test_regions.py
git commit -m "feat(stage2): region assignment pure geometry + off-face filter"
```

---

### Task 2: regions.py MediaPipe wrapper + no-face sentinel + overlay demo

The thin ML-adjacent shell: run Face Mesh once per image, extract the anchor
points from the 468 landmarks, assign every box, drop off-face ones, return
`None` when no face is found (the sentinel `assemble` turns into the no-face
branch). Plus the §6 "eyeball before metrics" overlay renderer as a `__main__`
demo. Logs decision D-019.

**Files:**
- Modify: `src/classification/regions.py` (append), `tests/test_regions.py` (append), `docs/DECISIONS.md` (append)

**Interfaces:**
- Consumes: `Anchors`, `assign_region` from Task 1.
- Produces (Task 5 relies on these exact names):
  - `extract_anchors(pts: list[tuple[float, float]]) -> Anchors` — `pts` is all 468 landmark points in **pixel** coords.
  - `assign(image, boxes) -> list[tuple[box, str]] | None` — `None` = no face found; otherwise `(box, region)` pairs with off-face boxes already dropped. `image` is RGB uint8 numpy; `boxes` are `(x, y, w, h, det_conf)` top-left pixel tuples.
  - `render(image, assignments) -> PIL.Image.Image` — boxes drawn colored by region.
  - Landmark index constants: `BROW_L=105, BROW_R=334, EYE_INNER_L=133, EYE_INNER_R=362, NOSE_TIP=1, NOSE_BOTTOM=2, MOUTH_L=61, MOUTH_R=291, LOWER_LIP=17, FACE_OVAL_IDS` (36-entry ordered ring).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_regions.py`:

```python
def _fake_landmarks():
    """468 points, only the indices extract_anchors reads are meaningful."""
    from src.classification.regions import (
        BROW_L, BROW_R, EYE_INNER_L, EYE_INNER_R, NOSE_TIP, NOSE_BOTTOM,
        MOUTH_L, MOUTH_R, LOWER_LIP, FACE_OVAL_IDS,
    )
    pts = [(0.0, 0.0)] * 468
    pts[BROW_L] = (70.0, 78.0)
    pts[BROW_R] = (130.0, 82.0)      # brow_y = mean = 80
    pts[EYE_INNER_L] = (80.0, 120.0)
    pts[EYE_INNER_R] = (120.0, 120.0)
    pts[NOSE_TIP] = (100.0, 150.0)
    pts[NOSE_BOTTOM] = (100.0, 180.0)
    pts[MOUTH_L] = (75.0, 220.0)
    pts[MOUTH_R] = (125.0, 220.0)    # center (100,220), half-width 25
    pts[LOWER_LIP] = (100.0, 235.0)
    ring = [(20, 0), (180, 0), (200, 40), (200, 260),
            (160, 300), (40, 300), (0, 260), (0, 40)]
    for k, idx in enumerate(FACE_OVAL_IDS):
        pts[idx] = ring[k % len(ring)]  # any on-ring point; ordering not asserted
    return pts


def test_extract_anchors_reads_the_right_landmarks():
    from src.classification.regions import extract_anchors
    a = extract_anchors(_fake_landmarks())
    assert a.brow_y == 80.0
    assert (a.eye_inner_left_x, a.eye_inner_right_x) == (80.0, 120.0)
    assert (a.nose_x, a.nose_bottom_y) == (100.0, 180.0)
    assert (a.mouth_cx, a.mouth_cy, a.mouth_half_width) == (100.0, 220.0, 25.0)
    assert a.lower_lip_y == 235.0
    assert len(a.oval_xy) == 36


def test_assign_no_face_returns_none():
    """Real MediaPipe on a black image: the no-face sentinel, not a crash."""
    from src.classification.regions import assign
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert assign(img, [(10, 10, 5, 5, 0.9)]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_regions.py -v`
Expected: the 8 Task-1 tests pass; the 2 new ones FAIL with `ImportError: cannot import name 'BROW_L'` / `cannot import name 'assign'`

- [ ] **Step 3: Write the implementation**

Append to `src/classification/regions.py`:

```python
# --- MediaPipe Face Mesh wrapper (the only non-pure part) -------------------
# Face Mesh 468-point topology indices — the ONLY landmarks we read.
BROW_L, BROW_R = 105, 334            # brow tops -> forehead line
EYE_INNER_L, EYE_INNER_R = 133, 362  # inner eye corners -> nose band edges
NOSE_TIP, NOSE_BOTTOM = 1, 2         # midline x / nose band floor
MOUTH_L, MOUTH_R = 61, 291           # lip corners -> perioral center+radius
LOWER_LIP = 17                       # chin line
# the standard ordered face-oval ring (36 points, clockwise from forehead top)
FACE_OVAL_IDS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def extract_anchors(pts) -> Anchors:
    """468 pixel-coord landmark points -> Anchors. min/max on the eye corners
    makes the nose band robust to left/right index conventions."""
    e1, e2 = pts[EYE_INNER_L][0], pts[EYE_INNER_R][0]
    (mx1, my1), (mx2, my2) = pts[MOUTH_L], pts[MOUTH_R]
    return Anchors(
        brow_y=(pts[BROW_L][1] + pts[BROW_R][1]) / 2.0,
        eye_inner_left_x=min(e1, e2), eye_inner_right_x=max(e1, e2),
        nose_x=pts[NOSE_TIP][0], nose_bottom_y=pts[NOSE_BOTTOM][1],
        mouth_cx=(mx1 + mx2) / 2.0, mouth_cy=(my1 + my2) / 2.0,
        mouth_half_width=abs(mx2 - mx1) / 2.0,
        lower_lip_y=pts[LOWER_LIP][1],
        oval_xy=[pts[i] for i in FACE_OVAL_IDS],
    )


def assign(image, boxes):
    """Run Face Mesh ONCE per image, then tag each box center with a region.

    Returns None if no face is found (the sentinel assemble turns into the
    no-face ConcernReport — bad angle/lighting is a real, common input, never
    a crash). Off-face boxes are dropped from the returned list.
    """
    import mediapipe as mp  # lazy: pure-geometry tests never touch mediapipe

    h, w = image.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                         max_num_faces=1) as fm:
        res = fm.process(image)  # expects RGB uint8 — our repo-wide format
    if not res.multi_face_landmarks:
        return None
    pts = [(lm.x * w, lm.y * h) for lm in res.multi_face_landmarks[0].landmark]
    anchors = extract_anchors(pts)
    out = []
    for b in boxes:
        x, y, bw, bh = b[:4]
        region = assign_region(x + bw / 2.0, y + bh / 2.0, anchors)
        if region is not None:
            out.append((b, region))
    return out


# --- eyeball tooling (spec §6: look before metrics) --------------------------
REGION_COLORS = {"forehead": "#e6194b", "nose": "#3cb44b",
                 "left_cheek": "#4363d8", "right_cheek": "#42d4f4",
                 "chin_jaw": "#f58231", "perioral": "#911eb4"}


def render(image, assignments):
    """Draw boxes colored by assigned region -> PIL Image."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(image)
    d = ImageDraw.Draw(im)
    for (b, region) in assignments:
        x, y, bw, bh = b[:4]
        d.rectangle([x, y, x + bw, y + bh], outline=REGION_COLORS[region], width=2)
        d.text((x, max(0, y - 12)), region, fill=REGION_COLORS[region])
    return im


if __name__ == "__main__":
    # manual check: python -m src.classification.regions FACE.jpg [OUT.png]
    # covers a face photo with a grid of small boxes and colors each by region
    # -> the §6 eyeball of region partitions + the off-face filter in one image.
    import sys
    import numpy as np
    from PIL import Image
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
    h, w = img.shape[:2]
    grid = [(x, y, 20, 20, 1.0)
            for y in range(0, h - 20, 40) for x in range(0, w - 20, 40)]
    result = assign(img, grid)
    if result is None:
        print("no face detected")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "regions_demo.png"
    render(img, result).save(out)
    print(f"{len(result)}/{len(grid)} grid boxes on-face -> {out}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_regions.py -v`
Expected: 10 passed (the no-face test takes a few seconds — MediaPipe init).

- [ ] **Step 5: Log decision D-019**

Append to `docs/DECISIONS.md` immediately after the D-018 section (before the trailing bullet list):

```markdown
### D-019 — Stage 2 region assignment: Face Mesh + landmark-anchored rules · LOCKED
MediaPipe Face Mesh runs once per image (CPU, no weights to manage); each
Stage-1 box center is assigned by explicit landmark-anchored geometric rules
(auditable — trust lives in readable logic), not a nearest-centroid black box.
The face-oval polygon doubles as the off-face false-positive filter. No face
found → no-face ConcernReport (low_light_flag + note), never a crash. Ceiling
accepted: box-center assignment can misplace a lesion sitting exactly on a
region boundary; the recommender only localises advice, so this is tolerable.
```

- [ ] **Step 6: Run the manual eyeball check (if a face photo is at hand)**

Run: `.venv/bin/python -m src.classification.regions <path-to-any-face-photo.jpg> /tmp/regions_demo.png`
Expected: prints `N/M grid boxes on-face -> /tmp/regions_demo.png`; open the PNG and confirm forehead/nose/cheeks/chin/perioral bands look sane and off-face grid boxes were dropped. If no photo is available, defer to the notebook's eval — but do it before trusting region metrics.

- [ ] **Step 7: Commit**

```bash
git add src/classification/regions.py tests/test_regions.py docs/DECISIONS.md
git commit -m "feat(stage2): mediapipe face-mesh wrapper, no-face sentinel, region overlay demo"
```

---

### Task 3: classifier.py — crop extraction + classifier interface (stub + torch)

The one ML piece, hidden behind an interface so the pipeline and its tests run
with no weights. `crop_with_context` is the knob that decides how much
surrounding morphology the model sees (pad 1.5× — ACNE04 boxes are already
loose per D-010). `StubClassifier` returns scripted probabilities;
`TorchClassifier` (MobileNetV3-small, 5-class head) is the prod path and hard-
errors on missing weights — never silently mislabel. Logs decision D-020.

**Files:**
- Create: `src/classification/classifier.py`, `tests/test_classifier.py`
- Modify: `docs/DECISIONS.md` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks (independent of regions).
- Produces (Task 5 and the notebook rely on these exact names):
  - `CLASSES = ["comedonal", "cystic", "inflammatory", "not_acne", "post_acne_mark"]` — alphabetical; index = model output index = torchvision `ImageFolder` order.
  - `crop_with_context(image, box, pad: float = 1.5, size: int = 112) -> np.ndarray` — `(size, size, 3)` uint8 RGB; replicate-pads at image edges (never shifts the box).
  - `StubClassifier(outputs: list[dict] | None = None)` with `.predict(crop) -> dict[str, float]` (all 5 classes as keys); scripted outputs are returned in order, last one repeats.
  - `TorchClassifier(weights: str, device: str = "cpu")` with the same `.predict` signature; raises `FileNotFoundError` if `weights` is absent; lazy-imports torch.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classifier.py`:

```python
"""Crop geometry + classifier interface. No torch, no weights needed."""
import numpy as np
from src.classification.classifier import CLASSES, StubClassifier, crop_with_context


def test_classes_locked_order():
    # alphabetical = torchvision ImageFolder order; a mismatch here means the
    # trained head's output indices no longer line up with predict()'s zip
    assert CLASSES == ["comedonal", "cystic", "inflammatory",
                       "not_acne", "post_acne_mark"]
    assert CLASSES == sorted(CLASSES)


def test_crop_center_box_shape_and_dtype():
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    crop = crop_with_context(img, (90, 90, 20, 20, 0.9), pad=1.5, size=112)
    assert crop.shape == (112, 112, 3)
    assert crop.dtype == np.uint8


def test_crop_edge_box_replicate_pads_not_shifts():
    img = np.zeros((100, 100, 3), np.uint8)
    img[0:3, 0:3] = (200, 10, 10)  # mark the true corner
    crop = crop_with_context(img, (0, 0, 10, 10), pad=2.0, size=64)
    assert crop.shape == (64, 64, 3)
    # replicate-pad keeps the corner pixel at the corner; a shifted box would
    # put black there instead
    assert tuple(crop[0, 0]) == (200, 10, 10)


def test_crop_degenerate_box_still_valid():
    img = np.zeros((100, 100, 3), np.uint8)
    crop = crop_with_context(img, (50, 50, 0, 0), pad=1.5, size=64)
    assert crop.shape == (64, 64, 3)


def test_stub_scripted_outputs_cycle_and_cover_all_classes():
    first = {"comedonal": 1.0, "inflammatory": 0.0, "cystic": 0.0,
             "post_acne_mark": 0.0, "not_acne": 0.0}
    s = StubClassifier([first])
    blank = np.zeros((112, 112, 3), np.uint8)
    p1, p2 = s.predict(blank), s.predict(blank)  # last output repeats
    for p in (p1, p2):
        assert set(p) == set(CLASSES)
        assert abs(sum(p.values()) - 1.0) < 1e-6
        assert max(p, key=p.get) == "comedonal"


def test_stub_default_is_valid_distribution():
    p = StubClassifier().predict(np.zeros((112, 112, 3), np.uint8))
    assert set(p) == set(CLASSES) and abs(sum(p.values()) - 1.0) < 1e-6


def test_torch_classifier_hard_errors_on_missing_weights():
    import pytest
    torch = pytest.importorskip("torch")  # skip cleanly where torch absent
    from src.classification.classifier import TorchClassifier
    with pytest.raises(FileNotFoundError):
        TorchClassifier("models/does_not_exist.pt")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_classifier.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.classification.classifier'`

- [ ] **Step 3: Write the implementation**

Create `src/classification/classifier.py`:

```python
"""Lesion-crop type classifier (Stage 2's one ML piece, spec §4.2).

Two implementations of one interface, predict(crop) -> {class: prob}:
- StubClassifier: scripted outputs, zero deps — lets assemble and every test
  run with no weights and no GPU (D-007 discipline).
- TorchClassifier: MobileNetV3-small, ImageNet-pretrained backbone with a
  5-class head, fine-tuned in Colab (notebooks/02_type_classifier.md). Hard
  error if weights are missing — never silently mislabel.

crop_with_context is the shared crop extractor: pad the (loose, D-010) box,
square it, replicate-pad at image edges, resize. `pad` decides how much
surrounding morphology (pustule head, ring of erythema) the model sees; it and
`size` are config values in configs/default.yaml.
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image

# ALPHABETICAL on purpose: torchvision ImageFolder sorts class folders, so
# the trained head's output index i is the i-th name here — same list at
# train and serve, an index swap is structurally impossible. Maps 1:1 onto
# schema concerns in assemble.CONCERN_MAP (spec §3); not_acne is dropped there.
CLASSES = ["comedonal", "cystic", "inflammatory", "not_acne", "post_acne_mark"]


def crop_with_context(image, box, pad: float = 1.5, size: int = 112):
    """(x, y, w, h[, conf]) top-left pixel box -> (size, size, 3) uint8 crop.

    Square window of side max(w, h) * pad around the box CENTER; when the
    window spills past the image edge we replicate-pad (never shift the box —
    shifting would move the lesion off-center).
    """
    x, y, w, h = box[:4]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(max(w, h) * pad, 8.0)  # 8px floor guards degenerate boxes
    half = side / 2.0
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    ih, iw = image.shape[:2]
    crop = image[max(0, y0):min(ih, y1), max(0, x0):min(iw, x1)]
    assert crop.size, f"box entirely outside image: {box}"  # can't happen via assemble (centers are on-face)
    pads = ((max(0, -y0), max(0, y1 - ih)), (max(0, -x0), max(0, x1 - iw)), (0, 0))
    if any(p for pair in pads for p in pair):
        crop = np.pad(crop, pads, mode="edge")
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


class StubClassifier:
    """Scripted stand-in with the real interface. `outputs` are returned in
    order; the last one repeats (so N boxes can each get a scripted class)."""

    def __init__(self, outputs=None):
        self.outputs = outputs or [{
            "comedonal": 0.05, "inflammatory": 0.80, "cystic": 0.05,
            "post_acne_mark": 0.05, "not_acne": 0.05,
        }]
        self._i = 0

    def predict(self, crop) -> dict:
        probs = self.outputs[min(self._i, len(self.outputs) - 1)]
        self._i += 1
        assert set(probs) == set(CLASSES), "stub outputs must cover all classes"
        return dict(probs)


class TorchClassifier:
    """Prod path: MobileNetV3-small + 5-class head. Lazy torch import so the
    rest of Stage 2 (and its tests) never needs torch installed."""

    def __init__(self, weights: str, device: str = "cpu"):
        import torch
        import torchvision
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"classifier weights missing: {weights} — train them via "
                "notebooks/02_type_classifier.md (do not run the prod path "
                "with a stub: never silently mislabel)")
        self._torch = torch
        model = torchvision.models.mobilenet_v3_small()
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, len(CLASSES))
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()
        self.model = model.to(device)
        self.device = device

    def predict(self, crop) -> dict:
        t = self._torch
        x = t.from_numpy(crop.astype("float32") / 255.0).permute(2, 0, 1)
        mean = t.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # ImageNet norm —
        std = t.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)   # must match training
        x = ((x - mean) / std).unsqueeze(0).to(self.device)
        with t.no_grad():
            probs = t.softmax(self.model(x)[0], dim=0).cpu().tolist()
        return dict(zip(CLASSES, probs))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_classifier.py -v`
Expected: 6 passed, 1 skipped (the torch test skips — torch not installed locally). If torch IS installed: 7 passed.

- [ ] **Step 5: Log decision D-020**

Append to `docs/DECISIONS.md` right after the D-019 entry:

```markdown
### D-020 — Stage 2 classifier taxonomy: 5 visual classes, hybrid training · LOCKED
Classes: comedonal (blackheads/whiteheads), inflammatory (papules/pustules),
cystic (nodules/cysts), post_acne_mark, not_acne. The comedonal/inflammatory
split is kept because RULES.md §1 gives them different first-line actives —
merging would blind the recommender. post_acne_mark is ONE combined
pigmentation+texture "scarring" bucket (flat-vs-textural is an information
ceiling for a single flat-lit phone photo, not a data gap); it maps to the
existing hyperpigmentation concern — no schema change — plus a mandatory
texture advisory ("topicals won't resolve texture; procedures need a
professional"). not_acne absorbs detector false positives and is dropped.
Training is hybrid: optional Kaggle acne-set pretrain (features only; dataset
quality UNVERIFIED — inspect before use) + fine-tune on self-labeled crops
from running the Stage 1 detector over train-eligible ACNE04 images. Self-
collected phone photos stay TEST-ONLY (D-014). Fallback if post_acne_mark has
too few crops (< ~30) to learn a stable class: collapse it into not_acne for
v1, accepting the raised cost — we lose the whole pigmentation recommendation,
not just a note. Partially advances D-012; comprehensive mark/scar coverage
stays deferred.
```

- [ ] **Step 6: Commit**

```bash
git add src/classification/classifier.py tests/test_classifier.py docs/DECISIONS.md
git commit -m "feat(stage2): lesion crop extraction + classifier interface (stub + torch)"
```

---

### Task 4: Recommender touch-ups — no-face guard + texture advisory

Consumer fixes the spec requires OUTSIDE `src/classification/` (§4.3, §11).
Today `recommend()` treats empty `concerns` as clear skin → maintenance
routine, which is wrong for "couldn't analyse". And hyperpigmentation arriving
from `post_acne_mark` boxes needs the texture advisory. Signal for the latter:
a `hyperpigmentation` concern **with `lesion_count` set** — only the counted-
marks path produces that (CONCERN_SCHEMA: hyperpigmentation normally doesn't
count discretely), so no schema change and no notes-string parsing.

**Files:**
- Modify: `src/recommendation/engine.py`, `docs/RULES.md`
- Create: `tests/test_engine.py`

**Interfaces:**
- Consumes: `ConcernReport`, `Concern`, `Product` from `src/recommendation/schema.py` (existing, unchanged).
- Produces (Task 5's e2e test relies on these):
  - `TEXTURE_ADVISORY: str` constant importable from `src.recommendation.engine`.
  - `recommend()` behaviour: empty `concerns` + `low_light_flag=True` → empty routine with a flag containing `"couldn't analyse"`; any `hyperpigmentation` concern with truthy `lesion_count` → `TEXTURE_ADVISORY` in `flags` (once).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engine.py`:

```python
"""Recommender touch-ups for Stage 2 (spec §4.3, §11)."""
from src.recommendation.engine import TEXTURE_ADVISORY, recommend
from src.recommendation.schema import Concern, ConcernReport, Product

CATALOG = [
    Product("p1", "Azelaic 10%", "x", "treatment", actives=["azelaic_acid"]),
    Product("p2", "Niacinamide", "x", "serum", actives=["niacinamide"]),
    Product("p3", "SPF50", "x", "spf"),
]


def test_no_face_is_not_maintenance():
    rep = ConcernReport("img1", concerns=[], low_light_flag=True,
                        notes="no face detected")
    rec = recommend(rep, CATALOG)
    assert any("couldn't analyse" in f for f in rec.flags)
    assert rec.target_actives == []
    assert all(not products for products in rec.routine.values())


def test_plain_clear_skin_still_maintenance():
    rec = recommend(ConcernReport("img2", concerns=[]), CATALOG)
    assert "maintenance routine" in rec.flags


def test_texture_advisory_on_counted_marks():
    rep = ConcernReport("img3", concerns=[
        Concern("hyperpigmentation", "forehead", 1, 0.8, lesion_count=2),
        Concern("hyperpigmentation", "left_cheek", 1, 0.8, lesion_count=1),
    ])
    rec = recommend(rep, CATALOG)
    assert rec.flags.count(TEXTURE_ADVISORY) == 1  # attached once, not per region
    assert rec.routine["spf"], "SPF stays mandatory with hyperpigmentation"


def test_no_advisory_for_uncounted_hyperpigmentation():
    rep = ConcernReport("img4", concerns=[
        Concern("hyperpigmentation", "forehead", 1, 0.8),  # lesion_count None
    ])
    assert TEXTURE_ADVISORY not in recommend(rep, CATALOG).flags
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_engine.py -v`
Expected: FAIL at collection — `ImportError: cannot import name 'TEXTURE_ADVISORY'`

- [ ] **Step 3: Implement the two engine changes**

In `src/recommendation/engine.py`, add the constant right after the
`INCOMPATIBLE` list (before `@dataclass class Recommendation`):

```python
# Stage 2 spec §3: post-acne marks are ONE bucket covering pigmentation AND
# texture. Topicals only help the pigmentation half; say so every time.
TEXTURE_ADVISORY = ("post-acne marks: topicals help the pigmentation; "
                    "textural scarring may need a professional — procedures, "
                    "topicals won't resolve texture (appearance-based note)")
```

Then inside `recommend()`, insert the no-face guard as the FIRST statement
after `flags: list[str] = []` (i.e. before the clear-skin branch):

```python
    # no-face / unusable image: empty concerns + low_light_flag means
    # "couldn't analyse", NOT clear skin (Stage 2 spec §4.3) — prescribing a
    # maintenance routine here would be confidently wrong
    if not report.concerns and report.low_light_flag:
        return Recommendation({c: [] for c in CATEGORIES}, [],
                              ["couldn't analyse — no face detected / image "
                               "unusable; retake the photo"])
```

And replace the two lines inside the concern loop:

```python
        if c.concern == "hyperpigmentation":
            needs_spf = True  # RULES.md §3, non-negotiable
```

with:

```python
        if c.concern == "hyperpigmentation":
            needs_spf = True  # RULES.md §3, non-negotiable
            if c.lesion_count and TEXTURE_ADVISORY not in flags:
                # counted marks = Stage 2's post_acne_mark path (spec §3);
                # area-based hyperpigmentation arrives with lesion_count=None
                flags.append(TEXTURE_ADVISORY)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_engine.py -v`
Expected: 4 passed

- [ ] **Step 5: Record the rule in RULES.md**

In `docs/RULES.md` §3, append one bullet to the `Rules:` list (after the
moisturizer bullet):

```markdown
- hyperpigmentation arriving as counted post-acne marks (`lesion_count` set)
  additionally carries the texture advisory: topicals address the pigmentation
  only; textural scarring needs professional procedures. (Stage 2 spec §3 —
  mirrors the cystic see-a-pro discipline for the part topicals can't fix.)
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -v`
Expected: all tests pass (regions 10, classifier 6+1skip, engine 4).

- [ ] **Step 7: Commit**

```bash
git add src/recommendation/engine.py tests/test_engine.py docs/RULES.md
git commit -m "feat(stage3): no-face guard + post-acne-mark texture advisory in recommender"
```

---

### Task 5: assemble.py — analyze() → ConcernReport + severity config

The orchestrator (spec §4.3): regions → crops → classifier → drop `not_acne` →
map `post_acne_mark` → `hyperpigmentation` → group by (concern, region) →
count → severity → mean confidence → `ConcernReport`. `derive_severity` is a
~5-line pure function living here (ponytail: split to severity.py only if
thresholds grow per-region-specific logic). Config gains the `classification:`
section. This task lands the end-to-end contract test: fake boxes + stub
classifier → schema-valid report that `recommend()` consumes — no GPU, no
weights.

**Files:**
- Create: `src/classification/assemble.py`, `tests/test_assemble.py`
- Modify: `configs/default.yaml`, `src/classification/__init__.py`

**Interfaces:**
- Consumes: `regions.assign` (Task 2), `classifier.crop_with_context` + `CLASSES` semantics (Task 3), `Concern`/`ConcernReport` from `src/recommendation/schema.py`, `recommend()`/`TEXTURE_ADVISORY` behaviour (Task 4).
- Produces (the public Stage-2 surface):
  - `analyze(image, boxes, classifier, config: dict | None = None, image_id: str = "unknown", assign_fn=None) -> ConcernReport` — `config` is the `classification:` section dict (None → loaded from `configs/default.yaml`); `assign_fn` defaults to `regions.assign` (injectable so the contract test needs no MediaPipe face).
  - `derive_severity(count: int, thresholds: list) -> int` — 0–4 ordinal.
  - `CONCERN_MAP: dict` — classifier class → schema concern (no `not_acne` key).
  - `src.classification.analyze` re-export.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_assemble.py`:

```python
"""End-to-end Stage 2 contract: fake boxes + StubClassifier + injected region
assignments -> schema-valid ConcernReport -> recommend() consumes it.
No GPU, no weights, no MediaPipe face needed (D-007 discipline)."""
import numpy as np
from src.classification.assemble import analyze, derive_severity
from src.classification.classifier import CLASSES, StubClassifier
from src.recommendation.engine import TEXTURE_ADVISORY, recommend
from src.recommendation.schema import Product

CFG = {"crop_pad": 1.5, "crop_size": 112, "weights": "unused",
       "severity_thresholds": {"default": [1, 3, 6, 10]}}
IMG = np.zeros((300, 200, 3), dtype=np.uint8)
CATALOG = [
    Product("p1", "BP gel", "x", "treatment", actives=["benzoyl_peroxide"]),
    Product("p2", "SPF50", "x", "spf"),
]


def probs(winner):
    p = {c: 0.05 for c in CLASSES}
    p[winner] = 0.80
    return p


def test_severity_boundaries():
    th = [1, 3, 6, 10]
    assert derive_severity(0, th) == 0
    assert derive_severity(1, th) == 1
    assert derive_severity(2, th) == 1
    assert derive_severity(3, th) == 2
    assert derive_severity(5, th) == 2
    assert derive_severity(6, th) == 3
    assert derive_severity(9, th) == 3
    assert derive_severity(10, th) == 4
    assert derive_severity(50, th) == 4


def test_analyze_end_to_end_contract():
    boxes = [(20, 50, 10, 10, 0.9), (40, 50, 10, 10, 0.9), (60, 50, 10, 10, 0.9),
             (90, 20, 10, 10, 0.9), (120, 50, 10, 10, 0.9)]
    assignments = [(boxes[0], "left_cheek"), (boxes[1], "left_cheek"),
                   (boxes[2], "left_cheek"), (boxes[3], "forehead"),
                   (boxes[4], "left_cheek")]
    stub = StubClassifier([probs("inflammatory"), probs("inflammatory"),
                           probs("inflammatory"), probs("post_acne_mark"),
                           probs("not_acne")])
    rep = analyze(IMG, boxes, stub, config=CFG, image_id="t1",
                  assign_fn=lambda img, bxs: assignments)
    # 3 inflammatory on left cheek -> severity 2; 1 mark on forehead ->
    # hyperpigmentation severity 1; the not_acne box vanishes entirely
    assert {(c.concern, c.region, c.severity, c.lesion_count) for c in rep.concerns} \
        == {("acne_inflammatory", "left_cheek", 2, 3),
            ("hyperpigmentation", "forehead", 1, 1)}
    assert all(abs(c.confidence - 0.80) < 1e-6 for c in rep.concerns)
    assert rep.overall_severity == 2          # hyperpigmentation never inflates it
    assert not rep.clear_skin
    rec = recommend(rep, CATALOG)             # Stage 3 consumes without error
    assert TEXTURE_ADVISORY in rec.flags      # counted marks carry the advisory
    assert rec.routine["spf"]                 # hyperpigmentation mandates SPF


def test_analyze_no_face_branch():
    rep = analyze(IMG, [(5, 5, 4, 4, 0.9)], StubClassifier(), config=CFG,
                  image_id="t2", assign_fn=lambda img, bxs: None)
    assert rep.low_light_flag and not rep.concerns and "no face" in rep.notes
    assert not rep.clear_skin                 # couldn't analyse != clear skin
    rec = recommend(rep, CATALOG)
    assert any("couldn't analyse" in f for f in rec.flags)


def test_all_not_acne_is_clear_skin():
    box = (20, 50, 10, 10, 0.9)
    rep = analyze(IMG, [box], StubClassifier([probs("not_acne")]), config=CFG,
                  assign_fn=lambda img, bxs: [(box, "nose")])
    assert rep.clear_skin and not rep.concerns and not rep.low_light_flag


def test_per_region_threshold_override():
    box = lambda i: (20 + 12 * i, 20, 10, 10, 0.9)
    boxes = [box(i) for i in range(3)]
    cfg = dict(CFG, severity_thresholds={"default": [1, 3, 6, 10],
                                         "forehead": [1, 2, 3, 4]})
    rep = analyze(IMG, boxes, StubClassifier([probs("comedonal")] * 3),
                  config=cfg, assign_fn=lambda i, b: [(b_, "forehead") for b_ in boxes])
    assert rep.concerns[0].severity == 3      # 3 lesions vs [1,2,3,4] -> 3, not 2


def test_public_reexport():
    from src.classification import analyze as public_analyze
    assert public_analyze is analyze
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_assemble.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.classification.assemble'`

- [ ] **Step 3: Write the implementation**

Create `src/classification/assemble.py`:

```python
"""Stage 2 orchestration: analyze(image, boxes) -> ConcernReport (spec §4.3).

The assembler, not a monolith: regions -> crops -> classifier -> group ->
severity -> the locked D-008 contract. Three of its four moving parts are pure
and deterministic; the classifier hides behind an interface, so this whole
module tests with a stub — no GPU, no weights.

image: RGB uint8 numpy (H, W, 3). boxes: Stage 1 output — (x, y, w, h,
det_confidence), pixel coords, TOP-LEFT anchored.
"""
from __future__ import annotations
import os
from statistics import fmean

import yaml

from src.recommendation.schema import Concern, ConcernReport
from . import regions
from .classifier import crop_with_context

# classifier class -> schema concern (spec §3). not_acne has no entry: those
# boxes are dropped entirely. post_acne_mark rides the existing
# hyperpigmentation concern (no schema change); its lesion_count doubles as
# the engine's texture-advisory signal.
CONCERN_MAP = {
    "comedonal": "acne_comedonal",
    "inflammatory": "acne_inflammatory",
    "cystic": "acne_cystic",
    "post_acne_mark": "hyperpigmentation",
}


def derive_severity(count: int, thresholds) -> int:
    """Per-REGION lesion count -> ordinal 0-4: how many thresholds the count
    clears. Hayashi grades whole-face counts; per-region counts are smaller,
    hence separate, config-tuned thresholds (spec §4.3 caveat)."""
    assert len(thresholds) == 4 and sorted(thresholds) == list(thresholds), \
        f"severity_thresholds must be 4 ascending counts, got {thresholds}"
    return sum(count >= t for t in thresholds)


def _load_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "configs", "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["classification"]


def analyze(image, boxes, classifier, config=None, image_id="unknown",
            assign_fn=None) -> ConcernReport:
    """The one public Stage-2 entry point (spec §1).

    classifier: anything with predict(crop) -> {class: prob} — TorchClassifier
    in prod, StubClassifier in tests. assign_fn: injectable region assigner
    (defaults to regions.assign) so the contract test needs no real face.
    """
    cfg = config or _load_config()
    assignments = (assign_fn or regions.assign)(image, boxes)
    if assignments is None:
        # no face: NOT clear skin. low_light_flag + note is the signal the
        # recommender's guard turns into "couldn't analyse" (spec §4.3).
        return ConcernReport(image_id=image_id, low_light_flag=True,
                             notes="no face detected")

    groups: dict = {}  # (concern, region) -> [confidence, ...]
    for box, region in assignments:
        crop = crop_with_context(image, box,
                                 pad=cfg["crop_pad"], size=cfg["crop_size"])
        probs = classifier.predict(crop)
        cls = max(probs, key=probs.get)
        if cls == "not_acne":
            continue
        groups.setdefault((CONCERN_MAP[cls], region), []).append(probs[cls])

    th_cfg = cfg["severity_thresholds"]
    concerns = [
        Concern(concern=concern, region=region,
                severity=derive_severity(len(confs),
                                         th_cfg.get(region, th_cfg["default"])),
                confidence=round(fmean(confs), 4),
                lesion_count=len(confs))
        for (concern, region), confs in sorted(groups.items())
    ]
    return ConcernReport(image_id=image_id, concerns=concerns,
                         clear_skin=not concerns)
```

Replace the contents of `src/classification/__init__.py` with:

```python
from .assemble import analyze  # the one public Stage-2 entry point (spec §1)
```

- [ ] **Step 4: Add the classification config section**

Append to `configs/default.yaml` (after the `recommendation:` block):

```yaml
classification:                       # Stage 2 (spec 2026-07-05)
  crop_pad: 1.5                       # box padding factor before square+resize —
                                      # decides how much surrounding morphology
                                      # the classifier sees; tune on the val set
  crop_size: 112
  weights: models/type_classifier.pt  # trained via notebooks/02_type_classifier.md
  severity_thresholds:                # per-REGION count -> ordinal 0-4
    default: [1, 3, 6, 10]            # >=1:1  >=3:2  >=6:3  >=10:4 — Hayashi is
                                      # whole-face; per-region counts are smaller.
                                      # Tune once Stage 1 produces real distributions.
    # forehead: [1, 4, 8, 12]         # per-region override goes here when tuned
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_assemble.py -v`
Expected: 6 passed

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -v`
Expected: everything passes (regions 10, classifier 6+1skip, engine 4, assemble 6).

- [ ] **Step 7: Commit**

```bash
git add src/classification/assemble.py src/classification/__init__.py configs/default.yaml tests/test_assemble.py
git commit -m "feat(stage2): assemble analyze() -> ConcernReport + severity config"
```

---

### Task 6: Notebook 02 — type-classifier training curriculum

The Colab walkthrough that produces `models/type_classifier.pt`, mirroring
notebook 01's cell-by-cell voice: inspect before parsing, eyeball before
metrics, decision gates written down. The self-labeled ACNE04 crop set built
in cells 5–6 is the real deliverable of the stage (spec §4.2). This is a
document task — no pytest cycle; the check is that every referenced repo
symbol actually exists.

**Files:**
- Create: `notebooks/02_type_classifier.md`

**Interfaces:**
- Consumes: `CLASSES`, `crop_with_context` (Task 3), `TorchClassifier` weights format (MobileNetV3-small, 5-class head, `state_dict`), config values `crop_pad=1.5` / `crop_size=112` (Task 5), Stage 1 `best.pt` from notebook 01.
- Produces: `models/type_classifier.pt` (when the user runs it in Colab; the file itself is gitignored).

- [ ] **Step 1: Write the notebook**

Create `notebooks/02_type_classifier.md` with exactly this content:

````markdown
# Notebook 02 — lesion-type classifier (Stage 2)

Cell-by-cell Colab walkthrough. Runtime → change runtime type → **GPU (T4)**.
The pipeline code (`src/classification/*.py`) is tested locally; the cells
below need a GPU and the real data, so they run in Colab.

Build order:
1. environment + get the code
2. (optional) Kaggle pretrain set — **verify it before believing it**
3. eyeball the pretrain data
4. pretrain the backbone
5. self-label: run the Stage 1 detector over ACNE04 → dump crops
6. hand-sort crops into the 5 classes ← **the real deliverable**
7. top up `not_acne` with FFHQ / on-face patches
8. fine-tune on the sorted crops
9. look at predictions before metrics
10. eval: confusion matrix, per-class recall, Fitzpatrick disaggregation
11. export weights
12. domain gap: self-collected phone crops (TEST ONLY — D-014)

Classes (locked, D-020 — alphabetical, matching both `CLASSES` in
`src/classification/classifier.py` and torchvision `ImageFolder`'s sort order):
`comedonal · cystic · inflammatory · not_acne · post_acne_mark`

---

### Cell 1 — environment

```python
# torch + torchvision ship preinstalled in Colab; add the rest
!pip install ultralytics -q
import sys; sys.path.insert(0, "/content/skinscan")   # git clone / upload the repo
from src.classification.classifier import CLASSES, crop_with_context
print(CLASSES)   # sanity: 5 classes, alphabetical (= ImageFolder order)
```

### Cell 2 — (optional) Kaggle pretrain set — VERIFY FIRST

⚠ The candidate set (`kaggle.com/code/zulqarnain11/acne-classification-using-cnn`)
is **unverified** — its exact dataset, class list, and quality were never
confirmed (spec §4.2). Its value would be *features*, not plug-and-play
weights: it classifies whole/region images, never a ~40px crop.

```python
!pip install kaggle -q
# upload your kaggle.json first: from google.colab import files; files.upload()
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d <SLUG-FROM-THE-KAGGLE-PAGE> -p /content/pretrain --unzip
import glob
for d in sorted(glob.glob("/content/pretrain/**/", recursive=True)):
    imgs = glob.glob(d + "*.jpg") + glob.glob(d + "*.png")
    if imgs: print(f"{d}: {len(imgs)} images")
```

**Decision gate (write the answer down):**
- Do its classes map onto ours (even coarsely, e.g. its "blackheads" → `comedonal`)?
- Are there enough images (>~1k) and do the labels survive eyeballing (cell 3)?
- **NO to either → skip cells 3–4 entirely.** ImageNet features are the
  fallback backbone; cells 5+ are unchanged either way.

### Cell 3 — eyeball the pretrain data

```python
import matplotlib.pyplot as plt
from PIL import Image
import random
dirs = sorted(glob.glob("/content/pretrain/*/"))
fig, axes = plt.subplots(len(dirs), 8, figsize=(16, 2 * len(dirs)))
for row, d in zip(axes, dirs):
    picks = random.sample(glob.glob(d + "*"), 8)
    for ax, p in zip(row, picks):
        ax.imshow(Image.open(p)); ax.set_title(d.split("/")[-2], fontsize=7); ax.axis("off")
plt.tight_layout()
```

Same rule as notebook 01 cell 5: if the labels look wrong here, they ARE wrong
— stop and rethink before training on them (D-011: Kaggle labels are
second-tier; validate by eyeballing before trusting).

### Cell 4 — pretrain the backbone

```python
import torch, torchvision
from torchvision import transforms, datasets

# map the Kaggle folders onto (a subset of) our class names first — one
# os.rename per folder — so the label semantics carry into fine-tuning
tf_train = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ds = datasets.ImageFolder("/content/pretrain_mapped", tf_train)
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)

model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(ds.classes))
model = model.cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
lossf = torch.nn.CrossEntropyLoss()
for epoch in range(5):
    model.train(); tot = 0.0
    for x, y in dl:
        opt.zero_grad()
        loss = lossf(model(x.cuda()), y.cuda())
        loss.backward(); opt.step(); tot += loss.item()
    print(f"epoch {epoch}: loss {tot / len(dl):.3f}")
torch.save(model.state_dict(), "/content/pretrained_backbone.pt")
```

### Cell 5 — self-label: run Stage 1 over ACNE04, dump crops

The whole point (spec §4.2): the Kaggle model never saw a ~40px lesion crop.
Whole-image training → per-crop inference is the exact train/serve mismatch
that caps quality; fine-tuning on real detector crops closes it.

```python
from ultralytics import YOLO
import numpy as np, os
from pathlib import Path
from PIL import Image

det = YOLO("/content/best.pt")   # Stage 1 weights from notebook 01
                                 # (runs/detect/train/weights/best.pt)
os.makedirs("/content/crops/unsorted", exist_ok=True)

# TRAIN-SPLIT ACNE04 images ONLY. Never the val/test split (leakage into the
# classifier's val set) and never self-collected photos (D-014: test-only).
train_imgs = sorted(glob.glob("/content/acne04_yolo/images/train/*.jpg"))

n = 0
for p in train_imgs:
    img = np.asarray(Image.open(p).convert("RGB"))
    for b in det(p, conf=0.25, verbose=False)[0].boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        # convert corners -> our top-left (x, y, w, h) contract, then use the
        # SAME crop function inference uses — pad/size must match configs/default.yaml
        crop = crop_with_context(img, (x1, y1, x2 - x1, y2 - y1), pad=1.5, size=112)
        Image.fromarray(crop).save(f"/content/crops/unsorted/{n:05d}_{Path(p).stem}.png")
        n += 1
print(n, "crops dumped")   # expect thousands; a few hundred sorted is enough
```

### Cell 6 — hand-sort into 5 classes (the real work)

Make the folders, then drag crops into them (Colab file pane, or download the
folder, sort locally, re-upload):

```python
for c in CLASSES:
    os.makedirs(f"/content/crops/sorted/{c}", exist_ok=True)
```

Sorting guide — decide by what's VISIBLE, not by what you'd palpate:
- `comedonal` — small dark plug (blackhead) or tiny pale bump (whitehead), no redness ring
- `inflammatory` — red papule or white-headed pustule with surrounding erythema
- `cystic` — large, deep, angry swelling; when unsure between inflammatory and cystic, pick inflammatory (cystic routes to a professional — keep its precision high)
- `post_acne_mark` — FLAT brown/red mark or textural pit where a lesion was; typically detector false-positives that landed on old marks — hunt these deliberately, it's the thinnest class
- `not_acne` — mole, shadow, hair, glasses edge, clear skin, anything else the detector hallucinated

Aim for a few hundred sorted crops total. Then count:

```python
counts = {c: len(glob.glob(f"/content/crops/sorted/{c}/*")) for c in CLASSES}
print(counts)
```

**Decision gate (D-020 fallback):** if `post_acne_mark` < ~30, collapse it
into `not_acne` for v1 (move the files, retrain the head with 4 classes) —
acknowledging the raised cost: we lose the entire pigmentation
recommendation, not just a note. Record the counts and the decision in the
training log either way.

### Cell 7 — top up `not_acne` (D-013)

```python
# random on-face non-lesion patches from FFHQ clear-skin images: sample boxes
# away from any detection, crop with the same function
import random
ffhq = sorted(glob.glob("/content/ffhq_subset/*.png"))[:200]
i = len(glob.glob("/content/crops/sorted/not_acne/*"))
for p in ffhq:
    img = np.asarray(Image.open(p).convert("RGB"))
    h, w = img.shape[:2]
    x, y = random.randint(w // 4, 3 * w // 4), random.randint(h // 3, 3 * h // 4)
    crop = crop_with_context(img, (x, y, 24, 24), pad=1.5, size=112)
    Image.fromarray(crop).save(f"/content/crops/sorted/not_acne/ffhq_{i:05d}.png")
    i += 1
```

### Cell 8 — fine-tune on the sorted crops

```python
from torch.utils.data import DataLoader, random_split

tf_val = transforms.Compose([
    transforms.Resize((112, 112)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
full = datasets.ImageFolder("/content/crops/sorted", tf_train)
# CLASSES is alphabetical precisely so this holds — if it ever fails, the
# exported head's indices would not match TorchClassifier.predict's zip
assert full.classes == list(CLASSES), f"folder/class mismatch: {full.classes}"
n_val = max(1, len(full) // 5)
train_ds, val_ds = random_split(full, [len(full) - n_val, n_val],
                                generator=torch.Generator().manual_seed(0))
val_ds.dataset = datasets.ImageFolder("/content/crops/sorted", tf_val)  # no flip on val

model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(full.classes))
try:   # graft the cell-4 pretrain if it exists; head shape may differ — skip it
    sd = torch.load("/content/pretrained_backbone.pt")
    model.load_state_dict({k: v for k, v in sd.items()
                           if not k.startswith("classifier.3")}, strict=False)
    print("pretrained backbone loaded")
except FileNotFoundError:
    print("no pretrain — ImageNet features only (fine, see cell 2 gate)")
model = model.cuda()

# class imbalance (cystic is rare, post_acne_mark thin): weighted loss
import collections
freq = collections.Counter(y for _, y in full.samples)
weights = torch.tensor([len(full) / freq[i] for i in range(len(full.classes))],
                       dtype=torch.float32).cuda()
lossf = torch.nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)   # low LR: shift gently
for epoch in range(15):
    model.train()
    for x, y in DataLoader(train_ds, batch_size=32, shuffle=True):
        opt.zero_grad(); lossf(model(x.cuda()), y.cuda()).backward(); opt.step()
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in DataLoader(val_ds, batch_size=64):
            correct += (model(x.cuda()).argmax(1).cpu() == y).sum().item()
            total += len(y)
    print(f"epoch {epoch}: val acc {correct / total:.3f}")
```

### Cell 9 — LOOK at predictions before metrics

Same discipline as notebook 01 cell 8. A grid per PREDICTED class — you're
hunting systematic confusions (marks predicted inflammatory? moles predicted
comedonal?), not admiring accuracy.

```python
model.eval()
by_pred = {c: [] for c in full.classes}
with torch.no_grad():
    for i in range(len(val_ds)):
        x, y = val_ds[i]
        pred = full.classes[model(x.unsqueeze(0).cuda()).argmax(1).item()]
        by_pred[pred].append((x, full.classes[y]))
fig, axes = plt.subplots(len(full.classes), 8, figsize=(16, 2 * len(full.classes)))
for row, c in zip(axes, full.classes):
    for ax, (x, true) in zip(row, by_pred[c][:8]):
        img = (x.permute(1, 2, 0).numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        ax.imshow(img); ax.set_title(f"pred {c} / true {true}", fontsize=6); ax.axis("off")
plt.tight_layout()
```

### Cell 10 — eval (D-017, D-016)

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

ys, ps, scores = [], [], []
with torch.no_grad():
    for x, y in DataLoader(val_ds, batch_size=64):
        out = torch.softmax(model(x.cuda()), dim=1).cpu()
        ps += out.argmax(1).tolist(); scores += out.tolist(); ys += y.tolist()
print(classification_report(ys, ps, target_names=full.classes, digits=3))
ConfusionMatrixDisplay(confusion_matrix(ys, ps), display_labels=full.classes).plot()

# per-class PR curves (spec §6) — thresholds matter more than argmax accuracy
# for the rare classes
from sklearn.metrics import PrecisionRecallDisplay
scores = np.array(scores)
for i, c in enumerate(full.classes):
    PrecisionRecallDisplay.from_predictions(
        [int(y == i) for y in ys], scores[:, i], name=c, ax=plt.gca())
plt.title("one-vs-rest PR curves")
```

Watch specifically (spec §6): `cystic` recall and `post_acne_mark` recall —
the rare/thin classes are exactly where a healthy headline accuracy hides
failure (Lesson-2: summary metrics compress failures).

**Fitzpatrick disaggregation (D-016, mandatory to ATTEMPT):** tag each val
image's Fitzpatrick group (I–VI, by eye, at the source-image level) in a dict
`{image_stem: group}`, join on the crop filename stems, and print the
classification report per group. If the val set is too small to disaggregate
meaningfully, **write that down as a finding** — "couldn't measure tone bias
with N=…" is itself a result, silence is not.

### Cell 11 — export weights

```python
torch.save(model.state_dict(), "/content/type_classifier.pt")
from google.colab import files; files.download("/content/type_classifier.pt")
```

Place it at `models/type_classifier.pt` in the repo (gitignored — weights
never commit). Then sanity-check the prod path locally:

```bash
.venv/bin/pip install torch torchvision   # first time only
.venv/bin/python -c "
from src.classification.classifier import TorchClassifier
import numpy as np
c = TorchClassifier('models/type_classifier.pt')
print(c.predict(np.zeros((112, 112, 3), dtype='uint8')))"
```

⚠ If you changed `pad`/`size` (or trained 4 classes via the cell-6 fallback),
update `configs/default.yaml` and `CLASSES` to match — train/serve settings
must be identical, that mismatch is the whole lesson of this stage.

### Cell 12 — domain gap: self-collected crops (TEST ONLY, D-014)

Run Stage 1 + this classifier over the self-collected phone photos, hand-label
the resulting crops, and report the same per-class metrics. The gap between
this number and cell 10's is the ACNE04→phone-photo domain gap — measuring it
is the point; training on these photos would both violate D-014 and destroy
the measurement.
````

- [ ] **Step 2: Verify every repo symbol the notebook references exists**

```bash
.venv/bin/python -c "
from src.classification.classifier import CLASSES, crop_with_context, TorchClassifier
from src.classification import analyze
print('notebook symbol refs ok:', CLASSES)"
grep -n "crop_pad: 1.5" configs/default.yaml && grep -n "crop_size: 112" configs/default.yaml
```

Expected: prints the 5 classes and both grep lines match (notebook's `pad=1.5, size=112` mirror the config).

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_type_classifier.md
git commit -m "docs(stage2): notebook 02 — type-classifier training curriculum"
```

---

## Done when

- `.venv/bin/pytest -v` is fully green (≈26 tests; 1 torch skip is fine).
- `python -m src.classification.regions <face.jpg>` renders a sane region overlay.
- `recommend(analyze(img, boxes, StubClassifier()), catalog)` round-trips: acne → actives, counted marks → texture advisory + SPF, no face → "couldn't analyse".
- Notebook 02 exists and its symbol references resolve; real weights land later via Colab (`models/type_classifier.pt`, gitignored).
