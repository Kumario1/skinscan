# Plan 001: Fix EXIF orientation mismatch between detector boxes and crop pixels

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `git diff --stat 1ebd544..HEAD -- src/classification/run_acne04_pipeline.py tests/`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

`analyze_image` runs YOLO on the image **path** (Ultralytics decodes with
OpenCV, which applies the EXIF Orientation tag) but crops pixels from a numpy
array loaded with `PIL.Image.open`, which does NOT apply EXIF orientation. For
any photo with an EXIF Orientation tag — which is nearly every phone photo, and
this repo's own self-collected test set is phone photos (DECISIONS.md D-014) —
the detector's box coordinates live in the rotated coordinate space while the
pixel array is in raw storage space. Every crop is then cut from the wrong
place (rotated/mirrored position), and the classifier silently judges the wrong
pixels. This was verified empirically: the same JPEG with Orientation=6 loads
as shape `(100, 200, 3)` via PIL and `(200, 100, 3)` via cv2/Ultralytics.

## Current state

- `src/classification/run_acne04_pipeline.py` — the end-to-end
  detector→classifier pipeline. The bug is at line 96:

```python
# src/classification/run_acne04_pipeline.py:95-103
def analyze_image(img_path, model, clf, out_dir, *, crop_size, crop_pad, max_boxes, conf, iou, imgsz, collage_tiles):
    image = np.asarray(Image.open(img_path).convert("RGB"))
    result = model.predict(
        str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )[0]
```

- Imports at the top of the file (line 11): `from PIL import Image, ImageDraw`
- Test convention in this repo: plain-python test files with an
  `if __name__ == "__main__":` runner that calls each test and prints `ok`
  — see `tests/test_pipeline_collage.py`. Tests insert the repo root on
  `sys.path`:

```python
# tests/test_pipeline_collage.py:8
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

## Environment facts

- You are in a fresh git worktree. `data/`, `models/`, `runs/`, `.venv/` are
  gitignored and therefore absent. Do NOT `pip install` anything.
- Use this interpreter for all commands:
  `/Users/princekumar/Documents/skinscan/.venv/bin/python`
  (has numpy, pillow, tensorflow, ultralytics, pyyaml, scikit-learn; pytest is
  NOT installed — run test files directly).

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Existing tests | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_pipeline_collage.py` | prints `ok`, exit 0 |
| New test | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_load_rgb.py` | prints `ok`, exit 0 |
| Syntax check | `/Users/princekumar/Documents/skinscan/.venv/bin/python -m py_compile src/classification/run_acne04_pipeline.py` | exit 0 |

## Scope

**In scope** (the only files you should modify):
- `src/classification/run_acne04_pipeline.py`
- `tests/test_load_rgb.py` (create)

**Out of scope** (do NOT touch):
- `src/classification/classifier.py` — `crop_with_context` is
  coordinate-space-agnostic; it is not the bug.
- Feeding the numpy array to `model.predict` instead of the path — Ultralytics
  treats raw arrays as BGR, so passing an RGB array would silently flip
  channels. Keep passing the path to YOLO; fix the PIL side.

## Git workflow

- Stay on the branch the worktree was created on; commit there.
- Commit message style (match `git log`): lowercase conventional prefix, e.g.
  `fix: apply exif orientation before cropping pipeline images`
- Do NOT push or open a PR.

## Steps

### Step 1: Add an EXIF-aware loader and use it

In `src/classification/run_acne04_pipeline.py`:

1. Change the PIL import (line 11) to include `ImageOps`:
   `from PIL import Image, ImageDraw, ImageOps`
2. Add a module-level helper directly above `analyze_image`:

```python
def load_rgb(path):
    """EXIF-corrected pixels — must match the orientation YOLO sees (cv2 applies EXIF)."""
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
```

3. In `analyze_image`, replace
   `image = np.asarray(Image.open(img_path).convert("RGB"))` with
   `image = load_rgb(img_path)`.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python -m py_compile src/classification/run_acne04_pipeline.py` → exit 0

### Step 2: Write the regression test

Create `tests/test_load_rgb.py` following the structure of
`tests/test_pipeline_collage.py` (same `sys.path` insert, same `__main__`
runner). One test:

- Build a 200×100 landscape RGB image, set EXIF tag 274 (Orientation) to 6,
  save as JPEG into a `tempfile.TemporaryDirectory`:

```python
img = Image.new("RGB", (200, 100), (255, 0, 0))
exif = img.getexif()
exif[274] = 6
img.save(path, exif=exif)
```

- Assert `load_rgb(path).shape == (200, 100, 3)` (orientation applied: the
  100-high landscape becomes 200-high portrait).
- Also assert a plain `np.asarray(Image.open(path)).shape == (100, 200, 3)`
  in the same test — this documents WHY the helper exists; if Pillow ever
  starts auto-applying EXIF, this line failing tells a maintainer the helper
  is now redundant.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_load_rgb.py` → prints `ok`

### Step 3: Confirm nothing else regressed

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_pipeline_collage.py` → prints `ok`

## Test plan

- New: `tests/test_load_rgb.py` — EXIF Orientation=6 JPEG loads transposed
  (the regression this plan fixes); raw PIL load shown untransposed.
- Pattern: `tests/test_pipeline_collage.py`.
- Verification: both test files print `ok`.

## Done criteria

ALL must hold:

- [ ] `grep -n "exif_transpose" src/classification/run_acne04_pipeline.py` → 1+ match
- [ ] `grep -n "np.asarray(Image.open" src/classification/run_acne04_pipeline.py` → no matches
- [ ] `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_load_rgb.py` → `ok`
- [ ] `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_pipeline_collage.py` → `ok`
- [ ] `git status --porcelain` shows no modified files outside the in-scope list

## STOP conditions

Stop and report back (do not improvise) if:

- Line 96 of `run_acne04_pipeline.py` doesn't match the "Current state" excerpt.
- The new test fails after the fix (would mean Pillow in this venv behaves
  differently than assumed — report the observed shapes).
- The fix appears to require touching `classifier.py` or Ultralytics internals.

## Maintenance notes

- If anyone later switches `model.predict` to take the in-memory array instead
  of the path, they must (a) convert RGB→BGR and (b) keep the EXIF transpose —
  the helper's docstring records this coupling.
- Reviewer should scrutinize: the test asserts both the fixed and the raw
  behavior, not just "no crash".
