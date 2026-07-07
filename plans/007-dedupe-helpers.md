# Plan 007: Deduplicate the three copy-pasted helpers

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `git diff --stat 1ebd544..HEAD -- src/ tests/`
> Expected prior changes: 001, 002, 004, 005, 006 (all listed in
> plans/README.md). The specific duplicated snippets below must still exist;
> otherwise STOP.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/006-batch-classifier-predict.md (same files; merged into your worktree)
- **Category**: tech-debt
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

Three helpers exist in two copies each; the copies have already started
drifting (different signatures for the same job). Every future fix must be
made twice or silently diverges:

1. `require(path, label)` — identical in `run_acne04_pipeline.py:35` and
   `check_acne04_detector.py:27`.
2. VOC XML box parsing — `check_acne04_detector.py:36` (`gt_boxes`) re-implements
   `voc_to_yolo.py:60` (`parse_voc_xml`).
3. Model labels-metadata reading — `run_acne04_pipeline.py:51`
   (`classifier_image_size`) re-implements the metadata read inside
   `AcneTypeClassifier.__init__` (`classifier.py:70-74`).

## Current state

- `src/classification/run_acne04_pipeline.py:35-37`:

```python
def require(path, label):
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")
```

  Same function verbatim at `src/detection/check_acne04_detector.py:27-29`.

- `src/detection/check_acne04_detector.py:36-43`:

```python
def gt_boxes(root, stem):
    xml = ET.parse(root / f"{stem}.xml").getroot()
    boxes = []
    for obj in xml.findall("object"):
        b = obj.find("bndbox")
        boxes.append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    return boxes
```

- `src/detection/voc_to_yolo.py:60-73` — `parse_voc_xml(xml_path)` returns
  `(w, h, list[Box])` where `Box` is a dataclass with
  `xmin/ymin/xmax/ymax` floats.

- `src/classification/run_acne04_pipeline.py:51-55`:

```python
def classifier_image_size(model_path, fallback=224):
    meta = Path(model_path).with_suffix(Path(model_path).suffix + ".labels.json")
    if not meta.exists():
        return fallback
    return int(json.loads(meta.read_text()).get("image_size", fallback))
```

- `src/classification/classifier.py:70-74` (inside `__init__`):

```python
        model_path = Path(model_path)
        meta = model_path.with_suffix(model_path.suffix + ".labels.json")
        metadata = json.loads(meta.read_text()) if meta.exists() else {}
```

- Existing test that pins `classifier_image_size` behavior:
  `tests/test_pipeline_collage.py::test_classifier_image_size_reads_metadata`
  — it must keep passing (keep the function as a thin wrapper; don't move the
  test).

## Environment facts

- Fresh git worktree; interpreter
  `/Users/princekumar/Documents/skinscan/.venv/bin/python`; no pytest —
  `__main__` runner convention.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Regression | run every file in `tests/` with the interpreter | each prints `ok` |
| Self-check | `/Users/princekumar/Documents/skinscan/.venv/bin/python src/classification/classifier.py` | prints `ok` |
| Syntax | `py_compile` both scripts | exit 0 |

## Scope

**In scope**:
- `src/utils.py` (create — `require` only)
- `src/classification/run_acne04_pipeline.py`
- `src/detection/check_acne04_detector.py`
- `src/classification/classifier.py` (add `read_model_metadata`)

**Out of scope**:
- `voc_to_yolo.py` — it is the canonical VOC parser; no changes there.
- Any behavior change; identical CLI output on identical input.
- Consolidating `draw_sheet` vs `render_sheet` — they render different things
  (crop grids vs GT-vs-pred sheets); merging them would couple two scripts for
  ~10 saved lines. Deliberately skipped.

## Git workflow

- Stay on the worktree's branch. Commit style:
  `refactor: single-source require, voc parsing, and labels metadata`
- Do NOT push.

## Steps

### Step 1: `require` → `src/utils.py`

Create `src/utils.py` containing exactly the `require` function (with a
one-line module docstring). Replace both copies with
`from ..utils import require` (both files are one package level below `src/`).
Delete both local definitions.

**Verify**: both `py_compile` commands exit 0.

### Step 2: `gt_boxes` reuses the canonical parser

In `check_acne04_detector.py`, replace the body of `gt_boxes` (keep the
function and its return shape — a list of `[xmin, ymin, xmax, ymax]` lists):

```python
from .voc_to_yolo import parse_voc_xml

def gt_boxes(root, stem):
    _, _, boxes = parse_voc_xml(str(root / f"{stem}.xml"))
    return [[b.xmin, b.ymin, b.xmax, b.ymax] for b in boxes]
```

Remove the now-unused `import xml.etree.ElementTree as ET` if nothing else in
the file uses it (check first).

**Verify**: `py_compile` exit 0; then a functional check without data — create
a minimal VOC XML in a temp dir and confirm both old shape and values:

```
/Users/princekumar/Documents/skinscan/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import tempfile, pathlib
from src.detection.check_acne04_detector import gt_boxes
xml = '<annotation><size><width>100</width><height>80</height><depth>3</depth></size><object><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object></annotation>'
d = tempfile.mkdtemp(); p = pathlib.Path(d)
(p / 'x.xml').write_text(xml)
print(gt_boxes(p, 'x'))
"
```
→ prints `[[1.0, 2.0, 30.0, 40.0]]`

### Step 3: Single metadata reader

In `classifier.py`, add a module-level function and use it in `__init__`:

```python
def read_model_metadata(model_path):
    model_path = Path(model_path)
    meta = model_path.with_suffix(model_path.suffix + ".labels.json")
    return json.loads(meta.read_text()) if meta.exists() else {}
```

In `run_acne04_pipeline.py`, reimplement `classifier_image_size` as a thin
wrapper (keeps its existing test green):

```python
def classifier_image_size(model_path, fallback=224):
    return int(read_model_metadata(model_path).get("image_size", fallback))
```

with the import added to the existing `from .classifier import ...` line.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_pipeline_collage.py` → `ok`

## Test plan

No new test files — this is a pure consolidation covered by existing tests:
`tests/test_pipeline_collage.py` (pins `classifier_image_size`), the Step 2
inline functional check, `classifier.py` self-check, and the rest of `tests/`.

## Done criteria

- [ ] `grep -rn "def require" src/ | wc -l` → exactly 1 (in `src/utils.py`)
- [ ] `grep -n "ET.parse" src/detection/check_acne04_detector.py` → no matches
- [ ] `grep -c "labels.json" src/classification/run_acne04_pipeline.py` → 0
- [ ] Step 2's inline check prints `[[1.0, 2.0, 30.0, 40.0]]`
- [ ] Every file in `tests/` prints `ok`; classifier self-check prints `ok`
- [ ] `git status --porcelain` clean outside the in-scope list

## STOP conditions

- Any excerpt above no longer matches (beyond documented prior-plan changes).
- The relative imports fail (e.g. scripts are run as plain files somewhere —
  they are documented and tested only as `-m` modules; if you find a plain-file
  invocation path in the repo, report it).

## Maintenance notes

- `src/utils.py` is for genuinely shared 5-line helpers only — a reviewer
  should push back if it starts collecting logic.
- Deliberately skipped: merging the two sheet renderers (different outputs,
  low payoff).
