# Plan 002: Make configs/default.yaml the real source of pipeline defaults

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `git diff --stat 1ebd544..HEAD -- src/ configs/ tests/`
> Changes from plan 001 (EXIF fix in `run_acne04_pipeline.py`) are EXPECTED and
> not drift. Any other mismatch with the "Current state" excerpts is a STOP
> condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/001-fix-exif-orientation.md (shares a file; merged into your worktree)
- **Category**: tech-debt
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

`configs/default.yaml` opens with "thresholds live here, not hard-coded" — but
nothing in `src/` imports yaml, and every knob in the file is duplicated as an
argparse default in up to three scripts. Changing the detector operating point
(conf/iou/imgsz) or a weights path in the YAML silently does nothing; changing
it in one script leaves the others stale. This plan makes the YAML the single
source of defaults, with CLI flags still overriding.

## Current state

- `configs/default.yaml` — the config file nothing reads. Relevant keys:

```yaml
detection:
  weights: models/detection/acne04_yolov8m_best.pt
  img_size: 1024
  conf_threshold: 0.07
  iou_threshold: 0.2
classification:
  crop_pad: 1.5
  crop_size: 224
  weights: models/classification/acne_model.keras
  local_data: data/raw/typeclassification/AcneDataset
```

- `src/classification/run_acne04_pipeline.py:16-32` — `parse_args()` hardcodes
  the same values as argparse defaults (`--detector`, `--classifier`, `--conf`
  0.07, `--iou` 0.2, `--imgsz` 1024, `--crop-pad` 1.5).
- `src/detection/check_acne04_detector.py:12-24` — `parse_args()` duplicates
  `--weights`, `--conf` 0.07, `--iou` 0.2, `--imgsz` 1024.
- `src/classification/train_type_classifier.py:15-16` — duplicates the data
  dir and output weights path:

```python
DEFAULT_DATA = Path("data/raw/typeclassification/AcneDataset")
DEFAULT_OUT = Path("models/classification/acne_model.keras")
```

- `pyyaml` is already in `requirements.txt` and importable in the venv.
- There is no existing config-loading code anywhere in `src/`.

## Environment facts

- Fresh git worktree; `data/`, `models/`, `runs/`, `.venv/` absent. Do NOT
  `pip install` anything.
- Interpreter: `/Users/princekumar/Documents/skinscan/.venv/bin/python`
  (numpy, pillow, tensorflow, ultralytics, pyyaml, scikit-learn; NO pytest —
  tests are plain-python `__main__` runners, see `tests/test_pipeline_collage.py`).

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| New test | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_config.py` | prints `ok` |
| Existing tests | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_pipeline_collage.py` | prints `ok` |
| EXIF test (from plan 001) | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_load_rgb.py` | prints `ok` |
| Arg smoke | `/Users/princekumar/Documents/skinscan/.venv/bin/python -c "import sys; sys.argv=['x','--help']; ..."` (see Step 3) | help text shows YAML-derived defaults |

## Scope

**In scope**:
- `src/config.py` (create)
- `src/classification/run_acne04_pipeline.py` (argparse defaults only)
- `src/detection/check_acne04_detector.py` (argparse defaults only)
- `src/classification/train_type_classifier.py` (module constants only)
- `configs/default.yaml` (comment header update only)
- `tests/test_config.py` (create)

**Out of scope**:
- `src/recommendation/*` — `recommendation.concern_confidence_cutoff` stays
  unread for now; plan 004 wires it.
- Any behavior change: same defaults, same flags, same outputs. This is a
  plumbing change only.
- Do not add a `--config` CLI flag or config schema validation — YAGNI for a
  single-config learning repo.

## Git workflow

- Stay on the worktree's branch. Commit style: `feat: load pipeline defaults from configs/default.yaml`
- Do NOT push.

## Steps

### Step 1: Create the loader

Create `src/config.py`:

```python
"""Single source for pipeline knobs: configs/default.yaml (RULES.md §5)."""
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)
```

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python -c "from src.config import load_config; c = load_config(); print(c['detection']['conf_threshold'])"` → prints `0.07`

### Step 2: Point the three scripts at it

In each file, load the config once at module or `parse_args` level
(`from ..config import load_config` for the packages under `src/`; note
`run_acne04_pipeline.py` and `train_type_classifier.py` live in
`src/classification/` so the relative import is `from ..config import
load_config`; `check_acne04_detector.py` in `src/detection/` likewise) and
replace only the duplicated literals:

- `run_acne04_pipeline.py` `parse_args()`:
  - `--detector` default → `Path(cfg["detection"]["weights"])`
  - `--classifier` default → `Path(cfg["classification"]["weights"])`
  - `--conf` → `cfg["detection"]["conf_threshold"]`
  - `--iou` → `cfg["detection"]["iou_threshold"]`
  - `--imgsz` → `cfg["detection"]["img_size"]`
  - `--crop-pad` → `cfg["classification"]["crop_pad"]`
- `check_acne04_detector.py` `parse_args()`: `--weights`, `--conf`, `--iou`,
  `--imgsz` from the same keys.
- `train_type_classifier.py`: `DEFAULT_DATA = Path(cfg["classification"]["local_data"])`,
  `DEFAULT_OUT = Path(cfg["classification"]["weights"])`, and the
  `--image-size` default → `cfg["classification"]["crop_size"]`.

Keep every CLI flag; flags still override. Do not change `--images`, `--out`,
`--limit`, or any other default not listed above.

**Verify**:
`/Users/princekumar/Documents/skinscan/.venv/bin/python -c "import sys; sys.argv = ['prog', '--help']" ; true` — instead run each help:
- `/Users/princekumar/Documents/skinscan/.venv/bin/python -m src.classification.run_acne04_pipeline --help` → exits 0, help shows `default: 0.07`-style values (argparse prints them if you use `argparse.ArgumentDefaultsHelpFormatter`; if the current parser doesn't, just confirm exit 0 — do NOT add the formatter, out of scope)
- `/Users/princekumar/Documents/skinscan/.venv/bin/python -m src.detection.check_acne04_detector --help` → exit 0
- `/Users/princekumar/Documents/skinscan/.venv/bin/python -m src.classification.train_type_classifier --help` → exit 0

### Step 3: Update the YAML header comment

The header currently reads `# SkinScan config — thresholds live here, not
hard-coded (RULES.md §5, §4)`. It is now true; extend it with one line naming
the loader: `# Loaded by src/config.py; CLI flags override.` No key changes.

**Verify**: `grep -n "src/config.py" configs/default.yaml` → 1 match

### Step 4: Test

Create `tests/test_config.py` (pattern: `tests/test_pipeline_collage.py`,
same sys.path insert + `__main__` runner):

- `test_load_config_has_pipeline_keys`: `load_config()` returns a dict where
  `cfg["detection"]["conf_threshold"]` is a float between 0 and 1,
  `cfg["detection"]["iou_threshold"]` is a float,
  `cfg["detection"]["img_size"]` is an int,
  `cfg["classification"]["crop_pad"]` is a float,
  and both `weights` values end in `.pt` / `.keras` respectively.
- `test_cli_defaults_come_from_config`: import
  `src.classification.run_acne04_pipeline` and, with `sys.argv = ["prog"]`,
  call `parse_args()`; assert `args.conf == cfg["detection"]["conf_threshold"]`
  and `args.imgsz == cfg["detection"]["img_size"]`.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_config.py` → `ok`

## Test plan

- `tests/test_config.py` as specified in Step 4 (2 tests).
- Regression: `tests/test_pipeline_collage.py` and `tests/test_load_rgb.py`
  still print `ok`.

## Done criteria

- [ ] `grep -rn "0.07" src/ --include="*.py"` → no matches (the literal now lives only in the YAML)
- [ ] `grep -rn "acne04_yolov8m_best" src/ --include="*.py"` → no matches
- [ ] `grep -rn "typeclassification" src/ --include="*.py"` → no matches
- [ ] All three `--help` invocations exit 0
- [ ] `tests/test_config.py`, `tests/test_pipeline_collage.py`, `tests/test_load_rgb.py` all print `ok`
- [ ] `git status --porcelain` clean outside the in-scope list

## STOP conditions

- `parse_args` in any script doesn't match the "Current state" description
  (beyond plan 001's EXIF change).
- Importing `src.config` from a script triggers a circular import.
- `tests/test_load_rgb.py` is absent (means plan 001 was not merged into your
  worktree — report, don't recreate it).

## Maintenance notes

- Future knobs go in the YAML first, then argparse reads them — a reviewer
  should reject any new hardcoded threshold in `src/`.
- Plan 004 adds `concern_report` keys to this YAML and wires
  `recommendation.concern_confidence_cutoff`.
- Deliberately skipped: config schema validation, multiple config files,
  `--config` flag — add only when a second config appears.
