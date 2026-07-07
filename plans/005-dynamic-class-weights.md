# Plan 005: Compute training class weights from the dataset, not hardcoded counts

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `git diff --stat 1ebd544..HEAD -- src/classification/train_type_classifier.py tests/`
> Plan 002 changed this file's `DEFAULT_DATA`/`DEFAULT_OUT`/`--image-size` to
> read from config — EXPECTED. The `class_weights()` function must still match
> the excerpt below; otherwise STOP.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/002-make-config-real.md (same file; merged into your worktree)
- **Category**: bug
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

`class_weights()` in the trainer hardcodes per-class image counts
`[735, 645, 621, 584, 193]`. The `--data` directory is a parameter and the
dataset has changed repeatedly (git history shows multiple harvest rounds).
The moment the data changes, the class weights are silently wrong — the model
still trains, converges, and reports plausible metrics, so nobody notices the
imbalance correction no longer matches reality.

## Current state

- `src/classification/train_type_classifier.py:30-43` — `count_split(root, split)`
  already counts images per class directory:

```python
def count_split(root, split):
    rows = []
    for d in sorted((root / split).iterdir()):
        if d.is_dir():
            rows.append((d.name, sum(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for p in d.iterdir())))
    return rows
```

- `src/classification/train_type_classifier.py:75-82` — the bug:

```python
def class_weights():
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    counts = [735, 645, 621, 584, 193]
    classes = np.array([0, 1, 2, 3, 4])
    y = np.repeat(classes, counts)
    return dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y)))
```

- Call site, `main()` (~line 118): `class_weight=class_weights(),`
- Label indices come from `tf.keras.preprocessing.image_dataset_from_directory`,
  which assigns them in **sorted directory-name order** — the same order
  `count_split` returns (it sorts `iterdir()`). So `count_split`'s row order
  maps 1:1 onto label indices.

## Environment facts

- Fresh git worktree; `data/` absent (gitignored) — the real dataset is NOT
  available; tests must use a temp-dir fixture.
- Interpreter: `/Users/princekumar/Documents/skinscan/.venv/bin/python`
  (numpy + scikit-learn available; no pytest — `__main__` runner convention,
  see `tests/test_pipeline_collage.py`).

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| New test | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_class_weights.py` | prints `ok` |
| Syntax | `/Users/princekumar/Documents/skinscan/.venv/bin/python -m py_compile src/classification/train_type_classifier.py` | exit 0 |
| Regression | run every file in `tests/` | each prints `ok` |

## Scope

**In scope**:
- `src/classification/train_type_classifier.py` (`class_weights` + its call site)
- `tests/test_class_weights.py` (create)

**Out of scope**:
- Any other part of the training script (datasets, model, callbacks, metadata).
- Actually running training (no data, no GPU here).

## Git workflow

- Stay on the worktree's branch. Commit style:
  `fix: derive class weights from the training directory`
- Do NOT push.

## Steps

### Step 1: Make class_weights read the data

Replace the function with one that derives counts from the train split via the
existing `count_split`:

```python
def class_weights(root):
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    counts = [n for _, n in count_split(root, "train")]
    if not counts or any(n == 0 for n in counts):
        raise SystemExit(f"empty class directory under {root}/train")
    classes = np.arange(len(counts))
    y = np.repeat(classes, counts)
    return dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y)))
```

Update the call site in `main()` to `class_weight=class_weights(args.data),`.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python -m py_compile src/classification/train_type_classifier.py` → exit 0

### Step 2: Test with a fixture

Create `tests/test_class_weights.py` (convention: `tests/test_pipeline_collage.py`).
Build a fixture in a `tempfile.TemporaryDirectory`: `train/A/` with 6 empty
`.jpg` files, `train/B/` with 2, `train/C/` with 4 (create with
`Path(...).touch()` — `count_split` only checks suffixes, never opens files).

1. `test_weights_inverse_to_counts` — `w = class_weights(Path(d))` returns
   keys `{0, 1, 2}`; `w[1] > w[2] > w[0]` (rarer class → larger weight); and
   `w[0] == pytest`-free exact check: `compute_class_weight("balanced", ...)`
   for counts `[6, 2, 4]` gives `total/(k*count)` = `12/(3*6), 12/(3*2), 12/(3*4)`
   → assert `abs(w[0] - 12/18) < 1e-9`, `abs(w[1] - 12/6) < 1e-9`,
   `abs(w[2] - 12/12) < 1e-9`.
2. `test_empty_class_dir_exits` — add empty dir `train/D/` with no images →
   `class_weights` raises `SystemExit`.
3. `test_non_image_files_ignored` — drop a `notes.txt` into `train/A/` →
   weights unchanged.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_class_weights.py` → `ok`

## Test plan

The 3 tests above; plus every existing `tests/` file still prints `ok`.

## Done criteria

- [ ] `grep -n "735" src/classification/train_type_classifier.py` → no matches
- [ ] `grep -n "class_weights(args.data)" src/classification/train_type_classifier.py` → 1 match
- [ ] `tests/test_class_weights.py` prints `ok`
- [ ] All other `tests/` files print `ok`
- [ ] `git status --porcelain` clean outside the in-scope list

## STOP conditions

- `class_weights()` or `count_split()` doesn't match the excerpts above.
- The importable path for the test breaks because `train_type_classifier`
  imports TF at module level (it doesn't at planning time — imports are inside
  functions; if that changed, report).

## Maintenance notes

- If a `not_acne` class is added later (plan 010's design), this now picks it
  up automatically — that's the point.
- Reviewer: check the test asserts exact balanced-weight values, not just
  "returns a dict".
