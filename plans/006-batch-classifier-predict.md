# Plan 006: Classify all crops of an image in one batched predict call

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `git diff --stat 1ebd544..HEAD -- src/classification/ tests/`
> Expected prior changes: 001 (load_rgb in run_acne04_pipeline.py), 002
> (argparse defaults), 004 (classifier.py vocab). The `predict()` method and
> the `analyze_image` box loop must still match the excerpts below; otherwise
> STOP.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/004-bridge-readiness-unify-vocab.md (same file; merged into your worktree)
- **Category**: perf
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

`analyze_image` calls `clf.predict(crop)` once per detector box — up to 16
separate TensorFlow `model.predict()` invocations per image, each with batch
size 1. Per-call graph/dispatch overhead dominates at this size; batching all
crops into one call removes ~15/16ths of that overhead per image and is the
obvious shape for any future serving path.

## Current state

- `src/classification/classifier.py` — `AcneTypeClassifier.predict` (post-004
  the file also has `concern_probs`; `predict` itself is unchanged from this
  excerpt):

```python
    def predict(self, crop):
        from PIL import Image
        from tensorflow.keras.applications.efficientnet import preprocess_input

        crop = np.asarray(crop)
        if crop.shape[:2] != (self.image_size, self.image_size):
            crop = np.asarray(Image.fromarray(crop).resize((self.image_size, self.image_size), Image.BILINEAR))
        x = preprocess_input(crop.astype(np.float32)[None, ...])
        probs = self.model.predict(x, verbose=0)[0]
        return dict(zip(self.classes, probs.astype(float).tolist()))
```

- `src/classification/classifier.py` — `StubClassifier` at the bottom has only
  `predict`.
- `src/classification/run_acne04_pipeline.py` — `analyze_image` box loop
  (line numbers pre-001; content unchanged by 001/002):

```python
    for box in result.boxes[:max_boxes]:
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        crop = crop_with_context(image, (x0, y0, x1 - x0, y1 - y0), pad=crop_pad, size=crop_size)
        crop_path = out_dir / f"{img_path.stem}_crop_{len(detections) + 1:02d}.jpg"
        Image.fromarray(crop).save(crop_path, quality=92)
        record = {
            "box": [x0, y0, x1, y1],
            "detector_conf": float(box.conf),
            "input_crop": str(crop_path),
        }
        if clf:
            probs = clf.predict(crop)
            label, prob = max(probs.items(), key=lambda kv: kv[1])
            sheet_items.append((crop, f"{label} {prob:.2f}"))
            record.update({"prediction": label, "probability": prob, "probs": probs})
        else:
            sheet_items.append((crop, f"conf {float(box.conf):.2f}"))
        detections.append(record)
```

- `classifier.py` has an `if __name__ == "__main__":` self-check block that
  must keep passing.

## Environment facts

- Fresh git worktree; `models/` absent → you cannot load the real Keras model.
  All required verification is model-free; TF itself IS importable.
- Interpreter: `/Users/princekumar/Documents/skinscan/.venv/bin/python`
  (no pytest — `__main__` runner convention).

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Classifier self-check | `/Users/princekumar/Documents/skinscan/.venv/bin/python src/classification/classifier.py` | prints `ok` |
| New test | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_predict_batch.py` | prints `ok` |
| Regression | run every file in `tests/` | each prints `ok` |

## Scope

**In scope**:
- `src/classification/classifier.py`
- `src/classification/run_acne04_pipeline.py` (the `analyze_image` loop only)
- `tests/test_predict_batch.py` (create)

**Out of scope**:
- Batching across images (one image's crops per call is enough).
- The detector call, collage/sheet rendering, JSON output shape —
  `predictions.json` must come out byte-identical in structure (same keys,
  same per-detection records, same ordering).

## Git workflow

- Stay on the worktree's branch. Commit style:
  `perf: batch crop classification into one predict call`
- Do NOT push.

## Steps

### Step 1: Add predict_batch, make predict delegate

In `AcneTypeClassifier`, factor the resize+preprocess into a helper and add a
batched path:

```python
    def _prepare(self, crop):
        from PIL import Image

        crop = np.asarray(crop)
        if crop.shape[:2] != (self.image_size, self.image_size):
            crop = np.asarray(Image.fromarray(crop).resize((self.image_size, self.image_size), Image.BILINEAR))
        return crop.astype(np.float32)

    def predict_batch(self, crops):
        from tensorflow.keras.applications.efficientnet import preprocess_input

        if not len(crops):
            return []
        x = preprocess_input(np.stack([self._prepare(c) for c in crops]))
        probs = self.model.predict(x, verbose=0)
        return [dict(zip(self.classes, p.astype(float).tolist())) for p in probs]

    def predict(self, crop):
        return self.predict_batch([crop])[0]
```

Give `StubClassifier` the same interface:

```python
    def predict_batch(self, crops):
        return [dict(zip(RAW_ACNE_CLASSES, self.probs.tolist())) for _ in crops]

    def predict(self, crop):
        return self.predict_batch([crop])[0]
```

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python src/classification/classifier.py` → `ok`

### Step 2: Restructure the analyze_image loop

Two passes: first loop over boxes to build crops + records (everything except
classification), then one `clf.predict_batch(crops)` call, then zip the
results back onto the records and sheet labels. Preserve exactly:

- per-record key order/content (`box`, `detector_conf`, `input_crop`, then
  `prediction`/`probability`/`probs` when clf is set),
- crop file naming (`_crop_01.jpg` numbering),
- sheet label text (`f"{label} {prob:.2f}"` with clf, `f"conf {...:.2f}"` without),
- the no-classifier (`clf is None`) path.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python -m py_compile src/classification/run_acne04_pipeline.py` → exit 0

### Step 3: Test

Create `tests/test_predict_batch.py` (convention: `tests/test_pipeline_collage.py`):

1. `test_stub_batch_matches_single` — `s = StubClassifier()`;
   `s.predict_batch([crop, crop])` returns 2 identical dicts equal to
   `s.predict(crop)`.
2. `test_prepare_resizes_and_casts` — build a real `AcneTypeClassifier`
   WITHOUT calling `__init__` (avoid model load):
   `clf = AcneTypeClassifier.__new__(AcneTypeClassifier); clf.image_size = 224`
   then `clf._prepare(np.zeros((100, 50, 3), np.uint8))` has shape
   `(224, 224, 3)` and dtype `float32`; a `(224, 224, 3)` input comes back
   same-shape without resizing.
3. `test_predict_batch_empty` — same `__new__` trick plus
   `clf.classes = [...]`; `clf.predict_batch([])` returns `[]` without
   touching `clf.model` (no model attribute set — would raise if touched).
4. `test_predict_batch_uses_one_model_call` — same `__new__` trick; set
   `clf.model` to a tiny fake with a `predict(x, verbose=0)` method that
   records call count and returns `np.tile([0.2, 0.2, 0.2, 0.2, 0.2], (len(x), 1))`;
   `clf.classes = RAW_ACNE_CLASSES`; call `predict_batch([crop1, crop2, crop3])`
   → 3 dicts, fake called exactly once with `x.shape[0] == 3`.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_predict_batch.py` → `ok`

## Test plan

The 4 tests above; classifier `__main__` self-check; all existing `tests/`
files still `ok`.

## Done criteria

- [ ] `grep -n "predict_batch" src/classification/classifier.py` → 2+ matches
- [ ] `grep -n "clf.predict(crop)" src/classification/run_acne04_pipeline.py` → no matches (loop now batches)
- [ ] `tests/test_predict_batch.py` prints `ok`
- [ ] `src/classification/classifier.py` self-check prints `ok`
- [ ] All other `tests/` files print `ok`
- [ ] `git status --porcelain` clean outside the in-scope list

## STOP conditions

- `predict()` or the box loop doesn't match the excerpts (beyond documented
  001/002/004 changes).
- Preserving the exact `predictions.json` record shape appears impossible
  with batching (it isn't — report what you hit).

## Maintenance notes

- Plan 010's future negative class changes `self.classes` length; nothing
  here assumes 5 classes.
- Reviewer: confirm records still pair the RIGHT probs with the RIGHT crop
  (ordering is by box index in both passes) — that's the one real risk in the
  restructure, and test 4's shape assertion doesn't cover cross-pairing; read
  the zip carefully.
