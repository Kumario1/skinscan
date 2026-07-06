# Stage 2 Type-Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status (2026-07-05):** Tasks 1–3 done. Task 4 done: 431 crops harvested from 120 detector-val faces at the locked operating point. Task 5/6: 192 hand-sorted — comedonal 54, inflammatory 51, post_acne_mark 47, cystic 24, not_acne 16 (+cell-4 negatives pending); 239 unsorted remain as top-up supply. **Decision: post_acne_mark ≥ 30 → GO, no collapse.** Task 7/8 v0 baseline (CPU, 162 train / 70 val): best macro-recall **0.491** @ epoch 8 — not_acne .79, comedonal .61, mark .53, inflammatory **.14** (bleeds into mark), cystic **.11** (15 train crops). v1 round: sort remaining 239 + severe-face harvest for cystic + oversampling/cosine/40 epochs (commit 7795daf). **v1 result (1290 train / 229 val, kaggle pretrain loaded): best-ckpt macro-recall 0.484 @ ep18, acc 0.54** — not_acne .70, mark .59, inflammatory .54, comedonal .32 (starved: 94 train vs mark's 636), cystic .27. v2 round: cell 2c mild/train-split harvest (comedonal+cystic supply) + inflammatory↔mark audit cell (drift vs ceiling) — commit b04e0d6. **v2 result: transfusion BACKFIRED.** 627 cystic crops (~8:1 kaggle:ACNE04) → cystic recall regressed 0.27→**0.09** (24/33 cystic→inflammatory); sampler balances classes not sub-domains, so "cystic" = "kaggle-cyst" and real ACNE04 cysts read inflammatory. Macro 0.44, acc 0.48. **Reverting kg_ crops** (sidelined to crops_kaggle_holdout) back to v1 baseline. **Key finding across v0–v2: macro stuck 0.44–0.49 across oversampling/pretrain/transfusion → data-or-task ceiling, not recipe.** The two failing classes (comedonal .27, cystic .09) are the *size*-defined ones, and crop_with_context (1.5× pad → resize 112) normalizes absolute size away → architectural ceiling. Fork: (A) ship v1-as-v0 with confidence-gated caveats (spec's concern_confidence_cutoff) + move to regions/assemble; (B) size-signal re-harvest (fixed-pixel window). Leaning A.

**Goal:** Ship the Stage 2 five-class lesion-crop classifier: curated self-labeled crop set, runnable Colab training notebook (`.ipynb`), trained weights that round-trip through `LesionClassifier`, and the §6 eval.

**Architecture:** All pure logic (class list, crop extraction, inference wrapper, stub) lives in `src/classification/classifier.py` and runs locally with no GPU/weights. GPU + labeling work runs in Colab from `notebooks/02_type_classifier.ipynb`, generated from the markdown curriculum so there is a single source of truth. Training data = Stage 1 detector crops from ACNE04, hand-sorted; self-collected phone photos are test-only.

**Tech Stack:** Python, numpy, Pillow, PyTorch + torchvision (MobileNetV3-small), ultralytics (harvest only), scikit-learn (metrics), nbformat (notebook generation), Colab T4.

## Global Constraints

- `CLASSES = ["comedonal", "cystic", "inflammatory", "not_acne", "post_acne_mark"]` — alphabetical because torchvision `ImageFolder` sorts class dirs; this order IS the label order. Never reorder.
- Concern schema is locked (D-008): classifier classes map onto existing concerns; no schema change.
- Fine-tune train data = ACNE04 detector crops only. Self-collected phone-photo crops are TEST/VAL ONLY, never train (D-014).
- Crop knobs: `crop_pad: 1.5`, `crop_size: 112` (`configs/default.yaml` → `classification:`).
- Weights path: `models/type_classifier_v0.pt` (config `classification.weights`).
- Augmentation must NOT jitter hue/saturation (red-vs-brown separates `inflammatory` from `post_acne_mark`).
- Train/val split by source-image stem: `int(md5(stem)) % 5 == 0` → val. Never split by crop.
- Model selection metric: val **macro recall** (imbalance-honest), not accuracy.
- Eval per §6: confusion matrix, per-class precision/recall; Fitzpatrick-disaggregated where the set supports it, honest note if not (D-016).
- Missing weights at inference = hard error (spec §7); tests use `StubClassifier`.
- Thin-class fallback (spec decision 3): if `post_acne_mark` < ~30 crops, move its crops into `not_acne/`, keep the empty dir → 5-class head simply never predicts it.

---

### Task 1: Verify the classifier module (already written)

**Files:**
- Exists: `src/classification/classifier.py`
- Exists: `src/classification/__init__.py`
- Exists: `configs/default.yaml` (classification block)

**Interfaces:**
- Produces: `CLASSES: list[str]` (order above); `crop_with_context(image: np.ndarray HxWx3 uint8, box: (x,y,w,h[,conf]), pad=1.5, size=112) -> np.ndarray size×size×3 uint8`; `build_net(pretrained=False) -> torchvision MobileNetV3-small with 5-class head`; `LesionClassifier(weights_path, device="cpu").predict(crop) -> dict[class, prob]`; `StubClassifier(probs=None).predict(crop) -> dict[class, prob]`.

- [ ] **Step 1: Run the module self-check**

Run: `uv run --quiet --with numpy --with pillow python src/classification/classifier.py`
Expected: `ok`

- [ ] **Step 2: Commit the module + config**

```bash
git add src/classification/ configs/default.yaml
git commit -m "feat: stage 2 lesion classifier module (crop, wrapper, stub)"
```

### Task 2: Harden the curriculum cells

Four small fixes to `notebooks/02_type_classifier.md` before it becomes the notebook source.

**Files:**
- Modify: `notebooks/02_type_classifier.md`

**Interfaces:**
- Produces: the exact cell sources Task 3 converts to `.ipynb`.

- [ ] **Step 1: Cell 2 markdown — fresh-session note.** After the "Prefer harvesting…" paragraph add:

```markdown
Fresh session? Re-stage the ACNE04 images first (notebook 01 cells 2–4), or
point `imgs` at a Drive copy of them.
```

- [ ] **Step 2: Cell 5 — collapse mechanics.** Append to the decision paragraph:

```markdown
Collapse mechanics: move the crops into `not_acne/` and **leave the empty
`post_acne_mark/` dir in place** — `ImageFolder` keeps the 5-class order, the
head stays 5-class, and the class simply never gets predicted (the intended
v1 behavior). No code or schema changes.
```

- [ ] **Step 3: Cell 7 — weight guard for an empty (collapsed) class.** Replace the `weight =` line with:

```python
weight = torch.tensor([len(train_idx) / (len(CLASSES) * max(counts[i], 1))
                       for i in range(len(CLASSES))])   # max(...,1): survives an empty class
```

- [ ] **Step 4: Cell 9 — small-val guards.** Replace the sample/loop opening with:

```python
for ax in axes.flat: ax.axis("off")          # blanks stay blank if val < 32
sample = random.sample(val_idx, min(32, len(val_idx)))
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_type_classifier.md
git commit -m "docs: harden notebook 02 cells (fresh session, collapse, guards)"
```

### Task 3: Generate + validate the runnable notebook

Convert the md curriculum to `notebooks/02_type_classifier.ipynb` (markdown between ```` ```python ```` fences → markdown cells, fence bodies → code cells). Generated, not hand-written JSON.

**Files:**
- Create: `notebooks/02_type_classifier.ipynb`
- Create (scratchpad, throwaway): `build_ipynb.py`

**Interfaces:**
- Consumes: `notebooks/02_type_classifier.md` (Task 2 final state).
- Produces: valid nbformat-v4 notebook, Colab GPU metadata, 9 code cells.

- [ ] **Step 1: Write the builder script** (scratchpad, not committed):

```python
# build_ipynb.py — one-shot md -> ipynb
import re
import nbformat as nbf

md = open("notebooks/02_type_classifier.md").read()
parts = re.split(r"```python\n(.*?)```", md, flags=re.S)  # even idx = md, odd = code
nb = nbf.v4.new_notebook()
nb.metadata.update({
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
})
for i, part in enumerate(parts):
    part = part.strip("\n")
    if not part.strip():
        continue
    nb.cells.append(nbf.v4.new_markdown_cell(part) if i % 2 == 0
                    else nbf.v4.new_code_cell(part))
nbf.validate(nb)
code = sum(c.cell_type == "code" for c in nb.cells)
assert code >= 10, f"suspiciously few code cells: {code}"   # floor, not exact — cells get added as the curriculum grows
assert any("from src.classification.classifier import CLASSES" in c.source
           for c in nb.cells if c.cell_type == "code"), "cell 1 import missing"
nbf.write(nb, "notebooks/02_type_classifier.ipynb")
print(f"ok: {len(nb.cells)} cells ({code} code)")
```

- [ ] **Step 2: Run it**

Run: `uv run --quiet --with nbformat python <scratchpad>/build_ipynb.py` (from repo root)
Expected: `ok: N cells (9 code)` and no validation error.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_type_classifier.ipynb
git commit -m "feat: runnable Colab notebook for stage 2 type classifier"
```

Optional (user's call — mirrors their deletion of `01_acne04_detector.md`): drop the md once the ipynb is the working artifact.

---

The remaining tasks run in **Colab (T4) + human labeling time** — they execute the notebook, cell numbers refer to it. Each has an acceptance gate; stop at a failed gate, don't train on a bad set.

### Task 4: Session setup + crop harvest (notebook cells 1–2)

**Consumes:** Stage 1 weights (`acne_y8n_v0` best.pt), ACNE04 images. **Produces:** `Drive/skinscan_stage2/crops_unsorted/*.png`, filenames `{imgstem}_{k}_{conf:.2f}.png`.

- [ ] Mount Drive, clone/upload repo, run cell 1. Expected print: the 5 classes in alphabetical order.
- [ ] Adjust the two paths in cell 2 (weights, images), run. Expected: `N crops` with N ≥ ~300 (harvest more images if short; prefer detector-val images, top up with train).
- [ ] Gate: open `crops_unsorted/` in Drive — crops are face-skin closeups, not garbage. A few obvious non-lesions is GOOD (they become `not_acne`/`post_acne_mark` labels).

### Task 5: Hand-sort + negatives (cells 3–4)

The real deliverable of the stage (spec §4.2). Budget hours, not minutes.

- [ ] Sort crops into `crops_labeled/<class>/` per the cell-3 table. Unsure → leave unsorted. Hunt `post_acne_mark` among low-conf crops.
- [ ] Run cell 4 (random on-face negatives). Eyeball every negative; delete any that landed on a lesion.
- [ ] Gate: all 5 dirs exist; `comedonal`/`inflammatory`/`not_acne` have dozens+ each; counts honest, no junk kept "to boost numbers".

### Task 6: Counts + thin-class decision (cell 5)

- [ ] Run cell 5, record the counts in the session log / DECISIONS.md if notable.
- [ ] If `post_acne_mark` < ~30: apply the collapse mechanics (move crops to `not_acne/`, keep empty dir) and note the raised cost — v1 loses the hyperpigmentation recommendation (spec §4.2).
- [ ] Gate: an explicit go/collapse decision exists before any training.

### Task 7: Train (cells 7–8; cell 6 pretrain deliberately skipped first)

Baseline first, no Kaggle pretrain — cell 6 is opt-in only if per-class recall disappoints (its dataset is unverified, spec §4.2).

- [ ] Run cell 7. Gate: `assert full.classes == CLASSES` passes; printed train counts + val size look sane (val ≈ 20%).
- [ ] Run cell 8 (20 epochs, AdamW 3e-4, weighted CE). Expected: per-epoch `val macro-recall`, best checkpoint saved to Drive as `type_classifier_v0.pt`.
- [ ] Gate: best macro-recall meaningfully above chance (0.2 for 5 classes). If not, suspect labels/split before hyperparameters.

### Task 8: Eval per spec §6 (cells 9–10)

- [ ] Run cell 9: read the red titles BEFORE metrics — is the confusion severity-flavored (inflammatory↔cystic) or treatment-flavored (post_acne_mark↔inflammatory, the expensive one)?
- [ ] Run cell 10: `classification_report` + confusion matrix. Record per-class recall for `cystic` and `post_acne_mark` specifically.
- [ ] Fitzpatrick: disaggregate if tone labels exist for source images; otherwise write the honest "too small to slice" note (D-016 — that note is a finding).
- [ ] When phone-photo crops exist: run ONLY cell 10's report on `crops_phone_test/` — the ACNE04-val vs phone gap is the domain-gap measurement (D-014).

### Task 9: Export + round-trip + wire-in (cell 11)

- [ ] Run cell 11. Expected: `LesionClassifier` loads the Drive weights on CPU and prints a prob dict whose argmax is plausible for the sample.
- [ ] Copy `type_classifier_v0.pt` → repo `models/type_classifier_v0.pt` (config already points there). Commit if repo policy allows ~10 MB binaries; otherwise note the Drive path in README.
- [ ] Gate: `LesionClassifier("models/type_classifier_v0.pt").predict(crop)` works locally (torch required) — the contract `assemble.py` will consume.

## Self-Review

- Spec coverage: §4.2 wrapper/stub/crop (Task 1), hybrid training with pretrain-as-option (Task 7 + cell 6), class sourcing incl. negatives + thin-class fallback (Tasks 5–6), §6 eval + D-016 + D-014 phone test set (Task 8), §7 hard-error weights (Task 1/9), config knobs (Task 1). Regions/assemble/no-face are OUT of scope here (separate plan).
- No placeholders: every step has exact code, command, or gate.
- Type consistency: `predict -> dict[class, prob]` used identically in Tasks 1, 8, 9; CLASSES order identical everywhere.
