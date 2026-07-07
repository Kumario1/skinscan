# Plan 010: Design the `not_acne` negative class for the Stage 2 classifier (design doc only — no training)

> **Executor instructions**: Follow this plan step by step. This is a DESIGN
> SPIKE: the deliverable is a document, not code. Run every verification
> command. If anything in the "STOP conditions" section occurs, stop and
> report. Your reviewer maintains `plans/README.md` — do not update it.
>
> **Drift check (run first)**: `ls docs/STAGE2_NEGATIVES_DESIGN.md 2>/dev/null`
> — if it exists, STOP (already designed).

## Status

- **Priority**: P2
- **Effort**: S (for the doc; the implementation it designs is L)
- **Risk**: LOW (no code changes)
- **Depends on**: plans/004-bridge-readiness-unify-vocab.md (reflects the post-004 vocabulary)
- **Category**: direction
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

The Stage 2 classifier is forced-choice among 5 acne types — softmax has no
"none of the above." Every detector false positive therefore BECOMES an acne
lesion in the type counts. The repo's own README documents the symptom: on the
maintainer's self-collected photo, all 16 detector candidates (detector conf
only 0.19–0.37) were classified as Pustules with classifier confidence up to
1.00. The groundwork already exists in the repo's decisions: D-013 locks FFHQ
as the clear-skin negative source, and git history shows negative-crop
harvesting existed in the deleted notebook era. This spike turns that into a
concrete, reviewable design the maintainer can execute on Colab later.

## Current state (read all of these before writing)

- `src/classification/classifier.py` — `RAW_ACNE_CLASSES` (5 classes),
  `RAW_TO_CONCERN` (post-plan-004: maps to schema IDs `acne_comedonal`/
  `acne_cystic`/`acne_inflammatory`), `concern_probs()` ignores unmapped
  classes — meaning a future `not_acne` class is ALREADY silently dropped from
  concern aggregation, which is exactly the desired end behavior.
- `src/classification/train_type_classifier.py` — the trainer: directory-per-class
  layout under `train/ valid/ test/`, class order asserted against
  `RAW_ACNE_CLASSES` (`if class_names != RAW_ACNE_CLASSES: raise`), class
  weights (post-plan-005: derived from directory counts), EfficientNetB0,
  labels metadata JSON written next to the model.
- `src/classification/run_acne04_pipeline.py` — the pipeline that would gain a
  rejection path once `not_acne` exists.
- `docs/DECISIONS.md` — D-011 (Kaggle type labels second-tier), D-013 (FFHQ
  negatives, LOCKED), D-014 (self-collected photos TEST-ONLY, never train).
- `README.md` §4 — the 16/16-Pustules custom-image result and its
  interpretation caveats.

## Environment facts

- Fresh git worktree; `data/`, `models/` absent. No training possible or
  wanted here. Interpreter (only for greps/inspection):
  `/Users/princekumar/Documents/skinscan/.venv/bin/python`.

## Scope

**In scope**:
- `docs/STAGE2_NEGATIVES_DESIGN.md` (create — the only deliverable)

**Out of scope**:
- ANY code change. No edits to classifier, trainer, pipeline, configs, tests.
- Downloading FFHQ or any dataset.
- Changing `RAW_ACNE_CLASSES` — that happens when the design is executed.

## Git workflow

- Stay on the worktree's branch. Commit style:
  `docs: design the not_acne negative class for stage 2`
- Do NOT push.

## Steps

### Step 1: Write `docs/STAGE2_NEGATIVES_DESIGN.md`

Required sections (use these exact headings so the done-criteria greps pass):

1. `## Problem` — forced-choice softmax; cite the README §4 numbers (16/16
   Pustules, detector conf 0.19–0.37, classifier conf 0.47–1.00) and explain
   why a low-quality detector box still yields a high-confidence type.
2. `## Options considered` — at least these three, each with 2–4 sentences of
   honest trade-offs:
   - **A. Sixth `not_acne` class** (recommended): retrain with negative crops;
     integrates with existing directory-per-class trainer; costs a training
     run + data prep.
   - **B. Probability/entropy threshold on the existing 5-class model**: no
     retraining, but calibration on out-of-distribution crops is exactly what
     softmax is bad at (the observed 1.00-confidence false positives argue
     against this alone).
   - **C. Two-stage gate (binary lesion/not-lesion, then type)**: cleanest
     separation, but a second model to train/version/serve — over-engineered
     for a learning project at this stage.
3. `## Recommendation` — Option A, with B's threshold kept as a cheap
   additional knob after retraining (they compose).
4. `## Negative data sources` — grounded in the repo's decisions:
   - FFHQ clear-skin faces (D-013): run the OWN detector at the locked
     operating point (conf 0.07, iou 0.2, imgsz 1024) over FFHQ faces; every
     resulting box is by-construction a false positive → harvest crops with
     the same `crop_with_context(pad=1.5, size=224)` used at inference (this
     matches train/inference distribution — important, say why).
   - Non-lesion facial regions from ACNE04 images (boxes sampled AWAY from GT
     annotations, IoU 0 with all GT boxes).
   - Explicitly EXCLUDED: self-collected photos (D-014, test-only — restate).
5. `## Dataset sizing & balance` — target the negative class at roughly the
   median existing class size (~600, comparable to Cyst's 645/Pustules' 584),
   split train/valid/test in the same ratios as the existing dataset
   (2778/921/918 ≈ 60/20/20); note class weights are now derived
   automatically from directories (plan 005).
6. `## Code changes required when executed` — enumerate precisely:
   - `RAW_ACNE_CLASSES` → `RAW_CLASSES` gains `"Not_acne"`? NO — instead
     document the naming decision: the new directory must sort into a known
     position (tf's `image_dataset_from_directory` orders alphabetically) and
     the trainer's class-order assert must be updated in the same commit; the
     labels metadata JSON already carries the class list, and
     `AcneTypeClassifier` reads classes from metadata, so INFERENCE code needs
     zero changes.
   - `RAW_TO_CONCERN` needs NO change (unmapped classes drop out of concern
     aggregation — cite `concern_probs`).
   - `run_acne04_pipeline.py`: counts will naturally include `not_acne`; the
     design should specify that `acne_type_counts` keeps reporting it (it is
     useful signal), while the future bridge ignores it.
7. `## Acceptance criteria` — measurable:
   - False-positive rate on a held-out FFHQ sheet: ≥80% of detector boxes on
     clear skin classified `not_acne`.
   - No macro-F1 regression > 2 points on the original 5 classes' test split.
   - README §4 custom-image test re-run: expected outcome is that some of the
     16 become `not_acne`.
8. `## Open questions` — at least: FFHQ licensing/size subset to use; whether
   Kaggle's dataset has usable negatives already; whether to also add
   `post_acne_mark` while retraining (recommend: no — one variable at a time).

Ground every claim in a file/line or README section — the doc should cite like
this plan does.

**Verify**: `for h in "## Problem" "## Options considered" "## Recommendation" "## Negative data sources" "## Dataset sizing" "## Code changes required" "## Acceptance criteria" "## Open questions"; do grep -q "$h" docs/STAGE2_NEGATIVES_DESIGN.md || echo "MISSING: $h"; done` → no output

## Test plan

None (doc-only). Verification is the heading grep above plus reviewer read.

## Done criteria

- [ ] `docs/STAGE2_NEGATIVES_DESIGN.md` exists with all 8 required headings
- [ ] The doc cites at least: README §4 numbers, D-013, D-014, `concern_probs`
      (`grep -c "D-013\|D-014\|concern_probs" docs/STAGE2_NEGATIVES_DESIGN.md` ≥ 3)
- [ ] `git diff --stat` shows exactly one new file
- [ ] No source files modified (`git status --porcelain` shows only the doc)

## STOP conditions

- You find an existing negatives design or a `not_acne` class already present
  in `classifier.py` (would mean the premise is stale).
- You cannot ground a required section in repo evidence — write the section
  with an explicit `OPEN:` marker instead of inventing facts, and note it in
  your report.

## Maintenance notes

- Executing this design is a future L-effort plan (data prep on Colab +
  retraining + re-running the README §4 test). The design doc is its spec.
- Plans 004/005/006 were written so execution needs minimal code change:
  vocabulary ignores unmapped classes, class weights auto-derive, batch
  predict is class-count-agnostic.
