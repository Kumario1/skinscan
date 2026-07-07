# Plan 008: Update DECISIONS D-018 to record the shipped YOLOv8m (docs only — do NOT touch the model)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `git diff --stat 1ebd544..HEAD -- docs/DECISIONS.md`
> Any change to this file since `1ebd544` is a STOP condition.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: docs
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

`docs/DECISIONS.md` D-018 locks the Stage 1 model as **YOLOv8-nano** ("step up
to -small only if it underfits"). The shipped, README-documented, config-referenced
model is **YOLOv8m** (`models/detection/acne04_yolov8m_best.pt`, F1=0.722 at
conf 0.07). The file's own discipline — "once a decision is LOCKED, don't
silently reverse it — if it needs to change, edit the entry and note the
change" — was violated. The maintainer has confirmed the shipped model is the
decision: update the doc to match reality. **The model, weights, config, and
README stay untouched.**

## Current state

- `docs/DECISIONS.md:139-148` — the entry to update:

```markdown
### D-018 — Stage 1 model: YOLOv8n, COCO-pretrained, fine-tune-all, single-class · LOCKED
Transfer learning, not from-scratch (1,457 images can't train a detector from
random init) and not a VLM API (teaches no CV, D-001). Ultralytics YOLOv8-nano,
COCO-pretrained weights, head reconfigured to 1 class (`lesion`), all layers
fine-tuned at a low LR (lr0≈0.001) to shift features gently without
catastrophic forgetting. Nano fits a free Colab T4; step up to -small only if
it underfits (config change, not a rewrite). Severity is NOT detected — it's
derived from lesion count/density per region (D-004), which is why we keep
ACNE04's Classification (count) labels. New eval vocabulary: IoU, mAP.
Workflow rule: eyeball predictions BEFORE reading metrics.
```

- Evidence of the shipped model (do not modify any of these):
  - `configs/default.yaml`: `weights: models/detection/acne04_yolov8m_best.pt`
  - `README.md` §1: `weights: models/detection/acne04_yolov8m_best.pt`, F1=0.722
    at conf=0.07 / IoU=0.2, imgsz 1024
- The doc's status legend (line 8) allows editing a LOCKED entry when the
  change is recorded in the entry itself.

## Environment facts

- Fresh git worktree. Docs-only plan; no interpreter needed beyond grep.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Confirm target | `grep -n "YOLOv8n" docs/DECISIONS.md` | 1 match at the D-018 heading (before edit) |
| Confirm done | `grep -n "YOLOv8m" docs/DECISIONS.md` | 1+ match (after edit) |

## Scope

**In scope**:
- `docs/DECISIONS.md` (the D-018 entry only)

**Out of scope**:
- `models/`, `configs/default.yaml`, `README.md`, any code. This plan changes
  one documentation entry, nothing else.
- Rewriting other decisions or resolving the trailing Q-B notes.

## Git workflow

- Stay on the worktree's branch. Commit style:
  `docs: record d-018 step-up to the shipped yolov8m`
- Do NOT push.

## Steps

### Step 1: Edit the D-018 entry

Update the heading and add a dated change note as the first line of the body,
preserving the original rationale below it. Target shape:

```markdown
### D-018 — Stage 1 model: YOLOv8m, COCO-pretrained, fine-tune-all, single-class · LOCKED
**Changed 2026-07-06:** originally locked as YOLOv8-nano; the shipped,
validated detector is YOLOv8-medium (`models/detection/acne04_yolov8m_best.pt`,
F1=0.722 at conf 0.07 / IoU 0.2, imgsz 1024 — see README §1). Entry updated to
match the shipped model per this file's own change rule; the model itself is
NOT changing. Original reasoning (still applies, with -nano → -medium):
Transfer learning, not from-scratch (1,457 images can't train a detector from
random init) and not a VLM API (teaches no CV, D-001). COCO-pretrained
weights, head reconfigured to 1 class (`lesion`), all layers fine-tuned at a
low LR (lr0≈0.001) to shift features gently without catastrophic forgetting.
Severity is NOT detected — it's derived from lesion count/density per region
(D-004), which is why we keep ACNE04's Classification (count) labels. New eval
vocabulary: IoU, mAP. Workflow rule: eyeball predictions BEFORE reading
metrics.
```

Note what is deliberately dropped from the old text: the "Nano fits a free
Colab T4; step up to -small only if it underfits" sentence — superseded by the
change note.

**Verify**: `grep -n "YOLOv8n" docs/DECISIONS.md` → no matches;
`grep -n "Changed 2026-07-06" docs/DECISIONS.md` → 1 match

## Test plan

None (docs-only). Verification is the two greps.

## Done criteria

- [ ] `grep -c "YOLOv8m" docs/DECISIONS.md` → ≥ 1
- [ ] `grep -c "YOLOv8n" docs/DECISIONS.md` → 0
- [ ] `git diff --stat` shows exactly one file changed: `docs/DECISIONS.md`
- [ ] The diff is confined to the D-018 entry (no other entries touched)

## STOP conditions

- The D-018 entry doesn't match the excerpt.
- You find yourself wanting to edit any file other than `docs/DECISIONS.md`.

## Maintenance notes

- If the detector is ever retrained at a different size, this entry gets
  another dated change note — that's the pattern this plan establishes.
