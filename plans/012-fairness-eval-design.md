# Plan 012: Design the Fitzpatrick-disaggregated evaluation (design doc only — no dataset, no code)

> **Executor instructions**: Follow this plan step by step. This is a DESIGN
> SPIKE: the deliverable is a document, not code. Run every verification
> command. If anything in the "STOP conditions" section occurs, stop and
> report. Your reviewer maintains `plans/README.md` — do not update it.
>
> **Drift check (run first)**: `ls docs/FAIRNESS_EVAL_DESIGN.md 2>/dev/null` —
> if it exists, STOP.

## Status

- **Priority**: P3
- **Effort**: S (doc; the harness it designs is M–L)
- **Risk**: LOW (no code changes)
- **Depends on**: none
- **Category**: direction
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

DECISIONS.md D-016 is LOCKED and says skin-tone-disaggregated evaluation is
**mandatory** — "the single most instructive eval here" — and
`configs/default.yaml` already carries `evaluation.disaggregate_by:
fitzpatrick`. Zero code or process exists for it. Skin-tone bias is the
documented failure mode of dermatology CV models, and this repo's headline
numbers (detector F1=0.722, classifier acc 91.18%) are exactly the kind of
summary metrics D-016 warns "compress failures." This spike produces the
design the maintainer can execute.

## Current state (read before writing)

- `docs/DECISIONS.md` D-016 (mandate, Fitzpatrick17k as tone source) and
  D-014 (self-collected photos are test-only).
- `configs/default.yaml`:

```yaml
evaluation:
  disaggregate_by: fitzpatrick      # D-016, mandatory
  fitzpatrick_source: fitzpatrick17k
```

- `src/detection/check_acne04_detector.py` — the existing detector eval:
  per-image predict → greedy IoU matching (`match_count`) → precision/recall/F1
  across a confidence sweep, JSON + render-sheet outputs. The natural host for
  a `group-by-tone` axis.
- `src/classification/train_type_classifier.py` — test-set
  classification_report / confusion matrix — the classifier eval to
  disaggregate.
- ACNE04 images live (locally) under `data/raw/acne04/Classification/JPEGImages`
  with VOC annotations; there are NO skin-tone labels in ACNE04.

## Environment facts

- Fresh git worktree; `data/` absent. Doc-only plan. You may use WebSearch (if
  available in your environment) to confirm Fitzpatrick17k facts; if
  unavailable, write from the repo evidence and mark external facts `OPEN:`.

## Scope

**In scope**:
- `docs/FAIRNESS_EVAL_DESIGN.md` (create — only deliverable)

**Out of scope**:
- Any code, any dataset download, any change to configs or DECISIONS.

## Git workflow

- Stay on the worktree's branch. Commit style:
  `docs: design the fitzpatrick-disaggregated eval (D-016)`
- Do NOT push.

## Steps

### Step 1: Write `docs/FAIRNESS_EVAL_DESIGN.md`

Required sections (exact headings):

1. `## Mandate` — quote D-016, name the two headline metrics it applies to
   (detector F1 from `check_acne04_detector.py`'s sweep; classifier test
   accuracy/per-class F1 from the trainer) and the config keys that already
   reserve the behavior.
2. `## The labeling problem` — ACNE04 has no tone labels; three options with
   trade-offs:
   - **Manual FST labeling** of the ACNE04 test split (+ the self-collected
     set): most defensible; needs a labeling protocol (Fitzpatrick I–VI,
     rater instructions, 2 raters + adjudication on disagreement); slow.
   - **Automatic ITA (Individual Typology Angle) estimation** from
     non-lesional skin pixels: cheap v0, but inflamed/lesional skin and
     bathroom lighting bias ITA — usable for a first cut, never for claims.
     The design must specify sampling non-lesion regions (outside GT boxes).
   - **Fitzpatrick17k as tone-labeled data**: per D-016 it is the tone-label
     source, but note honestly what it is (dermatology images labeled with
     FST) and what it is NOT (it is not ACNE04 — it can calibrate/validate an
     ITA estimator or train a tone classifier, not directly label ACNE04).
   - Recommendation: ITA v0 calibrated against a small manual sample; manual
     labels for the final eval.
3. `## Metrics & grouping` — per-FST-group (I–II / III–IV / V–VI pooled —
   justify pooling by expected group sizes): detector precision/recall/F1 at
   the locked operating point, classifier per-class recall; report group Ns
   and flag any group with N too small for stable estimates (state a floor,
   e.g. N≥30, and what to do below it: report with wide-CI caveat, don't
   hide).
4. `## Harness design` — a future `src/evaluation/disaggregate.py`: inputs
   (a tone-labels CSV `image_id,fst`, the existing eval outputs), reuse of
   `check_acne04_detector.py`'s matching internals, output shape (one row per
   group per metric, JSON + printed table). No code here — signatures and I/O
   contracts only.
5. `## Risks & honesty constraints` — ITA bias on inflamed skin; ACNE04's
   likely tone skew (dataset from Chinese hospitals — the design must say the
   expected consequence: some FST groups may have N≈0, and THAT FINDING is
   itself the instructive result per D-016); D-002 framing (cosmetic, not
   diagnostic claims).
6. `## Open questions` — at least: exact Fitzpatrick17k access/licensing;
   who does the second-rater pass; whether self-collected set (D-014) is
   large enough to report separately.

**Verify**: `for h in "## Mandate" "## The labeling problem" "## Metrics & grouping" "## Harness design" "## Risks & honesty constraints" "## Open questions"; do grep -q "$h" docs/FAIRNESS_EVAL_DESIGN.md || echo "MISSING: $h"; done` → no output

## Test plan

None (doc-only). Verification is the heading grep plus reviewer read.

## Done criteria

- [ ] `docs/FAIRNESS_EVAL_DESIGN.md` exists with all 6 required headings
- [ ] Cites D-016, D-014, D-002 and both eval scripts by path
      (`grep -c "D-016\|check_acne04_detector\|D-014\|D-002" docs/FAIRNESS_EVAL_DESIGN.md` ≥ 4)
- [ ] `git status --porcelain` shows only the new doc

## STOP conditions

- An existing fairness-eval design or `src/evaluation/` module turns up.
- You cannot verify an external fact and are tempted to invent it — use an
  `OPEN:` marker instead and note it in your report.

## Maintenance notes

- Executing this design is a future M–L plan (labeling pass + harness build).
- The harness should slot into `check_acne04_detector.py`'s existing
  JSON-outputs pattern, not replace it.
