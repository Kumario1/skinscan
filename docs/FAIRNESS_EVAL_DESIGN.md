# Fitzpatrick-disaggregated evaluation — design

Design spike for the skin-tone-disaggregated eval mandated by
`docs/DECISIONS.md` **D-016**. This is a design only: no dataset, no code, no
config change. The harness it describes is a future M–L plan. Every external
fact that this worktree cannot verify from the repo is tagged `OPEN:`.

## Mandate

D-016 (LOCKED) reads verbatim:

> **D-016 — Skin-tone-disaggregated evaluation is mandatory · LOCKED**
> Report error rates per Fitzpatrick skin-tone group, not just headline
> numbers. Skin-tone bias is the documented failure mode of these models. This
> is the Lesson-2 disaggregation principle (summary metrics compress failures)
> applied to a fairness axis, and it's the single most instructive eval here.
> Use Fitzpatrick17k for tone labels.

The two headline numbers this applies to — both currently reported as single
pooled scalars:

1. **Detector F1 = 0.722** at conf 0.07 / IoU 0.2, imgsz 1024
   (`README.md` §1). Produced by `src/detection/check_acne04_detector.py`: a
   confidence sweep over the ACNE04 test split, greedy IoU matching, then
   `precision/recall/F1` aggregated across *all* test images at once. The
   "locked operating point" is the `conf=0.07, iou=0.2` row of that sweep (the
   `m20` branch in the script).
2. **Classifier test accuracy = 91.18%** (macro F1 0.92, weighted F1 0.91;
   `README.md` §2). Produced by `src/classification/train_type_classifier.py`,
   which prints `sklearn` `classification_report` + `confusion_matrix` over
   `test_ds`. Per-class F1 exists for the five acne types
   (`Blackheads, Cyst, Papules, Pustules, Whiteheads` — `RAW_ACNE_CLASSES` in
   `src/classification/classifier.py`) but is reported pooled across tone.

Both scalars are exactly what D-016 warns about: they average over a skin-tone
axis that is the known failure mode, so they can hide a large per-group gap
while the pooled number looks fine.

The behavior is already reserved in `configs/default.yaml`:

```yaml
evaluation:
  disaggregate_by: fitzpatrick      # D-016, mandatory
  fitzpatrick_source: fitzpatrick17k
```

These two keys are load-bearing for this design: `disaggregate_by` names the
grouping axis, `fitzpatrick_source` names where tone labels come from. No code
reads them yet — this doc specifies what should.

## The labeling problem

ACNE04 ships **no skin-tone labels**. Its annotations are lesion bounding boxes
plus a severity/count grade (`src/detection/voc_to_yolo.py` parses the VOC XML;
there is no Fitzpatrick field). So there is nothing to group by until we
manufacture a tone label per image. Three ways to get one, with trade-offs:

### Option A — Manual Fitzpatrick (FST) labeling
Have human raters assign FST I–VI to each image in the ACNE04 **test split**
(`data/raw/acne04/Detection/VOC2007/ImageSets/Main/NNEW_test_0.txt`) plus the
self-collected set (D-014). This is the most defensible label and the only one
fit for a claim.

Protocol requirements:
- Fitzpatrick scale I–VI, judged from **non-lesional** facial skin (raters
  ignore the acne itself — inflammation reddens skin and inflates apparent
  tone).
- Written rater instructions with a reference chart; log ambient-lighting
  caveats (ACNE04 is heterogeneous clinical photography).
- **Two independent raters**; disagreements go to a third-rater / adjudication
  pass. Record inter-rater agreement (e.g. Cohen's κ) — a low κ is itself a
  finding about how labelable these photos are.
- Store one row per image; never overwrite, keep both raters' calls +
  adjudicated value.

Cost: slow, needs ≥2 people. This is the *final-eval* label.

### Option B — Automatic ITA (Individual Typology Angle) estimation
Estimate tone from pixels. Convert sampled skin pixels to CIELAB and compute

```
ITA° = arctan((L* − 50) / b*) × 180 / π
```

Higher ITA° ⇒ lighter skin; ITA° is binned into tone categories. This is a
cheap **v0** label that needs no annotators.

Hard constraints baked into the design:
- Sample **only non-lesional skin**. Concretely: sample skin-colored pixels
  from regions **outside every GT lesion box** (the boxes are already parsed by
  `check_acne04_detector.py` via `gt_boxes`), after a simple skin-color / face
  mask to drop hair, background, and specular highlights. Inflamed lesion
  pixels bias `a*`/`b*` and would corrupt ITA.
- ITA is biased by illumination — bathroom lighting and front-camera white
  balance (exactly the D-014 self-collected regime) shift `L*`. So ITA is
  usable for a **first cut / triage**, never for a published fairness claim.
- The exact ITA°→FST bin cutoffs are contested in the literature and must be
  **calibrated against a manual sample** (Option A) before the bins mean
  anything. `OPEN:` the precise cutoff table — do not hard-code textbook ITA
  thresholds without checking them against our own manually-labeled subset.

### Option C — Fitzpatrick17k as the tone-labeled corpus
Per D-016, Fitzpatrick17k is the *tone-label source*. Honest accounting of what
it is and is not:

- **What it is:** ~16,577 clinical dermatology images, each carrying a
  Fitzpatrick I–VI label and one of 114 skin-condition labels; FST assigned by
  non-expert annotators under a dynamic-consensus protocol with a
  dermatologist-labeled gold subset (Groh et al., 2021). It is a *legitimately
  tone-labeled* image corpus.
- **What it is NOT:** it is **not ACNE04**. Different sources, different
  conditions (114 dermatology diagnoses, not facial-acne close-ups), different
  framing. So Fitzpatrick17k cannot *directly* label an ACNE04 image.
- **How it actually helps:** it is training/validation fuel for the *tone
  estimator*, not a label join. Use it to (a) calibrate or validate an ITA
  estimator against human FST, or (b) train a small skin-tone classifier that
  we then run on ACNE04. Either way the ACNE04 label is inferred, and inferred
  labels get the same "triage, not claim" caveat as ITA.

### Recommendation
**ITA v0, calibrated against a small manual sample, for the first cut; manual
FST labels (Option A) for the final eval.** Fitzpatrick17k (Option C) is how we
sanity-check/calibrate the ITA estimator, not a shortcut around labeling
ACNE04.

## Metrics & grouping

**Grouping.** Report per FST group, pooled into three bins:

- **I–II** (light)
- **III–IV** (medium)
- **V–VI** (dark)

Justification for pooling six FST levels into three: per-level cells will be
thin, and thinner still once you also split by the five acne types (Stage 2) or
by lesion. Six-way × five-class cells would be mostly single digits and give
noise, not signal. Three bins keep each cell large enough to read while still
exposing the light-vs-dark gap that is the documented failure mode. Report the
group **N** alongside every metric so the pooling is auditable.

**Metrics per group:**
- **Detector**, at the locked operating point (`conf=0.07, iou=0.2, imgsz=1024`
  — no re-sweep per group; the operating point is fixed): precision, recall,
  F1, computed by summing per-image `pred / gt / match` counts *within* the
  group and applying the same P/R/F1 formulas `check_acne04_detector.py`
  already uses. Also report the pooled scalar so the disaggregated view sits
  next to the headline it decomposes.
- **Classifier**: per-class **recall** (and precision/F1) for each of the five
  acne types, per group; plus per-group overall accuracy. Recall per class ×
  group is the cell that reveals "the model misses cysts on dark skin."

**Small-group floor.** State a floor of **N ≥ 30** per pooled group for a
stable estimate. Below the floor:
- Still **report** the group — with a wide-confidence-interval caveat and the N
  printed loudly. Never hide or silently merge an undersized group; a
  suppressed group is the failure D-016 is built to prevent.
- Flag it explicitly (e.g. `low_n: true` in the output row) so downstream
  readers don't over-read the number.
- If a group is N ≈ 0, that absence is a first-class finding (see Risks).

## Harness design

A future `src/evaluation/disaggregate.py` (new package `src/evaluation/`; none
exists today). No code here — inputs, contracts, and output shape only.

**Inputs:**
1. **Tone-labels CSV** — `image_id,fst` (+ optional `fst_source`,
   `rater_agreement`, `low_confidence`). One row per evaluated image.
   - `image_id` uses each eval's *native* identifier so the join is exact: for
     the detector, the VOC stem (as produced by `split_ids` in
     `check_acne04_detector.py`); for the classifier, the test-set file's
     relative path under `data/raw/typeclassification/AcneDataset/test/`.
   - `fst` ∈ {1..6} (or the pooled bin) or empty if unlabelable.
2. **Existing per-image eval records** — the disaggregator does **not**
   re-run the model. It needs the two existing evals to emit, at the locked
   operating point, one record per image:
   - Detector: `{image_id, pred, gt, m20, m30}` — a per-image version of the
     counts `check_acne04_detector.py` currently only accumulates globally
     into `stats`. This means a small additive change to that script (emit a
     per-image JSON alongside `threshold_sweep.json`), **reusing** its existing
     `box_iou` / `match_count` internals unchanged — not a reimplementation.
   - Classifier: `{image_id, y_true, y_pred}` — a per-image dump from
     `train_type_classifier.py`'s existing `y_true` / `y_pred` arrays.

**Contract / reuse:**
- `disaggregate.py` is pure aggregation: left-join per-image records to the
  tone CSV on `image_id`, bucket into FST groups, sum counts (detector) or
  tally confusions (classifier) per group, apply the *same* P/R/F1 math the
  detector script already contains. It must not open images or load the model.
- Matching logic (`match_count`, `box_iou`) stays in
  `check_acne04_detector.py` and is imported, not duplicated — per the
  maintenance note, the harness slots into the existing JSON-outputs pattern,
  it does not replace the eval.

**Function surface (signatures / I-O contracts only, no bodies):**
- `load_tone_labels(csv_path) -> dict[str, int]` — `image_id → fst`; raises on
  duplicate `image_id`.
- `group_of(fst: int) -> str` — `{1,2}→"I-II"`, `{3,4}→"III-IV"`,
  `{5,6}→"V-VI"`.
- `disaggregate_detector(per_image_records, tone_labels) -> list[row]` — one
  row per group: `{group, n_images, pred, gt, matches, precision, recall, f1,
  low_n}`.
- `disaggregate_classifier(per_image_records, tone_labels, class_names)
  -> list[row]` — one row per (group × class): `{group, class, n, precision,
  recall, f1, low_n}` plus a per-group `accuracy` summary row.
- Images with no tone label go to an explicit `"unlabeled"` bucket and are
  **counted, never dropped** (silent drops hide coverage gaps).

**Output shape:** JSON (one row per group per metric, mirroring the row-list
shape of `threshold_sweep.json`) **plus** a printed table — same dual
JSON+stdout convention `check_acne04_detector.py` already uses. Written under
`runs/` (e.g. `runs/fairness_eval/`), consistent with the existing
`runs/detection_check/` outputs.

## Risks & honesty constraints

- **ITA bias on inflamed skin.** Acne inflames and reddens skin, shifting
  CIELAB `a*`/`b*`; ITA read over lesion pixels reads darker/redder than the
  person's constitutive tone. Mitigation is designed in (sample only
  outside-GT-box, non-lesional skin) but cannot be fully removed on heavily
  affected faces. So ITA labels stay "triage, not claim."
- **Illumination bias.** Front-camera white balance and bathroom lighting
  (the D-014 self-collected regime) move `L*` and therefore ITA. The
  self-collected set is where automatic tone estimation is *least* trustworthy
  and manual labeling matters most.
- **ACNE04's likely tone skew.** ACNE04 carries no demographic or skin-tone
  metadata, and its collection source is not documented in this repo.
  `OPEN:` the dataset's demographic composition — do not assert a specific
  national origin without a citation. What the design *can* state up front:
  the expected consequence is a **skewed FST distribution**, and one or more
  FST groups (most likely **V–VI**) may come back with **N ≈ 0**. Per D-016
  **that finding is itself the instructive result**: "the headline F1 is a
  light-skin F1 because the test set is almost entirely light-skinned" is
  exactly the failure a pooled scalar hides. The harness must surface an empty
  or tiny group loudly, not paper over it.
- **D-002 framing.** All outputs are cosmetic "concerns," never diagnostic
  claims (D-002). A fairness gap here reads as "the detector is less reliable
  on darker skin for *cosmetic acne concern* detection," never as a
  diagnostic-accuracy claim about a medical condition. Keep the language
  cosmetic in every reported table.
- **Inferred-label honesty.** Any metric computed on ITA- or classifier-
  inferred tone labels must be tagged as such in the output (`fst_source`), so
  a reader never mistakes an estimated-tone breakdown for a human-labeled one.

## Open questions

- `OPEN:` **Fitzpatrick17k access/licensing.** The dataset (Groh et al., 2021,
  `github.com/mattgroh/fitzpatrick17k`) is publicly described but this worktree
  could not confirm its redistribution license / terms of use. Confirm the
  license before building any tone estimator that ships derived weights or
  redistributes images.
- **Second-rater capacity.** Option A needs ≥2 raters + an adjudicator. Who
  does the second-rater pass, and do we have a board-certified reference for a
  gold subset (as Fitzpatrick17k used 312 gold images)?
- **Self-collected set size (D-014).** Is the self-collected test set (target
  ~100 images) large enough to report as its *own* disaggregated group, or only
  pooled? At ~100 images split three ways by tone, most bins fall under the
  N≥30 floor — likely report it pooled-with-caveat, not per-group. Revisit once
  the real N is known.
- `OPEN:` **ITA→FST cutoff table.** The exact ITA° bin boundaries to use are
  contested; fix them by calibrating against the manual sample rather than
  copying a textbook table.
- **ACNE04 demographic composition** — see Risks; unresolved until the manual
  labeling pass produces the actual FST distribution.

---

*Sources for external facts in this doc:*
[Fitzpatrick17k (Groh et al., 2021)](https://github.com/mattgroh/fitzpatrick17k),
[ACNE04 / LDL (Wu et al., ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf).
Repo facts cite `docs/DECISIONS.md`, `configs/default.yaml`,
`src/detection/check_acne04_detector.py`,
`src/classification/train_type_classifier.py`, and `README.md`.
