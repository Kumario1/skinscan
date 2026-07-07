# Stage 2 negative class (`Not_acne`) — design

Design spike only. No code changes here; this document is the spec for a future
L-effort execution (data prep on Colab + one retraining run + re-running the
README §4 test). Every claim below is anchored to a file/line, a `README.md`
section, or a decision ID.

## Problem

Stage 2 is a forced-choice softmax over exactly 5 acne types
(`RAW_ACNE_CLASSES = ["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]`,
`src/classification/classifier.py:7`; final layer is
`Dense(num_classes, activation="softmax")`, `classifier.py:73`). There is no
"none of the above" output. Every crop the detector hands it is therefore forced
onto one of the 5 types — a detector false positive does not get rejected, it
*becomes* an acne lesion in `acne_type_counts`
(`src/classification/run_acne04_pipeline.py:83-87,131`).

The repo already documents the symptom. README §4 ("My image test") ran the full
detector→classifier pipeline on a self-collected photo and got:

- **16** detector candidates, **16** classified `Pustules` (`type counts: Pustules=16`, README §4).
- classifier confidence range **0.47–1.00** (README §4).
- detector confidence range **0.19–0.37** (README §4).

The detector confidences (0.19–0.37) sit just above the operating-point floor
(conf 0.07, D-018) — these are weak, borderline boxes. Yet the classifier
reported confidence up to **1.00** on them. That is the core failure a softmax
head produces on out-of-distribution input: the exponential normalization always
sums to 1 across the 5 types, so even a crop of clear skin, a pore, or a shadow
is pushed onto the nearest type with an arbitrarily peaked probability. A high
classifier probability on a low-quality detector box is not evidence the crop is
a lesion — README §3 already warns "A high classifier probability on a bad
detector crop is still a bad result." The classifier has no vocabulary to say the
box was junk.

## Options considered

**A. Sixth `Not_acne` class (recommended).** Retrain the existing EfficientNetB0
with a negative class added as one more directory under `train/ valid/ test/`.
The trainer is already directory-per-class
(`image_dataset_from_directory`, `train_type_classifier.py:51-69`) and derives
class weights from directory counts (`class_weights`,
`train_type_classifier.py:77-86`), so a new class needs no trainer-logic change,
only data and the class-order assert. Inference is class-count-agnostic
(`predict_batch` zips `self.classes` from metadata,
`classifier.py:83-84,96-103`). Cost: one training run plus negative-crop
harvesting. This gives the model a real, learned reject region instead of a
post-hoc patch.

**B. Probability / entropy threshold on the existing 5-class model.** Requires no
retraining: reject a crop when its top-1 probability is below a cutoff (or its
entropy is high). But calibration on out-of-distribution crops is exactly what
softmax is worst at — the 1.00-confidence false positives observed in README §4
are the counter-example. A threshold that catches them would also throw away the
weak-but-real 0.47 detections. Cheap, but on this evidence it cannot be the
primary defense.

**C. Two-stage gate (binary lesion / not-lesion, then type).** A dedicated binary
model rejects non-lesions, then the 5-class type model runs only on survivors.
Cleanest conceptual separation and the binary head can be tuned independently.
But it is a second model to train, version, checkpoint, and serve — more moving
parts than a learning project at this stage warrants (D-001: optimize for lessons
learned per hour, not production robustness). Over-engineered for now.

## Recommendation

Take **Option A**: add a `Not_acne` class and retrain. It fits the existing
trainer and inference paths with near-zero code change and gives a learned reject
region rather than a calibration guess.

Keep **Option B as a cheap secondary knob** applied *after* retraining: once a
real `Not_acne` class exists, a probability floor on the surviving type
predictions composes with it — they are independent and stack. Do not ship B
alone (its weakness is the observed failure), and do not build C yet.

## Negative data sources

The negatives must match the train/inference distribution: the classifier only
ever sees `crop_with_context(image, box, pad=1.5, size=224)` crops
(`classifier.py:32-48`; the pipeline crops with `crop_pad=1.5`, `crop_size=224`
from `configs/default.yaml:12-13`). A negative built any other way (raw face
tiles, random rectangles) would teach the model to separate on crop *style*
rather than lesion presence. So every negative below is produced by the same crop
function fed real detector boxes.

1. **FFHQ clear-skin faces (D-013, LOCKED).** D-013 locks FFHQ as the clear-skin
   negative source "so the detector doesn't learn 'every face has acne,' and for
   false-positive-rate testing." Run the *own* YOLOv8m detector
   (`models/detection/acne04_yolov8m_best.pt`) at the locked operating point —
   **conf 0.07, IoU 0.2, imgsz 1024** (D-018; `configs/default.yaml:6-8`) — over
   FFHQ faces. On clear skin every box the detector returns is by construction a
   false positive, so harvest each with the same
   `crop_with_context(pad=1.5, size=224)` used at inference. This is the ideal
   negative: same detector, same operating point, same crop transform, same
   failure mode we want to reject — it samples the exact distribution of mistakes
   README §4 exposed.

2. **Non-lesion regions from ACNE04 images.** ACNE04 is dermatologist-boxed
   (D-010), so its GT boxes mark the lesions. Sample boxes *away* from every GT
   annotation (IoU 0 with all GT boxes in that image) and crop them the same way.
   These add in-domain, in-lighting negatives (clear cheek, forehead, background)
   that FFHQ's photographic style does not cover, and they cost no new download.

**Explicitly excluded: self-collected photos.** D-014 locks our own phone photos
(the README §4 image among them) as **TEST-ONLY**, never training. They stay a
held-out domain-gap probe; they do not enter the `Not_acne` training directory.
Restating D-014 here because the FFHQ harvest is superficially similar work and
the temptation to fold in self-collected negatives is real — do not.

## Dataset sizing & balance

Grounded reference points (README §2). The existing split is **train 2778 /
valid 921 / test 918** (≈ 60/20/20). Per-class *test* support is Blackheads 265,
Cyst 189, Papules 202, Pustules 205, Whiteheads 57 — the classes are already
uneven, and Whiteheads is the small tail.

Target: size `Not_acne` near the median existing class and split it in the same
≈ 60/20/20 ratio as the rest, so no split is dominated by negatives. Class
imbalance needs no manual correction — `class_weights` recomputes `balanced`
weights from the directory counts at train time
(`train_type_classifier.py:77-86`, plan 005), so whatever the final negative
count is, its loss weight self-adjusts. Aim to keep the negative class from
exceeding the largest real class, to avoid the model collapsing toward "reject
everything."

`OPEN:` The plan's concrete target ("~600, comparable to Cyst's 645 / Pustules'
584") could not be grounded in the repo — those full-dataset per-class totals are
not in `README.md`, `docs/`, or `notebooks/`; the only per-class counts committed
are the *test-split* supports above (57–265), because the training images live in
the gitignored `data/raw/typeclassification/AcneDataset`
(`configs/default.yaml:15`), absent from this worktree. At execution, read the
actual per-class directory counts with `train_type_classifier.py --inspect`
(`train_type_classifier.py:40-44`) and set the `Not_acne` target to that observed
median before harvesting.

## Code changes required when executed

Inference needs **zero** changes; the edits are confined to the trainer-facing
vocabulary and the stub/test fixtures.

- **New data directory.** Add `Not_acne/` under `train/ valid/ test/`. The name
  matters: `image_dataset_from_directory` orders classes **alphabetically**
  (`train_type_classifier.py:51-55`), so `Not_acne` inserts at index 2:
  `["Blackheads", "Cyst", "Not_acne", "Papules", "Pustules", "Whiteheads"]`.
  Whatever name is chosen fixes the label indices, so it must be decided once and
  never renamed.
- **`RAW_ACNE_CLASSES` (`classifier.py:7`)** → update to the 6-element list above,
  in the same commit as the data. This single edit also satisfies the trainer's
  class-order guard for free: the assert compares live `class_names` against
  `RAW_ACNE_CLASSES` (`train_type_classifier.py:106-107`), so once the constant
  matches the new alphabetical order the guard passes. `build_kaggle_efficientnet`
  defaults `num_classes=len(RAW_ACNE_CLASSES)` (`classifier.py:51`) and so tracks
  automatically; the trainer also passes `len(class_names)` explicitly
  (`train_type_classifier.py:110`), so the head width is correct either way.
- **`RAW_TO_CONCERN` (`classifier.py:8-14`)** → **NO change.** `Not_acne` is
  deliberately left unmapped. `concern_probs` only aggregates keys present in
  `RAW_TO_CONCERN` (guard `if raw in RAW_TO_CONCERN`, `classifier.py:23-29`), so
  an unmapped `Not_acne` mass silently drops out of concern aggregation — exactly
  the desired end behavior, and already regression-tested: `test_concern_vocab.py`
  feeds a literal `{"Blackheads": 0.5, "not_acne": 0.5}` and asserts the
  `not_acne` half is discarded (`tests/test_concern_vocab.py:33-35`).
- **`StubClassifier` (`classifier.py:115-124`)** → update its default 5-element
  `probs` to 6 elements. It hardcodes `[0.25, 0.2, 0.2, 0.2, 0.15]`
  (`classifier.py:117`) and zips against `RAW_ACNE_CLASSES`
  (`classifier.py:121`); if the constant grows to 6, `zip` truncates to 5 and
  silently drops a class, breaking the `set(p) == set(RAW_ACNE_CLASSES)` self-test
  (`classifier.py:136`) and `test_concern_vocab.py:41-42`.
- **`tests/test_concern_vocab.py:20`** → relax
  `assert set(RAW_TO_CONCERN) == set(RAW_ACNE_CLASSES)` to a proper-subset check
  (`set(RAW_TO_CONCERN) < set(RAW_ACNE_CLASSES)`), since `Not_acne` is
  intentionally in the class set but not in the concern map. `test_predict_batch.py`
  needs no edit — it derives its expected set from `RAW_ACNE_CLASSES` itself
  (`tests/test_predict_batch.py:58`).
- **`AcneTypeClassifier` inference (`classifier.py:77-109`)** → **NO change.** It
  reads `classes` from the labels-metadata JSON
  (`self.classes = list(classes or metadata.get("classes", ...))`,
  `classifier.py:83-84`), and that JSON is written from the live `class_names`
  each run (`train_type_classifier.py:141-148`). `predict_batch` zips
  `self.classes` with the model's output vector (`classifier.py:96-103`), so a
  6-wide model just yields 6-key dicts. Nothing in the inference path counts to 5.
- **`run_acne04_pipeline.py`** → **NO code change; a reporting decision.**
  `acne_type_counts` (`run_acne04_pipeline.py:83-87`) counts top-1 labels
  (`classifier.py:120-123`) and `analyze_image` passes `clf.classes` so the new
  `Not_acne` bucket appears in `acne_type_counts`
  (`run_acne04_pipeline.py:131`). Keep reporting it — a `Not_acne` count is useful
  signal about detector precision on a given image. The future concern bridge
  ignores it automatically via `concern_probs` (above), so no gating logic is
  needed in the pipeline.

## Acceptance criteria

Measurable, checkable at execution:

- **FFHQ reject rate.** On a held-out FFHQ sheet the detector never saw during
  negative harvesting, **≥ 80%** of the detector boxes (produced at the locked
  conf 0.07 / IoU 0.2 / imgsz 1024 operating point) are classified `Not_acne`.
  This directly measures the false-positive suppression D-013 exists for.
- **No type regression.** Macro-F1 on the *original* 5 classes' test split does
  not drop more than **2 points** from the current baseline (README §2: macro F1
  **0.92**). Evaluate on the same test images, ignoring `Not_acne` rows, using the
  classification report the trainer already prints
  (`train_type_classifier.py:134-139`).
- **README §4 re-run.** Re-run the exact README §4 command on the same
  self-collected image. Expected outcome: several of the 16 previously-`Pustules`
  crops flip to `Not_acne` (the weak 0.19–0.37 detector boxes are the prime
  candidates). Update README §4 with the new counts when it lands.

## Open questions

- **FFHQ licensing / subset.** FFHQ is large; which subset and how many faces to
  pull for the harvest, and confirm the license permits this research use
  (D-001 is research-only, but record it). Enough faces to hit the target
  negative count after the detector's per-face yield is known.
- **Kaggle negatives already present?** D-011 sources the type labels from Kaggle
  (second-tier). Check whether that Kaggle dataset ships a non-acne / clear
  category usable as negatives before harvesting FFHQ — it could cut the data-prep
  cost, though its crop style would still need the `crop_with_context` transform
  to match distribution.
- **Add `post_acne_mark` in the same run?** Tempting to also introduce a
  post-inflammatory-mark class while retraining. Recommend **no** — one variable
  at a time. Land `Not_acne`, measure it against the criteria above, then consider
  new positive classes separately so a regression is attributable.
- **Negative target count.** See the `OPEN:` in *Dataset sizing & balance* — the
  exact per-class median must be read from the live data directory
  (`train_type_classifier.py --inspect`) at execution, since the training counts
  are not committed to the repo.
