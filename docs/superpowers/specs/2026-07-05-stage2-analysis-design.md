# Stage 2 — Analysis Layer Design

**Status:** DRAFT (awaiting user review)
**Date:** 2026-07-05
**Depends on:** D-004 (two-stage), D-008 (concern schema is the contract),
D-011 (Kaggle type labels are second-tier), D-013 (FFHQ clear-skin negatives),
D-016 (Fitzpatrick eval), D-018 (Stage 1 detector).

---

## 1. What this stage is

Stage 2 is an **assembler**, not a monolith. It takes Stage 1's raw lesion
boxes and turns them into a valid `ConcernReport` (the locked contract in
`docs/CONCERN_SCHEMA.md`) that the already-built Stage 3 recommender consumes.

One public entry point:

```python
analyze(image, boxes) -> ConcernReport
```

Everything else is internal. `boxes` is Stage 1's output: a list of
`(x, y, w, h, det_confidence)` in pixel coords, single class `lesion`.

Four jobs, only **one** of which is ML:

| Job | Method | ML? |
|-----|--------|-----|
| Where → region | MediaPipe Face Mesh landmarks + geometric rules | no |
| What type | trained lesion-crop classifier | **yes** |
| How bad → severity | count lesions per (type, region) → ordinal 0–4 | no |
| Is it active acne | classifier `post_acne_mark` / `not_acne` classes + off-face filter | (part of the classifier) |

## 2. Scope & non-goals

**In scope:** region assignment, off-face false-positive rejection, a 5-class
lesion classifier, severity derivation, assembly into `ConcernReport`, a no-face
failure branch, and the Colab training notebook for the classifier.

**Out of scope (deliberately):**
- **Heuristic size/color baseline — dropped by user decision.** Not built. Can
  be added later as a sanity check if we ever doubt the model is earning its keep.
- **Comprehensive mark/scar detection.** The ACNE04-trained detector does not
  box flat marks or scars, so we cannot *find* all of them. `post_acne_mark`
  only triages a box that *happens* to land on one — an opportunistic partial
  signal (which starts feeding `hyperpigmentation`, §3), not full assessment.
  Comprehensive coverage stays D-012. (See §4.2.)
- **Whitehead-vs-blackhead and papule-vs-pustule granularity.** Collapsed into
  `comedonal` and `inflammatory` respectively (§3).
- **Full hyperpigmentation detector** — still D-012, deferred.

## 3. Taxonomy ⚠ CONFIRM AT REVIEW

The classifier trains on **5 visual classes**. They map 1:1 onto the schema's
concern vocabulary so the locked contract (D-008) is **not modified**:

| Classifier class | Covers | → Schema concern | Recommender behaviour |
|------------------|--------|------------------|-----------------------|
| `comedonal`      | blackheads, whiteheads | `acne_comedonal` | salicylic acid / adapalene |
| `inflammatory`   | papules, pustules | `acne_inflammatory` | benzoyl peroxide / azelaic |
| `cystic`         | nodules, cysts | `acne_cystic` | route to professional |
| `post_acne_mark` | post-acne marks — pigmentation **and** texture (your "scarring") | `hyperpigmentation` | niacinamide / vit C / azelaic + SPF, **plus** a texture advisory |
| `not_acne`       | mole, shadow, clear skin, detector FP | *(none — dropped)* | excluded entirely |

**⚠ The one override to confirm:** the user's list wrote "blackheads" and
"papules/comedonal". Papules are *inflammatory*, blackheads are *comedonal*, and
`RULES.md §1` gives them **different** actives — merging them blinds the
recommender. So this spec **splits** them: blackheads → `comedonal`, papules →
`inflammatory`. If you'd rather have a single coarse "bumps" class, say so and
we lose the comedonal/inflammatory treatment split.

**`post_acne_mark` = one combined "scarring" bucket** covering both flat
pigmentation marks and textural scarring. We deliberately do **not** split
flat-vs-textural: you can't reliably separate them from a single flat-lit phone
photo — even a dermatologist reaches for side-lighting or palpation. That's an
information ceiling, not a data gap, so one class is the honest v1 model.

It maps to the existing `hyperpigmentation` concern (**no schema change**):
- The topical pathway (`RULES.md`: niacinamide, vitamin C, azelaic acid, SPF)
  genuinely helps the **pigmentation** part.
- A mandatory **texture advisory** is attached: "textural scarring may need a
  professional (procedures — topicals won't resolve texture)." Mirrors the
  cystic see-a-pro discipline for the part topicals can't fix.

**Relationship to D-012:** this partially advances it — `post_acne_mark` is a
first, small, self-labeled pigmentation signal — but it is **not** the full
hyperpigmentation detector. Coverage is partial (only marks the detector happens
to box, §4.2), and comprehensive mark/scar assessment stays deferred (D-012).

## 4. Architecture

Three modules + one notebook. Three of the four moving parts are pure and
deterministic — testable locally with **no GPU and no weights**. Only the
classifier needs the model, and it hides behind an interface so the pipeline and
its tests run without it. That is D-007's discipline applied to Stage 2.

```
src/classification/
  regions.py      # landmarks -> region + off-face filter   (pure geometry)
  classifier.py   # crop+context -> 5-class probs            (ML wrapper + stub)
  assemble.py     # analyze(image, boxes) -> ConcernReport   (orchestration)
notebooks/
  02_type_classifier.md   # Colab training curriculum (mirrors notebook 01)
```

`severity` is a pure function living inside `assemble.py` (~10 lines) rather than
its own file. <!-- ponytail: fold severity into assemble; split to severity.py only if thresholds grow per-region-specific -->

### 4.1 regions.py — where (MediaPipe, no ML)

- Run MediaPipe Face Mesh **once per image** → 468 landmarks.
- Assign each box's **center** to a region using explicit landmark-anchored
  rules (auditable — matches the repo's "trust lives in readable logic"
  philosophy), not a nearest-centroid black box:
  - `forehead`: above the eyebrow line
  - `nose`: nose bridge/tip band, between the eyes
  - `left_cheek` / `right_cheek`: below eye, beside nose, split by face midline
  - `chin_jaw`: below the lower lip
  - `perioral`: within a radius of the mouth landmarks
- **Off-face rejection:** any box whose center falls outside the Face Mesh
  face-oval → dropped (stubble, hairline, neck moles gone for free).
- **No face found** → return a sentinel that `assemble` turns into the no-face
  branch (§4.3). Do not crash; bad angle/lighting is a real, common input.

Region vocabulary is the locked closed set from `CONCERN_SCHEMA.md`. Ceiling:
coarse box-center assignment can misplace a lesion sitting exactly on a region
boundary — acceptable, since the recommender only localises advice. <!-- ponytail: box-center assignment; upgrade to polygon containment only if boundary errors show up in eval -->

### 4.2 classifier.py — what type (the ML piece)

**Inference wrapper.** `LesionClassifier.predict(crop) -> probs` over the 5
classes. A `StubClassifier` with the same interface returns fixed probs so
`assemble` and its tests run with no weights.

**Crop extraction.** `crop_with_context(image, box, pad=1.5, size=112)`:
- Pad the box by a factor (default **1.5×** — ACNE04 boxes are already loose per
  D-010, so modest), square it, resize to `size`.
- **Replicate-pad** when a box sits at the image edge (don't shift the box).
- `pad` and `size` are **config values**, tuned on the self-labeled val set. The
  padding factor is the knob that decides how much surrounding morphology
  (pustule head, whitehead center, ring of erythema) the model gets to see.

**Backbone.** Transfer learning, mirroring D-018's philosophy — a small
ImageNet-pretrained backbone (MobileNetV3-small or ResNet18), head reconfigured
to 5 classes, fine-tuned at low LR. Small + fast + Colab-T4-friendly.

**Training = hybrid** (user decision):
1. **Pretrain** the backbone on the Kaggle acne-type dataset
   `tiswan14/acne-dataset-image` (the set behind the zulqarnain11 notebook —
   **verified 2026-07-05**): 2778 train images, 5 lesion-type classes
   (Blackheads 735 / Cyst 645 / Papules 621 / Pustules 584 / Whiteheads 193),
   mapped onto our comedonal / inflammatory / cystic. Its value is *features*,
   not plug-and-play weights — the notebook's own from-scratch Keras CNN is
   not used.
2. **Fine-tune** on the **self-labeled crop set** — a few hundred lesion crops
   from running Stage 1 on **ACNE04** images (train-eligible), hand-sorted into
   the 5 classes. This is the **real deliverable of the stage**: it is the only
   *train* data matched to the crop domain the model sees at inference.

**Train vs test sources (D-014 is load-bearing here):**
- **Train / fine-tune crops → ACNE04 detections only.** ACNE04 is
  train-eligible; fine-tuning on real detector crops fixes the whole-image →
  crop mismatch.
- **Self-collected phone-photo crops → TEST/VAL ONLY, never train (D-014).**
  They are the held-out set that measures the *second* mismatch — ACNE04 photos
  vs real phone photos (domain gap). Mixing them into training would both
  violate D-014 and destroy that measurement.

**Why the pretrain isn't enough alone:** the Kaggle model never saw a ~40px
lesion crop. Whole-image training → per-crop inference is the exact train/serve
mismatch that caps quality; the ACNE04-crop fine-tune closes it.

**Class-data sourcing & risks:**
- `not_acne`: cheap — random on-face non-lesion skin patches + FFHQ clear-skin
  crops (D-013).
- `cystic`: rare → class imbalance. Handle with weighted loss / oversampling;
  report per-class recall honestly.
- `post_acne_mark`: **thinnest class.** ACNE04 doesn't *label* marks/scars, but
  its images contain them — train examples come from ACNE04 detector
  false-positives that landed on a flat mark or scar, hand-relabeled
  `post_acne_mark` (self-collected ones can only serve as test, D-014). Collect
  deliberately, since this class is now worth more (§3). **Fallback:** if too few
  to learn a stable class, collapse into `not_acne` for v1 — but note the raised
  cost: we'd lose the whole pigmentation recommendation, not just a note. Decide
  from the self-labeled set's actual counts.

### 4.3 assemble.py — orchestrate → ConcernReport

`analyze(image, boxes)`:
1. `regions.assign(image, boxes)` → each box tagged with a region; off-face and
   (if no face) everything dropped.
2. For each surviving box: `crop_with_context` → `LesionClassifier.predict` →
   argmax class + confidence.
3. Drop `not_acne`. Map `post_acne_mark` → a `hyperpigmentation` concern entry
   and attach the texture advisory (§3).
4. Group the remaining lesions/marks by **(concern, region)**.
5. Per group: `lesion_count = len(group)`; `severity = derive_severity(count)`
   (count → ordinal 0–4, **per-region thresholds** from config — see caveat);
   `confidence` = mean of the group's per-crop classifier confidences.
   `overall_severity` is `max` over **acne** concerns only, so counted
   hyperpigmentation marks never inflate acne severity (already enforced by
   `schema.py`).
6. Emit one `Concern` per group → `ConcernReport`.

**No-face branch:** if MediaPipe finds no face, return a `ConcernReport` with
empty `concerns`, `low_light_flag=True`, and a note `"no face detected"`. ⚠
**Consumer touch-up needed:** the recommender currently treats empty `concerns`
as clear-skin → maintenance routine, which is wrong for "couldn't analyse". Add
a small guard in `recommend()` (or a `no_face` meta flag it respects) so no-face
returns "couldn't analyse", not a routine.

**Severity caveat:** Hayashi is a **whole-face** count scale; per-region counts
are smaller, so per-region thresholds ≠ whole-face thresholds. `derive_severity`
uses per-region thresholds (config, tuned once Stage 1 produces real
distributions). `overall_severity = max` across acne concerns is already
computed by `schema.py` and is left as-is.

## 5. Data flow

```
Stage 1 boxes ─┐
image ─────────┼─> regions.assign ─> on-face boxes+region
               │        │ (no face) ─────────────────────> no-face ConcernReport
               └─> crop_with_context ─> LesionClassifier ─> type+conf per box
                                            │
              drop not_acne;  post_acne_mark -> hyperpigmentation + texture note
                                            │
                        group by (type,region) -> count -> severity -> confidence
                                            │
                                     ConcernReport ─> Stage 3 recommend()
```

## 6. Evaluation (D-017, D-016)

- **Classifier:** confusion matrix, per-class precision/recall, PR curves — on
  the self-labeled **crop** val set (matched domain, the number that counts).
  `cystic` and `post_acne_mark` recall watched specifically (rare/thin classes).
- **Fitzpatrick-disaggregated** error where the self-labeled set has the tone
  coverage for it (D-016). Note honestly if the set is too small to disaggregate
  meaningfully — that itself is a finding.
- **Region assignment:** eyeball first (render boxes colored by assigned region
  on N faces) before trusting any aggregate — same "look before metrics" rule as
  Stage 1.

## 7. Error handling

| Case | Behaviour |
|------|-----------|
| No face detected | no-face `ConcernReport` (§4.3), recommender says "couldn't analyse" |
| Box outside face oval | dropped as off-face FP |
| Model weights absent | `StubClassifier` (tests) / hard error in prod path — do not silently mislabel |
| Crop at image edge | replicate-pad; if degenerate, low confidence |
| `post_acne_mark` class thin | fallback: collapse into `not_acne` (§4.2) |

## 8. Testing (ponytail: one runnable check per non-trivial unit)

- `regions.py`: synthetic landmark sets → assert a box lands in the expected
  region; a box outside the oval is dropped; no-face returns the sentinel.
- `derive_severity`: assert the count→ordinal **boundaries** (0/1 and each
  threshold edge).
- `assemble.py`: fake boxes + `StubClassifier` → assert a schema-valid
  `ConcernReport` that `recommend()` consumes without error. This is the
  end-to-end contract test, and it needs no GPU.
- Classifier quality lives in the notebook's eval (§6), not a unit test.

## 9. New dependencies

- `mediapipe` (region assignment). Add to `requirements.txt`.
- `torch`, `torchvision` (classifier) — already anticipated for Stage 1.

## 10. Decisions (resolved at review — 2026-07-05)

1. **Taxonomy split** — ✅ CONFIRMED. blackheads→`comedonal`, papules→
   `inflammatory`; keep the treatment split.
2. **Mark/scar handling** — ✅ RESOLVED. `post_acne_mark` = combined
   pigmentation+texture "scarring" bucket → existing `hyperpigmentation` concern
   (no schema change) + mandatory texture advisory (§3). Not split flat-vs-
   textural (information ceiling).
3. **Thin-class fallback** — ✅ CONFIRMED. Collapse `post_acne_mark`→`not_acne`
   only if too few crops to learn a stable class; raised cost acknowledged (§4.2).
4. **Region method** — ✅ RESOLVED. Landmark-anchored rules (auditable), not
   nearest-centroid.

## 11. Consumer touch-ups this stage requires (outside `src/classification/`)

- `recommend()` — add a no-face guard so empty `concerns` + no-face flag returns
  "couldn't analyse", not a maintenance routine (§4.3).
- `RULES.md` / `engine.py` — a texture-advisory note string for the
  `hyperpigmentation`-from-`post_acne_mark` path (§3). Small, additive.
- `configs/default.yaml` — `crop_pad`, `crop_size`, and per-region severity
  thresholds (§4.2, §4.3).
- `requirements.txt` — add `mediapipe` (§9).
