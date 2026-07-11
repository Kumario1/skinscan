# DECISIONS

Locked decisions for SkinScan. Same discipline as the GGC repo: once a decision
is LOCKED, don't silently reverse it — if it needs to change, edit the entry and
note the change. Each decision records the choice, the reasoning, and what it
rules out.

Status legend: **LOCKED** (decided, build on it) · **OPEN** (needs resolution) ·
**REVISIT** (locked for now, expected to change later).

---

## Scope & framing

### D-001 — This is a learning project, not a product · LOCKED
The goal is to learn computer vision by building a real, end-to-end system.
Optimize for lessons learned per hour, not for production-readiness, scale, or
launch. When a choice trades "more instructive" against "more shippable," pick
more instructive.

### D-002 — Cosmetic framing only; never diagnostic · LOCKED
Everything is framed as "concerns," never "conditions." The app never claims to
diagnose. No prescription-strength recommendations. This keeps us clear of
software-as-a-medical-device territory and is also just honest about what a
phone photo can support. Consequence: Stage 2 outputs "inflammatory-type acne
(appearance-based)" not "you have papulopustular acne."

### D-003 — Three concerns in scope, ranked · LOCKED
Priority order for concerns: (1) acne — location, type, severity; (2)
hyperpigmentation; (3) dry skin.
Acne is the anchor because ACNE04 gives us dermatologist-labeled data. Dry skin
is explicitly LOW priority — it's a poor single-photo visual classification
problem and will frustrate more than teach. It stays in the rules table but may
never get a real detector. See D-012.

---

## Architecture

### D-004 — Two-stage detect-then-classify, not end-to-end · LOCKED
Stage 1 detects lesions (where + how many + severity). Stage 2 crops each
detection and classifies type. This mirrors how real systems do it, gives
interpretable intermediate outputs, and lets us train the two models on
different datasets (ACNE04 for detection, Kaggle type-labeled data for
classification). Rules out a single monolithic multi-task model — worse for
learning, harder to debug.

### D-005 — Recommendation is a hybrid: rules gate, learned ranker reorders · LOCKED
**Amended 2026-07-09** (verbose-recommender milestone, issue #1): originally
"rules-based, not learned" — the concern→ingredient→product chain was a
hand-written rules table with no learned component, ruling out learned ranking
for v1. That was right while there was no data. The milestone adds a learned
ranker under a strict contract that keeps the trust model intact, so the entry
is updated per this file's own change rule — the rules-based v1 is NOT silently
reversed, it is bounded.

The recommender is now **hybrid**:
- The hand-written rules remain the auditable gate: concern → target actives →
  candidate products, conflict resolution, comedogenic down-ranking, and the
  dermatologist / pregnancy escalation flags. All correctness and trust live
  here (the DormRoom principle: trust in the auditable logic layer).
- A learned ranker may ONLY reorder rule-approved candidates *within a category*.
  It can NEVER introduce a product the rules didn't approve, override comedogenic
  down-ranking, or touch the dermatologist / pregnancy safety flags. The engine
  takes it as an optional injected object with a `score(product, profile)`
  interface, never imports sklearn itself, and treats `None` as "keep the rules
  ordering."

Ranker acceptance criteria live in D-022; the deliverable shape in D-019. Still
rules out a learned recommender that selects or gates products — the ranker only
orders what the rules already blessed.

### D-006 — The recommender reasons over ingredients, not products · LOCKED
CV output -> concern -> active ingredient(s) -> products containing them.
Products are interchangeable carriers of ingredients. This decouples the brain
(small ingredient rules table) from the catalog (swappable product CSV) and
means a stale catalog never corrupts the logic.

### D-007 — Build Stage 3 before Stages 1 & 2 · LOCKED
The recommender needs no ML — just rules and a CSV. Building it first, fed by
hand-faked detection output, forces us to lock the data contract (D-008) early,
which then defines exactly what the CV models must output. Same reasoning as
locking the GGC parser-registry contract before writing parsers.

---

## Data contracts

### D-008 — Concern schema is the contract between ML and rules · LOCKED
The interface between Stage 2 and Stage 3 is a fixed JSON schema (see
`docs/CONCERN_SCHEMA.md`). The CV side must produce it; the rules side consumes
only it. Neither side reaches across. Changing this schema is a deliberate,
logged act.

### D-009 — Catalog schema is fixed and ingredient-normalized · LOCKED
Products load into a fixed schema (see `docs/CATALOG_SCHEMA.md`) with a
normalized `actives` list. Raw INCI strings are parsed into canonical
ingredient IDs on import. We only normalize the ~30 actives the rules table
keys on; everything else is ignored. GGC parser-registry instinct: parse what
you use, ignore the rest.

---

## Datasets

### D-010 — ACNE04 is the anchor detection dataset · LOCKED
1,457 dermatologist-boxed facial images with severity + lesion count. Convert
to YOLO format. Known issue: boxes don't always capture full lesion extent —
we WILL look at predictions and confront this rather than trusting the labels.
Research use only; fine for D-001, revisit if this ever grows legs.

### D-011 — Acne-type labels come from Kaggle, treated as second-tier · LOCKED
Type classification (blackhead/whitehead/papule/pustule/cyst) uses Kaggle
type-labeled data. Label quality is weaker than ACNE04 — treat as a Stage 2
experiment, not a foundation. Validate labels by eyeballing before trusting.

### D-012 — Hyperpigmentation & dry skin have no good public dataset · OPEN
No clean public dataset for cosmetic hyperpigmentation or dry skin on otherwise
-normal faces. Options: weak Kaggle "skin concerns" sets, self-labeling, or
deprioritizing. Leaning toward: hyperpigmentation gets a small self-labeled
set later; dry skin stays rules-only (D-003). **Open until we commit.**

### D-013 — FFHQ provides the clear-skin negative class · LOCKED
Reuse FFHQ (already known from the camera project) as clear-skin negatives, so
the detector doesn't learn "every face has acne," and for false-positive-rate
testing.

### D-014 — Self-collected photos are TEST-ONLY · LOCKED
Our own phone photos (bathroom lighting, front camera) are a held-out test set
to measure domain gap. NEVER train on them. Even ~100 images as pure test data
tells us how badly ACNE04-trained models degrade in the wild.

### D-015 — Product catalog: Kaggle Sephora dataset · LOCKED
~8k products with ingredient lists + categories. Prices are stale — we don't
care (D-001, no live pricing). Rules out scraping retailers: no learning
benefit for CV, and it violates their ToS (unlike GGC's apartment complexes,
retailers gain nothing from our traffic).

**Extended 2026-07-09** (verbose-recommender milestone, issue #1): the same
Kaggle Sephora dataset (`nadyinky/sephora-products-and-skincare-reviews`)
supplies not just the catalog but ~1.1M reviews carrying per-reviewer skin_type,
skin_tone, rating, and is_recommended. Those reviews become the ranker's
training data (D-022) and feed the report's per-product review-stats lines. One
dataset, two uses; scraping remains ruled out.

---

## Evaluation

### D-016 — Skin-tone-disaggregated evaluation is mandatory · LOCKED
Report error rates per Fitzpatrick skin-tone group, not just headline numbers.
Skin-tone bias is the documented failure mode of these models. This is the
Lesson-2 disaggregation principle (summary metrics compress failures) applied
to a fairness axis, and it's the single most instructive eval here. Use
Fitzpatrick17k for tone labels.

### D-017 — Detection uses mAP/IoU; classification uses the usual suite · LOCKED
Stage 1 (detection): mAP, IoU, per-severity breakdown. Stage 2
(classification): the confusion-matrix / precision / recall / PR-curve suite
from the curriculum. Expect (per published results) decent single-class
detection but weak multi-class severity — we'll reproduce and reason about that.

---

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

---

## Verbose hybrid recommender milestone (issue #1)

### D-019 — Deliverable: Streamlit app over an importable pipeline · LOCKED
All analysis logic lives in importable modules; a CLI produces the report as
`report.json` + `report.md`, and the Streamlit app is a thin wrapper (cached
model loaders, intake form, markdown render, raw-JSON expander) with zero
business logic. CLI and app produce the identical report for identical inputs.
Rules out business logic in the UI layer; makes the whole analysis scriptable
and testable without a browser. Missing ranker artifacts (D-022) → degrade to
rules-only ordering (D-005), never fail.

### D-020 — Face regions: MediaPipe face-landmark polygons, grid fallback · LOCKED
Lesion boxes get a region label from the closed D-008 vocabulary (forehead,
nose, left/right cheek, chin_jaw, perioral) by point-in-polygon against
MediaPipe face landmarks — implemented with the Tasks FaceLandmarker (same
468-landmark topology FaceMesh used). A centroid outside every polygon snaps
to the nearest one and is reported in metadata (`forced_assignments`), never
silently. No face detected or MediaPipe unavailable → fall
back to a deterministic image-thirds grid, loudly flagged in the report. This
decides only *how* a box gets a region, not *which* regions exist (that stays
D-008). Keeps tests and constrained environments from hard-failing on the
landmark dependency.

### D-021 — Inference-time user profile: skin type, tone, pregnancy · LOCKED
At inference the user gives a short profile: skin_type required, from the closed
set combination/dry/normal/oily (matching the review vocabulary, D-015 / D-022);
skin tone estimated from non-lesional skin via ITA in CIELAB and bucketed
light/medium/deep, framed as triage not a claim (D-002), with self-reported tone
overriding the photo estimate; a pregnancy/nursing flag excludes retinoids with
conservative cosmetic framing (D-002). Intake asks only questions something
downstream consumes. Config carries the vocabulary, ITA cutoffs, and low-light
threshold.

### D-022 — Ranker acceptance criteria · LOCKED
The learned ranker (D-005) is an sklearn HistGradientBoostingClassifier
predicting is_recommended from product features × reviewer profile, trained on
the Sephora reviews (D-015) with a reviewer-disjoint deterministic split. It
ships ONLY if it beats BOTH a global-popularity baseline AND a
Bayesian-smoothed-rating baseline, on ROC-AUC and within-reviewer pairwise
ordering accuracy. Metrics are disaggregated by skin-tone bucket, including an
`unknown` bucket that is always reported, never dropped (D-016 discipline on the
recommendation axis). Fails the gate → ship rules-only (D-005 / D-019).

---

- **D-012** (hyperpigmentation/dry-skin data) — NON-BLOCKING. Acne path is
  fully unblocked; resolve when we get there.
- **Q-A** — Detection model choice. **RESOLVED → see D-018.**
- **Q-B** — How severity is represented in the concern schema (ordinal 1–4 vs
  continuous). Affects D-008. Resolve before finalizing schema. Currently
  drafted as ordinal 0–4 matching ACNE04.

## D-023 — Concern-efficacy labels: LLM-mined review text is the new ranking signal (2026-07-10)

**LOCKED.** Spec: `docs/superpowers/specs/2026-07-10-concern-efficacy-recommender-design.md`.
**AMENDED 2026-07-10 (operator constraint):** The operator authorized
OpenRouter with roughly $9 credit instead of Anthropic. Qwen3 235B A22B uses
10-review structured calls plus durable local spooling and identical-request
caching; the P2 hand-check explicitly validates the resulting attribution
tradeoff before the full pass can be approved.
Review texts are the only place *product × acne-type outcome* exists in the
D-015 dataset. A one-time OpenRouter pass (Qwen3 235B A22B, grouped
structured outputs) labels prefiltered reviews with (concern, outcome ∈
helped/worsened/unclear);
labels are cached locally (`review_concern_labels.jsonl`) and the API is never
called at inference or in tests. Ordered go/no-go gates: **P1** mention
density (executed 2026-07-10 — **PASS**, 970 catalog products with an n≥15
acne-concern cell vs the 300 floor); **P2** calibration (≥30% outcome-bearing
yield on a ~2k sample AND ≥85% maintainer agreement on a 50-review
hand-check); **P3** the bake-off (a concern-conditioned candidate ships only
if it beats the pooled StatsRanker champion on BOTH pooled metrics under the
D-022 harness — else the engine keeps its v2 contract). Aggregates live in
`concern_stats.json` (Bayesian-m smoothing toward per-concern priors,
skin-type sub-cells, fallback ladder concern → acne_general → pooled rating).

## D-024 — Ingredient KB + tier-2 catalog from beautyapi (CC-BY-NC-4.0) (2026-07-10)

**LOCKED.** Spec: `docs/superpowers/specs/2026-07-10-ingredient-kb-design.md`.
The `thebeautyapi/beautyproducts` HuggingFace dataset (~1k products with
per-ingredient comedogenicity/irritancy/functions/actives-rating) is licensed
**CC-BY-NC-4.0 (non-commercial)**. This is acceptable for this project's
research/portfolio use (consistent with D-001/D-010 research-use posture); a
commercial deployment would need the paid Beauty API or a differently-licensed
source. The raw file is a documented manual download into
`data/raw/beautyapi/beauty_data.jsonl` (gitignored, like the Sephora data) —
tests never touch it and run entirely on `tests/fixtures/beautyapi_sample.jsonl`.
Two derived artifacts: `ingredient_kb.json` (normalized-name → aggregated
metadata; max comedogenicity/irritancy on conflict, union of functions, "direct
actives" beats "supporting", collected aliases) and `catalog_tier2.json` (same
`Product` schema plus `tier: 2` / `no_outcome_data: true`; products not mapping
to the five catalog categories are dropped). The KB feeds an optional catalog
enrichment (KB-derived comedogenic flags superset the hand-list; per-product
`ingredient_match: {concern: float}`) and a pure `match_score`. **Honesty
property preserved:** ingredient-match is only a RANKING TIEBREAKER — review-
backed concern-stats (D-023) dominate, tier-2 fills a slot only when no tier-1
(review-backed) candidate exists, and the `no_outcome_data` flag carries
through. Without the KB file the importer is byte-identical to before
(regression-tested).

## D-025 — Revert to the 5-class classifier; 6-class retrain has a crop-domain confound (2026-07-10)

**LOCKED (until issue #5 retrain v2).** The Colab-retrained six-class model
(`acne_model_6class_v1_confounded.keras`, archived) predicts Not_acne ≈1.0 for
EVERY real detector crop — including ACNE04 lesions — while scoring perfectly
on AcneDataset test crops. Root cause: the five acne-class positives are
640×640 Roboflow mosaic images, but the Not_acne negatives were harvested
through the inference-time `crop_with_context` transform (224px padded,
upscaled detector crops). The model learned crop style, not acne; every
pipeline crop lands in the negative's domain. The D-022/issue-#5 acceptance
gates (5-class macro-F1 on dataset crops, FFHQ holdout reject, phantom-image
flip) all pass while the deployed pipeline is 100% broken — they cannot see
the confound.

Decision: `models/classification/acne_model.keras` is the ORIGINAL 5-class
model again (classes Blackheads/Cyst/Papules/Pustules/Whiteheads, sidecar
`.labels.json` records this). The engine still tolerates a missing Not_acne
class (bridge top-1 rejection is inert, issue #4 forward-compat unchanged).
Retrain v2 must (a) pass positives through the SAME `crop_with_context`
transform (ACNE04 ground-truth boxes) so both classes share the crop domain,
and (b) add an acceptance gate on REAL pipeline crops: on a known-acne image,
the share of detector crops classified Not_acne must stay below a threshold
(e.g. < 50%), which v1 fails at 100%.
