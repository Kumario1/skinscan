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

### D-005 — Recommendation is rules-based, not learned · LOCKED
The concern->ingredient->product chain is a hand-written rules table plus a
lookup, NOT a trained recommender. This is the DormRoom principle: trust lives
in the auditable logic layer. Rules out collaborative filtering / learned
ranking for v1 (no data for it anyway, and it would hide errors).

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

- **D-012** (hyperpigmentation/dry-skin data) — NON-BLOCKING. Acne path is
  fully unblocked; resolve when we get there.
- **Q-A** — Detection model choice. **RESOLVED → see D-018.**
- **Q-B** — How severity is represented in the concern schema (ordinal 1–4 vs
  continuous). Affects D-008. Resolve before finalizing schema. Currently
  drafted as ordinal 0–4 matching ACNE04.
