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

**Note 2026-07-10:** the learned slot is currently empty — the D-022 gate
rejected the trained model, so the reorderer shipping in the hook is the
statistical champion (`StatsRanker`, see D-022 amendment). The hybrid contract
(rules gate, hook only reorders, comedogenic/safety untouchable) is unchanged.

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

**Amended 2026-07-10 (gate executed — outcome recorded):** the gate ran on the
full ~1.1M-review dataset (plan 013). The learned model FAILED it: ROC-AUC
0.659 / pairwise 0.584 vs popularity 0.672/0.597 and Bayesian rating
0.666/0.609. A seven-probe investigation (`plans/ranker-v2-probe-evidence.md`)
showed the failure is structural (no sklearn variant, review-text feature, or
per-skin-type cell passes; per-skin-type ordering measurably LOSES to pooled).
Two consequences, per this file's change rule:

1. The failure mode "ship rules-only" is amended to **"ship the statistical
   champion"**: the engine's hook carries a `StatsRanker` ordering candidates by
   the Bayesian-smoothed pooled product rating from `review_stats.json` — the
   bake-off's measured winner (pairwise 0.609). Skin-type cells remain
   evidence-only (they hurt ordering: 0.606/0.596). Rules-only remains the
   degradation when stats artifacts are absent too.
2. The gate becomes a **ratchet**: a future learned model ships only if it
   beats the champion (the Bayesian baseline row in the eval — same score) on
   BOTH pooled metrics. The trainer already enforces this: the model artifact
   is written only on a gate pass.

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

## D-026 — Identification v2: SA-RPN reads native-resolution tiles; the zoom funnel is rejected (2026-07-11)

**LOCKED (replacing the shipped two-stage pipeline is gated on a consumer-photo
check).** With the SA-RPN detector trained and served (README §7, `sa-rpn/`),
there were two candidate ways to put a full face photo in front of a model
trained on 1024px clinical tiles:

- **zoom** — keep the shipped YOLOv8m as a stage-1 gatekeeper, cluster its
  lesion boxes, upscale each context crop to the model's 1024px input, and let
  SA-RPN re-detect inside those crops;
- **tile** — chunk the photo into native-resolution 1024px tiles (minimal
  count, evenly spaced, guaranteed minimum overlap so a seam lesion is always
  fully inside some tile), run every tile, dedupe across seams.

The A/B harness (`src/pipeline/compare_sarpn.py`) ran both funnels against the
same served epoch-15 checkpoint on 5 held-out AcneSCU validation images
(seed-42 split, 756 annotated lesions), scored by greedy IoU ≥ 0.3 matching:

```text
            recall           precision        exact-label acc   concern acc
zoom        0.04 (32/756)    0.52              0.44              0.62
tile        0.70 (530/756)   0.68              0.92              0.95
```

The zoom funnel fails **twice**: YOLO at imgsz 1024 downscales hi-res photos
~4×, so it forwards only 8–18 areas per face — recall is capped by the
gatekeeper before SA-RPN ever runs; and even on forwarded areas, upscaled
blurry crops halve SA-RPN's label accuracy (0.44 vs 0.92 on native pixels).
Rules out: any funnel where the old detector decides what the new model may
see, and any upscaled-crop input to SA-RPN.

**The v2 identification pipeline is therefore:** image → native-res 1024px
tiles → SA-RPN → cross-tile class-agnostic dedupe → D-020 regions →
ConcernReport via the fixed label map (closed/open comedones →
`acne_comedonal`; papules/pustules → `acne_inflammatory`; nodules →
`acne_cystic`; atrophic/hypertrophic scars + melasma → `hyperpigmentation`;
nevus/other dropped, same posture as Not_acne rejection) → the unchanged D-008
contract into the recommender. Severity keeps the bridge's existing lesion-count
thresholds. This is the first CV path that produces a non-acne concern —
AcneSCU's scar/melasma classes partially resolve D-012's hyperpigmentation
data blocker.

Caveats recorded honestly:
- The eval is on clinical images — the tile funnel's home domain (SA-RPN
  trained on exactly such tiles) and YOLO's out-of-domain territory. Before the
  YOLOv8m+EfficientNetB0 pipeline is retired from the e2e CLI, tiling must pass
  a self-collected consumer-photo check (D-014 photos): visual proof sheet +
  concern-report sanity, same real-pipeline discipline D-025 taught us.
- Cost: ~24 API calls per hi-res image (vs one YOLO pass); serving needs the
  legacy mmdet env (Lightning studio). The shipped local pipeline remains the
  fallback wherever the API is unreachable.
- The recommendation arm of the A/B was not exercised (catalog artifacts were
  absent on the eval machine); ConcernReports were produced and differ as the
  detection numbers imply — tile reports full-face severity (e.g. sev-4
  hyperpigmentation cells) where zoom reports sev-1 fragments.

## D-027 — Production cutover: SA-RPN native tiles are the sole default identifier (2026-07-12)

**LOCKED.** `src.pipeline.e2e` now defaults to the SA-RPN native-tile
identification path described in D-026/README §7a; the YOLOv8m +
EfficientNetB0 two-stage pipeline is retired from the default CLI. This entry
does **not** rewrite D-026's historical experiment record — the tile-vs-zoom
A/B numbers and the reasoning behind locking native-res tiling stand exactly
as run and reported. It supersedes five specific forward-looking statements
D-026 made about the pipeline that would come after it, now that the
implementation is complete and committed:

1. **Scar mapping.** D-026 wrote "atrophic/hypertrophic scars + melasma →
   `hyperpigmentation`." **Superseded:** scars (`atrophic_scar`,
   `hypertrophic_scar`) now map to their own concern, `acne_scarring`; only
   `melasma` maps to `hyperpigmentation`. `acne_scarring` joins the D-008
   closed concern vocabulary (`docs/CONCERN_SCHEMA.md`,
   `src/recommendation/schema.py: CONCERNS`). See `SARPN_LABEL_TO_CONCERN` in
   `src/pipeline/sarpn.py`.
2. **Severity thresholds.** D-026 wrote "severity keeps the bridge's existing
   lesion-count thresholds." **Superseded:** the SA-RPN bridge does NOT reuse
   `concern_report.severity_count_thresholds` — that config key remains the
   historical bridge's alone (`src/recommendation/bridge.py`). The SA-RPN
   path uses its own provisional, evidence-aware table under
   `sa_rpn.severity` (`configs/default.yaml`): per-concern lesion count via
   `bisect_right` over `count_thresholds` (comedonal `[1,8,20,40]`,
   inflammatory `[1,6,15,30]`, scarring `[1,3,8,20]`, hyperpigmentation
   `[1,4,10,25]`); any `nodule` forces severity to `nodule_severity` (4); 2
   affected regions floor severity at 2, `broad_region_count` (3+) floors it
   at 3; any `hypertrophic_scar` floors it at
   `hypertrophic_scar_min_severity` (3); a max retained detection score below
   `confidence_cutoff` (0.5) caps severity at 1. See `_severity`,
   `src/pipeline/sarpn.py`.
3. **Nevus/other.** D-026 wrote "nevus/other dropped, same posture as
   Not_acne rejection." **Superseded:** `nevus` and `other` are never
   concerns, but they are not dropped — they surface as visible
   `safety_observations` in `analysis.json`, gated by
   `sa_rpn.severity.professional_review` per-label count/confidence
   thresholds, plus an `unsupported_label` observation for any SA-RPN label
   outside `SARPN_LABEL_TO_CONCERN`/`SARPN_NON_ACTIONABLE_LABELS`. Nothing
   the service returns is silently discarded.
4. **API-unreachable fallback.** D-026's caveats wrote "the shipped local
   pipeline remains the fallback wherever the API is unreachable."
   **Superseded:** there is no fallback. A transport failure, a malformed
   response, or any tile that fails validation aborts the run
   (`SarpnTransportError` / `SarpnResponseError`, raised from
   `infer_native_tiles`/`_validated_detections` in `src/pipeline/sarpn.py`);
   `src.pipeline.e2e` exits non-zero and prints the sanitized failure to
   stderr. Identification runs before any output is staged, and a successful
   run publishes through an atomic staged swap (`_publish_staging`), so a
   failure never touches — and a success never partially overwrites — a
   previously published output directory. The YOLOv8m + EfficientNetB0
   pipeline (README §1-§4) is retained in the repository only as a
   **historical, evaluation-only** reference; `src.pipeline.e2e` does not
   import or call it (see
   `tests/test_e2e.py::test_importing_default_e2e_loads_no_legacy_models` for
   the full forbidden-import set: `ultralytics`, `tensorflow`,
   `src.classification.classifier`, `src.classification.run_acne04_pipeline`,
   `src.recommendation.bridge`, `src.recommendation.ranker`,
   `src.recommendation.concern_stats`).
5. **Concern contract.** D-026 wrote "the unchanged D-008 contract into the
   recommender." **Superseded:** the contract gained a V2 shape while keeping
   the D-008 vocabulary discipline — one `Concern` entry per concern (not per
   concern-region pair), carrying an aggregated `regions` list and an
   `evidence` block (`labels`, `max_confidence`, `affected_region_count`);
   `region` (singular) survives only as an internal backward-compatibility
   field, not part of the V2 JSON. `docs/CONCERN_SCHEMA.md` documents the V2
   shape as the default; the historical two-stage pipeline's bridge still
   produces the old one-entry-per-(concern, region) shape.

Also recorded, not previously stated in D-026:

- The recommendation engine became evidence-aware for the V2 contract
  (`docs/RULES.md` §5/§7): low-confidence concerns now contribute no actives
  at all (flag-only, where they previously still listed the ingredient under
  a "verify" tag); broad `acne_inflammatory` evidence (≥3 affected regions)
  can de-stack `benzoyl_peroxide` in favor of `azelaic_acid` when a real
  azelaic product exists; `acne_scarring` gets ceramides barrier support,
  mandatory SPF, and a professional-review flag above severity 3 or on
  `hypertrophic_scar` evidence; `hyperpigmentation`'s first-line actives
  dropped `vitamin_c` (`azelaic_acid, niacinamide` only); deep-tone guidance
  fires from the reported concern set independent of confidence; and
  `"unknown"` is a first-class `TONE_BUCKETS` member that stays neutral
  (never triggers deep-tone guidance) rather than being treated as risk.
- Production `src.pipeline.e2e` always calls `recommend(..., ranker=None)` —
  the deterministic rules-only order ships by default. The duck-typed ranker
  hook and `StatsRanker` (D-022) remain available to
  `src.recommendation.ranker` / standalone bake-off evaluation but nothing in
  the production e2e CLI activates them.
- The automated cutover gate is
  `tests/test_e2e.py::test_fixture_e2e_writes_complete_v2_artifact_set`: it
  exercises the full identify → region → tone → recommend → publish path
  against a fixture SA-RPN HTTP server and asserts the complete V2 artifact
  set (`analysis.json` with `schema_version: "2.0"`, optional `routine.json`,
  `detections.jpg`, `region_overlay.jpg`, `lesion_sheet.jpg`). It stands in
  for D-026's planned self-collected consumer-photo check: no real
  consumer-photo evaluation has been separately measured for this cutover, so
  no end-to-end accuracy number is claimed for the SA-RPN production
  pipeline — the detector metrics (README §7), the tile-vs-zoom A/B (§7a),
  and the recommender's rules discipline (`docs/RULES.md`) remain reported
  as separate, independent measurements, never fused into one score.

Rules out: any code path that imports or calls the YOLOv8m/EfficientNetB0
pipeline from the default `src.pipeline.e2e`; any silent local-model fallback
on SA-RPN failure; and describing D-026's own historical A/B numbers as if
they were re-measured under this cutover.
