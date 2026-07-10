# Concern-efficacy recommender — design

Date: 2026-07-10. Status: approved by maintainer (brainstorming session).
Supersedes the *direction* of the ranker work after plan 013's gate failure;
does not supersede the gate discipline itself, which this design extends.

## Why this redirect

Plan 013 trained a ranker on the ~1.1M Sephora reviews and it failed its own
D-022 acceptance gate; the seven-probe investigation
(`plans/ranker-v2-probe-evidence.md`) showed the failure is structural — the
existing feature columns cannot beat pooled product statistics, and
per-skin-type conditioning actively hurts. Plan 014 ships the honest fallback
(pooled `StatsRanker`).

The maintainer's goal is a system that *genuinely learns what works* and
consumes the verbose CV report (concern types, regions, severity) end-to-end,
instead of the report data decorating a routine chosen by five hardcoded
ingredient lists and ordered by a pooled review average.

The one untapped source of new signal in the data already on disk: **review
text**. ~926k reviews have `review_text`, and text is the only place where
*product × acne-type outcome* exists ("cleared my blackheads", "made my cystic
acne worse", "broke me out"). Probe 6 tested text only for pooled ordering;
concern-conditioned efficacy was never probed. This is also the PRD's own
declared v2 ("review-text NLP mining for concern-specific efficacy signals",
issue #1 Out of Scope list).

## Decision summary

Mine review texts into per-review (concern, outcome) labels via a one-time
LLM-assisted labeling pass; aggregate into per-product × concern efficacy
stats; bake off a trained model against concern-conditioned stats against the
pooled champion under the existing ratchet discipline; ship the winner behind
an inverted engine contract — **the scorer selects (whole catalog, conditioned
on the report's concern mix), the rules veto and structure** (safety,
comedogenic, pregnancy, slots, escalations). Rules stop being the selector but
remain non-negotiable as constraints.

## Architecture

### Offline pipeline (one-time, then cached; ordered go/no-go gates)

1. **Feasibility probe** (free). Word-boundary regex lexicon over
   `review_text` + `review_title`, mapping mentions to the closed concern
   vocabulary plus `acne_general` (unspecific "acne/breakout/blemish"
   mentions). Outputs: match counts per concern, per-product × concern
   cell-size histogram. **Gate P1:** ≥300 catalog products with an n≥15 raw
   mention cell for at least one acne concern. Below that, cells are too thin;
   the redirect stops and plan 014's StatsRanker is the end state.
2. **Calibration sample.** ~2k prefiltered texts LLM-labeled; maintainer
   hand-checks ~50. **Gate P2:** ≥30% of prefiltered texts yield a usable
   (concern, outcome ∈ {helped, worsened}) label, AND hand-check agreement
   ≥85%. Cost so far: ~$1.
3. **Full labeling pass.** All prefiltered texts (est. 200–400k) through the
   Anthropic Batch API (Haiku), strict JSON schema per review:

   ```json
   {"labels": [{"concern": "acne_comedonal|acne_inflammatory|acne_cystic|acne_general|hyperpigmentation|dryness",
                "outcome": "helped|worsened|unclear",
                "reviewer_has_condition": true}]}
   ```

   Texts truncated to ~1,200 chars. The prompt handles negation ("did NOT
   break me out") and attribution ("bought for wrinkles but it cleared my
   acne" → acne helped). Keyed by review row id; cached to
   `data/processed/review_concern_labels.parquet`; resumable — a crashed or
   re-run batch never re-bills already-labeled rows. Estimated one-time cost
   $30–80. Nothing downstream ever touches the API.
4. **Aggregation.** `data/processed/concern_stats.json`: per product ×
   concern → `{n, helped, worsened, help_rate, smoothed_score}`, skin_type
   sub-cells where n permits, per-concern global priors. Bayesian-m smoothing
   (same pattern as review_stats), with a fallback ladder:
   concern cell → `acne_general` cell → pooled rating (the StatsRanker
   score). A product with zero text evidence degrades to today's score —
   never undefined.
5. **Bake-off** (new decision **D-023**; same ratchet discipline as D-022).
   Reviewer-disjoint md5 split reused from plan 013. Eval on held-out
   reviewers' concern-labeled reviews: label = helped(1) vs worsened(0)
   (unclear dropped); metrics = pooled ROC-AUC + within-reviewer×concern
   pairwise ordering. Candidates:
   1. pooled StatsRanker (incumbent champion — the floor),
   2. concern-conditioned smoothed stats (no trained model),
   3. trained model (HGB or LightGBM over product actives/category/brand/
      price + concern & skin profile + efficacy aggregates).

   The winner ships. **Gate P3: the engine inversion ships only if some
   candidate beats the pooled champion on BOTH metrics.** If nothing does,
   the truth was learned for the price of the labeling run; StatsRanker
   stays and the engine keeps its v2 contract.

### Inference (local, deterministic, no API)

Verbose report + profile → the shipped scorer ranks catalog products per
category, conditioned on the report's actual concern mix and severities →
rules veto and structure. Cystic/severe and clear-skin paths keep today's
behavior byte-identical (the scorer never runs there). Step skeleton stays
cleanser → treatment → serum → moisturizer → SPF.

## Engine v3 (the inversion)

Today's `_build_routines` has one selection line (`if not matched: continue`)
and one sort key; v3 changes admission and ordering, everything else stays.

**New duck-typed interface** (dated D-005 amendment):

```python
class ConcernScorer:
    def score(self, product, report, profile) -> float          # higher = better
    def has_evidence(self, product_id, concerns) -> bool        # admission check
    def evidence(self, product_id, concern, skin_type) -> dict | None
```

Per category:

1. Target actives from concerns compute exactly as today (the dermatology
   prior; seeds the slot skeleton).
2. Candidates = active-matched products (as today) **plus** products the
   scorer admits on review evidence (concern cell n ≥ `min_cell_n`), capped
   at `admit_top_k` per category.
3. **Vetoes become product-level.** Pregnancy drops any retinoid-containing
   product outright (active-stripping alone would be bypassed by
   evidence-admission — a safety hole this design closes explicitly).
   Comedogenic partition remains the dominant sort term, unbeatable by any
   score.
4. Slot assignment runs over the union of target actives + admitted
   products' actives through the existing `_assign_slots` machinery
   unchanged (retinoids pin PM, conflicts split, SPF AM-only).
5. Sort key: `(len(p.comedogenic_flags), -scorer.score(p, report, profile))`.

### Degradation ladder

Each rung is exactly today's behavior at that rung:

| Artifacts present | Behavior |
|---|---|
| concern_stats (+ optional model bundle) | v3: admission + concern-conditioned ordering |
| pooled review_stats only | v2: StatsRanker reorders within rules-gated candidates, no admission |
| none | rules-only stable order (D-019) |

`load_scorer()` returns `ConcernScorer | StatsRanker | Ranker | None`; the
engine capability-checks (`has_evidence` present → v3 path).

## Report integration

Per-product "why" gains the concern line — *"31 reviewers with blackheads —
84% said it helped"* — via `evidence(product_id, concern, skin_type)` with
ladder tags, so a pooled fallback is never rendered as concern-specific.
Evidence-admitted products carry an explicit *"surfaced by reviewer evidence,
not ingredient rules"* note. The report's evidence → reasoning →
recommendation shape is unchanged.

## Testing

Fast suite stays pure-Python (no API, no ML weights):

- **Prefilter:** regex unit tests per concern bucket.
- **Labeling CLI:** takes an injectable labeler callable; tests use a stub
  returning canned JSON — covers cache/resume (a crashed batch never
  re-labels), schema validation, malformed-LLM-output handling (skip+count).
- **Aggregation:** dict-literal tests for smoothing math, ladder fallback,
  skin_type sub-cells, min cell sizes.
- **Bake-off harness:** engineered fixture where concern-conditioning
  provably wins; asserts the harness detects the winner and writes the eval
  artifact.
- **Engine v3:** hand-built concern stats — comedonal report ranks the
  blackhead-evidenced product first; cystic path byte-identical; pregnancy
  vetoes an evidence-admitted retinoid product; comedogenic beats any score;
  StatsRanker-only reproduces v2 order exactly; `None` → rules-only.
- The real labeling pass and bake-off are empirical; their tables get
  recorded like plan 013's gate outcome.

## Paper trail

- **D-005 dated amendment:** the hook may *admit* candidates (not just
  reorder), with the veto powers enumerated (comedogenic partition,
  pregnancy product-veto, cystic/clear-skin paths, slot rules).
- **New D-023:** labeling methodology, gates P1–P3, bake-off protocol,
  ratchet.
- **CONTEXT.md:** glossary entries for concern-stats and scorer; ranker entry
  updated.
- **Config keys** (all consumed, none speculative):

  ```yaml
  concern:
    labels_path: data/processed/review_concern_labels.parquet
    stats_path: data/processed/concern_stats.json
    eval_path: runs/concern/eval.json
    labeling_model: claude-haiku-4-5-20251001
    text_truncate_chars: 1200
    smoothing_m: 20
    min_cell_n: 15        # evidence-admission threshold
    admit_top_k: 3        # max evidence-admitted extras per category
    prefilter: {<concern>: [<word-boundary terms>], ...}
  ```

## Sequencing & scope

1. Plan 014 (StatsRanker floor) executes unchanged — it is the baseline and
   the terminal fallback.
2. Plan 015: feasibility probe + labeling CLI + aggregation (gates P1, P2,
   then the full pass).
3. Plan 016: bake-off harness + real run (gate P3).
4. Plan 017: engine v3 inversion + report evidence integration (only if P3
   passes).
5. Issues #9 (analyze()) and #10 (Streamlit) build on the final contract.

## Out of scope

- Scraping or any new data source (D-015 stands).
- LLM calls at inference time (labeling is offline-only).
- Medical claims; cosmetic-concern framing (D-002) untouched.
- Collaborative filtering / user feedback loops.
- Region-conditioned product choice: regions stay evidence/report material —
  which cheek a blackhead is on does not change the product.
