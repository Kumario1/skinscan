# recsys architecture

Standalone rebuild of the SkinScan recommendation half. Design decisions:

- **Coupling**: three file contracts only — reads `analysis.json` (schema 3) +
  `profile.json`, writes `recommendations.json`. Zero imports from `src/`.
- **Catalog = product details only.** Every ranking signal (reviews,
  popularity, ingredient analysis, concern efficacy, media) lives in its own
  versioned store, discovered through a registry. Signals are extensible:
  adding one = a build tool + a store file + a registry entry + one provider
  class in `signals.py`. Zero edits to pipeline/gates/composer.
- **LLM/AI only in batch tools** (`recsys/tools/`), cached and versioned —
  never at inference (enforced by `tests/test_no_network.py`).
- **Safety is structural, never score-driven**: hard vetoes with deterministic
  reason codes, session rules in the composer, D-002 cosmetic framing and
  doctor-referral passthrough in every output.
- **Cosmetic framing only** (no OTC-drug therapy layer): the treatment/serum
  slots recommend cosmetic products whose actives target the detected
  concerns; prescription-strength paths are a doctor conversation, per D-002.

## Data stores

| Store | Location | Committed? | Source / update | Freshness |
|---|---|---|---|---|
| Raw dump | `data/raw/sephora/*.csv` (repo level; main checkout only) | No | Kaggle snapshot, immutable | sha256 recorded in every derived artifact |
| Catalog full | `recsys/data/derived/catalog_full.json` | No | `tools/build_catalog.py` | rebuilt on importer/dump change |
| Catalog seed | `recsys/data/catalog/seed_catalog.json` + `seed_ids.txt` | **Yes** | `build_catalog.py --only-ids` | byte-identical rebuild is a test (`raw_dump` marker) |
| Knowledge | `recsys/data/knowledge/*.json` | **Yes** | hand-authored, PR-reviewed | tests pin the safety invariants |
| Review stats | `recsys/data/signals/review_stats.v1.json` | **Yes** (seed scope) | `build_review_stats.py`: n, mean, Bayesian-smoothed rating (m=20), per-skin-type cells (n≥20) | static per dump |
| Popularity | `recsys/data/signals/popularity.v1.json` | **Yes** (seed scope) | `build_popularity.py`: loves + full-dump same-category percentile | marked `snapshot-2023` |
| Ingredient analysis (Phase 1) | `signals/ingredient_analysis.v1.json` + JSONL cache | Store yes / cache no | LLM batch over INCI, keyed `(product_id, inci_sha256, prompt_version)` | re-labeled only on key change |
| Concern efficacy (Phase 2) | `signals/concern_efficacy.v1.json` + cache | Store yes / cache no | port of D-023 review mining; cells carry n | static per dump |
| Media/editorial (future) | `signals/media.v1.json` | when built | AI research + verification loop | freshness window via overlay |
| Verification overlay (Phase 3) | `recsys/data/verification/` | **Yes** | port of `src/recommendation/verification_loop.py` state machine | per-fact stale-flip windows |
| Signal registry | `recsys/data/signals/registry.json` | **Yes** | written by each build tool | engine refuses sha256-mismatched stores |

**Verification split**: mechanical facts (name, price, INCI, review stats) come
straight from the dump — provenance says so, no verification needed. Facts
backing safety claims (`spf`, `broad_spectrum`) are name-parsed in v0 and
marked `spf_source: "name_parse"` / `spf_value_from_name_parse_not_verified`
until the Phase 3 overlay upgrades them. LLM-derived stores are scoring
signals only, always labeled model-derived — safety gates key off the
deterministic INCI parser + `knowledge/safety_rules.json`, never LLM output.

## Pipeline stages (`pipeline.py`)

1. **load** — validate analysis/profile/catalog/registry; resolve profile
   precedence (file > `analysis.input_profile` > unknown; unknowns fail SAFE);
   sha256 everything. `contract_violation:<field>` on failure.
2. **select_targets** — concerns severity ≥ 1, ordered by (severity,
   confidence); `acne_cystic` flagged `referral_emphasis`.
3. **generate_candidates** (`candidates.py`) — carrier slots take the whole
   category; treatment/serum need an active matching a target concern. No
   targets (clear skin) → carrier-only maintenance routines.
4. **apply_gates** (`gates.py`) — deterministic reason codes, never
   score-overridden: `retinoid_pregnancy_status_excluded` (pregnant/trying/
   nursing/**unknown**), `profile_allergy:<x>`, `duplicates_current_active:<x>`,
   `spf_below_30_or_unknown`, `price_above_profile_cap`. Triage `derm_first`/
   `abstain` short-circuits to a `referral_only` document before any of this.
5. **score** (`signals.py` + `scoring.py`) — providers return
   `SignalScore(value 0..1, evidence, details)` or `None` (= neutral 0.5 + an
   uncertainty note; missing data is never a hidden penalty). Final = weighted
   mean; the full per-signal breakdown is retained (decomposability is
   tested). v0 signals: concern_fit, review_quality, popularity, price_value.
6. **compose** (`compose.py`) — archetypes are data (`archetypes.json`), not
   code branches. Session rules: SPF AM-only, retinoids PM-only, conflict
   pairs (BP×retinoid, BP×vitC, glycolic×retinoid) never share a session —
   split across AM/PM or reject the candidate. Greedy by score with
   per-candidate backtracking; budget cap via cheapest-swap loop; diversity
   guarantee vs best_overall.
7. **explain** (`explain.py`) — per-product `why` built from the same
   SignalScore objects the ranker used (no separate marketing-copy path),
   uncertainty flags, D-002 framing, referral passthrough.
8. **emit** — `sort_keys=True`, atomic write; `--generated-at` pins the only
   non-deterministic field.

## Output document (`recommendations.json`, schema `recsys-1`)

Top level: `engine` (version + git commit), `inputs` (analysis/profile
sha256s + profile source), `data_versions` (catalog/signals/knowledge
sha256s), `framing` (D-002), `triage` (level, referral reasons,
professional-review observations, see-doctor note), `status`
(`ok` | `referral_only`), `target_concerns`, `routines[5]`, `veto_log`,
`warnings`. Each routine: archetype, title, rationale, total price, `am[]` /
`pm[]` steps, `safety_checks`, notes. Each step: product identity, usage,
`why{summary, score, signals[{name, value, evidence}], uncertainty[]}` —
every number reproducible from the named store.

## Phases

| Phase | Content | Acceptance |
|---|---|---|
| 1 — Ingredient analysis | finish `tools/build_ingredient_analysis.py` (LLM batch, cache schema already real); `IngredientAnalysisSignal`; irritancy feeds the gentle archetype; comedogenic notes in explanations | ≥95% catalog coverage; deterministic reruns; no-network test green |
| 2 — Concern efficacy | port D-023 mining (`src/recommendation/concern_labels.py` + `concern_stats.py`); `ConcernEfficacySignal`; ladder concern cell → acne_general → pooled | cells for ≥300 products; "helped X% of n reviewers" evidence; weight scales with n |
| 3 — Verification overlay | port `verification_loop.py` scoped to recsys facts (SPF/broad_spectrum, discontinued flags); evidence bytes; stale-flip | `spf_source: "verified"` in output; loop never self-approves |
| 4 — Full catalog + hardening | full-scope stores in `derived/`; media-store interface; dummy-provider conformance test; golden-file eval harness | new signal addable with zero pipeline edits |
| 5 — Integration | `src/pipeline/e2e.py` optionally invokes the recsys CLI after writing analysis.json (subprocess/file handoff, no imports) | run dir carries both old `routine.json` and new `recommendations.json` |

Known v0 gaps (deliberate): no cadence data (a weekly peel can rank as a daily
treatment — Phase 3 restores label-verified cadence); no product
contraindications field (Phase 3); SPF from name-parse (Phase 3); review
signal is pooled star ratings, not concern-specific outcomes (Phase 2).
