# recsys architecture

Standalone rebuild of the SkinScan recommendation half. Design decisions:

- **Coupling**: three file contracts only — reads `analysis.json` (schema 3) +
  `profile.json`, writes `recommendations.json`. The **engine** imports nothing
  from `src/` — all 15 top-level modules, verified. The **tools** are not so
  clean: `tools/run_full_concern_pass.py` imports `src.config` and
  `src.recommendation.concern_labels`, reaching into three of its private
  functions (`_require_calibration_report`, `_configured_labeler_identity`,
  `_labeler`) to reuse the proven D-023 labeler and its budget guard rather than
  fork 1,656 lines. It is a batch path and cannot reach a recommendation — but
  it is real coupling: renaming a private in `concern_labels.py` breaks the
  Phase 2 pass.
- **Catalog = product details only.** Every ranking signal (reviews,
  popularity, ingredient analysis, concern efficacy) lives in its own
  versioned store, discovered through a registry. Signals are extensible:
  adding one = a build tool + a store file + a registry entry + one provider
  class in `signals.py`. Zero edits to pipeline/gates/composer.
- **LLM/AI only in batch tools** (`recsys/tools/`), cached and versioned —
  never at inference (enforced by `tests/test_no_network.py`).
- **Safety is structural, never score-driven**: hard vetoes with deterministic
  reason codes, session rules in the composer, D-002 cosmetic framing and
  doctor-referral passthrough in every output.
- **Cosmetic routines, prescriptions listed alongside.** The routine slots only
  ever hold cosmetic products. Prescription-strength products are catalogued and
  *listed* for a doctor conversation (D-033), never ranked into a routine —
  ranking one would assert it beats the cosmetics, and which therapy suits which
  concern is D-029 clinician-gated. See "Prescriptions" below.

## The flow, in one pass

```
  photo ──► SA-RPN detector ──► analysis.json ─┐
                                               ├──► recsys ──► recommendations.json ──► HTML
  user answers ──► profile.json ───────────────┘
```

`analysis.json` says *what the skin looks like* (concerns + severity + a triage
level). recsys never sees the photo, never runs a model, and never calls the
network. It answers one question: **given these concerns and this person, which
products are safe, and which of the safe ones are best?**

Order matters, and it is always the same:

1. **Eligibility first, ranking second.** A product is thrown out for being
   unsafe (retinoid + pregnancy) or wrong (not a daily format) *before* any
   score is computed. A high score can never rescue a vetoed product — that is
   why gates return reason codes instead of penalties.
2. **Ranking is a weighted mean of independent signals**, each of which can say
   "I don't know" (→ neutral 0.5 + an uncertainty note, never a hidden penalty).
3. **Composition applies session rules** (SPF in the morning, retinoids at
   night, never two conflicting actives in one session), then assembles five
   different routines from the same ranked pool.

Concretely, one pass, always this order — `pipeline.py` is the only place it
lives:

```
load ─► select_targets ─► generate_candidates ─► apply_gates ─►
list prescriptions & drop them ─► score ─► compose ─► explain ─► emit
```

## What each file does

**The engine** (pure, deterministic, no network — enforced by
`tests/test_no_network.py`):

| File | Does |
|---|---|
| `contracts.py` | The I/O boundary. Parses `analysis.json`/`profile.json`, resolves profile precedence (file > analysis > unknown), raises `contract_violation:<field>`. Unknowns fail *safe*, not silently. |
| `catalog.py` | Product identity only — no scores. Enforces the INCI contract: `actives` must parse out of the ingredient list, or the row is rejected. Holds the one exception, for label-stated drug actives (see Prescriptions). |
| `inci.py` | The deterministic ingredient parser. Turns an INCI string into canonical actives + comedogenic flags. Every safety gate keys off this, never off an LLM. |
| `knowledge.py` | Loads the hand-authored tables in `data/knowledge/` — which actives target which concern, which are retinoids, which conflict, the five archetypes. |
| `pipeline.py` | The orchestrator, and the only place the order above lives. Resolves and validates every input; picks targets (concerns severity ≥ 1, ordered by severity then confidence; `acne_cystic` flagged `referral_emphasis`); merges the optional drug catalog *before* applying the verification overlay, so approved facts reach drug rows too; stamps the output with the sha256 of every input and data file it used. Triage `derm_first`/`abstain` short-circuits to a `referral_only` document before any product logic runs. `emit` is `sort_keys=True` + atomic write; `--generated-at` pins the only non-deterministic field. |
| `candidates.py` | Per-slot shortlist. Carrier slots (cleanser/moisturizer/SPF) take the whole category; treatment/serum need an active that targets a detected concern. No targets (clear skin) → carrier-only maintenance routines. |
| `gates.py` | Hard vetoes with deterministic reason codes. Scores never participate and never override a veto. **HARD** safety always vetoes, in both modes: `retinoid_pregnancy_status_excluded` (pregnant/trying/nursing/**unknown**), `profile_allergy:<x>`, `product_contraindication:<x>`, `duplicates_current_active:<x>`, `ingredients_unknown` (no INCI *and* no parsed actives — nothing to reason about, so nothing can clear it), `spf_below_30_or_unknown`, `spf_not_broad_spectrum`, `price_above_profile_cap`. **SOFT** means *we haven't checked yet*, not *this is unsafe*: the eight verification-quality codes listed at `gates.py:36` (`role_not_verified`, `cadence_not_daily`, `cadence_unverified`, `treatment_label_unverified`, …). `strict` vetoes on them; `hybrid` turns them into quality flags and labels the step `category_derived`. That list is a **closed allowlist** — anything not on it is hard, so a *new* reason code defaults to vetoing rather than silently passing. Conflating the two is what made verified-only unusable; see "How it got here". |
| `signals.py` | The pluggable scoring inputs. Each provider returns `SignalScore(value 0..1, evidence, details)` or `None` — and `None` means neutral 0.5 plus an uncertainty note, never a hidden penalty. Six weighted signals: concern fit, concern efficacy, ingredient analysis, review quality, popularity, price value. Adding one touches this file only. |
| `scoring.py` | Weighted mean, weights per archetype. Every final score is decomposable back into named signals — tested, and visible in the output. |
| `compose.py` | Archetypes-as-data (`archetypes.json`), not code branches. Session rules: SPF AM-only, retinoids PM-only, conflict pairs (BP×retinoid, BP×vitC, glycolic×retinoid) never share a session — split across AM/PM or reject the candidate. Greedy by score with per-candidate backtracking, budget cap via a cheapest-swap loop, diversity guarantee vs `best_overall`. |
| `explain.py` | The "why" for each step, built from the *same* SignalScore objects the ranker used — there is no separate marketing-copy path. Also builds `prescription_options`, uncertainty flags, D-002 framing and the referral passthrough. |
| `verification.py` | The evidence overlay. Approved, hash-bound facts that upgrade a product from "inferred" to "verified". Stale evidence stops applying. |
| `evaluate.py` | Golden-file harness. Pins a full document so any behaviour change shows up as a diff. |
| `__main__.py` | The CLI. |

**The tools** (`recsys/tools/` — the *only* place network/LLM code is allowed;
all batch, cached, versioned, never at inference):

| File | Does |
|---|---|
| `build_catalog.py` | Kaggle Sephora CSV → catalog JSON (full or seed). |
| `build_review_stats.py` | Reviews CSV → n, mean, Bayesian-smoothed rating, per-skin-type cells. |
| `build_popularity.py` | Loves + same-category percentile. |
| `build_ingredient_analysis.py` | LLM batch over INCI → irritancy/comedogenic store. |
| `build_concern_efficacy.py` | Mined review labels → "helped X% of n reviewers with *this* concern". |
| `run_full_concern_pass.py` | The budget-capped Azure pass that produces those labels. |
| `import_verification.py` | Copies **already-approved** assertions in. Never approves anything. |
| `import_drug_catalog.py` | DailyMed drug rows → a recsys drug catalog (kept separate — see below). |
| `common.py` | Deterministic JSON writing + the signal registry. |

**Outside recsys**: `src/recommendation/verification_loop.py` is the state
machine that researches, ingests, reviews and approves catalog evidence;
`src/recommendation/import_dailymed.py` turns a DailyMed SPL into drug rows.
`tools/verify_e2e.py` checks all 18 stages against the real catalog;
`tools/render_routine_html.py` renders either document format for reading.

## How it got here

The order of these is the argument — each step exists because the previous one
broke on real data.

1. **Rebuild (2026-07-14).** The old `src/recommendation/` had accreted two
   engine generations plus a drug-therapy layer that dead-ended on a cosmetics
   catalog. recsys was started standalone, coupled by file contracts only, so
   the old engines could keep running untouched.
2. **Signals split out of the catalog.** The catalog holds identity; every
   ranking input lives in its own versioned store behind a registry. This is why
   a new signal costs one provider class and zero pipeline edits.
3. **Phases 1–2 added judgement the dump lacks** — what an ingredient list
   implies, and what reviewers *with the same concern* actually reported. Both
   are batch-mined and versioned; neither is trusted for safety.
4. **Phase 3 added evidence.** Facts that back a safety claim (SPF, cadence,
   whether it's even a face product) can't be guessed from a product name, so
   they need an approved, hash-bound snapshot of an authoritative page.
5. **Hybrid eligibility (2026-07-15)** — because Phase 3 was too honest. Only 13
   of 1,634 products (0.8%) were evidence-verified, so a real photo with severe
   scarring got 3 of 5 archetypes and *nothing* for the scarring. The fix was to
   notice that the gates were conflating two different things: "this is unsafe"
   and "we haven't verified this yet". Hard safety still vetoes; verification
   gaps became quality flags. Hybrid opens the whole 1,634-product catalog by
   category, labels each step `verified` or `category_derived`, and gives
   verified products a ranking nudge. Strict remains the default and is
   unchanged.
6. **Prescriptions (2026-07-15).** See below.

Governing decisions (`docs/DECISIONS.md`): **D-002** cosmetic framing, not
medical advice · **D-029** the therapy policy — which treatments exist for which
concern — is clinician-gated · **D-032** an identified agent may approve
*factual* catalog evidence after checking the source · **D-033** OTC status no
longer gates treatment eligibility, and prescription options may be surfaced
with a referral · **D-034** intended-area vetoes only a *stated* non-face area —
unknown and empty stay open, and `import_verification` refuses to silently drop
a fact the committed overlay asserts.

## Prescriptions

34 prescription acne products are catalogued from their FDA labels (DailyMed
SPLs). Three things make that safe:

- **They load under a different contract, not a weaker one.** A drug label
  publishes no INCI list, so `actives == parse_ingredients(inci)` cannot apply.
  It does something stronger: it names each active, states its exact strength,
  and cites itself — and the row is bound to those bytes by hash. Rows clearing
  *all* of that derive actives from `drug_actives`. Everything short falls back
  to the INCI rule, so a cosmetic cannot assert its way past derivation.
- **They ride in their own catalog file.** The signal stores are keyed by
  `catalog_full.json`'s sha256; merging drug rows into it would change that hash
  and silently strand every store.
- **Listed, never placed.** `prescription_options` surfaces the ones matching the
  detected concerns, with a referral note. They are read out of the gated pool
  and then *dropped from it, before ranking*, so the invariant holds by
  construction rather than by whichever way the ranking falls — which also keeps a product with no retail
  price out of every routine total. Hard safety still reaches them: a real run
  vetoes 6 Rx retinoids on pregnancy status before they can be listed.

## Data stores

| Store | Location | Committed? | Source / update | Freshness |
|---|---|---|---|---|
| Raw dump | `data/raw/sephora/*.csv` (repo level; main checkout only) | No | Kaggle snapshot, immutable | sha256 recorded in every derived artifact |
| Catalog full | `recsys/data/derived/catalog_full.json` | No | `tools/build_catalog.py` | rebuilt on importer/dump change |
| Catalog seed | `recsys/data/catalog/seed_catalog.json` + `seed_ids.txt` | **Yes** | `build_catalog.py --only-ids` | byte-identical rebuild is a test (`raw_dump` marker) |
| Knowledge | `recsys/data/knowledge/*.json` | **Yes** | hand-authored, PR-reviewed | tests pin the safety invariants |
| Review stats | `recsys/data/signals/review_stats.v1.json` | **Yes** (seed scope) | `build_review_stats.py`: n, mean, Bayesian-smoothed rating (m=20), per-skin-type cells (n≥20) | static per dump |
| Popularity | `recsys/data/signals/popularity.v1.json` | **Yes** (seed scope) | `build_popularity.py`: loves + full-dump same-category percentile | marked `snapshot-2023` |
| Ingredient analysis (Phase 1) | `signals/ingredient_analysis.v1.json` + JSONL cache | Store yes / cache no | pinned free Nemotron MoE over INCI, keyed `(product_id, inci_sha256, prompt_version)` | re-labeled only on key change |
| Concern efficacy (Phase 2) | `signals/concern_efficacy.v1.json` + cache | Store yes / cache no | proven D-023 labeler → file contract → deterministic recsys aggregate; cells carry n | static per dump |
| Verification overlay (Phase 3) | `recsys/data/verification/` | **Yes** | port of `src/recommendation/verification_loop.py` state machine | per-fact stale-flip windows |
| Signal registry | `recsys/data/signals/registry.json` | **Yes** | written by each build tool | engine refuses sha256-mismatched stores |

**Verification split**: mechanical facts (name, price, INCI, review stats) come
straight from the dump — provenance says so, no verification needed. Facts
backing safety claims (`spf`, `broad_spectrum`) are name-parsed in v0 and
marked `spf_source: "name_parse"` / `spf_value_from_name_parse_not_verified`
until the Phase 3 overlay upgrades them. LLM-derived stores are scoring
signals only, always labeled model-derived — safety gates key off the
deterministic INCI parser + `knowledge/safety_rules.json`, never LLM output.

## Output document (`recommendations.json`, schema `recsys-1`)

Top level: `engine` (version + git commit + `eligibility_mode`), `inputs`
(analysis/profile sha256s + profile source), `data_versions`
(catalog/`drug_catalog`/signals/knowledge/verification sha256s), `framing`
(D-002), `triage` (level, referral reasons, professional-review observations,
see-doctor note), `status` (`ok` | `partial` | `unavailable` |
`referral_only`), `target_concerns`, `routines[]`, `unavailable_archetypes[]`
(archetype + reasons — an archetype is never silently absent),
`prescription_options[]`, `veto_log`, `warnings`. Each routine: archetype,
title, rationale, total price, `am[]` / `pm[]` steps, `safety_checks`, notes.
Each step: product identity, usage, `verification` (`verified` |
`category_derived`), `prescription`, `why{summary, score, signals[{name, value,
evidence}], uncertainty[]}` — every number reproducible from the named store.

## Verifying it

```bash
# the whole chain, one command: photo -> SA-RPN -> analysis.json -> recsys
python -m src.pipeline.e2e --image <photo.jpg> --out runs/e2e/<name> \
  --recsys --recsys-data-root recsys/data/derived --recsys-eligibility-mode hybrid

python tools/verify_e2e.py                          # hybrid, full catalog
python tools/verify_e2e.py --mode strict --data-root recsys/data
python tools/render_routine_html.py <recommendations.json>   # readable page
```

Without `--recsys-eligibility-mode` the integrated path uses recsys's default
(`strict`), which on the full catalog gives 3 of 5 archetypes and no
prescription options — correct, but not what you usually want to look at.

`verify_e2e.py` checks 18 stages against the *real* catalog and a real photo's
analysis — the wiring fixtures cannot see — and re-runs the CLI in three
processes to confirm identical bytes. It mirrors `pipeline.py`'s own catalog
resolution so it cannot accidentally verify a different catalog than the engine
used. Passes 18/18 across strict/hybrid × seed/derived.

**The silent failure mode it exists to catch**, because it produces a plausible
answer with no error: running without `--data-root recsys/data/derived` falls
back to the bundled 60-product seed catalog. The run succeeds and the routines
look reasonable; nothing says you ranked 60 products instead of 1,634.

There used to be a second, worse one: `--catalog` without `--data-root` left the
signal stores keyed to a *different* catalog, and a `catalog_sha256` mismatch was
skipped with only a warning — so the ranker scored blind on neutral 0.5 and the
sole defence was a human remembering to check `warnings` was empty. Both halves
are gone: a mismatched store now **raises**, and `--catalog` no longer exists —
the catalog is whatever `--data-root` resolves to.

**Reproducibility, precisely.** Verified on two different photos, four runs of
one and two of the other:

- The **detector** is byte-identical for the same photo — 4 runs, one digest
  (202 detections; the second photo likewise, at 240).
- The **engine** is byte-identical across processes once `--generated-at` pins
  the clock.
- A **full e2e replay** (photo → analysis → recsys) differs in exactly two
  fields: `generated_at`, and `inputs.analysis_sha256` — because `analysis.json`
  embeds its own timestamp, so hashing it faithfully yields a new digest each
  run. Products, prices, routines and prescription options are identical. That
  is a correct provenance chain, not drift; a bit-exact e2e replay needs the
  analysis timestamp pinned too.

Severity reaches the ranking and is directional: on a photo where
`acne_inflammatory` fell from severity 4 to 3, every product targeting it scored
lower while a cleanser targeting only the *other* concerns scored **higher**
(0.7143 → 0.7692). Same-concern photos can still yield the same winners — that
is the weighting working, not ignoring the input.

## Phase status (2026-07-15)

| Phase | State |
|---|---|
| 1 — Ingredient analysis | **done** — store live, signal loads, ≥95% coverage |
| 2 — Concern efficacy | **in flight** — p11 policy passed the ≥85% gate at 86% (independent Opus audit); full paid pass running: 472 requests, 77,801 review rows, 0 errors. The store is not built yet, so `concern_efficacy` is absent from `data_versions.signals` and its 0.25 archetype weight currently scores neutral. Ranking sharpens when it lands. |
| 3 — Verification overlay | **done** — 14 overlay rows, of which 13 match a catalog product: **0.8% of 1,634**. (The 14th is a DailyMed row that matches a drug-catalog row, not a cosmetics product.) Evidence hash-bound, stale-flip wired. |
| 4 — Full catalog | **done** — 1,634 products; golden-file eval harness live |
| 5 — Integration | **done** — `src/pipeline/e2e.py --recsys` writes both documents |
| + Hybrid eligibility | **done** (2026-07-15, beyond the original plan) — see "How it got here" |
| + Prescriptions | **done** (2026-07-15) — 34 Rx products catalogued and listed |

Known gaps, honestly:

- `azelaic_acid_10` is an unfillable therapy path. Cosmetics do not declare
  per-active strengths, so no product can prove 10% — verified against the brand
  page, not assumed.
- Strict on the full catalog yields **3 of 5 archetypes** (`budget` and
  `gentle_sensitive` go unavailable: `required_role_missing:treatment`), because
  only 13 of 1,634 products — **0.8%** — are evidence-verified. That thinness is
  exactly why hybrid exists.
- Stale-flip is wired but only half-exercised. `cmd_refresh` flips a product's
  approved assertions on any `evidence_issues()` hit; only the snapshot branch
  has fired on real data — the five `stale` rows in `approved-combined.json`
  were flipped a day after retrieval, not by age. The age window
  (`FRESHNESS_DAYS`, `recsys/verification.py:14`) has never fired: every
  approved `retrieved_at` is 0–1 days old against its 90/180-day window. Tests
  cover it; production has not.

Unverified products still degrade honestly: cadence/contraindications remain
unknown, SPF stays name-parsed with an uncertainty flag, and missing model
signals are neutral rather than silently penalized. Approved overlay facts and
concern-efficacy cells replace those fallbacks product by product.
