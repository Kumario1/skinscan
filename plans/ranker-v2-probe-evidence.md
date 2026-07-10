# Ranker v2 probe evidence — why the D-022 gate is unreachable with sklearn-only tools

Investigation record, 2026-07-10 (advisor session, after plan 013's real-data gate
failure). Seven systematic probes on the full ~1.1M-review dataset, all using plan
013's harness (worktree `worktree-agent-a3cea0c90a5b9751f`, commit `65fb1b2`),
reviewer-disjoint md5 split, identical eval code. Probe scripts were scratchpad
throwaways; this file is the durable record.

## The gate (D-022, unchanged)

One score must beat BOTH baselines on BOTH pooled metrics:
ROC-AUC > popularity (0.6721) and > bayesian (0.6661);
pairwise > popularity (0.5967) and > bayesian (**0.6086** — the binding constraint).

## Probe results (test split, n=173,815)

| # | variant | AUC | pairwise | gate |
|---|---------|-----|----------|------|
| — | required | > 0.6722 | > 0.6086 | — |
| 1 | v1 shipped feature set (plan 013) | 0.6590 | 0.5841 | FAIL both |
| 2 | + f_pop (train-only smoothed per-product recommend rate) | **0.6740** | 0.5976 | FAIL pw |
| 2b | + f_pop out-of-fold (leak check — same result) | 0.6735 | 0.5964 | FAIL pw |
| 3 | + f_pop + f_bayes (rating prior as feature) | **0.6741** | 0.5976 | FAIL pw — feature ≠ objective |
| 3b | + per product×skin_type encoding (f_pop_skin) | 0.6677 | 0.5937 | FAIL — noisy cells overfit |
| 4 | HGB **regressor** on rating (objective switch) | 0.6674 | 0.6073 | FAIL both, pw close |
| 4b | regressor on **reviewer-centered** rating, alone | 0.6197 | **0.6320** | FAIL auc — pw decisively won |
| 4c | f_bayes + centered delta | 0.6610 | **0.6175** | FAIL auc |
| 4d | 50/50 z-blend clf + (f_bayes+delta) | 0.6713 | **0.6090** | FAIL auc by 0.0009 (test luck, see 5) |
| 5 | **validation-tuned** blend weight, frozen, single test eval | 0.6729 | 0.6036 | **FAIL** — best val min-margin −0.0014 at any w ∈ [0.30, 0.75] |
| 6 | + review-TEXT lexicon features (efficacy/complaint phrase rates, product + cell level, train-only aggregates) | 0.6731 | 0.6038 | **FAIL** — val min-margin −0.0016; text adds nothing |
| 7 | StatsRanker candidates (no model): per-skin-type cell stats vs pooled | see below | see below | cells LOSE to pooled on both metrics |

Probe 6 detail: phrases fire often (neg_rate 0.234, pos_rate 0.246 of 926k texted
rows) but are redundant with the rating they arrive with — a "broke me out" review
carries a 1-star rating, so every text aggregate is another flavor of "average
goodness." The maintainer-chosen "review-text v2" direction was probed before
planning and found not to add orthogonal signal at lexicon level; heavier text
features (TF-IDF/embeddings) face the same aggregate wall with lower prior odds.

Probe 7 detail (test split, m=20 smoothing, fallback to product level):

| orderer | AUC | pairwise |
|---|---|---|
| popularity (pooled product recommend) | **0.6721** | 0.5967 |
| bayesian (pooled product rating) | 0.6661 | **0.6086** |
| cell recommend + fallback ("people like you") | 0.6696 | 0.5955 |
| cell rating + fallback ("people like you") | 0.6636 | 0.6062 |

**Per-skin-type conditioning strictly hurts** — the interaction signal is weaker
than the noise thinner cells add. The dataset does not support personalized
ordering (learned or statistical) beyond pooled product stats. The skin-type
cells remain valid and useful as *descriptive evidence* in the report ("41
reviewers with oily skin — 4.3 avg"), just not as an ordering signal.

Context: base_rate 0.847, n_train 518,437, 234,171 reviews dropped (no catalog
product). Baselines move ~0.005 between val and test splits — every near-miss
above is within split noise.

## The mechanism (why this is structural, not a bug)

1. **Within one reviewer, every profile feature is constant.** The pairwise metric
   ranks one person's items, so profile features contribute nothing to it directly;
   only product-side signal orders within a reviewer.
2. **The two baselines are different specialists.** Popularity (per-product
   recommend rate) is the strong AUC opponent; bayesian (per-product smoothed mean
   rating) is the strong pairwise opponent — star ratings carry finer ordering
   granularity than the binary label the classifier optimizes.
3. **Reviewer rating-scale noise hides the ordering signal.** Centering the
   regression target per reviewer (probe 4b) beats bayesian's pairwise decisively
   (0.632 vs 0.609) — but discards the absolute level that drives AUC.
4. **One score can't cover both corners** with this feature set: the
   classifier/centered-regressor blend traces a Pareto frontier that passes just
   *below* the gate corner (probe 5's validation sweep: min-margin ≤ −0.0014 at
   every weight).

Beating both specialists with one generalist score needs genuinely new signal,
not recombination of the existing columns.

## What would plausibly add new signal

- **Review-text mining** — probed (probe 6): lexicon-level text is redundant with
  ratings; no orthogonal signal. Heavier text (TF-IDF/embeddings per product
  corpus) hits the same aggregate wall — any per-product summary of train reviews
  is a flavor of "average goodness," which the baselines already are. Low odds.
- **A learning-to-rank objective** (e.g. LightGBM LambdaRank) — optimizes pairwise
  directly. New dependency; would still rank via product-side aggregates, so the
  same ceiling applies; the AUC corner remains unproven. Low odds.
- NOT more encodings/objective tweaks of the current columns — probes 2–6
  exhausted that space; NOT per-skin-type cells — probe 7 shows they subtract.

## The constructive finding: the bake-off has a shippable winner

PRD user story 12 asks for "products ranked by how well-reviewed they are" — and
the strongest orderer this data supports is the **pooled Bayesian-smoothed product
stats** (probe 7), already computed into `review_stats.json` by plan 013. A
~20-line `StatsRanker` (score = smoothed product rating/recommend blend from the
stats file, no sklearn at inference) plugs into the same duck-typed engine hook,
beats rules-only catalog order by construction, and is more explainable than any
model ("4.3★ Bayesian-adjusted over 263 reviews"). Shipping it requires a dated
D-022/D-005 amendment (the gate's "fails → rules-only" becomes "fails → the
statistical champion ships; any future learned model must beat IT"), which is the
log's own change mechanism — shipping the bake-off winner, not moving goalposts.

## Consequences

- Plan 013's code is correct and merged-or-mergeable as-is; the gate guard means
  the pipeline runs **rules-only** (D-005/D-019) — the designed, honest
  degradation.
- `review_stats.json` (1,591 products × skin-type cells) is produced regardless
  and feeds the report's evidence lines; the report should read it directly since
  `load_ranker` returns `None` without a model (see plans/README.md forward note).
- D-022 itself needs no amendment: the gate did exactly its job. If it is ever
  amended (e.g. pairwise-only), that is a dated decision-log change, not a plan.
