# Plan 014: Ship the statistical champion — StatsRanker from review_stats + dated D-022 amendment

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not edit it.
>
> **Drift check (run first, AFTER the merge in "Git workflow")**:
> `git diff --stat 65fb1b2 -- src/recommendation/ranker.py tests/test_ranker.py`
> must be empty (you start from plan 013's code exactly), and
> `git diff --stat b197b6b -- src/recommendation/engine.py src/recommendation/schema.py src/pipeline/tone.py`
> must be empty. On a mismatch, STOP.

## Status

- **Priority**: P1 (completes the ranker story for milestone issue #1 / #8)
- **Effort**: S
- **Risk**: LOW — ~40 lines of pure-Python + tests + a documented decision amendment; no ML, no new deps.
- **Depends on**: plan 013 (branch `worktree-agent-a3cea0c90a5b9751f`, commit `65fb1b2`) — MUST be merged into your worktree first; see Git workflow.
- **Category**: direction
- **Planned at**: `main` @ `b197b6b` + plan-013 branch @ `65fb1b2`, 2026-07-10
- **Issue**: Kumario1/skinscan#8 (parent #1); evidence: `plans/ranker-v2-probe-evidence.md`

## Why this matters

Plan 013 built the learned ranker and its D-022 acceptance gate. On the real ~1.1M reviews the gate **failed**: the model (ROC-AUC 0.659 / pairwise 0.584) loses to both baselines (popularity 0.672/0.597, Bayesian rating 0.666/0.609). A seven-probe investigation (`plans/ranker-v2-probe-evidence.md`) established the failure is structural — no sklearn variant, text feature, or per-skin-type cell can pass — AND that the bake-off produced a shippable winner: the **pooled Bayesian-smoothed product rating**, already computed into `review_stats.json`. This plan ships that champion as a tiny `StatsRanker` through the engine's existing duck-typed hook, and records the outcome as a dated amendment to D-022/D-005 (the decision log's own change mechanism). Result: PRD story 12 ("products ranked by how well-reviewed they are… rather than alphabetical accident") is satisfied with the strongest orderer the data supports, and any future learned model must beat this champion to ship (the gate becomes a ratchet).

Empirical grounding (do not re-derive): pooled Bayesian rating orders test reviews at pairwise 0.6086 — the best of every candidate measured; per-skin-type cell variants measurably LOSE to pooled (cell rating 0.6062, cell recommend 0.5955), so **profile-conditioned ordering is explicitly rejected** — skin-type cells remain descriptive evidence only.

## Current state

(All excerpts from plan 013's code at `65fb1b2` — present after your merge.)

- `src/recommendation/ranker.py` — the module you extend. Relevant pieces:
  - `class Ranker` (learned; ~line 311): `score(product, profile)` returns P(is_recommended); `evidence(product_id, skin_type)` returns the stats cell with `n >= min_cell_size` else the `__all__` cell tagged `{"fallback": True, "cell": "all_reviewers"}`, `None` if the product is absent. Its evidence logic reads `self.cells = stats["cells"]` and `self.min_cell_size`.
  - `def load_ranker(config=None) -> "Ranker | None"` (~line 356): loads config, returns `None` when the model artifact is missing; loads stats if present.
  - `def build_review_stats(train_df, min_cell_size) -> dict` (~line 279): returns `{"min_cell_size": m, "base_rate": float, "cells": {product_id: {"__all__": {"n", "mean_rating", "pct_recommend"}, "<skin_type>": {...}}}}`. **Does NOT yet store the global mean rating** — you add it.
  - `train_pipeline(...)` (~line 423): always writes stats+eval; writes the model bundle only when `gate_passed`.
- `src/recommendation/engine.py` (DO NOT MODIFY) — the hook: sort key is `(len(p.comedogenic_flags), -ranker.score(p, profile))`; higher score = better; comedogenic partition always dominates.
- `tests/test_ranker.py` — 7 tests incl. `test_degrades_to_rules_only_when_model_absent`, which asserts `load_ranker` → `None` when **both** model and stats paths are absent (that stays true).
- `configs/default.yaml` — `ranker:` block already has every key you need: `review_stats_path`, `min_cell_size: 5`, `bayesian_prior_count: 20`. **Add no new config keys.**
- `docs/DECISIONS.md` — D-005 and D-022 as amended 2026-07-09; the log's header states its change rule: "once a decision is LOCKED, don't silently reverse it — if it needs to change, edit the entry and note the change."
- `CONTEXT.md` — glossary entry "**ranker** — The learned re-ranker that reorders rule-approved candidate products…".

## Commands you will need

Your worktree has no `.venv` (gitignored) — use the main repo's interpreter, run from your worktree root:

| Purpose | Command | Expected |
|---|---|---|
| Ranker tests | `/Users/princekumar/Documents/skinscan/.venv/bin/python -m pytest tests/test_ranker.py -q` | all pass |
| Full suite | `/Users/princekumar/Documents/skinscan/.venv/bin/python -m pytest -q` | all pass |
| Standalone | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_ranker.py` | prints `ok` |

There is no `data/` in the worktree; the real-artifact regeneration is the reviewer's step (they have the data). Fixture tests are your gate.

## Scope

**In scope** (only files you modify):
- `src/recommendation/ranker.py` — extend (StatsRanker, evidence extraction, `global_mean_rating` in stats, `load_ranker` three-way).
- `tests/test_ranker.py` — extend (5 new tests, one assertion update).
- `docs/DECISIONS.md` — dated amendments to D-022 and D-005 (text given verbatim in Step 4).
- `CONTEXT.md` — one glossary sentence (Step 4).
- `docs/CATALOG_SCHEMA.md` — one line in the "Review-stats & ranker artifacts" subsection (Step 4).
- `README.md` — one sentence after the ranker train command (Step 4).

**Out of scope** (do NOT touch):
- `src/recommendation/engine.py`, `src/recommendation/schema.py`, `src/pipeline/tone.py` — locked seams; StatsRanker plugs into the existing hook.
- `configs/default.yaml` — no new keys; reuse `bayesian_prior_count` and `min_cell_size`.
- The Streamlit app / report layer — later issues consume `load_ranker()` as-is.
- `plans/README.md` — reviewer maintains it.

## Git workflow

- Your worktree starts at `main` (`b197b6b`), which does NOT contain plan 013's code. **First action**: `git merge worktree-agent-a3cea0c90a5b9751f` (a local branch; worktrees share refs). Expect a clean merge (013 touched files main hasn't since `b197b6b`). Then run the drift check.
- Commit per logical unit; conventional commits, e.g. `feat: StatsRanker — ship the bake-off champion; D-022 dated amendment (#8)`.
- Do NOT push, PR, or merge to `main`.

## Steps

### Step 1: Add `global_mean_rating` to the stats artifact

In `build_review_stats`, add one key to the returned dict:
```python
"global_mean_rating": float(train_df["rating"].mean()),
```
(alongside `base_rate`). This is the smoothing prior StatsRanker needs; it is the same global mean the Bayesian baseline uses.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python -c "
import pandas as pd
from src.recommendation.ranker import build_review_stats
df = pd.DataFrame({'product_id':['P1','P1'],'skin_type':['oily','dry'],'rating':[4.0,5.0],'label':[1,0]})
s = build_review_stats(df, 5); print(round(s['global_mean_rating'],2))"` → `4.5`

### Step 2: Extract the evidence lookup and add `StatsRanker`

In `src/recommendation/ranker.py`:

**2a.** Extract the body of `Ranker.evidence` into a module-level function, and delegate:
```python
def evidence_cell(cells, min_cell_size, product_id, skin_type):
    """The report's per-product 'why' cell (shared by Ranker and StatsRanker):
    the product x skin_type cell when its n >= min_cell_size, else the '__all__'
    cell tagged fallback; None if the product is absent."""
    cell = cells.get(product_id)
    if cell is None:
        return None
    typed = cell.get(skin_type)
    if typed is not None and typed.get("n", 0) >= min_cell_size:
        return {**typed, "fallback": False, "cell": skin_type}
    all_cell = cell.get("__all__")
    if all_cell is None:
        return None
    return {**all_cell, "fallback": True, "cell": "all_reviewers"}
```
`Ranker.evidence` becomes `return evidence_cell(self.cells, self.min_cell_size, product_id, skin_type)`. Behavior identical — the existing evidence test must still pass unchanged.

**2b.** Add the champion:
```python
class StatsRanker:
    """The bake-off champion (D-022 amendment, 2026-07-10): orders candidates by
    the Bayesian-smoothed POOLED product rating from review_stats.json — the
    strongest orderer the review data supports (pairwise 0.609 on the D-022
    harness; see plans/ranker-v2-probe-evidence.md). Deliberately ignores the
    profile: per-skin-type cell ordering was measured and LOSES to pooled
    (0.606/0.596); skin-type cells stay evidence-only. No sklearn at inference.
    Same duck-typed hook as the learned Ranker; higher score = better."""

    def __init__(self, stats, m, min_cell_size):
        self.cells = stats.get("cells", {})
        self.min_cell_size = min_cell_size
        prior = float(stats.get("global_mean_rating", 0.0))
        self._prior = prior
        self._scores = {
            pid: (cell["__all__"]["n"] * cell["__all__"]["mean_rating"] + m * prior)
                 / (cell["__all__"]["n"] + m)
            for pid, cell in self.cells.items()
            if cell.get("__all__")
        }

    def score(self, product, profile) -> float:
        # profile intentionally unused (pooled beats per-profile; see class doc).
        return self._scores.get(product.product_id, self._prior)

    def evidence(self, product_id, skin_type):
        return evidence_cell(self.cells, self.min_cell_size, product_id, skin_type)
```

**2c.** Rework `load_ranker` to the three-way contract (docstring included):
```python
def load_ranker(config=None) -> "Ranker | StatsRanker | None":
    """Three-way loader (D-022 as amended 2026-07-10):
    - learned model artifact present  -> Ranker (a model exists only if it
      passed the ratcheted gate at training time);
    - model absent, review-stats present -> StatsRanker (the statistical
      champion: Bayesian-smoothed pooled product rating);
    - both absent -> None (rules-only catalog order, D-019)."""
```
Implementation: keep the existing model branch as-is; when the model path is missing but `review_stats_path` exists, load the stats JSON and return `StatsRanker(stats, config["ranker"]["bayesian_prior_count"], config["ranker"]["min_cell_size"])`; only when the stats file is ALSO missing return `None`. Note: `_config_at()` in the tests builds a minimal `ranker` config dict — it must now include `bayesian_prior_count` (add it there with value `20`); use `rcfg.get("bayesian_prior_count", 20)` defensively.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python -c "
from src.recommendation.ranker import StatsRanker, evidence_cell, load_ranker
s = StatsRanker({'global_mean_rating': 4.0, 'cells': {'PA': {'__all__': {'n': 100, 'mean_rating': 4.5, 'pct_recommend': 0.9}}}}, 20, 5)
class P: product_id='PA'
class Q: product_id='NOPE'
print(round(s.score(P(), None), 4), s.score(Q(), None))"` → `4.4167 4.0`

### Step 3: Tests

Extend `tests/test_ranker.py`:

1. `test_stats_ranker_orders_by_smoothed_rating` — hand-built stats dict, `m=20`, prior `4.0`: PA `{n:100, mean_rating:4.5}` → 4.4167; PB `{n:5, mean_rating:5.0}` → (25+80)/25 = 4.2. Assert `score(PA) > score(PB)` — **the smoothing test**: a 5.0-rated n=5 product must NOT outrank a well-attested 4.5.
2. `test_stats_ranker_unknown_product_gets_prior` — absent product scores exactly the prior; `profile=None` accepted.
3. `test_stats_ranker_through_engine` — train the fixture pipeline (existing `_train` helper) to produce a stats file; delete/point-away the model; `load_ranker` returns a `StatsRanker`; feed it to `recommend()` with the PA/PB/PC catalog from the existing reorder test and assert PC (comedogenic) still sorts last and PA/PB order follows their `__all__` smoothed ratings (fixture: PA all-cell mean_rating = (60·5 + 60·2)/120… **careful**: stats are train-rows-only — compute the expected order inside the test from the loaded stats file itself, not from hardcoded numbers: `assert ids[:2] == sorted(['PA','PB'], key=lambda p: -ranker._scores[p])`).
4. `test_load_ranker_three_way` — (a) model+stats → `Ranker`; (b) stats only → `StatsRanker`; (c) neither → `None`.
5. `test_stats_ranker_evidence_matches_ranker_evidence` — both classes return the identical dict for the same stats file (`evidence_cell` extraction is behavior-preserving), including the `normal`-skin-type fallback case.

Also UPDATE `test_degrades_to_rules_only_when_model_absent`: it currently passes a stats path that doesn't exist, so it still returns `None` — keep it, but rename/extend its docstring to say "both artifacts absent → rules-only" so the three-way semantics are pinned explicitly.

**Verify**: `…/python -m pytest tests/test_ranker.py -q` → 12 passed; `…/python tests/test_ranker.py` → `ok`; `…/python -m pytest -q` → all pass.

### Step 4: The dated amendments (exact text)

**4a. `docs/DECISIONS.md`** — append to the END of the D-022 entry (keep everything existing):

```markdown
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
```

**4b. `docs/DECISIONS.md`** — append one line to the END of D-005:

```markdown
**Note 2026-07-10:** the learned slot is currently empty — the D-022 gate
rejected the trained model, so the reorderer shipping in the hook is the
statistical champion (`StatsRanker`, see D-022 amendment). The hybrid contract
(rules gate, hook only reorders, comedogenic/safety untouchable) is unchanged.
```

**4c. `CONTEXT.md`** — in the **ranker** glossary entry, replace the first sentence
"The learned re-ranker that reorders rule-approved candidate products within a category by predicted fit for a profile (D-005 / D-022)."
with:
"The re-ranker that reorders rule-approved candidate products within a category (D-005 / D-022) — today a statistical champion (Bayesian-smoothed pooled product rating; `StatsRanker`), with the learned slot open to any model that beats it under the D-022 ratchet."
Keep the second sentence ("It only reorders…") unchanged.

**4d. `docs/CATALOG_SCHEMA.md`** — in the "Review-stats & ranker artifacts" subsection, add one line documenting the new `global_mean_rating` top-level key (the StatsRanker smoothing prior).

**4e. `README.md`** — directly after the ranker train command block, add:
```markdown
The trained model ships only if it passes the D-022 gate; otherwise the app
orders products by the Bayesian-smoothed review stats (`StatsRanker`) — see
`docs/DECISIONS.md` D-022 (2026-07-10 amendment).
```

**Verify**: `grep -c "2026-07-10" docs/DECISIONS.md` → ≥ 2; `grep -n "StatsRanker" CONTEXT.md README.md docs/DECISIONS.md docs/CATALOG_SCHEMA.md` → ≥ 1 match in each.

## Test plan

Step 3 is the test plan (5 new + 1 clarified; 12 total in `tests/test_ranker.py`). Pattern: the existing tests in the same file. Verification: full suite green.

## Done criteria

- [ ] `…/python -m pytest tests/test_ranker.py -q` → 12 passed
- [ ] `…/python -m pytest -q` → all pass, no regressions
- [ ] `load_ranker` three-way behavior covered by a test (model+stats / stats-only / neither)
- [ ] `git diff --stat b197b6b -- src/recommendation/engine.py src/recommendation/schema.py src/pipeline/tone.py configs/default.yaml` → **empty** (config untouched too)
- [ ] All four docs edits present (Step 4 verifies)
- [ ] Only in-scope files modified (`git status` clean of surprises)

## STOP conditions

- The merge of `worktree-agent-a3cea0c90a5b9751f` conflicts, or the post-merge drift check fails.
- The existing 013 tests break for any reason other than the documented `_config_at` addition — the evidence extraction (2a) must be behavior-preserving; if it isn't, something is wrong.
- Implementing the three-way `load_ranker` seems to require touching `engine.py` or the config schema — the boundary is wrong; stop.
- You find yourself wanting to make `StatsRanker.score` profile-dependent — that contradicts the measured evidence (probe 7); stop and report rather than "improve" it.

## Maintenance notes

- **The reviewer regenerates real artifacts** after approval: re-run the training CLI so `review_stats.json` gains `global_mean_rating` (the trainer needs no change beyond Step 1 — it writes stats regardless of gate outcome). Until regenerated, an old stats file lacks the key; `StatsRanker` then smooths toward `0.0` — the defensive `stats.get(..., 0.0)` keeps it running but mis-ordered, which is why regeneration is part of landing this.
- **The report issue** now gets its ranker score and evidence from ONE object (`load_ranker()` → StatsRanker in the normal case): `score()` is the same smoothed rating the "why" line can print. Verify the rendered report never presents a fallback (`all_reviewers`) cell as skin-type-specific.
- **Future model attempts**: the ratchet is automatic — the trainer's gate already compares against the champion's score (the Bayesian baseline row). Nothing to re-wire.
