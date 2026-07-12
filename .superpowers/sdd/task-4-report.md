# Task 4 Report — Evidence-Aware Deterministic Recommendations

## Status

Implemented. The deterministic recommendation engine now consumes SA-RPN V2 `ConcernEvidence` without activating a ranker, while retaining the optional ranker API and existing safety filters.

## TDD red commands and results

1. Low-confidence tracer:

```bash
python -m pytest tests/test_recommendation_engine.py::test_low_confidence_concern_is_visible_but_adds_no_strong_active -q
```

Result: **1 failed**. The prior engine incorrectly retained `salicylic_acid`, `adapalene`, and `azelaic_acid` for confidence 0.3.

2. Full Task 4 behavior set before implementation:

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py -q
```

Result: **8 failed, 39 passed**. Failures covered scarring support/guidance, changed pigmentation actives, broad-inflammation de-stacking, acne-before-scar ordering, deep-tone guidance, barrier support, and scarring ingredient metadata.

## Green commands and results

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py -q
```

Result: **47 passed in 0.04s**.

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py tests/test_ranker.py tests/test_concern_stats.py -q
```

Result: **65 passed in 2.16s**.

```bash
python tests/test_recommendation_engine.py
```

Result: **ok**.

```bash
git diff --check
```

Result: clean.

Runtime public-API observation used `recommend(...)` with broad inflammatory evidence, hypertrophic scarring, deep tone, and `ranker=None`. It returned deterministic targets `['azelaic_acid', 'niacinamide', 'ceramides']`, SPF, broad-inflammation/scarring/deep-tone flags, and no benzoyl peroxide. A low-confidence pigmentation probe returned no target actives, retained the verify flag, and still included supportive SPF.

## Full-suite note

```bash
python -m pytest -q
```

Result: collection error because `sa-rpn/test_client.py` interprets pytest's `-q` as an image path.

```bash
python -m pytest tests -q
```

Result: **218 passed, 2 failed, 1 deselected**. Both failures are pre-existing environment failures in `tests/test_predict_batch.py` because TensorFlow is not installed; they do not exercise the recommendation changes.

## Files changed

- `src/recommendation/engine.py`
- `src/recommendation/ingredient_kb.py`
- `tests/test_recommendation_engine.py`
- `tests/test_ingredient_kb.py`

## Behavior implemented

- Low-confidence concerns remain visible via multi-region verify flags but contribute no aggressive target active.
- Low-confidence pigmentation/scarring may still force supportive SPF.
- Cystic/severity-4 cases retain the existing soothe-only escalation short circuit.
- Scarring adds ceramides and SPF, with professional review for severity 3+ or hypertrophic evidence.
- Hyperpigmentation now targets azelaic acid and niacinamide, not vitamin C.
- Broad inflammatory acne removes benzoyl peroxide when azelaic acid is available and emits the required de-stacking flag.
- Active inflammatory acne is deterministically processed before scarring support.
- Any retained strong active adds ceramides once for barrier support.
- Deep tone adds sunscreen/irritation/PIH-prevention wording without changing efficacy targets; unknown tone adds nothing.
- `ranker=None` retains deterministic catalog and ingredient-match ordering; optional ranker APIs remain intact.
- Ingredient matching now includes ranking-only `acne_scarring` metadata.

## Self-review

- Confirmed pregnancy filtering still runs before conflict resolution and still removes retinoids.
- Confirmed retinoid PM pinning, incompatibility splitting, exfoliant cap, comedogenic partitioning, tier fallback, soothe-only filtering, and maintenance filtering remain covered and green.
- Confirmed cystic override is independent of concern order and preserves low-confidence verify flags.
- Confirmed supportive SPF is AM-only.
- Updated the standalone test runner to discover current `test_*` functions, avoiding stale renamed function references.
- No e2e ranker activation was added; Task 4 intentionally preserves the parameter only.

## Concerns

- The repository-wide pytest command collects `sa-rpn/test_client.py`, which behaves like a script at import time; use `python -m pytest tests ...` for the test suite until collection is isolated.
- The local environment lacks TensorFlow, leaving two unrelated classifier batch tests failing in the full `tests/` run.

## Review-finding fixes

A follow-up review identified two behavioral gaps and three test-strength gaps. They were fixed test-first.

### Review RED

```bash
python -m pytest tests/test_recommendation_engine.py -q
```

Result: **2 failed, 28 passed**.

- Broad inflammation removed benzoyl peroxide even when the catalog had no selectable azelaic-acid product.
- Deep-tone PIH guidance was absent when the relevant reported concern was below the active-confidence cutoff.

### Review GREEN

```bash
python -m pytest tests/test_recommendation_engine.py -q
```

Result: **30 passed in 0.03s**.

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py -q
```

Result: **50 passed in 0.03s**.

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py tests/test_ranker.py tests/test_concern_stats.py -q
```

Result: **68 passed in 2.22s**.

```bash
python tests/test_recommendation_engine.py
```

Result: **ok**.

```bash
python -m pytest tests -q
```

Result: **221 passed, 2 failed, 1 deselected**. The same two unrelated TensorFlow-missing classifier failures remain.

```bash
git diff --check
```

Result: clean.

### Review changes

- Broad-inflammation de-stacking now removes benzoyl peroxide only when an azelaic-acid product survives the catalog's tier-selection policy; a BP-only catalog retains BP and emits no de-stacking flag.
- Deep-tone PIH guidance now depends on reported inflammatory/scarring/pigmentation concerns, independent of confidence; low-confidence concerns still contribute no strong active.
- Acne-before-scar coverage now proves exact target ordering, scarring professional guidance, and SPF contribution.
- `ranker=None` coverage now proves descending ingredient-match ordering and stable catalog order for equal scores.
- Scarring coverage now proves the professional-review boundary between severity 2 and severity 3.

## Commits

- Initial Task 4 implementation: `a913786`
- Review-finding fix: `f95e24d`

## Remaining review-finding fixes

A second review required the broad-inflammation decision to use the exact final selection path and requested persistent safety/metadata regressions.

### Exact results

```bash
python -m pytest tests/test_recommendation_engine.py -q
```

Result: **33 passed in 0.02s**.

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py -q
```

Result: **53 passed in 0.03s**.

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py tests/test_ranker.py tests/test_concern_stats.py -q
```

Result: **71 passed in 3.00s**.

```bash
python tests/test_recommendation_engine.py
```

Result: **ok**.

```bash
python -m pytest tests -q
```

Result: **224 passed, 2 failed, 1 deselected**. The only failures remain the unrelated TensorFlow-missing classifier batch tests.

```bash
git diff --check
```

Result: clean.

### Changes

- Removed the duplicate approximate azelaic-selectability helper.
- Broad-inflammation de-stacking is now decided after the final target set, pregnancy filtering, and barrier support are assembled.
- The decision probes `_assign_slots` and `_build_routines`, the same slot/product/tier path used by final selection, and removes BP only if an azelaic product survives that path.
- Added a tier-shadowed regression: tier-2 azelaic in a serum category containing tier-1 niacinamide does not justify removing BP or adding the de-stacking flag.
- Added persistent low-confidence pigmentation and scarring tests proving verify flags, no concern active, and AM-only supportive SPF.
- Strengthened scarring ingredient metadata coverage to assert every required barrier/pigment-safe ingredient.

### Commit

- Remaining review-finding fix: `db23950`

## Final soothe-path review fix

The final review found that the cystic/severity-4 early return skipped required deep-tone PIH guidance.

### RED

```bash
python -m pytest tests/test_recommendation_engine.py::test_deep_tone_guidance_survives_cystic_soothe_only_short_circuit -q
```

Result: **1 failed**. The routine retained soothe-only targets and dermatologist escalation but omitted the deep-tone guidance flag.

### GREEN and compatibility evidence

```bash
python -m pytest tests/test_recommendation_engine.py -q
```

Result: **34 passed in 0.02s**.

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py -q
```

Result: **54 passed in 0.03s**.

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py tests/test_ranker.py tests/test_concern_stats.py -q
```

Result: **72 passed in 2.48s**.

```bash
python tests/test_recommendation_engine.py
```

Result: **ok**.

```bash
python -m pytest tests -q
```

Result: **225 passed, 2 failed, 1 deselected**. The same unrelated TensorFlow-missing classifier failures remain.

```bash
git diff --check
```

Result: clean.

### Change

- Centralized deep-tone PIH guidance in one helper and applied it before the cystic/severity-4 soothe-only return as well as the ordinary deterministic path.
- Regression proves deep tone plus cystic and relevant inflammatory/scarring evidence preserves exact soothe-only targets, dermatologist escalation, and deep-tone guidance.

### Commit

Recorded after this report update; final hash is reported in the completion response.
