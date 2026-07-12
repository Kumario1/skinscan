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

## Commit

Commit recorded after this report was prepared; final hash is reported in the completion response.
