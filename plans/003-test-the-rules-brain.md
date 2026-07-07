# Plan 003: Put tests on the rules engine and the YOLO geometry (and make the "unit-tested" claim true)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `git diff --stat 1ebd544..HEAD -- src/recommendation/ src/detection/voc_to_yolo.py tests/ requirements.txt`
> Changes from plans 001–002 are EXPECTED (they touch other files). Any
> mismatch between the "Current state" excerpts below and the live code is a
> STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none (001/002 merged into your worktree; they don't touch these files)
- **Category**: tests
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

`docs/RULES.md` calls the rules layer "where all correctness lives," and
DECISIONS.md D-007 says it was to be grown test-first — yet
`src/recommendation/engine.py` has zero tests. Separately,
`src/detection/voc_to_yolo.py`'s module docstring claims its geometry "is
unit-tested" — no such test exists anywhere in the repo (the only test file is
`tests/test_pipeline_collage.py`, 3 helper tests). This plan adds the missing
tests and makes the false docstring true. It also protects plan 004, which
builds new code on top of the engine.

## Current state

- `src/recommendation/engine.py` — the recommender: `recommend()`,
  `_resolve_conflicts()`, `_build_routine()`, tables `CONCERN_ACTIVES`,
  `INCOMPATIBLE`. Key behaviors to pin (read the file; excerpts):
  - clear skin / no concerns → maintenance routine, always SPF, flag
    `"maintenance routine"` (engine.py:48-52)
  - `has_cystic` or `overall_severity >= 4` → flag `"see a dermatologist"`,
    soothing actives only (engine.py:54-58)
  - confidence below cutoff → flag `f"{concern}@{region}: possible — verify"`
    (engine.py:67-68)
  - `overall_severity == 3` → flag `"consider a professional"` (engine.py:73-74)
  - `_resolve_conflicts` drops the LATER member of an incompatible pair and
    flags `f"{a}: held back (conflicts with earlier active)"` (engine.py:81-91)
  - `_build_routine` includes `spf`-category products only when `always_spf`;
    other categories filter on actives intersection; comedogenic-flagged
    products sort last within a category (engine.py:94-109)
- `src/recommendation/schema.py` — `Concern`, `ConcernReport`, `Product`
  dataclasses with assert-based validation; closed vocabularies `CONCERNS`,
  `REGIONS`, `CATEGORIES`.
- `src/detection/voc_to_yolo.py` — geometry `voc_box_to_yolo(Box, w, h)`
  (clamps out-of-bounds boxes, raises `ValueError` on degenerate boxes and on
  bad image size), `yolo_line`, parsers. Docstring line 13:
  `actually matters for correctness — it's unit-tested.` ← currently false.
- `src/detection/visualize_labels.py` — `yolo_to_corners` is the inverse
  transform (useful for a round-trip test).
- `requirements.txt` — 8 loose deps; no pytest.
- Test convention: plain-python files with `if __name__ == "__main__":`
  runner printing `ok`; `sys.path.insert(0, ...parents[1])` at top. See
  `tests/test_pipeline_collage.py`.

## Environment facts

- Fresh git worktree; `data/`, `models/`, `.venv/` absent. Do NOT `pip install`.
- Interpreter: `/Users/princekumar/Documents/skinscan/.venv/bin/python`
  (pytest NOT installed — that's why tests must keep the `__main__` runner
  convention; you'll add pytest to requirements.txt for future environments
  but must not install or rely on it).
- The recommendation package is pure Python (no TF import) — tests run in
  milliseconds.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| New engine tests | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_recommendation_engine.py` | prints `ok` |
| New geometry tests | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_voc_to_yolo.py` | prints `ok` |
| Existing tests | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_pipeline_collage.py` | prints `ok` |

## Scope

**In scope**:
- `tests/test_recommendation_engine.py` (create)
- `tests/test_voc_to_yolo.py` (create)
- `src/detection/voc_to_yolo.py` (docstring line 13 ONLY)
- `requirements.txt` (append `pytest` on its own line; do not install it)

**Out of scope**:
- Any behavior change to `engine.py`, `schema.py`, or `voc_to_yolo.py` code.
  If a test reveals what looks like a bug, write the test to pin CURRENT
  behavior and record the suspicion in your report NOTES — do not fix it here.
- `run_acne04_pipeline.py`, `classifier.py`.

## Git workflow

- Stay on the worktree's branch. Commit style:
  `test: cover the recommendation engine and voc->yolo geometry`
- Do NOT push.

## Steps

### Step 1: Engine tests

Create `tests/test_recommendation_engine.py` (convention per
`tests/test_pipeline_collage.py`). Build a small in-code catalog helper:

```python
def make_catalog():
    return [
        Product("p1", "SA Cleanser", "b", "cleanser", actives=["salicylic_acid"]),
        Product("p2", "BP Gel", "b", "treatment", actives=["benzoyl_peroxide"]),
        Product("p3", "Niacinamide Serum", "b", "serum", actives=["niacinamide"]),
        Product("p4", "Ceramide Cream", "b", "moisturizer", actives=["ceramides"]),
        Product("p5", "Sunscreen", "b", "spf", actives=[]),
        Product("p6", "Coconut Balm", "b", "moisturizer",
                actives=["ceramides"], comedogenic_flags=["coconut_oil"]),
        Product("p7", "Vit C Serum", "b", "serum", actives=["vitamin_c"]),
    ]
```

Tests (each a function; all called from `__main__`):

1. `test_clear_skin_maintenance` — `ConcernReport("img", clear_skin=True)` →
   flags == `["maintenance routine"]`, target contains `ceramides` and
   `hyaluronic_acid`, routine includes the spf product.
2. `test_cystic_escalates` — one `Concern("acne_cystic", "chin_jaw", 2, 0.9)` →
   `"see a dermatologist"` in flags; `benzoyl_peroxide` NOT in target;
   `centella` in target.
3. `test_severity_4_escalates` — `Concern("acne_inflammatory", "forehead", 4, 0.9)`
   → `"see a dermatologist"` in flags.
4. `test_comedonal_gets_first_line_actives_no_spf` —
   `Concern("acne_comedonal", "nose", 2, 0.9)` → `salicylic_acid` in target;
   routine["spf"] is empty (no hyperpigmentation → no SPF).
5. `test_hyperpigmentation_forces_spf_and_conflict_resolution` — concerns
   `[acne_inflammatory(0.9), hyperpigmentation(0.9)]` → spf product present;
   `benzoyl_peroxide` in target; `vitamin_c` NOT in target and a
   `"vitamin_c: held back"` -prefixed flag present (INCOMPATIBLE pair, BP
   appears first).
6. `test_low_confidence_flags_verify` — `Concern("acne_comedonal", "nose", 1, 0.3)`
   → some flag containing `"possible — verify"`; actives still in target.
7. `test_severity_3_professional_note` — severity-3 inflammatory concern →
   `"consider a professional"` in flags.
8. `test_comedogenic_downranked_last` — comedonal concern with catalog
   containing `p4` and `p6` (both ceramide moisturizers — note: neither
   matches acne actives, so instead target them via a dryness concern
   `Concern("dryness", "left_cheek", 1, 0.9)`) → `routine["moisturizer"]`
   has `p4` before `p6`.
9. `test_ordered_steps_follows_category_order` — any recommendation's
   `ordered_steps()` category sequence is a subsequence of
   `["cleanser", "treatment", "serum", "moisturizer", "spf"]`.

Import style: `from src.recommendation.engine import recommend` /
`from src.recommendation.schema import Concern, ConcernReport, Product`.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_recommendation_engine.py` → `ok`

### Step 2: Geometry tests

Create `tests/test_voc_to_yolo.py`:

1. `test_center_conversion` — `Box(10, 20, 30, 60)` in a 100×200 image →
   `(0.2, 0.2, 0.2, 0.2)` exactly.
2. `test_clamps_overspill` — `Box(-10, -10, 50, 50)` in 100×100 → clamped:
   `(0.25, 0.25, 0.5, 0.5)`.
3. `test_degenerate_raises` — `Box(150, 10, 190, 20)` in a 100×100 image
   (fully outside → degenerate after clamp) raises `ValueError`; also
   `voc_box_to_yolo(Box(1,1,2,2), 0, 100)` raises `ValueError`.
4. `test_yolo_line_format` — `yolo_line(0, (0.5, 0.5, 0.1, 0.1))` ==
   `"0 0.500000 0.500000 0.100000 0.100000"`.
5. `test_round_trip_with_visualizer` — convert `Box(40, 30, 80, 90)` in
   200×150, feed through `visualize_labels.yolo_to_corners`, assert corners
   come back within 1px of the original.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_voc_to_yolo.py` → `ok`

### Step 3: Make the docstring true and record pytest

- In `src/detection/voc_to_yolo.py` line 13, change
  `it's unit-tested.` → `it's unit-tested (tests/test_voc_to_yolo.py).`
- Append `pytest` as the last line of `requirements.txt` with the comment
  `pytest  # test runner; files also run standalone via __main__`.

**Verify**: `grep -n "tests/test_voc_to_yolo" src/detection/voc_to_yolo.py` → 1 match; `grep -c pytest requirements.txt` → 1

## Test plan

This plan IS the test plan: 9 engine tests + 5 geometry tests, all runnable
standalone. Also keep pytest compatibility: name everything `test_*` so
`pytest tests/` works in environments that have it.

## Done criteria

- [ ] Both new test files print `ok` with the venv interpreter
- [ ] `tests/test_pipeline_collage.py` still prints `ok`
- [ ] `git diff --stat` touches only the 4 in-scope files
- [ ] `grep -n "unit-tested" src/detection/voc_to_yolo.py` shows the test path
- [ ] No behavioral diff in `src/recommendation/` or geometry code:
      `git diff src/recommendation/ src/detection/voc_to_yolo.py` shows only the one docstring line

## STOP conditions

- Any described engine behavior (flags text, conflict-resolution order,
  spf gating) doesn't match what the code actually does — the excerpts above
  were read from `1ebd544`; if reality differs, report the difference instead
  of changing engine code or bending the test to pass silently.
- A test requires touching engine/schema source to pass.

## Maintenance notes

- Plan 004 (Stage 2→3 bridge) builds a `ConcernReport` producer; these tests
  are its safety net — run them after any engine change.
- The flag strings are asserted with `in`-containment, not equality, so
  wording tweaks won't break tests unnecessarily; a reviewer should check the
  executor kept assertions meaningful (not `assert True`).
