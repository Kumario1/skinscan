# Plan 004: Unify the concern vocabulary so the Stage 2→3 bridge can drop in later

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `git diff --stat 1ebd544..HEAD -- src/classification/classifier.py configs/default.yaml tests/`
> Changes from plans 001–003 are EXPECTED (001 touched run_acne04_pipeline.py,
> 002 touched configs + argparse defaults, 003 added tests). Any mismatch with
> the excerpts below in `classifier.py` is a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plans/002-make-config-real.md, plans/003-test-the-rules-brain.md
- **Category**: tech-debt
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

The repo has THREE concern vocabularies that don't match:
`src/classification/classifier.py` speaks `comedonal/cystic/inflammatory/
not_acne/post_acne_mark`; `src/recommendation/schema.py` (the locked D-008
contract) speaks `acne_comedonal/acne_inflammatory/acne_cystic/
hyperpigmentation/dryness`. The classifier's `predict_concerns()` therefore
returns keys the recommender cannot consume. The full Stage 2→Stage 3 bridge
(region assignment, severity derivation, ConcernReport construction) is
deliberately deferred by the maintainer — this plan only makes the vocabulary
single-sourced and schema-correct so the bridge can be built later without a
migration. `not_acne` as a future classifier class is a separate design
(docs/STAGE2_NEGATIVES_DESIGN.md, plan 010); it does not belong in today's
concern vocabulary.

## Current state

- `src/classification/classifier.py:7-16`:

```python
CONCERN_CLASSES = ["comedonal", "cystic", "inflammatory", "not_acne", "post_acne_mark"]
RAW_ACNE_CLASSES = ["Blackheads", "Cyst", "Papules", "Pustules", "Whiteheads"]
RAW_TO_CONCERN = {
    "Blackheads": "comedonal",
    "Whiteheads": "comedonal",
    "Cyst": "cystic",
    "Papules": "inflammatory",
    "Pustules": "inflammatory",
}
CLASSES = CONCERN_CLASSES  # old name kept for callers
```

- `src/classification/classifier.py:88-93`:

```python
    def predict_concerns(self, crop):
        out = {"comedonal": 0.0, "cystic": 0.0, "inflammatory": 0.0}
        for raw, prob in self.predict(crop).items():
            if raw in RAW_TO_CONCERN:
                out[RAW_TO_CONCERN[raw]] += prob
        return out
```

- `src/recommendation/schema.py:16-20` — the authoritative vocabulary:

```python
CONCERNS = {
    "acne_comedonal", "acne_inflammatory", "acne_cystic",
    "hyperpigmentation", "dryness",
}
```

- Usage check (verified at planning time): `CONCERN_CLASSES`, `CLASSES`, and
  `predict_concerns` are referenced NOWHERE outside `classifier.py` itself.
  `run_acne04_pipeline.py` imports only `AcneTypeClassifier, crop_with_context`.
- `configs/default.yaml` has no bridge-related keys yet.

## Environment facts

- Fresh git worktree; `data/`, `models/`, `.venv/` absent. Do NOT `pip install`.
- Interpreter: `/Users/princekumar/Documents/skinscan/.venv/bin/python`
  (no pytest — tests are `__main__` runners, see `tests/test_pipeline_collage.py`).
- `classifier.py` imports TF lazily inside methods; the module itself imports
  fine without a model, and its `__main__` self-check runs without TF weights.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Classifier self-check | `/Users/princekumar/Documents/skinscan/.venv/bin/python src/classification/classifier.py` | prints `ok` |
| New contract test | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_concern_vocab.py` | prints `ok` |
| Regression | run every file in `tests/` with the interpreter | each prints `ok` |

## Scope

**In scope**:
- `src/classification/classifier.py`
- `configs/default.yaml` (add bridge-readiness keys)
- `tests/test_concern_vocab.py` (create)

**Out of scope**:
- Building the actual bridge (ConcernReport construction, region assignment,
  severity-from-count) — explicitly deferred by the maintainer.
- `src/recommendation/*` — the schema is the fixed side of the contract;
  nothing there changes.
- `run_acne04_pipeline.py` — it doesn't use the concern vocabulary yet.
- Retraining or renaming any model class (`RAW_ACNE_CLASSES` stays exactly
  as-is; it matches the shipped model's labels metadata).

## Git workflow

- Stay on the worktree's branch. Commit style:
  `refactor: align classifier concern vocab with the D-008 schema`
- Do NOT push.

## Steps

### Step 1: Make classifier.py speak the schema vocabulary

In `src/classification/classifier.py`:

1. Delete `CONCERN_CLASSES` and the `CLASSES = CONCERN_CLASSES` alias
   (verified unused outside this file).
2. Rewrite the mapping to schema IDs, importing nothing (keep this module
   TF-light and dependency-free; duplicate strings are fine — the contract
   test in Step 3 keeps them honest):

```python
RAW_TO_CONCERN = {
    "Blackheads": "acne_comedonal",
    "Whiteheads": "acne_comedonal",
    "Cyst": "acne_cystic",
    "Papules": "acne_inflammatory",
    "Pustules": "acne_inflammatory",
}
```

3. Extract the aggregation into a pure module-level function so it is testable
   without a TF model, and make `predict_concerns` a thin wrapper:

```python
def concern_probs(raw_probs):
    """Aggregate raw class probabilities into D-008 schema concern IDs."""
    out = {c: 0.0 for c in sorted(set(RAW_TO_CONCERN.values()))}
    for raw, prob in raw_probs.items():
        if raw in RAW_TO_CONCERN:
            out[RAW_TO_CONCERN[raw]] += prob
    return out
```

and

```python
    def predict_concerns(self, crop):
        return concern_probs(self.predict(crop))
```

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python src/classification/classifier.py` → `ok`

### Step 2: Add the bridge-readiness config keys

Append to `configs/default.yaml` (values are the planned defaults for the
future bridge; nothing reads them yet — they lock the shape):

```yaml
concern_report:                     # consumed by the future Stage 2->3 bridge
  severity_count_thresholds: [1, 5, 10, 20]   # lesion count -> severity 1..4 (Q-B, ACNE04-aligned)
```

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python -c "from src.config import load_config; print(load_config()['concern_report']['severity_count_thresholds'])"` → `[1, 5, 10, 20]`

### Step 3: The contract test

Create `tests/test_concern_vocab.py` (convention: `tests/test_pipeline_collage.py`):

1. `test_mapping_targets_are_schema_concerns` —
   `set(RAW_TO_CONCERN.values()) <= CONCERNS` (import `CONCERNS` from
   `src.recommendation.schema`). This is the actual bridge-readiness
   guarantee: classifier output keys are valid recommender input keys.
2. `test_mapping_covers_all_model_classes` —
   `set(RAW_TO_CONCERN) == set(RAW_ACNE_CLASSES)`.
3. `test_concern_probs_aggregates_and_preserves_mass` — feed
   `{"Blackheads": 0.2, "Whiteheads": 0.1, "Cyst": 0.3, "Papules": 0.25, "Pustules": 0.15}`
   → `{"acne_comedonal": 0.3, "acne_cystic": 0.3, "acne_inflammatory": 0.4}`
   (compare with `abs(x - y) < 1e-9` per key) and the values sum to 1.0.
4. `test_concern_probs_ignores_unknown_keys` — an input with an extra
   `"not_acne": 0.5` key doesn't crash and doesn't appear in the output.
5. `test_stub_classifier_round_trip` —
   `concern_probs(StubClassifier().predict(None))` returns exactly the three
   acne concern keys.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_concern_vocab.py` → `ok`

## Test plan

The 5 tests in Step 3. Plus regression: every existing file in `tests/`
prints `ok`, and the classifier self-check prints `ok`.

## Done criteria

- [ ] `grep -n "CONCERN_CLASSES\|not_acne\|post_acne_mark" src/classification/classifier.py` → no matches
- [ ] `grep -rn "CLASSES" src/classification/classifier.py` → only `RAW_ACNE_CLASSES` matches
- [ ] `grep -n "acne_comedonal" src/classification/classifier.py` → 1+ match
- [ ] `tests/test_concern_vocab.py` prints `ok`
- [ ] All other files in `tests/` print `ok`
- [ ] `src/classification/classifier.py` self-check prints `ok`
- [ ] `git status --porcelain` clean outside the in-scope list

## STOP conditions

- `grep -rn "CONCERN_CLASSES\|predict_concerns\|from .classifier import CLASSES\|classifier.CLASSES" src/ tests/` reveals a caller outside
  `classifier.py` (the planning-time check said there is none; if one
  appeared, the deletion is not safe as written).
- The classifier `__main__` self-check fails after the edit.
- `src/config.py` does not exist (means plan 002 wasn't merged into your
  worktree — report, don't recreate it).

## Maintenance notes

- The future bridge should import `RAW_TO_CONCERN` / `concern_probs` from
  `classifier.py` and `CONCERNS/REGIONS` from `schema.py` — vocabulary is now
  single-sourced on the schema side; the contract test fails if they drift.
- `concern_report.severity_count_thresholds` in the YAML is reserved for the
  bridge; a reviewer should reject code that hardcodes severity thresholds
  elsewhere.
- Deferred (by maintainer decision): the bridge itself — ConcernReport
  construction, region assignment, severity derivation.
