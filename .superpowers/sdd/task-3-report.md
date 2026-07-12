# Task 3 Report

## Status

Implemented backward-compatible ConcernReport V2 evidence and the direct SA-RPN label/region/severity/safety bridge.

## Files

- `src/recommendation/schema.py`
- `src/pipeline/sarpn.py`
- `tests/test_sarpn.py`
- `.superpowers/sdd/task-3-report.md`

Historical `src/recommendation/bridge.py` was not changed.

## TDD evidence

### Red

Command:

```bash
python -m pytest tests/test_sarpn.py tests/test_bridge.py tests/test_recommendation_engine.py -q
```

Result: collection failed as expected because `ConcernEvidence` did not yet exist:

```text
ImportError: cannot import name 'ConcernEvidence' from 'src.recommendation.schema'
1 error in 0.45s
```

### Green

Command:

```bash
python -m pytest tests/test_sarpn.py tests/test_bridge.py tests/test_ranker.py tests/test_concern_stats.py -q
```

Result:

```text
87 passed in 12.74s
```

Compatibility command:

```bash
python -m pytest tests/test_recommendation_engine.py -q
```

Result:

```text
18 passed in 0.01s
```

Diff hygiene:

```bash
git diff --check
```

Result: clean.

## Runtime observation

Executed the bridge with Papule, Pustule, Nevus, and an unsupported label through the public module interface. It produced one aggregated inflammatory concern across sorted left/right cheek regions, separate papule/pustule evidence, mean confidence 0.84, max confidence 0.96, a non-actionable nevus safety observation with professional review, and an unsupported-label safety observation. No safety item created a concern.

## Compatibility

- First five positional `Concern` fields remain unchanged.
- Legacy constructors populate `regions` from singular `region`.
- Singular `region` remains canonical compatibility data.
- V2 `concern_to_dict` emits `regions` and `evidence`, not singular `region`.
- Historical bridge tests retain per-region behavior.
- Tone bucket `unknown` is accepted.

## Self-review

- Confirmed exact label normalization table lives only in `src/pipeline/sarpn.py`.
- Confirmed normalization casefolds, trims, and collapses spaces/hyphens/underscores.
- Confirmed provisional severity thresholds, regional escalation, nodule override, hypertrophic-scar minimum, and low-confidence cap.
- Confirmed nevus, other, and unsupported labels remain non-actionable.
- Confirmed `src/recommendation/bridge.py` is untouched.

## Concerns

None known. Runtime surface is currently the public Python module; the full e2e pipeline cutover is a later task.

## Review fixes

Addressed both Task 3 review findings with test-first fixes.

### Fix red

```bash
python -m pytest tests/test_sarpn.py -q
```

```text
2 failed, 60 passed in 9.85s
```

Expected failures proved that rebuilt observations replaced `label_name` with the normalized source field and that `concern_to_dict()` aliased `Concern.regions`.

### Fix green and compatibility

```bash
python -m pytest tests/test_sarpn.py tests/test_bridge.py tests/test_recommendation_engine.py -q
```

```text
90 passed in 9.80s
```

```bash
python -m pytest tests/test_sarpn.py tests/test_bridge.py tests/test_ranker.py tests/test_concern_stats.py -q
```

```text
90 passed in 12.12s
```

```bash
git diff --check
```

Result: clean.

### Review resolution

- The bridge normalizes `LesionObservation.label`, which is the Task 2 parser's response-label field, exactly once.
- Rebuilt observations preserve `LesionObservation.label_name` as the display/original label.
- `concern_to_dict()` now copies both `regions` and `evidence.labels`; payload mutation cannot alter the source concern.
- `acne_cystic` intentionally has no count-threshold entry. An explicit regression test confirms a nodule takes the severity-4 override before threshold lookup, preventing `KeyError`.

## Commit

Initial Task 3 commit: `1d86ca13aa40822281a9654db0f3020f964081c1`.
Review-fix commit exact hash is reported in the task completion response.
