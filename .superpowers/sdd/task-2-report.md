# Task 2 Report — Production SA-RPN Native-Tile HTTP Inference

## Status

Implemented and verified.

## TDD evidence

### Red

Command:

```bash
python -m pytest tests/test_sarpn.py -q
```

Exact result before production implementation:

```text
ERROR tests/test_sarpn.py
ImportError: cannot import name 'LesionObservation' from 'src.pipeline.sarpn'
1 error in 0.25s
```

The new tests could not collect because the Task 2 interfaces did not yet exist.

### Green

Focused command:

```bash
python -m pytest tests/test_sarpn.py -q
```

Exact final result:

```text
36 passed in 9.19s
```

Model-free regression command:

```bash
python -m pytest tests -q --ignore=tests/test_face_landmarker_real.py --ignore=tests/test_predict_batch.py
```

Exact final result:

```text
180 passed in 12.66s
```

A broader model-free attempt that omitted only the real face-landmarker test produced `179 passed, 2 failed`; both failures were pre-existing environment failures in `tests/test_predict_batch.py` because TensorFlow is not installed. No files under `src/classification` were modified, per constraint.

## Files changed

- `src/pipeline/sarpn.py`
- `tests/test_sarpn.py`
- `.superpowers/sdd/task-2-report.md`

## Implementation

- Preserved `SarpnSettings` and recursive severity immutability.
- Added immutable `Tile` and `LesionObservation` values.
- Added EXIF-transposed RGB loading.
- Retained the accepted evenly spaced tile-origin algorithm.
- Added native-resolution tiling with right/bottom edge coverage.
- Added JPEG quality-92 base64 request encoding.
- Added bounded concurrent HTTP inference using `ThreadPoolExecutor(max_workers=request_batch_size)` and the exact `(connect, read)` timeout tuple.
- Added strict validation of the production `count`/`detections` contract and every detection field.
- Added client-side minimum-score filtering.
- Added local clipping, full-image coordinate restoration, and full-image clipping.
- Added deterministic tile-order output after concurrent completion.
- Added class-agnostic greedy suppression using intersection over smaller area and strict `>` threshold comparison.
- Added fail-closed behavior for every HTTP, timeout, JSON, and response-contract failure; no retry and no YOLO fallback.

## Self-review

- Compared the client directly with `sa-rpn/serve.py:69-84`; corrected an initial review finding that fixtures incorrectly invented `label_name`. Human-readable `label_name` is now derived from the server's `label`.
- Added strict validation for the endpoint's required `count` and consistency with the detection list.
- Confirmed no edits under `src/detection` or `src/classification`, and no edits to `src/recommendation/bridge.py` or `sa-rpn/serve.py`.
- Confirmed `git diff --check` passes.
- Confirmed tests use `ThreadingHTTPServer`/`BaseHTTPRequestHandler` for actual HTTP behavior and only use recording sessions for exact timeout and non-standard NaN/Infinity payload injection.

## Concerns

- Python's `ThreadPoolExecutor` cannot forcibly interrupt an already-running request. On the first tile failure, queued futures are cancelled, but already-running requests finish or reach their configured timeout before executor shutdown completes. This still preserves the required fail-closed result and never retries.
- The full model-free suite cannot be run in this environment without excluding `tests/test_predict_batch.py`, because that pre-existing test imports TensorFlow and TensorFlow is absent.

## Commit

Initial implementation commit: `31a11018b6becf9fdf8424f80b4feedba207be4c` (`feat: add native-tile SA-RPN HTTP inference`).

## Review-finding fixes

- Replaced the executor-wide mutable `requests.Session` with a `session_factory`; every tile request obtains and closes its own session, so no mutable session is shared across worker threads.
- Preserved exact timeout verification through an injected recording session factory.
- Made the real HTTP fixture allocate response indexes under a lock using a monotonic counter, independent of request completion order.
- Removed fixture-side `count` injection. Every successful fixture response now states `count` explicitly.
- Added a real-HTTP missing-`count` rejection test and retained invalid/mismatched count validation.
- Added EXIF orientation coverage proving `load_rgb` transposes before inference input is produced.
- Added direct coverage that concurrent tiles receive distinct sessions and that each session is closed.

Review-fix focused command:

```bash
python -m pytest tests/test_sarpn.py -q
```

Exact result:

```text
39 passed in 9.74s
```

Review-fix model-free regression command:

```bash
python -m pytest tests -q --ignore=tests/test_face_landmarker_real.py --ignore=tests/test_predict_batch.py
```

Exact result:

```text
183 passed in 12.88s
```

Review-fix diff validation:

```bash
git diff --check
```

Exact result: success with no output.

Review-fix commit: recorded in git history immediately after the initial implementation commit.
