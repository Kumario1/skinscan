# SA-RPN V2 Production Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make `python -m src.pipeline.e2e` an SA-RPN HTTP-only production pipeline that emits V2 analysis and diagnostic artifacts, while retaining region/tone analysis and optional deterministic recommendations.

**Architecture:** Add one focused production module for SA-RPN HTTP transport, tile geometry, validation, class-agnostic deduplication, label normalization, concern bridging, severity, and diagnostic rendering. Rewrite the thin e2e module as orchestration and artifact publication only. Extend the existing recommendation schema compatibly so legacy YOLO/EfficientNet tools and tests remain intact, but V2 reports can aggregate evidence across regions.

**Tech Stack:** Python 3, `requests`, Pillow, NumPy, MediaPipe region mapper, existing ITA tone estimator, dataclasses, pytest, standard-library fixture HTTP server.

## Global Constraints

- Do not modify any source under `/Users/princekumar/Documents/skinscan/src/detection/` or `/Users/princekumar/Documents/skinscan/src/classification/`.
- Do not modify `/Users/princekumar/Documents/skinscan/src/recommendation/bridge.py`; it remains the historical EfficientNet bridge.
- Do not modify `/Users/princekumar/Documents/skinscan/sa-rpn/serve.py`; production must consume its existing `{"image": "<base64>"}` / `{"count": N, "detections": [...]}` contract.
- `src.pipeline.e2e` must not import or initialize `ultralytics`, TensorFlow, `AcneTypeClassifier`, `run_acne04_pipeline`, or the historical recommendation bridge.
- There is no YOLO/EfficientNet runtime fallback.
- Every tile must succeed. One timeout, HTTP error, invalid response, or invalid box fails identification.
- Cross-tile suppression remains class-agnostic intersection-over-smaller-area suppression, with higher-confidence detections retained first.
- Region mapping continues through `locate_regions()` and retains grid fallback metadata.
- Tone continues through `estimate_tone()`; `"unknown"` must be accepted by `UserProfile`.
- The default e2e must pass `ranker=None`; neither the learned ranker nor `ConcernStatsRanker` is activated.
- Recommendations are optional. Missing, unreadable, or invalid catalogs do not invalidate completed analysis.
- `analysis.json` is schema version `"2.0"`.
- Do not emit the old `predictions.json`, input collage, crop files, or classifier probability sheet from the default e2e.
- Final artifacts are published only after identification, region assignment, tone estimation, and concern construction complete.
- Normal tests require neither the SA-RPN/MMCV environment nor a live SA-RPN service.

---

## Current-State Findings That Shape the Implementation

1. `/Users/princekumar/Documents/skinscan/src/pipeline/e2e.py:18-22` statically imports the historical classifier bridge and concern-stats ranker. Lines 67-68 expose detector/classifier CLI arguments, and lines 76-91 instantiate YOLO and EfficientNet.
2. `/Users/princekumar/Documents/skinscan/src/pipeline/compare_sarpn.py:71-102` already contains the HTTP call and selected class-agnostic dedupe algorithm. Lines 166-189 contain the accepted native tile geometry. These are currently comparison-only and duplicated rather than production-owned.
3. `/Users/princekumar/Documents/skinscan/src/recommendation/schema.py:16-19` lacks `acne_scarring`; lines 27-39 model one `(concern, region)` entry and have no evidence fields. Lines 83-100 reject tone bucket `"unknown"`.
4. `/Users/princekumar/Documents/skinscan/src/recommendation/engine.py:128-133` flags low confidence but still adds all corresponding actives, directly conflicting with the V2 requirement.
5. `/Users/princekumar/Documents/skinscan/src/recommendation/engine.py:24` currently gives hyperpigmentation vitamin C, while the committed V2 policy calls for sunscreen, azelaic acid, and niacinamide.
6. `/Users/princekumar/Documents/skinscan/src/pipeline/regions.py:195-227` already returns all required method/fallback/reason metadata.
7. `/Users/princekumar/Documents/skinscan/src/pipeline/tone.py:143-175` already returns a valid `ToneEstimate("unknown", ...)`; only the downstream profile schema blocks it.
8. `/Users/princekumar/Documents/skinscan/sa-rpn/serve.py:69-84` confirms exact server labels come from checkpoint metadata and responses use `label`, `score`, and `bbox`.
9. Default `python -m pytest` currently collects `/sa-rpn/test_client.py`, which executes at import time and fails. `python -m pytest tests -q` currently reports 145 passing, two legacy TensorFlow-dependent failures, and one deselection in this environment. The cutover must not “fix” those failures by changing historical classifier source.
10. `/Users/princekumar/Documents/skinscan/docs/DECISIONS.md:364-379` conflicts with the committed spec by mapping scars to hyperpigmentation, retaining old severity thresholds, and promising local fallback. Documentation must explicitly supersede those parts of D-026.

## Planned File Map

### Create

- `/Users/princekumar/Documents/skinscan/src/pipeline/sarpn.py`
  - Production SA-RPN settings, typed observations, HTTP client, native tiling, coordinate restoration, response validation, class-agnostic dedupe, label normalization, concern bridge, severity, safety observations, and diagnostic rendering.
- `/Users/princekumar/Documents/skinscan/tests/test_sarpn.py`
  - Unit and fixture-HTTP tests for all production inference and bridge behavior.
- `/Users/princekumar/Documents/skinscan/tests/test_e2e.py`
  - CLI-level fixture e2e, artifact contract, optional catalog behavior, failed-identification publication behavior, and import guard.

### Modify

- `/Users/princekumar/Documents/skinscan/configs/default.yaml`
- `/Users/princekumar/Documents/skinscan/src/recommendation/schema.py`
- `/Users/princekumar/Documents/skinscan/src/recommendation/engine.py`
- `/Users/princekumar/Documents/skinscan/src/pipeline/e2e.py`
- `/Users/princekumar/Documents/skinscan/src/pipeline/compare_sarpn.py`
- `/Users/princekumar/Documents/skinscan/tests/test_config.py`
- `/Users/princekumar/Documents/skinscan/tests/test_recommendation_engine.py`
- `/Users/princekumar/Documents/skinscan/tests/test_compare_sarpn.py`
- `/Users/princekumar/Documents/skinscan/pytest.ini`
- `/Users/princekumar/Documents/skinscan/README.md`
- `/Users/princekumar/Documents/skinscan/docs/CONCERN_SCHEMA.md`
- `/Users/princekumar/Documents/skinscan/docs/RULES.md`
- `/Users/princekumar/Documents/skinscan/docs/DECISIONS.md`

### Explicitly Untouched

- `/Users/princekumar/Documents/skinscan/src/classification/classifier.py`
- `/Users/princekumar/Documents/skinscan/src/classification/run_acne04_pipeline.py`
- `/Users/princekumar/Documents/skinscan/src/detection/check_acne04_detector.py`
- `/Users/princekumar/Documents/skinscan/src/recommendation/bridge.py`
- `/Users/princekumar/Documents/skinscan/sa-rpn/serve.py`

---

## Proposed Production Interfaces

These names and signatures should be fixed before implementing dependent tasks.

```python
# src/pipeline/sarpn.py
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class SarpnSettings:
    endpoint_url: str
    tile_size: int
    tile_overlap: int
    connect_timeout_seconds: float
    read_timeout_seconds: float
    request_batch_size: int
    min_score: float
    dedupe_threshold: float
    severity: Mapping[str, object]

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "SarpnSettings": ...


@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class LesionObservation:
    normalized_label: str
    original_label: str
    confidence: float
    box: Box
    tile_index: int
    tile_box: Box
    region: str | None = None
    mapped_concern: str | None = None
    observation_status: str = "actionable"


@dataclass(frozen=True)
class SafetyObservation:
    code: str
    message: str
    labels: dict[str, int]
    count: int
    max_confidence: float
    professional_review: bool


class SarpnAnalysisError(RuntimeError):
    """Base failure for the SA-RPN identification stage."""


class SarpnTransportError(SarpnAnalysisError):
    """Connection, timeout, or non-success HTTP status."""


class SarpnResponseError(SarpnAnalysisError):
    """Malformed response, invalid label/score, or invalid bounding box."""


def load_rgb(path: Path) -> "np.ndarray": ...
def tile_origins(length: int, tile_size: int, stride: int) -> list[int]: ...
def make_tiles(image_shape: Sequence[int], tile_size: int, overlap: int) -> list[Tile]: ...
def infer_native_tiles(
    image_rgb: "np.ndarray",
    settings: SarpnSettings,
    *,
    session: "requests.Session | None" = None,
) -> list[LesionObservation]: ...
def dedupe_observations(
    observations: Sequence[LesionObservation],
    threshold: float,
) -> list[LesionObservation]: ...
def normalize_sarpn_label(label: str) -> str: ...
def build_sarpn_concern_report(
    image_id: str,
    observations: Sequence[LesionObservation],
    regions: Sequence[str],
    severity_config: Mapping[str, object],
    *,
    low_light_flag: bool = False,
) -> tuple["ConcernReport", list[LesionObservation], list[SafetyObservation]]: ...
def sanitize_endpoint(url: str) -> str: ...
def concern_to_dict(concern: "Concern") -> dict[str, object]: ...
def observation_to_dict(observation: LesionObservation) -> dict[str, object]: ...
def draw_detection_overlay(
    image_rgb: "np.ndarray",
    observations: Sequence[LesionObservation],
    output: Path,
) -> None: ...
def draw_region_overlay(
    image_rgb: "np.ndarray",
    observations: Sequence[LesionObservation],
    region_result: "RegionResult",
    output: Path,
) -> None: ...
def draw_lesion_sheet(
    image_rgb: "np.ndarray",
    observations: Sequence[LesionObservation],
    output: Path,
) -> None: ...
```

```python
# src/pipeline/e2e.py
@dataclass(frozen=True)
class PipelineResult:
    analysis: dict[str, object]
    routine: dict[str, object] | None
    output_dir: Path


def load_optional_catalog(path: Path) -> tuple[list[Product] | None, str | None]: ...
def routine_payload(
    report: ConcernReport,
    tone: ToneEstimate,
    region_mapping: Mapping[str, object],
    recommendation: Recommendation,
    top: int,
) -> dict[str, object]: ...
def run_pipeline(
    image_path: Path,
    output_dir: Path,
    *,
    settings: SarpnSettings,
    catalog_path: Path | None,
    face_landmarker_path: Path | None,
    skin_type: str,
    pregnant_or_nursing: bool,
    top: int,
    session: requests.Session | None = None,
) -> PipelineResult: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

```python
# src/recommendation/schema.py
@dataclass(frozen=True)
class ConcernEvidence:
    labels: dict[str, int] = field(default_factory=dict)
    max_confidence: float = 0.0
    affected_region_count: int = 0


@dataclass
class Concern:
    concern: str
    region: str
    severity: int
    confidence: float
    lesion_count: Optional[int] = None
    regions: list[str] = field(default_factory=list)
    evidence: ConcernEvidence = field(default_factory=ConcernEvidence)

    def __post_init__(self) -> None: ...
```

`Concern.region` remains as the first/canonical region strictly for source compatibility with existing positional constructors and historical code. V2 serializers emit `regions`, not `region`. `__post_init__` must set `regions = [region]` when legacy callers omit it and must validate every region.

---

## Task 1: Lock the SA-RPN Runtime Configuration Contract

**Files**

- Modify `/Users/princekumar/Documents/skinscan/configs/default.yaml:1-34`
- Modify `/Users/princekumar/Documents/skinscan/tests/test_config.py:9-19`
- Create `/Users/princekumar/Documents/skinscan/src/pipeline/sarpn.py`
- Create `/Users/princekumar/Documents/skinscan/tests/test_sarpn.py`

**Produces**

- `SarpnSettings`
- `SarpnSettings.from_config(config)`

- [ ] **Step 1: Add failing configuration tests**

Add assertions equivalent to:

```python
def test_load_config_has_sarpn_production_keys():
    cfg = load_config()
    sarpn = cfg["sa_rpn"]

    assert sarpn["endpoint_url"] == "http://localhost:8000/predict"
    assert sarpn["tile_size"] == 1024
    assert sarpn["tile_overlap"] == 128
    assert sarpn["connect_timeout_seconds"] == 5
    assert sarpn["read_timeout_seconds"] == 120
    assert sarpn["request_batch_size"] == 4
    assert sarpn["min_score"] == 0.3
    assert sarpn["dedupe_threshold"] == 0.5
    assert sarpn["severity"]["confidence_cutoff"] == 0.5
    assert set(sarpn["severity"]["count_thresholds"]) == {
        "acne_comedonal",
        "acne_inflammatory",
        "acne_scarring",
        "hyperpigmentation",
    }
```

```python
def test_sarpn_settings_reject_invalid_tile_geometry():
    cfg = load_config()
    cfg["sa_rpn"]["tile_overlap"] = cfg["sa_rpn"]["tile_size"]

    with pytest.raises(ValueError, match="tile_overlap"):
        SarpnSettings.from_config(cfg)
```

Also test non-positive batch size, invalid scores outside `[0, 1]`, and non-positive timeouts.

- [ ] **Step 2: Verify red**

Run:

```bash
python -m pytest tests/test_config.py tests/test_sarpn.py -q
```

Expected: failure because `sa_rpn` and `SarpnSettings` do not exist.

- [ ] **Step 3: Add exact configuration**

Preserve the existing `detection:` and `classification:` sections for historical commands. Add:

```yaml
sa_rpn:
  endpoint_url: http://localhost:8000/predict
  tile_size: 1024
  tile_overlap: 128
  connect_timeout_seconds: 5
  read_timeout_seconds: 120
  request_batch_size: 4
  min_score: 0.3
  dedupe_threshold: 0.5
  severity:
    confidence_cutoff: 0.5
    broad_region_count: 3
    nodule_severity: 4
    hypertrophic_scar_min_severity: 3
    count_thresholds:
      acne_comedonal: [1, 8, 20, 40]
      acne_inflammatory: [1, 6, 15, 30]
      acne_scarring: [1, 3, 8, 20]
      hyperpigmentation: [1, 4, 10, 25]
    professional_review:
      nevus: {min_count: 3, min_confidence: 0.8}
      other: {min_count: 5, min_confidence: 0.9}
```

These semantics are provisional but exact:

- Count threshold entries are inclusive starts for severities 1–4 using `bisect_right`.
- Two affected regions imply at least severity 2.
- Three or more affected regions imply at least severity 3.
- Any nodule yields severity 4.
- Any hypertrophic scar yields at least severity 3.
- If every retained detection for a non-nodule concern is below `confidence_cutoff`, cap severity at 1.
- Recommendation confidence remains separately enforced by the engine.

- [ ] **Step 4: Implement and validate `SarpnSettings.from_config`**

It must reject:

- `tile_size <= 0`
- `tile_overlap < 0`
- `tile_overlap >= tile_size`
- non-positive connection/read timeouts
- `request_batch_size <= 0`
- score or dedupe thresholds outside `[0, 1]`

- [ ] **Step 5: Verify green**

```bash
python -m pytest tests/test_config.py tests/test_sarpn.py -q
```

- [ ] **Step 6: Commit**

```bash
git add configs/default.yaml src/pipeline/sarpn.py tests/test_config.py tests/test_sarpn.py
git commit -m "feat: define SA-RPN production settings"
```

---

## Task 2: Implement HTTP Inference, Native Tiling, Validation, and Class-Agnostic Dedupe

**Files**

- Modify `/Users/princekumar/Documents/skinscan/src/pipeline/sarpn.py`
- Modify `/Users/princekumar/Documents/skinscan/tests/test_sarpn.py`

**Consumes**

- `SarpnSettings`
- Existing SA-RPN response contract from `/sa-rpn/serve.py:69-84`

**Produces**

- `Tile`
- `LesionObservation`
- `load_rgb`
- `tile_origins`
- `make_tiles`
- `infer_native_tiles`
- `dedupe_observations`
- `SarpnTransportError`
- `SarpnResponseError`

- [ ] **Step 1: Write tile-coverage and edge-restoration tests**

Cover:

```python
def test_make_tiles_covers_right_and_bottom_edges():
    tiles = make_tiles((1200, 2000, 3), tile_size=1024, overlap=128)

    assert len(tiles) == 6
    assert tiles[0] == Tile(index=0, x=0, y=0, width=1024, height=1024)
    assert tiles[-1].x + tiles[-1].width == 2000
    assert tiles[-1].y + tiles[-1].height == 1200
```

```python
def test_restored_boxes_are_clipped_to_full_image_bounds(fake_http_server):
    # Final tile begins away from origin and server returns a box extending
    # beyond that tile. The retained full-image box must end at width/height.
```

Retain the accepted evenly spaced origin algorithm currently at `compare_sarpn.py:166-173`.

- [ ] **Step 2: Write real HTTP fixture tests**

Use `http.server.ThreadingHTTPServer` and `BaseHTTPRequestHandler`, not a monkeypatch of `requests`.

The handler must:

- decode request JSON,
- verify `image` exists and is valid base64 JPEG,
- record each request,
- return deterministic `detections`.

Test exact timeout tuple by wrapping a recording `requests.Session`, while separate fixture tests prove actual HTTP behavior.

Required cases:

```python
def test_client_posts_base64_jpeg_and_validates_response(...)
def test_all_tiles_are_requested_and_results_restore_in_tile_order(...)
def test_one_http_500_fails_the_entire_analysis(...)
def test_one_timeout_fails_the_entire_analysis(...)
def test_missing_detections_key_is_rejected(...)
def test_non_list_detections_is_rejected(...)
def test_blank_label_is_rejected(...)
def test_boolean_or_out_of_range_score_is_rejected(...)
def test_reversed_or_zero_area_box_is_rejected(...)
def test_nan_or_infinite_box_coordinate_is_rejected(...)
def test_box_outside_tile_is_clipped_but_empty_after_clip_is_rejected(...)
```

Every exception message must include `tile <index>` and either the endpoint or response field.

- [ ] **Step 3: Write class-agnostic dedupe tests**

The important migration guard is different labels:

```python
def test_dedupe_is_class_agnostic_and_keeps_higher_confidence():
    observations = [
        LesionObservation(
            "papule", "Papule", 0.91, (10, 10, 40, 40), 0, (0, 0, 1024, 1024)
        ),
        LesionObservation(
            "pustule", "Pustule", 0.80, (12, 12, 39, 39), 1, (0, 0, 1024, 1024)
        ),
    ]

    kept = dedupe_observations(observations, threshold=0.5)

    assert kept == [observations[0]]
```

Also test equality semantics: preserve the existing `>` threshold comparison, not `>=`.

- [ ] **Step 4: Verify red**

```bash
python -m pytest tests/test_sarpn.py -q
```

- [ ] **Step 5: Implement bounded HTTP inference**

Implementation requirements:

- EXIF transpose and RGB conversion occur before any HTTP call.
- Encode each tile as JPEG quality 92.
- Use `requests.Session.post(..., json={"image": encoded}, timeout=(connect, read))`.
- Keep at most `request_batch_size` requests in flight via `ThreadPoolExecutor(max_workers=request_batch_size)`.
- Preserve deterministic final ordering by sorting tile results by `Tile.index`, regardless of completion order.
- Apply `min_score` client-side even if the server already filters.
- Translate tile-local boxes before dedupe.
- Clip local boxes to tile dimensions and restored boxes to full-image dimensions.
- Sort observations by descending confidence before class-agnostic suppression.
- Never retry automatically in V2; retries could duplicate load and hide partial service failures.

- [ ] **Step 6: Verify green**

```bash
python -m pytest tests/test_sarpn.py -q
```

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/sarpn.py tests/test_sarpn.py
git commit -m "feat: add native-tile SA-RPN HTTP inference"
```

---

## Task 3: Add ConcernReport V2 Evidence and the Direct SA-RPN Bridge

**Files**

- Modify `/Users/princekumar/Documents/skinscan/src/recommendation/schema.py:15-57,83-100`
- Modify `/Users/princekumar/Documents/skinscan/src/pipeline/sarpn.py`
- Modify `/Users/princekumar/Documents/skinscan/tests/test_sarpn.py`
- Run, but do not change unless broken, `/Users/princekumar/Documents/skinscan/tests/test_bridge.py`

**Compatibility strategy**

- Preserve the first five positional `Concern` fields exactly.
- Preserve `Concern.region` as canonical/legacy compatibility data.
- New V2 construction aggregates one `Concern` per concern across all regions.
- V2 JSON emits `regions` and `evidence`; it does not emit the compatibility-only singular `region`.
- Historical `build_concern_report()` remains unchanged and continues producing per-region entries.

- [ ] **Step 1: Add failing schema compatibility tests**

```python
def test_legacy_concern_constructor_populates_regions():
    concern = Concern("acne_inflammatory", "left_cheek", 2, 0.8, 4)

    assert concern.regions == ["left_cheek"]
    assert concern.evidence == ConcernEvidence()
```

```python
def test_user_profile_accepts_unknown_tone():
    profile = UserProfile(
        skin_type="combination",
        tone_bucket="unknown",
        tone_source="photo",
    )
    assert profile.tone_bucket == "unknown"
```

- [ ] **Step 2: Add exact label-normalization tests**

The normalization table must live only in `src/pipeline/sarpn.py`:

```python
SARPN_LABEL_TO_CONCERN = {
    "closed_comedo": "acne_comedonal",
    "open_comedo": "acne_comedonal",
    "papule": "acne_inflammatory",
    "pustule": "acne_inflammatory",
    "nodule": "acne_cystic",
    "atrophic_scar": "acne_scarring",
    "hypertrophic_scar": "acne_scarring",
    "melasma": "hyperpigmentation",
}

SARPN_NON_ACTIONABLE_LABELS = {"nevus", "other"}
```

Test all exact server spellings:

```python
@pytest.mark.parametrize(
    ("server_label", "normalized"),
    [
        ("Closed comedo", "closed_comedo"),
        ("open comedo", "open_comedo"),
        ("Papule", "papule"),
        ("Pustule", "pustule"),
        ("Nodule", "nodule"),
        ("Atrophic scar", "atrophic_scar"),
        ("Hypertrophic scar", "hypertrophic_scar"),
        ("Melasma", "melasma"),
        ("Nevus", "nevus"),
        ("other", "other"),
    ],
)
def test_normalize_exact_server_labels(server_label, normalized):
    assert normalize_sarpn_label(server_label) == normalized
```

Normalization must casefold, trim, and collapse spaces/hyphens/underscores to one underscore. Unknown nonblank labels remain normalized strings rather than raising.

- [ ] **Step 3: Add bridge and severity boundary tests**

Required cases:

- papules and pustules aggregate into one inflammatory concern,
- multiple regions become sorted unique `regions`,
- label counts remain separate,
- confidence is arithmetic mean of retained detection confidence,
- max confidence remains independent evidence,
- one/two/three-region escalation,
- count thresholds exactly below/at each boundary,
- one nodule yields severity 4,
- hypertrophic scar yields at least severity 3,
- low-confidence non-nodule burden is visible but capped at severity 1,
- no actionable observations yields `clear_skin=True`,
- nevus/other do not produce concerns,
- unknown labels remain in updated observations with `observation_status="unsupported"`,
- configured nevus/other count-or-confidence policy determines `professional_review`.

Expected concern construction:

```python
Concern(
    concern="acne_inflammatory",
    region="left_cheek",
    regions=["left_cheek", "right_cheek"],
    severity=2,
    confidence=0.84,
    lesion_count=9,
    evidence=ConcernEvidence(
        labels={"papule": 6, "pustule": 3},
        max_confidence=0.96,
        affected_region_count=2,
    ),
)
```

- [ ] **Step 4: Verify red**

```bash
python -m pytest tests/test_sarpn.py tests/test_bridge.py tests/test_recommendation_engine.py -q
```

- [ ] **Step 5: Extend the schema compatibly**

Exact schema changes:

```python
CONCERNS = {
    "acne_comedonal",
    "acne_inflammatory",
    "acne_cystic",
    "acne_scarring",
    "hyperpigmentation",
    "dryness",
}

TONE_BUCKETS = {"light", "medium", "deep", "unknown"}
```

`Concern.__post_init__` must:

1. Validate the concern and canonical `region`.
2. Set `regions = [region]` if empty.
3. Deduplicate `regions` while preserving order.
4. Validate every region.
5. Require canonical `region` to be present in `regions`.
6. Validate severity/confidence and `evidence.max_confidence`.
7. Require `evidence.affected_region_count == len(regions)` when evidence is populated.

- [ ] **Step 6: Implement the bridge**

The bridge takes parallel observations and regions, updates each observation with `region`, `mapped_concern`, and status, groups only actionable labels by concern, computes V2 concerns, and returns:

```python
(report, updated_observations, safety_observations)
```

Safety codes:

- `"nevus_observation"`
- `"other_observation"`
- `"unsupported_label"`

No safety observation may create a concern or target active.

- [ ] **Step 7: Verify green and historical compatibility**

```bash
python -m pytest \
  tests/test_sarpn.py \
  tests/test_bridge.py \
  tests/test_ranker.py \
  tests/test_concern_stats.py \
  -q
```

- [ ] **Step 8: Commit**

```bash
git add src/recommendation/schema.py src/pipeline/sarpn.py tests/test_sarpn.py
git commit -m "feat: bridge SA-RPN evidence into ConcernReport V2"
```

---

## Task 4: Make Recommendation Rules Evidence-Aware Without Activating a Ranker

**Files**

- Modify `/Users/princekumar/Documents/skinscan/src/recommendation/engine.py:19-27,100-147`
- Modify `/Users/princekumar/Documents/skinscan/tests/test_recommendation_engine.py`
- Modify `/Users/princekumar/Documents/skinscan/src/recommendation/ingredient_kb.py:38-64` only to add `acne_scarring` matching metadata
- Modify `/Users/princekumar/Documents/skinscan/tests/test_ingredient_kb.py` only for the new concern key

**Produces**

- Evidence-aware deterministic rules while preserving existing safety filters, AM/PM splitting, pregnancy handling, comedogenic ordering, tier handling, and optional ranker API.

- [ ] **Step 1: Replace the old low-confidence expectation with the V2 rule**

The existing test at `test_recommendation_engine.py:226-231` currently expects low-confidence concerns to add salicylic acid. Replace it with:

```python
def test_low_confidence_concern_is_visible_but_adds_no_strong_active():
    report = ConcernReport(
        "img",
        concerns=[Concern("acne_comedonal", "nose", 1, 0.3)],
    )

    rec = recommend(report, make_catalog())

    assert any("possible — verify" in flag for flag in rec.flags)
    assert "salicylic_acid" not in rec.target_actives
    assert "adapalene" not in rec.target_actives
    assert "azelaic_acid" not in rec.target_actives
```

- [ ] **Step 2: Add V2 behavior tests**

Add tests for:

1. `acne_scarring` gives `ceramides`, SPF, and professional guidance when severity ≥ 3 or evidence contains `hypertrophic_scar`.
2. Hyperpigmentation targets exactly `azelaic_acid` and `niacinamide`, with SPF; vitamin C is no longer introduced by this concern.
3. Broad inflammatory acne (`affected_region_count >= 3`) omits benzoyl peroxide when azelaic acid is available and adds an irritation-avoidance flag.
4. Any cystic concern overrides comedonal/inflammatory actives and retains the soothe-only path.
5. Active inflammatory acne is processed before scarring support.
6. Deep tone adds wording emphasizing sunscreen, irritation avoidance, and PIH prevention without changing efficacy claims.
7. Unknown tone adds no tone-specific flag and does not fail.
8. A strong active causes `ceramides` to be included for barrier support, matching the existing documented but currently unenforced rule.
9. `ranker=None` preserves deterministic catalog and ingredient-match ordering.
10. Existing pregnancy, PM retinoid, incompatibility, exfoliant cap, comedogenic, tier, soothe, and maintenance tests remain green.

- [ ] **Step 3: Verify red**

```bash
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py -q
```

- [ ] **Step 4: Update exact concern mappings**

```python
CONCERN_ACTIVES = {
    "acne_comedonal": ["salicylic_acid", "adapalene", "azelaic_acid"],
    "acne_inflammatory": ["benzoyl_peroxide", "azelaic_acid", "niacinamide"],
    "acne_cystic": ["centella"],
    "acne_scarring": ["ceramides"],
    "hyperpigmentation": ["azelaic_acid", "niacinamide"],
    "dryness": ["ceramides", "hyaluronic_acid", "glycerin"],
}
```

- [ ] **Step 5: Implement deterministic evidence policy**

Before `_assign_slots`:

- Always append a verify flag and `continue` for concerns below `conf_cutoff`.
- SPF is still allowed for low-confidence pigmentation/scarring because sunscreen is supportive, not an aggressive active.
- If a cystic concern exists, use the existing soothe-only short circuit.
- For broad inflammatory concern, remove `benzoyl_peroxide`, retain azelaic acid/niacinamide, and add:
  - `"broad inflammation: reduced strong-active stacking"`
- For scarring:
  - force SPF,
  - include ceramides,
  - add `"consider professional review for acne scarring"` when severity ≥ 3 or hypertrophic evidence is present.
- If any retained target belongs to `STRONG_ACTIVES`, add ceramides once.
- For deep tone plus inflammatory/scarring/pigmentation concern, add:
  - `"deeper tone: emphasize sunscreen and irritation avoidance to reduce post-inflammatory hyperpigmentation risk"`
- Do not pass a ranker from e2e. Keep the `ranker` parameter for standalone historical/evaluation callers.

Use a helper for multi-region flag text:

```python
def _concern_location(concern: Concern) -> str:
    return ",".join(concern.regions or [concern.region])
```

- [ ] **Step 6: Extend ingredient metadata**

Add an `acne_scarring` entry containing barrier-support and pigment-safe ingredient names, but keep it ranking-only:

```python
"acne_scarring": {
    "ceramide",
    "ceramides",
    "panthenol",
    "niacinamide",
    "azelaic acid",
    "centella",
}
```

- [ ] **Step 7: Verify green**

```bash
python -m pytest \
  tests/test_recommendation_engine.py \
  tests/test_ingredient_kb.py \
  tests/test_ranker.py \
  tests/test_concern_stats.py \
  -q
```

- [ ] **Step 8: Commit**

```bash
git add \
  src/recommendation/engine.py \
  src/recommendation/ingredient_kb.py \
  tests/test_recommendation_engine.py \
  tests/test_ingredient_kb.py
git commit -m "feat: make recommendations SA-RPN evidence aware"
```

---

## Task 5: Rewrite the Default E2E Around SA-RPN and Publish V2 Artifacts

**Files**

- Replace implementation of `/Users/princekumar/Documents/skinscan/src/pipeline/e2e.py:1-133`
- Modify `/Users/princekumar/Documents/skinscan/src/pipeline/sarpn.py`
- Create `/Users/princekumar/Documents/skinscan/tests/test_e2e.py`

**Consumes**

- `infer_native_tiles`
- `dedupe_observations`
- `locate_regions`
- `estimate_tone`
- `build_sarpn_concern_report`
- `recommend(..., ranker=None)`

**Produces**

- `analysis.json`
- optional `routine.json`
- `detections.jpg`
- `region_overlay.jpg`
- `lesion_sheet.jpg`

- [ ] **Step 1: Add import guard before rewriting e2e**

```python
def test_importing_default_e2e_loads_no_legacy_models():
    code = """
import sys
import src.pipeline.e2e
forbidden = {
    "tensorflow",
    "ultralytics",
    "src.classification.classifier",
    "src.classification.run_acne04_pipeline",
    "src.recommendation.bridge",
}
loaded = sorted(name for name in forbidden if name in sys.modules)
print(loaded)
raise SystemExit(bool(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 2: Add fixture HTTP e2e test**

Build a deterministic RGB JPEG under `tmp_path`; do not add a binary repository fixture. Use a fixture `fake_sarpn_server` backed by `ThreadingHTTPServer`.

Invoke the actual CLI function:

```python
exit_code = main([
    "--image", str(image_path),
    "--out", str(output_dir),
    "--api", fake_sarpn_server.url,
    "--catalog", str(catalog_path),
    "--face-landmarker", str(tmp_path / "missing.task"),
    "--tile-size", "1024",
    "--overlap", "128",
])
assert exit_code == 0
```

Assert the exact artifact set:

```python
assert {path.name for path in output_dir.iterdir()} == {
    "analysis.json",
    "routine.json",
    "detections.jpg",
    "region_overlay.jpg",
    "lesion_sheet.jpg",
}
```

Assert every image opens via Pillow and has non-zero dimensions.

Assert V2 JSON details:

```python
analysis = json.loads((output_dir / "analysis.json").read_text())
assert analysis["schema_version"] == "2.0"
assert analysis["pipeline"]["identifier"] == "sa-rpn-native-tiles"
assert analysis["pipeline"]["tile_size"] == 1024
assert analysis["pipeline"]["overlap"] == 128
assert analysis["region_mapping"]["method"] == "grid_fallback"
assert "missing" in analysis["region_mapping"]["reason"].lower()
assert analysis["recommendation_status"] == "complete"
assert analysis["detections"][0].keys() >= {
    "normalized_label",
    "original_label",
    "confidence",
    "box",
    "region",
    "mapped_concern",
    "source_tile",
}
```

Also assert:

- query strings and credentials in the fixture endpoint are not serialized,
- different-label overlap dedupes to one detection,
- unknown labels remain in detections with unsupported status,
- no classifier probabilities appear anywhere,
- no `predictions.json` or crop/collage artifacts appear.

- [ ] **Step 3: Add optional-catalog tests**

Missing catalog:

```python
assert main([... "--catalog", str(tmp_path / "missing.json")]) == 0
assert (output_dir / "analysis.json").exists()
assert not (output_dir / "routine.json").exists()
assert analysis["recommendation_status"] == "unavailable"
assert "missing" in analysis["recommendation_reason"].lower()
```

Unreadable/invalid JSON must behave the same way.

Recommendation exception handling should be tested by monkeypatching only `recommend`, after the real HTTP identification completes. Analysis and diagnostics must still publish, routine must not.

- [ ] **Step 4: Add failed-identification publication tests**

Seed the destination with a marker representing the last successful run, then cause tile 2 to return malformed JSON.

Assert:

- `main()` returns non-zero,
- no new `analysis.json` is published,
- no partial diagnostic files are published,
- the previous output directory remains intact,
- no legacy fallback request or import occurs.

- [ ] **Step 5: Verify red**

```bash
python -m pytest tests/test_e2e.py -q
```

- [ ] **Step 6: Rewrite e2e imports and CLI**

Top-level imports may include only lightweight modules plus:

```python
from ..config import load_config
from ..recommendation.engine import recommend
from ..recommendation.import_catalog import load_catalog
from ..recommendation.schema import Product, UserProfile
from .regions import locate_regions
from .sarpn import ...
from .tone import estimate_tone
```

Remove:

- detector/classifier CLI arguments,
- YOLO import,
- classifier imports,
- `ConcernStatsRanker`,
- historical bridge import,
- `predictions.json`.

New CLI options:

```text
--image PATH                  required
--out PATH                    default runs/e2e/<stem>
--api URL                     defaults from sa_rpn.endpoint_url
--catalog PATH                defaults paths.catalog_processed
--face-landmarker PATH        defaults paths.face_landmarker
--tile-size INT               defaults sa_rpn.tile_size
--overlap INT                 defaults sa_rpn.tile_overlap
--connect-timeout FLOAT       defaults config
--read-timeout FLOAT          defaults config
--request-batch-size INT      defaults config
--min-score FLOAT             defaults config
--dedupe-threshold FLOAT      defaults config
--skin-type VALUE             default combination
--pregnant                    boolean
--top INT                     default 5
```

- [ ] **Step 7: Build exact `analysis.json`**

Required top-level keys:

```python
{
    "schema_version": "2.0",
    "image_id": image_path.name,
    "pipeline": {
        "identifier": "sa-rpn-native-tiles",
        "endpoint": sanitize_endpoint(settings.endpoint_url),
        "tile_size": settings.tile_size,
        "overlap": settings.tile_overlap,
        "minimum_score": settings.min_score,
        "dedupe_threshold": settings.dedupe_threshold,
    },
    "detections": [...],
    "concerns": [...],
    "clear_skin": report.clear_skin,
    "skin_tone": asdict(tone),
    "region_mapping": region_result.metadata,
    "safety_observations": [asdict(item) for item in safety],
    "recommendation_status": "complete" | "unavailable",
    # recommendation_reason only when unavailable
}
```

`sanitize_endpoint()` must:

- preserve scheme, host, explicit port, and path,
- strip username/password,
- strip query and fragment,
- never serialize request headers or credentials.

- [ ] **Step 8: Render the three diagnostics**

`detections.jpg`:

- original image,
- every retained box,
- stable label color derived from a constant label-color map,
- normalized label and confidence,
- legend containing only present labels.

`region_overlay.jpg`:

- all polygons from `RegionResult.polygons`,
- lesion centroids,
- assigned region labels,
- title showing `metadata["method"]`,
- fallback grid naturally appears because it is already represented as polygons.

`lesion_sheet.jpg`:

- one padded crop per retained observation,
- normalized label,
- confidence,
- region,
- mapped concern or `safety`/`unsupported`,
- valid blank sheet with explanatory text when there are no detections.

- [ ] **Step 9: Stage and publish artifacts safely**

Use a sibling staging directory on the same filesystem:

1. Complete image loading and all HTTP inference before creating final artifacts.
2. Construct concern report in memory.
3. Render and serialize into staging.
4. Attempt recommendation; on failure update staged analysis to unavailable and omit routine.
5. If output exists, rename it to a sibling backup.
6. Atomically rename staging to output.
7. Remove backup only after publish succeeds.
8. If publish fails, restore backup.
9. Clean abandoned staging in `finally`.

This preserves an earlier successful output and prevents a failed identification from masquerading as a completed new analysis.

- [ ] **Step 10: Verify green**

```bash
python -m pytest \
  tests/test_e2e.py \
  tests/test_sarpn.py \
  tests/test_regions.py \
  tests/test_tone.py \
  tests/test_recommendation_engine.py \
  -q
```

- [ ] **Step 11: Manual fixture artifact inspection**

```bash
python -m pytest tests/test_e2e.py::test_fixture_e2e_writes_complete_v2_artifact_set -vv
```

Expected: PASS, with all five files opened and checked by the test.

- [ ] **Step 12: Commit**

```bash
git add src/pipeline/e2e.py src/pipeline/sarpn.py tests/test_e2e.py
git commit -m "feat: cut default e2e over to SA-RPN V2"
```

---

## Task 6: Reuse Production Geometry in the Comparison Harness and Fix Test Discovery

**Files**

- Modify `/Users/princekumar/Documents/skinscan/src/pipeline/compare_sarpn.py:27-102,164-213,265-305`
- Modify `/Users/princekumar/Documents/skinscan/tests/test_compare_sarpn.py:13-45,75-85`
- Modify `/Users/princekumar/Documents/skinscan/pytest.ini`

**Goal**

The historical zoom comparison remains available, but its native tile branch, HTTP validation, coordinate restoration, label mapping, concern bridge, and dedupe use production code.

- [ ] **Step 1: Change comparison tests to import production geometry**

Move tile/dedupe/bridge assertions to `tests/test_sarpn.py`. In `test_compare_sarpn.py`, retain only:

- zoom crop behavior,
- zoom coordinate mapping,
- cross-pipeline comparison metrics,
- a guard that the tile comparison path delegates to production.

Example:

```python
def test_tile_comparison_uses_production_inference(monkeypatch):
    called = {}

    def fake_infer(image, settings, session=None):
        called["image_shape"] = image.shape
        return []

    monkeypatch.setattr(compare_sarpn, "infer_native_tiles", fake_infer)
    # Invoke tile-only comparison helper.
    assert called["image_shape"] == image.shape
```

- [ ] **Step 2: Verify red**

```bash
python -m pytest tests/test_compare_sarpn.py tests/test_sarpn.py -q
```

- [ ] **Step 3: Remove duplicated production behavior from comparison**

Delete local production duplicates:

- `api_detect` for the tile path,
- `dedupe`,
- `tile_origins`,
- `tile_pipeline`,
- `report_from_detections`.

Import production equivalents from `.sarpn`.

The zoom branch may keep:

- `_square`
- `_overlaps`
- `zoom_crops`
- `zoom_pipeline`
- `yolo_boxes`

YOLO remains a lazy import inside `yolo_boxes`, so importing the comparison module remains lightweight.

Remove the import from `.e2e` at current line 42. The comparison harness should not make the user-facing default e2e module a utility dependency.

Do not activate `ConcernStatsRanker` in the tile production comparison. If historical ranker comparison is still useful, require an explicit future opt-in rather than loading it automatically.

- [ ] **Step 4: Restrict pytest discovery**

Update `pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -m "not real_models"
markers =
    real_models: slow checks that need local model/image artifacts
```

This prevents import-time execution of `sa-rpn/test_client.py` during normal tests without changing that historical manual script.

- [ ] **Step 5: Verify focused tests**

```bash
python -m pytest \
  tests/test_compare_sarpn.py \
  tests/test_sarpn.py \
  tests/test_e2e.py \
  -q
```

- [ ] **Step 6: Verify default collection no longer includes the manual client**

```bash
python -m pytest --collect-only -q
```

Expected: no `sa-rpn/test_client.py` entry.

- [ ] **Step 7: Run the default suite**

```bash
python -m pytest -q
```

Expected in a complete requirements environment: all model-free tests pass and `real_models` remains deselected.

Current environment warning: two pre-existing tests in `tests/test_predict_batch.py` fail because TensorFlow is not installed. Do not modify historical classifier source as part of this cutover. For immediate cutover verification in this environment, run:

```bash
python -m pytest -q --ignore=tests/test_predict_batch.py
```

The implementation should report that exclusion explicitly rather than claiming an unqualified full-suite pass.

- [ ] **Step 8: Commit**

```bash
git add \
  src/pipeline/compare_sarpn.py \
  tests/test_compare_sarpn.py \
  pytest.ini
git commit -m "refactor: share production SA-RPN geometry with comparison"
```

---

## Task 7: Update the V2 Contract and Production-Cutover Documentation

**Files**

- Modify `/Users/princekumar/Documents/skinscan/README.md:1-18,20-30,270-296,331-361,399-436,471-477`
- Modify `/Users/princekumar/Documents/skinscan/docs/CONCERN_SCHEMA.md:1-81`
- Modify `/Users/princekumar/Documents/skinscan/docs/RULES.md:14-23,82-130`
- Modify `/Users/princekumar/Documents/skinscan/docs/DECISIONS.md:329-384`

- [ ] **Step 1: Add a documentation test/checklist before editing**

Use repository searches as an executable red check:

```bash
grep -n "A two-stage acne analysis pipeline" README.md
grep -n "the shipped local pipeline remains the fallback" docs/DECISIONS.md
grep -n "severity keeps the bridge's existing lesion-count thresholds" docs/DECISIONS.md
grep -n "One entry.*per (concern, region)" docs/CONCERN_SCHEMA.md
```

All currently match text that must no longer describe the default V2 path.

- [ ] **Step 2: Update README default architecture**

The opening must state:

- SA-RPN native 1024px tiling is the sole default identifier.
- SA-RPN runs behind a required external HTTP service.
- YOLOv8m/EfficientNet remain historical and evaluation-only.
- No fallback occurs if the service is unavailable.
- Region and tone stages are retained.
- Recommendations are optional.
- Detector, concern mapping, and recommendation metrics remain separate.
- Do not imply a measured end-to-end accuracy.

Update the architecture diagram accordingly.

Retain the YOLO/EfficientNet metrics and history, but label them “historical pipeline” rather than default production.

- [ ] **Step 3: Update command and artifact documentation**

Default command:

```bash
.venv/bin/python -m src.pipeline.e2e \
  --image path/to/image.jpg \
  --api http://localhost:8000/predict \
  [--catalog data/processed/catalog.json] \
  [--skin-type dry] \
  [--pregnant]
```

Document:

```text
analysis.json
routine.json          optional
detections.jpg
region_overlay.jpg
lesion_sheet.jpg
```

Add real-service smoke command:

```bash
curl -fsS http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data "{\"image\":\"$(python -c 'import base64,sys; print(base64.b64encode(open(sys.argv[1],\"rb\").read()).decode())' path/to/image.jpg)\"}"
```

Then:

```bash
python -m src.pipeline.e2e \
  --image path/to/image.jpg \
  --api http://localhost:8000/predict \
  --out runs/e2e/sarpn-smoke
```

- [ ] **Step 4: Replace Concern Schema V1 with V2**

Document:

- `acne_scarring` as its own concern.
- Aggregated `regions`.
- Evidence labels, max confidence, affected region count.
- `region` remains an internal compatibility field but is not part of V2 JSON.
- Confidence is the mean retained detection score.
- Unknown labels and nevus/other remain observations, not concerns.
- The provisional severity table and exact YAML thresholds.

- [ ] **Step 5: Update rules documentation**

Change:

- Low-confidence concerns no longer introduce strong actives.
- Scarring policy.
- Hyperpigmentation V2 policy.
- Broad inflammatory stacking reduction.
- Deep-tone emphasis wording.
- Unknown-tone neutrality.
- Barrier support for strong actives.
- Default e2e uses `ranker=None`.
- Ranker interfaces remain available to standalone evaluation but are not production-activated.

- [ ] **Step 6: Add D-027 or an explicit D-026 supersession**

Do not silently rewrite the historical experiment. Append a decision titled:

```markdown
## D-027 — Production cutover: SA-RPN native tiles are the sole default identifier (2026-07-12)
```

It must explicitly supersede these D-026 statements:

- scars mapped to hyperpigmentation,
- existing classifier count thresholds reused,
- nevus/other simply dropped,
- local YOLO/EfficientNet fallback when the API is unreachable,
- unchanged old concern contract.

Record instead:

- scars map to `acne_scarring`,
- V2 evidence and provisional SA-RPN thresholds,
- nevus/other remain visible safety observations,
- service failure is explicit and terminal,
- YOLO/EfficientNet source remains historical/evaluation-only,
- default recommendation ranker is `None`,
- consumer-photo fixture artifact test is the automated cutover gate.

- [ ] **Step 7: Verify stale default claims are gone**

```bash
grep -n "the shipped local pipeline remains the fallback" docs/DECISIONS.md
grep -n "One entry.*per (concern, region)" docs/CONCERN_SCHEMA.md
grep -n "A two-stage acne analysis pipeline" README.md
```

Expected: no matches in current/default descriptions. Historical sections may still mention the old pipeline when clearly labeled historical.

- [ ] **Step 8: Run documentation-adjacent tests**

```bash
python -m pytest \
  tests/test_config.py \
  tests/test_sarpn.py \
  tests/test_recommendation_engine.py \
  tests/test_e2e.py \
  -q
```

- [ ] **Step 9: Commit**

```bash
git add \
  README.md \
  docs/CONCERN_SCHEMA.md \
  docs/RULES.md \
  docs/DECISIONS.md
git commit -m "docs: document SA-RPN V2 production cutover"
```

---

## Final Verification Sequence

Run in this order so failures are attributable:

```bash
python -m pytest tests/test_config.py -q
python -m pytest tests/test_sarpn.py -q
python -m pytest tests/test_recommendation_engine.py tests/test_ingredient_kb.py -q
python -m pytest tests/test_regions.py tests/test_tone.py -q
python -m pytest tests/test_e2e.py -q
python -m pytest tests/test_compare_sarpn.py -q
python -m pytest --collect-only -q
```

Current-environment focused suite:

```bash
python -m pytest tests -q --ignore=tests/test_predict_batch.py
```

Full suite after installing all declared requirements:

```bash
python -m pytest -q
```

Import guard independently:

```bash
python -c '
import sys
import src.pipeline.e2e
forbidden = [
    "tensorflow",
    "ultralytics",
    "src.classification.classifier",
    "src.classification.run_acne04_pipeline",
    "src.recommendation.bridge",
]
loaded = [name for name in forbidden if name in sys.modules]
print(loaded)
raise SystemExit(bool(loaded))
'
```

Expected output:

```text
[]
```

Source guard:

```bash
grep -nE \
  'ultralytics|tensorflow|AcneTypeClassifier|run_acne04_pipeline|recommendation\.bridge|ConcernStatsRanker' \
  src/pipeline/e2e.py
```

Expected: no matches.

Fixture artifact gate:

```bash
python -m pytest \
  tests/test_e2e.py::test_fixture_e2e_writes_complete_v2_artifact_set \
  -vv
```

Real-service smoke, when the SA-RPN endpoint is available:

```bash
python -m src.pipeline.e2e \
  --image path/to/consumer-photo.jpg \
  --api http://localhost:8000/predict \
  --out runs/e2e/sarpn-v2-smoke
```

Manually inspect:

- `runs/e2e/sarpn-v2-smoke/analysis.json`
- `runs/e2e/sarpn-v2-smoke/detections.jpg`
- `runs/e2e/sarpn-v2-smoke/region_overlay.jpg`
- `runs/e2e/sarpn-v2-smoke/lesion_sheet.jpg`
- `routine.json` only if a valid catalog is available

## Compatibility Hazards

1. **Concern aggregation change:** Historical reports are one entry per `(concern, region)`; SA-RPN V2 is one entry per concern with `regions`. Preserve `Concern.region` and positional construction so ranker and historical tests do not require a repository-wide rewrite.
2. **Tone `"unknown"`:** Adding it to `TONE_BUCKETS` changes profile validation but matches existing ranker behavior, which already serializes absent tone as `"unknown"`.
3. **Severity semantics:** Do not reuse `concern_report.severity_count_thresholds`. The old bridge remains on those thresholds; only the SA-RPN bridge consumes `sa_rpn.severity`.
4. **Server labels:** Checkpoint metadata may return spaces and title case. Normalize once and retain the original string in every observation.
5. **Boxes:** Validate booleans separately because Python treats `bool` as `int`. Reject NaN/infinity and zero-area boxes after clipping.
6. **Concurrency ordering:** Concurrent tile requests may finish out of order. Reassemble by tile index before dedupe and serialization.
7. **HTTP “batch size”:** The deployed endpoint accepts one image per request. Treat `request_batch_size` as the maximum number of requests in flight; do not invent an unsupported array payload.
8. **Deduplication drift:** Keep intersection-over-smaller-area, class-agnostic suppression, descending confidence, and strict `>` threshold.
9. **Recommendation failure:** Catch only the recommendation/catalog stage after report creation. Never catch an identification exception and label analysis complete.
10. **Output replacement:** Directly writing into an existing output directory can leave stale `routine.json`. Publish a fully staged directory and ensure unavailable recommendations cannot inherit an old routine.
11. **Comparison harness:** It can still import YOLO lazily for historical zoom, but production e2e and production SA-RPN modules must not import it.
12. **Default tests:** `pytest.ini` must restrict discovery to `/tests`; otherwise `sa-rpn/test_client.py` executes during collection.
13. **Legacy TensorFlow tests:** The present environment lacks TensorFlow even though it is in `requirements.txt`. Do not alter historical classifier code under this cutover merely to hide that environmental baseline.
14. **Documentation conflict:** D-026 contains statements now superseded by the committed V2 spec. Add a new logged decision rather than making the experiment’s historical record ambiguous.

## Recommended Implementation Order

1. Configuration and settings validation.
2. HTTP client, geometry, restoration, validation, and class-agnostic dedupe.
3. Backward-compatible ConcernReport V2 schema and direct SA-RPN bridge.
4. Evidence-aware recommendation changes.
5. E2E orchestration, diagnostics, staging, and fixture artifact gate.
6. Comparison harness reuse and pytest discovery.
7. Documentation cutover.
8. Focused suite, import guard, fixture artifact verification, then real-service smoke.

This ordering keeps each commit independently testable and avoids rewriting e2e against interfaces that are still moving.

### Critical Files for Implementation

- `/Users/princekumar/Documents/skinscan/src/pipeline/sarpn.py`
- `/Users/princekumar/Documents/skinscan/src/pipeline/e2e.py`
- `/Users/princekumar/Documents/skinscan/src/recommendation/schema.py`
- `/Users/princekumar/Documents/skinscan/src/recommendation/engine.py`
- `/Users/princekumar/Documents/skinscan/tests/test_e2e.py`