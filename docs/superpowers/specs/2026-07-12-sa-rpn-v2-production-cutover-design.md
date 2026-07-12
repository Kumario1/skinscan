# SA-RPN V2 Production Cutover Design

**Date:** 2026-07-12  
**Status:** Approved for implementation planning

## Goal

Make SA-RPN V2 native-resolution tiling the sole identification path used by the default end-to-end command. Remove YOLOv8m and EfficientNet from the default runtime data flow while leaving their existing source files untouched for historical and evaluation use.

Retain and adapt the proven downstream stages:

1. MediaPipe face-region mapping with deterministic grid fallback.
2. ITA skin-tone estimation.
3. ConcernReport construction.
4. Deterministic, safety-first skincare recommendations.
5. Versioned JSON and diagnostic-image output.

The recommendation stage remains optional when the product catalog is unavailable.

## Non-goals

- Deleting or rewriting existing YOLO/EfficientNet source, training, or evaluation code.
- Supporting a YOLO/EfficientNet runtime fallback.
- Running SA-RPN directly in the main Python environment.
- Activating the unvalidated concern-efficacy ranker or learned ranker.
- Diagnosing disease or generating treatment for nevus and generic `other` detections.
- Building a generalized multi-provider inference framework.

## Architecture

The default pipeline becomes:

```text
Input face photo
  -> EXIF orientation and RGB normalization
  -> overlapping native-resolution 1024x1024 tiles
  -> SA-RPN HTTP inference for every tile
  -> full-image coordinate restoration
  -> class-agnostic overlap-over-smaller-area deduplication
  -> typed SA-RPN lesion observations
  -> MediaPipe region mapping or grid fallback
  -> ITA skin-tone estimate
  -> SA-RPN ConcernReport bridge
  -> optional concern-aware recommendation engine
  -> V2 JSON and diagnostic images
```

`python -m src.pipeline.e2e` is the default user-facing entry point and becomes SA-RPN-only. It must not import or initialize Ultralytics, TensorFlow, or the crop classifier.

### Production boundaries

#### SA-RPN HTTP client

The client:

- Sends base64-encoded tile images to the configured LitServe endpoint.
- Validates response structure, labels, confidence scores, and bounding boxes.
- Uses explicit connection/read timeouts and bounded batch sizes.
- Treats an unavailable service, timeout, failed tile, or malformed response as an analysis failure.
- Never falls back to the old identification pipeline.

#### Native tile inference

The inference module:

- Produces overlapping tiles that cover the full image, including right and bottom edges.
- Uses the current default tile size of 1024 pixels and overlap of 128 pixels.
- Translates tile-local boxes into full-image coordinates.
- Clips restored boxes to image bounds.
- Applies the existing class-agnostic overlap-over-smaller-area suppression, as explicitly selected for this migration.
- Returns typed lesion observations containing label, confidence, full-image box, and source-tile metadata.

The existing comparison command should reuse this production implementation where practical so comparison and production geometry cannot drift.

#### Downstream analysis

The existing region mapper and tone estimator remain the downstream sources of region and tone data. The pipeline records whether MediaPipe landmarks or the grid fallback produced the region assignments, including the fallback reason.

The tone estimator may produce `unknown`. `unknown` is valid downstream input and results in neutral tone handling rather than a profile-construction failure.

## Configuration

The default runtime configuration replaces local detector/classifier settings with:

- SA-RPN endpoint URL.
- Tile size.
- Tile overlap.
- HTTP timeout.
- Request batch size.
- Minimum score threshold.
- Class-agnostic deduplication threshold.
- SA-RPN-specific severity thresholds.

Old detector/classifier configuration and source files may remain for their standalone historical tooling, but `src.pipeline.e2e` must not consume them.

## SA-RPN concern bridge

SA-RPN labels map directly to downstream concerns:

| SA-RPN label | Downstream concern |
|---|---|
| Closed comedo, open comedo | `acne_comedonal` |
| Papule, pustule | `acne_inflammatory` |
| Nodule | `acne_cystic` |
| Atrophic scar, hypertrophic scar | `acne_scarring` |
| Melasma | `hyperpigmentation` |
| Nevus, other | Non-actionable safety observation |

The bridge must normalize the exact label spelling returned by the SA-RPN server in one place. An unrecognized label remains visible as an unsupported observation but cannot select treatments.

Nevus and generic `other` detections:

- Are excluded from cosmetic concern and active selection.
- Remain visible in analysis and diagnostics.
- Produce a safety observation.
- May add professional-review guidance under a deterministic confidence/count policy.

## ConcernReport V2

The ConcernReport retains its stable core fields and adds evidence needed by recommendation rules:

```json
{
  "concern": "acne_inflammatory",
  "severity": 2,
  "confidence": 0.84,
  "lesion_count": 9,
  "regions": ["left_cheek", "right_cheek"],
  "evidence": {
    "labels": {
      "papule": 6,
      "pustule": 3
    },
    "max_confidence": 0.96,
    "affected_region_count": 2
  }
}
```

Concern confidence is derived deterministically from retained SA-RPN detections rather than classifier probability mass. Counts, label composition, maximum confidence, and affected-region count remain available independently so later calibration does not require changing the inference contract.

## Provisional SA-RPN severity

The old raw-count thresholds are not reused unchanged because SA-RPN native tiling has much higher lesion recall and no 16-detection cap.

A single isolated severity function combines:

- Retained lesion count.
- Lesion type.
- Number of affected face regions.
- Detection confidence.
- Presence of nodules or hypertrophic scars.

Initial semantics:

| Severity | Policy |
|---|---|
| 0 | No retained lesions for the concern |
| 1 | Small, localized burden |
| 2 | Moderate burden or multiple affected regions |
| 3 | High burden, broad distribution, or significant scarring |
| 4 | Nodular/cystic presentation or very high inflammatory burden |

Additional rules:

- A retained nodule triggers escalation regardless of total lesion count.
- Low-confidence detections remain visible but cannot independently trigger aggressive actives.
- Thresholds are deterministic, configurable, documented as provisional, and covered by boundary tests.
- Recalibration against a labeled consumer-photo set can replace threshold values without changing the bridge interface.

## Improved recommendation engine

The engine remains deterministic and safety-first. It consumes concern severity, confidence, lesion composition, and regional distribution rather than only a coarse concern name.

### Concern behavior

- **Comedonal acne:** salicylic acid, adapalene, and azelaic acid are eligible.
- **Inflammatory acne:** benzoyl peroxide, azelaic acid, and niacinamide are eligible.
- **Nodular/cystic acne:** gentle support only, no aggressive active stacking, and dermatologist escalation.
- **Acne scarring:** sunscreen and barrier support; active acne is prioritized; significant atrophic or hypertrophic scarring adds professional-review guidance.
- **Hyperpigmentation/melasma:** daily sunscreen, azelaic acid, and niacinamide; avoid irritation-heavy combinations.
- **Nevus/other:** no targeted cosmetic products; safety observation only.

### Evidence-aware rules

- A low-confidence concern is reported as possible but cannot introduce a strong active by itself.
- Broadly distributed inflammation reduces simultaneous strong actives.
- Nodules override ordinary acne recommendations.
- Active inflammatory acne is addressed before scar-focused products.
- Deeper ITA tone increases emphasis on irritation avoidance, sunscreen, and post-inflammatory hyperpigmentation prevention. It does not create unsupported efficacy differences.
- `unknown` tone applies neutral behavior.
- Pregnancy/nursing exclusions, PM-only retinoids, incompatibility separation, exfoliant limits, strong-active suppression on soothe paths, and comedogenic vetoes remain enforced.

### Product ordering

The migration does not activate a learned or concern-mined ranker. Eligible products are ordered by:

1. Hard safety and comedogenic filters.
2. Required routine slot.
3. Match to selected concern actives.
4. Existing catalog evidence and quality fields.
5. Stable deterministic fallback order.

### Optional recommendation stage

- Valid catalog: write `routine.json` and mark recommendation status `complete`.
- Missing or unreadable catalog: complete analysis and diagnostics, omit `routine.json`, and mark recommendation status `unavailable` with a reason.
- Recommendation failure must not erase a successfully completed identification and concern analysis.

## Output contract

The output directory contains:

```text
output/
  analysis.json
  routine.json          # present only after successful recommendation
  detections.jpg
  region_overlay.jpg
  lesion_sheet.jpg
```

### `analysis.json`

`analysis.json` is the V2 source of truth and includes:

```json
{
  "schema_version": "2.0",
  "pipeline": {
    "identifier": "sa-rpn-native-tiles",
    "tile_size": 1024,
    "overlap": 128
  },
  "detections": [],
  "concerns": [],
  "skin_tone": {},
  "region_mapping": {},
  "safety_observations": [],
  "recommendation_status": "complete"
}
```

The endpoint may be recorded in a sanitized form but credentials, tokens, and sensitive query parameters must never be serialized.

Each detection records its normalized label, original server label, confidence, full-image box, assigned region, mapped concern when actionable, and source-tile metadata.

### `routine.json`

The routine output retains existing field names and structure where their meaning remains valid. New scarring and pigmentation concerns may be represented through the enriched concern data without forcing unrelated routine consumers to understand raw SA-RPN responses.

## Diagnostic images

### Detection overlay

Draw every retained full-image detection with:

- A stable class color.
- SA-RPN label.
- Confidence.
- A legend for present classes.

### Region overlay

Draw MediaPipe face-region polygons when available, otherwise draw the fallback grid. Show lesion centroids and assigned regions. Record the region method in `analysis.json`.

### Lesion sheet

Replace the old classifier prediction sheet with lesion crops showing:

- Direct SA-RPN label.
- Confidence.
- Assigned region.
- Mapped concern or safety-observation status.

No classifier probability chart remains in the production output.

## Failure behavior

- Invalid or unreadable image: fail before API requests.
- SA-RPN unavailable or timed out: fail explicitly with non-zero exit; no old-model fallback.
- Any tile request failure: fail the analysis rather than silently understating severity.
- Malformed response or invalid box: fail with tile/request context.
- No detections: produce a valid clear-skin analysis and maintenance recommendation when a catalog is available.
- MediaPipe unavailable or no face found: use the grid fallback and record the reason.
- Tone cannot be estimated: use `unknown`.
- Catalog unavailable: analysis succeeds; recommendation is unavailable.
- Unknown SA-RPN label: preserve as unsupported evidence and block treatment selection.

Partial output from a failed identification run must not be presented as a completed analysis. Temporary files should be written separately and final outputs published only after identification and concern construction succeed.

## Verification strategy

The implementation leaves focused automated checks for:

1. Native tile coverage, edge handling, and coordinate restoration.
2. SA-RPN HTTP payload construction, response validation, timeout, and malformed-response behavior.
3. Existing class-agnostic deduplication behavior.
4. Exact SA-RPN label normalization and concern mapping.
5. Provisional severity boundaries and nodule escalation.
6. Nevus, generic `other`, and unknown-label safety handling.
7. MediaPipe-to-grid fallback metadata.
8. `unknown` tone acceptance.
9. Evidence-aware recommendation priority and all retained safety constraints.
10. Successful analysis without a product catalog.
11. End-to-end fixture execution using deterministic fake SA-RPN HTTP responses.
12. V2 JSON contract and diagnostic-image creation.
13. A source-level or import-level guard proving the default e2e path does not import YOLO, TensorFlow, or the crop classifier.

A real-service smoke command should also be documented for environments with access to the SA-RPN server, but normal tests must not require the legacy CUDA/MMCV environment.

## Documentation cutover

Update the README and decision documentation so they clearly state:

- SA-RPN native tiling is the sole default identification pipeline.
- The SA-RPN service is a required external runtime dependency.
- YOLO and EfficientNet remain historical/evaluation code only and are not runtime fallbacks.
- The recommendation engine consumes direct SA-RPN concern evidence.
- Detector, concern, and recommendation metrics remain separate; no unmeasured end-to-end accuracy claim is introduced.

## Completion criteria

The cutover is complete when:

1. The default e2e command runs only the native-tile SA-RPN path.
2. The default runtime imports no YOLO or EfficientNet inference code.
3. Region mapping, tone, ConcernReport, optional recommendations, V2 JSON, and diagnostics operate from SA-RPN detections.
4. Missing catalog data does not prevent analysis output.
5. The selected severity and safety rules are tested.
6. The default documentation and command examples describe the SA-RPN-only pipeline.
7. The focused automated suite passes.
8. A fixture-based e2e run produces the complete expected artifact set.
