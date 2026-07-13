# Concern Schema (Stage 2 → Stage 3 contract)

The fixed interface between the CV side and the rules side. The CV pipeline
produces exactly this; the recommender consumes exactly this. See DECISIONS.md
D-008.

**V2 is the default (D-026/D-027).** The production `src.pipeline.e2e`
SA-RPN path (README §7/§9) emits `schema_version: "2.0"` with the aggregated
shape documented below. The historical YOLOv8m + EfficientNetB0 two-stage
pipeline (README §1-§4, evaluation-only, not called by the default CLI) still
produces the older one-entry-per-(concern, region) shape through
`src/recommendation/bridge.py` — see the "V1 aggregation" note under Field
rules.

## Design intent

- **Face-anchored, not pixel-anchored.** The recommender cares about *regions*
  (forehead, cheeks, chin/jaw, nose), not exact coordinates. Pixel boxes are a
  Stage-1 internal detail; by the time we hit the contract they're summarized
  into regions. This keeps the rules stable even as detectors change.
- **Uncertainty is first-class.** Every concern carries a confidence. The rules
  layer decides how to treat low confidence (D-002: loud about uncertainty).
- **Concern vocabulary is a closed set.** The rules table keys on these exact
  strings. Adding a concern type is a logged decision.

## Closed concern vocabulary

```
acne_comedonal       # blackheads / whiteheads / open & closed comedones
acne_inflammatory    # papules / pustules
acne_cystic          # nodules / cysts (flagged for "see a professional")
acne_scarring        # atrophic / hypertrophic scars (V2, D-026/D-027) —
                      # its own concern; scars are NOT folded into
                      # hyperpigmentation
hyperpigmentation    # dark spots, post-acne marks, melasma (V2 SA-RPN
                      # source: melasma only — see src/pipeline/sarpn.py
                      # SARPN_LABEL_TO_CONCERN)
dryness               # rules-only for now, may lack a real detector (D-003/D-012)
```

## Closed region vocabulary

```
forehead · nose · left_cheek · right_cheek · chin_jaw · perioral
```

## Severity

Ordinal 0–4, aligned to ACNE04's grading so we don't have to remap.
`0 = none/clear, 1 = mild, 2 = moderate, 3 = significant, 4 = severe.`
(Q-B in DECISIONS: staying ordinal, not continuous.)

**V2 (SA-RPN bridge) severity is provisional and config-driven**
(`configs/default.yaml: sa_rpn.severity`) — it does **not** reuse
`concern_report.severity_count_thresholds` (that key is consumed only by the
historical bridge, `src/recommendation/bridge.py`). Rules apply in this order
(`_severity`, `src/pipeline/sarpn.py`):

| Rule | Effect |
|---|---|
| any `nodule` detection present | severity forced to `nodule_severity` (**4**) — short-circuits the rest |
| otherwise: lesion count → severity | per-concern count thresholds via `bisect_right` over `sa_rpn.severity.count_thresholds[concern]` |
| 2 affected regions | severity floored at **2** |
| affected regions ≥ `broad_region_count` (**3**) | severity floored at **3** |
| any `hypertrophic_scar` detection present | severity floored at `hypertrophic_scar_min_severity` (**3**) |
| max retained detection score < `confidence_cutoff` (**0.5**) | severity capped at **1** |

Current `count_thresholds` (`configs/default.yaml`):

```yaml
count_thresholds:
  acne_comedonal:     [1, 8, 20, 40]
  acne_inflammatory:  [1, 6, 15, 30]
  acne_scarring:      [1, 3, 8, 20]
  hyperpigmentation:  [1, 4, 10, 25]
```

`ConcernReport.overall_severity` (a derived property, not a stored field) is
still the max severity across `acne_*` concerns, used to trigger the cystic
"see a professional" path (`docs/RULES.md` §4).

## Schema

```json
{
  "schema_version": "2.0",
  "image_id": "string",
  "concerns": [
    {
      "concern": "acne_inflammatory",
      "regions": ["left_cheek", "right_cheek"],
      "severity": 2,
      "lesion_count": 9,
      "confidence": 0.71,
      "evidence": {
        "labels": {"papule": 5, "pustule": 4},
        "max_confidence": 0.91,
        "affected_region_count": 2
      }
    }
  ],
  "clear_skin": false,
  "low_light_flag": false,
  "notes": "free text, optional"
}
```

## Field rules

- `concerns` may be empty → `clear_skin: true`, recommender returns a
  maintenance routine.
- **V2 aggregation: one entry per concern**, not per (concern, region) pair.
  Inflammatory acne on both cheeks is a **single** `acne_inflammatory` entry
  with `regions: ["left_cheek", "right_cheek"]` and
  `evidence.affected_region_count: 2`. The `Concern` dataclass
  (`src/recommendation/schema.py`) keeps a singular `region` field internally
  — the canonical first entry of `regions`, kept only for backward-compatible
  positional construction and code that still reads `concern.region` — but it
  is **not** part of the V2 JSON payload above.
  - *V1 aggregation (historical):* the two-stage pipeline's bridge
    (`src/recommendation/bridge.py`) still constructs one `Concern` per
    (concern, region) pair, each with a single-element `regions` list — the
    original v1 shape this document used to describe as the default.
- `lesion_count` is the count of retained detections that produced the
  concern; still nullable for concerns without a discrete count (dryness).
- `confidence` is the **mean of the retained detection scores** that produced
  the concern (`sum(scores) / len(scores)`), in `[0, 1]`. Below a configurable
  threshold (`recommendation.concern_confidence_cutoff`, `configs/`), the
  recommender flags the concern `"possible — verify"` and adds **no actives**
  for it — see `docs/RULES.md` §5 for the V2 confidence-gating change (it
  previously still listed the ingredient under the flag; it no longer does).
- `evidence` (V2 only) carries the raw signal behind the aggregation:
  `labels` (per-source-label detection counts, e.g.
  `{"papule": 5, "pustule": 4}`), `max_confidence` (highest single retained
  detection score), and `affected_region_count` (must equal `len(regions)`
  whenever evidence is non-default — enforced in `Concern.__post_init__`).
- Unknown SA-RPN labels, and the closed `nevus`/`other` labels
  (`SARPN_NON_ACTIONABLE_LABELS`), never become concerns. They surface as
  `safety_observations` in `analysis.json` instead — non-actionable
  `nevus`/`other` observations (gated by per-label count/confidence
  thresholds, `sa_rpn.severity.professional_review`) and an
  `unsupported_label` observation for anything outside the label map —
  visible, never silently dropped.

## What this deliberately excludes

- No pixel coordinates in the concern schema itself (SA-RPN detection boxes
  are reported separately, in `analysis["detections"]`).
- No product info (that's Stage 3's job).
- No skin-type-wide labels ("oily skin") — this contract works per-concern,
  optionally per-region, only.
