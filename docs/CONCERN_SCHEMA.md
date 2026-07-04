# Concern Schema (Stage 2 → Stage 3 contract)

The fixed interface between the CV side and the rules side. The CV pipeline
produces exactly this; the recommender consumes exactly this. See DECISIONS.md
D-008.

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
acne_comedonal       # blackheads / whiteheads
acne_inflammatory    # papules / pustules
acne_cystic          # nodules / cysts (flagged for "see a professional")
hyperpigmentation    # dark spots, post-acne marks, melasma-like
dryness              # rules-only for now, may lack a real detector (D-003/D-012)
```

## Closed region vocabulary

```
forehead · nose · left_cheek · right_cheek · chin_jaw · perioral
```

## Severity

Ordinal 0–4, aligned to ACNE04's grading so we don't have to remap.
`0 = none/clear, 1 = mild, 2 = moderate, 3 = significant, 4 = severe.`
(Q-B in DECISIONS: staying ordinal, not continuous.)

## Schema

```json
{
  "schema_version": "1.0",
  "image_id": "string",
  "overall_severity": 0,
  "concerns": [
    {
      "concern": "acne_inflammatory",
      "region": "left_cheek",
      "severity": 2,
      "lesion_count": 7,
      "confidence": 0.81
    }
  ],
  "meta": {
    "clear_skin": false,
    "low_light_flag": false,
    "notes": "free text, optional"
  }
}
```

## Field rules

- `concerns` may be empty → `clear_skin: true`, recommender returns a
  maintenance routine.
- One entry **per (concern, region) pair**. Inflammatory acne on both cheeks =
  two entries. This lets the recommender localize advice.
- `lesion_count` is nullable (hyperpigmentation/dryness don't count discretely).
- `confidence` in [0,1]. Below a configurable threshold (configs/), the
  recommender still lists the ingredient but tags it "possible — verify."
- `overall_severity` = max severity across acne concerns, used to trigger the
  cystic "see a professional" path.

## What this deliberately excludes

- No pixel coordinates (Stage 1 internal).
- No product info (that's Stage 3's job).
- No skin-type-wide labels ("oily skin") — v1 works per-concern-per-region only.
