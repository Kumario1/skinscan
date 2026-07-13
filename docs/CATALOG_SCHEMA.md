# Catalog Schema (product data contract)

How products are represented after import. Raw source data (Kaggle Sephora,
D-015) is messy; it gets normalized into this on import. See DECISIONS.md D-009.

## Design intent

- **Ingredient-normalized.** The raw INCI string is parsed once, at import, into
  a canonical `actives` list. The recommender never parses ingredient text at
  query time — it filters on `actives`. (D-006: reason over ingredients.)
- **Parse only what we use.** We normalize the ~30 actives the rules table keys
  on, plus a comedogenic flag list. Every other ingredient string is discarded.
  GGC parser-registry instinct.
- **Prices are decorative.** Stored if present, never trusted, never used in
  logic. (D-001: no live pricing.)

## Product schema v2 (D-029)

```json
{
  "catalog_schema_version": "2",
  "product_id": "string",
  "name": "CeraVe Foaming Facial Cleanser",
  "brand": "CeraVe",
  "category": "cleanser",
  "actives": ["niacinamide", "ceramides"],
  "comedogenic_flags": [],
  "price_usd": 15.99,
  "price_is_stale": true,
  "intended_areas": ["face"],
  "routine_roles": ["cleanser"],
  "format": "gel",
  "exposure": "rinse_off",
  "drug_actives": [],
  "otc_drug": false,
  "label_source": "https://authoritative.example/label",
  "label_verified_at": "2026-07-13T00:00:00Z",
  "broad_spectrum": null,
  "spf": null,
  "comedogenic_claim": "unknown",
  "irritant_features": [],
  "contraindications": [],
  "evidence_roles": ["cleanser"],
  "evidence_grade": "verified_label",
  "cadence": "per_label",
  "cadence_source": "https://authoritative.example/label",
  "amount": null,
  "amount_source": null
}
```

`actives` is the complete set of canonical actives known to be carried from
the ingredient list. It is not proof of delivered strength or therapeutic
role. `drug_actives` holds only independently verified active/strength/source
tuples. Legacy rows still load with `catalog_schema_version: "legacy"` and
unknown/empty v2 fields, but gain no eligibility by inference.

Storage validity is deliberately broader than recommendation eligibility.
Unknown metadata is preserved, then the importer writes a deterministic
quarantine report listing each unavailable role and stable reason codes.

## Closed category vocabulary

Routine ordering depends on this being a closed set:

```
cleanser · treatment · serum · moisturizer · spf
```

Products that don't map to one of these are dropped at import (v1 scope).

## Canonical actives (the ~30 that matter)

The normalizer maps raw INCI variants → these IDs. This list IS the vocabulary
the rules table (RULES.md) keys on. Kept deliberately small.

```
Acne / exfoliation:   salicylic_acid, benzoyl_peroxide, adapalene,
                      azelaic_acid, glycolic_acid, lactic_acid, mandelic_acid
Pigmentation:         niacinamide, vitamin_c, alpha_arbutin, tranexamic_acid,
                      kojic_acid, retinol
Barrier / hydration:  ceramides, hyaluronic_acid, glycerin, squalane,
                      panthenol, centella
Soothing:             allantoin, madecassoside, zinc
Sun:                  spf (category-level, not an active)
```

## Comedogenic flag list (negative signals)

Products containing these get flagged, so the recommender can down-rank them for
acne-prone concerns. Adds noticeable "smartness" for one more lookup (D-006 note).

```
coconut_oil · isopropyl_myristate · isopropyl_palmitate ·
algae_extract · certain cocoa/wheat-germ derivatives
```

## Normalizer requirements

- Case-insensitive, punctuation-tolerant.
- Handles parenthetical aliases: "Ascorbic Acid (Vitamin C)" → `vitamin_c`.
- Handles common synonyms: "Sodium Hyaluronate" → `hyaluronic_acid`;
  "Tocopherol" is ignored (not in our active list).
- Fuzzy-match spelling variants for the ~30 actives only; exact-miss on anything
  else = silently dropped.
- Import is idempotent and logged: report how many products got ≥1 active vs
  zero (zero-active products are still valid carriers, e.g. plain moisturizers).

## Sephora import: category mapping (D-015)

The real Kaggle dump (`product_info.csv`, ~8.5k rows) is imported with
`import_catalog.py --format sephora`. A per-row adapter renames its columns
(`product_id → product_id` — **preserved**, load-bearing for joining reviews;
`product_name → name`, `brand_name → brand`, `ingredients`, `price_usd → price`)
and maps its taxonomy onto the closed five categories. Everything downstream of
the adapter is unchanged — same normalizer, same schema.

Category rule: keep only `primary_category == "Skincare"`, then this exact-string
table on `(secondary_category, tertiary_category)`. **Verified against the actual
CSV** (8,494 rows, 39 skincare pairs, 2026-07-09) — not written from memory.

| → category | Sephora (secondary / tertiary) pairs |
|------------|--------------------------------------|
| cleanser | Cleansers / Face Wash & Cleansers · Cleansers / Makeup Removers · Cleansers / Face Wipes · Cleansers / _(empty)_ |
| treatment storage category | Cleansers / Toners · Cleansers / Exfoliators · Treatments / Facial Peels · Treatments / Blemish & Acne Treatments · Masks / Face Masks · Masks / Sheet Masks |
| serum | Treatments / Face Serums |
| moisturizer storage category | Moisturizers / Moisturizers · Mists & Essences · Face Oils · Night Creams · Decollete & Neck Creams · _(empty)_ |
| spf | Sunscreen / Face Sunscreen · Sunscreen / _(empty)_ |

These mappings preserve a coarse historical storage category only. They do
not assign v3 routine eligibility. Source facts survive separately: removers
and wipes are rinse-off cleansing formats; exfoliators are scrubs; peels and
masks remain peel/mask exposures with no daily treatment role; decollete/neck
cream has intended area `neck` and no facial moisturizer role. Only explicit
verification overlay data may add a role, source, strength, claim, cadence, or
amount.

Everything else is **dropped and counted** in the import log's
`dropped_by_category` breakdown: non-Skincare primaries (Makeup, Hair, …) and
skincare pairs with no step in the closed vocabulary — eye care, lip balms, gift
sets, mini sizes, high-tech tools, wellness/supplements, self-tanners, BB/CC
creams, body sunscreen, blotting papers. On the full dump this keeps **1,634
products** (all five categories non-empty, ~89% with ≥1 canonical active).

## Verification overlay and quarantine

The optional overlay is a JSON object keyed by `product_id`. Every supplied
field is schema-validated with product/field context, then merged in sorted
product order. Verified drug actives are also included in the complete
carried-active set so safety checks cannot miss an unrelated active. The raw
Sephora/BeautyAPI import never manufactures strengths, label URLs,
verification timestamps, broad-spectrum claims, contraindications, cadence,
or amounts.

Treatment quarantine checks face area, treatment role, daily format,
leave-on/approved exposure, verified drug active strength, label source, and
timestamp. SPF additionally requires explicit broad-spectrum `true` and
numeric SPF ≥30. Unknown comedogenic/contraindication metadata remains unknown,
not false. Masks, scrubs, peels, neck products, and trace-active rinse-off
cleansers cannot masquerade as verified leave-on therapy.

## What this deliberately excludes

- No reviews / ratings / loves on the catalog itself — popularity ranking
  exists (D-028) but lives in the review-stats artifact and the ranker, never
  on the catalog schema.
- No concentration claims derived from INCI lists. Strength may exist only in
  `drug_actives` when an authoritative overlay supplies it.
- No stock / availability.

## Review-stats & ranker artifacts

The reviews are a **separate artifact**, not part of the catalog schema (D-009
unchanged) — `data/raw/sephora/reviews_*.csv` join to the catalog on the
preserved Sephora `product_id`. The learned ranker (D-022, `ranker.py`) turns
them into two files:

**`review_stats.json`** — per-product × skin-type evidence, from train rows only:

```json
{
  "min_cell_size": 5,
  "base_rate": 0.83,
  "global_mean_rating": 4.19,
  "cells": {
    "P480274": {
      "__all__": {"n": 812, "mean_rating": 4.4, "pct_recommend": 0.86},
      "oily":    {"n": 210, "mean_rating": 4.2, "pct_recommend": 0.81}
    }
  },
  "loves": {"P480274": 118000}
}
```

`Ranker.evidence(product_id, skin_type)` returns the `<skin_type>` cell when its
`n >= min_cell_size`, else the `__all__` cell tagged `{"fallback": true,
"cell": "all_reviewers"}`; `None` when the product is absent.

`global_mean_rating` (top-level) is the train-rows mean rating — the smoothing
prior `StatsRanker` shrinks each product's pooled rating toward (D-022 amendment).

`loves` (top-level, D-028) — `{product_id: loves_count}` joined from
`product_info.csv` at stats-build time, catalog products only; feeds
`StatsRanker`'s popularity nudge (`ranker.popularity_weight`). Absent when the
product-info file wasn't available at build time — the nudge degrades to 0.

**`ranker.joblib`** — the model bundle (joblib dict), written **only when the
D-022 gate passes**: `{"model"` (HistGradientBoostingClassifier on
`is_recommended`), `"brand_vocab"`, `"active_vocab"`, `"feature_columns"`,
`"base_rate"}`. `feature_columns`/`brand_vocab`/`active_vocab` let inference
reconstruct the exact training columns (anti-skew). A disaggregated
`eval.json` (pooled + per-tone-bucket ROC-AUC/pairwise) is always written.
