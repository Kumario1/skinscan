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

## Product schema

```json
{
  "product_id": "string",
  "name": "CeraVe Foaming Facial Cleanser",
  "brand": "CeraVe",
  "category": "cleanser",
  "actives": ["niacinamide", "ceramides"],
  "comedogenic_flags": [],
  "price_usd": 15.99,
  "price_is_stale": true,
  "raw_ingredients": "aqua, niacinamide, ..."
}
```

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
| cleanser | Cleansers / Face Wash & Cleansers · Cleansers / Toners · Cleansers / Makeup Removers · Cleansers / Face Wipes · Cleansers / _(empty)_ |
| treatment | Cleansers / Exfoliators · Treatments / Facial Peels · Treatments / Blemish & Acne Treatments · Masks / Face Masks · Masks / Sheet Masks |
| serum | Treatments / Face Serums |
| moisturizer | Moisturizers / Moisturizers · Mists & Essences · Face Oils · Night Creams · Decollete & Neck Creams · _(empty)_ |
| spf | Sunscreen / Face Sunscreen · Sunscreen / _(empty)_ |

Non-obvious calls: Toners/Removers/Wipes are the cleansing phase; Exfoliators and
Masks are the treatment step regardless of Sephora's grouping; Mists/Essences/
Oils/Night/Neck creams are leave-on moisturizers.

Everything else is **dropped and counted** in the import log's
`dropped_by_category` breakdown: non-Skincare primaries (Makeup, Hair, …) and
skincare pairs with no step in the closed vocabulary — eye care, lip balms, gift
sets, mini sizes, high-tech tools, wellness/supplements, self-tanners, BB/CC
creams, body sunscreen, blotting papers. On the full dump this keeps **1,634
products** (all five categories non-empty, ~89% with ≥1 canonical active).

## What this deliberately excludes

- No reviews / ratings (v1 doesn't rank on popularity).
- No concentration data (INCI lists don't provide it reliably).
- No stock / availability.
