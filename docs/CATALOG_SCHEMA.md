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

## What this deliberately excludes

- No reviews / ratings (v1 doesn't rank on popularity).
- No concentration data (INCI lists don't provide it reliably).
- No stock / availability.
