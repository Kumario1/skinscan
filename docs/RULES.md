# RULES

The brain. Concern → ingredient mappings, interaction constraints, and routine
ordering. Hand-written, auditable, ~40 rows. This is the SkinScan equivalent of
GGC's locked DECISIONS — small, human-owned, and where all correctness lives.

**Sources to cite when filling real values:** dermatology association guidance
(AAD), INCIDecoder ingredient explainers. This file is the plan; the machine-
readable version lives in `src/recommendation/rules.py` / a YAML config.

**Not medical advice (D-002).** These are appearance-based, cosmetic-tier
mappings. Cystic → always route to "see a professional," never treat.

## 1. Concern → recommended actives

| Concern             | First-line actives            | Also helpful               | Avoid                     |
|---------------------|-------------------------------|----------------------------|---------------------------|
| acne_comedonal      | salicylic_acid, adapalene     | azelaic_acid, mandelic     | heavy comedogenic oils    |
| acne_inflammatory   | benzoyl_peroxide, azelaic_acid| adapalene, niacinamide     | over-exfoliation          |
| acne_cystic         | (route to professional)       | soothing only: centella    | DIY strong actives        |
| hyperpigmentation   | niacinamide, vitamin_c        | azelaic_acid, alpha_arbutin, tranexamic_acid, retinol | unprotected sun (mandate SPF) |
| dryness             | ceramides, hyaluronic_acid    | glycerin, squalane, panthenol | foaming/stripping cleansers, high-% actives |

## 2. Interaction constraints (don't co-recommend in same routine step)

| Ingredient A     | Ingredient B     | Rule                                          |
|------------------|------------------|-----------------------------------------------|
| benzoyl_peroxide | retinol/adapalene| Don't combine same step (degradation/irritation); split AM/PM |
| benzoyl_peroxide | vitamin_c        | Don't layer; separate routines                |
| glycolic/lactic  | retinol          | Don't stack same night (irritation)           |
| multiple strong exfoliants | —      | Cap at one primary chemical exfoliant per routine |

## 3. Routine ordering (output structure)

Fixed order, matches catalog categories:
```
cleanser → treatment → serum → moisturizer → spf (AM only)
```
Rules:
- SPF is ALWAYS included when hyperpigmentation is present (non-negotiable — it's
  the highest-leverage step for pigmentation).
- If AM/PM split is triggered by an interaction constraint, output two routines.
- Moisturizer with ceramides always included when dryness present OR when a
  strong active (BP, retinoid) is recommended (barrier support).

## 4. Severity modifiers

| overall_severity | Behavior                                                   |
|------------------|------------------------------------------------------------|
| 0                | Maintenance routine only (gentle cleanser, moisturizer, SPF)|
| 1–2              | First-line actives, standard routine                        |
| 3                | First-line actives + "consider a professional" note         |
| 4 or any cystic  | Minimal soothing routine + strong "see a dermatologist" flag; do NOT recommend aggressive actives |

## 5. Confidence handling

- concern confidence ≥ threshold (configs/) → normal recommendation.
- below threshold → still surface the ingredient, tagged "possible — verify at a
  counter or with a professional." (D-002: loud uncertainty.)

## 6. Comedogenic down-ranking

When any acne concern is present, products carrying `comedogenic_flags`
(CATALOG_SCHEMA) are down-ranked, not excluded — surfaced last with a note.

---

**Open (Q-B related):** exact severity thresholds and confidence cutoff are
config values, not locked here — tune empirically once Stage 1 produces real
distributions.
