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
| benzoyl_peroxide | retinol/adapalene| Don't combine same step; time-split (retinoid→PM, BP→AM) |
| benzoyl_peroxide | vitamin_c        | Don't layer; time-split across slots (they COEXIST — see below) |
| glycolic/lactic  | retinol          | Don't stack same night (retinoid→PM, AHA→the other slot) |
| multiple strong exfoliants | —      | Cap at one primary chemical exfoliant per routine |

### 2a. Time-split resolution (Engine v2, D-005/D-021 aware)

Incompatible pairs are no longer resolved by dropping one active — they are
split across AM/PM slots so both survive. Deterministic algorithm
(`engine._assign_slots`):

1. **Default** — every active is eligible for both slots (`{AM, PM}`).
2. **Retinoid pin** — retinol/adapalene pin to **PM** (photosensitivity).
3. **Exfoliant cap** — a second chemical exfoliant (glycolic/lactic/mandelic/
   salicylic) is dropped with a "one chemical exfoliant per routine" flag.
4. **Pair shrink** — for each INCOMPATIBLE pair that still shares a slot:
   - if one member is already pinned to a single slot, the other takes the
     **complement** (e.g. adapalene=PM ⇒ benzoyl_peroxide=AM);
   - if both are still free, the **later-listed active takes its preferred slot**
     (`PREFERRED_SLOT`: BP→AM, vitamin_c→AM, AHAs→PM) and the **earlier active
     takes the complement**.
5. **Terminal drop** — only a pair still sharing a slot after step 4 (both pinned
   to the same single slot) falls back to the legacy behavior: drop the later
   active with a "held back (conflicts with earlier active)" flag.

**benzoyl_peroxide + vitamin_c** (both prefer AM): the later-listed active wins
its preference, so **vitamin_c→AM and benzoyl_peroxide→PM**. They coexist across
slots; neither is held back.

### 2b. Pregnancy / nursing (D-021)

If the `UserProfile.pregnant_or_nursing` flag is set, retinoids (retinol,
adapalene) are stripped from the target actives **before** conflict resolution,
with the flag: *"retinoids omitted (pregnancy/nursing) — cosmetic guidance only,
confirm with your doctor."* Other actives (e.g. salicylic_acid, azelaic_acid)
are unaffected. Not medical advice (D-002).

### 2c. Ranker re-ordering (D-005)

An optional learned ranker may be passed to `recommend(...)`. It is duck-typed
(`ranker.score(product, profile)`) and the engine imports **no** ML. The ranker
**only reorders** rule-approved candidates within a single slot × category — it
can never add, remove, or re-slot a product, nor touch flags. The comedogenic
partition (§6) ALWAYS dominates the sort key `(len(comedogenic_flags),
-score)`, so a comedogenic product never outranks a clean one regardless of
score. `ranker=None` degrades to the deterministic rules-only order (D-019).

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
