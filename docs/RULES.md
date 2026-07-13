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
| acne_comedonal      | salicylic_acid, adapalene, azelaic_acid | mandelic          | heavy comedogenic oils    |
| acne_inflammatory   | benzoyl_peroxide, azelaic_acid, niacinamide | adapalene       | over-exfoliation          |
| acne_cystic         | (route to professional)       | soothing only: centella    | DIY strong actives        |
| acne_scarring       | ceramides (barrier support) + SPF (mandate) | —             | aggressive actives without professional review |
| hyperpigmentation   | azelaic_acid, niacinamide     | alpha_arbutin, tranexamic_acid, retinol | unprotected sun (mandate SPF) |
| dryness             | ceramides, hyaluronic_acid, glycerin | squalane, panthenol | foaming/stripping cleansers, high-% actives |

**V2 changes** (`src/recommendation/engine.py: CONCERN_ACTIVES`, D-026/D-027):

- `acne_scarring` is now its own concern — V2 SA-RPN scars
  (`atrophic_scar`/`hypertrophic_scar`) no longer fold into
  `hyperpigmentation` (only `melasma` does). First-line is barrier-support
  `ceramides`; SPF is mandatory whenever `acne_scarring` is present (§3); a
  severity ≥ 3 or any `hypertrophic_scar` evidence additionally flags
  `"consider professional review for acne scarring"`.
- `hyperpigmentation` first-line dropped `vitamin_c` — it is now
  `azelaic_acid, niacinamide` only (`STRONG_ACTIVES` still contains
  `vitamin_c` for interaction-constraint/soothe-path purposes; it's just no
  longer proposed as a first-line hyperpigmentation active).

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
     takes the complement**; an active absent from `PREFERRED_SLOT` takes the
     complement of the earlier active's preference (PM when that is also
     unlisted).
5. **Terminal drop** — only a pair still sharing a slot after step 4 (both pinned
   to the same single slot) falls back to the legacy behavior: drop the later
   active with a "held back (conflicts with earlier active)" flag.

**benzoyl_peroxide + vitamin_c** (both prefer AM): the later-listed active wins
its preference, so **vitamin_c→AM and benzoyl_peroxide→PM**. They coexist across
slots; neither is held back.

At the product level, a product must satisfy EVERY matched target active's
slot pins (set intersection, not union): a product bundling a PM-pinned
retinoid with an AM-eligible active appears only in PM, and a product whose
matched actives have disjoint slots is excluded entirely.

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
- SPF is ALWAYS included when `hyperpigmentation` OR `acne_scarring` is
  present (V2: extended to scarring — non-negotiable, it's the
  highest-leverage step for pigmentation and for protecting healing scar
  tissue). `engine.py`'s `needs_spf` flag is set from the concern's presence
  alone, independent of its confidence — see §7.
- If AM/PM split is triggered by an interaction constraint, output two routines.
- Moisturizer with ceramides always included when dryness present OR when ANY
  `engine.STRONG_ACTIVES` member (chemical exfoliants, retinol/adapalene,
  benzoyl_peroxide, azelaic_acid, vitamin_c, gluconolactone, willow_bark) is
  in `target_actives` (barrier support) — not only BP/retinoid.

## 4. Severity modifiers

| overall_severity | Behavior                                                   |
|------------------|------------------------------------------------------------|
| 0                | Maintenance routine only (gentle cleanser, moisturizer, SPF)|
| 1–2              | First-line actives, standard routine                        |
| 3                | First-line actives + "consider a professional" note         |
| 4 or any cystic  | Minimal soothing routine + strong "see a dermatologist" flag; do NOT recommend aggressive actives |

"Do NOT recommend aggressive actives" is enforced at the PRODUCT level on the
soothe and maintenance (severity 0) paths: a product is excluded when it
carries ANY strong active (`engine.STRONG_ACTIVES` — exfoliating acids,
BP, retinoids, azelaic, vitamin C, plus the PHA/botanical exfoliant sources
gluconolactone and willow bark), even when it also contains a matching gentle
active — an "SA + hyaluronic acid serum" stays out. Products whose NAME
markets exfoliation (AHA/BHA/PHA, "exfoliating", "peel", "resurfacing") are
excluded there too, since citrus-juice acids and enzymes don't parse out of
INCI. Sub-threshold concerns keep their "possible — verify" flags on the
escalation path (§5 applies everywhere).

## 5. Confidence handling

- concern confidence ≥ threshold (`recommend()`'s `conf_cutoff` parameter,
  `src/recommendation/engine.py`, default 0.5 — the `configs/default.yaml`
  key `recommendation.concern_confidence_cutoff` is not currently wired to
  it) → normal recommendation: first-line actives for that concern are added
  to `target_actives`.
- **V2 change:** below threshold → the concern contributes **no actives at
  all** (strong or gentle) to `target_actives`; only the flag
  `"{concern}@{regions}: possible — verify"` is emitted (D-002: loud
  uncertainty). This replaces the earlier behavior of still listing the
  ingredient under a "verify" tag. `needs_spf` (hyperpigmentation/
  acne_scarring, §3) and the deep-tone flag (§7) are exceptions — both are
  decided from the concern's presence in the report, not its confidence, so
  they still apply even when the concern itself contributed no actives.
- The cystic/severe soothe-only escalation (§4, `overall_severity >= 4` or
  any `acne_cystic`) bypasses per-concern confidence gating entirely: every
  low-confidence concern in the report still gets its "possible — verify"
  flag, but the target list is fixed to `centella, ceramides,
  hyaluronic_acid` regardless of confidence.

## 6. Comedogenic down-ranking

When any acne concern is present, products carrying `comedogenic_flags`
(CATALOG_SCHEMA) are down-ranked, not excluded — surfaced last with a note.

## 7. V2 evidence-aware adjustments (SA-RPN bridge)

These apply to reports whose concerns carry V2 `evidence` (`labels`,
`max_confidence`, `affected_region_count` — `docs/CONCERN_SCHEMA.md`), i.e.
the production `src.pipeline.e2e` SA-RPN path.

- **Broad-inflammatory de-stacking.** When an `acne_inflammatory` concern's
  `evidence.affected_region_count >= 3` (`broad_inflammation`) and both
  `benzoyl_peroxide` and `azelaic_acid` are in `target_actives`, the engine
  builds a probe routine first; only if `azelaic_acid` actually surfaces a
  catalog product in that probe does it drop `benzoyl_peroxide` from
  `target_actives` and flag `"broad inflammation: reduced strong-active
  stacking"`. If no azelaic-containing product exists in the catalog,
  benzoyl_peroxide is kept — an empty treatment slot is worse than a
  slightly heavier one.
- **Deep-tone emphasis wording.** `profile.tone_bucket == "deep"` AND the
  report contains `acne_inflammatory`, `acne_scarring`, or
  `hyperpigmentation` → flag `"deeper tone: emphasize sunscreen and
  irritation avoidance to reduce post-inflammatory hyperpigmentation risk"`.
  Decided from the full reported-concern set, independent of per-concern
  confidence (a low-confidence hyperpigmentation concern still triggers it).
- **Unknown-tone neutrality.** `TONE_BUCKETS` (`src/recommendation/schema.py`)
  includes `"unknown"` alongside `light`/`medium`/`deep` — matching the
  photo-tone estimator, which reports `"unknown"` rather than guessing when
  too few skin pixels are sampled (`src/pipeline/tone.py`). Unknown (or
  absent) tone never triggers the deep-tone guidance above and is otherwise
  treated exactly like `light`/`medium`: the recommender stays neutral by
  default rather than assuming risk when tone can't be estimated, while still
  reporting the bucket honestly (D-016 discipline: never silently dropped).
- **Ranker default.** Production `src.pipeline.e2e` always calls
  `recommend(report, catalog, profile=profile, ranker=None)` — every run
  ships the deterministic rules-only order (§2c, D-019). The duck-typed
  ranker hook and `StatsRanker` (D-022) remain available to
  `src.recommendation.ranker` / standalone bake-off evaluation only; nothing
  in the production e2e CLI activates them.

---

**Open (Q-B related):** exact severity thresholds and confidence cutoff are
config values, not locked here — tune empirically once Stage 1 produces real
distributions.
