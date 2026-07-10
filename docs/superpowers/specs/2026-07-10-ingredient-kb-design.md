# Ingredient knowledge base + tier-2 catalog — design

Date: 2026-07-10. Status: approved by maintainer (brainstorming session).
Complements the concern-efficacy recommender design (same date); does not
change its gates or the review-labeling data layer (plan 015).

## Goal

Use an external ingredient dataset to (a) build an ingredient-level knowledge
base, (b) enrich the existing Sephora catalog with better comedogenic flags
and a per-concern ingredient-match score, and (c) add a second-tier product
catalog that fills slots only when no review-backed product can.

## Source dataset

`thebeautyapi/beautyproducts` on HuggingFace — `beauty_data.jsonl` (~11 MB,
~1–10k products; the "180k" figure is their paid API, out of scope). Each
product row carries a structured ingredient list with per-ingredient
`comedogenicity`, `irritancy`, `functions`, and an actives `rating`
("direct actives" / "supporting actives").

License: **CC-BY-NC-4.0** (non-commercial). Acceptable for this project's
research/portfolio use; a commercial deployment would need the paid API or a
different source. Record this in DECISIONS.md.

Rejected sources (evaluated 2026-07-10):
- `Scanalyze/Skincare-ingredients` (HF): one 86 KB CSV, duplicate rows,
  generic one-line descriptions. No signal beyond the beautyapi metadata.
- Kaggle INCI list: ingredient descriptions, superseded by beautyapi
  per-ingredient fields.
- Kaggle skincare-products-clean-dataset: ~1k products with plain-text
  ingredient strings — strictly worse than the Sephora catalog we have.

Download is a documented manual step (like the Sephora data), into
`data/raw/beautyapi/beauty_data.jsonl`. Tests never touch the network.

## Deliverables

### 1. Ingredient KB — `data/processed/ingredient_kb.json`

New module `src/recommendation/ingredient_kb.py`, `build_kb(rows) -> dict`:

- Walk every product's ingredient entries; aggregate by normalized ingredient
  name (same normalization `import_catalog.py` uses: lowercase, strip
  punctuation/whitespace).
- Keep per ingredient: `comedogenicity: int|None`, `irritancy: int|None`,
  `functions: list[str]`, `rating: str|None`, plus alias names encountered
  (`label_name`, `other_names`, `ph_eur_name`) so catalog lookup hits more.
- Conflicting values across rows → max comedogenicity / max irritancy
  (conservative); union of functions; "direct actives" wins over
  "supporting actives" for rating.
- CLI build step (`python -m src.recommendation.ingredient_kb`) reads the raw
  JSONL and writes the KB. Deterministic: sorted keys, no RNG.

### 2. Ingredient-match score

`match_score(raw_ingredients: str, concern: str, kb: dict) -> float` in the
same module:

- Hand-curated `CONCERN_ACTIVES` table in the module: concern → set of
  beneficial ingredient names/functions, extending the D-006 actives map
  (e.g. `acne_comedonal` → salicylic acid, retinol/retinal, azelaic acid,
  niacinamide…). Auditable, no ML.
- Score: +1 per matched beneficial active, discounted by INCI position
  (ingredient lists are concentration-ordered, so a match at position 2
  counts more than at position 40); −1 per ingredient with
  comedogenicity ≥ 3 when the concern is an acne concern. Squash to [0, 1].
- Deterministic and pure — takes the KB as an argument, no I/O.

### 3. Catalog enrichment (tier 1)

`import_catalog.py` gains an optional KB pass:

- Comedogenic flags come from KB lookups — a superset of the current
  ~30-entry hand-list, which stays as fallback when the KB is absent.
- Each product gets `ingredient_match: {concern: float}` persisted in
  `catalog.json`.
- Without the KB file present, the importer behaves exactly as today
  (backwards compatible; fast tests don't need the KB).

### 4. Tier-2 catalog — `data/processed/catalog_tier2.json`

- Same importer path run over beautyproducts rows, mapped into the existing
  `Product` schema plus `tier: 2` and `no_outcome_data: true`.
- Products that don't map to one of the five catalog categories are dropped.
- `engine.py` slot-filling: tier-2 products are considered **only when a slot
  has no tier-1 candidate**, and the `no_outcome_data` flag carries through
  to the routine output so the UI/consumer can label them honestly.

## Ranking integration

The ingredient-match score is a **prior/tiebreaker, not a blended feature**:
within a slot, review-backed concern-stats (plan 015) dominate; the match
score orders products with equal or absent stats. No weights to tune. This
preserves the honesty property from the concern-efficacy design — outcome
data beats ingredient plausibility whenever it exists.

## Testing

- Fixture: ~5 handwritten beautyproducts-style JSONL rows checked into
  `tests/fixtures/`.
- KB aggregation: conflicting comedogenicity → max; alias lookup works.
- Match score: a known fixture product ranks above a known-worse one for a
  given concern; comedogenic penalty applies only to acne concerns;
  position discount ordering.
- Tier-2 fallback: a slot with no tier-1 candidate pulls a tier-2 product,
  flagged `no_outcome_data`; a slot with tier-1 candidates never does.
- No network calls in the fast suite; importer without KB file unchanged
  (regression test).

## Out of scope

- The paid Beauty API (full 180k catalog).
- Learned/blended ranking weights.
- UI changes beyond passing the `no_outcome_data` flag through.
- Kaggle dataset ingestion.
