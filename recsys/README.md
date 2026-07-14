# recsys — acne product recommendation system

Standalone rebuild of the recommendation half of SkinScan. Couples to the rest
of the repo through **three file contracts only**: reads `analysis.json`
(schema 3) + `profile.json`, writes `recommendations.json`. No imports from
`src/`. Stdlib only outside `recsys/tools/`.

## Run it

```bash
python -m recsys recommend \
  --analysis runs/e2e/v3-pr17-valid-104/analysis.json \
  --profile tests/fixtures/profile_complete.json \
  --out runs/recsys/v3-pr17-valid-104/recommendations.json
```

Output: 5 routine archetypes (best overall, budget, gentle/sensitive, minimal,
comprehensive), each a safety-valid AM/PM routine with a per-product `why`
built from the exact signal values the ranker used.

## Tests

```bash
python -m pytest recsys/tests -q
```

The `raw_dump`-marked test additionally verifies the committed seed catalog is
a byte-identical rebuild from the Kaggle dump (skips when the dump is absent).

## Rebuild the data

The raw Kaggle Sephora dump lives at the repo level (`data/raw/sephora/`,
gitignored — main checkout only). All committed artifacts regenerate
deterministically (no timestamps → byte-identical):

```bash
RAW=/Users/princekumar/Documents/skinscan/data/raw/sephora
python -m recsys.tools.build_catalog --raw-dir $RAW \
  --only-ids recsys/data/catalog/seed_ids.txt --out recsys/data/catalog/seed_catalog.json
python -m recsys.tools.build_review_stats --raw-dir $RAW \
  --catalog recsys/data/catalog/seed_catalog.json \
  --out recsys/data/signals/review_stats.v1.json --data-root recsys/data
python -m recsys.tools.build_popularity --raw-dir $RAW \
  --catalog recsys/data/catalog/seed_catalog.json \
  --out recsys/data/signals/popularity.v1.json --data-root recsys/data

# Paid, offline Phase 1 build (requires OPENROUTER_API_KEY; default cap: 100 new labels)
python -m recsys.tools.build_ingredient_analysis \
  --catalog recsys/data/catalog/seed_catalog.json \
  --out recsys/data/signals/ingredient_analysis.v1.json --data-root recsys/data
```

Full-catalog twins go to `recsys/data/derived/` (gitignored).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data architecture, pipeline
stages, and the phase plan (ingredient analysis, concern efficacy, verification
overlay, full catalog, integration).
