# Catalog verification

Human-approved schema-v2 overlays belong here. Approval records must include
authoritative source URLs, retrieval timestamps and content hashes, reviewer
identity, and approval time. Proposed or stale assertions never affect product
eligibility.

`catalog_completeness.json` records the current release inventory gap. Regenerate
it with:

```bash
python -m src.recommendation.catalog_completeness \
  data/processed/catalog.json data/processed/catalog_tier2.json \
  --out data/verification/catalog_completeness.json
```

The command exits non-zero until every support role has 25 eligible products
and every modeled treatment path has at least one exact eligible product.
