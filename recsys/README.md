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

To publish both recommendation contracts atomically from the SA-RPN pipeline:

```bash
python -m src.pipeline.e2e \
  --image path/to/image.jpg --api http://localhost:8000/predict \
  --recsys --out runs/e2e/sample
```

The run directory keeps the existing `routine.json` and adds the standalone
`recommendations.json`. Use `--recsys-data-root recsys/data/derived` and
`--recsys-catalog recsys/data/derived/catalog_full.json` for the local full
catalog build.

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

# Free, offline Phase 1 build (OPENROUTER_API_KEY or OPENROUTER_KEY).
# The pinned free MoE is recorded in every cache/store entry.
python -m recsys.tools.build_ingredient_analysis \
  --catalog recsys/data/catalog/seed_catalog.json \
  --out recsys/data/signals/ingredient_analysis.v1.json --data-root recsys/data

# Phase 2 uses semantic model labels plus the versioned literal-policy layer.
# p7 passed the independent 50-row exact-set gate at 44/50 (88%).
python -m src.recommendation.concern_labels probe
python -m src.recommendation.concern_labels calibrate
python -m src.recommendation.concern_labels label --yes --p2-approved
python -m recsys.tools.build_concern_efficacy \
  --labels data/processed/review_concern_labels.jsonl \
  --catalog recsys/data/catalog/seed_catalog.json \
  --out recsys/data/signals/concern_efficacy.v1.json --data-root recsys/data

# Import only already-approved assertions; this command never approves facts.
python -m recsys.tools.import_verification \
  --source data/verification/approved-combined.json \
  --source-evidence data/verification/evidence \
  --out-root recsys/data/verification
```

Full-catalog twins go to `recsys/data/derived/` (gitignored). Build signal
stores under `derived/signals/` with `--data-root recsys/data/derived`, then run
with `--data-root recsys/data/derived`; static knowledge and verification fall
back to the committed data root.

Golden evaluation:

```bash
python -m recsys.evaluate recsys/eval/cases.json
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data architecture, pipeline
stages, and the phase plan (ingredient analysis, concern efficacy, verification
overlay, full catalog, integration).
