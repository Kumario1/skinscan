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
# Calibration writes a 50-row audit sample. Record the independently measured
# exact-set agreement only after reviewing that artifact; the full pass requires
# yield >=30%, agreement >=85%, and at least 50 audited rows in the report.
python -m src.recommendation.concern_labels probe
python -m src.recommendation.concern_labels calibrate --n 50
python -m src.recommendation.concern_labels calibrate --n 50 \
  --audit-file runs/concern/calibration_audit.json
python -m src.recommendation.concern_labels label --yes
python -m recsys.tools.build_concern_efficacy \
  --labels data/processed/review_concern_labels.jsonl \
  --catalog recsys/data/catalog/seed_catalog.json \
  --out recsys/data/signals/concern_efficacy.v1.json --data-root recsys/data

# Import only already-approved assertions; this command never approves facts.
# It refuses to drop a fact the committed overlay asserts (a re-verification
# that supersedes an assertion without re-asserting all of its facts silently
# quarantines the product). Re-assert the fact against a source that states it,
# or pass --allow-fact-loss when no source does and the loss is intended.
python -m recsys.tools.import_verification \
  --source data/verification/approved-combined.json \
  --source-evidence data/verification/evidence \
  --out-root recsys/data/verification

# Prescription/OTC drug rows, from labels the verification loop already fetched.
# Its own catalog on purpose: the signal stores are keyed by catalog_full.json's
# sha256, so merging drug rows into that file would strand every store.
python -m recsys.tools.import_drug_catalog \
  --source data/processed/catalog_drug.json \
  --out recsys/data/derived/catalog_drug.json
```

A drug label publishes no INCI list, so drug rows are the one exception to
"actives must parse out of the INCI". They earn it: the label names each active,
states its exact strength, and cites itself as the source, and the row is bound
to the label bytes by hash — checked in `catalog.py`, and enforced per active.
Anything short of that falls back to the INCI rule.

Prescriptions are **listed, never placed**. `prescription_options` surfaces the
ones that fit the reported concerns so a user can raise them with a doctor
(D-033); ranking one into a routine would assert it beats the cosmetics, and
which therapy suits which concern stays D-029 clinician-gated. They are read out
of the gated pool and then dropped from it, so no routine total can quietly
include a product that has no retail price.

When Azure is configured, the labeler requires `TARGET_URL` (or
`AZURE_OPENAI_ENDPOINT`), `AZURE_KEY` (or `AZURE_OPENAI_API_KEY`), the exact
`AZURE_OPENAI_DEPLOYMENT`, and explicit `AZURE_INPUT_PRICE_PER_MILLION` /
`AZURE_OUTPUT_PRICE_PER_MILLION` values. The full pass refuses partial Azure
configuration, writes an ignored token ledger, and enforces both the configured
$40 cumulative ceiling and 900-request ceiling. Each in-flight request reserves
its conservative maximum cost before submission, so retries and concurrency
cannot bypass those limits.

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
