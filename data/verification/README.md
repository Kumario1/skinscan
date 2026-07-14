# Catalog verification

Reviewer-approved schema-v2 overlays belong here. Approval records must include
authoritative source URLs, retrieval timestamps and content hashes, reviewer
identity and type (`human` or `agent`), and approval time. Proposed or stale
assertions never affect product eligibility.

After an identified human or AI agent has reviewed every source and fact in a proposed batch,
create a signed overlay without hand-editing JSON:

```bash
python -m src.recommendation.approve_verification \
  data/verification/catalog-verification-batch-001.json \
  --out data/verification/catalog-verification-batch-001-approved.json \
  --reviewer-id '<reviewer-id>' \
  --reviewer-type agent \
  --acknowledge-reviewed
```

The command validates and signs the reviewed assertions; it does not retrieve
or independently verify their sources.

Agent approval applies only to factual catalog evidence. It does not satisfy
the clinician approval gates for triage, therapy policy, or instructions in
D-029. The command refuses unsigned acknowledgement, blank reviewer identities,
unknown reviewer types, mixed status input, malformed facts, or an output where
any product lacks an approved patch. It never overwrites the proposed evidence
file.

`catalog_completeness.json` records the current release inventory gap. Regenerate
it with:

```bash
python -m src.recommendation.catalog_completeness \
  data/processed/catalog.json data/processed/catalog_tier2.json \
  --out data/verification/catalog_completeness.json
```

The command exits non-zero until every support role has 25 eligible products
and every modeled treatment path has at least one exact eligible product.

## Loop orchestrator

`python -m src.recommendation.verification_loop run` is the resumable command
that drives the whole cycle: rebuild catalogs from raw sources plus every
approved overlay, mark stale evidence, select the next batch from coverage
shortfalls, validate proposed research, and report stopping criteria. It owns
`loop_manifest.json` (per-product states: researching, proposed, approved,
eligible, quarantined, refresh_due, rejected), `batches/<N>/` (research brief,
proposed.json, REVIEW.md, approved.json), `evidence/<sha256>` (exact retrieved
bytes for every assertion), `audits/`, and `dailymed-pool.json` (new base rows
from `discover`; an overlay can only enrich a product ID that already exists in
a base catalog). Research, review, and `approve` remain reviewer actions; the
orchestrator never signs anything itself. `status` exits 0 only when coverage,
unmatched IDs, in-flight products, snapshots, freshness, and audits all pass.
See `.claude/skills/catalog-loop/SKILL.md` for the agent runbook.
