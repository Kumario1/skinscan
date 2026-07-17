# Research brief - verification batch 006

Rules (fail closed - see data/verification/README.md and D-032):
- Sources must be the manufacturer's own product page or a regulatory label
  (DailyMed SPL). HTTPS only. Never retailer listings, search snippets, or
  name-based inference.
- Match the exact brand, product, size, strength, and variant of the catalog
  row. Discontinued or mismatched variant => reject the product
  (`verification_loop reject --batch 006 --product <ID> --reason ...`).
- For every source: save the exact retrieved bytes to
  `data/verification/evidence/<sha256-of-bytes>` and record `source_url`,
  `retrieved_at` (UTC ISO-8601), `source_sha256`.
- Assert only facts the source explicitly states, in
  `data/verification/batches/006/proposed.json` (schema below). Facts may
  not repeat across a product's assertions.
- When done run: `python -m src.recommendation.verification_loop ingest --batch 006`

```json
{"schema_version": "2", "products": [
  {"product_id": "<ID>", "assertions": [
    {"status": "proposed", "source_url": "https://...",
      "retrieved_at": "2026-01-01T00:00:00Z", "source_sha256": "<64 hex>",
      "facts": {"routine_roles": ["cleanser"], "...": "..."}}]}]}
```

## P484080 - Mini Faded Serum for Dark Spots & Discoloration (Topicals) -> azelaic_acid_10
- Treatment path target: exactly [azelaic_acid 10%] verified via a current authoritative label (DailyMed SPL) or the manufacturer's own page (facts.drug_actives + label fields; D-033: OTC status recorded but not required).
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- drug_active_not_verified: facts.drug_actives [{name, strength, source}]
- label_source_missing: facts.label_source (authoritative label URL)
- label_verification_timestamp_missing: facts.label_verified_at

## dailymed:66858a82-cfda-4ba9-aac7-cf65499f4b1a:21922-052:adapalene-0.1%+benzoyl_peroxide-2.5% - Adapalene and Benzoyl Peroxide Gel, 0.1%/2.5% (DailyMed SPL) -> adapalene_0_1_benzoyl_peroxide_2_5
- Treatment path target: exactly [adapalene 0.1%, benzoyl_peroxide 2.5%] verified via a current authoritative label (DailyMed SPL) or the manufacturer's own page (facts.drug_actives + label fields; D-033: OTC status recorded but not required).
- routine_role_not_verified: facts.routine_roles must include the target role

## P482320 - Mini Glowscreen Sunscreen SPF 40 PA+++ with Hyaluronic Acid + Niacinamide (Supergoop!) -> sunscreen
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- broad_spectrum_not_verified: facts.broad_spectrum true per Drug Facts label
- spf_below_30_or_unknown: facts.spf (integer >= 30) per Drug Facts label
- label_source_missing: facts.label_source (authoritative label URL)
- label_verification_timestamp_missing: facts.label_verified_at
- noncomedogenic_claim_not_verified: facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it

## P399623 - Luminous Dewy Skin Mist (Tatcha) -> moisturizer
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- noncomedogenic_claim_not_verified: facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it
