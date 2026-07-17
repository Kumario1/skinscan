# Research brief - verification batch 009

Rules (fail closed - see data/verification/README.md and D-032):
- Sources must be the manufacturer's own product page or a regulatory label
  (DailyMed SPL). HTTPS only. Never retailer listings, search snippets, or
  name-based inference.
- Match the exact brand, product, size, strength, and variant of the catalog
  row. Discontinued or mismatched variant => reject the product
  (`verification_loop reject --batch 009 --product <ID> --reason ...`).
- For every source: save the exact retrieved bytes to
  `data/verification/evidence/<sha256-of-bytes>` and record `source_url`,
  `retrieved_at` (UTC ISO-8601), `source_sha256`.
- Assert only facts the source explicitly states, in
  `data/verification/batches/009/proposed.json` (schema below). Facts may
  not repeat across a product's assertions.
- When done run: `python -m src.recommendation.verification_loop ingest --batch 009`

```json
{"schema_version": "2", "products": [
  {"product_id": "<ID>", "assertions": [
    {"status": "proposed", "source_url": "https://...",
      "retrieved_at": "2026-01-01T00:00:00Z", "source_sha256": "<64 hex>",
      "facts": {"routine_roles": ["cleanser"], "...": "..."}}]}]}
```

## P454383 - PLAY Everyday Sunscreen Lotion SPF 50 PA++++ (Supergoop!) -> sunscreen
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- broad_spectrum_not_verified: facts.broad_spectrum true per Drug Facts label
- spf_below_30_or_unknown: facts.spf (integer >= 30) per Drug Facts label
- label_source_missing: facts.label_source (authoritative label URL)
- label_verification_timestamp_missing: facts.label_verified_at
- noncomedogenic_claim_not_verified: facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it

## P392235 - The Camellia Oil 2-in-1 Makeup Remover & Cleanser (Tatcha) -> cleanser
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)

## P434548 - Honeymoon Glow AHA Resurfacing Night Serum (Farmacy) -> moisturizer
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- noncomedogenic_claim_not_verified: facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it

## P456410 - Squalane + Zinc Sheer Mineral Sunscreen SPF 30 PA +++ (Biossance) -> sunscreen
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- broad_spectrum_not_verified: facts.broad_spectrum true per Drug Facts label
- spf_below_30_or_unknown: facts.spf (integer >= 30) per Drug Facts label
- label_source_missing: facts.label_source (authoritative label URL)
- label_verification_timestamp_missing: facts.label_verified_at
- noncomedogenic_claim_not_verified: facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it

## P173726 - Facial Cotton (Shiseido) -> cleanser
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)

## P392245 - Virgin Marula Luxury Face Oil (Drunk Elephant) -> moisturizer
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- noncomedogenic_claim_not_verified: facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it

## P419222 - Umbra Sheer Physical Daily Defense SPF 30 (Drunk Elephant) -> sunscreen
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- broad_spectrum_not_verified: facts.broad_spectrum true per Drug Facts label
- spf_below_30_or_unknown: facts.spf (integer >= 30) per Drug Facts label
- label_source_missing: facts.label_source (authoritative label URL)
- label_verification_timestamp_missing: facts.label_verified_at
- noncomedogenic_claim_not_verified: facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it

## P455364 - Oat Cleansing Balm (The INKEY List) -> cleanser
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
