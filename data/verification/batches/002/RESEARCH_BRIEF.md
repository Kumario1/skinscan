# Research brief - verification batch 002

Rules (fail closed - see data/verification/README.md and D-032):
- Sources must be the manufacturer's own product page or a regulatory label
  (DailyMed SPL). HTTPS only. Never retailer listings, search snippets, or
  name-based inference.
- Match the exact brand, product, size, strength, and variant of the catalog
  row. Discontinued or mismatched variant => reject the product
  (`verification_loop reject --batch 002 --product <ID> --reason ...`).
- For every source: save the exact retrieved bytes to
  `data/verification/evidence/<sha256-of-bytes>` and record `source_url`,
  `retrieved_at` (UTC ISO-8601), `source_sha256`.
- Assert only facts the source explicitly states, in
  `data/verification/batches/002/proposed.json` (schema below). Facts may
  not repeat across a product's assertions.
- When done run: `python -m src.recommendation.verification_loop ingest --batch 002`

```json
{"schema_version": "2", "products": [
  {"product_id": "<ID>", "assertions": [
    {"status": "proposed", "source_url": "https://...",
      "retrieved_at": "2026-01-01T00:00:00Z", "source_sha256": "<64 hex>",
      "facts": {"routine_roles": ["cleanser"], "...": "..."}}]}]}
```

## P188306 - Acne Solutions All-Over Clearing Treatment Oil-Free (CLINIQUE) -> treatment
- Re-verification: prior evidence went stale or lacks a snapshot. Re-fetch every source and re-assert these facts fresh: amount, amount_source, cadence, cadence_source, contraindications, drug_actives, evidence_grade, evidence_roles, exposure, format, intended_areas, label_effective_date, label_source, label_verified_at, label_version, ndc_product_code, otc_drug, routine_roles, source_hash, source_set_id

## P469517 - CLEAR Oil-Free Moisturizer (Paula's Choice) -> moisturizer
- Re-verification: prior evidence went stale or lacks a snapshot. Re-fetch every source and re-assert these facts fresh: cadence, cadence_source, comedogenic_claim, evidence_grade, evidence_roles, exposure, format, intended_areas, routine_roles

## P469520 - RESIST Perfectly Balanced Foaming Cleanser (Paula's Choice) -> cleanser
- Re-verification: prior evidence went stale or lacks a snapshot. Re-fetch every source and re-assert these facts fresh: cadence, cadence_source, evidence_grade, evidence_roles, exposure, format, intended_areas, routine_roles

## P476733 - Mineral Mattescreen Sunscreen SPF 40 PA+++ (Supergoop!) -> sunscreen
- Re-verification: prior evidence went stale or lacks a snapshot. Re-fetch every source and re-assert these facts fresh: broad_spectrum, cadence, cadence_source, comedogenic_claim, evidence_grade, evidence_roles, exposure, format, intended_areas, label_source, label_verified_at, routine_roles, spf

## P427411 - Azelaic Acid 10% Suspension Brightening Cream (The Ordinary) -> azelaic_acid_10
- Treatment path target: exactly [azelaic_acid 10%] verified via a current DailyMed SPL (facts.drug_actives + otc_drug + label fields).
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- otc_status_not_verified: facts.otc_drug true, per a current DailyMed SPL
- drug_active_not_verified: facts.drug_actives [{name, strength, source}]
- label_source_missing: facts.label_source (authoritative label URL)
- label_verification_timestamp_missing: facts.label_verified_at

## P411387 - Superfood Antioxidant Cleanser (Youth To The People) -> cleanser
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)

## P427421 - Protini Polypeptide Firming Refillable Moisturizer (Drunk Elephant) -> moisturizer
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- noncomedogenic_claim_not_verified: facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it

## P454380 - Unseen Sunscreen SPF 40 PA+++ (Supergoop!) -> sunscreen
- routine_role_not_verified: facts.routine_roles must include the target role
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)
- broad_spectrum_not_verified: facts.broad_spectrum true per Drug Facts label
- spf_below_30_or_unknown: facts.spf (integer >= 30) per Drug Facts label
- label_source_missing: facts.label_source (authoritative label URL)
- label_verification_timestamp_missing: facts.label_verified_at
- noncomedogenic_claim_not_verified: facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it
