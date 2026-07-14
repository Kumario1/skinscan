# Verified Catalog Resources

## Knowledge

- [Catalog schema](docs/CATALOG_SCHEMA.md)
  Canonical product, evidence, and quarantine contract. Use when deciding which
  facts an agent must collect.
- [Verification importer](src/recommendation/import_catalog.py)
  Primary implementation of overlay validation, application, quarantine, and
  completeness accounting.
- [Approval signer](src/recommendation/approve_verification.py)
  Guarded transition from proposed assertions to an attributable approved
  overlay. It signs reviewed evidence; it does not perform the research.
- [DailyMed importer](src/recommendation/import_dailymed.py)
  Reproducible regulatory-label candidate discovery for topical OTC drugs.
- [Batch 001 review](data/verification/BATCH_001_REVIEW.md)
  First worked example of the research and review standard.

## Gaps

- No single orchestrator currently performs target selection, web evidence
  capture, conflict resolution, approval, import, and stopping as one loop.
- No automated source-refresh policy currently turns aged approvals stale.
- Manufacturer-page research is not yet normalized like DailyMed ingestion.
