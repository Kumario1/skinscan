# Catalog verification batch 002 review

Reviewer: `claude-fable-5-catalog-loop` (agent). Research and review performed
2026-07-14. Every fact below was re-checked against the exact snapshot bytes in
`data/verification/evidence/<source_sha256>` after ingest (30 literal-quote
checks, all passing). Retrieval was curl over HTTPS; snapshots are the exact
response bytes. D-032: this approval covers factual catalog evidence only.

## Proposed products

| Role | Catalog product | Product ID | Evidence result |
|---|---|---|---|
| Treatment | Clinique Acne Solutions All-Over Clearing Treatment | `P188306` | Fresh DailyMed SPL (set `0d0cfec0`, version 3, effective 2025-08-15) verifies human OTC lotion, benzoyl peroxide 2.5%, NDC `49527-117`, "Treats Acne", thin layer one to three times daily, sensitive-skin warnings. Snapshot hash `f819499…` is byte-identical to the batch-001 recorded hash, confirming label stability. Clinique.com is bot-walled (403), so `intended_areas` could **not** be re-asserted — product will quarantine on `intended_area_not_face` until a brand-page fetch succeeds. |
| Moisturizer | Paula's Choice CLEAR Oil-Free Moisturizer | `P469517` | Current official page (product renamed "CLEAR Oil-Free Niacinamide Moisturizer", same SKU 3800 and URL) states "won't clog pores or cause breakouts" (non-comedogenic claim), day/night use per FAQ, "sheer, ceramide-infused lotion". The renamed page no longer states face use anywhere, so `intended_areas` was **not** asserted — quarantines on `intended_area_not_face`. |
| Cleanser | Paula's Choice RESIST Perfectly Balanced Foaming Cleanser | `P469520` | Official page: "Use as the first step in your AM/PM skincare routine", apply "to your face and eye area", "Rinse well". Full support facts asserted. |
| Moisturizer | Drunk Elephant Protini Polypeptide Firming Refillable Moisturizer | `P427421` | Official page (exact refillable variant, ID 999DE00000103): "Apply morning and night to a clean, dry face". Page makes **no** non-comedogenic claim, so none was asserted — quarantines on `noncomedogenic_claim_not_verified`. |
| Sunscreen | Supergoop! Mineral Mattescreen SPF 40 | `P476733` | Two non-overlapping assertions. DailyMed SPL "Mineral Mattescreen SPF 40 - Untinted" (set `08671d39`, version 5, effective 2026-06-04, marketing active): "Broad Spectrum Sunscreen SPF 40 PA +++", reapply-every-2-hours directions. Official supergoop.com page: "Non-comedogenic" bullet and "matte sunscreen for face". Note: the catalog row does not pin a tint; SPF/actives/claims are identical across Untinted and tinted SPLs, and asserted facts hold for all tints. |
| Sunscreen | Supergoop! Unseen Sunscreen SPF 40 | `P454380` | DailyMed SPL "Unseen Sunscreen Broad Spectrum SPF 40" (set `0670214e`, version 10, effective 2024-08-07) has marketing status **active** with no end date, so the catalog's SPF 40 variant is still a valid label even though supergoop.com now sells the reformulated SPF 50 (the SPF 40 page handle 404s). Only label facts asserted; no current brand page exists for this variant, so `intended_areas`/`comedogenic_claim` were **not** asserted — quarantines on both until resolved. |

## Rejected during research

- `P427411` The Ordinary Azelaic Acid 10% Suspension — the treatment role
  requires `otc_drug: true` from an authoritative label, but azelaic acid 10%
  has no legitimate drug label: DailyMed shows only Rx 15%+ products and
  grey-market unapproved listings from trading companies. The Ordinary's
  product is a cosmetic. The `azelaic_acid_10` path cannot be filled by any
  real product under current engine rules — policy decision needed.
- `P411387` Youth To The People Superfood Cleanser — youthtothepeople.com
  returns 403 on every endpoint (page, .js, .json); no manufacturer evidence
  retrievable without a real browser session.

## Checklist

- [x] Every source is the manufacturer's own page or a DailyMed SPL, HTTPS.
- [x] Exact retrieved bytes saved content-addressed; hashes recomputed at review.
- [x] Brand, product, strength, and variant checked against the catalog row
      (Mattescreen tint note and Unseen SPF 40/50 distinction documented above).
- [x] Only facts literally stated by each source were asserted; gaps left
      unasserted and their quarantine consequences documented.
- [x] No facts repeat across a product's assertions.
- [x] Rejects recorded in the manifest with reasons.
