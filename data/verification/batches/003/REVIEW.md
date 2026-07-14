# Catalog verification batch 003 review

Reviewer: `claude-fable-5-catalog-loop` (agent), 2026-07-14. Facts re-checked
against exact snapshot bytes in `data/verification/evidence/<source_sha256>`
after ingest (12 literal-quote checks passing). curl over HTTPS; snapshots are
exact response bytes. D-032: factual catalog evidence only.

## Proposed products

| Role | Catalog product | Product ID | Evidence result |
|---|---|---|---|
| Cleanser | Farmacy Green Clean Makeup Removing Cleansing Balm | `P417238` | Official page (canonical `green-clean-cleansing-balm`): "makeup remover balm and face cleanser in one", "To remove, rinse with warm water", routine block "Green Clean Cleansing Balm → Cleanser … Morning + Evening". |
| Moisturizer | Tatcha The Dewy Skin Cream | `P441101` | Official page: "Massage onto face, neck and décolletage… Use daily, morning and night", product info block "Dermatologist Tested, Non-Comedogenic, Cruelty-free". Page subtitle is now "Replenishing and Plumping Moisturizer" vs catalog "Plumping & Hydrating Moisturizer" — same product, same handle, treated as a rename. |
| Sunscreen | Supergoop! Glowscreen SPF 40 | `P456218` | Two non-overlapping assertions. DailyMed SPL "Glow Screen SPF 40" (set `21569637`, version 8, effective 2026-01-09, marketing active): "Broad Spectrum Sunscreen / SPF 40 PA+++", reapply-every-2-hours directions. Official page (canonical `glowscreen-spf-40`, product record `YGroup_glowscreen40`): manufacturer attributes "Body Part:Face" and "Preferences:Non-comedogenic". Shade variants (Sunrise etc.) share SPF/actives; base SPL used, same convention as batch 002's Mattescreen. |

## Rejected during research

- `P7880` fresh Soy Hydrating Gentle Face Cleanser — fresh.com is an Estée
  Lauder property behind the same WAF as clinique.com (403 on all fetches); no
  manufacturer evidence retrievable without a real browser.
- Carried from batch 002 (re-selected pre-fix, re-rejected): `P427411`
  (azelaic 10% policy gap), `P411387` (YTTP bot wall), `P427421` (page makes
  no non-comedogenic claim), `P454380` (SPF 40 variant retired at
  manufacturer; only SPF 50 sold). `select` now excludes rejected products.

## Checklist

- [x] Sources are manufacturer pages or DailyMed SPLs, HTTPS only.
- [x] Exact bytes snapshotted content-addressed; hashes recomputed at review.
- [x] Brand/product/strength/variant checked (Tatcha rename and Glowscreen
      shade note documented above).
- [x] Only facts literally stated were asserted; no cross-assertion repeats.
- [x] Rejects recorded with reasons.
