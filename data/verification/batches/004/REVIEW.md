# Catalog verification batch 004 review

Reviewer: `claude-fable-5-catalog-loop` (agent), 2026-07-14. Facts re-checked
against exact snapshot bytes in `data/verification/evidence/<source_sha256>`
after ingest (18 literal-quote checks passing; the P456392 check failing is
what exposed its discontinuation). curl over HTTPS; snapshots are exact
response bytes. D-032: factual catalog evidence only.

## Proposed products

| Role | Catalog product | Product ID | Evidence result |
|---|---|---|---|
| Moisturizer | Tatcha The Water Cream | `P418218` | Official page (canonical `the-water-cream-lightweight-moisturizer`): "Massage onto face, neck and décolletage… Use daily, morning and night"; product info block "Dermatologist Tested, Non-Comedogenic, Cruelty-free". Page subtitle now "Lightweight Pore-Refining" vs catalog "Oil-Free Pore Minimizing" — same product/handle, rename. |
| Cleanser | Tatcha The Deep Cleanse Gentle Exfoliating Cleanser | `P427536` | Official page, exact title match: "Massage with wet hands onto wet face. Rinse thoroughly", FAQ "Use daily". Cadence asserted as `daily` (the page states daily use without specifying AM/PM). |
| Cleanser | Tatcha The Rice Wash | `P461537` | Official page: FAQ "designed for daily use, morning and night"; "massage onto face. Rinse." One PDP covers all sizes (50/120/240 ml); the page `<title>` says "(Travel Size)" but the full-size 120 ml variant is on the page. Subtitle now "Creamy Rice Powder Cleanser" vs catalog "Skin-Softening Cleanser" — rename. |
| Sunscreen | Shiseido Clear Sunscreen Stick SPF 50+ | `P429242` | DailyMed SPL exact title match (set `15a786f2`, version 2, effective 2024-12-23, marketing **active**): "BROAD SPECTRUM / SPF 50+ / For Face/Body / WATER RESISTANT (80 MINUTES)", directions "apply liberally 15 minutes before sun exposure". `intended_areas: face` comes from the label itself. shiseido.com no longer sells this product (410; successor line is "Ultimate Sun Protector Clear Stick SPF 60"), so no page-level non-comedogenic claim exists — quarantines on `noncomedogenic_claim_not_verified` unless label standards change. SPF asserted as 50 (label floor for "50+"). |

## Rejected during research

- `P456392` innisfree Daily UV Defense SPF 36 — every marketing act in the
  current SPL is `completed` (last lot expiries 2026–2027) and the site
  redirects the SPF 36 handle to an SPF 50 successor. Discontinued variant.
  Caught by the snapshot re-check (the "active" assertion failed).
- `P470259` Topicals Faded Serum — cosmetic azelaic serum; same
  `azelaic_acid_10` policy gap as P427411.
- `P409800` SEPHORA COLLECTION Cleansing & Exfoliating Wipes — house brand;
  sephora.com is the manufacturer page and is bot-walled (403).
- `P248407` First Aid Beauty Ultra Repair Cream — page verified (face+body
  cream, "Apply from head to toe once or twice daily") but makes no
  non-comedogenic claim; cannot clear the moisturizer role.

## Checklist

- [x] Sources are manufacturer pages or DailyMed SPLs, HTTPS only.
- [x] Exact bytes snapshotted content-addressed; hashes recomputed at review.
- [x] Brand/product/strength/variant checked (two Tatcha renames, Rice Wash
      title quirk, Shiseido successor-line distinction documented).
- [x] Only facts literally stated were asserted; no cross-assertion repeats.
- [x] Rejects recorded with reasons, including one discontinued variant.
