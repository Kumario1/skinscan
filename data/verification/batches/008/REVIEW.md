# Catalog verification batch 008

Status: **approved by agent `claude-fable-5` at 2026-07-16**

Two new moisturizers via Firecrawl manufacturer pages, plus DailyMed SPL
follow-up assertions for the two Supergoop sunscreens whose batch-005
manufacturer pages could not state the SPF drug facts. Every snapshot
re-opened; hashes recomputed from the stored bytes and matched.

## Proposed products

| Role | Catalog product | Product ID | Evidence result |
|---|---|---|---|
| moisturizer | Barrier+ Triple Lipid-Peptide Face Cream (Skinfix), 1.7 oz | `P442840` | Page states twice-daily AM/PM use, non-comedogenic. |
| moisturizer | Ultra Facial Moisturizing Cream with Squalane (Kiehl's), 1.7 oz | `P421996` | Page states twice-daily face use, non-comedogenic. |
| sunscreen | Glow Stick Sunscreen SPF 50 (Supergoop!) | `P429953` | Current DailyMed SPL states Broad Spectrum SPF 50, OTC, marketing active. |
| sunscreen | (Re)setting Mineral Powder SPF 35 (Supergoop!) | `P467976` | Current DailyMed SPL (Translucent) states Broad Spectrum SPF 35, OTC, marketing active. |

## Evidence re-check

### P442840 — `fd4e913851a3743ff1a34e913ec9a1dd41c112b3d9d5455172caa8ffda11132c` (1,519,735 bytes)
`https://skinfix.com/products/lipid-peptide-cream`. H1 "Triple Lipid-Peptide
Cream" — Skinfix's 2025 rebrand dropped the "Barrier+" prefix; same product,
and the catalog's 1.7 oz variant is offered (Full 1.7 oz plus refill/mini/XL).
"Day + Night — Apply 1-2 pumps to clean, dry skin after serum and eye cream /
Use every day, twice a day—in the morning and at night" → am_pm cadence,
daily_support. "Triple Lipid-Peptide Cream was tested to be non-comedogenic"
→ claimed_noncomedogenic. The rebranded page names no application area, so
intended_areas is deliberately absent.

### P421996 — `8a145512021bf98b40ccb28b43508ca287ed1c00c30d2e96b3889937af560344` (1,527,767 bytes)
`https://www.kiehls.com/skincare/face-moisturizers/ultra-facial-cream-with-squalane/622.html`.
H1 "Ultra Facial Cream with Squalane"; 1.7 fl oz / 50 ml variant offered.
"apply a dime-sized amount of our hydrating facial cream to skin... Use twice
daily, both day and night" → am_pm, face, daily_support. "Fragrance-free,
non-comedogenic, and dermatologist-tested" → claimed_noncomedogenic.

### P429953 — `3326fa27972f33906b0d77c0feb8baca16b92156cfa191ca2fdd38fa2d3fd189` (22,852 bytes)
DailyMed SPL 61e543e3-9ace-6440-e053-2991aa0a9647, title "Glow Stick SPF 50"
[SUPERGOOP, LLC], version 13, effective 20260109, marketing status ACTIVE
(not discontinued). Document title and label state Broad Spectrum SPF 50;
OTC monograph sunscreen. Facts asserted are exactly the SPF-label set the
batch-005 manufacturer page could not provide; no fact key overlaps the
batch-005 assertion (enforced by ingest).

### P467976 — `3aed13366b285a59bd1f27227d779473002f093873deb77ff2a8be1d72caddc3` (20,640 bytes)
DailyMed SPL af5c9801-26ce-5c60-e053-2995a90ab457, title "(Re) Setting 100%
Mineral Powder Translucent Broad Spectrum Suncreen SPF 35" [SUPERGOOP, LLC],
version 6, effective 20260109, marketing ACTIVE. The catalog row carries no
shade; DailyMed lists four shade SPLs (Translucent/Light/Medium/Deep) all
stating identical Broad Spectrum SPF 35 zinc-oxide facts — the Translucent
(default shade) label is cited.

## Rejected during research

- **P248404 — Pure Skin Face Cleanser (First Aid Beauty). Substantive reject.**
  The manufacturer URL now serves the renamed, reformulated "Ultra Gentle
  Cream-to-Foam Face Cleanser" in 2/6/10 oz; the original product and its
  5 oz variant are gone.
- **P481989 — Watermelon Glow Niacinamide Sunscreen SPF 50 (Glow Recipe).
  Substantive reject.** Product URL 302-redirects to the homepage (Shopify
  deleted-product behaviour), product .json 404s, and the current Glow Recipe
  SPF line no longer includes it — discontinued.
- Carried from earlier this batch: `P126301` (no usage frequency stated on
  the Clinique page) and `P394639` (no non-comedogenic claim on lgbeauty.com);
  their batch-005 facts stand.

## Checklist

- [x] Sources are manufacturer pages or DailyMed SPLs over HTTPS
- [x] Brand/product/size variant match confirmed (or rejected); Supergoop shade
      caveat documented above
- [x] Evidence bytes snapshotted under their sha256; hashes re-verified
- [x] Every asserted fact restates something the source explicitly states
- [x] No fact repeats a product's earlier-batch assertions
- [x] Rejected products removed from proposed.json
