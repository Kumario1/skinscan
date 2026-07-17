# Catalog verification batch 009

Status: **approved by agent `claude-fable-5` at 2026-07-16**

Firecrawl manufacturer pages plus one DailyMed SPL. Every snapshot re-opened;
hashes recomputed from the stored bytes and matched (also enforced by ingest).

## Proposed products

| Role | Catalog product | Product ID | Evidence result |
|---|---|---|---|
| sunscreen | PLAY Everyday Sunscreen Lotion SPF 50 (Supergoop!), 2.4 oz | `P454383` | Page states daily face+body use, non-comedogenic; current DailyMed SPL states Broad Spectrum SPF 50, OTC, marketing active. |
| cleanser | The Camellia Oil 2-in-1 Makeup Remover & Cleanser (Tatcha), 150 mL | `P392235` | Page states rinse-off oil cleanser, non-comedogenic; **no usage frequency stated → no cadence asserted** (stays quarantined on cadence). |
| cleanser | Oat Cleansing Balm (The INKEY List), 150 mL | `P455364` | Page states AM/PM rinse-off use. |

## Evidence re-check

### P454383 — page `308c9d78ececc30941c9e30009fb316e5e4a631276c8752f432468820bcd6574` (3,457,632 bytes) + SPL `18017378d37e1eacc196be655be99c2f7e0a223d6aa2f6f78b152a745a81c449` (60,823 bytes)
`https://supergoop.com/products/everyday-sunscreen`, H1 "PLAY Everyday Lotion
SPF 50" (catalog name variant "PLAY Everyday Sunscreen Lotion SPF 50 PA++++";
same product — PA rating is the Asian-market label). Sizes offered include the
catalog's 2.4 fl oz. "Apply PLAY Everyday Lotion generously and evenly across
your face and body... Remember to reapply at least every 2 hours" → per_label
cadence, face+body areas. "Non-comedogenic so it won't clog pores." Second
assertion cites DailyMed SPL 0428eebc (title "Play Everyday Lotion SPF 50",
SUPERGOOP), version 9, effective 20260604, marketing ACTIVE, Broad Spectrum
SPF 50 — the drug-label facts the page cannot carry. The two assertions share
no fact keys (enforced by ingest).

### P392235 — `27dcf2d0e5516e77efa8733fba929e804511fe6d95aed5d3040d2cb90c441664` (1,986,327 bytes)
`https://tatcha.com/products/the-camellia-cleansing-oil-and-makeup-remover`.
Sold today as "The Camellia Cleansing Oil" — Tatcha shortened the name; same
2-in-1 oil cleanser/makeup remover. Site labels the 150 mL bottle 5.0 fl oz
vs the catalog's 5.1 oz — same 150 mL variant, label rounding. "Massage into
skin in circular motions... Rinse with warm water" → rinse_off oil.
"Non-comedogenic" stated. No application frequency stated anywhere → cadence
and cadence_source deliberately absent; the row remains quarantined on
instruction_cadence (same treatment as P126301 in batch 005).

### P455364 — `28e4d69f5550ca82daf4f1f4a30d61278518b9f457a34c002e43fd0240f69c1b` (2,300,314 bytes)
`https://www.theinkeylist.com/products/oat-cleansing-balm-150ml`. Catalog's
150 mL variant offered. "Use AM and PM. Massage a raspberry-sized amount onto
dry skin. Add warm water to emulsify, then rinse" → am_pm, rinse_off, daily
support. The page names no application area → intended_areas absent (absence
stays open, never favorable). No non-comedogenic claim → not asserted.

## Rejected during research

- **P173726 — Facial Cotton (Shiseido).** Cotton pads, an accessory, not a
  cleanser — category mismatch, can never carry cleanser usage facts.
- **P434548 — Honeymoon Glow AHA Resurfacing Night Serum (Farmacy).**
  farmacybeauty.com now sells "Honey Glow 17% AHA + BHA Resurfacing Acid
  Serum"; renamed AND reformulated — the catalog formulation no longer exists.
- **P456410 — Squalane + Zinc Sheer Mineral Sunscreen SPF 30 (Biossance).**
  Manufacturer URL redirects to the replacement Squalane + Daily Mineral
  SPF 50; the SPF 30 product is discontinued.
- **P392245 — Virgin Marula Luxury Face Oil (Drunk Elephant).** Face oil with
  no non-comedogenic claim — cannot clear moisturizer quarantine (same
  standard as P427415, P248407).
- **P419222 — Umbra Sheer Physical Daily Defense SPF 30 (Drunk Elephant).**
  US manufacturer page returns HTTP 410 Gone; absent from the US
  sun-protection collection — discontinued.

## Checklist

- [x] Sources are manufacturer pages or DailyMed SPLs over HTTPS
- [x] Brand/product/size variant match confirmed (or rejected); name-change
      lineages (Tatcha shortening, Supergoop PA-rating suffix) documented
- [x] Evidence bytes snapshotted under their sha256; hashes re-verified
- [x] Every asserted fact restates something the source explicitly states;
      unstated facts deliberately omitted
- [x] No fact repeats across a product's assertions
- [x] Rejected products removed from proposed.json
