# Catalog verification batch 007 (serum role)

Status: **approved by agent `claude-fable-5` at 2026-07-16**

Manually queued batch (see RESEARCH_BRIEF.md): recsys's serum slot needs
overlay-verified usage facts on cosmetic serums. Cosmetic facts only — no
drug_actives or treatment-path claims are asserted anywhere in this batch.

## Proposed products

| Role | Catalog product | Product ID | Evidence result |
|---|---|---|---|
| serum | Niacinamide 10% + Zinc 1% Oil Control Serum (The Ordinary), 30 mL | `P427417` | Manufacturer page states face application, morning and evening. |
| serum | Azelaic Acid 10% Suspension Brightening Cream (The Ordinary), 30 mL | `P427411` | Manufacturer page states face application, morning and evening; product is a suspension. |
| serum | Alpha Arbutin 2% + HA Hyperpigmentation Serum (The Ordinary), 30 mL | `P427412` | Manufacturer page states face application, morning and evening. |
| serum | Salicylic Acid 2% Anhydrous Solution Pore Clearing Serum (The Ordinary), 30 mL | `P479732` | Manufacturer page states face application, morning and evening, and a sensitive-skin exclusion. |

## Evidence re-check

Every snapshot was re-opened; hashes recomputed from the stored bytes and
matched (also enforced by `ingest`). All sources are the manufacturer's own
HTTPS product pages (theordinary.com), each matching the exact brand, product,
strength, and the catalog row's 30 mL variant.

### P427417 — evidence `09cd2186e866dc53977ffd6bdf8527f158345b77d706dd5eda799a243d7e1773` (407,237 bytes)
Page title: "Niacinamide 10% + Zinc 1% Oil Control Serum | The Ordinary".
How to Use: "Apply a few drops to the face in the morning and evening."
- `routine_roles: ["serum"]`, `format: "serum"` — the product's own name and
  page classify it as a serum.
- `exposure: "leave_on"` — directions apply with no rinse-off step.
- `cadence: "am_pm"` + `cadence_source` — "in the morning and evening".
- `intended_areas: ["face"]` — "to the face".

### P427411 — evidence `66007051596787473b988e64f50473c7a36d58f2ce06fed57c94621a908e0890` (392,879 bytes)
Page title: "Azelaic Acid Suspension 10% for Exfoliation & Even Skin Tone".
How to Use: "Apply a small amount to the face in the morning and evening.
Avoid the eye contour and contact with eyes and mouth."
- `format: "suspension"` — the product's official name states it.
- Other facts as above (leave_on, am_pm, face).
- Note: this product was rejected in batch 003 **as an `azelaic_acid_10`
  drug-path candidate** (no DailyMed SPL). That rejection stands; nothing here
  asserts drug facts. This batch requeues it only for the cosmetic serum role.

### P427412 — evidence `3d25cf4c621762f86429cff20753e18512246909d8368e9ffcb93aad5553a43d` (406,788 bytes)
Page title: "Alpha Arbutin 2% + HA Brightening Serum for Uneven Skin".
How to Use: "Apply a few drops to the face in the morning and evening. Use sun
protection during the day."
- Facts as above (serum, leave_on, am_pm, face).

### P479732 — evidence `84cec47c22d3694ec987de57002d1c70798d56c6bd3008e1eb3745b9b0178340` (395,416 bytes)
Page title: "The Ordinary | Salicylic Acid 2% Anhydrous Solution".
How to Use: "Apply in the morning and evening. Apply a small dot to the target
area or apply a small amount over the face for general use. Do not use on
sensitive, peeling or compromised skin."
- `format: "solution"` — the product's official name states it.
- `contraindications: ["sensitive"]` — "Do not use on sensitive ... skin".
- Other facts as above (leave_on, am_pm, face).

## Rejected during research

**`P427419` — Hyaluronic Acid 2% + B5 Hydrating Serum → serum. Variant reject.**
The manufacturer page for the original formulation (retrieved cleanly,
sha256 `e35638ee2f50184ae8adc683e249af3975059a1a62dbb0f13dc00c056c11bbe5`, not
retained in evidence/) offers 30 mL, 100 mL, and 240 mL only; the catalog row
is the 2 oz / 60 mL variant. Exact size match is required, so the product was
rejected and removed from proposed.json.

## Checklist

- [x] Every source is the manufacturer's own page over HTTPS
- [x] Brand, product, strength, and size variant match the catalog row (or rejected)
- [x] Evidence bytes snapshotted under their sha256; hashes re-verified
- [x] Every asserted fact restates something the page explicitly says
- [x] No drug_actives / treatment-path claims in a cosmetic batch
- [x] Rejected products removed from proposed.json
