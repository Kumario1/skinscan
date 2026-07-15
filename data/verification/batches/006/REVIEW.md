# Catalog verification batch 006

Status: **approved by agent `claude-fable-5` at 2026-07-15**

Researched and prepared on 2026-07-15. One product is proposed; the other three
selected into this batch remain outstanding (see below). The production overlay
retains `status: "proposed"` as the immutable research input; the separately
signed approved overlay grants catalog eligibility.

## Proposed products

| Role | Catalog product | Product ID | Evidence result |
|---|---|---|---|
| Treatment (`adapalene_0_1_benzoyl_peroxide_2_5`) | Adapalene and Benzoyl Peroxide Gel USP, 0.1%/2.5% (Encube Ethicals) | `dailymed:66858a82-cfda-4ba9-aac7-cf65499f4b1a:21922-052:adapalene-0.1%+benzoyl_peroxide-2.5%` | Current DailyMed SPL states the treatment indication verbatim. |

### Evidence re-check

Source: `https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/66858a82-cfda-4ba9-aac7-cf65499f4b1a.xml`
Snapshot: `data/verification/evidence/9c48a920fc2552528ca348351860cf97af57aa0d078fa69236cbfab834ce8271`
(94,075 bytes; hash recomputed from the stored bytes and matched before approval).

Re-opened the snapshot and checked the single asserted fact against it:

- `routine_roles: ["treatment"]` — INDICATIONS & USAGE states: "Adapalene and
  benzoyl peroxide gel USP, 0.1% / 2.5% is indicated for the topical treatment
  of acne vulgaris in patients 9 years of age and older." The label states the
  treatment role directly; nothing is inferred from the product name.

Variant match confirmed against the catalog row, which is derived from this same
SPL: document type `34391-3 HUMAN PRESCRIPTION DRUG LABEL`, route TOPICAL, form
gel, NDC `21922-052`, and structured actives adapalene 0.1% + benzoyl peroxide
2.5% — the exact strengths the `adapalene_0_1_benzoyl_peroxide_2_5` path
requires. Per D-033 the path does not require OTC status; `otc_drug: false` is
recorded on the row, so presentation routes it to a clinician.

No other fact was asserted. The row already carries drug_actives, label_source,
label_verified_at, cadence, exposure and intended_areas from the label itself,
so the brief listed `routine_role_not_verified` as its only outstanding reason.

## Rejected during research

**`P484080` — Topicals Faded Mini → `azelaic_acid_10`. Substantive reject.**
The brand's own page was retrieved cleanly (HTTP 200, canonical
`https://mytopicals.com/products/faded-mini-original-copy`, 461,395 bytes,
evidence `853c3ca4c434c83a54f58d6ad17740c6a17a03161b84293a5ff07297e36f5ff8`,
hash recomputed from the stored bytes and matched). Variant is the Mini (H1
"Faded Mini", 15ml). Azelaic acid appears exactly once, as a benefit line —
"Azelaic Acid — Helps prevent the appearance of hyperpigmentation and smooth
skin texture" — with **no percentage attached**. The only percentages anywhere
on the page are "100% odorless formula" and user-survey figures (92.31%,
94.87%). The path requires exactly azelaic acid 10%, so this product cannot
fill it. This is the general case, not an accident of one SKU: cosmetics do not
declare per-active strengths, which is precisely why D-033 records that azelaic
acid 10% exists only as cosmetics — and why the path stays unfillable by them.

**`P482320` (Supergoop! Mini Glowscreen → sunscreen) and `P399623` (Tatcha
Luminous Dewy Skin Mist → moisturizer). Not verifiable today.**
Both brands answer automated HTTPS requests with `HTTP 429 local_rate_limited`
(bot management), including with a browser-like User-Agent and after honouring
`Retry-After`. Evidence for this catalog must be hash-verifiable bytes from the
manufacturer's own page, and working around a site's bot protection is not an
acceptable way to obtain them — so no snapshot exists and no fact is asserted.
Rejected rather than left in flight; requeue when a human can fetch the page.
Two rendered-DOM transcripts and two JSON blobs gathered that way during
research were discarded before review: a DOM transcript is a derived artifact,
not the server's bytes, so it cannot be re-verified against the source at all.

Note for whoever requeues `P399623`: Tatcha's own copy calls the product a
"hydrator", not a moisturizer — worth resolving against the catalog's role
vocabulary before asserting `routine_roles`.

## Checklist

- [x] Source is a regulatory label (DailyMed SPL) over HTTPS.
- [x] Exact bytes stored under `evidence/<sha256>`; hash recomputed and matched.
- [x] Every asserted fact quoted from the source; nothing inferred.
- [x] Variant (brand, strength, form, NDC) matches the catalog row.
- [x] No fact repeats across the product's assertions.
- [x] Agent approval covers factual catalog evidence only (D-032). It does not
      satisfy D-029's clinician gates: this fills a *catalog* path, and the
      therapy policy that decides whether the path is offered stays gated.
