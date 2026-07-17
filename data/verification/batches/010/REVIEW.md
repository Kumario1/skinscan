# Batch 010 review

Reviewer: claude-agent (catalog loop session 2026-07-16). Research fanned out to
six brand-grouped subagents; every snapshot re-opened and every fact re-checked
against the stored bytes before approval (hash + content spot-checks below).

## Evidence results (10 proposed)

| Product | Source | Snapshot | Facts re-checked |
|---|---|---|---|
| P426836 Beste No. 9 Jelly Cleanser (Drunk Elephant) | drunkelephant.com PDP | e99470c5… hash ✓ | "Use nightly" present → pm_daily; cleanser role ✓ |
| P442566 Slaai Makeup-Melting Butter Cleanser (Drunk Elephant) | drunkelephant.com PDP | 8d93fbe7… hash ✓ | "In the evening" present → pm_daily; cleanser ✓ |
| P400259 C-Firma Fresh Vitamin-C Day Serum (Drunk Elephant) | drunkelephant.com PDP | 9355e6eb… hash ✓ | "In the morning" present → am_daily; serum ✓; 28 mL variant ✓ |
| P392246 T.L.C. Framboos Glycolic Night Serum (Drunk Elephant) | drunkelephant.com PDP | 7554ee30… hash ✓ | "In the evening" present → pm_daily; serum ✓; 30 mL ✓ |
| P297516 Checks and Balances Frothy Face Wash (Origins) | origins.com PDP (Firecrawl through bot wall) | 3f4824ff… hash ✓ | "gentle enough for daily use" present → daily; cleanser ✓; 5 oz variant ✓ |
| P460516 Papaya Sorbet Enzyme Cleansing Balm (Glow Recipe) | glowrecipe.com PDP | f07bb5d3… hash ✓ | FAQ "safe for everyday (and 2x a day) use? Yes" → am_pm; cleanser ✓; 100 mL ✓ |
| P444718 Squalane Cleanser (The Ordinary) | theordinary.com PDP (current slug, 50 ml in stock) | 66b9b647… hash ✓ | "ideal for daily use" + "non-comedogenic and soap-free" → daily + claimed_noncomedogenic |
| P427420 Multi-Peptide + HA Serum (The Ordinary) | theordinary.com PDP, 60 ml variant selected & InStock | 10b02c52… hash ✓ | "morning and evening" → am_pm; serum ✓; 60 mL variant confirmed sold |
| P427413 Lactic Acid 10% + HA 2% (The Ordinary) | theordinary.com PDP | b04dcb56… hash ✓ | "once daily, ideally in the evening" → pm_daily; serum ✓; 30 mL ✓ |
| P481169 The Silk Sunscreen SPF 50 (Tatcha) | tatcha.com PDP + DailyMed SPL 7799042a… | 2a8616e3… / 3a1ead63… hashes ✓ | Page: "Non-comedogenic", reapply every 2 hours, 50 mL ✓. SPL: "Broad Spectrum", "SPF 50", all 5 `<statusCode code="active">` (not discontinued) |

All 12 snapshots exist under `data/verification/evidence/`, sha256 re-computed
and matching the assertions; every asserted string located in the stored bytes.
The Ordinary pages were reached via current slugs after confirming titles (old
slugs 301 to wrong products — checked per prior-batch warning).

## Rejects (14, recorded via `verification_loop reject`)

- P270594 Bobbi Brown Vitamin Enriched Face Base — page retrieved, variant
  matched, but zero non-comedogenic claim (only a negative review mentions clogging).
- P432668 D-Bronzi Bronzing Drops — renamed product; page states no usage frequency.
- P419221 Umbra Tinte Physical Daily Defense — gone from drunkelephant.com
  (sitemap + search); replaced by Umbra Tinte Mineral Cream SPF 30.
- P466123 Watermelon Glow Niacinamide Dew Drops — page states no usage frequency.
- P428819 Watermelon Pink Juice Oil-Free Moisturizer — 60 mL variant no longer
  sold (current product: 50/25 mL only).
- P479327 Plum Plump Hyaluronic Moisturizer — no non-comedogenic claim (only
  cross-sell blurbs for other products).
- P393718 Sunday Riley Luna Night Oil — no non-comedogenic claim.
- P456398 Shiseido Ultimate Sun Protector SPF 50+ — discontinued at
  manufacturer (404s; category lists only SPF 60+); SPL still active but no PDP.
- P416563 Biossance Squalane + Vitamin C Rose Firming Oil — name/variant
  mismatch with current "Rose Oil"; no non-comedogenic claim.
- P394624 belif True Cream Moisturizing Bomb — live page (lgbeauty.com) but no
  non-comedogenic claim.
- P416815 OLEHENRIKSEN Find Your Balance Cleanser — 404, absent from products
  sitemap (discontinued).
- P469522 Paula's Choice RESIST SPF 50 — no DailyMed SPL exists; noncomedogenic
  claim absent.
- P466154 Supergoop! Daily Dose Vitamin C + SPF 40 — removed from supergoop.com
  (discontinued).
- P429516 COOLA Sun Silk Drops SPF 30 — renamed/reformulated (different
  actives per old vs new SPLs); exact variant no longer sold.

## Checklist

- [x] Every proposed assertion re-checked against its stored snapshot bytes
- [x] sha256 of every snapshot re-computed and matches `source_sha256`
- [x] Exact brand / product / size variant confirmed for each verified row
- [x] Cadence asserted only where the source states frequency; source URL recorded
- [x] `claimed_noncomedogenic` only where the page states it
- [x] Sunscreen (P481169) has a DailyMed SPL assertion: broad spectrum, SPF ≥ 30,
      label_source, label_verified_at; marketing status active
- [x] All rejects recorded with reasons; removed from proposed.json
- [x] No facts repeated across a product's assertions
