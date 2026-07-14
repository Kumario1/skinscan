# Verification overlay (Phase 3 — scaffold only)

Facts that back **safety-relevant claims** get verified here; everything
mechanical (name, price, INCI, review stats) comes straight from the dump and
needs no verification.

What will live here, ported from the proven loop in
`src/recommendation/verification_loop.py` (state machine: candidate →
researching → proposed → approved → eligible | quarantined | rejected):

- `approved.json` — approved fact assertions keyed by `product_id`
  (initially: verified `spf` + `broad_spectrum`; discontinued/reformulated
  flags; later: media/editorial claims).
- `evidence/<sha256>` — exact retrieved source bytes per assertion.
- Per-fact freshness windows with stale-flip; the loop never self-approves.

Until Phase 3, SPF values are parsed from product names and every output marks
them `spf_source: "name_parse"` / `spf_value_from_name_parse_not_verified`.
