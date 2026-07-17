# Verification overlay

Facts that back **safety-relevant claims** get verified here; everything
mechanical (name, price, INCI, review stats) comes straight from the dump and
needs no verification.

Assertions are approved by the proven loop in
`src/recommendation/verification_loop.py` (state machine: candidate →
researching → proposed → approved → eligible | quarantined | rejected):

- `approved.json` — imported approved fact assertions keyed by `product_id`
  (initially: verified `spf` + `broad_spectrum`; later: media/editorial claims).
  Not discontinued flags: the loop rejects a discontinued SKU at research time
  and drops it, so no assertion can ever carry one.
- `evidence/<sha256>` — exact retrieved source bytes per assertion.
- Per-fact freshness windows with stale-flip; the importer never self-approves.

Products without a current approved assertion keep the honest name-parse
fallback and `spf_value_from_name_parse_not_verified` uncertainty flag.
