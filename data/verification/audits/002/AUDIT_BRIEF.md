# Random audit 002

Re-fetch each source and confirm every fact is still explicitly supported. Record with:
`python -m src.recommendation.verification_loop audit --record pass|fail --notes ...`

## P456218
- source: https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/21569637-332a-461b-bf6c-33af6a83a155.xml
- retrieved_at: 2026-07-14T18:21:17.942408Z
- sha256: ec588ae92b30d15eff5ca69d48730131defd6daa6336341ac6f4e43fe8f72093
- facts: `{"broad_spectrum": true, "cadence": "per_label", "cadence_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/21569637-332a-461b-bf6c-33af6a83a155.xml", "evidence_grade": "regulatory_label", "exposure": "leave_on", "format": "cream", "label_effective_date": "20260109", "label_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/21569637-332a-461b-bf6c-33af6a83a155.xml", "label_verified_at": "2026-07-14", "label_version": "8", "otc_drug": true, "source_set_id": "21569637-332a-461b-bf6c-33af6a83a155", "spf": 40}`

## P469520
- source: https://www.paulaschoice.com/resist-perfectly-balanced-foaming-cleanser/783.html
- retrieved_at: 2026-07-14T18:13:41.489734Z
- sha256: e794c409bc51ab5dd020632398cd4c9599794bdb770eade112fe5ac9110a95e2
- facts: `{"cadence": "am_pm", "cadence_source": "https://www.paulaschoice.com/resist-perfectly-balanced-foaming-cleanser/783.html", "evidence_grade": "manufacturer_product_page", "evidence_roles": ["daily_support"], "exposure": "rinse_off", "format": "cleanser", "intended_areas": ["face"], "routine_roles": ["cleanser"]}`

## P479732
- source: https://theordinary.com/en-us/salicylic-acid-2-anhydrous-solution-exfoliator-100442.html
- retrieved_at: 2026-07-16T17:59:18Z
- sha256: 84cec47c22d3694ec987de57002d1c70798d56c6bd3008e1eb3745b9b0178340
- facts: `{"cadence": "am_pm", "cadence_source": "https://theordinary.com/en-us/salicylic-acid-2-anhydrous-solution-exfoliator-100442.html", "contraindications": ["sensitive"], "exposure": "leave_on", "format": "solution", "intended_areas": ["face"], "routine_roles": ["serum"]}`

## P441101
- source: https://tatcha.com/products/the-dewy-skin-cream
- retrieved_at: 2026-07-14T18:21:17.942408Z
- sha256: 965b7cce539295e0c4916690a543da1e65a36fef07f920b8750b2b646b66f4b1
- facts: `{"cadence": "am_pm", "cadence_source": "https://tatcha.com/products/the-dewy-skin-cream", "comedogenic_claim": "claimed_noncomedogenic", "evidence_grade": "manufacturer_product_page", "evidence_roles": ["daily_support"], "exposure": "leave_on", "format": "cream", "intended_areas": ["face"], "routine_roles": ["moisturizer"]}`

## P461537
- source: https://tatcha.com/products/rice-wash-soft-cream-cleanser
- retrieved_at: 2026-07-14T18:27:51.301104Z
- sha256: 615436bd1e5352eb4f544e71e6cfd9232f12e9f25185077bdcb9404d2cca9784
- facts: `{"cadence": "am_pm", "cadence_source": "https://tatcha.com/products/rice-wash-soft-cream-cleanser", "evidence_grade": "manufacturer_product_page", "evidence_roles": ["daily_support"], "exposure": "rinse_off", "format": "cleanser", "intended_areas": ["face"], "routine_roles": ["cleanser"]}`
