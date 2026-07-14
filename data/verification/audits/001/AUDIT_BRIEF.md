# Random audit 001

Re-fetch each source and confirm every fact is still explicitly supported. Record with:
`python -m src.recommendation.verification_loop audit --record pass|fail --notes ...`

## P461537
- source: https://tatcha.com/products/rice-wash-soft-cream-cleanser
- retrieved_at: 2026-07-14T18:27:51.301104Z
- sha256: 615436bd1e5352eb4f544e71e6cfd9232f12e9f25185077bdcb9404d2cca9784
- facts: `{"cadence": "am_pm", "cadence_source": "https://tatcha.com/products/rice-wash-soft-cream-cleanser", "evidence_grade": "manufacturer_product_page", "evidence_roles": ["daily_support"], "exposure": "rinse_off", "format": "cleanser", "intended_areas": ["face"], "routine_roles": ["cleanser"]}`

## P417238
- source: https://www.farmacybeauty.com/products/green-clean-cleansing-balm
- retrieved_at: 2026-07-14T18:21:17.942408Z
- sha256: 1da3b1a06a95b394dd5745c609c40c583f104ccdd8f669aa1cbe6953274f40aa
- facts: `{"cadence": "am_pm", "cadence_source": "https://www.farmacybeauty.com/products/green-clean-cleansing-balm", "evidence_grade": "manufacturer_product_page", "evidence_roles": ["daily_support"], "exposure": "rinse_off", "format": "cleanser", "intended_areas": ["face"], "routine_roles": ["cleanser"]}`

## P188306
- source: https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml
- retrieved_at: 2026-07-14T18:13:41.489734Z
- sha256: f81949978b0534ebcd1c56e3a53d1fa5e7b2536aea7bfd9fbcff21e4266e28da
- facts: `{"amount": "thin_layer", "amount_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml", "cadence": "per_label", "cadence_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml", "contraindications": ["sensitive"], "drug_actives": [{"name": "benzoyl_peroxide", "source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml", "strength": "2.5%"}], "evidence_grade": "regulatory_label", "evidence_roles": ["acne_treatment"], "exposure": "leave_on", "format": "lotion", "label_effective_date": "20250815", "label_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml", "label_verified_at": "2026-07-14", "label_version": "3", "ndc_product_code": "49527-117", "otc_drug": true, "routine_roles": ["treatment"], "source_hash": "f81949978b0534ebcd1c56e3a53d1fa5e7b2536aea7bfd9fbcff21e4266e28da", "source_set_id": "0d0cfec0-5d18-c347-e063-6394a90a912c"}`
