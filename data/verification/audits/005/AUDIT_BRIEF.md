# Random audit 005

Re-fetch each source and confirm every fact is still explicitly supported. Record with:
`python -m src.recommendation.verification_loop audit --record pass|fail --notes ...`

## P400259
- source: https://www.drunkelephant.com/collections/serums/c-firma-fresh-vitamin-c-day-serum-812343034358.html
- retrieved_at: 2026-07-16T20:55:29Z
- sha256: 9355e6eb20c0ec324a637402d7f74f7da8274d83a8921be1e8d15f1fd0e04b32
- facts: `{"cadence": "am_daily", "cadence_source": "https://www.drunkelephant.com/collections/serums/c-firma-fresh-vitamin-c-day-serum-812343034358.html", "routine_roles": ["serum"]}`

## P392235
- source: https://tatcha.com/products/the-camellia-cleansing-oil-and-makeup-remover
- retrieved_at: 2026-07-16T18:51:15Z
- sha256: 27dcf2d0e5516e77efa8733fba929e804511fe6d95aed5d3040d2cb90c441664
- facts: `{"comedogenic_claim": "claimed_noncomedogenic", "evidence_grade": "manufacturer_product_page", "exposure": "rinse_off", "format": "oil", "routine_roles": ["cleanser"]}`

## P427411
- source: https://theordinary.com/en-us/azelaic-acid-suspension-10-exfoliator-100407.html
- retrieved_at: 2026-07-16T18:00:15Z
- sha256: 66007051596787473b988e64f50473c7a36d58f2ce06fed57c94621a908e0890
- facts: `{"cadence": "am_pm", "cadence_source": "https://theordinary.com/en-us/azelaic-acid-suspension-10-exfoliator-100407.html", "exposure": "leave_on", "format": "suspension", "intended_areas": ["face"], "routine_roles": ["serum"]}`

## P188306
- source: https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml
- retrieved_at: 2026-07-14T18:13:41.489734Z
- sha256: f81949978b0534ebcd1c56e3a53d1fa5e7b2536aea7bfd9fbcff21e4266e28da
- facts: `{"amount": "thin_layer", "amount_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml", "cadence": "per_label", "cadence_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml", "contraindications": ["sensitive"], "drug_actives": [{"name": "benzoyl_peroxide", "source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml", "strength": "2.5%"}], "evidence_grade": "regulatory_label", "evidence_roles": ["acne_treatment"], "exposure": "leave_on", "format": "lotion", "label_effective_date": "20250815", "label_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml", "label_verified_at": "2026-07-14", "label_version": "3", "ndc_product_code": "49527-117", "otc_drug": true, "routine_roles": ["treatment"], "source_hash": "f81949978b0534ebcd1c56e3a53d1fa5e7b2536aea7bfd9fbcff21e4266e28da", "source_set_id": "0d0cfec0-5d18-c347-e063-6394a90a912c"}`

## P427420
- source: https://theordinary.com/en-us/multi-peptide-ha-serum-100613.html?dwvar_100613_size=60ml&quantity=1
- retrieved_at: 2026-07-16T20:55:30Z
- sha256: 10b02c529ced4f48eb1520fdb71bb5e1ff35fd9dce68cf5bf0f7bc8e90c71eba
- facts: `{"cadence": "am_pm", "cadence_source": "https://theordinary.com/en-us/multi-peptide-ha-serum-100613.html?dwvar_100613_size=60ml&quantity=1", "routine_roles": ["serum"]}`
