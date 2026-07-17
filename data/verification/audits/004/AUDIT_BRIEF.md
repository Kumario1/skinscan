# Random audit 004

Re-fetch each source and confirm every fact is still explicitly supported. Record with:
`python -m src.recommendation.verification_loop audit --record pass|fail --notes ...`

## P467976
- source: https://supergoop.com/products/mineral-setting-powder
- retrieved_at: 2026-07-16T18:31:23Z
- sha256: 92befeb7eb991b2b2815da7803360b18e291330b8f1d9556bb1f820f1acef232
- facts: `{"cadence": "per_label", "cadence_source": "https://supergoop.com/products/mineral-setting-powder", "comedogenic_claim": "claimed_noncomedogenic", "evidence_grade": "manufacturer_product_page", "exposure": "leave_on", "format": "powder", "intended_areas": ["face", "neck"], "routine_roles": ["sunscreen"]}`

## P461537
- source: https://tatcha.com/products/rice-wash-soft-cream-cleanser
- retrieved_at: 2026-07-14T18:27:51.301104Z
- sha256: 615436bd1e5352eb4f544e71e6cfd9232f12e9f25185077bdcb9404d2cca9784
- facts: `{"cadence": "am_pm", "cadence_source": "https://tatcha.com/products/rice-wash-soft-cream-cleanser", "evidence_grade": "manufacturer_product_page", "evidence_roles": ["daily_support"], "exposure": "rinse_off", "format": "cleanser", "intended_areas": ["face"], "routine_roles": ["cleanser"]}`

## P429242
- source: https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/15a786f2-2939-4aa2-a413-b661ff6d2dd9.xml
- retrieved_at: 2026-07-14T18:27:51.301104Z
- sha256: 9be337200f65be18e894a7b9c7b0ea1a1be478eb1f536ce291cdfcb2d0711ce7
- facts: `{"broad_spectrum": true, "cadence": "per_label", "cadence_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/15a786f2-2939-4aa2-a413-b661ff6d2dd9.xml", "evidence_grade": "regulatory_label", "evidence_roles": ["daily_support"], "exposure": "leave_on", "format": "stick", "intended_areas": ["face"], "label_effective_date": "20241223", "label_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/15a786f2-2939-4aa2-a413-b661ff6d2dd9.xml", "label_verified_at": "2026-07-14", "label_version": "2", "otc_drug": true, "routine_roles": ["sunscreen"], "source_set_id": "15a786f2-2939-4aa2-a413-b661ff6d2dd9", "spf": 50}`

## P427536
- source: https://tatcha.com/products/the-deep-cleanse
- retrieved_at: 2026-07-14T18:27:51.301104Z
- sha256: 9fca6f5ab6c4afb4ff12199777242a49bb5c6871145239a7e79bc9a1c708b259
- facts: `{"cadence": "daily", "cadence_source": "https://tatcha.com/products/the-deep-cleanse", "evidence_grade": "manufacturer_product_page", "evidence_roles": ["daily_support"], "exposure": "rinse_off", "format": "cleanser", "intended_areas": ["face"], "routine_roles": ["cleanser"]}`

## P429953
- source: https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/61e543e3-9ace-6440-e053-2991aa0a9647.xml
- retrieved_at: 2026-07-16T18:42:11Z
- sha256: 3326fa27972f33906b0d77c0feb8baca16b92156cfa191ca2fdd38fa2d3fd189
- facts: `{"broad_spectrum": true, "label_effective_date": "20260109", "label_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/61e543e3-9ace-6440-e053-2991aa0a9647.xml", "label_verified_at": "2026-07-16", "label_version": "13", "otc_drug": true, "source_set_id": "61e543e3-9ace-6440-e053-2991aa0a9647", "spf": 50}`
