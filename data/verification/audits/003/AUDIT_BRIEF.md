# Random audit 003

Re-fetch each source and confirm every fact is still explicitly supported. Record with:
`python -m src.recommendation.verification_loop audit --record pass|fail --notes ...`

## P429242
- source: https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/15a786f2-2939-4aa2-a413-b661ff6d2dd9.xml
- retrieved_at: 2026-07-14T18:27:51.301104Z
- sha256: 9be337200f65be18e894a7b9c7b0ea1a1be478eb1f536ce291cdfcb2d0711ce7
- facts: `{"broad_spectrum": true, "cadence": "per_label", "cadence_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/15a786f2-2939-4aa2-a413-b661ff6d2dd9.xml", "evidence_grade": "regulatory_label", "evidence_roles": ["daily_support"], "exposure": "leave_on", "format": "stick", "intended_areas": ["face"], "label_effective_date": "20241223", "label_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/15a786f2-2939-4aa2-a413-b661ff6d2dd9.xml", "label_verified_at": "2026-07-14", "label_version": "2", "otc_drug": true, "routine_roles": ["sunscreen"], "source_set_id": "15a786f2-2939-4aa2-a413-b661ff6d2dd9", "spf": 50}`

## P394639
- source: https://lgbeauty.com/products/the-true-cream-aqua-bomb
- retrieved_at: 2026-07-16T18:32:25Z
- sha256: 9361d83914c604958187529ae517424cff3520465fc81561a43355867d2b45d6
- facts: `{"cadence": "am_pm", "cadence_source": "https://lgbeauty.com/products/the-true-cream-aqua-bomb", "evidence_grade": "manufacturer_product_page", "evidence_roles": ["daily_support"], "exposure": "leave_on", "format": "cream", "intended_areas": ["face"], "routine_roles": ["moisturizer"]}`

## P429953
- source: https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/61e543e3-9ace-6440-e053-2991aa0a9647.xml
- retrieved_at: 2026-07-16T18:42:11Z
- sha256: 3326fa27972f33906b0d77c0feb8baca16b92156cfa191ca2fdd38fa2d3fd189
- facts: `{"broad_spectrum": true, "label_effective_date": "20260109", "label_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/61e543e3-9ace-6440-e053-2991aa0a9647.xml", "label_verified_at": "2026-07-16", "label_version": "13", "otc_drug": true, "source_set_id": "61e543e3-9ace-6440-e053-2991aa0a9647", "spf": 50}`

## P454380
- source: https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0670214e-520b-4126-8f28-71c765117276.xml
- retrieved_at: 2026-07-14T18:13:41.489734Z
- sha256: a87bd1db66aefb21e6bed3dbd55f3f3bc148dea8d031c1331458b28ad5aefaaf
- facts: `{"broad_spectrum": true, "cadence": "per_label", "cadence_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0670214e-520b-4126-8f28-71c765117276.xml", "evidence_grade": "regulatory_label", "exposure": "leave_on", "format": "cream", "label_effective_date": "20240807", "label_source": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0670214e-520b-4126-8f28-71c765117276.xml", "label_verified_at": "2026-07-14", "label_version": "10", "otc_drug": true, "routine_roles": ["sunscreen"], "source_set_id": "0670214e-520b-4126-8f28-71c765117276", "spf": 40}`

## P441101
- source: https://tatcha.com/products/the-dewy-skin-cream
- retrieved_at: 2026-07-14T18:21:17.942408Z
- sha256: 965b7cce539295e0c4916690a543da1e65a36fef07f920b8750b2b646b66f4b1
- facts: `{"cadence": "am_pm", "cadence_source": "https://tatcha.com/products/the-dewy-skin-cream", "comedogenic_claim": "claimed_noncomedogenic", "evidence_grade": "manufacturer_product_page", "evidence_roles": ["daily_support"], "exposure": "leave_on", "format": "cream", "intended_areas": ["face"], "routine_roles": ["moisturizer"]}`
