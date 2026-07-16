# Lesion-Specific Care Evidence Report

Status: **research complete; independent dermatologist review pending; not production-approved**

Policy proposal: `lesion_care_policy.proposed.json`

Source ledger: `lesion_care_source_manifest.json`

Evidence cutoff and retrieval date: 2026-07-16

Jurisdictional frame: United States; NICE is used as a current, high-quality clinical-practice guideline where it supplies operational detail not stated in the US guideline abstract.

## 1. Gate decision

The Phase 0 research gate is complete enough to submit for clinical review. It is **not approved**. No reviewer identity, credentials, review date, report hash acceptance, policy version acceptance, or exception list exists.

Consequently:

- no production recommendation code, model-analysis contract, or runtime therapy policy was changed;
- the proposed JSON deliberately uses a non-production schema and sets `production_eligible` to `false`;
- the application must not turn these findings into treatment instructions until an independent qualified dermatologist approves every label path, active, formulation, strength, contraindication, escalation rule, and user-facing phrase;
- catalog verification is not clinical approval, and clinical approval will not make an inadequately evidenced product eligible.

## 2. Research question and method

### Scope

Each detector label was researched separately: `closed_comedo`, `open_comedo`, `papule`, `pustule`, `nodule`, `atrophic_scar`, `hypertrophic_scar`, `melasma`, `nevus`, and `other`.

For each label, the review asked:

1. What can the visible finding reasonably mean?
2. What important lookalikes cannot be excluded from a photograph?
3. What history, physical examination, dermoscopy, Wood lamp examination, palpation, or biopsy may be needed?
4. What findings require referral or faster escalation?
5. Which management channels have support: retail, prescription discussion, procedure, monitoring, referral, or abstention?
6. For a retail path, what exact active, strength, role, formulation, exposure, cadence, contraindication, review period, and stop condition are supported?

### Source hierarchy and inclusion rules

The evidence hierarchy was applied in this order:

1. current clinical-practice guidelines;
2. systematic reviews and meta-analyses;
3. randomized trials when a higher-level source did not answer the question;
4. regulatory monographs, labels, safety communications, and enforcement statements;
5. professional-association patient guidance for morphology, examination expectations, and plain-language safety wording;
6. product manufacturers only for exact SKU facts, never for clinical efficacy.

Included sources had to be authoritative, applicable to a named label or its safety boundary, available in exact retrievable bytes, and specific enough to map to an auditable claim. Current sources were preferred; an older source was retained when it was the current regulatory monograph or no more recent source replaced it.

Excluded as clinical evidence: retailer copy, product reviews, influencer/social content, search snippets, manufacturer efficacy claims, non-specific wellness advice, and any claim that could not be mapped to a source and label. No individual RCT was added merely to increase source count when a current guideline or systematic review already answered the question.

### Search record

The search covered PubMed, NICE, FDA/AccessData, AAD, ACOG, and USPSTF. The principal queries were:

- `2024 AAD acne guideline PubMed acne management`
- `NICE NG198 acne recommendations referral scarring pregnancy`
- `AAD acne whiteheads blackheads papules pustules nodules`
- `AAD acne lookalikes rosacea perioral dermatitis folliculitis`
- `AAD depressed and raised acne scar treatment`
- `Cochrane silicone gel sheeting hypertrophic scars`
- `AAD melasma treatment iron oxide sunscreen`
- `melasma topical treatment systematic review PubMed`
- `AAD mole nevus ABCDE diagnosis treatment`
- `FDA OTC mole removal warning`
- `FDA OTC acne monograph benzoyl peroxide salicylic acid`
- `ACOG pregnancy acne topical ingredients`

The source manifest records the database or site, URL, publication date where available, retrieval timestamp, exact snapshot path, byte count, SHA-256 digest, query mapping, and inclusion decision.

## 3. Evidence synthesis by exact label

The following sections describe evidence, not approved instructions. “Proposed path” means the conservative machine-readable transformation in the proposed policy, subject to independent review.

### `closed_comedo`

**Finding meaning and limits.** AAD describes a whitehead as a closed plugged pore that appears white or flesh-colored. A photograph may show that morphology, but it cannot confirm acne or exclude milia, folliculitis, keratosis pilaris, perioral dermatitis, or other bumps. Detector confidence must therefore describe evidence quality, not diagnostic probability. [S06, S07]

**Evidence-supported management.** The 2024 AAD guideline strongly recommends topical retinoids and conditionally recommends salicylic acid and azelaic acid for acne. NICE recommends a 12-week first-line course chosen by clinical severity and preference and provides fixed-combination strengths; it notes that benefit may not be visible for 6–8 weeks and that alternate-day or short-contact introduction can reduce irritation. The FDA OTC acne monograph permits salicylic acid 0.5%–2%. FDA labeling supports adapalene 0.1% for acne in people 12 and older, with once-daily label directions and irritation precautions. [S01–S04]

**Safety boundary and proposed path.** NICE says topical retinoids are contraindicated in pregnancy and when planning pregnancy; ACOG says topical retinoids are generally avoided during pregnancy. The proposal therefore defers adapalene for pregnancy, trying, nursing, or unknown status until the reviewer resolves exact wording. Retail matching requires an exact regulatory drug active and strength, not a cosmetic “retinol” substitute. Diagnostic uncertainty, deep painful lesions, scarring, or treatment failure route to a clinician. [S02, S04, S05]

### `open_comedo`

**Finding meaning and limits.** AAD describes a blackhead as an open plugged pore whose surface darkens after exposure to air, not because the pore contains dirt. Pigmented and follicular lookalikes cannot all be excluded from a single image. [S06]

**Evidence-supported management.** The active evidence is the same comedonal-acne evidence as for a closed comedo: topical retinoid, salicylic acid, and selected combination paths, with 6–8 weeks before early benefit and review around 12 weeks. [S01–S04]

**Safety boundary and proposed path.** The proposal allows only exact, label-supported adapalene 0.1% or salicylic acid 0.5%–2% candidates after intake and product gates. It forbids extraction instructions, antibiotic monotherapy, and substitution of an unverified cosmetic retinoid. [S02–S05]

### `papule`

**Finding meaning and limits.** AAD describes a papule as an inflamed pimple without visible pus. A photograph cannot establish etiology or exclude rosacea, folliculitis, perioral dermatitis, or another inflammatory eruption. [S06, S07]

**Evidence-supported management.** AAD strongly recommends benzoyl peroxide and topical retinoids and supports combining different mechanisms. NICE recommends fixed topical combinations and prohibits topical antibiotic monotherapy, oral antibiotic monotherapy, and simultaneous topical-plus-oral antibiotic use as an unstructured regimen. The FDA monograph permits benzoyl peroxide 2.5%–10%. [S01–S03]

**Safety boundary and proposed path.** A retail candidate must have exact Drug Facts, role, exposure, cadence, contraindications, and strength. The proposed path favors a single benzoyl-peroxide or exact adapalene-containing treatment rather than stacking products. Severe pain, rapid spread, systemic symptoms, scarring, diagnostic uncertainty, or failure after a reviewed course escalates. [S01–S03]

### `pustule`

**Finding meaning and limits.** AAD describes a pustule as an inflamed bump containing pus. A photograph cannot rule out folliculitis or infection, so the visible morphology is not a diagnosis. [S06, S07]

**Evidence-supported management.** The inflammatory-acne evidence is the same as for papules: benzoyl peroxide and topical retinoids are strongly recommended, combination mechanisms are preferred where needed, and antibiotic monotherapy is discouraged. [S01–S03]

**Safety boundary and proposed path.** The proposal uses the same exact-product gate as papules and explicitly forbids popping or lancing advice. Rapid spread, fever, eye involvement, severe pain, deep lesions, or scarring requires clinician review. [S02, S07]

### `nodule`

**Finding meaning and limits.** AAD describes a nodule as a deep, painful, firm lesion without pus. A photograph cannot establish depth, tenderness, fluctuation, or scarring risk; palpation and history matter. [S06]

**Evidence-supported management.** NICE recommends referral for nodulocystic acne. AAD strongly recommends isotretinoin for severe acne, acne causing psychosocial burden or scarring, or acne failing standard oral or topical therapy; it also recognizes clinician-administered intralesional corticosteroid for larger lesions. [S01, S02]

**Safety boundary and proposed path.** This is clinician-first. There is no automatic retail treatment match. Prescription and procedural options are discussion topics only, not start/stop instructions. Severe pain, rapid worsening, systemic symptoms, scarring, or psychosocial burden increases urgency. [S01, S02]

### `atrophic_scar`

**Finding meaning and limits.** Depressed acne scars include subtypes that respond differently to procedures. A photograph cannot reliably distinguish scar subtype, active acne, enlarged pores, or other texture and cannot assess tethering or skin mechanics. A dermatologist examines the scar and skin before choosing therapy. [S08]

**Evidence-supported management.** Active acne should be controlled before scar procedures. AAD describes procedure selection by scar type, including microneedling, laser, filler, chemical methods, punch techniques, and surgery; selected mild scarring may improve with a topical retinoid or salicylic acid. NICE recommends specialist referral when severe scarring persists for a year after acne clears and lists CO2 laser, punch elevation, and glycolic-acid peel as possible specialist options. [S02, S08]

**Safety boundary and proposed path.** This is clinician-first and receives no automatic retail product. FDA warns that high-strength chemical peels used without professional supervision can cause burns, infection, pigment change, and scarring. [S18]

### `hypertrophic_scar`

**Finding meaning and limits.** A raised scar must be distinguished from keloid, ongoing inflammation, infection, and another raised lesion. The application cannot confirm the scar type or whether the wound is fully healed from a photograph. [S09]

**Evidence-supported management.** AAD states that silicone sheets or ointments may reduce features of a raised scar, should be used only after the wound closes, and may require daily use for months; rash or skin breakdown is a stop condition. The 2021 Cochrane review found the randomized evidence for silicone gel sheeting in hypertrophic scars to be sparse and generally very low certainty. [S09, S10]

**Safety boundary and proposed path.** This is clinician-first. Silicone is retained as a discussion option, not an automatic retail match, until a clinician confirms the context and the catalog contains a dedicated evidence-complete scar product. Generic dimethicone in a moisturizer is not treated as evidence of a silicone scar-care formulation. [S09, S10]

### `melasma`

**Finding meaning and limits.** AAD notes that a dermatologist may diagnose melasma by examination and may use dermoscopy, Wood lamp examination, or occasionally biopsy to exclude another condition. A photograph cannot exclude post-inflammatory pigment change, medication-related pigmentation, a nevus, or malignancy. [S11]

**Evidence-supported management.** AAD emphasizes daily broad-spectrum SPF 30+ sun protection and visible-light protection with iron oxide in a tinted sunscreen. It lists prescription hydroquinone and triple combination and clinician-selected azelaic acid, kojic acid, or vitamin C; results commonly take 3–12 months. A 2022 systematic review found support for triple-combination therapy, hydroquinone, tretinoin, and visible-light-protective sunscreen while also finding heterogeneous studies and evidence ranging from very low to high certainty. [S11, S12]

**Safety boundary and proposed path.** The only proposed automatic retail role is a tinted, broad-spectrum SPF 30+ facial sunscreen with deterministically parsed iron oxides and exact SKU evidence. FDA states that OTC hydroquinone skin-lightening products are not lawfully marketed under the OTC framework and warns about harms such as ochronosis; FDA also warns against unsupervised high-strength peels. Pigment treatment and procedures remain clinician discussion. A new, changing, asymmetric, bleeding, itching, or painful spot routes out of the melasma path. [S13, S18]

### `nevus`

**Finding meaning and limits.** A model label cannot determine that a mole is benign or act as skin-cancer screening. AAD uses change over time and the ABCDE warning features— asymmetry, border, color, diameter, and evolution—along with symptoms and examination; a dermatologist may use removal and microscopy when a lesion is suspicious. [S14–S16]

**Evidence-supported management.** Most stable moles need no product treatment. New or changing lesions, an outlier lesion, bleeding, itching, pain, or a new mole after age 30 deserve dermatology review. The USPSTF “insufficient evidence” statement concerns clinician visual screening of asymptomatic people and does not justify reassurance for a symptomatic or changing spot. [S14–S16, S19]

**Safety boundary and proposed path.** Monitoring or referral only; no product match. FDA states there are no legally marketed OTC drugs for mole removal and warns that self-treatment can delay cancer diagnosis and cause injury or scarring. [S17]

### `other`

**Finding meaning and limits.** This label is an explicit unsupported result, not a miscellaneous treatment bucket. It may contain benign, infectious, inflammatory, drug-related, or malignant conditions whose management conflicts.

**Safety boundary and proposed path.** No product is matched and no nearest-label substitution is allowed. The application should explain that the image did not support a named care pathway. New or changing lesions, bleeding, itching, pain, rapid spread, fever, eye involvement, systemic symptoms, or persistent concern route to clinical evaluation. This conservative abstention is a policy inference from the documented diagnostic uncertainty and lesion safety sources, not a separately validated treatment claim. [S07, S16, S19]

## 4. Auditable claim extraction

Each row is one claim and uses exactly one of the permitted management channels. “Source strength” is the source-stated recommendation or evidence status, not ingredient concentration. When two sources support different parts of a row, their URLs, dates, retrieval times, and hashes appear in the same order.

| Claim | Detector label | Proposed app wording | Channel | Ingredient or therapy | Intended role | Formulation | Source strength | Population and exclusions | Safety constraints | Expected review period | Source URL | Publication date | Retrieval date | Snapshot SHA-256 | Researcher conclusion | Unresolved uncertainty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | closed_comedo | The image model found features most consistent with a closed comedo; a photograph cannot confirm acne. | refer | Diagnostic confirmation | Clinical assessment | History and skin examination | Association differential-diagnosis guidance | People with an acne-like bump; no diagnosis from image alone | Lookalikes require examination | Refer when uncertain | [S07](https://www.aad.org/public/diseases/acne/really-acne/stubborn-acne) | not stated | 2026-07-16T22:15:15Z | `dc66acea4512cedee9ff014a3a7d7940902f66317f61f00098d82f03e67cdade` | Use non-diagnostic finding wording. | Photo-only diagnostic performance is not established. |
| C02 | closed_comedo | If clinically approved and intake is complete, an exact adapalene acne drug may be considered. | retail | Adapalene 0.1% | Treatment | FDA-submitted OTC leave-on gel label | AAD strong topical-retinoid recommendation; product-specific regulatory label | Age 12+; pregnancy/trying excluded by proposal; nursing or unknown deferred | Irritation, allergy, damaged skin, duplicate active | Benefit may take 6–8 weeks; policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S04](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=330bc2b3-e064-4189-9c47-9c5a50aa4dde)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>updated 2026-06-01<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:28:02Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`f2ba90bdaece79884d9e7bfdeb9e63e3c761d440cf160df9daace3b1ff031de2`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Evidence supports a candidate class, not automatic placement. | Exact introductory cadence and nursing rule need dermatologist approval. |
| C03 | closed_comedo | If clinically approved and product evidence is exact, salicylic acid may be considered. | retail | Salicylic acid 0.5%–2% | Treatment | FDA-compliant OTC acne drug with verified exposure | AAD conditional recommendation; FDA monograph concentration | Complete intake; no allergy, conflict, or product contraindication | Do not infer drug indication or concentration from cosmetic INCI alone | Policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S03](https://www.accessdata.fda.gov/drugsatfda_docs/omuf/monographs/OTC%20Monograph_M006-Topical%20Acne%20drug%20products%20for%20OTC%20Human%20Use%2011.23.2021.pdf)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>2021-11-23<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`33caa0d71f5648cffd08af2cdced3c9d2b2b20dcbf006068e6379ae399f0a80e`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Conditional clinical evidence exists, but current catalog coverage is unfilled. | Leave-on versus rinse-off wording needs clinical review. |
| C04 | open_comedo | The image model found features most consistent with an open comedo; the dark surface is not trapped dirt. | monitor | Gentle observation while confirming the finding | Finding only | No product | Association morphology guidance | General public with a compatible visible pore | Do not recommend extraction | Refer if uncertain or changing | [S06](https://www.aad.org/public/diseases/acne/really-acne/symptoms) | not stated | 2026-07-16T22:15:15Z | `ffbbffa2cdd51bff85ad5bb0cb896e8365d277cfd8a9e8dfd2d970e881670b6f` | Compatible morphology can be described without diagnosis. | Pigmented or follicular lookalikes remain. |
| C05 | open_comedo | If clinically approved and intake is complete, an exact adapalene acne drug may be considered. | retail | Adapalene 0.1% | Treatment | FDA-submitted OTC leave-on gel label | AAD strong topical-retinoid recommendation; product-specific regulatory label | Age 12+; pregnancy/trying excluded by proposal; nursing or unknown deferred | Irritation, allergy, damaged skin, duplicate active | Benefit may take 6–8 weeks; policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S04](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=330bc2b3-e064-4189-9c47-9c5a50aa4dde)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>updated 2026-06-01<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:28:02Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`f2ba90bdaece79884d9e7bfdeb9e63e3c761d440cf160df9daace3b1ff031de2`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Evidence supports a candidate class, not automatic placement. | Exact introductory cadence and nursing rule need dermatologist approval. |
| C06 | open_comedo | If clinically approved and product evidence is exact, salicylic acid may be considered. | retail | Salicylic acid 0.5%–2% | Treatment | FDA-compliant OTC acne drug with verified exposure | AAD conditional recommendation; FDA monograph concentration | Complete intake; no allergy, conflict, or product contraindication | Do not infer drug indication or concentration from cosmetic INCI alone | Policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S03](https://www.accessdata.fda.gov/drugsatfda_docs/omuf/monographs/OTC%20Monograph_M006-Topical%20Acne%20drug%20products%20for%20OTC%20Human%20Use%2011.23.2021.pdf)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>2021-11-23<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`33caa0d71f5648cffd08af2cdced3c9d2b2b20dcbf006068e6379ae399f0a80e`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Conditional clinical evidence exists, but current catalog coverage is unfilled. | Leave-on versus rinse-off wording needs clinical review. |
| C07 | papule | If clinically approved and product evidence is exact, benzoyl peroxide may be considered. | retail | Benzoyl peroxide 2.5%–10% | Treatment | FDA-compliant OTC acne drug; reviewed exposure required | AAD strong recommendation; FDA monograph concentration | Complete intake; no allergy, conflict, or product contraindication | Irritation and bleaching warning; do not stack duplicate actives | Benefit may take 6–8 weeks; policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S03](https://www.accessdata.fda.gov/drugsatfda_docs/omuf/monographs/OTC%20Monograph_M006-Topical%20Acne%20drug%20products%20for%20OTC%20Human%20Use%2011.23.2021.pdf)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>2021-11-23<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`33caa0d71f5648cffd08af2cdced3c9d2b2b20dcbf006068e6379ae399f0a80e`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | One exact benzoyl-peroxide SKU currently clears the neutral coverage gate. | No universal best concentration is established here. |
| C08 | papule | A clinician may discuss fixed-combination topical therapy if a single retail path is unsuitable. | prescription_discussion | Adapalene plus benzoyl peroxide or another reviewed combination | Clinician-managed treatment | Fixed topical combination | NICE first-line recommendation; AAD strong component recommendations | Clinician-assessed acne; pregnancy and individual contraindications apply | No app-generated prescription start, stop, or dose | Review at 12 weeks | [S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations)<br>[S01](https://pubmed.ncbi.nlm.nih.gov/38300170/) | 2021-06-25, updated 2026-04-30<br>2024-05 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8`<br>`0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7` | Combination therapy is a clinician discussion option. | Exact choice depends on assessed severity and history. |
| C09 | papule | Do not use an antibiotic as acne monotherapy. | unsupported | Topical or oral antibiotic monotherapy | None | None | NICE do-not-use recommendation | People with acne | Antimicrobial stewardship; a clinician may use an antibiotic only within an appropriate regimen | Clinician-defined | [S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z | `de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Automatic monotherapy is prohibited. | Not a prohibition on every clinician-managed combination. |
| C10 | pustule | If clinically approved and product evidence is exact, benzoyl peroxide may be considered. | retail | Benzoyl peroxide 2.5%–10% | Treatment | FDA-compliant OTC acne drug; reviewed exposure required | AAD strong recommendation; FDA monograph concentration | Complete intake; no allergy, conflict, or product contraindication | Irritation and bleaching warning; do not pop or lance | Benefit may take 6–8 weeks; policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S03](https://www.accessdata.fda.gov/drugsatfda_docs/omuf/monographs/OTC%20Monograph_M006-Topical%20Acne%20drug%20products%20for%20OTC%20Human%20Use%2011.23.2021.pdf)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>2021-11-23<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`33caa0d71f5648cffd08af2cdced3c9d2b2b20dcbf006068e6379ae399f0a80e`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | One exact benzoyl-peroxide SKU currently clears the neutral coverage gate. | A photograph cannot exclude folliculitis or infection. |
| C11 | pustule | A clinician may discuss fixed-combination topical therapy. | prescription_discussion | Adapalene plus benzoyl peroxide or another reviewed combination | Clinician-managed treatment | Fixed topical combination | NICE first-line recommendation | Clinician-assessed acne; pregnancy and individual contraindications apply | No app-generated prescription start, stop, or dose | Review at 12 weeks | [S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z | `de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Combination therapy is a clinician discussion option. | Exact choice depends on assessed severity and history. |
| C12 | pustule | Do not use an antibiotic as acne monotherapy. | unsupported | Topical or oral antibiotic monotherapy | None | None | NICE do-not-use recommendation | People with acne | Antimicrobial stewardship; clinician regimen required | Clinician-defined | [S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z | `de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Automatic monotherapy is prohibited. | Not a prohibition on every clinician-managed combination. |
| C13 | nodule | A photograph cannot confirm depth; this finding needs clinician assessment. | refer | Nodular or nodulocystic acne assessment | Clinical assessment | History and palpation | NICE referral recommendation | Suspected nodular or nodulocystic acne | Severe pain, rapid worsening, systemic symptoms, scarring, or uncertainty increase urgency | Prompt; urgency clinician-assessed | [S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z | `de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | No retail treatment match. | Photo cannot establish depth or severity. |
| C14 | nodule | A dermatologist may discuss isotretinoin when severe, scarring, burdensome, or refractory acne is confirmed. | prescription_discussion | Isotretinoin | Clinician-managed systemic treatment | Oral prescription | AAD strong recommendation in specified contexts | Severe, scarring, psychosocially burdensome, or standard-therapy-resistant acne | Full systemic and pregnancy-safety program required | Clinician-defined | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/) | 2024-05 | 2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7` | Discussion only; never an app start/stop instruction. | The model cannot establish qualifying severity. |
| C15 | nodule | A dermatologist may consider an injection for a selected large lesion. | procedure | Intralesional corticosteroid | Clinician-administered procedure | Injection | AAD good-practice option for larger lesions | Selected large inflammatory lesions after examination | Procedure risks and diagnosis require clinician management | Clinician-defined | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/) | 2024-05 | 2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7` | Procedure discussion only. | The accessible abstract supplies limited procedural detail. |
| C16 | atrophic_scar | Control active acne before treating a depressed scar. | refer | Active-acne control and scar-subtype assessment | Clinical assessment | Dermatologist examination | AAD professional guidance | People with acne breakouts and depressed scars | Do not begin scar procedures while active acne is uncontrolled | Clinician-defined | [S08](https://www.aad.org/public/diseases/acne/derm-treat/scars/treatment) | 2023-12-08 | 2026-07-16T22:15:15Z | `f32004f01f832c27c094341e98dfe2af98ab7e87e6bed0abb595696d014ebcec` | Clinician-first; no automatic product. | Photo cannot reliably subtype or assess tethering. |
| C17 | atrophic_scar | A dermatologist may select a procedure after confirming scar subtype and pigment risk. | procedure | Microneedling, laser, filler, punch technique, chemical reconstruction, or surgery | Clinician-administered procedure | Subtype-specific | AAD professional guidance; NICE specialist options | Confirmed depressed scar after active-acne control | Procedure choice and pigment risk are individualized | Clinician-defined; severe persistent scars may be referred after one year | [S08](https://www.aad.org/public/diseases/acne/derm-treat/scars/treatment)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2023-12-08<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `f32004f01f832c27c094341e98dfe2af98ab7e87e6bed0abb595696d014ebcec`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Procedure discussion only. | No universal procedure or course. |
| C18 | atrophic_scar | Do not use a high-strength chemical peel without professional supervision. | unsupported | Unsupervised chemical peel | None | High-concentration acid peel | FDA safety warning | Consumers considering peel products | Burns, infection, pigment change, and scarring | Seek care for injury | [S18](https://www.fda.gov/drugs/drug-alerts-and-statements/fda-warns-against-purchasing-or-using-chemical-peel-skin-products-without-professional-supervision) | 2024-07-30 | 2026-07-16T22:15:15Z | `fa2e2a8bfe9ef8556a72dbfe1b9cad5c1d47773297d97b9eb1044ad0cb7e1b5a` | Block retail matching for unsupervised high-strength peels. | Does not compare supervised procedures. |
| C19 | hypertrophic_scar | A photograph cannot confirm hypertrophic scar versus keloid or another raised lesion. | refer | Raised-scar confirmation | Clinical assessment | History and examination | AAD professional guidance | People with a raised lesion or scar | Open wound, infection, rapid growth, bleeding, or diagnostic uncertainty | Prompt if changing or symptomatic | [S09](https://www.aad.org/public/diseases/a-z/scars-treatment) | not stated | 2026-07-16T22:15:15Z | `72cf92f4e85afca63c416845f356df9c1190648dbfb7d9dc9405e88a37527672` | Clinician-first. | The source does not validate photo classification. |
| C20 | hypertrophic_scar | A clinician may discuss silicone sheet or gel after confirming a raised scar and closed wound. | refer | Silicone sheet or silicone gel | Scar-care evidence role; not a new routine role | Dedicated sheet or gel | AAD professional guidance | Confirmed raised scar on fully closed skin | Stop for rash or skin breakdown; no generic dimethicone substitution | Often months if clinician recommends | [S09](https://www.aad.org/public/diseases/a-z/scars-treatment) | not stated | 2026-07-16T22:15:15Z | `72cf92f4e85afca63c416845f356df9c1190648dbfb7d9dc9405e88a37527672` | Discussion only; catalog is unfilled. | Exact product and duration wording need review. |
| C21 | hypertrophic_scar | Evidence is too uncertain for an automatic silicone recommendation. | unsupported | Silicone gel sheeting | None | Sheet | Cochrane systematic review; generally very-low-certainty evidence | People with hypertrophic scars in small heterogeneous trials | Do not present benefit as established | Clinician-defined reassessment | [S10](https://pubmed.ncbi.nlm.nih.gov/34564840/) | 2021-09-26 | 2026-07-16T22:15:15Z | `101c3727a4b38ebc01c5f189df0799adfcca65bb38ba13fac247b07ba6145812` | No automatic retail match. | Few small trials and imprecision. |
| C22 | melasma | If clinically approved, a tinted broad-spectrum SPF 30+ sunscreen with exact iron-oxide evidence may be considered. | retail | Iron-oxide visible-light photoprotection | Sunscreen | Tinted leave-on face sunscreen, SPF 30+ | AAD professional guidance | People with a pattern consistent with melasma; new/changing/symptomatic spots excluded | Exact SKU, SPF, broad spectrum, exposure, role, and iron-oxide INCI required | Improvement may take 3–12 months | [S11](https://www.aad.org/public/diseases/a-z/melasma-treatment) | 2022-02-15 | 2026-07-16T22:15:15Z | `3390d004857f9fa68dbac9ccd308d4ccb8795d1268f885cb3c7f89b2abb3ab98` | Three exact catalog SKUs clear the neutral coverage gate. | Optimal tint and iron-oxide concentration are not specified. |
| C23 | melasma | A dermatologist may discuss hydroquinone or triple-combination therapy after confirming melasma. | prescription_discussion | Hydroquinone or triple combination | Clinician-managed pigment treatment | Topical prescription | AAD guidance; systematic review with evidence from very low to high certainty | Clinician-confirmed melasma | Irritation, paradoxical darkening, and individual contraindications | Clinician-defined; visible improvement may take months | [S11](https://www.aad.org/public/diseases/a-z/melasma-treatment)<br>[S12](https://pubmed.ncbi.nlm.nih.gov/35290681/) | 2022-02-15<br>2022 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `3390d004857f9fa68dbac9ccd308d4ccb8795d1268f885cb3c7f89b2abb3ab98`<br>`aa3b49b70eb61667003fc509ac2359584427772960d93a500b6a24eb8eb189aa` | Prescription discussion only. | Trials are heterogeneous and exact regimen is clinician-specific. |
| C24 | melasma | Do not match an OTC hydroquinone skin-lightening product. | unsupported | OTC hydroquinone skin lightener | None | OTC skin-lightening product | FDA regulatory safety communication | US consumers | Products are not lawfully marketed under the OTC framework; ochronosis and other harms | Seek clinical advice for injury or worsening | [S13](https://www.fda.gov/drugs/drug-safety-communications/fda-works-protect-consumers-potentially-harmful-otc-skin-lightening-products) | 2022-04-19 | 2026-07-16T22:15:15Z | `ec8881a2c34445da24e698217e58f52bc69c2953a52f2a890c0910431d94fcf9` | Block US retail matching. | Regulatory conclusion is jurisdiction-specific. |
| C25 | melasma | A changing or uncertain pigment pattern needs clinician assessment and may require dermoscopy, Wood lamp examination, or biopsy. | refer | Melasma differential diagnosis | Clinical assessment | Examination with optional diagnostic tools | AAD professional guidance | Pigment pattern suspected to be melasma | New, changing, asymmetric, bleeding, itching, painful, or uncertain spot exits retail path | Prompt if concerning | [S11](https://www.aad.org/public/diseases/a-z/melasma-treatment) | 2022-02-15 | 2026-07-16T22:15:15Z | `3390d004857f9fa68dbac9ccd308d4ccb8795d1268f885cb3c7f89b2abb3ab98` | Photo cannot confirm melasma. | Test selection is clinician-specific. |
| C26 | melasma | Do not use a high-strength chemical peel without professional supervision. | unsupported | Unsupervised chemical peel | None | High-concentration acid peel | FDA safety warning | Consumers considering peel products | Burns, infection, pigment change, and scarring | Seek care for injury | [S18](https://www.fda.gov/drugs/drug-alerts-and-statements/fda-warns-against-purchasing-or-using-chemical-peel-skin-products-without-professional-supervision) | 2024-07-30 | 2026-07-16T22:15:15Z | `fa2e2a8bfe9ef8556a72dbfe1b9cad5c1d47773297d97b9eb1044ad0cb7e1b5a` | Block unsupervised peel matching. | Does not compare supervised procedures. |
| C27 | nevus | A stable mole generally needs monitoring, not a product. | monitor | Observation | Monitoring | No product | AAD professional guidance | Stable, asymptomatic mole after appropriate assessment | Do not reassure that an image proves benignity | Follow clinician-directed monitoring | [S14](https://www.aad.org/public/diseases/a-z/moles-treatment) | not stated | 2026-07-16T22:15:15Z | `6e7449ce21cda8b72cd972814d555676a3382cfb48f1aaceb311616cd2914e72` | No product match. | Model output is not skin-cancer screening. |
| C28 | nevus | Any ABCDE feature or evolution needs prompt dermatology review. | refer | Dermoscopic assessment and possible biopsy | Clinical assessment | Dermatologist examination | AAD warning-sign guidance | New or changing mole or spot | Asymmetry, border, color, diameter, or evolution | Prompt | [S15](https://www.aad.org/public/diseases/a-z/moles-symptoms) | not stated | 2026-07-16T22:15:15Z | `535bd4b6e8faf3b2d05c3ff5e6ded4964e159fcbed4b3533b340237736d6c30e` | Refer; never reassure from model confidence. | ABCDE does not identify every melanoma. |
| C29 | nevus | A spot that differs, itches, bleeds, hurts, or changes needs dermatology review. | refer | Lesion examination | Clinical assessment | Dermatologist examination | AAD escalation guidance | Symptomatic or changing mole | Symptoms or change override a stable-looking photograph | Prompt | [S16](https://www.aad.org/public/diseases/a-z/when-is-a-mole-a-problem) | not stated | 2026-07-16T22:15:15Z | `f6591155dd0a9d613e830406d7f7289b96fc14207f46927e54a23fad83ebf665` | Refer. | Urgency depends on clinical context. |
| C30 | nevus | Never match or recommend a home mole-removal product. | unsupported | OTC mole remover or home removal | None | Any acid, freezing, burning, cutting, or home remedy | FDA regulatory enforcement; AAD professional guidance | US consumers with a mole or skin tag | Delayed cancer diagnosis, injury, infection, and scarring | Clinical examination if removal is desired or lesion is concerning | [S17](https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/ariella-naturals-632509-08042022)<br>[S14](https://www.aad.org/public/diseases/a-z/moles-treatment) | 2022-08-04<br>not stated | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `c4e8aa557153a8abb5630b9d713dcfb632fa969226a285e97fb1bff526ac724e`<br>`6e7449ce21cda8b72cd972814d555676a3382cfb48f1aaceb311616cd2914e72` | Absolute no-product rule. | FDA conclusion is jurisdiction-specific. |
| C31 | other | The image did not support a named lesion-specific care pathway. | unsupported | No therapy | None | No product | Conservative safety-policy inference from professional differential-diagnosis guidance | Any result assigned `other` | Do not substitute the nearest label; refer for new, changing, bleeding, itching, painful, rapidly spreading, ocular, febrile, or systemic findings | No treatment period; refer for red flags or persistent concern | [S07](https://www.aad.org/public/diseases/acne/really-acne/stubborn-acne) | not stated | 2026-07-16T22:15:15Z | `dc66acea4512cedee9ff014a3a7d7940902f66317f61f00098d82f03e67cdade` | Explicit abstention is safer than an unsupported product path. | No condition-specific clinical efficacy claim is possible. |
| C32 | closed_comedo | A clinician may discuss azelaic acid when the retail candidates are unsuitable or unsuccessful. | prescription_discussion | Azelaic acid | Clinician-managed acne treatment | Topical; exact strength and vehicle clinician-selected | AAD conditional recommendation for acne | Clinician-assessed acne; no automatic detector-only placement | Exact strength, cadence, pregnancy context, interactions, and irritation counseling require clinical review | Policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Clinician discussion only; no current retail coverage. | The guideline abstract supplies no exact azelaic-acid strength or formulation. |
| C33 | open_comedo | A clinician may discuss azelaic acid when the retail candidates are unsuitable or unsuccessful. | prescription_discussion | Azelaic acid | Clinician-managed acne treatment | Topical; exact strength and vehicle clinician-selected | AAD conditional recommendation for acne | Clinician-assessed acne; no automatic detector-only placement | Exact strength, cadence, pregnancy context, interactions, and irritation counseling require clinical review | Policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Clinician discussion only; no current retail coverage. | The guideline abstract supplies no exact azelaic-acid strength or formulation. |
| C34 | papule | A clinician may discuss azelaic acid as an alternative within an assessed inflammatory-acne plan. | prescription_discussion | Azelaic acid | Clinician-managed acne treatment | Topical; exact strength and vehicle clinician-selected | AAD conditional recommendation for acne | Clinician-assessed acne; no automatic detector-only placement | Exact strength, cadence, pregnancy context, interactions, and irritation counseling require clinical review | Policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Clinician discussion only. | The guideline abstract supplies no exact azelaic-acid strength or formulation. |
| C35 | pustule | A clinician may discuss azelaic acid as an alternative within an assessed inflammatory-acne plan. | prescription_discussion | Azelaic acid | Clinician-managed acne treatment | Topical; exact strength and vehicle clinician-selected | AAD conditional recommendation for acne | Clinician-assessed acne; no automatic detector-only placement | Exact strength, cadence, pregnancy context, interactions, and irritation counseling require clinical review | Policy review at 12 weeks | [S01](https://pubmed.ncbi.nlm.nih.gov/38300170/)<br>[S02](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations) | 2024-05<br>2021-06-25, updated 2026-04-30 | 2026-07-16T22:15:15Z<br>2026-07-16T22:15:15Z | `0f89baf2e84ba8a1e50604aea41c05d454c3e1b025f67e676871fe5a5003d9b7`<br>`de62fcfcb356c5e276bbf3bb03ec1405b62637a37f0f975d8d662cd3b38277b8` | Clinician discussion only. | The guideline abstract supplies no exact azelaic-acid strength or formulation. |
| C36 | melasma | After confirming melasma, a dermatologist may select azelaic acid, kojic acid, or vitamin C within an individualized plan. | refer | Azelaic acid, kojic acid, or vitamin C | Clinician-selected pigment treatment | Topical; exact strength, vehicle, and combination not specified | AAD professional treatment guidance | Clinician-confirmed melasma; a new, changing, or symptomatic pigment lesion is excluded | Irritation, pregnancy context, interactions, and paradoxical darkening require clinician review | Visible improvement may take 3–12 months | [S11](https://www.aad.org/public/diseases/a-z/melasma-treatment) | 2022-02-15 | 2026-07-16T22:15:15Z | `3390d004857f9fa68dbac9ccd308d4ccb8795d1268f885cb3c7f89b2abb3ab98` | Clinician-selected option only; no automatic product match. | The source does not supply a universal concentration, formulation, or comparative effect. |
| C37 | atrophic_scar | After scar confirmation and active-acne control, a dermatologist may discuss a topical retinoid or salicylic acid for selected mild scarring. | refer | Topical retinoid or salicylic acid | Clinician-selected mild-scar treatment | Topical; exact strength, vehicle, and cadence not specified | AAD professional treatment guidance | Selected mild acne scarring after active acne is controlled | Pregnancy, irritation, scar subtype, and duplicate-active risks require assessment; no automatic product | Clinician-defined | [S08](https://www.aad.org/public/diseases/acne/derm-treat/scars/treatment) | 2023-12-08 | 2026-07-16T22:15:15Z | `f32004f01f832c27c094341e98dfe2af98ab7e87e6bed0abb595696d014ebcec` | Discussion only; clinician-first status remains. | No effect size or exact regimen is supplied. |

## 5. Safety policy separated from clinical evidence

The sources above do not directly specify the application’s complete control logic. The following are conservative safety-policy transformations proposed for reviewer approval:

- **Unknown is not favorable.** A missing age, pregnancy status, allergy, sensitivity, medication, current-active, duration, pain/depth, or conditional lesion answer defers the affected active path.
- **Exact label, not grouped concern.** `closed_comedo`, `open_comedo`, `papule`, `pustule`, and `nodule` may share some evidence, but their output status and escalation differ. Scar and pigment labels are not collapsed.
- **Confidence wording.** Model confidence is never rendered as the chance that the person has a disease.
- **One product per requested role.** Within an AM or PM session, the selector may return at most one product for each requested routine role; an empty eligible set stays empty.
- **Prescription/procedure boundary.** The app names discussion options but never instructs starting, stopping, dosing, obtaining, or performing one.
- **Nevus and `other`.** These labels cannot produce a product match.
- **No gap filling.** A missing exact product remains an explicit coverage gap.

These transformations are identifiable in the proposed policy by reason codes and global rules. They must be reviewed as policy, not misrepresented as direct quotations from guidelines.

## 6. Product-evidence audit

### Inputs and reproducibility

The audit loaded the current full and drug catalogs, applied the current approved verification overlay at a fixed timestamp of 2026-07-16 UTC, used deterministic INCI parsing, and evaluated current hard gates against a complete neutral audit profile (age 30, normal skin, self-reported medium tone, pregnancy not applicable, and no allergies, medications, sensitivities, current actives, or price cap). This profile is only a coverage probe; it is not a user or a clinical authorization.

| Input | SHA-256 |
|---|---|
| `recsys/data/derived/catalog_full.json` | `e6c1d00309945aee0752c09f666e2f449d5a64caacaf6676306f583d3f9ba726` |
| `recsys/data/derived/catalog_drug.json` | `ad0e14f586a75af48e157af1daa715ac71898a07bf3b55ed4b2ed8ad3cad1a6e` |
| `recsys/data/verification/approved.json` | `cdfe6433ad5983b0992ca8c7057a75eff270b10779b0e8acd89d247726b9d5bf` |
| `recsys/data/knowledge/concern_actives.json` | `74e2202d03adafccfc263d6747766ef5171fd77626a31a7d9af696c20811d6a7` |
| `recsys/data/knowledge/safety_rules.json` | `c6c18809c6898db32f4d1ebdbb1e2b63dfc90759ab0c5b6ac1229d939c4d49ec` |

Inventory: 1,667 products (1,634 full-catalog rows plus 33 drug rows). The overlay supplied current assertions for 38 products and produced no staleness warning at the audit timestamp.

### Active and regulatory coverage

| Parsed active | Products containing active | Regulatory drug rows | OTC / prescription | Verification status across containing products | OTC rows clearing current neutral-profile treatment gates |
|---|---:|---:|---:|---|---:|
| adapalene | 4 | 4 | 1 / 3 | 4 partial | 0 |
| benzoyl peroxide | 13 | 8 | 2 / 6 | 8 partial, 5 unverified | 1 |
| salicylic acid | 181 | 0 | 0 / 0 | 2 partial, 179 unverified | 0 |
| azelaic acid | 17 | 1 | 0 / 1 | 1 partial, 16 unverified | 0 |
| retinol | 66 | 0 | 0 / 0 | 66 unverified | 0 |
| tretinoin | 11 | 11 | 0 / 11 | 11 partial | 0 |

Interpretation:

- ingredient presence is not proof of an acne-drug indication, strength, exposure, cadence, or contraindication;
- the single gate-clearing OTC treatment is benzoyl peroxide (`P188306`) under the audit profile;
- the catalog’s OTC adapalene/benzoyl-peroxide combination is blocked by missing treatment-role and contraindication verification;
- prescription rows remain discussion options and are never placed into a retail routine role;
- cosmetic retinol is not substituted for adapalene, and cosmetic salicylic-acid presence is not promoted to an OTC acne-drug claim.

### Exact-variant candidate audit

These are the only current exact variants that can fill a proposed retail target under the neutral coverage profile. Every evidence source is an exact SKU manufacturer page or regulatory label preserved in the existing content-addressed verification format and copied into this research package. `partial` is a completeness rank, not clinical approval.

| Product | Proposed exact-label use | Exact relevant facts | Current status | Neutral-profile gate result | Exact-variant evidence |
|---|---|---|---|---|---|
| `P188306` Clinique Acne Solutions All-Over Clearing Treatment Oil-Free | `papule`, `pustule` | Benzoyl peroxide 2.5%; treatment; leave-on lotion; per-label cadence; OTC acne-drug label | partial | eligible | [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/0d0cfec0-5d18-c347-e063-6394a90a912c.xml), snapshot `f81949978b0534ebcd1c56e3a53d1fa5e7b2536aea7bfd9fbcff21e4266e28da` |
| `P456218` Supergoop! Glowscreen Sunscreen SPF 40 | `melasma` | Iron Oxides; SPF 40; broad spectrum; sunscreen role; leave-on cream | partial | eligible | [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/21569637-332a-461b-bf6c-33af6a83a155.xml), snapshot `ec588ae92b30d15eff5ca69d48730131defd6daa6336341ac6f4e43fe8f72093`; [exact manufacturer variant](https://supergoop.com/products/glowscreen-spf-40), snapshot `6042206c437fdd22e2d07ad2ec26af697b741a1d70ff665bb9bde3e63088e92a` |
| `P467976` Supergoop! (Re)setting 100% Mineral Powder Sunscreen SPF 35 | no fill | May contain Iron Oxides (CI 77492); SPF 35; broad spectrum; sunscreen role; leave-on powder | partial | blocked: `format_not_allowed_for_role:powder` | [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/af5c9801-26ce-5c60-e053-2995a90ab457.xml), snapshot `3aed13366b285a59bd1f27227d779473002f093873deb77ff2a8be1d72caddc3`; [exact manufacturer variant](https://supergoop.com/products/mineral-setting-powder), snapshot `92befeb7eb991b2b2815da7803360b18e291330b8f1d9556bb1f820f1acef232` |
| `P476733` Supergoop! Mineral Mattescreen Sunscreen SPF 40 | `melasma` | Iron Oxides CI 77491/77492/77499; SPF 40; broad spectrum; sunscreen role; leave-on lotion | partial | eligible | [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/08671d39-3be2-d666-e063-6294a90a6207.xml), snapshot `293660367c20a287ca9e7983d33ad784a3b3c3ec81e9e651cc9c7ca724d2066e`; [exact manufacturer variant](https://supergoop.com/products/smooth-and-poreless-mattescreen), snapshot `b9f85343d7a3688329a766f1dd33459ae9bebf5aa3bf27243511417f086ed652` |
| `P481169` Tatcha The Silk Sunscreen Mineral Broad Spectrum SPF 50 | `melasma` | Iron Oxides CI 77491; SPF 50; broad spectrum; sunscreen role; leave-on exposure | partial | eligible; format is unknown and therefore lowers completeness but does not veto under D-036 | [DailyMed label](https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/7799042a-d2de-480e-a8a3-827b3dc27907.xml), snapshot `3a1ead63262eccc7983484f4ffcb157801e14338bfe6885dc86d0ff58a6ff5ba`; [exact manufacturer variant](https://tatcha.com/products/the-silk-sunscreen-spf-50), snapshot `2a8616e3ec997ecc1be5c8e375663f3eb065c802e05243f2060656587a1c98d6` |

### Iron-oxide tinted sunscreen coverage

Deterministic matching used `iron oxide`, `iron oxides`, `ferric oxide`, `ferrous oxide`, or CI 77491/77492/77499 in normalized INCI text.

- 34 sunscreen-category products contain an iron-oxide token;
- 31 report SPF 30 or greater;
- 4 have verified SPF, broad-spectrum status, sunscreen role, and leave-on exposure;
- 3 of those 4 clear the current neutral-profile hard gates;
- one verified powder sunscreen is rejected because powder is not an allowed primary sunscreen format;
- all four evidence-complete candidates are still `partial`, not `verified`, under the current completeness rubric.

This is enough catalog presence to test an approved melasma sunscreen path later, but it is not permission to expose that path now.

### Silicone scar-care coverage

Three product names contain “scar,” but none has a dedicated scar-care evidence role or exact, reviewed silicone-sheet/silicone-gel formulation evidence. This proposed evidence role does not change the repository's closed routine-role vocabulary:

- `P377368` D-Scar Scar Diminishing Serum — allantoin and glycerin;
- `P446423` InvisiScar Post-Acne Resurfacing Treatment — salicylic acid and vitamin C;
- `P483128` Scar Gel Treatment — allantoin and glycerin.

The catalog contains many products with vehicle silicones such as dimethicone. Counting those as scar treatments would be a false inference. Dedicated verified silicone scar-care coverage is therefore **zero**.

### Coverage conclusion

Current coverage is incomplete but usable as an honest constraint. Verification status is a ranking signal, not a blanket exclusion. Hard safety, exact-active, strength, role, exposure, cadence, contraindication, and conflict facts remain mandatory. `label_source` and `label_verified_at` alone affect rank; a regulatory label becomes mandatory when the reviewed therapeutic intent specifically claims OTC or prescription drug status. An unknown intended area or format lowers completeness, while an explicitly non-face area or known incompatible format vetoes. Unfilled paths must be surfaced as gaps, not silently replaced.

| Exact detector label | Coverage status | Safe current catalog coverage |
|---|---|---|
| `closed_comedo` | `unfilled` | No OTC adapalene or exact salicylic-acid acne-drug row clears the current treatment gates. |
| `open_comedo` | `unfilled` | No OTC adapalene or exact salicylic-acid acne-drug row clears the current treatment gates. |
| `papule` | `partial_coverage_pending_clinical_approval` | `P188306`, benzoyl peroxide 2.5%; one exact variant. |
| `pustule` | `partial_coverage_pending_clinical_approval` | `P188306`, benzoyl peroxide 2.5%; one exact variant. |
| `nodule` | `unfilled_by_policy` | Clinician-first; no retail target is allowed. |
| `atrophic_scar` | `unfilled_by_policy` | Clinician-first and procedure discussion; no retail target is allowed. |
| `hypertrophic_scar` | `unfilled` | Zero dedicated, verified silicone sheet or gel products. |
| `melasma` | `partial_coverage_pending_clinical_approval` | Three exact iron-oxide SPF variants: `P456218`, `P476733`, `P481169`. |
| `nevus` | `unfilled_by_policy` | Monitoring or referral only; no product target is allowed. |
| `other` | `unfilled_by_policy` | Unsupported; no product target is allowed. |

## 7. Independent dermatologist review packet

The reviewer must independently confirm or reject every item below and record exceptions:

1. finding wording and photo-only limits for each of the 10 labels;
2. lookalikes, red flags, referral urgency, and stop/escalate conditions;
3. every retail active, strength, formulation, exposure, cadence statement, age boundary, pregnancy/nursing rule, allergy rule, interaction rule, and duplicate-active rule;
4. whether `closed_comedo` and `open_comedo` may share the proposed active candidates while retaining distinct findings;
5. whether papule and pustule can share the proposed inflammatory path;
6. whether nodules, atrophic scars, and raised scars must remain clinician-first;
7. whether silicone may ever be a retail match and the minimum confirmation, wound, product-evidence, duration, and stop requirements;
8. whether melasma can produce an iron-oxide sunscreen match before clinical confirmation and which pigment red flags force referral;
9. absolute prohibition of mole-removal and `other`-label treatment matching;
10. user-facing uncertainty, review-period, and reason-code wording;
11. the proposed intake contract and all “unknown means defer” decisions;
12. the source set, extraction accuracy, policy transformations, US applicability, and unresolved uncertainties.

Approval is valid only when the record contains reviewer identity, credentials, affiliation, date, accepted report SHA-256, accepted policy version, and exceptions. A partial approval must identify the exact accepted and rejected label paths; rejected or unreviewed paths remain disabled.

## 8. Unresolved questions for the reviewer

- Should nursing be an automatic retail-retinoid exclusion or a clinician-discussion deferral?
- May an exact salicylic-acid OTC acne drug be used for both open and closed comedones, and must the application restrict it to leave-on formulations?
- Is a single benzoyl-peroxide path appropriate for both papules and pustules without a clinician-assessed severity measure?
- Which symptoms or lesion counts should change referral urgency without implying that the image model assessed acne severity?
- Can photoprotection be shown for a probable melasma label before diagnosis, or should any new pigment finding be clinician-first?
- What constitutes adequate “clinician-confirmed raised scar” evidence for a later silicone product path?
- Which time horizon and outcome wording are acceptable for acne, pigment, and scar follow-up?
- Are any additional skin-of-color cautions required for irritation, post-inflammatory hyperpigmentation, or procedures?

## 9. Phase 0 acceptance result

| Acceptance item | Result |
|---|---|
| All 10 labels researched separately | Pass |
| Clinical evidence, safety policy, and product evidence separated | Pass |
| Exact source snapshots and hashes captured | Pass |
| Proposed machine-readable 10-label policy produced | Pass |
| Catalog coverage measured without gap substitution | Pass |
| Independent dermatologist approval recorded | **Fail — pending** |
| Production policy or recommendation behavior changed | **No — correctly blocked** |

The next authorized step is independent clinical review of this packet. Production implementation remains blocked until that review is complete and recorded.
