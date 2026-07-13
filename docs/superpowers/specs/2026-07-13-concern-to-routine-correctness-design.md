# Concern-to-routine correctness — audit and remediation design

Date: 2026-07-13. Status: repository-controlled P0/P1 implementation complete;
release blocked on the external gates listed below. Proposed originally from
an audit of every available SA-RPN v2
E2E run. This document is a release-correctness spec, not a claim that a photo
can diagnose acne or replace a clinician.

## Implementation note (2026-07-13)

The v3 decision, therapy-policy, catalog verification/quarantine, hard
eligibility, eligible-only ranking, one-per-role composition, whole-regimen
validation, schema-3 artifact/provenance, release evaluation, and resumable
batch seams are implemented with public regression tests. The audit evidence
and numbers below are retained unchanged as the reason for the remediation.

Implementation completion is not clinical approval or release eligibility.
The repository still lacks qualified clinician policy approval, an adequate
calibration cohort, an external clinician-reviewed set, a verified real catalog
overlay, and immutable remote detector identity. Synthetic test policy/catalog
fixtures prove only code behavior. Release tooling therefore reports these
missing inputs as blockers rather than weakening or fabricating a gate.

## Executive verdict

The current recommendation mechanism is internally deterministic, but the
result is not yet a trustworthy acne regimen. The main problem is not one bad
product. Errors compound across five stages:

1. detector labels and counts are treated as clinical facts;
2. a count-derived severity of 4 is coupled to `soothe_escalation`, even when
   no nodule is present;
3. candidates qualify when they match **any** target or support active;
4. pooled product ratings and popularity rank candidates, not acne efficacy or
   suitability for the proposed use;
5. five products in each category are serialized as a routine instead of one
   regimen plus clearly separated alternatives.

On the 18 unique images represented by the available run artifacts:

- only 2 images are in the deterministic seed-42 validation split; the other
  16 are detector-training images, so the runs cannot serve as an honest
  generalization evaluation;
- at IoU 0.5, predictions have 52.6% geometry precision, 70.6% geometry
  recall, and 91.7% label accuracy among geometry-matched detections;
- nodule recall is 4/13 (30.8%), while one non-nodule image is incorrectly
  escalated by two predicted nodules;
- the stored recommendation mode agrees with the independent disposition in
  9/18 cases (50%);
- 12/18 stored routine artifacts (66.7%) no longer match a replay through the
  current code;
- the batch log records 30 attempts, 15 failed attempts, and 5 of 19 requested
  images without a logged completion. Failed batches restart work rather than
  checkpointing per image.

The release decision is therefore **do not present the current output as a
personalized acne treatment routine**. It may remain an experimental cosmetic
decision-support output while the P0 gates in this spec are implemented and
clinically reviewed.

## Evidence boundary

This audit uses the AcneSCU XML annotations as dataset ground truth for model
evaluation, not as a medical diagnosis of the photographed people. Product
recommendations below express a regimen pattern. They do not establish that a
specific user can safely use a drug or cosmetic.

The clinical baseline is deliberately narrow and comes from authoritative
guidance:

- The [American Academy of Dermatology acne guideline](https://www.aad.org/member/clinical-quality/guidelines/acne)
  recommends benzoyl peroxide, topical retinoids, salicylic acid, and azelaic
  acid, and recommends combining treatments with different mechanisms when
  appropriate.
- [NICE NG198](https://www.nice.org.uk/guidance/ng198/chapter/Recommendations)
  recommends a non-alkaline syndet cleanser, avoiding oil-based or comedogenic
  skin-care products, a defined 12-week first-line course, and referral for
  nodulocystic acne. It also says to consider referral for scarring or
  persistent pigment change and to treat ongoing acne to prevent further
  scarring.
- The [FDA/DailyMed adapalene 0.1% and benzoyl peroxide 2.5% label](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=fc8e868f-9699-4ae6-83d5-fa27789336cd)
  specifies a thin once-daily application, sun protection, irritation
  precautions, and avoidance of additional drying or irritating products.

These sources support product classes and safety constraints. They do not
support centella, sheet masks, clay masks, neck products, anti-aging oils, or
trace-active cleansers as substitutes for first-line acne treatment.

## What was audited

### Artifacts

- 19 primary run directories representing 18 unique source images; the two
  smoke runs reuse one image.
- 3 partial update reruns under `runs/e2e/updates_results`.
- `analysis.json`, `routine.json`, collages, lesion sheets, batch logs, and the
  matching AcneSCU VOC XML annotations.
- current concern construction, recommendation engine, catalog importer,
  pooled-statistics ranker, E2E writer, and relevant tests.

### Method

1. Reconstruct the deterministic seed-42 AcneSCU split used by the training
   preparation script.
2. Greedily pair predicted and annotated boxes one-to-one by IoU and inspect
   unmatched predictions, misses, and class confusion.
3. Rebuild concern counts from the XML annotations and compare the resulting
   safety decision with the stored prediction-derived decision.
4. Independently adjudicate the routine disposition:
   - any annotated nodule: prompt dermatology review and only a minimal
     supportive routine pending that review;
   - no annotated nodule: treat active acne with a proven topical option when
     user safety inputs permit, and add rather than substitute a referral when
     scarring, pigment change, or severity warrants it.
5. Replay each stored recommendation through current code and compare its
   semantic fields.
6. Inspect the actual products admitted and ranked for both recommendation
   modes, including their catalog category and raw ingredients.

## Per-run decision audit

`C/I/N/S/P` means comedonal, inflammatory, nodule, scar, and pigment counts.
Counts are not clinical grades; they are included to expose how detector error
propagates into the decision.

| Run | Split | GT C/I/N/S/P | Pred C/I/N/S/P | Stored mode | Independent disposition |
|---|---|---:|---:|---|---|
| sarpn-v2-random-120 | TRAIN | 28/17/0/7/1 | 37/30/2/18/3 | soothe | Treat active acne; referral if indicated |
| sarpn-v2-random-146 | TRAIN | 36/31/0/17/19 | 45/40/0/26/15 | soothe | Treat active acne; referral if indicated |
| sarpn-v2-random-147 | TRAIN | 33/24/0/9/58 | 34/21/0/17/160 | treatment | Treat active acne; referral if indicated |
| sarpn-v2-random-148 | TRAIN | 47/4/0/3/6 | 46/7/0/12/15 | soothe | Treat active acne; referral if indicated |
| sarpn-v2-random-15 | TRAIN | 11/15/0/6/28 | 17/15/0/8/31 | treatment | Treat active acne; referral if indicated |
| sarpn-v2-random-166 | TRAIN | 18/26/0/38/4 | 29/36/0/31/5 | soothe | Treat active acne; referral if indicated |
| sarpn-v2-random-17 | TRAIN | 9/11/4/20/0 | 10/16/1/50/15 | soothe | Derm-first supportive pending review |
| sarpn-v2-random-197 | VALID | 41/70/4/14/15 | 57/70/1/33/15 | soothe | Derm-first supportive pending review |
| sarpn-v2-random-230 | VALID | 23/12/1/13/10 | 27/13/0/19/18 | treatment | Derm-first supportive pending review |
| sarpn-v2-random-231 | TRAIN | 25/7/0/20/4 | 25/9/0/16/9 | treatment | Treat active acne; referral if indicated |
| sarpn-v2-random-247 | TRAIN | 50/47/0/12/11 | 66/62/1/24/21 | soothe | Treat active acne; referral if indicated |
| sarpn-v2-random-252 | TRAIN | 25/32/1/10/2 | 38/40/0/13/2 | soothe | Derm-first supportive pending review |
| sarpn-v2-random-262 | TRAIN | 15/15/0/1/25 | 21/13/0/5/43 | treatment | Treat active acne; referral if indicated |
| sarpn-v2-random-274 | TRAIN | 25/14/1/39/18 | 43/23/0/75/27 | soothe | Derm-first supportive pending review |
| sarpn-v2-random-45 | TRAIN | 27/13/0/26/10 | 20/15/0/24/11 | soothe | Treat active acne; referral if indicated |
| sarpn-v2-random-73 | TRAIN | 140/27/0/16/42 | 179/32/0/52/42 | soothe | Treat active acne; referral if indicated |
| sarpn-v2-random-76 | TRAIN | 12/15/2/53/43 | 26/22/2/45/30 | soothe | Derm-first supportive pending review |
| sarpn-v2-smoke-2 | TRAIN | 21/30/0/3/0 | 52/36/0/7/2 | soothe | Treat active acne; referral if indicated |

The most important counterexamples are:

- `random-120`: two false nodule predictions suppress active treatment;
- `random-230`: an annotated nodule is labeled as a papule and the routine
  remains in treatment mode;
- `random-252` and `random-274`: annotated nodules are also labeled as
  papules, although the count-derived severity happens to trigger soothing;
- `random-147`: predicted pigment burden is almost three times the annotation
  count, demonstrating that a plausible-looking severity can still be badly
  calibrated;
- `smoke-2`: the dataset's `other` class is never emitted by the model and is
  partly absorbed into acne classes.

## Detector and concern findings

### Evaluation contamination

The available E2E directory is mostly an in-sample demonstration. Sixteen of
18 unique images occur in detector training. No release metric may aggregate
these images with held-out images. The two validation images are useful
counterexamples, but too small to certify a safety-sensitive classifier.

### IoU 0.5 results across the 18 unique images

| Quantity | Result |
|---|---:|
| Annotated lesions | 1,755 |
| Predictions | 2,354 |
| Correct-label matches | 1,136 |
| Wrong-label geometry matches | 103 |
| Missed annotations | 516 |
| Unmatched predictions | 1,115 |
| Geometry recall | 70.6% |
| Geometry precision | 52.6% |
| Label accuracy among matched geometry | 91.7% |

The precision result matters as much as recall because counts drive severity,
affected-region floors, treatment targets, and referral language. In addition,
943/2,354 predictions (40.1%) required forced region assignment, so region
coverage should not currently be treated as high-confidence evidence.

### Class-level same-label recall

| Class | Correct / GT | Recall |
|---|---:|---:|
| atrophic scar | 190/293 | 64.8% |
| closed comedo | 266/398 | 66.8% |
| hypertrophic scar | 10/14 | 71.4% |
| melasma | 178/296 | 60.1% |
| nevus | 80/97 | 82.5% |
| nodule | 4/13 | 30.8% |
| open comedo | 114/188 | 60.6% |
| other | 0/46 | 0.0% |
| papule | 260/361 | 72.0% |
| pustule | 34/49 | 69.4% |

`nodule` is the weakest safety-relevant class. The current rule gives one
detector label the power to suppress treatment, but the detector both misses
and invents that label. The correct response is a calibrated triage decision
with an uncertainty/abstention path, not a lower threshold applied directly
to raw detections.

### Saturated nevus flag

All 18 images produce a `nevus_observation` professional-review flag. The
rule fires when either the count or maximum score crosses its threshold, so a
small number of high-score detections is sufficient. This may be a defensible
safety posture, but a flag with 100% prevalence on this sample is not
discriminating. It must be separately validated and clearly described as
observational, not diagnostic.

## Recommendation findings

### 1. Referral and treatment are incorrectly made mutually exclusive

`soothe_escalation` activates when `has_cystic` is true **or** overall
severity reaches 4. It removes evidence-based acne actives and selects support
ingredients such as centella, ceramides, and hyaluronic acid.

That is conservative for genuinely nodulocystic or diagnostically uncertain
presentations. It is not logical for every high count or scar-heavy image.
Guidance supports treating ongoing acne to reduce future scarring while also
referring when appropriate. The output needs two independent axes:

- `triage_level` and `referral_reasons`;
- `therapy_disposition` and an eligible treatment plan.

### 2. Active union is not product suitability

A product currently qualifies if it contains any target active. Support
ingredients are common, so they admit products that do not deliver the core
therapy. Examples from current replay include:

- clay masks and sheet masks ranked as `treatment`;
- anti-aging oil-serums admitted for hydrating/support ingredients;
- a decollete/neck serum mapped to `moisturizer`;
- trace-salicylic cleansers outranking leave-on acne treatment;
- a benzoyl-peroxide lotion surviving removal of benzoyl peroxide from the
  target set because the same product also contains ceramides.

Matching must be product-role-aware and must inspect every carried active, not
just the active that admitted the product.

### 3. The ranker optimizes reviews, not acne outcomes

`StatsRanker` is pooled Bayesian rating plus a popularity nudge. It cannot
answer whether a product is effective for comedonal, inflammatory, or pigment
concerns. In a treatment replay, a direct 10% azelaic-acid product is around
41st among serum candidates and is appended only by the current coverage
repair. Salicylic masks, scrubs, and peels can rank highly because customer
ratings are not a measure of daily acne-regimen suitability.

The existing concern-efficacy design remains useful, but outcome evidence must
operate only after a hard role, format, concentration, intended-area, and
safety eligibility gate.

### 4. A product menu is serialized as a routine

Five products are emitted in each of five broad categories. A user can
reasonably read that as 25 products to stack. The recent coverage promotion can
also append a sixth product after the top-five truncation. This is unsafe and
operationally ambiguous.

The selected regimen must contain at most one product per role. Alternatives
must be separate and mutually exclusive. Cadence, amount, ramp-up, AM/PM
placement, conflicts, and review timing must be machine-readable.

### 5. Required safety context is absent

The output silently defaults skin type and does not require age, pregnancy or
trying-to-conceive status, allergies, eczema/rosacea or sensitivity, current
actives and medications, treatment history, duration, pain, or prior scarring.
Without these inputs the system cannot safely choose a retinoid, avoid
duplicate irritants, or know whether self-care has already failed.

## What this system should recommend

### When a nodule/cystic presentation is credible or the model abstains

Recommend prompt professional assessment. Until that assessment, recommend
only a minimal supportive regimen:

1. gentle non-alkaline facial cleanser;
2. noncomedogenic moisturizer if needed;
3. broad-spectrum SPF 30 or higher;
4. avoid picking, scrubs, peels, and stacking new irritants.

Do not substitute five centella-containing products for referral. Do not
automatically start or stop a medicine based only on a photo.

### When active acne is present without the derm-first gate

After collecting the safety profile, compose one treatment course rather than
a shopping list:

1. gentle cleanser;
2. **one** proven treatment path, such as a properly labeled adapalene /
   benzoyl-peroxide product where appropriate, benzoyl peroxide alone when a
   retinoid is unsuitable, or azelaic acid as an alternative matching the
   concern and safety profile;
3. noncomedogenic moisturizer;
4. broad-spectrum SPF 30 or higher.

Salicylic acid may be an eligible alternative, but its presence does not make
a rinse-off cleanser, scrub, peel, or clay mask interchangeable with a
leave-on treatment. Product identity, concentration, exposure, and label
directions must control that decision.

### Product-level audit disposition

Some catalog products could be eligible alternatives after label and profile
validation. Concrete candidates worth validating are Paula's Choice `CLEAR
Daily Skin Clearing Treatment with 2.5% Benzoyl Peroxide`, Paula's Choice `10%
Azelaic Acid Booster`, The INKEY List `SuperSolutions 10% Azelaic Serum Redness
Relief Solution`, and The Ordinary `Azelaic Acid 10% Suspension Brightening
Cream`. Their names and catalog actives are much closer to the intended
therapy than the masks and support serums that currently outrank them. This is
a shortlist for eligibility review, not an instruction that every user should
use any of them; concentration, full formula, label, age, pregnancy status,
current regimen, sensitivity, and intended use still have to pass the gates.

The audit does **not** approve any current product for every user. The
following current candidate types should be rejected or demoted before
ranking:

- masks, scrubs, peels, cleansing balms, and makeup removers as the core acne
  treatment;
- neck/decollete products for facial moisturizer slots;
- anti-aging products admitted only by a support ingredient;
- products whose therapeutic active or concentration is not verified;
- products with a carried conflicting or contraindicated active;
- duplicate treatment steps and multiple alternatives serialized as selected.

## Target architecture

```text
image evidence
  -> calibrated concern evidence + uncertainty
  -> triage/referral decision
  -> therapy plan (active class, strength band, exposure, cadence)
  -> hard catalog eligibility
  -> concern-efficacy ranking among eligible equivalents
  -> one-per-role regimen composer
  -> whole-regimen safety validator
  -> versioned artifact + explanation
```

Ranking is intentionally late. It must never repair a bad clinical decision or
make an ineligible product eligible.

### Decision model

Replace the binary `treatment` / `soothe_escalation` coupling with:

```json
{
  "triage_level": "routine|routine_plus_review|derm_first|abstain",
  "referral_reasons": ["suspected_nodule", "scarring_risk"],
  "therapy_disposition": "active_treatment|supportive_only|maintenance|defer",
  "decision_evidence": [
    {
      "concern": "acne_nodular",
      "probability": 0.71,
      "quality": "low",
      "source": "detector_v3"
    }
  ]
}
```

Rules:

- credible nodule/cystic evidence or unresolved high-risk uncertainty leads to
  `derm_first` or `abstain` and `supportive_only`;
- count severity alone cannot suppress all active treatment;
- scarring or persistent pigment concern may add `routine_plus_review` while
  active acne still receives an eligible treatment plan;
- detector confidence is calibrated against the downstream triage decision,
  not interpreted as a probability by default;
- no diagnosis or malignancy claim is emitted from a photo.

### Therapy plan

The decision layer outputs therapeutic intent, not ingredient keywords:

```json
{
  "course_weeks": 12,
  "review_at_weeks": 12,
  "primary": {
    "therapy": "azelaic_acid",
    "strength_band": "verified_otc_or_labeled",
    "exposure": "leave_on",
    "cadence": "per_label",
    "role": "treatment"
  },
  "alternatives": [
    {"therapy": "benzoyl_peroxide", "reason": "retinoid_not_selected"}
  ],
  "support_roles": ["cleanser", "moisturizer", "sunscreen"]
}
```

Exact drug instructions must come from verified labeling and be constrained by
age, pregnancy/trying-to-conceive state, current medicines, sensitivity, and
clinician-reviewed policy. Recommendation prose must not invent directions.

### Catalog contract v2

Each recommendable SKU requires these verified fields:

```json
{
  "intended_area": ["face"],
  "routine_roles": ["treatment"],
  "format": "gel",
  "exposure": "leave_on",
  "drug_actives": [{"name": "azelaic_acid", "strength": "10%"}],
  "otc_drug": false,
  "label_source": "manufacturer_or_regulator_url",
  "label_verified_at": "2026-07-13",
  "broad_spectrum": null,
  "spf": null,
  "comedogenic_claim": "unknown",
  "irritant_features": [],
  "contraindications": [],
  "evidence_roles": ["acne_treatment_alternative"],
  "evidence_grade": "guideline_class_plus_verified_product_form"
}
```

Additional requirements:

- distinguish active concentration from mere ingredient-list presence;
- distinguish face from neck, body, lip, eye, or makeup use;
- distinguish rinse-off, short-contact, leave-on, mask, scrub, and peel;
- store SPF value and broad-spectrum status for sunscreen eligibility;
- retain a source and verification timestamp for medical-label fields;
- use `unknown` rather than infer absence for fragrance, comedogenicity, or
  contraindications;
- quarantine products missing a hard field from the corresponding role.

### Hard eligibility

Before scoring, a product must pass all constraints for its proposed role:

1. correct intended area and allowed format;
2. correct exposure and cadence class;
3. direct match to the therapy plan, including verified strength where
   required;
4. all carried actives checked for contraindication, duplication, and
   conflict;
5. profile constraints checked;
6. sunscreen has verified broad-spectrum SPF 30+;
7. no unresolved required field.

Support actives may explain or break ties among eligible support products.
They may not admit a product to a therapeutic role.

### Ranking

Rank only products that are therapeutically interchangeable within a role.
The score order is:

1. hard safety and role eligibility (veto, not score);
2. concern-specific outcome evidence with minimum sample and provenance;
3. tolerability/profile fit;
4. evidence quality and data completeness;
5. user budget and preference;
6. pooled rating/popularity only as a final tie-breaker.

Never allow a review score to overpower an intended-area, format, strength,
or safety constraint. If concern-specific evidence is unavailable, say so and
fall back transparently; do not describe pooled ratings as acne efficacy.

### Composer and validator

The composer returns one selected product per role and separate alternatives:

```json
{
  "selected_regimen": {
    "am": ["cleanser", "moisturizer_if_needed", "sunscreen"],
    "pm": ["cleanser", "treatment", "moisturizer_if_needed"]
  },
  "selected_products": {
    "cleanser": "sku-a",
    "treatment": "sku-b",
    "moisturizer": "sku-c",
    "sunscreen": "sku-d"
  },
  "alternatives": {
    "treatment": ["sku-e", "sku-f"]
  }
}
```

The final validator rejects the artifact if:

- a required therapy role is missing without an explicit reason;
- more than one product is selected for a role;
- an alternative appears in the selected regimen;
- a product is placed outside its intended area or role;
- treatment strength/exposure is unverified;
- products duplicate or conflict through any carried active;
- a profile contraindication is violated;
- a mask, scrub, or peel is scheduled as a daily leave-on treatment;
- the explanation promises an active that the selected product does not
  deliver.

## E2E artifact contract

Every `analysis.json` and `routine.json` must include:

```json
{
  "schema_version": "3",
  "generated_at": "RFC-3339 timestamp",
  "source_image_sha256": "...",
  "dataset": {"name": "AcneSCU", "sample_id": "...", "split": "valid"},
  "code": {"git_commit": "...", "dirty": false},
  "models": {"detector_sha256": "...", "classifier_sha256": "..."},
  "config_sha256": "...",
  "catalog_sha256": "...",
  "ranker_sha256": "...",
  "input_profile": {"...": "explicit values or unknown"},
  "replay_key": "hash of all semantic inputs"
}
```

Artifacts with a mismatched replay key are stale and must not be mixed into an
evaluation. Dirty-code runs may be used for local debugging but must be marked
and excluded from a release report.

## E2E harness and release metrics

### Evaluation sets

Maintain three explicit sets:

1. **component validation:** held-out AcneSCU images only;
2. **decision counterfactual:** the same held-out images run twice, once from
   predictions and once from the actual AcneSCU VOC XML annotation path; the
   XML identity is hashed into replay and the oracle path skips detector HTTP;
3. **external clinical review set:** consented, representative consumer photos
   with clinician-reviewed triage and regimen-disposition labels.

Training images may remain visual smoke tests but may not contribute to a
release metric. Split membership must be asserted before execution.

### Required metrics

- detector precision/recall per class and operating point;
- nodule triage sensitivity, specificity, positive predictive value, and
  abstention rate;
- prediction-versus-oracle triage confusion;
- prediction-versus-clinician disposition agreement;
- referral-reason precision/recall;
- therapy-plan agreement and target coverage;
- product-role precision;
- contraindication/conflict violations;
- selected-product count per role;
- explanation-to-product consistency;
- artifact replay equality and freshness;
- per-image attempt count, failure class, latency, and completion rate.

Aggregate lesion accuracy is not a proxy for the safety decision. Report all
safety-critical metrics with sample counts and confidence intervals.

### Regression fixtures

Pin at least these current failures:

- false nodule escalation: `random-120`;
- missed nodule under-triage: `random-230`;
- missed nodules masked by count severity: `random-252`, `random-274`;
- pigment overcount: `random-147`;
- unsupported `other` absorption: `smoke-2`;
- BP carried through a non-BP support-active match;
- direct azelaic treatment displaced by masks/support serums;
- neck serum admitted as facial moisturizer;
- coverage promotion exceeding the role limit;
- stale artifact rejected by replay key.

## Batch reliability

Make the E2E runner an idempotent per-image job:

- bounded retry with exponential backoff and jitter for transient model/API
  errors;
- persist a manifest row after each stage and image;
- resume from the last verified stage without rerunning completed images;
- atomic artifact writes followed by schema validation;
- stable run ID plus per-attempt ID;
- explicit terminal states: `complete`, `retryable_failed`,
  `permanent_failed`, `stale`;
- batch exit non-zero if any requested image lacks a terminal artifact;
- summary includes requested, completed, failed, retried, and skipped counts.

## Delivery plan

### P0 — release blockers

1. **Honest evaluation and provenance**
   - add split assertions and artifact hashes to `src/pipeline/e2e.py`;
   - reject mixed/stale runs in the evaluation CLI;
   - regenerate the audit on held-out and external sets.
2. **Separate triage from treatment**
   - replace the severity-4 blanket in `src/recommendation/engine.py` with the
     decision model above;
   - add abstention and independent referral reasons;
   - have the policy and thresholds reviewed by a qualified clinician.
3. **Safety-calibrate nodule routing**
   - calibrate the operating point on held-out data;
   - evaluate the downstream triage decision, not just box mAP;
   - route uncertain high-risk cases to abstention.
4. **Role-based single regimen**
   - implement one selected product per role and separate alternatives in
     `src/recommendation/schema.py` and `engine.py`;
   - add the whole-regimen validator.
5. **Close the carried-active safety hole**
   - validate all actives on every admitted SKU after target de-stacking;
   - prohibit a removed or contraindicated active from re-entering through a
     support-active match.

### P1 — recommendation quality and operability

1. Extend `src/recommendation/import_catalog.py` and schema with role, area,
   format, exposure, strength, label, sunscreen, and verification fields;
   rebuild and quarantine incomplete SKUs.
2. Admit products by therapy role, not active union; reject masks, peels,
   scrubs, and non-face products from core facial roles.
3. Put the concern-efficacy scorer behind hard eligibility, then bake it off
   against the pooled ranker on held-out outcome evidence.
4. Add the required user safety intake and preserve `unknown` rather than
   silently defaulting `skin_type=combination`.
5. Add machine-readable cadence, amount source, ramp-up, conflict, and review
   timing.
6. Add per-image retry, checkpoint, resume, and atomic validation to the E2E
   runner.

### P2 — follow-up and personalization

1. Budget-aware alternatives among clinically equivalent products.
2. Longitudinal follow-up at the policy-defined review interval, including
   tolerability and worsening.
3. Clinician adjudication tooling for abstained or discordant cases.
4. Post-market monitoring for catalog drift, label changes, and product
   discontinuation.

## Acceptance gates

P0 is complete only when all of the following hold:

- 100% of release-evaluation images are declared held-out or external; zero
  detector-training images contribute to reported performance;
- every artifact contains the required provenance and replays byte-identically
  in semantic fields from its replay key;
- a clinician approves the triage labels, referral rules, treatment
  dispositions, and numeric nodule operating gate before a release threshold
  is frozen;
- the frozen held-out set meets that clinician-approved nodule sensitivity,
  positive-predictive-value, and abstention gate, with sample counts and
  confidence intervals reported;
- prediction-to-clinician disposition agreement meets the frozen gate and no
  safety-critical disagreement is hidden by aggregate accuracy;
- count-derived severity without high-risk evidence cannot by itself force
  supportive-only care;
- every selected treatment has verified area, role, format, exposure, active,
  and where applicable strength and label source;
- exactly zero selected products violate profile constraints, carried-active
  constraints, intended area, or role;
- at most one product is selected per role and all alternatives are outside the
  selected regimen;
- all 18 current unique cases are replayable with zero unmarked artifact drift;
- a batch can resume after an injected transient failure and finishes with an
  accurate terminal status for every requested image.

Do not invent a nodule sensitivity target from this 18-image audit. The
operating gate is a clinical risk decision and needs enough held-out positive
and negative cases to estimate it honestly.

## Test plan

### Unit tests

- decision table for nodule, high-count non-nodule, scarring, pigment,
  uncertainty, and clear-skin cases;
- product eligibility matrix by role/area/format/exposure/strength/profile;
- all-carried-active conflict and contraindication tests;
- one-per-role composer and alternatives separation;
- validator rejects every enumerated invalid state;
- provenance/replay hash stability and stale detection;
- retry classification, checkpoint, and resume.

### Integration tests

- prediction-derived versus XML-oracle counterfactual on held-out fixtures;
- catalog import to therapy plan to selected regimen with verified sample
  products;
- concern scorer cannot admit a hard-ineligible product regardless of score;
- generated explanation exactly reflects the selected SKU and verified label;
- interrupted batch resumes without recomputing completed image stages.

### Release report

Generate one versioned report containing cohort declaration, split proof,
model/config/catalog hashes, class metrics, triage metrics, disposition
confusion, product-role violations, artifact freshness, and batch completion.
No manually copied metrics and no unversioned collages count as a release gate.

## Current repository verification

At audit time, recommendation-focused tests pass: 178 passed. The full suite
reports 297 passed, 1 deselected, and 2 failed because TensorFlow is not
installed and `predict_batch` imports it before the empty-input and fake-model
test paths. Those two environment/import failures are separate from this
recommendation audit and should not be represented as recommendation
regressions.

## Supersession and compatibility

This spec narrows and strengthens the earlier concern-efficacy recommender
design. Concern-efficacy evidence remains valuable, but it cannot admit
products ahead of role and safety validation. Existing recommendation JSON is
v2 input only; the v3 output is intentionally not byte-compatible because the
old five-products-per-category shape is ambiguous. A migration reader may
display old runs, but it must label them `legacy` and must not compare them to
v3 without regeneration.
