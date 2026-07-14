# Concern-to-routine correctness v3 implementation plan

**Spec:** `docs/superpowers/specs/2026-07-13-concern-to-routine-correctness-design.md`
**Repository baseline:** `3bed21d` (`main`)
**Plan date:** 2026-07-13
**Execution style:** vertical red-green-refactor slices, then a two-axis standards/spec review
**Primary objective:** implement every repository-controlled P0 and P1 requirement in the spec without inventing clinical approval, calibration results, verified labels, or held-out/external evidence that the repository does not contain.

## Outcome

Replace the current count-coupled, active-union, multi-product menu with a versioned v3 pipeline that:

1. represents triage/referral independently from therapy disposition;
2. abstains when a safety-critical detector signal is not calibrated by an approved policy;
3. turns therapeutic intent into a role-aware plan;
4. admits products only after hard area, role, format, exposure, strength, label, sunscreen, profile, and carried-active checks;
5. ranks only eligible, therapeutically interchangeable candidates;
6. selects at most one product per role and keeps alternatives separate;
7. validates the whole regimen before serialization;
8. writes reproducible v3 artifacts with hashes and a replay key;
9. rejects training, dirty, stale, or mixed artifacts from release evaluation;
10. supports bounded retry, per-stage checkpointing, resume, and accurate batch terminal states.

The implementation must remain honest when required information is missing. An empty or deferred treatment role with a machine-readable reason is correct; silently falling back to an unverified product is not.

## Baseline and working-tree constraints

At plan creation:

- Recommendation/E2E focused baseline: `168` targeted tests pass (`test_recommendation_engine`, `test_import_catalog`, `test_e2e`, `test_sarpn`).
- The spec itself is untracked and is in scope.
- `src/pipeline/e2e.py` and `tests/test_e2e.py` contain an uncommitted coverage-promotion change. Preserve its intent as regression evidence, but v3 must supersede the behavior because promotion can exceed the one-product-per-role limit.
- `runs/e2e/batch-2026-07-13.log`, `.claude/`, `AcneSCU.v1-acnescu-original.voc/`, and untracked run directories are pre-existing user artifacts and must not be edited, deleted, staged, or committed.
- Do not reset, restore, checkout, clean, or otherwise discard pre-existing work.
- Use `apply_patch` for source/document edits. Formatting tools may perform mechanical rewrites.
- Stage only files that are part of this plan.

Before editing, record:

```bash
git status --short
git diff -- src/pipeline/e2e.py tests/test_e2e.py
git rev-parse HEAD
```

Use the recorded `HEAD` as the fixed review point. The final diff against it is expected to include the already-started in-scope E2E test/change after it has been reconciled with v3.

## Non-negotiable safety and truthfulness rules

- Never treat detector confidence as a calibrated probability. A probability field may be populated only by a named calibrator/policy that marks the value calibrated.
- Never invent a nodule threshold, sensitivity target, PPV target, clinical label, label direction, drug strength, sunscreen claim, contraindication, or product verification timestamp.
- High count/severity without high-risk evidence may add review/referral language but cannot by itself force `supportive_only`.
- Unreviewed or unresolved safety-critical nodule evidence routes to `abstain` plus `supportive_only`, not to a photo-derived diagnosis.
- Referral and active treatment are separate axes. Scarring/pigment concerns may add `routine_plus_review` without suppressing an otherwise eligible therapy plan.
- Unknown is data. Do not silently default skin type, pregnancy status, age, current actives, medications, allergies, sensitivity conditions, treatment history, duration, pain, or prior scarring.
- Hard eligibility is a veto. Ranker/scorer output cannot admit or repair an ineligible product.
- Every carried active is evaluated, including actives unrelated to the match that admitted the product.
- No medicine directions may be synthesized from ingredient names. Cadence, amount, ramp-up, and review language must name a verified source or remain unknown/deferred.
- Existing v2 artifacts may be read only through an explicit legacy reader. They must not be compared with v3 artifacts as if semantically equivalent.
- Release evaluation rejects training images, unknown split, stale replay keys, dirty code, unknown detector identity, and mixed semantic inputs.
- A locally useful experimental run may contain unknown provenance fields, but must be visibly marked non-release-eligible.

## Target module boundaries

Keep the public interfaces small and put policy detail behind focused modules:

```text
src/recommendation/schema.py       v3 contracts and legacy-safe parsing
src/recommendation/decision.py     concern evidence -> care decision
src/recommendation/therapy.py      decision/profile -> therapeutic intent
src/recommendation/eligibility.py  plan/product/profile -> hard veto result
src/recommendation/composer.py     eligible ranked products -> one regimen
src/recommendation/validator.py    whole-artifact invariant validation
src/recommendation/engine.py       orchestration and deterministic fallbacks
src/pipeline/provenance.py         hashes, canonical JSON, replay/freshness
src/pipeline/e2e.py                single-image orchestration/public CLI
src/pipeline/batch.py              manifest, retry, checkpoint, resume
src/evaluation/e2e_release.py      release cohort validation and metrics
```

If implementation reveals that two adjacent modules are trivial wrappers, combining them is allowed, but the decision, hard-eligibility, composition, validation, provenance, and batch seams must remain independently testable through public functions.

## V3 public contracts to lock before broad implementation

Use dataclasses plus closed string vocabularies, following the current repository style. JSON serializers must be explicit; do not rely on recursive `asdict` when versioning or omission rules matter.

### Care decision

```python
DecisionEvidence(
    concern: str,
    probability: float | None,
    quality: str,              # high | medium | low | unknown
    source: str,
    calibrated: bool,
    reasons: list[str],
)

CareDecision(
    triage_level: str,         # routine | routine_plus_review | derm_first | abstain
    referral_reasons: list[str],
    therapy_disposition: str,  # active_treatment | supportive_only | maintenance | defer
    evidence: list[DecisionEvidence],
    policy_version: str | None,
    policy_reviewed: bool,
)
```

The serializer key is `decision_evidence`, matching the spec even if the Python field is shorter.

### Therapy plan

```python
TherapyOption(
    therapy: str,
    strength_band: str,
    exposure: str,
    cadence: str,
    role: str,
    reason: str | None,
)

TherapyPlan(
    course_weeks: int | None,
    review_at_weeks: int | None,
    primary: TherapyOption | None,
    alternatives: list[TherapyOption],
    support_roles: list[str],
    deferred_reasons: list[str],
    policy_version: str | None,
)
```

Unreviewed policy or missing required intake must not fabricate a primary therapy. It should retain explainable alternatives/intent only if policy permits that representation, and serialize `primary: null` with `deferred_reasons`.

### Product contract v2

Retain the existing normalized `actives` list as the complete known carried-active vocabulary and add:

```python
VerifiedActive(name: str, strength: str | None, source: str | None)

Product(
    ...existing fields...,
    intended_areas: list[str],
    routine_roles: list[str],
    format: str,                    # unknown is valid storage, never hard eligibility
    exposure: str,                  # unknown | rinse_off | short_contact | leave_on | mask | scrub | peel
    drug_actives: list[VerifiedActive],
    otc_drug: bool | None,
    label_source: str | None,
    label_verified_at: str | None,
    broad_spectrum: bool | None,
    spf: int | None,
    comedogenic_claim: str,         # unknown | claimed_noncomedogenic | not_claimed
    irritant_features: list[str],
    contraindications: list[str],
    evidence_roles: list[str],
    evidence_grade: str,
    cadence: str | None,
    cadence_source: str | None,
    amount: str | None,
    amount_source: str | None,
)
```

Legacy catalog rows must still load, with every new field defaulting to unknown/empty. Legacy loading must never make a row eligible by inference.

### User safety profile

Expand `UserProfile` with explicit unknown-capable fields:

```python
skin_type: combination | dry | normal | oily | unknown
tone_bucket: light | medium | deep | unknown
tone_source: self_report | photo | unknown
age_years: int | None
pregnancy_status: pregnant | trying | nursing | not_pregnant | not_applicable | unknown
allergies: list[str]
sensitivity_conditions: list[str]   # eczema, rosacea, sensitive, or operator-defined values
current_actives: list[str]
current_medications: list[str]
treatment_history: list[str]
acne_duration_weeks: int | None
painful_or_deep_lesions: bool | None
prior_scarring: bool | None
max_price_usd: float | None
```

Support a narrow migration from the old `pregnant_or_nursing: bool` input, but never collapse `unknown` into `False` in new payloads.

### Recommendation/regimen

```python
RoutineInstruction(
    role: str,
    slot: str,                 # am | pm
    cadence: str,
    amount: str | None,
    source: str | None,
)

Recommendation(
    decision: CareDecision,
    therapy_plan: TherapyPlan,
    selected_products: dict[str, Product],
    selected_regimen: dict[str, list[RoutineInstruction]],
    alternatives: dict[str, list[Product]],
    eligibility_rejections: dict[str, list[str]],
    explanation: list[dict[str, object]],
    flags: list[str],
    validation_errors: list[str],
)
```

`Recommendation` construction is successful only if `validation_errors` is empty. The E2E serializer must refuse to emit `routine.json` for an invalid regimen and must store an explicit recommendation failure reason in `analysis.json`.

## Execution sequence

Each task below is a vertical slice. For every behavior: add one public-interface test, confirm it fails for the expected reason, implement the minimum behavior, rerun that focused test, then rerun the owning test file before moving on. Do not write all tests up front.

### Task 0 — Protect the baseline and add v3 test helpers

**Files:**

- `tests/conftest.py` or a small new `tests/recommendation_fixtures.py`
- no production behavior change

**Steps:**

1. Record the baseline status/diff/commit described above.
2. Add factory helpers only when the first v3 test needs them: verified cleanser, treatment, moisturizer, sunscreen, complete profile, active-acne report, uncalibrated nodule report.
3. Keep factories explicit: test products should show all hard fields so a failure identifies the exact missing constraint.
4. Do not convert existing tests wholesale. Migrate each old fixture only as its public behavior changes.

**Verify:**

```bash
pytest -q tests/test_recommendation_engine.py tests/test_import_catalog.py tests/test_e2e.py tests/test_sarpn.py
```

### Task 1 — Lock v3 schemas and legacy-safe parsing

**Files:**

- modify `src/recommendation/schema.py`
- modify `src/recommendation/import_catalog.py`
- add `tests/test_recommendation_schema_v3.py`
- extend `tests/test_import_catalog.py`

**RED/GREEN slices:**

1. A complete v3 `Product` round-trips through `product_dict`/`load_catalog`, including nested verified actives.
2. A legacy catalog row loads with unknown v2 fields and is visibly `legacy`, but does not gain eligible roles.
3. `UserProfile` preserves unknown pregnancy status; the old boolean input maps only when explicitly supplied.
4. `CareDecision`, `TherapyPlan`, and regimen serializers reject values outside closed vocabularies.
5. Stable serializer ordering makes the same semantic object byte-identical under repeated serialization.

**Implementation notes:**

- Add `Product.from_dict`/`to_dict` rather than passing nested dicts directly to `Product(**row)`.
- If a `catalog_schema_version` field is added, use `2` for enriched rows and `1`/`legacy` for old rows; do not confuse it with E2E schema version `3`.
- Assertions may remain consistent with current project style, but user-controlled JSON loading should raise `ValueError` with field context rather than leak an `AssertionError`.
- Keep ranker inputs (`product_id`, actives, category, brand, price) source-compatible.

**Focused verify:**

```bash
pytest -q tests/test_recommendation_schema_v3.py tests/test_import_catalog.py tests/test_ranker.py
```

### Task 2 — Add an explicit safety-profile intake path

**Files:**

- modify `src/recommendation/schema.py`
- modify `src/pipeline/e2e.py`
- add/extend `tests/test_e2e.py`
- add `tests/fixtures/profile_complete.json`
- add `tests/fixtures/profile_unknown.json`

**Public behavior:**

- `--profile <json>` loads the full profile.
- Narrow CLI overrides may remain for convenience, but the default profile serializes all fields as unknown; it must not default to `skin_type=combination` or pregnancy `False`.
- Invalid age, pregnancy status, active IDs, or negative budget fails before detector HTTP work.
- `analysis.json` and `routine.json` carry the exact normalized `input_profile`.
- Missing treatment-critical fields lead to a deferred primary treatment, not guessed values.

**RED/GREEN slices:**

1. Parser default yields explicit unknowns.
2. Complete JSON profile round-trips to artifact input.
3. Invalid profile aborts before the fake SA-RPN server receives a request.
4. Old `--pregnant` is either migrated with a deprecation flag or replaced by `--pregnancy-status`; do not let both disagree silently.

### Task 3 — Build the independent care-decision layer

**Files:**

- add `src/recommendation/decision.py`
- add `tests/test_care_decision.py`
- update `src/recommendation/engine.py` only after the decision API is green
- update `docs/RULES.md` once behavior is locked

**Public function:**

```python
decide_care(report: ConcernReport, policy: TriagePolicy) -> CareDecision
```

`TriagePolicy` must include an identifier/version, approval state, calibrator identity, and optional reviewed operating thresholds. The repository may ship an explicitly unreviewed conservative default, but must not label it clinician-approved.

**Decision table tests, one at a time:**

1. Clear skin -> `routine` + `maintenance`.
2. Active acne, severity 1–3, no high-risk evidence -> `routine` + `active_treatment` when policy/profile allow downstream planning.
3. Severity 4 caused only by counts/coverage, no nodule evidence -> not `supportive_only`; use `routine_plus_review` + `active_treatment` and a count/severity referral reason if policy calls for review.
4. Scarring risk -> `routine_plus_review` while preserving `active_treatment`.
5. Persistent pigment concern -> optional `routine_plus_review` while preserving treatment.
6. Calibrated nodule probability above an approved gate -> `derm_first` + `supportive_only` + `suspected_nodule`.
7. Uncalibrated raw nodule detection -> `abstain` + `supportive_only` + `unvalidated_nodule_evidence`.
8. Safety-critical uncertain/low-quality nodule evidence near an approved abstention band -> `abstain` + `supportive_only`.
9. Low-confidence non-high-risk concern is represented in evidence and may defer that concern; it must not be rewritten as a probability.
10. No diagnosis/malignancy wording appears in serialized decision reasons.

**Regression names:** include `random_120_high_count_without_real_nodule_does_not_suppress_treatment` and `random_230_oracle_nodule_routes_derm_first` so the audit counterexamples remain visible.

### Task 4 — Add reviewed-policy loading and therapy planning

**Files:**

- add `src/recommendation/therapy.py`
- add `tests/test_therapy_plan.py`
- modify `configs/default.yaml`
- add `docs/THERAPY_POLICY_SCHEMA.md`
- add only synthetic policy fixtures under `tests/fixtures/`; do not add a fake production approval file

**Public functions:**

```python
load_therapy_policy(path: Path | None) -> TherapyPolicy
plan_therapy(decision, report, profile, policy) -> TherapyPlan
```

**Required behavior:**

- Missing/unreviewed production policy keeps support roles but returns `primary=None`, `therapy_disposition=defer` where active selection would otherwise occur, and a reason such as `clinician_reviewed_policy_missing`.
- A synthetic approved test policy can exercise product-independent paths for azelaic acid, benzoyl peroxide, and an explicitly permitted retinoid combination.
- Pregnancy/trying status excludes retinoid paths; unknown pregnancy status defers retinoid selection rather than treating unknown as false.
- Age, sensitivity conditions, current actives/medications, prior treatment, duration, and pain are exposed to policy checks.
- `course_weeks`, `review_at_weeks`, cadence, and amount are copied from the reviewed policy/label source only. Unknown stays `None`/`per_label` with a source requirement.
- `derm_first`/`abstain` produces only cleanser, moisturizer-if-needed, and sunscreen support roles plus avoidance guidance; it does not auto-start or auto-stop medicine.

**Tests:**

1. Missing policy defers active primary.
2. Synthetic approved policy chooses a deterministic path for a complete eligible profile.
3. Pregnancy/trying and unknown status prevent a retinoid primary.
4. Existing current active creates a duplication/conflict planning reason.
5. Derm-first plan contains no active treatment role.
6. No cadence or amount appears without a named source.

### Task 5 — Extend catalog import with a verification overlay and quarantine report

**Files:**

- modify `src/recommendation/import_catalog.py`
- modify `docs/CATALOG_SCHEMA.md`
- extend `tests/test_import_catalog.py`
- add `tests/fixtures/catalog_verification_sample.json`
- add malformed/incomplete verification fixtures as needed

**Interface:**

```bash
python -m src.recommendation.import_catalog \
  --csv ... --format sephora --verification verified_products.json \
  --out catalog.json --quarantine-out catalog_quarantine.json
```

**Rules:**

- The raw Sephora/BeautyAPI source may preserve source taxonomy as provenance, but must not manufacture verified drug strengths, label URLs, broad-spectrum claims, cadence, amount, contraindications, or timestamps.
- Verification overlay is keyed by `product_id`, schema-validated, and applied deterministically.
- A product may be stored with unknown fields, but the import report lists every role it is quarantined from and exact reason codes.
- Face/neck/body/eye/lip distinctions must survive mapping. Stop mapping decollete/neck items to a facial moisturizer role.
- Masks, scrubs, peels, makeup removers, balms, and rinse-off cleansers retain accurate format/exposure; they cannot masquerade as leave-on treatment.
- SPF role requires explicit broad-spectrum `True` and numeric SPF >= 30.
- Drug actives distinguish verified strength from ingredient-list occurrence.
- All new output remains deterministic/idempotent.

**RED/GREEN slices:**

1. Verified azelaic treatment receives face/treatment/leave-on/strength/source metadata.
2. Same product without overlay remains stored but quarantined from treatment.
3. Neck serum is not eligible for face moisturizer.
4. Mask/scrub/peel cannot claim leave-on treatment.
5. SPF 50 without broad-spectrum verification is quarantined; broad-spectrum SPF 30 passes the catalog completeness check.
6. Unknown contraindication/comedogenic data remains unknown rather than false.
7. Malformed overlay identifies product and field.
8. Repeated imports are byte-identical, including quarantine report.

Do not rebuild or commit the real catalog unless a real verification overlay exists. Fixture-level proof is sufficient for the code task; real SKU verification remains an operator/data task.

### Task 6 — Implement hard role eligibility and the carried-active veto

**Files:**

- add `src/recommendation/eligibility.py`
- add `tests/test_product_eligibility.py`

**Public function:**

```python
check_eligibility(
    product: Product,
    role: str,
    therapy: TherapyOption | None,
    profile: UserProfile,
    selected_products: Mapping[str, Product] = {},
) -> EligibilityResult
```

`EligibilityResult` contains `eligible: bool` plus stable machine-readable reason codes and optional field paths.

**Eligibility matrix:**

- intended area includes `face`;
- proposed role is in `routine_roles`;
- format is allowed for role;
- exposure matches the therapy/support role;
- treatment directly matches therapy and verified strength band;
- label source and verification timestamp exist where required;
- every `actives` and every `drug_actives` entry is checked against profile contraindications, current actives/medications, and already selected products;
- removed or contraindicated therapy cannot re-enter through a support-active match;
- cleanser cannot satisfy leave-on therapy merely because it lists a trace active;
- sunscreen is verified broad-spectrum SPF 30+;
- required unknown field produces a veto, not a score penalty.

**Required regression tests:**

1. BP product matched through ceramides is rejected after BP is removed/contraindicated.
2. Retinoid-carrying niacinamide product is rejected for pregnancy/trying/unknown policy where required.
3. Trace salicylic rinse-off cleanser cannot fill treatment.
4. Direct verified azelaic leave-on treatment passes.
5. Neck serum fails facial moisturizer.
6. Mask, scrub, and peel fail daily leave-on treatment.
7. Unverified strength/source fails treatment.
8. Unverified sunscreen claim fails SPF.
9. Duplicate/conflicting carried actives across two roles fail.
10. Support ingredient can explain/tie-break an eligible support product but cannot admit it to treatment.

### Task 7 — Rank eligible equivalents, compose one regimen, validate the whole result

**Files:**

- add `src/recommendation/composer.py`
- add `src/recommendation/validator.py`
- add `tests/test_regimen_composer.py`
- add `tests/test_regimen_validator.py`
- adapt `src/recommendation/ranker.py` only as required for a richer score explanation

**Ranking order:**

1. Hard eligibility has already passed and is absent from the numeric score.
2. Concern-specific outcome evidence with sample size/provenance.
3. Tolerability/profile fit.
4. Evidence quality and completeness.
5. User budget/preference when price is usable and explicitly accepted.
6. Pooled rating/popularity as final deterministic tie-breaker only.

If concern evidence is absent, tag the fallback as pooled/general and do not call it acne efficacy. Preserve the existing `StatsRanker` as a final tie-break input; do not let it see ineligible candidates.

**Composer behavior:**

- Select at most one product for each requested role.
- A product may not occupy multiple selected roles unless policy explicitly permits the role combination and validator confirms schedule/active safety; default is one role per SKU.
- Keep alternatives in `alternatives[role]`; never append them to selected AM/PM steps.
- Alternatives must be independently eligible and ordered, with a configurable small limit that does not change selected counts.
- AM/PM lists contain role instructions, not repeated product menus.
- Supportive-only regimen contains cleanser, moisturizer-if-needed, sunscreen; no treatment.
- Active regimen contains one treatment if a fully eligible product exists; otherwise returns a missing-role reason and does not pretend completion.

**Validator invariants, each with a failing test:**

1. Required therapy role missing without reason.
2. More than one selected product for a role.
3. Alternative appears in selected products or AM/PM steps.
4. Product used outside intended area/role.
5. Treatment strength/exposure/source unverified.
6. Duplicate or conflicting carried active.
7. Profile contraindication.
8. Mask/scrub/peel scheduled as daily leave-on.
9. Explanation claims an active/strength not delivered by selected product.
10. Instruction cadence/amount lacks source.
11. Same role/SKU repeated within or across slots unexpectedly.
12. SPF scheduled in PM or missing from a plan that requires it.

**Regression:** replace the current deep-carrier coverage-promotion expectation with `coverage_promotion_cannot_create_a_second_selected_treatment`. The v3 result should select the best eligible treatment and place lower-ranked carriers only in alternatives.

### Task 8 — Rebuild engine orchestration around decision -> plan -> eligibility -> rank -> compose -> validate

**Files:**

- refactor `src/recommendation/engine.py`
- rewrite `tests/test_recommendation_engine.py` incrementally around public v3 behavior
- retain a clearly named legacy adapter only if a current historical caller needs it
- update `docs/RULES.md`, `CONTEXT.md`, and `docs/DECISIONS.md`

**Public API:**

```python
recommend(
    report: ConcernReport,
    catalog: list[Product],
    profile: UserProfile,
    *,
    triage_policy: TriagePolicy,
    therapy_policy: TherapyPolicy,
    concern_scorer=None,
    pooled_ranker=None,
) -> Recommendation
```

No silent profile default in the v3 API. Any compatibility wrapper must construct an explicit unknown profile and mark output experimental/legacy.

**Orchestration tests:**

1. Severity-4 high-count non-nodule case retains active-treatment disposition.
2. Uncalibrated nodule case abstains/supports only.
3. Scarring adds review while active therapy remains possible.
4. All product eligibility occurs before any scorer/ranker call; a spy ranker never sees rejected products.
5. Direct eligible azelaic treatment outranks support serums/masks because those are not candidates for its role.
6. Carried-active BP cannot re-enter after plan de-stacking.
7. Exactly one selected product per role; alternatives separate.
8. Missing verified treatment yields explicit defer/missing-role output, not a mask or serum substitution.
9. Maintenance and supportive paths use the same hard role/area/SPF checks.
10. Concern scorer fallback and pooled ranker are truthfully tagged.

Remove or isolate obsolete mechanisms whose semantics conflict with v3:

- binary `Recommendation.mode` as the primary decision;
- five products per category as the routine contract;
- active-union admission;
- post-truncation coverage promotion;
- severity-4 blanket supportive-only routing;
- name-substring heuristics as a substitute for verified format/area fields. Heuristics may remain import warnings/quarantine aids, never proof of eligibility.

### Task 9 — Serialize schema-version 3 analysis/routine artifacts

**Files:**

- refactor `src/pipeline/e2e.py`
- extend `tests/test_e2e.py`
- optionally add `src/pipeline/artifacts.py` if serialization would otherwise dominate orchestration

**Artifact behavior:**

- `analysis.json` and `routine.json` use `schema_version: "3"` exactly.
- Both include the same provenance envelope and exact `input_profile`.
- Analysis includes care decision even when catalog/recommendation is unavailable, so triage/referral does not depend on product data.
- Routine includes `decision`, `therapy_plan`, `selected_regimen`, `selected_products`, `alternatives`, eligibility/fallback explanations, validation status, and no category menus.
- If regimen validation fails, do not write a routine artifact. Analysis records `recommendation_status=invalid` and stable reasons.
- Legacy v2 artifact reader labels output `legacy`; no v2 writer remains on the production path.
- CLI output surfaces `triage_level`, referral reasons, therapy disposition, recommendation status, and release eligibility without diagnosis wording.

**E2E tests:**

1. Complete fixture writes v3 artifact set.
2. Decision exists without catalog.
3. At most one selected product per role in serialized output.
4. Alternatives are disjoint.
5. Profile unknowns are explicit.
6. Invalid recommendation does not erase analysis.
7. Existing atomic directory publication and concurrency protections remain green.

### Task 10 — Add deterministic provenance, replay key, and stale detection

**Files:**

- add `src/pipeline/provenance.py`
- add `tests/test_provenance.py`
- integrate into `src/pipeline/e2e.py`
- extend `tests/test_e2e.py`

**Public helpers:**

```python
sha256_file(path) -> str
canonical_json_bytes(value) -> bytes
build_provenance(inputs, *, clock, git_reader) -> dict
compute_replay_key(semantic_inputs) -> str
validate_artifact_freshness(artifact, current_inputs=None) -> list[str]
```

**Semantic inputs to hash:**

- source image bytes;
- declared dataset name/sample ID/split and split-proof metadata;
- normalized input profile;
- sanitized effective pipeline config;
- detector identity/hash;
- classifier identity/hash or explicit `not_applicable` for the SA-RPN-only production path;
- catalog bytes/hash or explicit unavailable state;
- concern scorer/ranker artifact hash or explicit none;
- triage and therapy policy bytes/hash;
- code commit and dirty state;
- schema version.

Exclude volatile/render-only values such as `generated_at`, output directory, attempt ID, and diagnostic image encoding from the replay key.

**CLI provenance inputs:**

- add a dataset manifest or explicit `--dataset-name`, `--sample-id`, `--dataset-split` (`train|valid|test|external|smoke|unknown`);
- add remote detector identity/hash input because a URL alone does not identify model weights;
- local experimental runs may use unknown, but release mode must reject it.

**Tests:**

1. Hash stability independent of dict insertion order.
2. Generated timestamp does not alter replay key.
3. Profile/config/catalog/model/policy/source changes do alter replay key.
4. Dirty code is recorded.
5. Missing artifact file has explicit unknown/unavailable identity.
6. Analysis and routine share provenance/replay key.
7. Mutated stale artifact is rejected.
8. Legacy v2 artifact is labeled legacy and excluded from equality comparison.

### Task 11 — Build honest evaluation and release-report tooling

**Files:**

- add `src/evaluation/__init__.py`
- add `src/evaluation/e2e_release.py`
- add `tests/test_e2e_release.py`
- add small JSON/XML fixtures only; do not duplicate the full dataset
- document CLI in `README.md`

**Inputs:**

- v3 run directories;
- cohort manifest with dataset/split proof;
- optional AcneSCU VOC oracle annotations;
- optional clinician-reviewed triage/disposition labels;
- frozen threshold/policy metadata.

**Preflight:**

- reject training or unknown-split images from a release report;
- reject dirty-code artifacts;
- reject stale/mismatched replay keys;
- reject mixed detector/config/catalog/policy hashes unless the cohort is explicitly stratified and no aggregate is claimed;
- reject duplicate source image SHA/sample IDs;
- report missing clinician labels/calibration as blocked gates, not zero scores.

**Metrics:**

- per-class detector precision/recall at named operating point;
- nodule triage sensitivity, specificity, PPV, and abstention rate;
- prediction-vs-oracle triage confusion;
- prediction-vs-clinician disposition agreement when labels exist;
- referral-reason precision/recall;
- therapy-plan agreement and target coverage;
- product-role precision;
- contraindication/conflict violation count;
- selected-product count per role;
- explanation/product consistency;
- artifact freshness/replay equality;
- per-image attempts, failure class, latency, and completion.

Include sample counts and a deterministic binomial confidence interval implementation (Wilson is acceptable) for safety-critical rates. Do not use aggregate lesion accuracy as a proxy for triage.

**Counterfactual path:** run the same held-out manifest through prediction-derived concern evidence and annotation-derived oracle concern evidence, preserving distinct replay inputs/source tags. A release report compares them but never mixes them into one cohort.

**Tests:**

1. One training sample causes preflight failure.
2. Unknown/dirty/stale/mixed sample causes named failure.
3. Tiny known confusion fixture yields exact metric counts and intervals.
4. Missing clinician labels returns `blocked`, not a made-up agreement.
5. Selected-product/role and validation violations are counted from artifacts.
6. Report JSON is deterministic apart from an injected timestamp.

### Task 12 — Refactor single-image work into resumable stages

**Files:**

- refactor `src/pipeline/e2e.py`
- add `src/pipeline/batch.py`
- add `tests/test_batch_pipeline.py`

**Stages:**

```text
identified -> regions_and_concerns -> decision_and_recommendation -> rendered -> published
```

Persist a validated atomic checkpoint after every completed stage. The identification checkpoint must contain enough normalized observations to resume without another SA-RPN request. Later checkpoints must include their input replay fragment so they are invalidated if source/config/model/profile/catalog/policy changes.

**Manifest contract:**

```json
{
  "run_id": "stable request-derived id",
  "requested": 3,
  "images": {
    "sample-id": {
      "source_image_sha256": "...",
      "state": "complete|retryable_failed|permanent_failed|stale|in_progress",
      "last_verified_stage": "identified",
      "attempts": [{"attempt_id": "...", "failure_class": null, "latency_ms": 10}],
      "artifact_dir": "...",
      "replay_key": "..."
    }
  }
}
```

Every manifest/checkpoint write uses temp-file + `fsync` where practical + same-directory `os.replace`, followed by schema validation/read-back.

**Retry policy:**

- retry only transient transport/timeouts/5xx classes;
- never retry response-contract/schema/profile/catalog/policy errors;
- bounded attempts;
- exponential backoff with bounded jitter;
- injectable sleeper/random/clock in tests;
- stable run ID across resume, unique attempt ID per try.

**Resume behavior:**

- complete + fresh -> skip;
- partial + fresh -> resume after `last_verified_stage`;
- stale checkpoint -> mark stale and recompute from first affected stage;
- retryable failed below attempt cap -> retry;
- permanent failed or exhausted -> terminal failure;
- interrupted manifest can be reopened without losing completed images.

**Exit/summary:**

- exit non-zero if any requested image lacks a valid terminal artifact;
- print/write requested, completed, failed, retried, skipped, stale, and total attempt counts;
- an optional recommendation failure may still be a completed analysis only when the artifact contract explicitly represents it; missing identification never counts complete.

**Tests:**

1. Inject transient identification failure, then success; assert retry/backoff/attempt rows.
2. Inject permanent malformed response; assert no retry.
3. Interrupt after identification; resume without additional fake-server request.
4. Complete image is skipped on rerun.
5. Changed catalog/profile resumes from recommendation rather than identification.
6. Changed detector/source invalidates identification.
7. Atomic-write failure leaves last valid manifest readable.
8. Summary/exit code counts every requested image exactly once.

### Task 13 — Pin all audit regressions at public seams

**Files:**

- add `tests/fixtures/concern_correctness_cases.json`
- extend decision/eligibility/composer/evaluation tests

Represent current audited failures as compact semantic fixtures, not model-dependent full-image tests:

- `random-120`: false-nodule/high-count escalation does not let severity alone suppress active treatment; uncalibrated nodule signal abstains rather than pretending certainty.
- `random-230`: annotation-derived nodule evidence takes derm-first/supportive path.
- `random-252` and `random-274`: missed prediction nodule vs oracle nodule produces a visible counterfactual disagreement.
- `random-147`: pigment overcount does not masquerade as calibrated probability and appears in decision evidence quality.
- `smoke-2`: unsupported/other observations remain safety evidence and are not absorbed into acne therapy targets.
- BP product admitted through ceramides is vetoed when BP is removed/contraindicated.
- Direct verified azelaic leave-on treatment is not displaced by mask/support serum.
- Neck/decollete product cannot fill facial moisturizer.
- Coverage repair cannot exceed one selected product per role.
- Stale replay key is rejected.

Each fixture should state whether it represents prediction or oracle evidence and must not be counted as held-out release evidence merely because it is a regression test.

### Task 14 — Documentation and compatibility migration

**Files:**

- update `README.md`
- update `CONTEXT.md`
- update `docs/RULES.md`
- update `docs/CATALOG_SCHEMA.md`
- update `docs/CONCERN_SCHEMA.md`
- append dated decisions to `docs/DECISIONS.md`
- update the spec status/implementation note without altering audit evidence

Document:

- v3 architecture and the independent decision axes;
- explicit unknown profile intake;
- catalog verification/quarantine flow;
- one-per-role selected regimen vs alternatives;
- provenance/replay and release eligibility;
- batch/retry/resume CLI;
- evaluation cohort rules and blocked external gates;
- legacy v2 reader behavior and prohibition on v2/v3 comparison;
- the fact that code completion is not clinical approval or a release claim.

Ensure old README claims that production E2E uses `ranker=None` or emits v2 menus are corrected to the actual shipped behavior.

### Task 15 — Full verification, review, remediation, and commit

**Fast checks during work:**

```bash
pytest -q tests/test_care_decision.py
pytest -q tests/test_therapy_plan.py
pytest -q tests/test_product_eligibility.py
pytest -q tests/test_regimen_composer.py tests/test_regimen_validator.py
pytest -q tests/test_recommendation_engine.py
pytest -q tests/test_provenance.py tests/test_e2e.py
pytest -q tests/test_e2e_release.py tests/test_batch_pipeline.py
```

**Final checks:**

```bash
pytest -q
git diff --check
git status --short
```

If the two known TensorFlow import-order tests still fail only because TensorFlow is unavailable, verify that they are the same environment failures recorded by the spec and report them exactly. Do not label new failures environmental without reproducing the baseline failure signature.

**Required review:**

Run the repository `review` workflow against the recorded baseline commit and this spec:

- Standards sources: `CONTEXT.md`, `docs/DECISIONS.md`, `docs/RULES.md`, `docs/CATALOG_SCHEMA.md`, `docs/CONCERN_SCHEMA.md`, `README.md`, and any discovered agent guidance.
- Spec source: `docs/superpowers/specs/2026-07-13-concern-to-routine-correctness-design.md`.
- Spawn parallel standards and spec review agents as required by the review workflow.
- Fix all correctness/safety findings, add regression tests, and rerun focused + full suites.
- Repeat review if any high-impact finding required structural changes.

**Commit:**

- Inspect the staged diff and confirm no run artifacts, raw datasets, `.claude`, model weights, or unrelated user files are staged.
- Commit the complete in-scope change on the current branch with an intentional message such as `feat: implement concern-to-routine correctness v3`.
- If multiple small commits were made during execution, each must remain coherent and green; otherwise use one final commit.

## Acceptance checklist for repository-controlled work

- [ ] Schema v3 artifacts and explicit legacy v2 reader exist.
- [ ] Decision layer separates triage/referral from therapy disposition.
- [ ] Severity/count without high-risk evidence cannot force supportive-only care.
- [ ] Uncalibrated high-risk evidence abstains; no raw confidence is labeled probability.
- [ ] Missing reviewed therapy policy or safety intake defers treatment honestly.
- [ ] Product contract stores all v2 verification fields and unknowns.
- [ ] Importer applies verified overlay and writes deterministic quarantine reasons.
- [ ] Hard eligibility checks area, role, format, exposure, therapy, strength, labels, sunscreen, profile, and all carried actives.
- [ ] Rankers see eligible products only and pooled stats are final tie-breakers.
- [ ] One selected product per role; alternatives separate.
- [ ] Whole-regimen validator rejects every enumerated invalid state.
- [ ] E2E analysis carries decision even without a catalog.
- [ ] Both artifacts carry source/config/code/model/catalog/ranker/policy/profile provenance and one replay key.
- [ ] Release preflight rejects train/unknown/dirty/stale/mixed runs.
- [ ] Release metrics include sample counts and confidence intervals.
- [ ] Batch retry/checkpoint/resume/terminal summary behaviors are tested.
- [ ] All named audit regressions are pinned.
- [ ] Docs describe real behavior and external blockers.
- [ ] Focused and full tests are green, except any exactly reproduced pre-existing environment-only TensorFlow failures.
- [ ] Standards and spec reviews have no unresolved high-impact findings.
- [ ] Only in-scope files are committed.

## External release gates that code must expose but cannot satisfy

These are not implementation excuses; they are required stop conditions. The executor must finish all code/test/documentation work above, then mark release status blocked until the corresponding real evidence is supplied.

1. **Clinician policy approval:** no qualified clinician approval artifact currently exists for triage labels, referral rules, therapy dispositions, numeric nodule operating/abstention gates, therapy paths, or instructions.
2. **Adequate calibration cohort:** the 18-image audit cannot define a nodule sensitivity/PPV target. A sufficiently large frozen held-out positive/negative cohort is required.
3. **External clinical review set:** no consented representative set with clinician triage/regimen-disposition labels is present.
4. **Verified real catalog overlay:** real product role/area/format/exposure/strength/label/SPF verification must come from authoritative product/regulatory sources and be timestamped; test fixtures are not approval.
5. **Remote detector identity:** the production HTTP service must expose or be configured with an immutable detector artifact hash for release runs.

The final implementation may report `release_status: blocked` with these missing gates. It must never weaken a gate, substitute audit images, or present fixture data as proof to make the status pass.

## P2 follow-up boundary

The spec's P2 items are intentionally downstream of P0/P1 correctness:

- budget-aware ordering may be implemented now only among already eligible equivalents and only when price freshness/meaning is explicit;
- longitudinal follow-up requires a separately reviewed policy and product/user data-retention design;
- clinician adjudication tooling depends on the external review workflow;
- catalog drift monitoring depends on an authoritative verified catalog source.

Do not create speculative storage, UI, notification, or monitoring systems in this execution. Record these as follow-up work after the v3 core is green and the external gates have owners.
