# Therapy policy schema v1

The therapy policy is the reviewed, product-independent boundary between a
`CareDecision` and `TherapyPlan` (D-029). The repository deliberately ships no
production policy. `load_therapy_policy(None)` returns an explicitly unreviewed
policy; an active-treatment decision is then deferred with
`clinician_reviewed_policy_missing` rather than receiving a guessed therapy.

This is cosmetic decision-support code, not medical advice or clinical
approval. The synthetic policy under `tests/fixtures/` is marked `test_only`
and exists solely to exercise code paths.

## Top-level object

```json
{
  "policy_id": "stable-policy-id",
  "version": "immutable-version",
  "reviewed": true,
  "reviewed_by": "qualified-reviewer-or-approval-artifact-id",
  "test_only": false,
  "support_roles": ["cleanser", "moisturizer", "sunscreen"],
  "paths": []
}
```

- `policy_id` and `version` are required non-empty strings and form the
  serialized policy identity.
- `reviewed` is required. A reviewed production policy requires `reviewed_by`.
- `test_only` permits a fixture to omit `reviewed_by`; such a policy is never
  release evidence.
- `support_roles` is optional; the conservative default is cleanser,
  moisturizer, and sunscreen.
- `paths` is an ordered list. The first profile-compatible path becomes the
  primary and later compatible paths become alternatives.

## Path object

```json
{
  "therapy": "azelaic_acid",
  "strength_band": "10%",
  "exposure": "leave_on",
  "cadence": "per_label",
  "cadence_source": "https://authoritative.example/label",
  "amount": null,
  "amount_source": null,
  "role": "treatment",
  "reason": "reviewed-first-line-path",
  "course_weeks": 12,
  "review_at_weeks": 12,
  "min_age_years": 12,
  "max_age_years": null,
  "excluded_pregnancy_statuses": ["pregnant", "trying", "nursing"],
  "excluded_sensitivity_conditions": ["eczema"],
  "conflicting_actives": ["azelaic_acid"],
  "conflicting_medications": [],
  "excluded_treatment_history": ["azelaic_acid_failed"],
  "min_acne_duration_weeks": 4,
  "max_acne_duration_weeks": null,
  "required_painful_or_deep_lesions": false,
  "required_prior_scarring": null,
  "requires_known": ["age_years", "pregnancy_status"],
  "concerns": ["acne_comedonal", "acne_inflammatory"]
}
```

`therapy`, `strength_band`, `exposure`, `cadence`, and `role` are required.
Medicine cadence always requires `cadence_source`; an amount requires
`amount_source`. Course/review/age/duration values are positive integers or
null. Constraint lists contain normalized profile values. Treatment history,
duration, painful/deep lesions, and prior scarring are first-class constraints;
a mismatch excludes the path and a required unknown defers it. Unknown fields
named by `requires_known` defer the path. Retinoid paths always defer on unknown
pregnancy status and always exclude pregnant, trying, and nursing states,
regardless of whether a malformed or incomplete policy attempted to omit one.

Unknown top-level/path keys, scalar coercion (for example `1` as a boolean),
invalid reviewer types, and non-finite numeric values are rejected. Policy
validation is intentionally strict so misspelled safety constraints cannot be
silently ignored.

The planner does not start/stop medicine for `derm_first` or `abstain`; those
dispositions produce support roles only and explicit guidance to avoid
self-starting or stopping medicine pending professional review. Product
selection happens later and must independently verify therapy, strength,
exposure, source, timestamp, profile constraints, and every carried active.

## Release boundary

A real policy needs qualified review of therapy paths, referral/disposition
semantics, contraindications, course/review timing, and every instruction
source. A checked JSON shape is not that review. Release evaluation remains
blocked until clinician policy approval and calibration/external evidence are
provided outside this repository.
