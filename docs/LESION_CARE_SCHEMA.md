# Lesion care schema 4

Schema 4 is the active detector-to-care-to-product contract for the synthetic
MVP. It preserves exact labels through the complete workflow:

`closed_comedo`, `open_comedo`, `papule`, `pustule`, `nodule`,
`atrophic_scar`, `hypertrophic_scar`, `melasma`, `nevus`, and `other`.

## Scope gate

The policy in `lesion_care_policy.proposed.json` is bound to the evidence
report hash and the recorded high-reasoning AI research audit. It is eligible
only when all of these are true:

- environment is development or test;
- both `synthetic_profile` and `fixture_image` are attested;
- the policy, evidence-report, and source-manifest hashes match both the audit
  and compiled trust roots;
- the policy remains explicitly production-ineligible.

The integrated CLI requires `--mvp-synthetic`, an explicit `test` environment,
and `--mvp-fixture-manifest`. The code pins that manifest's own digest before
reading its claims; the trusted manifest then hash-pins the image bytes and
approved raw-profile→normalized-profile pairs while restricting dataset names
and split proofs. Authorization, decoding, inference, rendering, and provenance
all use one immutable image buffer. The flag or a caller-created manifest can
never authorize an input, and a rewritten policy/source-manifest pair cannot
replace the audited artifacts. Omitting any prerequisite, using an unregistered
image/profile pair, or running in production defers every detected pathway. The
audit is not clinical approval. Recsys independently rechecks every artifact
and input pair before accepting schema 4, and schema 4 cannot substitute an
external profile file.

## Analysis contract

`lesion_findings[]` always has exactly ten entries. Each entry contains the
exact label, count, regions, mean/max detector confidence, and evidence source.
Confidence describes detector evidence quality; `decision_evidence.probability`
is always null unless a separate calibrated model is introduced.

`care_pathways[]` also has exactly ten entries. Each path carries its status,
exact retail active specifications, product roles, separately channelled
clinician options, reason codes, source IDs, unknown required answers, and
policy identity. Status is one of:

- `not_detected`
- `retail_eligible`
- `clinician_only`
- `deferred`
- `monitoring_only`
- `unsupported`

Grouped `concerns` remain serialized for schema-3 consumers but are deprecated
and ignored by schema-4 care and product parsing.

## Intake and care rules

Conditional unknowns defer only the affected path. These include new/changing
spots; separate bleeding, itching, pain, and other symptoms; acne control;
scar duration and wound/diagnosis state; and pigment onset during pregnancy or
hormonal medication use. Referral and retail eligibility are independent, so a
melasma path may require diagnostic confirmation while still permitting a safe
retail sunscreen target.

Prescriptions and procedures are clinician options, never catalog targets.
`nevus` is monitoring/referral only and `other` is unsupported; both prohibit
treatment-product matching.

## Product ownership and compatibility

Recsys is the only product selector. It consumes only exact
`retail_eligible` paths, applies deterministic active and safety gates, ranks
verification rather than blanket-excluding incomplete retail products, and
selects at most one product per slot and emits zero or one selected regimen.
Other evaluated archetypes are reported as unselected, not serialized as
additional routines. Melasma sunscreen requires parsed iron
oxide. Hypertrophic scar products require the dedicated `scar_care` role,
exact silicone evidence, a closed wound, and clinician-confirmed scar type.

`recommendations.json` emits `target_lesions`, the source `care_pathways`, and
one `lesion_coverage` row per detected label. Coverage statuses are
`covered_by_product`, `clinician_only`, `deferred`, `unfilled`,
`monitoring_only`, or `unsupported`. A product can cover multiple labels, but
an unrelated active never counts as coverage. The compatibility
`routine.json` is projected from this recsys result; the old selector is not on
the active path.

Schema 3 remains readable for one migration release. Its exact targets are
derived only from `concerns[].evidence.labels`, never from the grouped concern
name.

## Rollout telemetry

The release evaluator reports, per exact label, detected sample count, product
coverage and unfilled rates, referral rate, pathway outcomes, and safety
deferrals. It also reports selected-product verification distribution. These
metrics must be reviewed before schema 3 is retired.
