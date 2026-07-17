"""Frozen I/O contracts: what recsys reads (analysis.json, profile.json) and
the vocabularies it validates against. The output document is assembled in
pipeline.py; its schema is documented in ARCHITECTURE.md.

Unknown fields in the inputs are ignored (forward-compatible); unknown VALUES
in closed vocabularies are contract violations.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .inci import CANONICAL_ACTIVES, _SAFETY_RULES_PATH

SCHEMA_VERSION = "recsys-1"
ANALYSIS_SCHEMA_VERSION = "4"
LEGACY_ANALYSIS_SCHEMA_VERSION = "3"

# Trust root for the single AI-audited, synthetic-only MVP policy. Schema-4
# inputs must match these immutable artifact identities; a caller cannot mint a
# retail pathway by setting scope_authorized=true in arbitrary JSON.
TRUSTED_SCHEMA4_POLICY = {
    "identity": "skinscan-lesion-care-us:2026-07-16-mvp.1",
    "sha256": "8d24399f076e7721a96b1fc6cf59174bb94390b655a15eb326c8d3a722ffbae1",
    "report_sha256": "8e201d6d869a67e8635f29cc43452138e760561ae0e34d2b2b4b8ba1d6e4344d",
    "source_manifest_sha256": (
        "5f6f69e5d468ce947ae2eb6e321a8476267dd8cf1381758a2017011e97c50a2f"
    ),
    "fixture_manifest_sha256": (
        "10d018abc93ffb84a9ecee0b79cb16dbeaea92ff68108f9f3222f592fd001508"
    ),
}
TRUSTED_SCHEMA4_IMAGE_SHA256S = frozenset({
    "15ac6670480316bb7f7ae83d3846ffcdc0a4c952a526186000283c378f32a7b0",
    "3641d770996c5c09358956e0da15f4e03bb3b67099db9227fa6faff7dfac9e2b",
})
TRUSTED_SCHEMA4_PROFILE_PAIRS = frozenset({
    (
        "a362655a68cda891c4841185801906961e22e198936ef3c1de9d55c2bc104a9e",
        "9a7b21cd69a0254a9b372473079f4ae19a7f2f0c30afe0e21890e821ddcfff4c",
    ),
    (
        "98148a01ad8da87339e01977d539813926d0d7b2b96f21d34ea1c1bbf72bbfc7",
        "196e888934136051a1beb499a6d7babd30e51bfcb4dba56dc82e7a7a9a1fe658",
    ),
    (
        "cf1ce100f13f85d89c75ffbfdfe048db7dbf9134270a72c25ac76ddce0ddae02",
        "ab380708a4267edc72df828d81330bea4d2122061013d98f16f7eea91a6c2e09",
    ),
})

LESION_TYPES = (
    "closed_comedo", "open_comedo", "papule", "pustule", "nodule",
    "atrophic_scar", "hypertrophic_scar", "melasma", "nevus", "other",
)
LESION_TYPE_SET = frozenset(LESION_TYPES)
CARE_PATHWAY_STATUSES = frozenset({
    "not_detected", "retail_eligible", "clinician_only", "deferred",
    "monitoring_only", "unsupported",
})

CONCERNS = (
    "acne_comedonal", "acne_inflammatory", "acne_cystic", "acne_scarring",
    "hyperpigmentation", "dryness",
)
TRIAGE_LEVELS = ("routine", "routine_plus_review", "derm_first", "abstain")
THERAPY_DISPOSITIONS = ("active_treatment", "supportive_only", "maintenance", "defer")
SKIN_TYPES = ("combination", "dry", "normal", "oily", "unknown")
TONE_BUCKETS = ("light", "medium", "deep", "unknown")
TONE_SOURCES = ("self_report", "photo", "unknown")
PREGNANCY_STATUSES = (
    "pregnant", "trying", "nursing", "not_pregnant", "not_applicable", "unknown",
)
SLOTS = ("cleanser", "treatment", "serum", "moisturizer", "spf")

# The closed vocabulary a profile may declare it is already using. gates.py
# matches these against product.actives by exact set intersection, so an
# un-normalized value ("Retinol", "salicylic acid") would intersect nothing and
# silently fail the duplicate-active HARD gate open rather than veto. Ported
# from src.recommendation.schema.KNOWN_ACTIVE_IDS, whose UserProfile raises on
# the same input.
KNOWN_ACTIVE_IDS = frozenset(CANONICAL_ACTIVES.values())


def _declarable_active_ids(path: Path = _SAFETY_RULES_PATH) -> frozenset[str]:
    """KNOWN_ACTIVE_IDS plus every active safety_rules.json names (retinoids +
    prescription_actives), read from the SAME packaged file inci.py derives the
    pregnancy scan from. The synonym table alone cannot express a prescription:
    the drug door mints "tretinoin" into product.actives, but no cosmetic INCI
    ever parses to it, so a truthful "I am on tretinoin" hard-errored the whole
    run — and the duplicate-active gate could therefore never fire for exactly
    the users with the most dangerous current actives. Deriving both sides from
    one file makes profile-vocabulary-lags-catalog unrepresentable."""
    rules = json.loads(path.read_text(encoding="utf-8"))
    extra = set(rules.get("retinoids") or []) | set(rules.get("prescription_actives") or [])
    return KNOWN_ACTIVE_IDS | extra


DECLARABLE_ACTIVE_IDS = _declarable_active_ids()


class ContractViolation(ValueError):
    """Input does not satisfy a recsys contract. str(exc) starts with
    'contract_violation:<field>'."""

    def __init__(self, field_name: str, detail: str):
        super().__init__(f"contract_violation:{field_name}: {detail}")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ConcernFinding:
    concern: str
    severity: int
    confidence: float
    lesion_count: int | None
    regions: tuple[str, ...]
    evidence_labels: dict[str, int]


@dataclass(frozen=True)
class LesionFinding:
    lesion_type: str
    count: int
    regions: tuple[str, ...]
    mean_detector_confidence: float | None
    max_detector_confidence: float | None
    evidence_source: str


@dataclass(frozen=True)
class CarePathway:
    lesion_type: str
    status: str
    retail_target_actives: tuple[str, ...]
    retail_target_specs: tuple[dict, ...]
    required_product_roles: tuple[str, ...]
    clinician_options: tuple[dict, ...]
    reason_codes: tuple[str, ...]
    policy_source_ids: tuple[str, ...]
    required_answers: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisInput:
    schema_version: str
    lesion_findings: tuple[LesionFinding, ...]
    care_pathways: tuple[CarePathway, ...]
    concerns: tuple[ConcernFinding, ...]
    skin_tone_bucket: str
    safety_observations: tuple[dict, ...]  # {code, professional_review} kept verbatim
    triage_level: str
    referral_reasons: tuple[str, ...]
    therapy_disposition: str
    policy_reviewed: bool
    therapy_policy_reviewed: bool
    therapy_policy_identity: str | None
    therapy_policy_sha256: str | None
    therapy_plan: dict
    therapy_primary: dict | None
    therapy_support_roles: tuple[str, ...]
    therapy_deferred_reasons: tuple[str, ...]
    input_profile: dict
    source_image_sha256: str | None
    generated_at: str | None
    analysis_sha256: str


def _optional_confidence(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not 0 <= float(value) <= 1):
        raise ContractViolation(field_name, f"expected finite 0..1 or null, got {value!r}")
    return float(value)


def _load_lesion_findings(data: dict) -> tuple[LesionFinding, ...]:
    raw = data.get("lesion_findings")
    if not isinstance(raw, list) or len(raw) != len(LESION_TYPES):
        raise ContractViolation("lesion_findings", "expected exactly 10 entries")
    findings: list[LesionFinding] = []
    seen: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ContractViolation(f"lesion_findings[{index}]", "expected an object")
        lesion_type = row.get("lesion_type")
        if lesion_type not in LESION_TYPE_SET:
            raise ContractViolation(
                f"lesion_findings[{index}].lesion_type", f"unknown {lesion_type!r}"
            )
        if lesion_type in seen:
            raise ContractViolation(
                f"lesion_findings[{index}].lesion_type", f"duplicate {lesion_type!r}"
            )
        seen.add(str(lesion_type))
        count = row.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ContractViolation(
                f"lesion_findings[{index}].count", "expected a non-negative integer"
            )
        regions = row.get("regions")
        if not isinstance(regions, list) or not all(isinstance(item, str) for item in regions):
            raise ContractViolation(
                f"lesion_findings[{index}].regions", "expected a string list"
            )
        mean = _optional_confidence(
            row.get("mean_detector_confidence"),
            f"lesion_findings[{index}].mean_detector_confidence",
        )
        maximum = _optional_confidence(
            row.get("max_detector_confidence"),
            f"lesion_findings[{index}].max_detector_confidence",
        )
        if count == 0 and (mean is not None or maximum is not None):
            raise ContractViolation(
                f"lesion_findings[{index}]", "zero-count finding must have null confidence"
            )
        if count > 0 and (mean is None or maximum is None or mean > maximum):
            raise ContractViolation(
                f"lesion_findings[{index}]", "detected finding requires mean <= max confidence"
            )
        source = row.get("evidence_source")
        if not isinstance(source, str) or not source:
            raise ContractViolation(
                f"lesion_findings[{index}].evidence_source", "expected a non-empty string"
            )
        findings.append(LesionFinding(
            str(lesion_type), count, tuple(regions), mean, maximum, source,
        ))
    if seen != LESION_TYPE_SET:
        raise ContractViolation("lesion_findings", f"missing {sorted(LESION_TYPE_SET - seen)}")
    return tuple(findings)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractViolation(field_name, "expected a string list")
    return tuple(value)


def _load_care_pathways(data: dict) -> tuple[CarePathway, ...]:
    raw = data.get("care_pathways")
    if not isinstance(raw, list) or len(raw) != len(LESION_TYPES):
        raise ContractViolation("care_pathways", "expected exactly 10 entries")
    pathways: list[CarePathway] = []
    seen: set[str] = set()
    allowed_roles = {"treatment", "sunscreen", "scar_care"}
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ContractViolation(f"care_pathways[{index}]", "expected an object")
        lesion_type = row.get("lesion_type")
        if lesion_type not in LESION_TYPE_SET or lesion_type in seen:
            raise ContractViolation(
                f"care_pathways[{index}].lesion_type", f"unknown or duplicate {lesion_type!r}"
            )
        seen.add(str(lesion_type))
        status = row.get("status")
        if status not in CARE_PATHWAY_STATUSES:
            raise ContractViolation(
                f"care_pathways[{index}].status", f"unknown {status!r}"
            )
        specs = row.get("retail_target_actives")
        if not isinstance(specs, list) or not all(isinstance(item, dict) for item in specs):
            raise ContractViolation(
                f"care_pathways[{index}].retail_target_actives", "expected an object list"
            )
        active_ids: list[str] = []
        for spec_index, spec in enumerate(specs):
            active_id = spec.get("active_id")
            if not isinstance(active_id, str) or not active_id:
                raise ContractViolation(
                    f"care_pathways[{index}].retail_target_actives[{spec_index}].active_id",
                    "expected a non-empty string",
                )
            active_ids.append(active_id)
            for field_name in ("strength", "formulation"):
                if not isinstance(spec.get(field_name), str) or not spec[field_name]:
                    raise ContractViolation(
                        f"care_pathways[{index}].retail_target_actives"
                        f"[{spec_index}].{field_name}",
                        "expected a non-empty audited value",
                    )
            minimum_age = spec.get("minimum_age_years")
            if minimum_age is not None and (
                not isinstance(minimum_age, int)
                or isinstance(minimum_age, bool)
                or minimum_age < 0
            ):
                raise ContractViolation(
                    f"care_pathways[{index}].retail_target_actives"
                    f"[{spec_index}].minimum_age_years",
                    "expected a non-negative integer or null",
                )
        roles = _string_tuple(
            row.get("required_product_roles"),
            f"care_pathways[{index}].required_product_roles",
        )
        if set(roles) - allowed_roles:
            raise ContractViolation(
                f"care_pathways[{index}].required_product_roles", "unknown product role"
            )
        clinician_options = row.get("clinician_options")
        if not isinstance(clinician_options, list) or not all(
            isinstance(item, dict) and isinstance(item.get("channel"), str)
            and isinstance(item.get("option"), str) for item in clinician_options
        ):
            raise ContractViolation(
                f"care_pathways[{index}].clinician_options", "expected channel/option objects"
            )
        reason_codes = _string_tuple(
            row.get("reason_codes"), f"care_pathways[{index}].reason_codes"
        )
        sources = _string_tuple(
            row.get("policy_source_ids"), f"care_pathways[{index}].policy_source_ids"
        )
        required_answers = _string_tuple(
            row.get("required_answers"), f"care_pathways[{index}].required_answers"
        )
        if status == "retail_eligible" and (not active_ids or not roles):
            raise ContractViolation(
                f"care_pathways[{index}]", "retail path requires an active and role"
            )
        if lesion_type in {"nevus", "other"} and (active_ids or roles):
            raise ContractViolation(
                f"care_pathways[{index}]", f"{lesion_type} cannot select products"
            )
        pathways.append(CarePathway(
            str(lesion_type), str(status), tuple(active_ids), tuple(dict(item) for item in specs),
            roles, tuple(dict(item) for item in clinician_options), reason_codes,
            sources, required_answers,
        ))
    if seen != LESION_TYPE_SET:
        raise ContractViolation("care_pathways", f"missing {sorted(LESION_TYPE_SET - seen)}")
    return tuple(pathways)


def load_analysis(path: str | Path, allow_unreviewed: bool = False) -> AnalysisInput:
    path = Path(path)
    raw_bytes = path.read_bytes()
    try:
        data = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ContractViolation("analysis", f"invalid JSON: {exc}") from exc
    schema_version = str(data.get("schema_version"))
    if schema_version not in {ANALYSIS_SCHEMA_VERSION, LEGACY_ANALYSIS_SCHEMA_VERSION}:
        raise ContractViolation(
            "schema_version",
            f"expected {ANALYSIS_SCHEMA_VERSION!r} (or legacy '3'), "
            f"got {data.get('schema_version')!r}",
        )

    concerns = []
    seen_concerns: set[str] = set()
    # Schema 4 keeps this field for display compatibility only.  Intentionally
    # do not parse or validate it: care and product logic cannot depend on it.
    raw_concerns = data.get("concerns") or [] if schema_version == "3" else []
    for i, c in enumerate(raw_concerns):
        name = c.get("concern")
        if name not in CONCERNS:
            raise ContractViolation(f"concerns[{i}].concern", f"unknown {name!r}")
        if name in seen_concerns:
            raise ContractViolation(f"concerns[{i}].concern", f"duplicate {name!r}")
        seen_concerns.add(name)
        severity = c.get("severity")
        if not isinstance(severity, int) or isinstance(severity, bool) or not 0 <= severity <= 4:
            raise ContractViolation(f"concerns[{i}].severity", f"expected int 0..4, got {severity!r}")
        confidence_raw = c.get("confidence")
        if confidence_raw is not None and (
            not isinstance(confidence_raw, (int, float)) or isinstance(confidence_raw, bool)
        ):
            raise ContractViolation(
                f"concerns[{i}].confidence", f"expected a number or null, got {confidence_raw!r}"
            )
        confidence = float(confidence_raw or 0.0)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ContractViolation(
                f"concerns[{i}].confidence", f"expected finite 0..1, got {confidence!r}"
            )
        concerns.append(ConcernFinding(
            concern=name,
            severity=severity,
            confidence=confidence,
            lesion_count=c.get("lesion_count"),
            regions=tuple(c.get("regions") or []),
            evidence_labels=dict((c.get("evidence") or {}).get("labels") or {}),
        ))

    decision = data.get("decision") or {}
    triage = decision.get("triage_level")
    if triage not in TRIAGE_LEVELS:
        raise ContractViolation("decision.triage_level", f"unknown {triage!r}")
    disposition = decision.get("therapy_disposition")
    if disposition not in THERAPY_DISPOSITIONS:
        raise ContractViolation("decision.therapy_disposition", f"unknown {disposition!r}")
    policy_reviewed = decision.get("policy_reviewed")
    if not isinstance(policy_reviewed, bool):
        raise ContractViolation("decision.policy_reviewed", "expected a boolean")
    if (schema_version == "3" and triage in {"derm_first", "abstain"}
            and disposition == "active_treatment"):
        raise ContractViolation(
            "decision.therapy_disposition",
            f"{triage} requires treatment to remain deferred",
        )
    policy_key = "lesion_care" if schema_version == "4" else "therapy"
    therapy_policy = (data.get("policies") or {}).get(policy_key)
    if not isinstance(therapy_policy, dict):
        raise ContractViolation(f"policies.{policy_key}", "expected an object")
    therapy_policy_reviewed = (
        therapy_policy.get("audit_approved") if schema_version == "4"
        else therapy_policy.get("reviewed")
    )
    if not isinstance(therapy_policy_reviewed, bool):
        raise ContractViolation(f"policies.{policy_key}.audit_approved", "expected a boolean")
    therapy_policy_identity = therapy_policy.get("identity")
    therapy_policy_sha256 = therapy_policy.get("sha256")
    if schema_version == "4":
        for key, expected in TRUSTED_SCHEMA4_POLICY.items():
            if therapy_policy.get(key) != expected:
                raise ContractViolation(
                    f"policies.lesion_care.{key}", "does not match the trusted MVP artifact"
                )
        if therapy_policy_reviewed is not True:
            raise ContractViolation(
                "policies.lesion_care.audit_approved", "trusted audit approval required"
            )
        if therapy_policy.get("scope_authorized") is not True:
            raise ContractViolation(
                "policies.lesion_care.scope_authorized",
                "synthetic fixture scope must be authorized",
            )
        if therapy_policy.get("input_scope") != "synthetic_fixture":
            raise ContractViolation(
                "policies.lesion_care.input_scope", "expected synthetic_fixture"
            )
        dataset = data.get("dataset")
        if not isinstance(dataset, dict) or dataset.get("name") not in {"fixture", "synthetic"}:
            raise ContractViolation("dataset.name", "expected a trusted fixture dataset")
        if dataset.get("split_proof") not in {"synthetic-test-fixture", "fixture"}:
            raise ContractViolation("dataset.split_proof", "expected fixture provenance")
        source_image_sha256 = data.get("source_image_sha256")
        if source_image_sha256 not in TRUSTED_SCHEMA4_IMAGE_SHA256S:
            raise ContractViolation(
                "source_image_sha256", "does not match an authorized fixture image"
            )
        if therapy_policy.get("fixture_image_sha256") != source_image_sha256:
            raise ContractViolation(
                "policies.lesion_care.fixture_image_sha256",
                "must match source_image_sha256",
            )
        input_profile = data.get("input_profile")
        if not isinstance(input_profile, dict):
            raise ContractViolation("input_profile", "expected an object")
        normalized_profile_sha256 = hashlib.sha256(json.dumps(
            input_profile, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if (
            therapy_policy.get("fixture_normalized_profile_sha256")
            != normalized_profile_sha256
            or (
                therapy_policy.get("fixture_profile_sha256"),
                normalized_profile_sha256,
            ) not in TRUSTED_SCHEMA4_PROFILE_PAIRS
        ):
            raise ContractViolation(
                "policies.lesion_care.fixture_profile_sha256",
                "raw and resolved profile hashes are not an authorized pair",
            )

    plan = data.get("therapy_plan")
    if not isinstance(plan, dict):
        raise ContractViolation("therapy_plan", "expected an object")
    support_roles = plan.get("support_roles")
    required_support_roles = {"cleanser", "moisturizer", "sunscreen"}
    if (
        not isinstance(support_roles, list)
        or len(support_roles) != len(required_support_roles)
        or set(support_roles) != required_support_roles
    ):
        raise ContractViolation(
            "therapy_plan.support_roles",
            "expected cleanser, moisturizer, and sunscreen exactly once",
        )
    deferred = plan.get("deferred_reasons")
    if not isinstance(deferred, list) or not all(isinstance(x, str) for x in deferred):
        raise ContractViolation("therapy_plan.deferred_reasons", "expected string list")
    primary = plan.get("primary")
    if schema_version == "4" and primary is not None:
        raise ContractViolation(
            "therapy_plan.primary", "schema 4 uses plural exact-label pathways"
        )
    if primary is not None:
        if not isinstance(primary, dict):
            raise ContractViolation("therapy_plan.primary", "expected object or null")
        required = ("therapy", "strength_band", "exposure", "cadence", "role")
        missing = [key for key in required
                   if not isinstance(primary.get(key), str) or not primary[key]]
        if missing:
            raise ContractViolation("therapy_plan.primary", f"missing fields {missing}")
        if primary["role"] != "treatment":
            raise ContractViolation(
                "therapy_plan.primary.role", "recsys supports treatment intent only"
            )
        if disposition != "active_treatment":
            raise ContractViolation(
                "therapy_plan.primary",
                "primary treatment requires active_treatment disposition",
            )
        if not allow_unreviewed:
            if not therapy_policy_reviewed:
                raise ContractViolation(
                    "therapy_plan.primary",
                    "primary treatment requires a reviewed therapy policy",
                )
            if not isinstance(therapy_policy_identity, str) or not therapy_policy_identity:
                raise ContractViolation(
                    "policies.therapy.identity", "reviewed policy requires a named identity"
                )
            if (
                not isinstance(therapy_policy_sha256, str)
                or len(therapy_policy_sha256) != 64
                or any(c not in "0123456789abcdef" for c in therapy_policy_sha256.lower())
            ):
                raise ContractViolation(
                    "policies.therapy.sha256", "reviewed policy requires a sha256 digest"
                )
        if not isinstance(primary.get("cadence_source"), str) or not primary["cadence_source"]:
            raise ContractViolation(
                "therapy_plan.primary.cadence_source", "expected a named source"
            )
        amount = primary.get("amount")
        if amount is not None and (not isinstance(amount, str) or not amount):
            raise ContractViolation(
                "therapy_plan.primary.amount", "expected a non-empty string or null"
            )
        if amount is not None and (
            not isinstance(primary.get("amount_source"), str)
            or not primary["amount_source"]
        ):
            raise ContractViolation(
                "therapy_plan.primary.amount_source",
                "required when amount is specified",
            )
        if not isinstance(plan.get("policy_version"), str) or not plan["policy_version"]:
            raise ContractViolation("therapy_plan.policy_version", "expected a named policy")
        if not allow_unreviewed and plan["policy_version"] != therapy_policy_identity:
            raise ContractViolation(
                "therapy_plan.policy_version",
                "must match policies.therapy.identity",
            )

    bucket = (data.get("skin_tone") or {}).get("bucket", "unknown")
    if bucket not in TONE_BUCKETS:
        raise ContractViolation("skin_tone.bucket", f"unknown {bucket!r}")

    if schema_version == "4":
        lesion_findings = _load_lesion_findings(data)
        care_pathways = _load_care_pathways(data)
        if {item.lesion_type for item in lesion_findings} != {
            item.lesion_type for item in care_pathways
        }:
            raise ContractViolation("care_pathways", "must match lesion_findings labels")
    else:
        # One-release schema-3 migration: derive exact labels only from the
        # detector-label counts already embedded in legacy evidence.  The
        # grouped concern name itself is never consulted downstream.
        migrated: list[LesionFinding] = []
        for lesion_type in LESION_TYPES:
            rows = [item for item in concerns if item.evidence_labels.get(lesion_type, 0)]
            count = sum(item.evidence_labels.get(lesion_type, 0) for item in rows)
            confidences = [item.confidence for item in rows]
            migrated.append(LesionFinding(
                lesion_type=lesion_type,
                count=count,
                regions=tuple(sorted({region for item in rows for region in item.regions})),
                mean_detector_confidence=(
                    sum(confidences) / len(confidences) if confidences else None
                ),
                max_detector_confidence=max(confidences) if confidences else None,
                evidence_source="legacy_schema3_detector_labels",
            ))
        lesion_findings = tuple(migrated)
        primary_active = primary.get("therapy") if isinstance(primary, dict) else None
        care_pathways = tuple(CarePathway(
            lesion_type=item.lesion_type,
            status=(
                "retail_eligible"
                if item.count and primary_active and item.lesion_type in {
                    "closed_comedo", "open_comedo", "papule", "pustule"
                }
                else "clinician_only" if item.count else "not_detected"
            ),
            retail_target_actives=((str(primary_active),) if item.count and primary_active
                                    and item.lesion_type in {
                                        "closed_comedo", "open_comedo", "papule", "pustule"
                                    } else ()),
            retail_target_specs=(({"active_id": str(primary_active)},)
                                 if item.count and primary_active and item.lesion_type in {
                                     "closed_comedo", "open_comedo", "papule", "pustule"
                                 } else ()),
            required_product_roles=(("treatment",) if item.count and primary_active
                                    and item.lesion_type in {
                                        "closed_comedo", "open_comedo", "papule", "pustule"
                                    } else ()),
            clinician_options=(),
            reason_codes=("schema3_exact_label_migration",) if item.count else (),
            policy_source_ids=(),
            required_answers=(),
        ) for item in lesion_findings)

    observations = tuple(
        {"code": o.get("code"), "professional_review": bool(o.get("professional_review"))}
        for o in (data.get("safety_observations") or [])
    )
    return AnalysisInput(
        schema_version=schema_version,
        lesion_findings=lesion_findings,
        care_pathways=care_pathways,
        concerns=tuple(concerns),
        skin_tone_bucket=bucket,
        safety_observations=observations,
        triage_level=triage,
        referral_reasons=tuple(decision.get("referral_reasons") or []),
        therapy_disposition=disposition,
        policy_reviewed=policy_reviewed,
        therapy_policy_reviewed=therapy_policy_reviewed,
        therapy_policy_identity=therapy_policy_identity,
        therapy_policy_sha256=therapy_policy_sha256,
        therapy_plan=dict(plan),
        therapy_primary=dict(primary) if primary is not None else None,
        therapy_support_roles=tuple(support_roles),
        therapy_deferred_reasons=tuple(deferred),
        input_profile=dict(data.get("input_profile") or {}),
        source_image_sha256=data.get("source_image_sha256"),
        generated_at=data.get("generated_at"),
        analysis_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


@dataclass(frozen=True)
class Profile:
    skin_type: str = "unknown"
    tone_bucket: str = "unknown"
    tone_source: str = "unknown"
    pregnancy_status: str = "unknown"
    age_years: int | None = None
    allergies: tuple[str, ...] = ()
    sensitivity_conditions: tuple[str, ...] = ()
    current_actives: tuple[str, ...] = ()
    current_medications: tuple[str, ...] = ()
    treatment_history: tuple[str, ...] = ()
    acne_duration_weeks: int | None = None
    finding_duration_weeks: int | None = None
    painful_or_deep_lesions: bool | None = None
    prior_scarring: bool | None = None
    spot_new_or_changing: bool | None = None
    spot_bleeding_itching_or_painful: bool | None = None
    spot_bleeding: bool | None = None
    spot_itching: bool | None = None
    spot_painful: bool | None = None
    spot_other_symptoms: bool | None = None
    active_acne_controlled: bool | None = None
    scar_duration_months: int | None = None
    pregnancy_or_hormonal_medication_onset: bool | None = None
    abcde_change_present: bool | None = None
    wound_closed: bool | None = None
    scar_diagnosis_confirmed_by_clinician: bool | None = None
    max_price_usd: float | None = None
    unknown_fields: frozenset[str] = frozenset()
    source: str = "unknown"  # "file" | "analysis.input_profile" | "unknown"
    profile_sha256: str | None = None


def _profile_list(
    data: dict, field_name: str, allowed: frozenset[str] | None = None
) -> tuple[str, ...]:
    """Free text by default: allergies are matched against raw INCI by
    inci.allergy_matches, which normalizes and resolves synonyms itself.
    `allowed` closes the vocabulary for fields matched by exact identity
    downstream, where an unrecognized value fails a gate open instead of loud.
    """
    value = data.get(field_name)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractViolation(f"profile.{field_name}", "expected a list of strings")
    if allowed is not None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ContractViolation(f"profile.{field_name}", f"unknown active IDs {unknown}")
    return tuple(value)


def _profile_optional_int(data: dict, field_name: str, *, maximum: int | None = None) -> int | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractViolation(f"profile.{field_name}", "expected an integer or null")
    if value < 0 or (maximum is not None and value > maximum):
        limit = f"0..{maximum}" if maximum is not None else "non-negative"
        raise ContractViolation(f"profile.{field_name}", f"expected {limit} integer or null")
    return value


def _profile_optional_bool(data: dict, field_name: str) -> bool | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ContractViolation(f"profile.{field_name}", "expected boolean or null")
    return value


def _profile_from_dict(data: dict, source: str, sha256: str | None) -> Profile:
    if not isinstance(data, dict):
        raise ContractViolation("profile", "expected an object")
    skin_type = data.get("skin_type") or "unknown"
    if skin_type not in SKIN_TYPES:
        raise ContractViolation("profile.skin_type", f"unknown {skin_type!r}")
    tone = data.get("tone_bucket") or "unknown"
    if tone not in TONE_BUCKETS:
        raise ContractViolation("profile.tone_bucket", f"unknown {tone!r}")
    tone_source = data.get("tone_source") or "unknown"
    if tone_source not in TONE_SOURCES:
        raise ContractViolation("profile.tone_source", f"unknown {tone_source!r}")
    pregnancy = data.get("pregnancy_status") or "unknown"
    if pregnancy not in PREGNANCY_STATUSES:
        raise ContractViolation("profile.pregnancy_status", f"unknown {pregnancy!r}")
    price = data.get("max_price_usd")
    if price is not None and (
        not isinstance(price, (int, float))
        or isinstance(price, bool)
        or not math.isfinite(price)
        or price < 0
    ):
        raise ContractViolation(
            "profile.max_price_usd", "expected a finite non-negative number or null"
        )
    declared_unknowns = data.get("unknown_fields") or []
    if not isinstance(declared_unknowns, list) or not all(
        isinstance(item, str) for item in declared_unknowns
    ):
        raise ContractViolation("profile.unknown_fields", "expected a string list")
    return Profile(
        skin_type=skin_type,
        tone_bucket=tone,
        tone_source=tone_source,
        pregnancy_status=pregnancy,
        age_years=_profile_optional_int(data, "age_years", maximum=130),
        allergies=_profile_list(data, "allergies"),
        sensitivity_conditions=_profile_list(data, "sensitivity_conditions"),
        current_actives=_profile_list(data, "current_actives", allowed=DECLARABLE_ACTIVE_IDS),
        current_medications=_profile_list(data, "current_medications"),
        treatment_history=_profile_list(data, "treatment_history"),
        acne_duration_weeks=_profile_optional_int(data, "acne_duration_weeks"),
        finding_duration_weeks=_profile_optional_int(data, "finding_duration_weeks"),
        painful_or_deep_lesions=_profile_optional_bool(data, "painful_or_deep_lesions"),
        prior_scarring=_profile_optional_bool(data, "prior_scarring"),
        spot_new_or_changing=_profile_optional_bool(data, "spot_new_or_changing"),
        spot_bleeding_itching_or_painful=_profile_optional_bool(
            data, "spot_bleeding_itching_or_painful"
        ),
        spot_bleeding=_profile_optional_bool(data, "spot_bleeding"),
        spot_itching=_profile_optional_bool(data, "spot_itching"),
        spot_painful=_profile_optional_bool(data, "spot_painful"),
        spot_other_symptoms=_profile_optional_bool(data, "spot_other_symptoms"),
        active_acne_controlled=_profile_optional_bool(data, "active_acne_controlled"),
        scar_duration_months=_profile_optional_int(data, "scar_duration_months"),
        pregnancy_or_hormonal_medication_onset=_profile_optional_bool(
            data, "pregnancy_or_hormonal_medication_onset"
        ),
        abcde_change_present=_profile_optional_bool(data, "abcde_change_present"),
        wound_closed=_profile_optional_bool(data, "wound_closed"),
        scar_diagnosis_confirmed_by_clinician=_profile_optional_bool(
            data, "scar_diagnosis_confirmed_by_clinician"
        ),
        max_price_usd=float(price) if price is not None else None,
        unknown_fields=frozenset(set(declared_unknowns) | {
            field_name for field_name in (
                "allergies", "sensitivity_conditions", "current_actives",
                "current_medications", "treatment_history",
            ) if field_name not in data or data.get(field_name) is None
        }),
        source=source,
        profile_sha256=sha256,
    )


def resolve_profile(profile_path: str | Path | None, analysis: AnalysisInput) -> Profile:
    """Precedence: explicit --profile file > analysis.input_profile > all-unknown.
    Missing fields are unknown; unknowns fail SAFE downstream (see gates.py)."""
    if profile_path is not None:
        raw = Path(profile_path).read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractViolation("profile", f"invalid JSON: {exc}") from exc
        return _profile_from_dict(data, "file", hashlib.sha256(raw).hexdigest())
    if analysis.input_profile:
        return _profile_from_dict(analysis.input_profile, "analysis.input_profile", None)
    return _profile_from_dict({}, "unknown", None)
