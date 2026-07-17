"""Exact-label lesion care policy for the synthetic MVP.

This module is deliberately product independent.  It translates the audited
ten-label research artifact into analysis-schema-4 findings and pathways.  The
standalone recommender owns every SKU decision.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping


LESION_TYPES = (
    "closed_comedo",
    "open_comedo",
    "papule",
    "pustule",
    "nodule",
    "atrophic_scar",
    "hypertrophic_scar",
    "melasma",
    "nevus",
    "other",
)
LESION_TYPE_SET = frozenset(LESION_TYPES)

# Trust root for the only fixture manifest the synthetic MVP is allowed to
# execute. A caller-provided manifest is data, not authority.
TRUSTED_MVP_FIXTURE_MANIFEST_SHA256 = (
    "10d018abc93ffb84a9ecee0b79cb16dbeaea92ff68108f9f3222f592fd001508"
)
TRUSTED_LESION_CARE_POLICY_SHA256 = (
    "8d24399f076e7721a96b1fc6cf59174bb94390b655a15eb326c8d3a722ffbae1"
)
TRUSTED_LESION_CARE_REPORT_SHA256 = (
    "8e201d6d869a67e8635f29cc43452138e760561ae0e34d2b2b4b8ba1d6e4344d"
)
TRUSTED_LESION_CARE_SOURCE_MANIFEST_SHA256 = (
    "5f6f69e5d468ce947ae2eb6e321a8476267dd8cf1381758a2017011e97c50a2f"
)

PATHWAY_STATUSES = frozenset({
    "not_detected",
    "retail_eligible",
    "clinician_only",
    "deferred",
    "monitoring_only",
    "unsupported",
})

_SYMPTOM_FIELDS = (
    "spot_bleeding",
    "spot_itching",
    "spot_painful",
    "spot_other_symptoms",
)


@dataclass(frozen=True)
class LesionCarePolicy:
    policy_id: str
    version: str
    labels: dict[str, dict]
    source_path: str
    sha256: str
    report_sha256: str
    manifest_sha256: str
    audit_approved: bool
    scope_authorized: bool
    scope_reasons: tuple[str, ...]
    intake_contract: dict

    @property
    def identity(self) -> str:
        return f"{self.policy_id}:{self.version}"


@dataclass(frozen=True)
class MvpFixtureAuthorization:
    authorized: bool
    manifest_sha256: str | None
    reasons: tuple[str, ...]
    image_sha256: str | None = None
    profile_sha256: str | None = None
    normalized_profile_sha256: str | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping_sha256(value: Mapping[str, object]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected a non-empty string")
    return value


def authorize_mvp_fixture_inputs(
    manifest_path: str | Path | None,
    *,
    image_bytes: bytes,
    profile_path: str | Path | None,
    environment: str | None,
    dataset_name: str,
    split_proof: str | None,
    normalized_profile: Mapping[str, object] | None = None,
) -> MvpFixtureAuthorization:
    """Authorize only hash-pinned synthetic fixtures, never a bare CLI claim."""
    reasons: list[str] = []
    if manifest_path is None:
        return MvpFixtureAuthorization(False, None, ("fixture_manifest_required",))
    path = Path(manifest_path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return MvpFixtureAuthorization(False, None, ("fixture_manifest_missing",))
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    profile_sha256 = _sha256(Path(profile_path)) if profile_path is not None else None
    normalized_profile_sha256 = (
        _mapping_sha256(normalized_profile) if normalized_profile is not None else None
    )
    if manifest_sha256 != TRUSTED_MVP_FIXTURE_MANIFEST_SHA256:
        return MvpFixtureAuthorization(
            authorized=False,
            manifest_sha256=manifest_sha256,
            reasons=("fixture_manifest_not_trusted",),
            image_sha256=image_sha256,
            profile_sha256=profile_sha256,
            normalized_profile_sha256=normalized_profile_sha256,
        )
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MVP fixture manifest {path}: invalid JSON: {exc}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != (
        "skinscan-mvp-fixtures-1"
    ):
        raise ValueError("MVP fixture manifest: unsupported schema")
    if environment not in set(manifest.get("authorized_environments") or []):
        reasons.append(f"fixture_environment_not_authorized:{environment or 'unknown'}")
    if dataset_name not in set(manifest.get("dataset_names") or []):
        reasons.append(f"fixture_dataset_not_authorized:{dataset_name}")
    if split_proof not in set(manifest.get("split_proofs") or []):
        reasons.append("fixture_split_proof_not_authorized")
    if image_sha256 not in set(manifest.get("image_sha256s") or []):
        reasons.append("fixture_image_hash_not_authorized")
    raw_profile_authorizations = manifest.get("profile_authorizations")
    if not isinstance(raw_profile_authorizations, list) or not all(
        isinstance(item, Mapping)
        and isinstance(item.get("raw_sha256"), str)
        and isinstance(item.get("normalized_sha256"), str)
        for item in raw_profile_authorizations
    ):
        raise ValueError("MVP fixture manifest: invalid profile_authorizations")
    profile_authorizations = {
        (item["raw_sha256"], item["normalized_sha256"])
        for item in raw_profile_authorizations
    }
    if profile_path is None:
        reasons.append("fixture_profile_file_required")
    if normalized_profile is None:
        reasons.append("fixture_normalized_profile_required")
    if (
        profile_path is not None
        and normalized_profile is not None
        and (profile_sha256, normalized_profile_sha256) not in profile_authorizations
    ):
        reasons.append("fixture_profile_pair_not_authorized")
    return MvpFixtureAuthorization(
        authorized=not reasons,
        manifest_sha256=manifest_sha256,
        reasons=tuple(reasons),
        image_sha256=image_sha256,
        profile_sha256=profile_sha256,
        normalized_profile_sha256=normalized_profile_sha256,
    )


def load_lesion_care_policy(
    path: str | Path,
    *,
    report_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    environment: str | None = None,
    input_types: Iterable[str] = (),
    scope_prerequisite_reasons: Iterable[str] = (),
) -> LesionCarePolicy:
    """Load and validate the audited policy without widening its MVP scope."""
    path = Path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"lesion care policy {path}: invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("lesion_care_policy: expected an object")
    if value.get("schema_version") != "lesion-care-proposal-1":
        raise ValueError("lesion_care_policy.schema_version: unsupported")

    raw_labels = value.get("labels")
    if not isinstance(raw_labels, list) or len(raw_labels) != len(LESION_TYPES):
        raise ValueError("lesion_care_policy.labels: expected exactly 10 entries")
    labels: dict[str, dict] = {}
    for index, row in enumerate(raw_labels):
        if not isinstance(row, Mapping):
            raise ValueError(f"lesion_care_policy.labels[{index}]: expected an object")
        lesion_type = row.get("lesion_type")
        if lesion_type not in LESION_TYPE_SET:
            raise ValueError(
                f"lesion_care_policy.labels[{index}].lesion_type: unknown {lesion_type!r}"
            )
        if lesion_type in labels:
            raise ValueError(f"lesion_care_policy.labels: duplicate {lesion_type!r}")
        care_path = row.get("care_path")
        if not isinstance(care_path, Mapping):
            raise ValueError(
                f"lesion_care_policy.labels[{index}].care_path: expected an object"
            )
        source_ids = row.get("source_ids")
        if (not isinstance(source_ids, list) or not source_ids
                or not all(isinstance(item, str) and item for item in source_ids)):
            raise ValueError(
                f"lesion_care_policy.labels[{index}].source_ids: "
                "expected at least one authoritative source or abstention source"
            )
        labels[str(lesion_type)] = dict(row)
    if set(labels) != LESION_TYPE_SET:
        missing = sorted(LESION_TYPE_SET - set(labels))
        raise ValueError(f"lesion_care_policy.labels: missing {missing}")

    for lesion_type in ("nevus", "other"):
        path_value = labels[lesion_type]["care_path"]
        if path_value.get("retail_target_actives") or path_value.get("required_product_roles"):
            raise ValueError(f"lesion_care_policy.{lesion_type}: product matching is forbidden")

    audit = value.get("ai_research_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("lesion_care_policy.ai_research_audit: expected an object")
    audit_approved = audit.get("approved") is True
    if audit.get("approved_policy_version") != value.get("version"):
        audit_approved = False
    report_sha256 = _require_string(
        audit.get("report_sha256"), "lesion_care_policy.ai_research_audit.report_sha256"
    )
    if (
        report_sha256 != TRUSTED_LESION_CARE_REPORT_SHA256
        or report_path is None
        or _sha256(Path(report_path)) != TRUSTED_LESION_CARE_REPORT_SHA256
    ):
        audit_approved = False

    policy_sha256 = hashlib.sha256(raw).hexdigest()
    if policy_sha256 != TRUSTED_LESION_CARE_POLICY_SHA256:
        audit_approved = False
    manifest_path = Path(manifest_path) if manifest_path is not None else (
        path.parent / "lesion_care_source_manifest.json"
    )
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
    except FileNotFoundError:
        manifest = None
        manifest_raw = b""
        audit_approved = False
    except json.JSONDecodeError as exc:
        raise ValueError(f"lesion care manifest {manifest_path}: invalid JSON: {exc}") from exc
    if not isinstance(manifest, Mapping):
        audit_approved = False
    else:
        if hashlib.sha256(manifest_raw).hexdigest() != (
            TRUSTED_LESION_CARE_SOURCE_MANIFEST_SHA256
        ):
            audit_approved = False
        manifest_audit = manifest.get("ai_research_audit")
        if not isinstance(manifest_audit, Mapping):
            audit_approved = False
        else:
            audit_approved = audit_approved and all((
                manifest_audit.get("approved") is True,
                manifest_audit.get("mvp_development_changes_authorized") is True,
                manifest_audit.get("production_changes_authorized") is False,
                manifest_audit.get("real_user_use_authorized") is False,
                manifest_audit.get("policy_version") == value.get("version"),
                manifest_audit.get("policy_sha256") == policy_sha256,
                manifest_audit.get("report_sha256") == report_sha256,
            ))

    authorized_environments = set(value.get("authorized_environments") or [])
    authorized_inputs = set(value.get("authorized_input_types") or [])
    requested_inputs = set(input_types)
    scope_reasons: list[str] = list(scope_prerequisite_reasons)
    if not audit_approved:
        scope_reasons.append("mvp_ai_research_audit_not_valid")
    if value.get("mvp_synthetic_eligible") is not True or value.get("test_only") is not True:
        scope_reasons.append("policy_not_synthetic_mvp_eligible")
    if environment not in authorized_environments:
        scope_reasons.append(f"environment_not_authorized:{environment or 'unknown'}")
    for input_type in sorted(requested_inputs - authorized_inputs):
        scope_reasons.append(f"input_type_not_authorized:{input_type}")
    if not requested_inputs or not {"synthetic_profile", "fixture_image"}.issubset(
        requested_inputs
    ):
        scope_reasons.append("synthetic_profile_and_fixture_image_required")
    if value.get("production_eligible") is not False:
        scope_reasons.append("production_scope_must_remain_disabled")

    intake_contract = value.get("intake_contract")
    if not isinstance(intake_contract, Mapping):
        raise ValueError("lesion_care_policy.intake_contract: expected an object")
    return LesionCarePolicy(
        policy_id=_require_string(value.get("policy_id"), "lesion_care_policy.policy_id"),
        version=_require_string(value.get("version"), "lesion_care_policy.version"),
        labels=labels,
        source_path=str(path),
        sha256=policy_sha256,
        report_sha256=report_sha256,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        audit_approved=audit_approved,
        scope_authorized=not scope_reasons,
        scope_reasons=tuple(scope_reasons),
        intake_contract=dict(intake_contract),
    )


def build_lesion_findings(
    observations: Iterable[object], *, evidence_source: str
) -> list[dict]:
    """Aggregate retained detector observations without collapsing labels."""
    grouped: dict[str, list[object]] = defaultdict(list)
    for observation in observations:
        label = getattr(observation, "label", None)
        if label in LESION_TYPE_SET:
            grouped[label].append(observation)

    findings: list[dict] = []
    for lesion_type in LESION_TYPES:
        rows = grouped.get(lesion_type, [])
        confidences = [float(getattr(row, "score", 0.0)) for row in rows]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidences):
            raise ValueError(f"lesion_findings.{lesion_type}: invalid detector confidence")
        findings.append({
            "lesion_type": lesion_type,
            "count": len(rows),
            "regions": sorted({
                str(region) for region in (getattr(row, "region", None) for row in rows)
                if region
            }),
            "mean_detector_confidence": (
                round(sum(confidences) / len(confidences), 6) if confidences else None
            ),
            "max_detector_confidence": round(max(confidences), 6) if confidences else None,
            "evidence_source": evidence_source,
        })
    return findings


def _is_unknown(profile: Mapping[str, object], field_name: str) -> bool:
    declared_unknowns = profile.get("unknown_fields") or []
    if isinstance(declared_unknowns, (list, tuple, set)) and field_name in declared_unknowns:
        return True
    if field_name not in profile:
        return True
    value = profile.get(field_name)
    return value is None or value == "unknown"


def _unknown_required_answers(
    lesion_type: str,
    profile: Mapping[str, object],
    policy: LesionCarePolicy,
    *,
    include_policy_critical: bool,
) -> list[str]:
    unknown: list[str] = []
    if include_policy_critical:
        for field_name in policy.intake_contract.get("policy_critical_required") or []:
            # Existing acne duration is the compatibility name for finding duration.
            if field_name == "finding_duration_weeks" and not _is_unknown(
                profile, "acne_duration_weeks"
            ):
                continue
            if _is_unknown(profile, str(field_name)):
                unknown.append(str(field_name))

    conditional = policy.intake_contract.get("conditional") or {}
    fields = conditional.get(lesion_type) or []
    for field_name in fields:
        field_name = str(field_name)
        if field_name == "spot_bleeding_itching_or_painful":
            if not _is_unknown(profile, field_name):
                continue
            unknown.extend(
                item for item in _SYMPTOM_FIELDS if _is_unknown(profile, item)
            )
        elif _is_unknown(profile, field_name):
            unknown.append(field_name)
    return list(dict.fromkeys(unknown))


def _normalized_target_actives(care: Mapping[str, object]) -> list[dict]:
    """Add structured safety facts already stated in the audited policy text."""
    normalized: list[dict] = []
    for raw in care.get("retail_target_actives") or []:
        spec = dict(raw)
        eligibility = str(spec.get("eligibility") or "")
        match = re.search(r"\bage\s+(\d+)\s+or\s+older\b", eligibility, re.I)
        if match:
            spec["minimum_age_years"] = int(match.group(1))
        normalized.append(spec)
    return normalized


def _age_filter_actives(
    target_actives: list[dict], profile: Mapping[str, object]
) -> tuple[list[dict], list[str]]:
    age = profile.get("age_years")
    if not isinstance(age, int) or isinstance(age, bool):
        return target_actives, []
    eligible: list[dict] = []
    reasons: list[str] = []
    for spec in target_actives:
        minimum = spec.get("minimum_age_years")
        if isinstance(minimum, int) and age < minimum:
            reasons.append(
                f"active_age_excluded:{spec.get('active_id', 'unknown')}:{minimum}"
            )
        else:
            eligible.append(spec)
    return eligible, reasons


def _clinician_channel(option: str) -> str:
    lower = option.lower()
    if any(word in lower for word in (
        "laser", "microneedling", "surgery", "injection", "cryotherapy",
        "filler", "punch", "biopsy", "dermoscop", "procedure", "removal",
    )):
        return "procedure"
    if any(word in lower for word in (
        "retinoid", "isotretinoin", "hydroquinone", "prescription", "oral therapy",
        "systemic therapy", "azelaic acid", "combination therapy",
    )):
        return "prescription_discussion"
    return "clinician_discussion"


def build_care_pathways(
    findings: Iterable[Mapping[str, object]],
    profile: Mapping[str, object],
    policy: LesionCarePolicy,
) -> list[dict]:
    """Translate each exact finding into one independent, product-free path."""
    by_type = {str(item.get("lesion_type")): item for item in findings}
    pathways: list[dict] = []
    for lesion_type in LESION_TYPES:
        finding = by_type[lesion_type]
        count = int(finding.get("count") or 0)
        row = policy.labels[lesion_type]
        care = row["care_path"]
        target_actives = _normalized_target_actives(care)
        required_roles = list(care.get("required_product_roles") or [])
        unknowns = _unknown_required_answers(
            lesion_type,
            profile,
            policy,
            include_policy_critical=bool(target_actives and required_roles),
        ) if count else []
        reason_codes = list(care.get("reason_codes") or [])
        reason_codes.extend(f"required_intake_unknown:{name}" for name in unknowns)
        target_actives, age_reasons = _age_filter_actives(target_actives, profile)
        reason_codes.extend(age_reasons)

        if count == 0:
            status = "not_detected"
        elif not policy.scope_authorized:
            status = "deferred"
            reason_codes.extend(policy.scope_reasons)
        elif lesion_type == "nevus":
            status = "monitoring_only"
        elif lesion_type == "other":
            status = "unsupported"
        elif not target_actives or not required_roles:
            status = "clinician_only"
        elif unknowns:
            status = "deferred"
        else:
            status = "retail_eligible"

        pathways.append({
            "lesion_type": lesion_type,
            "status": status,
            "retail_target_actives": target_actives if status == "retail_eligible" else [],
            "required_product_roles": required_roles if status == "retail_eligible" else [],
            "clinician_options": [
                {"channel": _clinician_channel(str(option)), "option": str(option)}
                for option in care.get("clinician_options") or []
            ],
            "reason_codes": list(dict.fromkeys(reason_codes)),
            "policy_source_ids": list(row.get("source_ids") or []),
            "required_answers": unknowns,
            "policy_id": policy.identity,
        })
    return pathways


def _evidence_quality(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    if value >= 0.8:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


def decide_exact_label_care(
    findings: Iterable[Mapping[str, object]], pathways: Iterable[Mapping[str, object]],
) -> dict:
    """Derive referral and retail axes from exact labels only."""
    finding_rows = [row for row in findings if int(row.get("count") or 0) > 0]
    path_by_type = {str(row.get("lesion_type")): row for row in pathways}
    detected = {str(row.get("lesion_type")) for row in finding_rows}
    referrals: list[str] = []
    if "nodule" in detected:
        referrals.append("nodule_clinician_assessment")
    if detected & {"atrophic_scar", "hypertrophic_scar"}:
        referrals.append("scar_subtype_assessment")
    if "melasma" in detected:
        referrals.append("pigment_diagnosis_confirmation")
    if "nevus" in detected:
        referrals.append("nevus_monitoring_or_assessment")
    if "other" in detected:
        referrals.append("unsupported_finding_assessment")

    statuses = {path_by_type[label]["status"] for label in detected}
    if "nodule" in detected:
        triage = "derm_first"
    elif referrals:
        triage = "routine_plus_review"
    else:
        triage = "routine"
    if "retail_eligible" in statuses:
        disposition = "active_treatment"
    elif "deferred" in statuses:
        disposition = "defer"
    elif detected & {"nodule", "atrophic_scar", "hypertrophic_scar"}:
        disposition = "supportive_only"
    else:
        disposition = "maintenance"

    return {
        "triage_level": triage,
        "referral_reasons": referrals,
        "therapy_disposition": disposition,
        "decision_evidence": [
            {
                "lesion_type": str(row["lesion_type"]),
                "probability": None,
                "quality": _evidence_quality(row.get("max_detector_confidence")),
                "source": row.get("evidence_source"),
                "calibrated": False,
                "reasons": ["exact_detector_label", f"lesion_count_{row['count']}"],
            }
            for row in finding_rows
        ],
        "policy_reviewed": False,
    }


def exact_label_therapy_plan(pathways: Iterable[Mapping[str, object]], policy: LesionCarePolicy) -> dict:
    """Compatibility envelope; exact paths remain plural and product independent."""
    detected = [row for row in pathways if row.get("status") != "not_detected"]
    return {
        "policy_version": policy.identity,
        "lesion_types": [row["lesion_type"] for row in detected],
        "primary": None,
        "alternatives": [],
        "support_roles": ["cleanser", "moisturizer", "sunscreen"],
        "deferred_reasons": [
            reason
            for row in detected if row.get("status") == "deferred"
            for reason in row.get("reason_codes") or []
        ],
        "clinician_options": [
            {"lesion_type": row["lesion_type"], **option}
            for row in detected
            for option in row.get("clinician_options") or []
        ],
    }
