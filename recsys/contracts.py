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
ANALYSIS_SCHEMA_VERSION = "3"

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
class AnalysisInput:
    concerns: tuple[ConcernFinding, ...]
    skin_tone_bucket: str
    safety_observations: tuple[dict, ...]  # {code, professional_review} kept verbatim
    triage_level: str
    referral_reasons: tuple[str, ...]
    therapy_disposition: str
    policy_reviewed: bool
    therapy_policy_reviewed: bool
    therapy_plan: dict
    therapy_primary: dict | None
    therapy_support_roles: tuple[str, ...]
    therapy_deferred_reasons: tuple[str, ...]
    input_profile: dict
    source_image_sha256: str | None
    generated_at: str | None
    analysis_sha256: str


def load_analysis(path: str | Path) -> AnalysisInput:
    path = Path(path)
    raw_bytes = path.read_bytes()
    try:
        data = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ContractViolation("analysis", f"invalid JSON: {exc}") from exc
    if str(data.get("schema_version")) != ANALYSIS_SCHEMA_VERSION:
        raise ContractViolation(
            "schema_version",
            f"expected {ANALYSIS_SCHEMA_VERSION!r}, got {data.get('schema_version')!r}",
        )

    concerns = []
    seen_concerns: set[str] = set()
    for i, c in enumerate(data.get("concerns") or []):
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
    if triage in {"derm_first", "abstain"} and disposition == "active_treatment":
        raise ContractViolation(
            "decision.therapy_disposition",
            f"{triage} requires treatment to remain deferred",
        )
    therapy_policy = (data.get("policies") or {}).get("therapy")
    if not isinstance(therapy_policy, dict):
        raise ContractViolation("policies.therapy", "expected an object")
    therapy_policy_reviewed = therapy_policy.get("reviewed")
    if not isinstance(therapy_policy_reviewed, bool):
        raise ContractViolation("policies.therapy.reviewed", "expected a boolean")
    therapy_policy_identity = therapy_policy.get("identity")
    therapy_policy_sha256 = therapy_policy.get("sha256")

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
        if plan["policy_version"] != therapy_policy_identity:
            raise ContractViolation(
                "therapy_plan.policy_version",
                "must match policies.therapy.identity",
            )

    bucket = (data.get("skin_tone") or {}).get("bucket", "unknown")
    if bucket not in TONE_BUCKETS:
        raise ContractViolation("skin_tone.bucket", f"unknown {bucket!r}")

    observations = tuple(
        {"code": o.get("code"), "professional_review": bool(o.get("professional_review"))}
        for o in (data.get("safety_observations") or [])
    )
    return AnalysisInput(
        concerns=tuple(concerns),
        skin_tone_bucket=bucket,
        safety_observations=observations,
        triage_level=triage,
        referral_reasons=tuple(decision.get("referral_reasons") or []),
        therapy_disposition=disposition,
        policy_reviewed=policy_reviewed,
        therapy_policy_reviewed=therapy_policy_reviewed,
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
    painful_or_deep_lesions: bool | None = None
    prior_scarring: bool | None = None
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
        painful_or_deep_lesions=_profile_optional_bool(data, "painful_or_deep_lesions"),
        prior_scarring=_profile_optional_bool(data, "prior_scarring"),
        max_price_usd=float(price) if price is not None else None,
        unknown_fields=frozenset(
            field_name for field_name in (
                "allergies", "sensitivity_conditions", "current_actives",
                "current_medications", "treatment_history",
            ) if field_name not in data or data.get(field_name) is None
        ),
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
