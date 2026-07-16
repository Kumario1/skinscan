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

from .inci import CANONICAL_ACTIVES

SCHEMA_VERSION = "recsys-1"
ANALYSIS_SCHEMA_VERSION = "3"

CONCERNS = (
    "acne_comedonal", "acne_inflammatory", "acne_cystic", "acne_scarring",
    "hyperpigmentation", "dryness",
)
TRIAGE_LEVELS = ("routine", "routine_plus_review", "derm_first", "abstain")
REFERRAL_ONLY_TRIAGE = ("derm_first", "abstain")
SKIN_TYPES = ("combination", "dry", "normal", "oily", "unknown")
TONE_BUCKETS = ("light", "medium", "deep", "unknown")
TONE_SOURCES = ("self_report", "photo", "unknown")
PREGNANCY_STATUSES = (
    "pregnant", "trying", "nursing", "not_pregnant", "not_applicable", "unknown",
)
SLOTS = ("cleanser", "treatment", "serum", "moisturizer", "spf")

# The closed vocabulary a profile may declare it is already using, derived from
# the INCI parser's synonym table so it cannot drift from what the catalog can
# actually carry. gates.py matches these against product.actives by exact set
# intersection, so an un-normalized value ("Retinol", "salicylic acid") would
# intersect nothing and silently fail the duplicate-active HARD gate open rather
# than veto. Ported from src.recommendation.schema.KNOWN_ACTIVE_IDS, whose
# UserProfile raises on the same input.
KNOWN_ACTIVE_IDS = frozenset(CANONICAL_ACTIVES.values())


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
        current_actives=_profile_list(data, "current_actives", allowed=KNOWN_ACTIVE_IDS),
        current_medications=_profile_list(data, "current_medications"),
        treatment_history=_profile_list(data, "treatment_history"),
        acne_duration_weeks=_profile_optional_int(data, "acne_duration_weeks"),
        painful_or_deep_lesions=_profile_optional_bool(data, "painful_or_deep_lesions"),
        prior_scarring=_profile_optional_bool(data, "prior_scarring"),
        max_price_usd=float(price) if price is not None else None,
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
