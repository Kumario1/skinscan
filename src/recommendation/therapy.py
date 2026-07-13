"""Reviewed-policy loading and product-independent therapy planning."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

from .schema import CareDecision, ConcernReport, TherapyOption, TherapyPlan, UserProfile


RETINOID_THERAPIES = {
    "adapalene", "retinol", "retinal", "adapalene_benzoyl_peroxide",
}
POLICY_FIELDS = {
    "policy_id", "version", "reviewed", "reviewed_by", "test_only",
    "support_roles", "paths",
}
PATH_FIELDS = {
    "therapy", "strength_band", "exposure", "cadence", "role", "reason",
    "cadence_source", "amount", "amount_source", "course_weeks",
    "review_at_weeks", "min_age_years", "max_age_years",
    "excluded_pregnancy_statuses", "excluded_sensitivity_conditions",
    "conflicting_actives", "conflicting_medications", "requires_known", "concerns",
    "excluded_treatment_history", "min_acne_duration_weeks",
    "max_acne_duration_weeks", "required_painful_or_deep_lesions",
    "required_prior_scarring",
}


@dataclass(frozen=True)
class TherapyPath:
    option: TherapyOption
    course_weeks: int | None = None
    review_at_weeks: int | None = None
    min_age_years: int | None = None
    max_age_years: int | None = None
    excluded_pregnancy_statuses: tuple[str, ...] = ()
    excluded_sensitivity_conditions: tuple[str, ...] = ()
    conflicting_actives: tuple[str, ...] = ()
    conflicting_medications: tuple[str, ...] = ()
    excluded_treatment_history: tuple[str, ...] = ()
    requires_known: tuple[str, ...] = ()
    min_acne_duration_weeks: int | None = None
    max_acne_duration_weeks: int | None = None
    required_painful_or_deep_lesions: bool | None = None
    required_prior_scarring: bool | None = None
    concerns: tuple[str, ...] = ("acne_comedonal", "acne_inflammatory")


@dataclass(frozen=True)
class TherapyPolicy:
    policy_id: str
    version: str
    reviewed: bool
    reviewed_by: str | None = None
    paths: tuple[TherapyPath, ...] = ()
    support_roles: tuple[str, ...] = ("cleanser", "moisturizer", "sunscreen")
    source_path: str | None = None
    test_only: bool = False

    @property
    def identifier(self) -> str:
        return f"{self.policy_id}:{self.version}"


def unreviewed_therapy_policy() -> TherapyPolicy:
    return TherapyPolicy(
        policy_id="missing-clinician-reviewed-policy",
        version="none",
        reviewed=False,
    )


def _optional_int(value: object, field_path: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_path}: expected a positive integer or null")
    return value


def _tuple_strings(value: object, field_path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_path}: expected a list of strings")
    return tuple(value)


def _optional_bool(value: object, field_path: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_path}: expected a boolean or null")
    return value


def _load_path(value: object, index: int) -> TherapyPath:
    field_path = f"therapy_policy.paths[{index}]"
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_path}: expected an object")
    unknown = set(value) - PATH_FIELDS
    if unknown:
        raise ValueError(f"{field_path}: unknown fields {sorted(unknown)}")
    required = ("therapy", "strength_band", "exposure", "cadence", "role")
    for key in required:
        if not isinstance(value.get(key), str) or not value.get(key):
            raise ValueError(f"{field_path}.{key}: expected a non-empty string")
    cadence_source = value.get("cadence_source")
    amount = value.get("amount")
    amount_source = value.get("amount_source")
    if not isinstance(cadence_source, str) or not cadence_source:
        raise ValueError(f"{field_path}.cadence_source: expected a non-empty string")
    if amount is not None and not isinstance(amount, str):
        raise ValueError(f"{field_path}.amount: expected string or null")
    if amount_source is not None and not isinstance(amount_source, str):
        raise ValueError(f"{field_path}.amount_source: expected string or null")
    if amount is not None and not amount_source:
        raise ValueError(f"{field_path}.amount_source: required when amount is present")
    reason = value.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(f"{field_path}.reason: expected string or null")
    requires_known = _tuple_strings(
        value.get("requires_known"), f"{field_path}.requires_known"
    )
    allowed_profile_fields = set(UserProfile.__dataclass_fields__) - {"pregnant_or_nursing"}
    invalid_required = set(requires_known) - allowed_profile_fields
    if invalid_required:
        raise ValueError(
            f"{field_path}.requires_known: unknown profile fields {sorted(invalid_required)}"
        )
    option = TherapyOption(
        therapy=value["therapy"], strength_band=value["strength_band"],
        exposure=value["exposure"], cadence=value["cadence"], role=value["role"],
        reason=reason, cadence_source=cadence_source,
        amount=amount, amount_source=amount_source,
    )
    return TherapyPath(
        option=option,
        course_weeks=_optional_int(value.get("course_weeks"), f"{field_path}.course_weeks"),
        review_at_weeks=_optional_int(
            value.get("review_at_weeks"), f"{field_path}.review_at_weeks"
        ),
        min_age_years=_optional_int(value.get("min_age_years"), f"{field_path}.min_age_years"),
        max_age_years=_optional_int(value.get("max_age_years"), f"{field_path}.max_age_years"),
        excluded_pregnancy_statuses=_tuple_strings(
            value.get("excluded_pregnancy_statuses"),
            f"{field_path}.excluded_pregnancy_statuses",
        ),
        excluded_sensitivity_conditions=_tuple_strings(
            value.get("excluded_sensitivity_conditions"),
            f"{field_path}.excluded_sensitivity_conditions",
        ),
        conflicting_actives=_tuple_strings(
            value.get("conflicting_actives"), f"{field_path}.conflicting_actives"
        ),
        conflicting_medications=_tuple_strings(
            value.get("conflicting_medications"), f"{field_path}.conflicting_medications"
        ),
        excluded_treatment_history=_tuple_strings(
            value.get("excluded_treatment_history"),
            f"{field_path}.excluded_treatment_history",
        ),
        requires_known=requires_known,
        min_acne_duration_weeks=_optional_int(
            value.get("min_acne_duration_weeks"), f"{field_path}.min_acne_duration_weeks"
        ),
        max_acne_duration_weeks=_optional_int(
            value.get("max_acne_duration_weeks"), f"{field_path}.max_acne_duration_weeks"
        ),
        required_painful_or_deep_lesions=_optional_bool(
            value.get("required_painful_or_deep_lesions"),
            f"{field_path}.required_painful_or_deep_lesions",
        ),
        required_prior_scarring=_optional_bool(
            value.get("required_prior_scarring"), f"{field_path}.required_prior_scarring"
        ),
        concerns=_tuple_strings(value.get("concerns"), f"{field_path}.concerns")
        or ("acne_comedonal", "acne_inflammatory"),
    )


def load_therapy_policy(path: Path | None) -> TherapyPolicy:
    if path is None:
        return unreviewed_therapy_policy()
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return TherapyPolicy(
            "missing-clinician-reviewed-policy", "none", False, source_path=str(path)
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"therapy policy {path}: invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("therapy_policy: expected an object")
    unknown = set(value) - POLICY_FIELDS
    if unknown:
        raise ValueError(f"therapy_policy: unknown fields {sorted(unknown)}")
    for key in ("policy_id", "version"):
        if not isinstance(value.get(key), str) or not value.get(key):
            raise ValueError(f"therapy_policy.{key}: expected a non-empty string")
    reviewed = value.get("reviewed")
    if not isinstance(reviewed, bool):
        raise ValueError("therapy_policy.reviewed: expected a boolean")
    reviewed_by = value.get("reviewed_by")
    test_only = value.get("test_only", False)
    if reviewed_by is not None and not isinstance(reviewed_by, str):
        raise ValueError("therapy_policy.reviewed_by: expected string or null")
    if not isinstance(test_only, bool):
        raise ValueError("therapy_policy.test_only: expected a boolean")
    if reviewed and not reviewed_by and not test_only:
        raise ValueError("therapy_policy.reviewed_by: required for reviewed production policy")
    raw_paths = value.get("paths", [])
    if not isinstance(raw_paths, list):
        raise ValueError("therapy_policy.paths: expected a list")
    support_roles = _tuple_strings(value.get("support_roles"), "therapy_policy.support_roles")
    return TherapyPolicy(
        policy_id=value["policy_id"], version=value["version"], reviewed=reviewed,
        reviewed_by=reviewed_by,
        paths=tuple(_load_path(item, index) for index, item in enumerate(raw_paths)),
        support_roles=support_roles or ("cleanser", "moisturizer", "sunscreen"),
        source_path=str(path), test_only=test_only,
    )


def _unknown_required(profile: UserProfile, field_name: str) -> bool:
    value = getattr(profile, field_name, None)
    return value is None or value == "unknown"


def _path_reasons(path: TherapyPath, profile: UserProfile) -> list[str]:
    therapy = path.option.therapy
    reasons: list[str] = []
    for field_name in path.requires_known:
        if _unknown_required(profile, field_name):
            reasons.append(f"required_profile_unknown:{field_name}")
    if path.min_age_years is not None:
        if profile.age_years is None:
            reasons.append("required_profile_unknown:age_years")
        elif profile.age_years < path.min_age_years:
            reasons.append("age_below_policy_minimum")
    if path.max_age_years is not None and profile.age_years is not None:
        if profile.age_years > path.max_age_years:
            reasons.append("age_above_policy_maximum")
    if therapy in RETINOID_THERAPIES and profile.pregnancy_status in {
        "pregnant", "trying", "nursing",
    }:
        reasons.append(f"pregnancy_status_excludes:{therapy}")
    elif profile.pregnancy_status in path.excluded_pregnancy_statuses:
        reasons.append(f"pregnancy_status_excludes:{therapy}")
    if therapy in RETINOID_THERAPIES and profile.pregnancy_status == "unknown":
        reasons.append(f"pregnancy_status_unknown_defers:{therapy}")
    for condition in sorted(set(profile.sensitivity_conditions) & set(
        path.excluded_sensitivity_conditions
    )):
        reasons.append(f"sensitivity_condition_excludes:{condition}")
    for active in sorted(set(profile.current_actives) & (
        set(path.conflicting_actives) | {therapy}
    )):
        reasons.append(f"current_active_conflict:{active}")
    for medication in sorted(set(profile.current_medications) & set(
        path.conflicting_medications
    )):
        reasons.append(f"current_medication_conflict:{medication}")
    for history in sorted(set(profile.treatment_history) & set(
        path.excluded_treatment_history
    )):
        reasons.append(f"treatment_history_excludes:{history}")
    duration = profile.acne_duration_weeks
    if path.min_acne_duration_weeks is not None:
        if duration is None:
            reasons.append("required_profile_unknown:acne_duration_weeks")
        elif duration < path.min_acne_duration_weeks:
            reasons.append("acne_duration_below_policy_minimum")
    if (path.max_acne_duration_weeks is not None and duration is not None
            and duration > path.max_acne_duration_weeks):
        reasons.append("acne_duration_above_policy_maximum")
    for field_name, required in (
        ("painful_or_deep_lesions", path.required_painful_or_deep_lesions),
        ("prior_scarring", path.required_prior_scarring),
    ):
        if required is not None:
            actual = getattr(profile, field_name)
            if actual is None:
                reasons.append(f"required_profile_unknown:{field_name}")
            elif actual is not required:
                reasons.append(f"profile_value_mismatch:{field_name}")
    return reasons


def plan_therapy(
    decision: CareDecision,
    report: ConcernReport,
    profile: UserProfile,
    policy: TherapyPolicy,
) -> TherapyPlan:
    support_roles = list(policy.support_roles)
    if decision.therapy_disposition in {"supportive_only", "maintenance"}:
        guidance = (
            ["avoid_self_start_or_stop_medicine_pending_professional_review"]
            if decision.therapy_disposition == "supportive_only" else []
        )
        return TherapyPlan(
            None, None, None, [], support_roles, guidance, policy.identifier,
        )
    if not policy.reviewed:
        return TherapyPlan(
            None, None, None, [], support_roles,
            ["clinician_reviewed_policy_missing"], policy.identifier,
        )

    reported = {concern.concern for concern in report.concerns}
    eligible: list[TherapyPath] = []
    deferred: list[str] = []
    for path in policy.paths:
        if not reported.intersection(path.concerns):
            continue
        reasons = _path_reasons(path, profile)
        if reasons:
            deferred.extend(reasons)
        else:
            eligible.append(path)
    if not eligible:
        if not deferred:
            deferred.append("no_policy_path_for_reported_concerns")
        return TherapyPlan(None, None, None, [], support_roles,
                           list(dict.fromkeys(deferred)), policy.identifier)

    primary_path, *alternative_paths = eligible
    alternatives = [
        TherapyOption(
            path.option.therapy, path.option.strength_band, path.option.exposure,
            path.option.cadence, path.option.role,
            reason=path.option.reason or "eligible_policy_alternative",
            cadence_source=path.option.cadence_source,
            amount=path.option.amount, amount_source=path.option.amount_source,
        )
        for path in alternative_paths
    ]
    return TherapyPlan(
        primary_path.course_weeks,
        primary_path.review_at_weeks,
        primary_path.option,
        alternatives,
        support_roles,
        list(dict.fromkeys(deferred)),
        policy.identifier,
    )
