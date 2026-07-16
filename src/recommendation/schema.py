"""Versioned recommendation data contracts.

The concern contract remains compatible with the historical SA-RPN bridge.
Catalog/recommendation v3 fields are explicit and unknown-capable: loading a
legacy row preserves it for display but never invents eligibility metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections import Counter
from typing import Mapping, Optional


CONCERNS = {
    "acne_comedonal", "acne_inflammatory", "acne_cystic", "acne_scarring",
    "hyperpigmentation", "dryness",
}
REGIONS = {
    "forehead", "nose", "left_cheek", "right_cheek", "chin_jaw", "perioral",
}
CATEGORIES = ["cleanser", "treatment", "serum", "moisturizer", "spf"]
ROUTINE_ROLES = {"cleanser", "treatment", "moisturizer", "sunscreen"}
PRODUCT_EXPOSURES = {
    "unknown", "rinse_off", "short_contact", "leave_on", "mask", "scrub", "peel",
}
PRODUCT_AREAS = {"face", "neck", "body", "eye", "lip", "unknown"}
# Areas that positively place a product somewhere other than the face. "unknown"
# and an empty list are absence of evidence, not evidence of another area.
NON_FACE_AREAS = PRODUCT_AREAS - {"face", "unknown"}
TRIAGE_LEVELS = {"routine", "routine_plus_review", "derm_first", "abstain"}
THERAPY_DISPOSITIONS = {"active_treatment", "supportive_only", "maintenance", "defer"}
EVIDENCE_QUALITIES = {"high", "medium", "low", "unknown"}
SLOTS = {"am", "pm"}


def excludes_face(intended_areas) -> bool:
    """True only when a product names its areas and the face is not among them.

    An OTC drug label states a target ("cover the entire affected area") but
    almost never names the face, so requiring an explicit "face" vetoes every
    label-verified product and leaves the fact satisfiable only by inventing it.
    Veto on a positive claim to another area instead; unknown/empty stays open.
    """
    areas = set(intended_areas)
    return "face" not in areas and bool(areas & NON_FACE_AREAS)


def _closed(field_name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name}: expected one of {sorted(allowed)}, got {value!r}")


def _string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name}: expected a list of strings")
    return list(value)


# --- concern side (Stage 2 -> decision) -----------------------------------
@dataclass(frozen=True)
class ConcernEvidence:
    labels: dict[str, int] = field(default_factory=dict)
    max_confidence: float = 0.0
    affected_region_count: int = 0
    source: str = "prediction"


@dataclass
class Concern:
    concern: str
    region: str
    severity: int
    confidence: float
    lesion_count: Optional[int] = None
    regions: list[str] = field(default_factory=list)
    evidence: ConcernEvidence = field(default_factory=ConcernEvidence)

    def __post_init__(self) -> None:
        _closed("concern", self.concern, CONCERNS)
        _closed("region", self.region, REGIONS)
        if not self.regions:
            self.regions = [self.region]
        self.regions = list(dict.fromkeys(self.regions))
        for region in self.regions:
            _closed("regions", region, REGIONS)
        if self.region not in self.regions:
            raise ValueError("region: canonical region must be present in regions")
        if not 0 <= self.severity <= 4:
            raise ValueError("severity: expected 0..4")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence: expected 0..1")
        if not 0.0 <= self.evidence.max_confidence <= 1.0:
            raise ValueError("evidence.max_confidence: expected 0..1")
        if (self.evidence.labels or self.evidence.max_confidence
                or self.evidence.affected_region_count):
            if self.evidence.affected_region_count != len(self.regions):
                raise ValueError("evidence.affected_region_count must match regions")


@dataclass
class ConcernReport:
    image_id: str
    concerns: list[Concern] = field(default_factory=list)
    clear_skin: bool = False
    low_light_flag: bool = False
    notes: str = ""

    @property
    def overall_severity(self) -> int:
        acne = [c.severity for c in self.concerns if c.concern.startswith("acne_")]
        return max(acne) if acne else 0

    @property
    def has_cystic(self) -> bool:
        return any(c.concern == "acne_cystic" for c in self.concerns)


# --- catalog v2 ------------------------------------------------------------
@dataclass(frozen=True)
class VerifiedActive:
    name: str
    strength: str | None = None
    source: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object], field_path: str = "drug_actives") -> "VerifiedActive":
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_path}: expected an object")
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field_path}.name: expected a non-empty string")
        for key in ("strength", "source"):
            item = value.get(key)
            if item is not None and not isinstance(item, str):
                raise ValueError(f"{field_path}.{key}: expected string or null")
        return cls(name=name, strength=value.get("strength"), source=value.get("source"))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "strength": self.strength, "source": self.source}


@dataclass
class Product:
    # Historical fields stay first for positional source compatibility.
    product_id: str
    name: str
    brand: str
    category: str
    actives: list[str] = field(default_factory=list)
    comedogenic_flags: list[str] = field(default_factory=list)
    price_usd: Optional[float] = None
    price_is_stale: bool = True
    ingredient_match: dict[str, float] = field(default_factory=dict)
    tier: int = 1
    no_outcome_data: bool = False

    # Catalog contract v2. Unknown/empty values are storage-valid and are hard
    # vetoes when the corresponding role needs them.
    intended_areas: list[str] = field(default_factory=list)
    routine_roles: list[str] = field(default_factory=list)
    format: str = "unknown"
    exposure: str = "unknown"
    drug_actives: list[VerifiedActive] = field(default_factory=list)
    otc_drug: bool | None = None
    label_source: str | None = None
    label_verified_at: str | None = None
    broad_spectrum: bool | None = None
    spf: int | None = None
    comedogenic_claim: str = "unknown"
    irritant_features: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    evidence_roles: list[str] = field(default_factory=list)
    evidence_grade: str = "unknown"
    cadence: str | None = None
    cadence_source: str | None = None
    amount: str | None = None
    amount_source: str | None = None
    source_set_id: str | None = None
    ndc_product_code: str | None = None
    label_version: str | None = None
    label_effective_date: str | None = None
    source_hash: str | None = None
    catalog_schema_version: str = "legacy"

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"category: unknown category {self.category!r}")
        if not isinstance(self.format, str) or not self.format:
            raise ValueError("format: expected a non-empty string")
        _closed("exposure", self.exposure, PRODUCT_EXPOSURES)
        _closed(
            "comedogenic_claim",
            self.comedogenic_claim,
            {"unknown", "claimed_noncomedogenic", "not_claimed"},
        )
        invalid_roles = set(self.routine_roles) - ROUTINE_ROLES
        if invalid_roles:
            raise ValueError(f"routine_roles: unknown roles {sorted(invalid_roles)}")
        invalid_areas = set(self.intended_areas) - PRODUCT_AREAS
        if invalid_areas:
            raise ValueError(f"intended_areas: unknown areas {sorted(invalid_areas)}")
        for field_name in ("otc_drug", "broad_spectrum"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name}: expected boolean or null")
        if (self.spf is not None
                and (not isinstance(self.spf, int) or isinstance(self.spf, bool)
                     or self.spf < 0)):
            raise ValueError("spf: expected a non-negative integer or null")
        for field_name in (
            "label_source", "label_verified_at", "cadence", "cadence_source",
            "amount", "amount_source",
            "source_set_id", "ndc_product_code", "label_version",
            "label_effective_date", "source_hash",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name}: expected string or null")
        if not isinstance(self.evidence_grade, str):
            raise ValueError("evidence_grade: expected a string")
        converted: list[VerifiedActive] = []
        for index, active in enumerate(self.drug_actives):
            if isinstance(active, VerifiedActive):
                converted.append(active)
            elif isinstance(active, Mapping):
                converted.append(VerifiedActive.from_dict(active, f"drug_actives[{index}]"))
            else:
                raise ValueError(f"drug_actives[{index}]: expected object")
        self.drug_actives = converted

    @property
    def is_legacy(self) -> bool:
        return str(self.catalog_schema_version).lower() in {"1", "legacy"}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Product":
        if not isinstance(value, Mapping):
            raise ValueError("product: expected an object")
        required = ("product_id", "name", "brand", "category")
        for key in required:
            if not isinstance(value.get(key), str):
                raise ValueError(f"product.{key}: expected a string")
        legacy = not any(
            key in value
            for key in ("catalog_schema_version", "intended_areas", "routine_roles", "exposure")
        )
        kwargs = dict(value)
        kwargs["actives"] = _string_list(value.get("actives"), "product.actives")
        kwargs["comedogenic_flags"] = _string_list(
            value.get("comedogenic_flags"), "product.comedogenic_flags"
        )
        for key in (
            "intended_areas", "routine_roles", "irritant_features", "contraindications",
            "evidence_roles",
        ):
            kwargs[key] = _string_list(value.get(key), f"product.{key}")
        raw_drug_actives = value.get("drug_actives", [])
        if not isinstance(raw_drug_actives, list):
            raise ValueError("product.drug_actives: expected a list")
        kwargs["drug_actives"] = [
            VerifiedActive.from_dict(item, f"product.drug_actives[{index}]")
            for index, item in enumerate(raw_drug_actives)
        ]
        kwargs.setdefault("catalog_schema_version", "legacy" if legacy else "2")
        known = set(cls.__dataclass_fields__)
        unknown = set(kwargs) - known
        if unknown:
            raise ValueError(f"product: unknown fields {sorted(unknown)}")
        try:
            return cls(**kwargs)
        except TypeError as exc:
            raise ValueError(f"product: invalid fields: {exc}") from exc

    def to_dict(self) -> dict[str, object]:
        # Explicit order is intentional; canonical JSON may sort it later, but
        # callers that preserve insertion order still receive stable bytes.
        return {
            "catalog_schema_version": self.catalog_schema_version,
            "product_id": self.product_id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "actives": list(self.actives),
            "comedogenic_flags": list(self.comedogenic_flags),
            "price_usd": self.price_usd,
            "price_is_stale": self.price_is_stale,
            "ingredient_match": dict(sorted(self.ingredient_match.items())),
            "tier": self.tier,
            "no_outcome_data": self.no_outcome_data,
            "intended_areas": list(self.intended_areas),
            "routine_roles": list(self.routine_roles),
            "format": self.format,
            "exposure": self.exposure,
            "drug_actives": [active.to_dict() for active in self.drug_actives],
            "otc_drug": self.otc_drug,
            "label_source": self.label_source,
            "label_verified_at": self.label_verified_at,
            "broad_spectrum": self.broad_spectrum,
            "spf": self.spf,
            "comedogenic_claim": self.comedogenic_claim,
            "irritant_features": list(self.irritant_features),
            "contraindications": list(self.contraindications),
            "evidence_roles": list(self.evidence_roles),
            "evidence_grade": self.evidence_grade,
            "cadence": self.cadence,
            "cadence_source": self.cadence_source,
            "amount": self.amount,
            "amount_source": self.amount_source,
            "source_set_id": self.source_set_id,
            "ndc_product_code": self.ndc_product_code,
            "label_version": self.label_version,
            "label_effective_date": self.label_effective_date,
            "source_hash": self.source_hash,
        }


# --- explicit safety profile ----------------------------------------------
SKIN_TYPES = {"combination", "dry", "normal", "oily", "unknown"}
TONE_BUCKETS = {"light", "medium", "deep", "unknown"}
TONE_SOURCES = {"self_report", "photo", "unknown"}
PREGNANCY_STATUSES = {
    "pregnant", "trying", "nursing", "not_pregnant", "not_applicable", "unknown",
}
KNOWN_ACTIVE_IDS = {
    "salicylic_acid", "benzoyl_peroxide", "adapalene", "azelaic_acid",
    "glycolic_acid", "lactic_acid", "mandelic_acid", "niacinamide",
    "vitamin_c", "alpha_arbutin", "tranexamic_acid", "kojic_acid", "retinol",
    "retinal", "ceramides", "hyaluronic_acid", "glycerin", "squalane",
    "panthenol", "centella", "allantoin", "madecassoside", "zinc",
    "gluconolactone", "willow_bark",
}


@dataclass
class UserProfile:
    skin_type: str = "unknown"
    tone_bucket: str = "unknown"
    tone_source: str = "unknown"
    # Legacy boolean remains accepted only as an explicit migration input.
    pregnant_or_nursing: bool | None = None
    age_years: int | None = None
    pregnancy_status: str = "unknown"
    allergies: list[str] = field(default_factory=list)
    sensitivity_conditions: list[str] = field(default_factory=list)
    current_actives: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    treatment_history: list[str] = field(default_factory=list)
    acne_duration_weeks: int | None = None
    painful_or_deep_lesions: bool | None = None
    prior_scarring: bool | None = None
    max_price_usd: float | None = None

    def __post_init__(self) -> None:
        _closed("skin_type", self.skin_type, SKIN_TYPES)
        if self.tone_bucket is None:  # narrow old-payload migration
            self.tone_bucket = "unknown"
        _closed("tone_bucket", self.tone_bucket, TONE_BUCKETS)
        _closed("tone_source", self.tone_source, TONE_SOURCES)
        _closed("pregnancy_status", self.pregnancy_status, PREGNANCY_STATUSES)
        if (self.pregnant_or_nursing is not None
                and not isinstance(self.pregnant_or_nursing, bool)):
            raise ValueError("pregnant_or_nursing: expected boolean or null")
        if self.pregnant_or_nursing is not None:
            migrated = "pregnant" if self.pregnant_or_nursing else "not_pregnant"
            if self.pregnancy_status != "unknown" and self.pregnancy_status != migrated:
                raise ValueError("pregnancy_status conflicts with pregnant_or_nursing")
            self.pregnancy_status = migrated
        if (self.age_years is not None
                and (not isinstance(self.age_years, int) or isinstance(self.age_years, bool)
                     or not 0 <= self.age_years <= 130)):
            raise ValueError("age_years: expected an integer 0..130 or null")
        if (self.acne_duration_weeks is not None
                and (not isinstance(self.acne_duration_weeks, int)
                     or isinstance(self.acne_duration_weeks, bool)
                     or self.acne_duration_weeks < 0)):
            raise ValueError("acne_duration_weeks: expected a non-negative integer or null")
        if (self.max_price_usd is not None
                and (not isinstance(self.max_price_usd, (int, float))
                     or isinstance(self.max_price_usd, bool)
                     or not math.isfinite(self.max_price_usd)
                     or self.max_price_usd < 0)):
            raise ValueError("max_price_usd: expected a finite non-negative number or null")
        for field_name in ("painful_or_deep_lesions", "prior_scarring"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name}: expected boolean or null")
        for key in (
            "allergies", "sensitivity_conditions", "current_actives",
            "current_medications", "treatment_history",
        ):
            value = getattr(self, key)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{key}: expected a list of strings")
        invalid_actives = set(self.current_actives) - KNOWN_ACTIVE_IDS
        if invalid_actives:
            raise ValueError(f"current_actives: unknown active IDs {sorted(invalid_actives)}")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "UserProfile":
        if not isinstance(value, Mapping):
            raise ValueError("profile: expected an object")
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"profile: unknown fields {sorted(unknown)}")
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ValueError(f"profile: invalid fields: {exc}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "skin_type": self.skin_type,
            "tone_bucket": self.tone_bucket,
            "tone_source": self.tone_source,
            "age_years": self.age_years,
            "pregnancy_status": self.pregnancy_status,
            "allergies": list(self.allergies),
            "sensitivity_conditions": list(self.sensitivity_conditions),
            "current_actives": list(self.current_actives),
            "current_medications": list(self.current_medications),
            "treatment_history": list(self.treatment_history),
            "acne_duration_weeks": self.acne_duration_weeks,
            "painful_or_deep_lesions": self.painful_or_deep_lesions,
            "prior_scarring": self.prior_scarring,
            "max_price_usd": self.max_price_usd,
        }


# --- care, therapeutic intent, and regimen --------------------------------
@dataclass(frozen=True)
class DecisionEvidence:
    concern: str
    probability: float | None
    quality: str
    source: str
    calibrated: bool
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _closed("quality", self.quality, EVIDENCE_QUALITIES)
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability: expected 0..1 or null")
        if self.probability is not None and not self.calibrated:
            raise ValueError("probability may be populated only for calibrated evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "concern": self.concern,
            "probability": self.probability,
            "quality": self.quality,
            "source": self.source,
            "calibrated": self.calibrated,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CareDecision:
    triage_level: str
    referral_reasons: list[str]
    therapy_disposition: str
    evidence: list[DecisionEvidence]
    policy_version: str | None
    policy_reviewed: bool

    def __post_init__(self) -> None:
        _closed("triage_level", self.triage_level, TRIAGE_LEVELS)
        _closed("therapy_disposition", self.therapy_disposition, THERAPY_DISPOSITIONS)

    def to_dict(self) -> dict[str, object]:
        return {
            "triage_level": self.triage_level,
            "referral_reasons": list(self.referral_reasons),
            "therapy_disposition": self.therapy_disposition,
            "decision_evidence": [item.to_dict() for item in self.evidence],
            "policy_version": self.policy_version,
            "policy_reviewed": self.policy_reviewed,
        }


@dataclass(frozen=True)
class TherapyOption:
    therapy: str
    strength_band: str
    exposure: str
    cadence: str
    role: str
    reason: str | None = None
    cadence_source: str | None = None
    amount: str | None = None
    amount_source: str | None = None

    def __post_init__(self) -> None:
        _closed("exposure", self.exposure, PRODUCT_EXPOSURES)
        _closed("role", self.role, ROUTINE_ROLES)
        if not self.therapy or not self.strength_band or not self.cadence:
            raise ValueError("therapy, strength_band, and cadence must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "therapy": self.therapy,
            "strength_band": self.strength_band,
            "exposure": self.exposure,
            "cadence": self.cadence,
            "role": self.role,
            "reason": self.reason,
            "cadence_source": self.cadence_source,
            "amount": self.amount,
            "amount_source": self.amount_source,
        }


@dataclass(frozen=True)
class TherapyPlan:
    course_weeks: int | None
    review_at_weeks: int | None
    primary: TherapyOption | None
    alternatives: list[TherapyOption]
    support_roles: list[str]
    deferred_reasons: list[str]
    policy_version: str | None

    def __post_init__(self) -> None:
        invalid = set(self.support_roles) - ROUTINE_ROLES
        if invalid:
            raise ValueError(f"support_roles: unknown roles {sorted(invalid)}")
        for field_name, value in (
            ("course_weeks", self.course_weeks), ("review_at_weeks", self.review_at_weeks)
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{field_name}: expected a positive integer or null")

    def to_dict(self) -> dict[str, object]:
        return {
            "course_weeks": self.course_weeks,
            "review_at_weeks": self.review_at_weeks,
            "primary": self.primary.to_dict() if self.primary else None,
            "alternatives": [item.to_dict() for item in self.alternatives],
            "support_roles": list(self.support_roles),
            "deferred_reasons": list(self.deferred_reasons),
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class RoutineInstruction:
    role: str
    slot: str
    cadence: str
    amount: str | None
    source: str | None

    def __post_init__(self) -> None:
        _closed("role", self.role, ROUTINE_ROLES)
        _closed("slot", self.slot, SLOTS)
        if not self.cadence:
            raise ValueError("cadence: expected a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "slot": self.slot,
            "cadence": self.cadence,
            "amount": self.amount,
            "source": self.source,
        }


@dataclass
class EligibilityDiagnostics:
    """Compact role-level eligibility evidence with optional debug detail."""

    requested_roles: list[str]
    collect_details: bool = False
    _eligible_counts: Counter[str] = field(default_factory=Counter, repr=False)
    _rejected_counts: Counter[str] = field(default_factory=Counter, repr=False)
    _reason_counts: dict[str, Counter[str]] = field(default_factory=dict, repr=False)
    _details: dict[str, dict[str, list[str]]] = field(default_factory=dict, repr=False)
    missing_roles: list[str] = field(default_factory=list)

    def record(self, role: str, product_id: str, reasons: list[str]) -> None:
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            self._rejected_counts[role] += 1
            self._reason_counts.setdefault(role, Counter()).update(reasons)
        else:
            self._eligible_counts[role] += 1
        if self.collect_details:
            self._details.setdefault(role, {})[product_id] = reasons

    def reject_previously_eligible(
        self, role: str, product_id: str, reasons: list[str]
    ) -> None:
        """Replace a first-pass eligible result with a contextual rejection."""
        if self._eligible_counts[role]:
            self._eligible_counts[role] -= 1
        reasons = list(dict.fromkeys(reasons))
        self._rejected_counts[role] += 1
        self._reason_counts.setdefault(role, Counter()).update(reasons)
        if self.collect_details:
            self._details.setdefault(role, {})[product_id] = reasons

    def mark_missing(self, role: str) -> None:
        if role not in self.missing_roles:
            self.missing_roles.append(role)

    def role_has_missing_reason(self, role: str) -> bool:
        return role in self.missing_roles

    def to_summary(self, selected_roles: list[str] | None = None) -> dict[str, object]:
        selected = selected_roles or []
        roles: dict[str, object] = {}
        for role in self.requested_roles:
            roles[role] = {
                "eligible_count": self._eligible_counts[role],
                "rejected_count": self._rejected_counts[role],
                "rejection_reason_counts": dict(sorted(
                    self._reason_counts.get(role, Counter()).items()
                )),
            }
        return {
            "requested_roles": list(self.requested_roles),
            "selected_roles": [role for role in self.requested_roles if role in selected],
            "missing_roles": [role for role in self.requested_roles if role in self.missing_roles],
            "roles": roles,
        }

    def debug_payload(self) -> dict[str, object] | None:
        if not self.collect_details:
            return None
        return {
            "schema_version": "1",
            "rejections": {
                role: {
                    product_id: reasons
                    for product_id, reasons in sorted(outcomes.items()) if reasons
                }
                for role, outcomes in sorted(self._details.items())
            },
        }


@dataclass
class Recommendation:
    decision: CareDecision
    therapy_plan: TherapyPlan
    selected_products: dict[str, Product]
    selected_regimen: dict[str, list[RoutineInstruction]]
    alternatives: dict[str, list[Product]]
    eligibility_diagnostics: EligibilityDiagnostics
    explanation: list[dict[str, object]]
    flags: list[str]
    validation_errors: list[str]

    @property
    def valid(self) -> bool:
        return not self.validation_errors

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.to_dict(),
            "therapy_plan": self.therapy_plan.to_dict(),
            "selected_products": {
                role: product.to_dict() for role, product in sorted(self.selected_products.items())
            },
            "selected_regimen": {
                slot: [item.to_dict() for item in self.selected_regimen.get(slot, [])]
                for slot in ("am", "pm")
            },
            "alternatives": {
                role: [product.to_dict() for product in products]
                for role, products in sorted(self.alternatives.items())
            },
            "explanation": [dict(item) for item in self.explanation],
            "flags": list(self.flags),
        }

    @property
    def eligibility_rejections(self) -> dict[str, list[str]]:
        """Compatibility view: role-level reasons, plus detail only in debug mode."""
        result = {
            f"role:{role}": ["no_eligible_product"]
            for role in self.eligibility_diagnostics.missing_roles
        }
        debug = self.eligibility_diagnostics.debug_payload()
        if debug:
            for role, rows in debug["rejections"].items():
                for product_id, reasons in rows.items():
                    result[f"{role}:{product_id}"] = list(reasons)
        return result
