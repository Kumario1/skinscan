"""Hard product-role eligibility. Scoring never participates in this module."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

from .schema import Product, TherapyOption, UserProfile


RETINOIDS = {"adapalene", "retinol", "retinal", "tretinoin"}
TREATMENT_ACTIVES = RETINOIDS | {
    "benzoyl_peroxide", "azelaic_acid", "salicylic_acid", "glycolic_acid",
    "lactic_acid", "mandelic_acid",
}
ACTIVE_CONFLICTS = {
    frozenset(("benzoyl_peroxide", "retinol")),
    frozenset(("benzoyl_peroxide", "adapalene")),
    frozenset(("benzoyl_peroxide", "vitamin_c")),
    frozenset(("glycolic_acid", "retinol")),
    frozenset(("glycolic_acid", "adapalene")),
}
ALLOWED_FORMATS = {
    "cleanser": {"cleanser", "gel", "foam", "cream", "bar", "wash"},
    "treatment": {"gel", "cream", "lotion", "serum", "suspension", "solution"},
    "moisturizer": {"cream", "lotion", "gel", "balm", "emulsion"},
    "sunscreen": {"sunscreen", "cream", "lotion", "gel", "fluid", "stick"},
}
ROLE_EXPOSURES = {
    "cleanser": {"rinse_off"},
    "treatment": {"leave_on", "short_contact"},
    "moisturizer": {"leave_on"},
    "sunscreen": {"leave_on"},
}


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    field_paths: dict[str, str] = field(default_factory=dict)


def _carried(product: Product) -> set[str]:
    return set(product.actives) | {active.name for active in product.drug_actives}


def _matches_strength(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    if expected in {"verified", "verified_otc_or_labeled", "per_label"}:
        return bool(actual.strip())
    return actual.strip().lower() == expected.strip().lower()


def check_eligibility(
    product: Product,
    role: str,
    therapy: TherapyOption | None,
    profile: UserProfile,
    selected_products: Mapping[str, Product] | None = None,
) -> EligibilityResult:
    """Return every hard veto in deterministic order."""
    selected_products = selected_products or {}
    reasons: list[str] = []
    paths: dict[str, str] = {}

    def reject(code: str, path: str) -> None:
        if code not in reasons:
            reasons.append(code)
            paths[code] = path

    if product.is_legacy:
        reject("catalog_schema_legacy", "catalog_schema_version")
    if "face" not in product.intended_areas:
        reject("intended_area_not_face", "intended_areas")
    if role not in product.routine_roles:
        reject("routine_role_not_verified", "routine_roles")
    if product.format == "unknown":
        reject("format_unknown", "format")
    elif product.format not in ALLOWED_FORMATS.get(role, set()):
        reject("format_not_allowed_for_role", "format")
    if product.exposure == "unknown":
        reject("exposure_unknown", "exposure")
    elif product.exposure not in ROLE_EXPOSURES.get(role, set()):
        reject("exposure_not_allowed_for_role", "exposure")
    if product.exposure in {"mask", "scrub", "peel"}:
        reject("non_daily_format_for_role", "exposure")

    carried = _carried(product)
    if role == "treatment":
        if product.otc_drug is not True:
            reject("otc_status_not_verified", "otc_drug")
        if therapy is None:
            reject("therapy_missing_for_treatment_role", "therapy")
        else:
            matches = [active for active in product.drug_actives
                       if active.name == therapy.therapy]
            if not matches:
                reject("therapy_active_not_directly_verified", "drug_actives")
            elif not any(_matches_strength(active.strength, therapy.strength_band)
                         for active in matches):
                reject("therapy_strength_not_verified", "drug_actives.strength")
            if therapy.exposure != product.exposure:
                reject("therapy_exposure_mismatch", "exposure")
            if (therapy.cadence != "per_label"
                    and product.cadence != therapy.cadence):
                reject("therapy_cadence_mismatch", "cadence")
            if therapy.amount is not None and product.amount != therapy.amount:
                reject("therapy_amount_mismatch", "amount")
            if matches and not all(active.source for active in matches):
                reject("drug_active_source_missing", "drug_actives.source")
        if not product.label_source:
            reject("label_source_missing", "label_source")
        if not product.label_verified_at:
            reject("label_verification_timestamp_missing", "label_verified_at")
    elif carried & TREATMENT_ACTIVES:
        reject("carried_treatment_active_in_support_role", "actives")

    if role == "sunscreen":
        if product.broad_spectrum is not True:
            reject("broad_spectrum_not_verified", "broad_spectrum")
        if product.spf is None or product.spf < 30:
            reject("spf_below_30_or_unknown", "spf")
        if not product.label_source:
            reject("label_source_missing", "label_source")
        if not product.label_verified_at:
            reject("label_verification_timestamp_missing", "label_verified_at")

    if role in {"moisturizer", "sunscreen"} and (
        product.comedogenic_claim != "claimed_noncomedogenic"
    ):
        reject("noncomedogenic_claim_not_verified", "comedogenic_claim")

    if not product.cadence:
        reject("instruction_cadence_unknown", "cadence")
    if not product.cadence_source:
        reject("instruction_cadence_source_missing", "cadence_source")
    if product.amount is not None and not product.amount_source:
        reject("instruction_amount_source_missing", "amount_source")

    status = profile.pregnancy_status
    if carried & RETINOIDS and status in {"pregnant", "trying", "nursing", "unknown"}:
        reject("retinoid_pregnancy_status_excluded", "drug_actives")
    for allergy in sorted(set(profile.allergies) & carried):
        reject(f"profile_allergy:{allergy}", "actives")
    for contraindication in sorted(set(product.contraindications)):
        if contraindication in profile.sensitivity_conditions:
            reject(f"profile_contraindication:{contraindication}", "contraindications")
        if contraindication in profile.current_medications:
            reject(f"medication_contraindication:{contraindication}", "contraindications")
        if contraindication == status:
            reject(f"pregnancy_contraindication:{contraindication}", "contraindications")
    for duplicate in sorted(carried & set(profile.current_actives)):
        reject(f"duplicates_current_active:{duplicate}", "actives")

    selected_carried: set[str] = set()
    for selected in selected_products.values():
        selected_carried.update(_carried(selected))
    for duplicate in sorted(carried & selected_carried):
        reject(f"duplicates_selected_active:{duplicate}", "actives")
    for active in sorted(carried):
        for other in sorted(selected_carried):
            if frozenset((active, other)) in ACTIVE_CONFLICTS:
                reject(f"conflicts_selected_active:{active}:{other}", "actives")

    return EligibilityResult(not reasons, reasons, paths)
