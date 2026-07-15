"""Hard vetoes with deterministic reason codes. Scores never participate here
and never override a veto.

Semantics ported from src/recommendation/eligibility.py: pregnancy status
unknown/trying/nursing/pregnant excludes retinoids (unknown is data, never a
favorable default), allergies and current actives veto on any carried active,
SPF products must be >= min_spf.
"""
from __future__ import annotations

from dataclasses import dataclass

from .catalog import CatalogProduct
from .contracts import Profile
from .inci import allergy_matches, contains_retinoid
from .knowledge import Knowledge


@dataclass(frozen=True)
class Veto:
    product_id: str
    slot: str
    reason: str

    def to_dict(self) -> dict:
        return {"product_id": self.product_id, "slot": self.slot, "reason": self.reason}


def profile_gate_reasons(
    product: CatalogProduct, slot: str, profile: Profile, knowledge: Knowledge
) -> list[str]:
    reasons: list[str] = []
    actives = set(product.actives)
    expected_role = "sunscreen" if slot == "spf" else slot
    if product.discontinued:
        reasons.append("product_discontinued")
    if "face" not in product.intended_areas:
        reasons.append("intended_area_not_verified:face")
    if expected_role not in product.routine_roles:
        reasons.append(f"role_not_verified:{expected_role}")
    expected_exposure = "rinse_off" if slot == "cleanser" else "leave_on"
    if product.exposure != expected_exposure:
        reasons.append(f"exposure_not_verified:{expected_exposure}")
    if product.cadence is None or not product.cadence_source:
        reasons.append("cadence_unverified")
    elif product.cadence not in ("am", "pm", "am_pm", "daily", "once_daily",
                                 "twice_daily", "per_label"):
        reasons.append("cadence_not_daily")
    profile_contraindications = (
        set(profile.sensitivity_conditions)
        | set(profile.current_medications)
        | {profile.pregnancy_status}
    )
    for condition in sorted(set(product.contraindications) & profile_contraindications):
        reasons.append(f"product_contraindication:{condition}")
    if profile.pregnancy_status in knowledge.pregnancy_excluded_statuses and (
        actives & knowledge.retinoids or contains_retinoid(product.inci)
    ):
        reasons.append("retinoid_pregnancy_status_excluded")
    for allergy in sorted(set(profile.allergies)):
        if allergy_matches(allergy, product.inci, product.actives):
            reasons.append(f"profile_allergy:{allergy.strip().lower()}")
    for duplicate in sorted(actives & set(profile.current_actives)):
        reasons.append(f"duplicates_current_active:{duplicate}")
    if slot in ("cleanser", "moisturizer", "spf"):
        for active in sorted(actives & knowledge.treatment_actives):
            reasons.append(f"treatment_active_in_support_role:{active}")
    if slot == "treatment":
        verified_drug_actives = {
            active.get("name") for active in product.drug_actives
            if isinstance(active, dict)
        }
        if not verified_drug_actives or not (verified_drug_actives & actives):
            reasons.append("treatment_active_unverified")
        if product.format in ("mask", "peel", "scrub"):
            reasons.append(f"treatment_format_not_daily_leave_on:{product.format}")
        if not product.label_source or not product.label_verified_at:
            reasons.append("treatment_label_unverified")
    if slot == "spf":
        if product.spf is None or product.spf < knowledge.min_spf:
            reasons.append("spf_below_30_or_unknown")
        if product.broad_spectrum is False:
            reasons.append("spf_not_broad_spectrum")
        elif product.broad_spectrum is not True:
            reasons.append("spf_broad_spectrum_unverified")
    if (
        profile.max_price_usd is not None
        and product.price_usd is not None
        and product.price_usd > profile.max_price_usd
    ):
        reasons.append("price_above_profile_cap")
    return reasons


def apply_profile_gates(
    candidates_by_slot: dict[str, list[CatalogProduct]],
    profile: Profile,
    knowledge: Knowledge,
) -> tuple[dict[str, list[CatalogProduct]], list[Veto]]:
    kept: dict[str, list[CatalogProduct]] = {}
    vetoes: list[Veto] = []
    for slot, products in candidates_by_slot.items():
        kept[slot] = []
        for product in products:
            reasons = profile_gate_reasons(product, slot, profile, knowledge)
            if reasons:
                vetoes.extend(Veto(product.product_id, slot, r) for r in reasons)
            else:
                kept[slot].append(product)
    return kept, vetoes


def duplicate_active_reasons(
    candidate: CatalogProduct,
    selected: list[CatalogProduct],
    knowledge: Knowledge,
) -> list[str]:
    """Cross-product duplicate veto: only carried TREATMENT actives count —
    repeating benign support ingredients (glycerin in cleanser + moisturizer)
    is normal, not therapeutic duplication."""
    selected_actives: set[str] = set()
    for product in selected:
        selected_actives |= set(product.actives)
    duplicates = set(candidate.actives) & selected_actives & knowledge.treatment_actives
    return [f"duplicates_selected_active:{d}" for d in sorted(duplicates)]
