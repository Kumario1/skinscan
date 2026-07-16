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


# Verification-QUALITY reasons: the product's usage facts (role/area/exposure/
# cadence) or drug/label proof are not individually evidence-verified. In hybrid
# eligibility these do NOT veto — the product is still slotted by its catalog
# category and its facts derived from safe defaults; they instead lower ranking
# and are surfaced as a "category-derived, not individually verified" label.
# Everything NOT listed here is a HARD safety veto (ingredient/profile/price)
# and always excludes the product — a new reason defaults to hard.
SOFT_REASON_PREFIXES = frozenset({
    "intended_area_not_verified",
    "role_not_verified",
    "exposure_not_verified",
    "cadence_unverified",
    "cadence_not_daily",
    "treatment_active_unverified",
    "treatment_label_unverified",
    "spf_broad_spectrum_unverified",
})


def _reason_is_soft(reason: str) -> bool:
    return reason.split(":", 1)[0] in SOFT_REASON_PREFIXES


# Mirrors src.recommendation.schema.excludes_face -- an OTC drug label states a
# target ("cover the entire affected area") but almost never names the face, so
# requiring an explicit "face" vetoes every label-verified product and leaves
# the fact satisfiable only by inventing it. Veto a positive claim to another
# area instead; unknown/empty stays open.
_NON_FACE_AREAS = frozenset({"neck", "body", "eye", "lip"})


def _excludes_face(intended_areas) -> bool:
    areas = set(intended_areas)
    return "face" not in areas and bool(areas & _NON_FACE_AREAS)


def profile_gate_reasons(
    product: CatalogProduct, slot: str, profile: Profile, knowledge: Knowledge
) -> list[str]:
    reasons: list[str] = []
    actives = set(product.actives)
    expected_role = "sunscreen" if slot == "spf" else slot
    # Every ingredient gate below reads actives/inci, so a row carrying neither
    # clears all of them vacuously -- a "+Retinol" moisturizer with nothing
    # parsed passes the pregnancy exclusion because there is no active to match
    # and no INCI to scan. Unknown is data, never a favorable default. A drug row
    # publishes no INCI but names its actives, and a plain moisturizer carries a
    # real INCI and no actives; both are known. Only neither is unknown.
    if not product.inci and not actives:
        reasons.append("ingredients_unknown")
    if _excludes_face(product.intended_areas):
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
    strict: bool = True,
) -> tuple[dict[str, list[CatalogProduct]], list[Veto]]:
    """Keep the products that survive their gate reasons; return the rest as
    vetoes.

    strict=True (default): any reason vetoes — the fail-closed, evidence-only
    posture (only individually-verified products enter routines).
    strict=False (hybrid): only HARD safety reasons veto; SOFT verification
    reasons do not, so the product stays eligible (slotted by catalog category)
    but ranks below verified products and is labeled category-derived at
    explain time. Ingredient/profile/price safety is identical in both modes.
    """
    kept: dict[str, list[CatalogProduct]] = {}
    vetoes: list[Veto] = []
    for slot, products in candidates_by_slot.items():
        kept[slot] = []
        for product in products:
            reasons = profile_gate_reasons(product, slot, profile, knowledge)
            blocking = (
                reasons if strict else [r for r in reasons if not _reason_is_soft(r)]
            )
            if blocking:
                vetoes.extend(Veto(product.product_id, slot, r) for r in blocking)
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
