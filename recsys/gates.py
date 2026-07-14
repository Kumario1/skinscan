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
    if actives & knowledge.retinoids and (
        profile.pregnancy_status in knowledge.pregnancy_excluded_statuses
    ):
        reasons.append("retinoid_pregnancy_status_excluded")
    for allergy in sorted(actives & set(profile.allergies)):
        reasons.append(f"profile_allergy:{allergy}")
    for duplicate in sorted(actives & set(profile.current_actives)):
        reasons.append(f"duplicates_current_active:{duplicate}")
    if slot == "spf" and (product.spf is None or product.spf < knowledge.min_spf):
        reasons.append("spf_below_30_or_unknown")
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
