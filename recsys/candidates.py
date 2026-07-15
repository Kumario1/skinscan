"""Per-slot candidate generation from the catalog.

Carrier slots (cleanser/moisturizer/spf) take the whole category; active slots
(treatment/serum) take only products whose actives intersect the union of the
target concerns' actives. With no targets (clear skin) the active slots are
empty — routines degrade to maintenance (cleanse/moisturize/protect).
"""
from __future__ import annotations

from .catalog import CatalogProduct
from .knowledge import Knowledge
from .signals import TargetConcern

CARRIER_SLOTS = ("cleanser", "moisturizer", "spf")
ACTIVE_SLOTS = ("treatment", "serum")


def generate_candidates(
    catalog: list[CatalogProduct],
    targets: tuple[TargetConcern, ...],
    knowledge: Knowledge,
    strict: bool = True,
) -> dict[str, list[CatalogProduct]]:
    """strict=True: treatment candidates must match a target concern on their
    verified drug_actives (evidence-only). strict=False (hybrid): treatments may
    also match on INCI-parsed actives, so an unverified acne treatment whose
    label proof isn't in yet still becomes a candidate (ranked below verified,
    surfaced as category-derived)."""
    target_actives: set[str] = set()
    for t in targets:
        target_actives |= knowledge.concern_actives.get(t.concern, frozenset())
    by_slot: dict[str, list[CatalogProduct]] = {s: [] for s in CARRIER_SLOTS + ACTIVE_SLOTS}
    for product in sorted(catalog, key=lambda p: p.product_id):
        if product.category in CARRIER_SLOTS:
            by_slot[product.category].append(product)
        elif product.category in ACTIVE_SLOTS:
            if product.category == "treatment":
                therapeutic_actives = {active.get("name") for active in product.drug_actives}
                if not strict:
                    therapeutic_actives |= set(product.actives)
            else:
                therapeutic_actives = set(product.actives)
            if therapeutic_actives & target_actives:
                by_slot[product.category].append(product)
    return by_slot
