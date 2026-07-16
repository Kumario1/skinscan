"""Per-slot candidate generation from the catalog.

Carrier slots take the whole category. Treatment takes only products whose
verified drug actives match reviewed therapy intent; concern/INCI matching
never creates treatment intent (D-029). Serum is not a reviewed plan role.
"""
from __future__ import annotations

from .catalog import CatalogProduct
from .knowledge import Knowledge
from .signals import TargetConcern

CARRIER_SLOTS = ("cleanser", "moisturizer", "spf")
ACTIVE_SLOTS = ("treatment", "serum")
GENERIC_STRENGTH_BANDS = frozenset({
    "verified", "verified_otc_or_labeled", "per_label",
})
COMBINATION_THERAPIES = {
    "adapalene_benzoyl_peroxide": (
        "adapalene_0.1%_bp_2.5%",
        {"adapalene": "0.1%", "benzoyl_peroxide": "2.5%"},
    ),
}


def _matches_strength(actual: object, expected: str) -> bool:
    if not isinstance(actual, str) or not actual.strip():
        return False
    return (
        expected in GENERIC_STRENGTH_BANDS
        or actual.strip().lower() == expected.strip().lower()
    )


def generate_candidates(
    catalog: list[CatalogProduct],
    targets: tuple[TargetConcern, ...],
    knowledge: Knowledge,
    strict: bool = True,
    therapy_primary: dict | None = None,
) -> dict[str, list[CatalogProduct]]:
    """The legacy arguments stay for source compatibility; only
    `therapy_primary` may admit a treatment."""
    by_slot: dict[str, list[CatalogProduct]] = {s: [] for s in CARRIER_SLOTS + ACTIVE_SLOTS}
    for product in sorted(catalog, key=lambda p: p.product_id):
        if product.category in CARRIER_SLOTS:
            by_slot[product.category].append(product)
        elif product.category == "treatment" and therapy_primary is not None:
            declared = {
                active.get("name"): active.get("strength")
                for active in product.drug_actives
                if isinstance(active, dict)
            }
            expected_therapy = therapy_primary["therapy"]
            expected_combination = COMBINATION_THERAPIES.get(expected_therapy)
            if expected_combination is not None:
                expected_band, expected_actives = expected_combination
                matched = (
                    therapy_primary["strength_band"] == expected_band
                    and declared == expected_actives
                )
            else:
                matched = (
                    set(declared) == {expected_therapy}
                    and _matches_strength(
                        declared.get(expected_therapy), therapy_primary["strength_band"]
                    )
                )
            if (
                matched
                and product.exposure == therapy_primary["exposure"]
                and (
                    therapy_primary["cadence"] == "per_label"
                    or product.cadence == therapy_primary["cadence"]
                )
                and (
                    therapy_primary.get("amount") is None
                    or product.amount == therapy_primary["amount"]
                )
            ):
                by_slot["treatment"].append(product)
    return by_slot
