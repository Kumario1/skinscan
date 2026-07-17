"""Per-slot candidate generation from the catalog.

Carrier slots take the whole category. Treatment takes only products whose
verified drug actives match reviewed therapy intent; concern/INCI matching
never creates treatment intent (D-029). Serum is not a reviewed plan role:
it admits cosmetic products whose actives target a selected concern.
"""
from __future__ import annotations

from .catalog import CatalogProduct
from .knowledge import Knowledge
from .signals import TargetLesion

CARRIER_SLOTS = ("cleanser", "moisturizer", "spf")
ACTIVE_SLOTS = ("treatment", "serum", "scar_care")
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


def _matches_policy_strength(actual: object, policy_strength: object) -> bool:
    if not isinstance(actual, str) or not isinstance(policy_strength, str):
        return False
    try:
        value = float(actual.strip().rstrip("%"))
    except ValueError:
        return actual.strip().lower() == policy_strength.strip().lower()
    text = policy_strength.replace(" ", "").rstrip("%")
    if "%-" in policy_strength:
        text = policy_strength.replace(" ", "").replace("%", "")
    if "-" in text:
        try:
            low, high = (float(part) for part in text.split("-", 1))
        except ValueError:
            return False
        return low <= value <= high
    try:
        return value == float(text)
    except ValueError:
        return False


def generate_candidates(
    catalog: list[CatalogProduct],
    targets: tuple[TargetLesion, ...],
    knowledge: Knowledge,
    strict: bool = True,
    therapy_primary: dict | None = None,
) -> dict[str, list[CatalogProduct]]:
    """The legacy arguments stay for source compatibility; only
    `therapy_primary` may admit a treatment."""
    by_slot: dict[str, list[CatalogProduct]] = {s: [] for s in CARRIER_SLOTS + ACTIVE_SLOTS}
    treatment_specs = {
        str(spec.get("active_id")): spec
        for target in targets if "treatment" in target.required_roles
        for spec in getattr(target, "target_specs", ())
        if isinstance(spec, dict) and spec.get("active_id")
    }
    # TargetLesion keeps only normalized active IDs in its public constructor;
    # when no full strength spec is attached, a label-verified drug fact still
    # proves exact active identity and the safety gates remain fail closed.
    treatment_actives = {
        active for target in targets if "treatment" in target.required_roles
        for active in target.target_actives
    }
    melasma_spf_required = any(
        target.lesion_type == "melasma" and "sunscreen" in target.required_roles
        for target in targets
    )
    scar_care_required = any(
        "scar_care" in target.required_roles for target in targets
    )
    for product in sorted(catalog, key=lambda p: p.product_id):
        if product.category in CARRIER_SLOTS:
            if product.category == "spf" and melasma_spf_required and (
                "iron_oxides" not in product.actives
            ):
                continue
            by_slot[product.category].append(product)
        elif product.category == "treatment" and (therapy_primary is not None or treatment_actives):
            declared = {
                active.get("name"): active.get("strength")
                for active in product.drug_actives
                if isinstance(active, dict)
            }
            if therapy_primary is None:
                matched_actives = set(declared) & treatment_actives
                matched = bool(matched_actives) and set(declared).issubset(treatment_actives)
                for active in matched_actives:
                    spec = treatment_specs.get(active)
                    if spec and spec.get("strength") and not _matches_policy_strength(
                        declared.get(active), spec["strength"]
                    ):
                        matched = False
                expected_exposure = "leave_on"
                expected_cadence = "per_label"
                expected_amount = None
            else:
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
                expected_exposure = therapy_primary["exposure"]
                expected_cadence = therapy_primary["cadence"]
                expected_amount = therapy_primary.get("amount")
            if (
                matched
                and product.exposure == expected_exposure
                and (
                    expected_cadence == "per_label"
                    or product.cadence == expected_cadence
                )
                and (
                    expected_amount is None or product.amount == expected_amount
                )
            ):
                by_slot["treatment"].append(product)
        elif product.category == "serum":
            wanted = frozenset().union(*(
                frozenset(t.target_actives) or knowledge.lesion_actives.get(
                    t.lesion_type, frozenset()
                )
                for t in targets
            )) if targets else frozenset()
            # Admitted only when it targets a detected concern AND carries no
            # treatment-class active that no detected concern calls for: a
            # retinol serum is a dryness candidate by its humectants alone,
            # and a retinoid must not reach a patient with no indication.
            if set(product.actives) & wanted and not (
                set(product.actives) & knowledge.treatment_actives - wanted
            ):
                by_slot["serum"].append(product)
        elif product.category == "scar_care" and scar_care_required:
            if (
                "scar_care" in product.evidence_roles
                and "silicone_scar_care" in product.actives
                and product.format in {"silicone_sheet", "silicone_gel"}
            ):
                by_slot["scar_care"].append(product)
    return by_slot
