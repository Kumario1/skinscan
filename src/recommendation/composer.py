"""Rank already-eligible equivalents and compose one product per role."""
from __future__ import annotations

from collections.abc import Mapping

from .schema import (
    CareDecision,
    EligibilityDiagnostics,
    Product,
    Recommendation,
    RoutineInstruction,
    TherapyPlan,
    UserProfile,
)


ROLE_ORDER = ("cleanser", "treatment", "moisturizer", "sunscreen")
# Every grade the catalogs actually carry is mapped: an unmapped grade takes the
# fallback below, which must never let an undeclared string outrank a product
# that honestly says "unknown".
EVIDENCE_GRADE_SCORE = {
    "verified_label": 3.0,
    "guideline_class_plus_verified_product_form": 3.0,
    "reviewed_policy": 3.0,
    "regulatory_label": 3.0,
    "synthetic_test": 2.0,
    "complete": 1.0,
    "manufacturer_product_page": 1.0,
    "pending_review": 0.0,  # imported, not yet reviewed — no better than unknown
    "unknown": 0.0,
}
UNGRADED_SCORE = 0.0


def _score(scorer, product: Product, profile: UserProfile) -> float:
    if scorer is None:
        return 0.0
    value = scorer.score(product, profile)
    if isinstance(value, Mapping):
        value = value.get("score", 0.0)
    return float(value)


def rank_equivalents(
    products: list[Product],
    profile: UserProfile,
    *,
    concern_scorer=None,
    pooled_ranker=None,
) -> tuple[list[Product], str]:
    """Order a homogeneous eligible role bucket with pooled stats last."""
    mode = "concern_specific" if concern_scorer is not None else "pooled_general_fallback"

    def key(product: Product) -> tuple[float, float, float, float, float, str]:
        concern = _score(concern_scorer, product, profile)
        tolerability = -float(len(product.irritant_features))
        evidence = EVIDENCE_GRADE_SCORE.get(product.evidence_grade, UNGRADED_SCORE)
        budget = 0.0
        if (profile.max_price_usd is not None and product.price_usd is not None
                and not product.price_is_stale):
            budget = 1.0 if product.price_usd <= profile.max_price_usd else -1.0
        pooled = _score(pooled_ranker, product, profile)
        return (-concern, -tolerability, -evidence, -budget, -pooled, product.product_id)

    return sorted(products, key=key), mode


def _instruction(role: str, product: Product, plan: TherapyPlan, slot: str) -> RoutineInstruction:
    if role == "treatment" and plan.primary is not None:
        option = plan.primary
        cadence = product.cadence if option.cadence == "per_label" else option.cadence
        source = (
            product.cadence_source if option.cadence == "per_label"
            else option.cadence_source
        )
        amount = option.amount if option.amount is not None else product.amount
        if option.amount is not None:
            source = option.amount_source or source
        elif product.amount is not None:
            source = product.amount_source or source
        return RoutineInstruction(
            role, slot, cadence or "unknown", amount, source,
        )
    return RoutineInstruction(
        role, slot, product.cadence or "unknown", product.amount, product.cadence_source,
    )


def _assign_roles(requested, ranked_by_role) -> dict[str, Product]:
    """Fill as many roles as the eligible SKUs allow, each role taking the
    best-ranked product it can hold.

    One SKU can serve two roles (a moisturizer with SPF), and claiming it for
    the earlier role must not cost the later one its slot when the earlier role
    has another option — SPF is non-negotiable (RULES.md 3). So a contested SKU
    re-homes its current holder instead of being refused. With no contested SKU
    this is exactly first-choice-per-role.
    """
    holder: dict[str, str] = {}       # product_id -> role holding it
    chosen: dict[str, Product] = {}   # role -> product

    def claim(role: str, tried: set[str]) -> bool:
        for product in ranked_by_role.get(role, ()):
            if product.product_id in tried:
                continue
            tried.add(product.product_id)
            current = holder.get(product.product_id)
            if current is None or claim(current, tried):
                holder[product.product_id] = role
                chosen[role] = product
                return True
        return False

    for role in requested:
        claim(role, set())
    return chosen


def compose_regimen(
    decision: CareDecision,
    therapy_plan: TherapyPlan,
    eligible_by_role: Mapping[str, list[Product]],
    profile: UserProfile,
    *,
    eligibility_diagnostics: EligibilityDiagnostics | None = None,
    eligibility_rejections: Mapping[str, list[str]] | None = None,
    concern_scorer=None,
    pooled_ranker=None,
    alternative_limit: int = 2,
) -> Recommendation:
    if alternative_limit < 0:
        raise ValueError("alternative_limit must be non-negative")
    requested = list(therapy_plan.support_roles)
    if therapy_plan.primary is not None:
        requested.append(therapy_plan.primary.role)
    requested = [role for role in ROLE_ORDER if role in requested]

    alternatives: dict[str, list[Product]] = {}
    diagnostics = eligibility_diagnostics or EligibilityDiagnostics(
        requested, collect_details=bool(eligibility_rejections)
    )
    for key, reasons in (eligibility_rejections or {}).items():
        if ":" in key and not key.startswith("role:"):
            role, product_id = key.split(":", 1)
            diagnostics.record(role, product_id, list(reasons))
    explanation: list[dict[str, object]] = []

    ranked_by_role: dict[str, list[Product]] = {}
    ranking_modes: dict[str, str] = {}
    for role in requested:
        ranked_by_role[role], ranking_modes[role] = rank_equivalents(
            list(eligible_by_role.get(role, [])), profile,
            concern_scorer=concern_scorer, pooled_ranker=pooled_ranker,
        )
    selected = _assign_roles(requested, ranked_by_role)

    # A SKU selected for any role is no longer an alternative anywhere.
    selected_ids = {product.product_id for product in selected.values()}
    for role in requested:
        product = selected.get(role)
        if product is None:
            diagnostics.mark_missing(role)
            continue
        alternatives[role] = [
            candidate for candidate in ranked_by_role[role]
            if candidate.product_id not in selected_ids
        ][:alternative_limit]
        item: dict[str, object] = {
            "role": role,
            "product_id": product.product_id,
            "ranking_basis": ranking_modes[role],
        }
        if role == "treatment" and therapy_plan.primary is not None:
            item["delivered_active"] = therapy_plan.primary.therapy
            matching = [active for active in product.drug_actives
                        if active.name == therapy_plan.primary.therapy]
            item["strength"] = matching[0].strength if matching else None
        # D-033 relaxed eligibility to let verified prescription-strength rows
        # through "while advising the user to see a doctor" -- this is that
        # advice. Eligibility was this engine's only Rx control, so without the
        # annotation a placed Rx product is indistinguishable from an OTC one
        # in the output. `is not True`: otc_drug is optional, and a drug whose
        # label has not proven it OTC is presented as a prescription -- unknown
        # is data, never a favorable default (recsys/explain.is_prescription is
        # the same rule).
        if product.drug_actives and product.otc_drug is not True:
            item["prescription"] = True
            item["referral_note"] = (
                "prescription — a doctor or dermatologist can advise on and "
                "prescribe this"
            )
        explanation.append(item)

    regimen: dict[str, list[RoutineInstruction]] = {"am": [], "pm": []}
    for role in ROLE_ORDER:
        product = selected.get(role)
        if product is None:
            continue
        if role == "sunscreen":
            regimen["am"].append(_instruction(role, product, therapy_plan, "am"))
        elif role == "treatment":
            regimen["pm"].append(_instruction(role, product, therapy_plan, "pm"))
        else:
            regimen["am"].append(_instruction(role, product, therapy_plan, "am"))
            regimen["pm"].append(_instruction(role, product, therapy_plan, "pm"))

    recommendation = Recommendation(
        decision=decision,
        therapy_plan=therapy_plan,
        selected_products=selected,
        selected_regimen=regimen,
        alternatives=alternatives,
        eligibility_diagnostics=diagnostics,
        explanation=explanation,
        flags=[],
        validation_errors=[],
    )
    from .validator import validate_recommendation
    recommendation.validation_errors = validate_recommendation(recommendation, profile)
    return recommendation
