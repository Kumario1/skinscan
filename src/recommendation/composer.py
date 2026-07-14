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
EVIDENCE_GRADE_SCORE = {
    "verified_label": 3.0,
    "guideline_class_plus_verified_product_form": 3.0,
    "reviewed_policy": 3.0,
    "synthetic_test": 2.0,
    "complete": 1.0,
    "unknown": 0.0,
}


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
        evidence = EVIDENCE_GRADE_SCORE.get(product.evidence_grade, 0.5)
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

    selected: dict[str, Product] = {}
    alternatives: dict[str, list[Product]] = {}
    diagnostics = eligibility_diagnostics or EligibilityDiagnostics(
        requested, collect_details=bool(eligibility_rejections)
    )
    for key, reasons in (eligibility_rejections or {}).items():
        if ":" in key and not key.startswith("role:"):
            role, product_id = key.split(":", 1)
            diagnostics.record(role, product_id, list(reasons))
    explanation: list[dict[str, object]] = []
    used_ids: set[str] = set()

    for role in requested:
        ranked, ranking_mode = rank_equivalents(
            list(eligible_by_role.get(role, [])), profile,
            concern_scorer=concern_scorer, pooled_ranker=pooled_ranker,
        )
        ranked = [product for product in ranked if product.product_id not in used_ids]
        if not ranked:
            diagnostics.mark_missing(role)
            continue
        selected[role] = ranked[0]
        used_ids.add(ranked[0].product_id)
        alternatives[role] = ranked[1:1 + alternative_limit]
        item: dict[str, object] = {
            "role": role,
            "product_id": ranked[0].product_id,
            "ranking_basis": ranking_mode,
        }
        if role == "treatment" and therapy_plan.primary is not None:
            item["delivered_active"] = therapy_plan.primary.therapy
            matching = [active for active in ranked[0].drug_actives
                        if active.name == therapy_plan.primary.therapy]
            item["strength"] = matching[0].strength if matching else None
        explanation.append(item)

    # A multi-role SKU can be an earlier role's alternative and a later role's
    # selected product. Alternatives are presentation choices, so remove any
    # SKU selected anywhere after all roles have been composed.
    selected_ids = {product.product_id for product in selected.values()}
    alternatives = {
        role: [product for product in products if product.product_id not in selected_ids]
        for role, products in alternatives.items()
    }

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
