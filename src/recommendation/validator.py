"""Whole-regimen invariant validation."""
from __future__ import annotations

from .eligibility import check_eligibility
from .schema import Product, Recommendation, UserProfile


def _carried(product: Product) -> set[str]:
    return set(product.actives) | {active.name for active in product.drug_actives}


def validate_recommendation(
    recommendation: Recommendation,
    profile: UserProfile,
) -> list[str]:
    errors: list[str] = []

    def add(code: str) -> None:
        if code not in errors:
            errors.append(code)

    required = set(recommendation.therapy_plan.support_roles)
    if recommendation.therapy_plan.primary is not None:
        required.add(recommendation.therapy_plan.primary.role)
    for role in sorted(required):
        if role not in recommendation.selected_products:
            reasons = recommendation.eligibility_rejections.get(f"role:{role}", [])
            if not reasons:
                add(f"required_role_missing_without_reason:{role}")

    # The public type is one product per key; reject malformed callers that
    # bypass it with a list and enforce default one-SKU-per-role.
    product_ids: dict[str, str] = {}
    selected_so_far: dict[str, Product] = {}
    for role, product in recommendation.selected_products.items():
        if not isinstance(product, Product):
            add(f"more_than_one_or_invalid_selected_product:{role}")
            continue
        if product.product_id in product_ids:
            add(f"sku_selected_for_multiple_roles:{product.product_id}")
        product_ids[product.product_id] = role
        therapy = recommendation.therapy_plan.primary if role == "treatment" else None
        result = check_eligibility(product, role, therapy, profile, selected_so_far)
        for reason in result.reasons:
            add(f"selected_product_ineligible:{role}:{reason}")
        selected_so_far[role] = product

    selected_ids = set(product_ids)
    for role, products in recommendation.alternatives.items():
        seen: set[str] = set()
        for product in products:
            if not isinstance(product, Product):
                add(f"invalid_alternative:{role}")
                continue
            if product.product_id in selected_ids:
                add(f"alternative_is_selected:{role}:{product.product_id}")
            if product.product_id in seen:
                add(f"duplicate_alternative:{role}:{product.product_id}")
            seen.add(product.product_id)
            therapy = recommendation.therapy_plan.primary if role == "treatment" else None
            other_selected = {
                selected_role: selected_product
                for selected_role, selected_product in recommendation.selected_products.items()
                if selected_role != role and isinstance(selected_product, Product)
            }
            result = check_eligibility(product, role, therapy, profile, other_selected)
            for reason in result.reasons:
                add(f"alternative_ineligible:{role}:{product.product_id}:{reason}")

    scheduled: dict[tuple[str, str], int] = {}
    for slot in ("am", "pm"):
        for instruction in recommendation.selected_regimen.get(slot, []):
            if instruction.slot != slot:
                add(f"instruction_slot_mismatch:{instruction.role}")
            if instruction.role not in recommendation.selected_products:
                add(f"instruction_has_no_selected_product:{instruction.role}")
            key = (slot, instruction.role)
            scheduled[key] = scheduled.get(key, 0) + 1
            if scheduled[key] > 1:
                add(f"role_repeated_in_slot:{slot}:{instruction.role}")
            if not instruction.source:
                add(f"instruction_source_missing:{slot}:{instruction.role}")
            if not instruction.cadence or instruction.cadence == "unknown":
                add(f"instruction_cadence_unknown:{slot}:{instruction.role}")
    if ("pm", "sunscreen") in scheduled:
        add("sunscreen_scheduled_pm")
    if ("sunscreen" in recommendation.selected_products
            and ("am", "sunscreen") not in scheduled):
        add("required_sunscreen_not_scheduled_am")
    if (scheduled.get(("am", "treatment"), 0)
            + scheduled.get(("pm", "treatment"), 0) > 1):
        add("treatment_repeated_across_slots")

    for item in recommendation.explanation:
        product_id = item.get("product_id")
        role = item.get("role")
        product = recommendation.selected_products.get(role) if isinstance(role, str) else None
        if product is None or product.product_id != product_id:
            add("explanation_product_mismatch")
            continue
        claimed = item.get("delivered_active")
        if claimed is not None and claimed not in _carried(product):
            add(f"explanation_active_not_delivered:{claimed}")
        strength = item.get("strength")
        if strength is not None:
            verified = {active.strength for active in product.drug_actives
                        if active.name == claimed}
            if strength not in verified:
                add(f"explanation_strength_not_delivered:{strength}")

    return errors
