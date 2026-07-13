from src.recommendation.composer import compose_regimen
from src.recommendation.schema import (
    CareDecision, DecisionEvidence, Product, TherapyOption, TherapyPlan, UserProfile,
    VerifiedActive,
)


def profile(**overrides):
    values = {"skin_type": "oily", "age_years": 25,
              "pregnancy_status": "not_pregnant"}
    values.update(overrides)
    return UserProfile(**values)


def decision(disposition="active_treatment"):
    return CareDecision("routine", [], disposition, [], "synthetic", True)


def product(product_id, role, active=None, grade="verified_label"):
    category = "spf" if role == "sunscreen" else role
    values = dict(
        product_id=product_id, name=product_id, brand="Example", category=category,
        intended_areas=["face"], routine_roles=[role],
        format={"cleanser": "cleanser", "treatment": "gel",
                "moisturizer": "cream", "sunscreen": "sunscreen"}[role],
        exposure="rinse_off" if role == "cleanser" else "leave_on",
        comedogenic_claim=("claimed_noncomedogenic"
                           if role in {"moisturizer", "sunscreen"} else "not_claimed"),
        cadence="per_label", cadence_source=f"synthetic://{product_id}",
        evidence_grade=grade, catalog_schema_version="2",
    )
    if role == "treatment":
        values.update(
            actives=[active], drug_actives=[VerifiedActive(active, "10%", "synthetic://label")],
            label_source="synthetic://label", label_verified_at="2026-07-13",
        )
    if role == "sunscreen":
        values.update(broad_spectrum=True, spf=30, label_source="synthetic://spf",
                      label_verified_at="2026-07-13")
    return Product(**values)


def active_plan():
    option = TherapyOption("azelaic_acid", "10%", "leave_on", "per_label",
                           "treatment", cadence_source="synthetic://label")
    return TherapyPlan(12, 12, option, [],
                       ["cleanser", "moisturizer", "sunscreen"], [], "synthetic")


def test_coverage_promotion_cannot_create_a_second_selected_treatment():
    candidates = {
        "cleanser": [product("cleanser", "cleanser")],
        "treatment": [product("best", "treatment", "azelaic_acid"),
                      product("deep-carrier", "treatment", "azelaic_acid")],
        "moisturizer": [product("moisturizer", "moisturizer")],
        "sunscreen": [product("spf", "sunscreen")],
    }
    result = compose_regimen(decision(), active_plan(), candidates, profile())
    assert result.selected_products["treatment"].product_id == "best"
    assert [item.product_id for item in result.alternatives["treatment"]] == ["deep-carrier"]
    assert len([role for role in result.selected_products if role == "treatment"]) == 1
    assert result.validation_errors == []


def test_alternatives_are_configurable_and_never_scheduled():
    plan = active_plan()
    candidates = {
        "treatment": [product("a", "treatment", "azelaic_acid"),
                      product("b", "treatment", "azelaic_acid"),
                      product("c", "treatment", "azelaic_acid")],
    }
    result = compose_regimen(
        decision(), plan, candidates, profile(), alternative_limit=1
    )
    assert len(result.alternatives["treatment"]) == 1
    assert [step.role for step in result.selected_regimen["pm"]] == ["treatment"]


def test_supportive_only_plan_contains_no_treatment():
    plan = TherapyPlan(None, None, None, [],
                       ["cleanser", "moisturizer", "sunscreen"], [], "synthetic")
    candidates = {
        "cleanser": [product("cleanser", "cleanser")],
        "treatment": [product("aza", "treatment", "azelaic_acid")],
        "moisturizer": [product("moist", "moisturizer")],
        "sunscreen": [product("spf", "sunscreen")],
    }
    result = compose_regimen(
        decision("supportive_only"), plan, candidates, profile()
    )
    assert "treatment" not in result.selected_products
    assert all(step.role != "treatment" for steps in result.selected_regimen.values()
               for step in steps)


def test_missing_treatment_is_explicit_and_not_substituted():
    result = compose_regimen(
        decision(), active_plan(),
        {"treatment": []}, profile(),
    )
    assert "treatment" not in result.selected_products
    assert result.eligibility_rejections["role:treatment"] == ["no_eligible_product"]
    assert result.validation_errors == []


def test_concern_scorer_precedes_pooled_stats_and_fallback_is_truthful():
    a = product("a", "treatment", "azelaic_acid")
    b = product("b", "treatment", "azelaic_acid")

    class Score:
        def __init__(self, values): self.values = values
        def score(self, item, _profile): return self.values[item.product_id]

    result = compose_regimen(
        decision(), active_plan(), {"treatment": [a, b]}, profile(),
        concern_scorer=Score({"a": 1, "b": 0}), pooled_ranker=Score({"a": 0, "b": 100}),
    )
    assert result.selected_products["treatment"] == a
    assert result.explanation[0]["ranking_basis"] == "concern_specific"
    fallback = compose_regimen(
        decision(), active_plan(), {"treatment": [a, b]}, profile(),
        pooled_ranker=Score({"a": 0, "b": 100}),
    )
    assert fallback.selected_products["treatment"] == b
    assert fallback.explanation[0]["ranking_basis"] == "pooled_general_fallback"
