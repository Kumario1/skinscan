import pytest

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
            otc_drug=True,
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


# --- one SKU, two roles -------------------------------------------------------

def spf_moisturizer(product_id="spf-moisturizer", grade="verified_label"):
    """A real shelf item: a moisturizer that is also a broad-spectrum sunscreen."""
    return Product(
        product_id=product_id, name=product_id, brand="Example", category="moisturizer",
        intended_areas=["face"], routine_roles=["moisturizer", "sunscreen"],
        format="cream", exposure="leave_on",
        comedogenic_claim="claimed_noncomedogenic",
        cadence="per_label", cadence_source=f"synthetic://{product_id}",
        evidence_grade=grade, catalog_schema_version="2",
        broad_spectrum=True, spf=50,
        label_source="synthetic://spf", label_verified_at="2026-07-13",
    )


def test_multi_role_sku_does_not_cost_a_mandatory_role_its_slot():
    """SPF is non-negotiable (RULES.md 3). When the only sunscreen-eligible SKU
    is also the best moisturizer, it must serve sunscreen and the moisturizer
    role must take its next option -- not be dropped as unavailable."""
    shared = spf_moisturizer()
    plain = product("plain-moisturizer", "moisturizer", grade="complete")
    candidates = {
        "cleanser": [product("cleanser", "cleanser")],
        "treatment": [product("best", "treatment", "azelaic_acid")],
        "moisturizer": [shared, plain],   # shared outranks plain on evidence
        "sunscreen": [shared],
    }
    result = compose_regimen(decision(), active_plan(), candidates, profile())

    assert result.selected_products["sunscreen"].product_id == "spf-moisturizer"
    assert result.selected_products["moisturizer"].product_id == "plain-moisturizer"
    assert result.validation_errors == []
    # and the SKU serving sunscreen is not also offered as a moisturizer option
    assert "spf-moisturizer" not in [
        item.product_id for item in result.alternatives.get("moisturizer", [])
    ]


def test_a_genuinely_unfillable_role_is_still_reported_missing():
    """Re-homing must not invent coverage: with no sunscreen-eligible SKU at
    all, the role stays missing."""
    candidates = {
        "cleanser": [product("cleanser", "cleanser")],
        "treatment": [product("best", "treatment", "azelaic_acid")],
        "moisturizer": [product("moisturizer", "moisturizer")],
        "sunscreen": [],
    }
    result = compose_regimen(decision(), active_plan(), candidates, profile())
    assert "sunscreen" not in result.selected_products


def test_a_single_shared_sku_is_never_double_booked():
    """Only one product exists and it serves both roles. It can hold exactly one
    slot, so exactly one role is filled and the other is honestly reported
    missing -- the SKU is never counted against two roles at once.

    Which of the two wins is deliberately not asserted: no assignment fills both,
    and nothing in RULES.md establishes a priority for that contention.
    """
    shared = spf_moisturizer()
    candidates = {
        "cleanser": [product("cleanser", "cleanser")],
        "treatment": [product("best", "treatment", "azelaic_acid")],
        "moisturizer": [shared],
        "sunscreen": [shared],
    }
    result = compose_regimen(decision(), active_plan(), candidates, profile())
    selected = {role: p.product_id for role, p in result.selected_products.items()}

    filled = [role for role in ("moisturizer", "sunscreen") if role in selected]
    assert len(filled) == 1, f"one SKU cannot fill two roles, got {filled}"
    assert selected[filled[0]] == "spf-moisturizer"


def test_uncontested_roles_still_take_their_first_choice():
    """The common case must be untouched: no shared SKU -> best product per role."""
    candidates = {
        "cleanser": [product("cleanser", "cleanser")],
        "treatment": [product("best", "treatment", "azelaic_acid", grade="verified_label"),
                      product("second", "treatment", "azelaic_acid", grade="complete")],
        "moisturizer": [product("m-best", "moisturizer", grade="verified_label"),
                        product("m-second", "moisturizer", grade="complete")],
        "sunscreen": [product("spf", "sunscreen")],
    }
    result = compose_regimen(decision(), active_plan(), candidates, profile())
    assert result.selected_products["treatment"].product_id == "best"
    assert result.selected_products["moisturizer"].product_id == "m-best"
    assert result.selected_products["sunscreen"].product_id == "spf"


# --- evidence grade ranking ---------------------------------------------------

def test_every_grade_the_catalogs_carry_outranks_an_ungraded_product():
    """regulatory_label / manufacturer_product_page / pending_review all appear
    in the real catalogs; none may silently fall to a default that outranks an
    honest 'unknown'."""
    from src.recommendation.composer import EVIDENCE_GRADE_SCORE, rank_equivalents

    for grade in ("regulatory_label", "manufacturer_product_page", "pending_review"):
        assert grade in EVIDENCE_GRADE_SCORE, grade
    assert EVIDENCE_GRADE_SCORE["regulatory_label"] > EVIDENCE_GRADE_SCORE["synthetic_test"]
    assert EVIDENCE_GRADE_SCORE["pending_review"] == EVIDENCE_GRADE_SCORE["unknown"]

    ranked, _ = rank_equivalents(
        [product("ungraded", "moisturizer", grade="not_a_declared_grade"),
         product("known", "moisturizer", grade="unknown")],
        profile(),
    )
    assert [item.product_id for item in ranked] == ["known", "ungraded"], (
        "an undeclared grade must never outrank a product that declares unknown"
    )


def test_a_regulatory_label_outranks_a_test_fixture():
    from src.recommendation.composer import rank_equivalents

    ranked, _ = rank_equivalents(
        [product("fixture", "moisturizer", grade="synthetic_test"),
         product("real", "moisturizer", grade="regulatory_label")],
        profile(),
    )
    assert [item.product_id for item in ranked] == ["real", "fixture"]


def test_reassignment_chains_through_more_than_one_role():
    """A two-hop re-home: sunscreen wants the SKU moisturizer holds, and
    moisturizer's only other option is the SKU cleanser holds, so cleanser must
    step aside too. All four roles are fillable and all four must be filled."""
    shared_spf = spf_moisturizer("shared-spf")            # moisturizer + sunscreen
    dual_wash = Product(                                   # cleanser + moisturizer
        product_id="dual-wash", name="dual-wash", brand="Example", category="moisturizer",
        intended_areas=["face"], routine_roles=["cleanser", "moisturizer"],
        format="cream", exposure="leave_on",
        comedogenic_claim="claimed_noncomedogenic",
        cadence="per_label", cadence_source="synthetic://dual-wash",
        evidence_grade="verified_label", catalog_schema_version="2",
    )
    plain_cleanser = product("plain-cleanser", "cleanser", grade="complete")
    candidates = {
        # each earlier role's favourite is the next role's only fallback
        "cleanser": [dual_wash, plain_cleanser],
        "treatment": [product("best", "treatment", "azelaic_acid")],
        "moisturizer": [shared_spf, dual_wash],
        "sunscreen": [shared_spf],
    }
    result = compose_regimen(decision(), active_plan(), candidates, profile())
    selected = {role: p.product_id for role, p in result.selected_products.items()}

    assert selected == {
        "cleanser": "plain-cleanser",
        "treatment": "best",
        "moisturizer": "dual-wash",
        "sunscreen": "shared-spf",
    }, selected
    assert len(set(selected.values())) == 4, "no SKU may serve two roles"


def test_assignment_is_deterministic_under_contention():
    shared = spf_moisturizer()
    candidates = {
        "cleanser": [product("cleanser", "cleanser")],
        "treatment": [product("best", "treatment", "azelaic_acid")],
        "moisturizer": [shared, product("plain", "moisturizer", grade="complete")],
        "sunscreen": [shared],
    }
    runs = {
        tuple(sorted((role, p.product_id) for role, p in
                     compose_regimen(decision(), active_plan(), candidates,
                                     profile()).selected_products.items()))
        for _ in range(5)
    }
    assert len(runs) == 1


# --- ranking axes -------------------------------------------------------------
# Mutation testing showed these were unpinned: perturbing the grade scores, the
# tolerability sign, or the budget comparison killed nothing.

from src.recommendation.composer import EVIDENCE_GRADE_SCORE, UNGRADED_SCORE, rank_equivalents


def _ranked(products, prof=None, **kwargs):
    ranked, _ = rank_equivalents(products, prof or profile(), **kwargs)
    return [p.product_id for p in ranked]


def test_evidence_grades_form_the_intended_total_order():
    """The whole ladder, not just neighbouring pairs: a grade silently moving
    band would reorder every tie."""
    ladder = [
        ("verified_label", 3.0), ("guideline_class_plus_verified_product_form", 3.0),
        ("reviewed_policy", 3.0), ("regulatory_label", 3.0),
        ("synthetic_test", 2.0),
        ("complete", 1.0), ("manufacturer_product_page", 1.0),
        ("pending_review", 0.0), ("unknown", 0.0),
    ]
    assert EVIDENCE_GRADE_SCORE == dict(ladder)
    assert UNGRADED_SCORE == 0.0, "an undeclared grade never outranks an honest unknown"


def test_a_more_irritating_product_ranks_below_a_gentler_equal():
    gentle = product("gentle", "moisturizer")
    harsh = product("harsh", "moisturizer")
    harsh.irritant_features = ["fragrance", "denatured_alcohol"]
    assert _ranked([harsh, gentle]) == ["gentle", "harsh"]


def test_tolerability_outranks_evidence_completeness():
    """RULES.md 8.5 order: concern, then tolerability, then evidence."""
    gentle_thin = product("gentle-thin", "moisturizer", grade="unknown")
    harsh_proven = product("harsh-proven", "moisturizer", grade="verified_label")
    harsh_proven.irritant_features = ["fragrance"]
    assert _ranked([harsh_proven, gentle_thin]) == ["gentle-thin", "harsh-proven"]


def test_a_product_within_budget_outranks_one_over_it():
    cheap = product("cheap", "moisturizer")
    dear = product("dear", "moisturizer")
    cheap.price_usd, dear.price_usd = 20.0, 80.0
    cheap.price_is_stale = dear.price_is_stale = False
    assert _ranked([dear, cheap], profile(max_price_usd=50.0)) == ["cheap", "dear"]


def test_the_budget_bound_is_inclusive():
    exact = product("exact", "moisturizer")
    over = product("over", "moisturizer")
    exact.price_usd, over.price_usd = 50.0, 50.01
    exact.price_is_stale = over.price_is_stale = False
    assert _ranked([over, exact], profile(max_price_usd=50.0)) == ["exact", "over"]


def test_a_stale_price_is_not_used_to_judge_budget():
    """Ranking on a price we no longer trust would quietly mis-sort; with no
    usable price the axis is neutral and the product_id tiebreak decides."""
    stale_cheap = product("a-stale", "moisturizer")
    fresh_dear = product("b-fresh", "moisturizer")
    stale_cheap.price_usd, fresh_dear.price_usd = 10.0, 80.0
    stale_cheap.price_is_stale, fresh_dear.price_is_stale = True, False
    # the over-budget fresh price is penalised; the stale cheap one is neutral
    assert _ranked([fresh_dear, stale_cheap], profile(max_price_usd=50.0)) == \
        ["a-stale", "b-fresh"]


def test_no_budget_in_the_profile_leaves_the_axis_neutral():
    a = product("a", "moisturizer")
    b = product("b", "moisturizer")
    a.price_usd, b.price_usd = 900.0, 5.0
    a.price_is_stale = b.price_is_stale = False
    assert _ranked([b, a], profile()) == ["a", "b"], "product_id breaks the tie, not price"


def test_ties_break_on_product_id_so_ordering_is_deterministic():
    assert _ranked([product("c", "moisturizer"), product("a", "moisturizer"),
                    product("b", "moisturizer")]) == ["a", "b", "c"]


def test_a_scorer_may_return_a_mapping_with_a_score_key():
    class MappingScore:
        def score(self, item, _profile):
            return {"score": 1.0 if item.product_id == "b" else 0.0, "why": "..."}

    assert _ranked([product("a", "treatment", "azelaic_acid"),
                    product("b", "treatment", "azelaic_acid")],
                   concern_scorer=MappingScore()) == ["b", "a"]


def test_alternative_limit_must_be_non_negative():
    with pytest.raises(ValueError, match="alternative_limit must be non-negative"):
        compose_regimen(decision(), active_plan(),
                        {"treatment": [product("a", "treatment", "azelaic_acid")]},
                        profile(), alternative_limit=-1)


def test_alternative_limit_zero_selects_without_offering_alternatives():
    result = compose_regimen(
        decision(), active_plan(),
        {"treatment": [product("a", "treatment", "azelaic_acid"),
                       product("b", "treatment", "azelaic_acid")]},
        profile(), alternative_limit=0)
    assert result.selected_products["treatment"].product_id == "a"
    assert result.alternatives["treatment"] == []


# --- directions: what the user is actually told to do -------------------------
# _instruction resolves cadence/amount/source between the therapy option and the
# product label. Mutation testing showed every branch of it was unpinned.

def _treat_plan(cadence="per_label", cadence_source="synthetic://label",
                amount=None, amount_source=None):
    option = TherapyOption("azelaic_acid", "10%", "leave_on", cadence, "treatment",
                           cadence_source=cadence_source, amount=amount,
                           amount_source=amount_source)
    return TherapyPlan(12, 12, option, [], ["cleanser", "moisturizer", "sunscreen"],
                       [], "synthetic")


def _pm_treatment(plan, **product_fields):
    item = product("t", "treatment", "azelaic_acid")
    for key, value in product_fields.items():
        setattr(item, key, value)
    result = compose_regimen(decision(), plan, {"treatment": [item]}, profile())
    return next(s for s in result.selected_regimen["pm"] if s.role == "treatment")


def test_per_label_cadence_defers_to_the_product_label():
    step = _pm_treatment(_treat_plan(cadence="per_label"),
                         cadence="once_daily", cadence_source="label://aza")
    assert step.cadence == "once_daily"
    assert step.source == "label://aza", "the label is cited, not the policy"


def test_an_explicit_policy_cadence_overrides_the_product_label():
    step = _pm_treatment(_treat_plan(cadence="twice_weekly",
                                     cadence_source="policy://reviewed"),
                         cadence="once_daily", cadence_source="label://aza")
    assert step.cadence == "twice_weekly"
    assert step.source == "policy://reviewed"


def test_a_product_with_no_cadence_is_reported_unknown_not_blank():
    step = _pm_treatment(_treat_plan(cadence="per_label"), cadence=None)
    assert step.cadence == "unknown"


def test_the_policy_amount_wins_and_cites_its_own_source():
    step = _pm_treatment(
        _treat_plan(amount="pea_sized", amount_source="policy://amount"),
        amount="a thin layer", amount_source="label://amount")
    assert step.amount == "pea_sized"
    assert step.source == "policy://amount"


def test_the_product_amount_is_used_when_the_policy_states_none():
    step = _pm_treatment(_treat_plan(amount=None),
                         amount="a thin layer", amount_source="label://amount")
    assert step.amount == "a thin layer"
    assert step.source == "label://amount"


def test_an_amount_with_no_source_of_its_own_keeps_the_cadence_source():
    """source must never go blank: it falls back rather than being dropped."""
    step = _pm_treatment(
        _treat_plan(cadence="twice_weekly", cadence_source="policy://reviewed",
                    amount="pea_sized", amount_source=None),
        cadence="once_daily")
    assert step.amount == "pea_sized"
    assert step.source == "policy://reviewed"


def test_a_support_step_is_described_purely_from_its_own_label():
    """No therapy option applies to a cleanser; its directions are the label's."""
    item = product("c", "cleanser")
    item.cadence, item.cadence_source, item.amount = "twice_daily", "label://cleanser", "a dab"
    result = compose_regimen(decision(), active_plan(), {"cleanser": [item]}, profile())
    step = next(s for s in result.selected_regimen["am"] if s.role == "cleanser")
    assert (step.cadence, step.amount, step.source) == ("twice_daily", "a dab", "label://cleanser")


def test_the_explanation_names_the_delivered_active_and_its_strength():
    item = product("t", "treatment", "azelaic_acid")
    result = compose_regimen(decision(), active_plan(), {"treatment": [item]}, profile())
    entry = next(e for e in result.explanation if e["role"] == "treatment")
    assert entry["delivered_active"] == "azelaic_acid"
    assert entry["strength"] == "10%"


def test_strength_is_none_when_the_product_does_not_carry_the_planned_active():
    item = product("t", "treatment", "benzoyl_peroxide")
    result = compose_regimen(decision(), active_plan(), {"treatment": [item]}, profile())
    entry = next(e for e in result.explanation if e["role"] == "treatment")
    assert entry["delivered_active"] == "azelaic_acid"
    assert entry["strength"] is None, "no matching drug active means no strength to claim"
