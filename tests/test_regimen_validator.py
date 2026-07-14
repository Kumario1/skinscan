from src.recommendation.schema import RoutineInstruction
from src.recommendation.validator import validate_recommendation

from test_regimen_composer import active_plan, decision, product, profile
from src.recommendation.composer import compose_regimen


def valid_recommendation():
    return compose_regimen(
        decision(), active_plan(),
        {
            "cleanser": [product("cleanser", "cleanser")],
            "treatment": [product("aza", "treatment", "azelaic_acid"),
                          product("aza-alt", "treatment", "azelaic_acid")],
            "moisturizer": [product("moist", "moisturizer")],
            "sunscreen": [product("spf", "sunscreen")],
        },
        profile(),
    )


def test_valid_composed_regimen_has_no_errors():
    assert validate_recommendation(valid_recommendation(), profile()) == []


def test_required_role_missing_without_reason_is_rejected():
    rec = valid_recommendation()
    del rec.selected_products["treatment"]
    rec.eligibility_rejections.pop("role:treatment", None)
    assert "required_role_missing_without_reason:treatment" in validate_recommendation(
        rec, profile()
    )


def test_alternative_cannot_also_be_selected():
    rec = valid_recommendation()
    rec.alternatives["treatment"] = [rec.selected_products["treatment"]]
    assert any(error.startswith("alternative_is_selected")
               for error in validate_recommendation(rec, profile()))


def test_alternative_must_be_eligible_with_other_selected_roles():
    rec = valid_recommendation()
    invalid = product("bad-alt", "treatment", "azelaic_acid")
    invalid.label_source = None
    rec.alternatives["treatment"] = [invalid]
    errors = validate_recommendation(rec, profile())
    assert any("alternative_ineligible:treatment:bad-alt:label_source_missing" in error
               for error in errors)


def test_product_outside_area_or_role_is_rejected():
    rec = valid_recommendation()
    rec.selected_products["moisturizer"].intended_areas = ["neck"]
    errors = validate_recommendation(rec, profile())
    assert any("intended_area_not_face" in error for error in errors)


def test_unverified_treatment_strength_or_exposure_is_rejected():
    rec = valid_recommendation()
    rec.selected_products["treatment"].drug_actives = []
    rec.selected_products["treatment"].exposure = "mask"
    errors = validate_recommendation(rec, profile())
    assert any("therapy_active_not_directly_verified" in error for error in errors)
    assert any("non_daily_format_for_role" in error for error in errors)


def test_duplicate_carried_active_and_profile_contraindication_are_rejected():
    rec = valid_recommendation()
    rec.selected_products["moisturizer"].actives = ["azelaic_acid"]
    errors = validate_recommendation(rec, profile(current_actives=["azelaic_acid"]))
    assert any("duplicates_current_active:azelaic_acid" in error for error in errors)


def test_explanation_must_match_delivered_active_and_strength():
    rec = valid_recommendation()
    treatment = next(item for item in rec.explanation if item["role"] == "treatment")
    treatment["delivered_active"] = "benzoyl_peroxide"
    treatment["strength"] = "20%"
    errors = validate_recommendation(rec, profile())
    assert "explanation_active_not_delivered:benzoyl_peroxide" in errors
    assert "explanation_strength_not_delivered:20%" in errors


def test_instruction_requires_source_and_cadence():
    rec = valid_recommendation()
    rec.selected_regimen["pm"][0] = RoutineInstruction(
        rec.selected_regimen["pm"][0].role, "pm", "unknown", None, None
    )
    errors = validate_recommendation(rec, profile())
    assert any(error.startswith("instruction_source_missing") for error in errors)
    assert any(error.startswith("instruction_cadence_unknown") for error in errors)


def test_same_role_is_not_repeated_and_sunscreen_is_am_only():
    rec = valid_recommendation()
    rec.selected_regimen["pm"].append(
        RoutineInstruction("sunscreen", "pm", "per_label", None, "synthetic://spf")
    )
    rec.selected_regimen["am"].append(rec.selected_regimen["am"][0])
    errors = validate_recommendation(rec, profile())
    assert "sunscreen_scheduled_pm" in errors
    assert any(error.startswith("role_repeated_in_slot") for error in errors)
