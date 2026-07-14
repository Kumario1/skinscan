import json

import pytest

from src.recommendation.schema import (
    CareDecision,
    DecisionEvidence,
    Product,
    RoutineInstruction,
    TherapyOption,
    TherapyPlan,
    UserProfile,
    VerifiedActive,
)


def _complete_product() -> Product:
    return Product(
        "aza-10", "Azelaic 10%", "Example", "treatment",
        actives=["azelaic_acid"],
        intended_areas=["face"], routine_roles=["treatment"], format="gel",
        exposure="leave_on",
        drug_actives=[VerifiedActive("azelaic_acid", "10%", "https://label.test/aza")],
        otc_drug=False, label_source="https://label.test/aza",
        label_verified_at="2026-07-13", comedogenic_claim="not_claimed",
        evidence_roles=["acne_treatment_alternative"], evidence_grade="synthetic_test",
        cadence="per_label", cadence_source="https://label.test/aza",
        catalog_schema_version="2",
    )


def test_complete_product_round_trips_nested_verified_actives():
    original = _complete_product()
    loaded = Product.from_dict(original.to_dict())
    assert loaded == original
    assert isinstance(loaded.drug_actives[0], VerifiedActive)


def test_legacy_product_stays_visible_but_gains_no_eligible_roles():
    product = Product.from_dict({
        "product_id": "old", "name": "Old Serum", "brand": "B",
        "category": "serum", "actives": ["azelaic_acid"],
    })
    assert product.is_legacy
    assert product.routine_roles == []
    assert product.intended_areas == []
    assert product.exposure == "unknown"


def test_profile_unknown_is_not_collapsed_to_not_pregnant():
    assert UserProfile().to_dict()["pregnancy_status"] == "unknown"
    assert UserProfile.from_dict({"pregnant_or_nursing": False}).pregnancy_status == "not_pregnant"
    assert UserProfile.from_dict({"pregnant_or_nursing": True}).pregnancy_status == "pregnant"


@pytest.mark.parametrize("payload,field", [
    ({"age_years": True}, "age_years"),
    ({"age_years": 20.5}, "age_years"),
    ({"acne_duration_weeks": 3.5}, "acne_duration_weeks"),
    ({"painful_or_deep_lesions": "no"}, "painful_or_deep_lesions"),
    ({"prior_scarring": 1}, "prior_scarring"),
    ({"max_price_usd": True}, "max_price_usd"),
    ({"pregnant_or_nursing": "false"}, "pregnant_or_nursing"),
])
def test_profile_rejects_wrong_scalar_types(payload, field):
    with pytest.raises(ValueError, match=field):
        UserProfile.from_dict(payload)


def test_closed_vocabulary_objects_reject_invalid_values_with_context():
    with pytest.raises(ValueError, match="triage_level"):
        CareDecision("maybe", [], "defer", [], None, False)
    with pytest.raises(ValueError, match="probability"):
        DecisionEvidence("x", 0.7, "high", "raw", False, [])
    with pytest.raises(ValueError, match="exposure"):
        TherapyOption("x", "known", "rub_on", "per_label", "treatment")
    with pytest.raises(ValueError, match="slot"):
        RoutineInstruction("cleanser", "noon", "daily", None, "label")


def test_explicit_serializers_are_stable():
    product = _complete_product()
    option = TherapyOption(
        "azelaic_acid", "10%", "leave_on", "per_label", "treatment",
        cadence_source="https://label.test/aza",
    )
    plan = TherapyPlan(12, 12, option, [], ["cleanser", "moisturizer", "sunscreen"], [], "test")
    payload = {
        "product": product.to_dict(),
        "plan": plan.to_dict(),
    }
    first = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    second = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert first == second
