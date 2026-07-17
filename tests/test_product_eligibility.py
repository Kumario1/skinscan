from dataclasses import replace

from src.recommendation.eligibility import check_eligibility
from src.recommendation.schema import Product, TherapyOption, UserProfile, VerifiedActive


def profile(**overrides):
    values = {
        "skin_type": "oily", "tone_bucket": "medium", "tone_source": "self_report",
        "age_years": 25, "pregnancy_status": "not_pregnant",
    }
    values.update(overrides)
    return UserProfile(**values)


def treatment(active="azelaic_acid", strength="10%", **overrides):
    values = {
        "product_id": f"{active}-product", "name": active, "brand": "Example",
        "category": "treatment", "actives": [active], "intended_areas": ["face"],
        "routine_roles": ["treatment"], "format": "gel", "exposure": "leave_on",
        "drug_actives": [VerifiedActive(active, strength, "https://label.test")],
        "otc_drug": True,
        "label_source": "https://label.test", "label_verified_at": "2026-07-13",
        "comedogenic_claim": "not_claimed", "evidence_grade": "synthetic_test",
        "cadence": "per_label", "cadence_source": "https://label.test",
        "catalog_schema_version": "2",
    }
    values.update(overrides)
    return Product(**values)


def option(active="azelaic_acid", strength="10%"):
    return TherapyOption(active, strength, "leave_on", "per_label", "treatment",
                         cadence_source="https://label.test")


def support(role, **overrides):
    category = "spf" if role == "sunscreen" else role
    values = {
        "product_id": f"support-{role}", "name": role, "brand": "Example",
        "category": category, "intended_areas": ["face"], "routine_roles": [role],
        "format": "sunscreen" if role == "sunscreen" else (
            "cleanser" if role == "cleanser" else "cream"
        ),
        "exposure": "rinse_off" if role == "cleanser" else "leave_on",
        "comedogenic_claim": (
            "claimed_noncomedogenic" if role in {"moisturizer", "sunscreen"}
            else "not_claimed"
        ),
        "catalog_schema_version": "2",
        "cadence": "per_label", "cadence_source": "https://support.test",
    }
    if role == "sunscreen":
        values.update(broad_spectrum=True, spf=30, label_source="https://spf.test",
                      label_verified_at="2026-07-13")
    values.update(overrides)
    return Product(**values)


def test_direct_verified_azelaic_leave_on_treatment_passes():
    assert check_eligibility(treatment(), "treatment", option(), profile()).eligible


def test_non_otc_treatment_is_not_rejected_for_otc_status():
    # D-033: verified actives + label suffice; OTC status no longer gates
    for value in (False, None):
        result = check_eligibility(
            treatment(otc_drug=value), "treatment", option(), profile()
        )
        assert result.eligible


def test_treatment_cadence_must_match_concrete_policy_direction():
    product = treatment(cadence="twice_daily")
    planned = TherapyOption(
        "azelaic_acid", "10%", "leave_on", "once_daily", "treatment",
        cadence_source="synthetic://policy",
    )
    result = check_eligibility(product, "treatment", planned, profile())
    assert "therapy_cadence_mismatch" in result.reasons


def test_bp_carried_through_ceramides_is_vetoed_from_support_role():
    product = support(
        "moisturizer", actives=["ceramides", "benzoyl_peroxide"],
        drug_actives=[VerifiedActive("benzoyl_peroxide", "2.5%", "https://bp.test")],
    )
    result = check_eligibility(product, "moisturizer", None, profile())
    assert "carried_treatment_active_in_support_role" in result.reasons


def test_retinoid_carrying_support_product_is_rejected_when_pregnancy_unknown():
    product = support(
        "moisturizer", actives=["niacinamide", "retinol"],
        drug_actives=[VerifiedActive("retinol", "0.1%", "https://retinol.test")],
    )
    result = check_eligibility(product, "moisturizer", None,
                               profile(pregnancy_status="unknown"))
    assert "retinoid_pregnancy_status_excluded" in result.reasons


def test_trace_salicylic_rinse_off_cleanser_cannot_fill_leave_on_treatment():
    cleanser = support(
        "cleanser", actives=["salicylic_acid"], routine_roles=["cleanser", "treatment"],
        drug_actives=[VerifiedActive("salicylic_acid", None, None)],
    )
    result = check_eligibility(
        cleanser, "treatment", option("salicylic_acid", "2%"), profile()
    )
    assert "format_not_allowed_for_role" in result.reasons
    assert "therapy_exposure_mismatch" in result.reasons


def test_neck_serum_fails_facial_moisturizer():
    product = support("moisturizer", intended_areas=["neck"])
    assert "intended_area_not_face" in check_eligibility(
        product, "moisturizer", None, profile()
    ).reasons


def test_unstated_area_is_not_a_non_face_claim():
    # An OTC drug label says "cover the entire affected area" and never names
    # the face, so an unstated area must not veto: requiring an explicit "face"
    # is satisfiable only by inventing the fact. A stated face wins over a
    # co-stated body.
    for areas in ([], ["unknown"], ["face", "body"]):
        assert "intended_area_not_face" not in check_eligibility(
            support("moisturizer", intended_areas=areas), "moisturizer", None, profile()
        ).reasons


def test_mask_scrub_and_peel_fail_daily_leave_on_treatment():
    for exposure in ("mask", "scrub", "peel"):
        product = treatment(exposure=exposure, format="gel")
        result = check_eligibility(product, "treatment", option(), profile())
        assert "non_daily_format_for_role" in result.reasons


def test_unverified_strength_and_source_fail_treatment():
    product = treatment(
        drug_actives=[VerifiedActive("azelaic_acid", None, None)], label_source=None
    )
    result = check_eligibility(product, "treatment", option(), profile())
    assert "therapy_strength_not_verified" in result.reasons
    assert "drug_active_source_missing" in result.reasons
    assert "label_source_missing" in result.reasons


def test_unverified_sunscreen_claim_fails_and_verified_spf30_passes():
    unverified = support("sunscreen", broad_spectrum=None, spf=50)
    assert "broad_spectrum_not_verified" in check_eligibility(
        unverified, "sunscreen", None, profile()
    ).reasons
    assert check_eligibility(support("sunscreen"), "sunscreen", None, profile()).eligible


def test_duplicate_or_conflicting_carried_actives_across_roles_fail():
    selected = {"treatment": treatment("benzoyl_peroxide", "2.5%")}
    duplicate = support("moisturizer", actives=["benzoyl_peroxide"],
                        drug_actives=[])
    result = check_eligibility(duplicate, "moisturizer", None, profile(), selected)
    assert "duplicates_selected_active:benzoyl_peroxide" in result.reasons


def test_support_ingredient_cannot_admit_product_to_treatment():
    serum = treatment(
        "niacinamide", None,
        drug_actives=[], actives=["niacinamide"],
    )
    result = check_eligibility(serum, "treatment", option(), profile())
    assert "therapy_active_not_directly_verified" in result.reasons
