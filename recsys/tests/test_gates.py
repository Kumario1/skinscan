from pathlib import Path

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.gates import (
    SOFT_REASON_PREFIXES,
    _reason_is_soft,
    apply_profile_gates,
    duplicate_active_reasons,
    profile_gate_reasons,
)
from recsys.knowledge import load_knowledge

K = load_knowledge(Path(__file__).parents[1] / "data" / "knowledge")


def product(pid="p1", category="treatment", actives=(), spf=None, price=None,
            inci=("Water",), **verified):
    role = "sunscreen" if category == "spf" else category
    product_format = verified.pop(
        "format",
        "cleanser" if category == "cleanser"
        else "sunscreen" if category == "spf"
        else "cream",
    )
    defaults = {
        "intended_areas": ("face",),
        "routine_roles": (role,),
        "exposure": "rinse_off" if category == "cleanser" else "leave_on",
        "cadence": "am_pm" if category != "treatment" else "per_label",
        "cadence_source": "https://example.test/label",
        "label_source": "https://example.test/label",
        "label_verified_at": "2026-07-14",
        "contraindications_verified": True,
        "drug_actives": tuple(
            {"name": active, "strength": "verified", "source": "https://example.test/label"}
            for active in actives
        ) if category == "treatment" else (),
    }
    defaults.update(verified)
    return CatalogProduct(
        product_id=pid, name=pid, brand="b", category=category,
        price_usd=price, size=None, format=product_format, spf=spf,
        spf_source="name_parse" if spf else None,
        inci=tuple(inci), inci_sha256="", actives=tuple(actives),
        **defaults,
    )


def test_soft_reason_prefixes_membership_is_pinned_exactly():
    """D-029 permits no evidence gap to become a ranking penalty."""
    assert SOFT_REASON_PREFIXES == frozenset()


def test_ingredient_and_profile_safety_reasons_are_never_classified_soft():
    """_reason_is_soft splits on ':' so a reason carrying a payload
    ("profile_allergy:fragrance") is classified by its prefix alone. Every
    ingredient/profile/price reason must fail that check -- if one ever passes,
    it stops vetoing in hybrid and the product reaches the user.
    """
    for reason in (
        "retinoid_pregnancy_status_excluded",
        "profile_allergy:fragrance",
        "duplicates_current_active:retinol",
        "product_contraindication:warfarin",
        "treatment_active_in_support_role:retinol",
        "treatment_format_not_daily_leave_on:peel",
        "cadence_not_daily",
        "spf_below_30_or_unknown",
        "spf_not_broad_spectrum",
        "price_above_profile_cap",
    ):
        assert _reason_is_soft(reason) is False, reason


def test_verification_quality_gaps_are_hard():
    for reason in (
        "intended_area_not_verified:face",
        "role_not_verified:treatment",
        "exposure_not_verified:leave_on",
        "cadence_unverified",
        "treatment_active_unverified",
        "contraindications_unverified",
        "format_unverified",
        "amount_source_unverified",
        "spf_broad_spectrum_unverified",
    ):
        assert _reason_is_soft(reason) is False, reason


def test_treatment_requires_explicit_contraindication_evidence():
    treatment = product(
        actives=["azelaic_acid"], contraindications_verified=False,
    )
    assert "contraindications_unverified" in profile_gate_reasons(
        treatment, "treatment", Profile(pregnancy_status="not_pregnant"), K
    )


def test_support_requires_explicit_contraindications_or_approved_daily_support():
    profile = Profile(pregnancy_status="not_pregnant")
    unknown = product(
        category="moisturizer", contraindications_verified=False,
        daily_support_verified=False,
    )
    approved = product(
        category="moisturizer", contraindications_verified=False,
        daily_support_verified=True,
    )

    assert "contraindications_unverified" in profile_gate_reasons(
        unknown, "moisturizer", profile, K
    )
    assert "contraindications_unverified" not in profile_gate_reasons(
        approved, "moisturizer", profile, K
    )


def test_format_and_amount_source_are_hard_requirements():
    profile = Profile(pregnancy_status="not_pregnant")
    assert "format_unverified" in profile_gate_reasons(
        product(format=None), "treatment", profile, K
    )
    assert "format_not_allowed_for_role:mist" in profile_gate_reasons(
        product(format="mist"), "treatment", profile, K
    )
    assert "amount_source_unverified" in profile_gate_reasons(
        product(amount="thin_layer", amount_source=None), "treatment", profile, K
    )


def test_an_unclassified_reason_defaults_to_hard():
    """A reason added to profile_gate_reasons without touching the soft table
    must veto in both modes. The default has to be the safe one -- that is what
    makes forgetting to classify a new reason harmless.
    """
    assert _reason_is_soft("some_reason_nobody_has_classified_yet") is False
    assert _reason_is_soft("some_reason_nobody_has_classified_yet:payload") is False


def test_pregnancy_unknown_vetoes_retinoids():
    retinol = product(actives=["retinol"])
    for status in ("pregnant", "trying", "nursing", "unknown"):
        reasons = profile_gate_reasons(retinol, "treatment", Profile(pregnancy_status=status), K)
        assert "retinoid_pregnancy_status_excluded" in reasons, status
    assert profile_gate_reasons(retinol, "treatment", Profile(pregnancy_status="not_pregnant"), K) == []


def test_allergy_and_current_active_vetoes():
    niacinamide = product(actives=["niacinamide"])
    profile = Profile(pregnancy_status="not_pregnant", allergies=("niacinamide",))
    assert "profile_allergy:niacinamide" in profile_gate_reasons(niacinamide, "serum", profile, K)
    profile = Profile(pregnancy_status="not_pregnant", current_actives=("niacinamide",))
    assert "duplicates_current_active:niacinamide" in profile_gate_reasons(niacinamide, "serum", profile, K)


def test_allergy_vetoes_non_active_ingredient_via_full_inci():
    # Fragrance is never a parsed active, so only a full-INCI scan can catch it.
    fragranced = product(
        category="moisturizer", actives=["glycerin"],
        inci=["Water", "Glycerin", "Parfum (Fragrance)", "Limonene"],
    )
    profile = Profile(pregnancy_status="not_pregnant", allergies=("fragrance",))
    assert "profile_allergy:fragrance" in profile_gate_reasons(fragranced, "moisturizer", profile, K)
    # A product with no fragrance is not vetoed on this allergen.
    clean = product(category="moisturizer", actives=["glycerin"], inci=["Water", "Glycerin"])
    assert "profile_allergy:fragrance" not in profile_gate_reasons(clean, "moisturizer", profile, K)


def test_allergy_free_text_resolves_onto_canonical_active():
    # "salicylic acid"/"BHA" must map onto the canonical salicylic_acid token
    # even when the product carries only the parsed active, not raw INCI.
    bha_product = product(actives=["salicylic_acid"])
    for declared in ("salicylic acid", "BHA"):
        profile = Profile(pregnancy_status="not_pregnant", allergies=(declared,))
        reasons = profile_gate_reasons(bha_product, "treatment", profile, K)
        assert any(r.startswith("profile_allergy:") for r in reasons), declared


def test_pregnancy_vetoes_cosmetic_retinoid_ester_from_inci():
    # Retinyl Palmitate never resolves to a canonical retinoid active; only an
    # INCI marker scan catches it. Actives here are deliberately non-retinoid.
    ester = product(
        actives=["salicylic_acid"],
        inci=["Salicylic Acid", "Retinyl Palmitate", "Aloe Vera Leaf Extract"],
    )
    for status in ("pregnant", "trying", "nursing", "unknown"):
        reasons = profile_gate_reasons(ester, "treatment", Profile(pregnancy_status=status), K)
        assert "retinoid_pregnancy_status_excluded" in reasons, status
    # Not excluded for a non-pregnant profile.
    assert "retinoid_pregnancy_status_excluded" not in profile_gate_reasons(
        ester, "treatment", Profile(pregnancy_status="not_pregnant"), K
    )


def test_unparseable_ingredients_veto_hard_even_in_hybrid():
    # Shape of a real catalog row: "+Retinol Vitamin C Moisturizer" that carries
    # no INCI and no parsed actives. Every ingredient gate reads one of those two
    # fields, so the pregnancy exclusion has nothing to match and passes
    # vacuously. Hybrid relaxes only verification-quality reasons, so the veto
    # has to hold there -- ranking is not a safety mechanism.
    profile = Profile(pregnancy_status="pregnant")
    blank = product(pid="P448932", category="moisturizer", actives=(), inci=())
    reasons = profile_gate_reasons(blank, "moisturizer", profile, K)
    assert "ingredients_unknown" in reasons
    assert "retinoid_pregnancy_status_excluded" not in reasons
    kept, vetoes, _flags = apply_profile_gates(
        {"moisturizer": [blank]}, profile, K, strict=False
    )
    assert kept["moisturizer"] == []
    assert "ingredients_unknown" in {v.reason for v in vetoes}


def test_known_ingredients_without_parsed_actives_are_not_unknown():
    # A plain moisturizer parses to no actives; its ingredients are still known.
    profile = Profile(pregnancy_status="not_pregnant")
    plain = product(category="moisturizer", actives=(),
                    inci=["Water", "Glycerin", "Cetearyl Alcohol"])
    assert "ingredients_unknown" not in profile_gate_reasons(
        plain, "moisturizer", profile, K
    )
    # A drug label publishes no INCI but names its actives -- also known.
    drug = product(actives=("azelaic_acid",), inci=())
    assert "ingredients_unknown" not in profile_gate_reasons(
        drug, "treatment", profile, K
    )


def test_spf_gate():
    weak = product(category="spf", spf=15)
    unknown = product(category="spf", spf=None)
    unverified = product(category="spf", spf=30, broad_spectrum=None)
    narrow = product(category="spf", spf=30, broad_spectrum=False)
    strong = product(category="spf", spf=30, broad_spectrum=True)
    profile = Profile(pregnancy_status="not_pregnant")
    assert "spf_below_30_or_unknown" in profile_gate_reasons(weak, "spf", profile, K)
    assert "spf_below_30_or_unknown" in profile_gate_reasons(unknown, "spf", profile, K)
    assert "spf_broad_spectrum_unverified" in profile_gate_reasons(
        unverified, "spf", profile, K
    )
    assert "spf_not_broad_spectrum" in profile_gate_reasons(narrow, "spf", profile, K)
    assert profile_gate_reasons(strong, "spf", profile, K) == []


def test_price_cap():
    pricey = product(price=50.0)
    unknown = product(price=None)
    profile = Profile(pregnancy_status="not_pregnant", max_price_usd=20.0)
    assert "price_above_profile_cap" in profile_gate_reasons(pricey, "treatment", profile, K)
    assert "price_unknown_for_profile_cap" in profile_gate_reasons(
        unknown, "treatment", profile, K
    )


def test_non_daily_products_are_vetoed():
    profile = Profile(pregnancy_status="not_pregnant")
    assert "cadence_not_daily" in profile_gate_reasons(
        product(cadence="weekly"), "treatment", profile, K
    )
    assert profile_gate_reasons(
        product(actives=("salicylic_acid",), cadence="per_label"),
        "treatment", profile, K,
    ) == []
    sensitive = Profile(
        pregnancy_status="not_pregnant", sensitivity_conditions=("sensitive",)
    )
    assert "product_contraindication:sensitive" in profile_gate_reasons(
        product(contraindications=("sensitive",)), "treatment", sensitive, K
    )


def test_a_verified_non_daily_cadence_vetoes_in_both_modes():
    """cadence_not_daily is emitted only from the elif -- cadence IS verified and
    positively says something a daily AM/PM routine cannot honour ("weekly").
    That is a checked fact, not the "we have not checked yet" gap the soft table
    exists for, so hybrid must not downgrade it: doing so puts a label-verified
    weekly exfoliant into a daily routine. Latent while every catalog row has a
    null cadence, reachable the moment the overlay sets one.
    """
    weekly = product("w1", "treatment", actives=["salicylic_acid"], cadence="weekly")
    profile = Profile(pregnancy_status="not_pregnant")
    for strict in (True, False):
        kept, vetoes, flags = apply_profile_gates(
            {"treatment": [weekly]}, profile, K, strict=strict
        )
        assert kept["treatment"] == [], strict
        assert [v.reason for v in vetoes] == ["cadence_not_daily"], strict
        assert flags == {}, strict


def test_an_unverified_cadence_vetoes_in_every_mode():
    unknown_cadence = product(
        "c1", "treatment", actives=["salicylic_acid"], cadence=None, cadence_source=None
    )
    kept, vetoes, flags = apply_profile_gates(
        {"treatment": [unknown_cadence]}, Profile(pregnancy_status="not_pregnant"), K,
        strict=False,
    )
    assert kept["treatment"] == []
    assert [v.reason for v in vetoes] == ["cadence_unverified"]
    assert flags == {}


def test_medication_and_pregnancy_contraindications_are_vetoed():
    profile = Profile(
        pregnancy_status="pregnant", current_medications=("warfarin",)
    )
    reasons = profile_gate_reasons(
        product(contraindications=("warfarin", "pregnant")),
        "treatment", profile, K,
    )
    assert "product_contraindication:warfarin" in reasons
    assert "product_contraindication:pregnant" in reasons


def test_hard_role_eligibility_vetoes_unverified_or_wrong_products():
    profile = Profile(pregnancy_status="not_pregnant")
    assert "role_not_verified:moisturizer" in profile_gate_reasons(
        product(category="moisturizer", routine_roles=("cleanser",)),
        "moisturizer", profile, K,
    )
    assert "intended_area_not_verified:face" in profile_gate_reasons(
        product(category="moisturizer", intended_areas=("body",)),
        "moisturizer", profile, K,
    )
    assert "treatment_active_in_support_role:retinol" in profile_gate_reasons(
        product(category="moisturizer", actives=("retinol",)),
        "moisturizer", profile, K,
    )


def test_unstated_area_is_not_a_non_face_claim():
    # A regulatory label states "cover the entire affected area" and never names
    # the face, so an unstated area must not veto: requiring an explicit "face"
    # is satisfiable only by inventing the fact. A stated face wins over a
    # co-stated body.
    profile = Profile(pregnancy_status="not_pregnant")
    for areas in ((), ("unknown",), ("face", "body")):
        assert "intended_area_not_verified:face" not in profile_gate_reasons(
            product(category="moisturizer", intended_areas=areas),
            "moisturizer", profile, K,
        )


def test_a_stated_area_outside_the_known_vocabulary_is_still_a_non_face_claim():
    """The non-face check must derive from the product's own areas rather than
    intersect an enumerated list, because an enumerated list is a copy of the
    area vocabulary that nothing keeps in sync: add "scalp" to the vocabulary,
    forget this list, and a scalp product becomes eligible for a facial routine.
    Any stated area that is not the face is a claim to somewhere that is not the
    face, whether or not this module has heard of it.
    """
    profile = Profile(pregnancy_status="not_pregnant")
    for areas in (("scalp",), ("hand",), ("scalp", "body"), ("underarm", "unknown")):
        assert "intended_area_not_verified:face" in profile_gate_reasons(
            product(category="moisturizer", intended_areas=areas), "moisturizer", profile, K
        ), areas


def test_contraindication_matching_folds_case_and_surrounding_whitespace():
    """Declared conditions and product contraindications are both free text off
    the evidence overlay -- there is no closed vocabulary to validate either
    against -- so exact matching fails this HARD gate open on a capital letter.
    """
    sensitive = Profile(pregnancy_status="not_pregnant", sensitivity_conditions=("Sensitive ",))
    assert "product_contraindication:sensitive" in profile_gate_reasons(
        product(contraindications=("sensitive",)), "treatment", sensitive, K
    )
    medicated = Profile(pregnancy_status="not_pregnant", current_medications=("Warfarin",))
    assert "product_contraindication:warfarin" in profile_gate_reasons(
        product(contraindications=("WARFARIN",)), "treatment", medicated, K
    )


def test_treatment_requires_verified_drug_active_and_safe_format():
    profile = Profile(pregnancy_status="not_pregnant")
    assert "treatment_active_unverified" in profile_gate_reasons(
        product(actives=("salicylic_acid",), drug_actives=()),
        "treatment", profile, K,
    )
    assert "treatment_format_not_daily_leave_on:peel" in profile_gate_reasons(
        product(actives=("salicylic_acid",), format="peel"),
        "treatment", profile, K,
    )


def test_apply_profile_gates_vetoes_hard_safety_reasons_in_hybrid_too():
    """Every other test in this file calls profile_gate_reasons directly and so
    bypasses the strict/hybrid fork, which is the branch that actually decides
    what reaches a user. Hybrid relaxes verification quality, never ingredient
    safety: a retinoid must not survive for a pregnant user in either mode.
    """
    retinol = product("r1", "treatment", actives=["retinol"])
    for strict in (True, False):
        kept, vetoes, flags = apply_profile_gates(
            {"treatment": [retinol]}, Profile(pregnancy_status="pregnant"), K, strict=strict
        )
        assert kept["treatment"] == [], strict
        assert [v.reason for v in vetoes] == ["retinoid_pregnancy_status_excluded"], strict
        assert flags == {}, strict


def test_apply_profile_gates_never_downgrades_missing_evidence():
    unverified = product("u1", "moisturizer", actives=["glycerin"], exposure=None)
    profile = Profile(pregnancy_status="not_pregnant")

    for strict in (True, False):
        kept, vetoes, flags = apply_profile_gates(
            {"moisturizer": [unverified]}, profile, K, strict=strict
        )
        assert kept["moisturizer"] == []
        assert [v.reason for v in vetoes] == ["exposure_not_verified:leave_on"]
        assert flags == {}


def test_a_hard_reason_vetoes_even_when_soft_reasons_accompany_it():
    """Hybrid splits one product's reasons into two lists. The hard one has to
    win: a product that is both unverified AND unsafe must not be kept because
    its soft reasons were filtered out of the blocking set first.
    """
    unsafe_and_unverified = product(
        "x1", "moisturizer", actives=["glycerin"], exposure=None,
        inci=["Water", "Glycerin", "Parfum (Fragrance)"],
    )
    profile = Profile(pregnancy_status="not_pregnant", allergies=("fragrance",))
    kept, vetoes, flags = apply_profile_gates(
        {"moisturizer": [unsafe_and_unverified]}, profile, K, strict=False
    )
    assert kept["moisturizer"] == []
    assert [v.reason for v in vetoes] == [
        "exposure_not_verified:leave_on", "profile_allergy:fragrance"
    ]
    assert flags == {}


def test_multiple_evidence_gaps_all_veto():
    partial = product(
        "p9", "moisturizer", actives=["glycerin"],
        exposure=None, cadence=None, cadence_source=None,
    )
    kept, vetoes, flags = apply_profile_gates(
        {"moisturizer": [partial]}, Profile(pregnancy_status="not_pregnant"), K, strict=False
    )
    assert kept["moisturizer"] == []
    assert [v.reason for v in vetoes] == [
        "exposure_not_verified:leave_on", "cadence_unverified",
    ]
    assert flags == {}


def test_duplicate_treatment_actives_only():
    salicylic_serum = product("p2", "serum", ["salicylic_acid"])
    selected_treatment = product("p3", "treatment", ["salicylic_acid"])
    assert duplicate_active_reasons(salicylic_serum, [selected_treatment], K) == [
        "duplicates_selected_active:salicylic_acid"
    ]
    # benign support ingredients repeat freely
    glycerin_moisturizer = product("p4", "moisturizer", ["glycerin"])
    glycerin_cleanser = product("p5", "cleanser", ["glycerin"])
    assert duplicate_active_reasons(glycerin_moisturizer, [glycerin_cleanser], K) == []
