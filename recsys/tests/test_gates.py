from pathlib import Path

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.gates import duplicate_active_reasons, profile_gate_reasons
from recsys.knowledge import load_knowledge

K = load_knowledge(Path(__file__).parents[1] / "data" / "knowledge")


def product(pid="p1", category="treatment", actives=(), spf=None, price=None, **verified):
    role = "sunscreen" if category == "spf" else category
    product_format = verified.pop("format", None)
    defaults = {
        "intended_areas": ("face",),
        "routine_roles": (role,),
        "exposure": "rinse_off" if category == "cleanser" else "leave_on",
        "cadence": "am_pm" if category != "treatment" else "per_label",
        "cadence_source": "https://example.test/label",
        "label_source": "https://example.test/label",
        "label_verified_at": "2026-07-14",
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
        inci=(), inci_sha256="", actives=tuple(actives),
        **defaults,
    )


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
    profile = Profile(pregnancy_status="not_pregnant", max_price_usd=20.0)
    assert "price_above_profile_cap" in profile_gate_reasons(pricey, "treatment", profile, K)


def test_verified_discontinued_and_non_daily_products_are_vetoed():
    profile = Profile(pregnancy_status="not_pregnant")
    assert "product_discontinued" in profile_gate_reasons(
        product(discontinued=True), "treatment", profile, K
    )
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
