from pathlib import Path

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.gates import duplicate_active_reasons, profile_gate_reasons
from recsys.knowledge import load_knowledge

K = load_knowledge(Path(__file__).parents[1] / "data" / "knowledge")


def product(pid="p1", category="treatment", actives=(), spf=None, price=None, **verified):
    return CatalogProduct(
        product_id=pid, name=pid, brand="b", category=category,
        price_usd=price, size=None, format=None, spf=spf,
        spf_source="name_parse" if spf else None,
        inci=(), inci_sha256="", actives=tuple(actives),
        **verified,
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
    strong = product(category="spf", spf=30)
    profile = Profile(pregnancy_status="not_pregnant")
    assert "spf_below_30_or_unknown" in profile_gate_reasons(weak, "spf", profile, K)
    assert "spf_below_30_or_unknown" in profile_gate_reasons(unknown, "spf", profile, K)
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
    assert profile_gate_reasons(product(cadence="per_label"), "treatment", profile, K) == []
    sensitive = Profile(
        pregnancy_status="not_pregnant", sensitivity_conditions=("sensitive",)
    )
    assert "product_contraindication:sensitive" in profile_gate_reasons(
        product(contraindications=("sensitive",)), "treatment", sensitive, K
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
