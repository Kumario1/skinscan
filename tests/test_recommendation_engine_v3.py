from pathlib import Path

from src.recommendation.decision import TriagePolicy, conservative_unreviewed_policy
from src.recommendation.engine import recommend
from src.recommendation.schema import Concern, ConcernEvidence, ConcernReport, Product, UserProfile
from src.recommendation.therapy import load_therapy_policy

from test_regimen_composer import product


POLICY = Path(__file__).parent / "fixtures" / "therapy_policy_synthetic.json"


def profile():
    return UserProfile(skin_type="oily", age_years=25, pregnancy_status="not_pregnant")


def triage():
    return TriagePolicy("synthetic", "1", True)


def catalog():
    return [
        product("cleanser", "cleanser"),
        product("aza", "treatment", "azelaic_acid"),
        product("aza-alt", "treatment", "azelaic_acid"),
        product("moist", "moisturizer"),
        product("spf", "sunscreen"),
    ]


def active_report(severity=2):
    return ConcernReport("active", concerns=[
        Concern("acne_inflammatory", "forehead", severity, 0.9, lesion_count=30),
    ])


def run(report, products=None, **kwargs):
    return recommend(
        report, products if products is not None else catalog(), profile(),
        triage_policy=kwargs.pop("triage_policy", triage()),
        therapy_policy=load_therapy_policy(POLICY),
        collect_eligibility_details=True,
        **kwargs,
    )


def test_severity_four_non_nodule_retains_active_treatment():
    result = run(active_report(4))
    assert result.decision.triage_level == "routine_plus_review"
    assert result.decision.therapy_disposition == "active_treatment"
    assert result.selected_products["treatment"].product_id == "aza"


def test_uncalibrated_nodule_abstains_and_supports_only():
    report = ConcernReport("nodule", concerns=[
        Concern("acne_cystic", "chin_jaw", 4, 0.95,
                evidence=ConcernEvidence({"nodule": 1}, 0.95, 1)),
    ])
    result = run(report, triage_policy=conservative_unreviewed_policy())
    assert (result.decision.triage_level, result.decision.therapy_disposition) == (
        "abstain", "supportive_only"
    )
    assert "treatment" not in result.selected_products


def test_scarring_adds_review_while_therapy_remains_active():
    report = active_report()
    report.concerns.append(Concern("acne_scarring", "left_cheek", 3, 0.9))
    result = run(report)
    assert result.decision.triage_level == "routine_plus_review"
    assert result.decision.therapy_disposition == "active_treatment"


def test_ranker_and_scorer_never_see_hard_rejected_product():
    rejected = Product(
        "mask", "Azelaic Mask", "Example", "treatment", actives=["azelaic_acid"],
        intended_areas=["face"], routine_roles=["treatment"], format="gel",
        exposure="mask", catalog_schema_version="2",
        cadence="daily", cadence_source="synthetic://mask",
    )
    seen = []

    class Spy:
        def score(self, item, _profile):
            seen.append(item.product_id)
            return 100 if item.product_id == "mask" else 0

    result = run(active_report(), catalog() + [rejected], concern_scorer=Spy())
    assert "mask" not in seen
    assert result.selected_products["treatment"].product_id == "aza"


def test_scorer_never_sees_product_rejected_by_selected_role_context():
    products = catalog()
    products[0].actives = ["glycerin"]
    duplicate_moisturizer = product("duplicate-moist", "moisturizer")
    duplicate_moisturizer.actives = ["glycerin"]
    products.append(duplicate_moisturizer)
    seen = []

    class Spy:
        def score(self, item, _profile):
            seen.append(item.product_id)
            return 100

    result = run(active_report(), products, concern_scorer=Spy())
    assert "duplicate-moist" not in seen
    assert "duplicates_selected_active:glycerin" in result.eligibility_rejections[
        "moisturizer:duplicate-moist"
    ]


def test_support_choice_cannot_suppress_an_available_primary_treatment():
    shared_cleanser = product("a-cleanser", "cleanser")
    shared_cleanser.actives = ["niacinamide"]
    plain_cleanser = product("b-cleanser", "cleanser")
    aza = product("aza-niacinamide", "treatment", "azelaic_acid")
    aza.actives.append("niacinamide")
    products = [shared_cleanser, plain_cleanser, aza,
                product("moist", "moisturizer"), product("spf", "sunscreen")]
    result = run(active_report(), products)
    assert result.selected_products["treatment"].product_id == "aza-niacinamide"
    assert result.selected_products["cleanser"].product_id == "b-cleanser"
    assert result.decision.therapy_disposition == "active_treatment"


def test_carried_bp_cannot_reenter_through_support_role():
    bp_moist = product("bp-moist", "moisturizer")
    bp_moist.actives = ["ceramides", "benzoyl_peroxide"]
    result = run(active_report(), catalog() + [bp_moist])
    assert result.selected_products["moisturizer"].product_id == "moist"
    assert "carried_treatment_active_in_support_role" in result.eligibility_rejections[
        "moisturizer:bp-moist"
    ]


def test_one_selected_product_per_role_and_alternatives_are_disjoint():
    result = run(active_report())
    assert set(result.selected_products) == {"cleanser", "treatment", "moisturizer", "sunscreen"}
    selected_ids = {item.product_id for item in result.selected_products.values()}
    alternative_ids = {item.product_id for items in result.alternatives.values() for item in items}
    assert selected_ids.isdisjoint(alternative_ids)
    assert result.validation_errors == []


def test_missing_verified_treatment_is_an_explicit_missing_role():
    products = [item for item in catalog() if item.category != "treatment"]
    result = run(active_report(), products)
    assert "treatment" not in result.selected_products
    assert result.decision.therapy_disposition == "defer"
    assert result.eligibility_rejections["role:treatment"] == ["no_eligible_product"]


def test_maintenance_uses_same_hard_spf_and_role_checks():
    bad_spf = product("bad-spf", "sunscreen")
    bad_spf.broad_spectrum = None
    result = run(ConcernReport("clear", clear_skin=True), [
        product("cleanser", "cleanser"), product("moist", "moisturizer"), bad_spf,
    ])
    assert "sunscreen" not in result.selected_products
    assert "broad_spectrum_not_verified" in result.eligibility_rejections[
        "sunscreen:bad-spf"
    ]
