"""Unit tests for the Stage 3 rules engine (src/recommendation/engine.py).

RULES.md calls the rules layer "where all correctness lives" and DECISIONS.md
D-007 says it was to be grown test-first — these tests pin the current
behavior so future work on the engine has a safety net. Pure Python, no ML;
runs in milliseconds. Standalone via __main__ (pytest not required) but named
test_* so `pytest tests/` also works.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendation.engine import recommend
from src.recommendation.schema import (
    Concern, ConcernReport, Product, UserProfile, CATEGORIES,
)


def make_catalog():
    return [
        Product("p1", "SA Cleanser", "b", "cleanser", actives=["salicylic_acid"]),
        Product("p2", "BP Gel", "b", "treatment", actives=["benzoyl_peroxide"]),
        Product("p3", "Niacinamide Serum", "b", "serum", actives=["niacinamide"]),
        Product("p4", "Ceramide Cream", "b", "moisturizer", actives=["ceramides"]),
        Product("p5", "Sunscreen", "b", "spf", actives=[]),
        Product("p6", "Coconut Balm", "b", "moisturizer",
                actives=["ceramides"], comedogenic_flags=["coconut_oil"]),
        Product("p7", "Vit C Serum", "b", "serum", actives=["vitamin_c"]),
        Product("p8", "Adapalene Gel", "b", "treatment", actives=["adapalene"]),
        Product("p10", "Ceramide Lotion", "b", "moisturizer", actives=["ceramides"]),
    ]


class StubRanker:
    """Duck-typed ranker (D-005): only reorders, never adds/removes/flags."""
    def __init__(self, scores):
        self.scores = scores

    def score(self, product, profile):
        return self.scores.get(product.product_id, 0.0)


def _product_ids(products):
    return [p.product_id for p in products]


def test_soothe_routine_excludes_strong_active_products():
    """RULES.md §4 — cystic: 'do NOT recommend aggressive actives'. A product
    must not sneak into the soothe routine via a gentle active (e.g. an SA serum
    that also lists hyaluronic_acid)."""
    catalog = make_catalog() + [
        Product("p11", "SA + HA Serum", "b", "serum",
                actives=["salicylic_acid", "hyaluronic_acid"]),
        Product("p12", "Plain HA Serum", "b", "serum", actives=["hyaluronic_acid"]),
        Product("p13", "Glycolic Ceramide Cream", "b", "moisturizer",
                actives=["glycolic_acid", "ceramides"]),
        Product("p15", "PHA Resurfacing Toner", "b", "cleanser",
                actives=["gluconolactone", "glycerin", "hyaluronic_acid"]),
        # INCI-clean (citrus-juice "AHA") but named as an exfoliant — the
        # soothe path vetoes by name too, since the vocabulary can't see it.
        Product("p16", "AHA Liquid Exfoliating Treatment", "b", "treatment",
                actives=["niacinamide", "hyaluronic_acid"]),
    ]
    report = ConcernReport("img", concerns=[Concern("acne_cystic", "chin_jaw", 2, 0.9)])
    rec = recommend(report, catalog)
    everything = [pid for prods in rec.routine.values() for pid in _product_ids(prods)]
    assert "p11" not in everything, everything
    assert "p13" not in everything, everything
    assert "p15" not in everything, everything
    assert "p16" not in everything, everything
    assert "p12" in everything, everything


def test_maintenance_routine_excludes_strong_active_products():
    """RULES.md §4 severity 0 — maintenance is 'gentle cleanser, moisturizer,
    SPF'; strong-active products must not match via a bundled gentle active."""
    catalog = make_catalog() + [
        Product("p11", "SA + HA Serum", "b", "serum",
                actives=["salicylic_acid", "hyaluronic_acid"]),
    ]
    rec = recommend(ConcernReport("img", clear_skin=True), catalog)
    everything = [pid for prods in rec.routine.values() for pid in _product_ids(prods)]
    assert "p11" not in everything, everything


def test_multi_active_product_respects_all_slot_pins():
    """RULES.md §2a — a product carrying a PM-pinned retinoid must not land in
    AM just because another of its target actives is AM-eligible."""
    catalog = make_catalog() + [
        Product("p14", "Adapalene + Niacinamide Gel", "b", "treatment",
                actives=["adapalene", "niacinamide"]),
    ]
    report = ConcernReport("img", concerns=[
        Concern("acne_comedonal", "forehead", 2, 0.9),
        Concern("acne_inflammatory", "left_cheek", 2, 0.9),
    ])
    rec = recommend(report, catalog)
    assert "p14" not in _product_ids(rec.routines["AM"]["treatment"])
    assert "p14" in _product_ids(rec.routines["PM"]["treatment"])


def test_cystic_path_keeps_low_confidence_verify_flags():
    """RULES.md §5 — loud uncertainty applies on the escalation path too."""
    report = ConcernReport("img", concerns=[
        Concern("acne_cystic", "chin_jaw", 1, 0.3),
        Concern("acne_comedonal", "left_cheek", 2, 0.9),
    ])
    rec = recommend(report, make_catalog())
    assert "see a dermatologist" in rec.flags
    assert any("acne_cystic" in f and "verify" in f for f in rec.flags), rec.flags


def test_clear_skin_maintenance():
    report = ConcernReport("img", clear_skin=True)
    rec = recommend(report, make_catalog())
    assert rec.flags == ["maintenance routine"], rec.flags
    assert "ceramides" in rec.target_actives
    assert "hyaluronic_acid" in rec.target_actives
    # always_spf -> the spf product is present
    assert "p5" in _product_ids(rec.routine["spf"])


def test_cystic_escalates():
    report = ConcernReport("img", concerns=[Concern("acne_cystic", "chin_jaw", 2, 0.9)])
    rec = recommend(report, make_catalog())
    assert "see a dermatologist" in rec.flags
    assert "benzoyl_peroxide" not in rec.target_actives
    assert "centella" in rec.target_actives


def test_severity_4_escalates():
    report = ConcernReport("img", concerns=[Concern("acne_inflammatory", "forehead", 4, 0.9)])
    rec = recommend(report, make_catalog())
    assert "see a dermatologist" in rec.flags


def test_comedonal_gets_first_line_actives_no_spf():
    report = ConcernReport("img", concerns=[Concern("acne_comedonal", "nose", 2, 0.9)])
    rec = recommend(report, make_catalog())
    assert "salicylic_acid" in rec.target_actives
    # no hyperpigmentation -> SPF not forced
    assert rec.routine["spf"] == []


def test_hyperpigmentation_forces_spf_and_conflict_resolution():
    report = ConcernReport("img", concerns=[
        Concern("acne_inflammatory", "left_cheek", 2, 0.9),
        Concern("hyperpigmentation", "left_cheek", 2, 0.9),
    ])
    rec = recommend(report, make_catalog())
    # hyperpigmentation forces SPF into the routine
    assert "p5" in _product_ids(rec.routine["spf"])
    # BP + vitamin_c are INCOMPATIBLE but now COEXIST across slots (Engine v2):
    # both prefer AM -> the later-listed active (vitamin_c) takes AM, BP takes PM.
    assert "benzoyl_peroxide" in rec.target_actives
    assert "vitamin_c" in rec.target_actives
    assert rec.slot_assignment["vitamin_c"] == {"AM"}, rec.slot_assignment
    assert rec.slot_assignment["benzoyl_peroxide"] == {"PM"}, rec.slot_assignment
    assert not any("vitamin_c: held back" in f for f in rec.flags), rec.flags
    # vitamin_c serum -> AM only; BP treatment -> PM only
    assert "p7" in _product_ids(rec.routines["AM"]["serum"])
    assert "p7" not in _product_ids(rec.routines["PM"]["serum"])
    assert "p2" in _product_ids(rec.routines["PM"]["treatment"])
    assert "p2" not in _product_ids(rec.routines["AM"]["treatment"])


def test_bp_retinoid_time_split():
    # comedonal (adapalene) + inflammatory (benzoyl_peroxide) -> both survive,
    # split across slots: retinoid pinned PM, BP shifted to AM. No held-back flag.
    report = ConcernReport("img", concerns=[
        Concern("acne_comedonal", "nose", 2, 0.9),
        Concern("acne_inflammatory", "forehead", 2, 0.9),
    ])
    rec = recommend(report, make_catalog())
    assert "benzoyl_peroxide" in rec.target_actives
    assert "adapalene" in rec.target_actives
    assert rec.slot_assignment["benzoyl_peroxide"] == {"AM"}, rec.slot_assignment
    assert rec.slot_assignment["adapalene"] == {"PM"}, rec.slot_assignment
    assert not any("held back" in f for f in rec.flags), rec.flags
    # BP treatment (p2) AM only; adapalene treatment (p8) PM only
    assert "p2" in _product_ids(rec.routines["AM"]["treatment"])
    assert "p2" not in _product_ids(rec.routines["PM"]["treatment"])
    assert "p8" in _product_ids(rec.routines["PM"]["treatment"])
    assert "p8" not in _product_ids(rec.routines["AM"]["treatment"])


def test_spf_never_in_pm():
    report = ConcernReport("img", concerns=[
        Concern("hyperpigmentation", "left_cheek", 2, 0.9),
    ])
    rec = recommend(report, make_catalog())
    assert "p5" in _product_ids(rec.routines["AM"]["spf"])
    assert rec.routines["PM"]["spf"] == [], rec.routines["PM"]["spf"]


def test_pregnancy_omits_retinoids():
    report = ConcernReport("img", concerns=[
        Concern("acne_comedonal", "nose", 2, 0.9),
    ])
    profile = UserProfile(skin_type="oily", pregnant_or_nursing=True)
    rec = recommend(report, make_catalog(), profile=profile)
    assert "adapalene" not in rec.target_actives
    assert "retinol" not in rec.target_actives
    assert "salicylic_acid" in rec.target_actives
    assert any("pregnancy" in f for f in rec.flags), rec.flags


def test_ranker_reorders_but_comedogenic_dominates():
    report = ConcernReport("img", concerns=[Concern("dryness", "left_cheek", 1, 0.9)])
    # p6 is comedogenic with the HIGHEST score; it must still sort last.
    ranker = StubRanker({"p4": 0.1, "p10": 0.9, "p6": 5.0})
    rec = recommend(report, make_catalog(), ranker=ranker)
    moist = _product_ids(rec.routines["AM"]["moisturizer"])
    assert moist == ["p10", "p4", "p6"], moist


def test_ranker_none_preserves_order():
    report = ConcernReport("img", concerns=[Concern("dryness", "left_cheek", 1, 0.9)])
    rec = recommend(report, make_catalog(), ranker=None)
    moist = _product_ids(rec.routines["AM"]["moisturizer"])
    # stable sort by comedogenic count only -> catalog order preserved
    assert moist == ["p4", "p10", "p6"], moist


def test_low_confidence_flags_verify():
    report = ConcernReport("img", concerns=[Concern("acne_comedonal", "nose", 1, 0.3)])
    rec = recommend(report, make_catalog())
    assert any("possible — verify" in f for f in rec.flags), rec.flags
    # low confidence still contributes actives to the target
    assert "salicylic_acid" in rec.target_actives


def test_severity_3_professional_note():
    report = ConcernReport("img", concerns=[Concern("acne_inflammatory", "forehead", 3, 0.9)])
    rec = recommend(report, make_catalog())
    assert "consider a professional" in rec.flags


def test_comedogenic_downranked_last():
    report = ConcernReport("img", concerns=[Concern("dryness", "left_cheek", 1, 0.9)])
    rec = recommend(report, make_catalog())
    moisturizers = _product_ids(rec.routine["moisturizer"])
    # p4 (clean) and p6 (comedogenic) both match ceramides; p6 sorts last
    assert moisturizers.index("p4") < moisturizers.index("p6"), moisturizers


def test_ordered_steps_follows_category_order():
    report = ConcernReport("img", concerns=[
        Concern("acne_comedonal", "nose", 2, 0.9),
        Concern("hyperpigmentation", "left_cheek", 2, 0.9),
    ])
    rec = recommend(report, make_catalog())
    steps = [c for c, _products in rec.ordered_steps()]
    # steps must be a subsequence of the canonical category order
    it = iter(CATEGORIES)
    assert all(c in it for c in steps), steps


if __name__ == "__main__":
    test_soothe_routine_excludes_strong_active_products()
    test_maintenance_routine_excludes_strong_active_products()
    test_multi_active_product_respects_all_slot_pins()
    test_cystic_path_keeps_low_confidence_verify_flags()
    test_clear_skin_maintenance()
    test_cystic_escalates()
    test_severity_4_escalates()
    test_comedonal_gets_first_line_actives_no_spf()
    test_hyperpigmentation_forces_spf_and_conflict_resolution()
    test_bp_retinoid_time_split()
    test_spf_never_in_pm()
    test_pregnancy_omits_retinoids()
    test_ranker_reorders_but_comedogenic_dominates()
    test_ranker_none_preserves_order()
    test_low_confidence_flags_verify()
    test_severity_3_professional_note()
    test_comedogenic_downranked_last()
    test_ordered_steps_follows_category_order()
    print("ok")
