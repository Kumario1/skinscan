from pathlib import Path

from recsys.catalog import CatalogProduct
from recsys.compose import Step, preferred_usage, try_place
from recsys.knowledge import load_knowledge
from recsys.scoring import ScoredCandidate

K = load_knowledge(Path(__file__).parents[1] / "data" / "knowledge")


def product(pid, category, actives=()):
    return CatalogProduct(
        product_id=pid, name=pid, brand="b", category=category,
        price_usd=10.0, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=tuple(actives),
    )


def step(prod, slot, usage):
    return Step(slot, ScoredCandidate(prod, 0.5, ()), usage)


def test_pinned_sessions():
    assert preferred_usage(product("p1", "spf"), "spf", K) == ("AM", True)
    assert preferred_usage(product("p2", "treatment", ["retinol"]), "treatment", K) == ("PM", True)
    assert preferred_usage(product("p3", "cleanser"), "cleanser", K) == ("AM_PM", True)


def test_session_preferences():
    assert preferred_usage(product("p1", "serum", ["vitamin_c"]), "serum", K)[0] == "AM"
    assert preferred_usage(product("p2", "serum", ["glycolic_acid"]), "serum", K)[0] == "PM"
    # slot defaults spread actives across the day
    assert preferred_usage(product("p3", "treatment", ["azelaic_acid"]), "treatment", K)[0] == "AM"
    assert preferred_usage(product("p4", "serum", ["niacinamide"]), "serum", K)[0] == "PM"


def test_conflict_splits_across_sessions():
    retinol_pm = step(product("p1", "treatment", ["retinol"]), "treatment", "PM")
    bp = product("p2", "serum", ["benzoyl_peroxide"])
    usage, reason = try_place(bp, "serum", [retinol_pm], K)
    assert reason is None
    assert usage == "AM"  # BP prefers AM anyway; never shares PM with the retinoid


def test_conflict_with_both_sessions_is_vetoed():
    bp_cleanser = step(product("p1", "cleanser", ["benzoyl_peroxide"]), "cleanser", "AM_PM")
    retinol = product("p2", "treatment", ["retinol"])
    usage, reason = try_place(retinol, "treatment", [bp_cleanser], K)
    assert usage is None
    # candidate's active first, then the already-selected product's active
    assert reason == "conflicts_selected_active:retinol:benzoyl_peroxide"


def test_vitamin_c_flips_when_am_conflicts():
    bp_am = step(product("p1", "treatment", ["benzoyl_peroxide"]), "treatment", "AM")
    vitc = product("p2", "serum", ["vitamin_c"])
    usage, reason = try_place(vitc, "serum", [bp_am], K)
    assert reason is None
    assert usage == "PM"  # AM preference flips away from the BP conflict
