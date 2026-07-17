"""The total-budget reducer (recsys/compose.py::_reduce_to_budget).

It swaps the most expensive step for a cheaper candidate until the routine fits.
Its docstring promises termination via "strictly-decreasing prices", so the
loop's exits -- fits, nothing cheaper, nothing priced -- are pinned here.
"""
from pathlib import Path

from recsys.catalog import CatalogProduct
from recsys.compose import ComposedRoutine, Step, _reduce_to_budget
from recsys.knowledge import load_knowledge
from recsys.scoring import ScoredCandidate

K = load_knowledge(Path(__file__).parents[1] / "data" / "knowledge")


def _product(pid, category, price):
    return CatalogProduct(
        product_id=pid, name=pid, brand="b", category=category,
        price_usd=price, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=(),
    )


def _scored(prod):
    return ScoredCandidate(prod, 0.5, ())


def _routine(*steps):
    return ComposedRoutine(archetype={}, steps=list(steps))


def _step(pid, slot, price, usage="AM"):
    return Step(slot, _scored(_product(pid, slot, price)), usage)


def test_a_routine_already_within_budget_is_untouched():
    routine = _routine(_step("cleanser-a", "cleanser", 10.0),
                       _step("moisturizer-a", "moisturizer", 15.0))
    _reduce_to_budget(routine, {}, 100.0, K)
    assert [s.scored.product.product_id for s in routine.steps] == [
        "cleanser-a", "moisturizer-a"]
    assert routine.notes == []


def test_the_most_expensive_step_is_swapped_for_a_cheaper_one():
    routine = _routine(_step("cleanser-a", "cleanser", 10.0),
                       _step("moisturizer-lux", "moisturizer", 90.0))
    filtered = {"moisturizer": [_scored(_product("moisturizer-lux", "moisturizer", 90.0)),
                                _scored(_product("moisturizer-cheap", "moisturizer", 20.0))]}
    _reduce_to_budget(routine, filtered, 50.0, K)

    assert routine.total_price_usd == 30.0
    assert {s.scored.product.product_id for s in routine.steps} == {
        "cleanser-a", "moisturizer-cheap"}
    assert "budget_swap:moisturizer:moisturizer-lux->moisturizer-cheap" in routine.notes


def test_it_keeps_swapping_until_the_total_fits():
    routine = _routine(_step("cleanser-lux", "cleanser", 60.0),
                       _step("moisturizer-lux", "moisturizer", 60.0))
    filtered = {
        "cleanser": [_scored(_product("cleanser-cheap", "cleanser", 5.0))],
        "moisturizer": [_scored(_product("moisturizer-cheap", "moisturizer", 5.0))],
    }
    _reduce_to_budget(routine, filtered, 20.0, K)

    assert routine.total_price_usd == 10.0
    assert len([n for n in routine.notes if n.startswith("budget_swap:")]) == 2
    assert "over_total_budget" not in routine.notes


def test_no_cheaper_candidate_is_reported_rather_than_looping():
    routine = _routine(_step("moisturizer-lux", "moisturizer", 90.0))
    _reduce_to_budget(routine, {"moisturizer": []}, 10.0, K)

    assert routine.notes == ["over_total_budget"]
    assert routine.total_price_usd == 90.0, "an unaffordable routine is reported, not falsified"


def test_an_equally_priced_alternative_is_never_swapped_in():
    """The termination guarantee is "strictly-decreasing prices": a same-price
    swap would leave the total unchanged and the loop would spin forever."""
    routine = _routine(_step("moisturizer-a", "moisturizer", 90.0))
    filtered = {"moisturizer": [_scored(_product("moisturizer-b", "moisturizer", 90.0))]}
    _reduce_to_budget(routine, filtered, 10.0, K)

    assert routine.notes == ["over_total_budget"]
    assert [s.scored.product.product_id for s in routine.steps] == ["moisturizer-a"]


def test_unpriced_steps_cannot_be_swapped_away():
    """A step with no price contributes nothing to the total and offers nothing
    to trade; the reducer must stop instead of spinning on it."""
    routine = _routine(_step("cleanser-a", "cleanser", 60.0))
    routine.steps.append(Step("moisturizer", _scored(
        _product("moisturizer-unpriced", "moisturizer", None)), "AM"))
    _reduce_to_budget(routine, {"cleanser": [], "moisturizer": []}, 10.0, K)

    assert routine.notes == ["over_total_budget"]


def test_a_routine_with_no_priced_step_at_all_terminates():
    routine = _routine()
    routine.steps.append(Step("moisturizer", _scored(
        _product("moisturizer-unpriced", "moisturizer", None)), "AM"))
    _reduce_to_budget(routine, {}, 10.0, K)
    assert routine.total_price_usd is None
