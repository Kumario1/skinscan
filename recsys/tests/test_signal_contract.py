"""Invariants every signal provider must honour, whatever its store says.

SignalScore.value is documented as 0..1 (recsys/signals.py). A provider that
leaks a value outside that range silently reweights every routine, so the
bound is tested here against adversarial stores rather than trusted.
"""
from pathlib import Path

import pytest

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.knowledge import load_knowledge
from recsys.signals import (
    ConcernEfficacySignal,
    ConcernFitSignal,
    IngredientAnalysisSignal,
    PopularitySignal,
    PriceValueSignal,
    ReviewQualitySignal,
    ScoringContext,
    TargetConcern,
)

K = load_knowledge(Path(__file__).parents[1] / "data" / "knowledge")


def _product(product_id="p1", category="treatment", price_usd=20.0, actives=("adapalene",)):
    return CatalogProduct(
        product_id=product_id, name="Test", brand="Test", category=category,
        price_usd=price_usd, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=tuple(actives),
    )


def _ctx(targets=(("acne_inflammatory", 3, 0.9),), skin_type="oily", category_prices=None):
    return ScoringContext(
        targets=tuple(TargetConcern(*t) for t in targets),
        profile=Profile(pregnancy_status="not_pregnant", skin_type=skin_type),
        knowledge=K, category_prices=category_prices or {},
    )


def _store_signals():
    """Every store-backed provider, primed with a store that knows nothing."""
    return [
        ConcernEfficacySignal({"products": {}}, {"version": "v1"}),
        IngredientAnalysisSignal({"products": {}}, {"version": "v1"}),
        PopularitySignal({"products": {}}, {"version": "v1"}),
        ReviewQualitySignal({"products": {}}, {"version": "v1"}),
    ]


def test_store_signals_abstain_for_an_unknown_product():
    """No entry means no opinion -- never a crash and never a zero score, which
    would be an actively negative verdict the store never made."""
    for signal in _store_signals():
        assert signal.score(_product(), "treatment", _ctx()) is None, signal.name


def test_built_in_signals_abstain_when_they_have_nothing_to_go_on():
    assert ConcernFitSignal().score(_product(), "treatment", _ctx(targets=())) is None
    # a lone price has no peers to compare against
    assert PriceValueSignal().score(
        _product(), "treatment", _ctx(category_prices={"treatment": (20.0,)})) is None
    assert PriceValueSignal().score(
        _product(price_usd=None), "treatment",
        _ctx(category_prices={"treatment": (10.0, 20.0)})) is None


@pytest.mark.parametrize("prices, price, expected", [
    ((10.0, 20.0, 30.0), 10.0, 1.0),   # cheapest
    ((10.0, 20.0, 30.0), 30.0, 0.0),   # dearest
    ((10.0, 20.0, 30.0), 20.0, 0.5),   # middle
    ((10.0, 10.0), 10.0, 0.0),         # tied
])
def test_price_value_ranks_against_category_peers(prices, price, expected):
    score = PriceValueSignal().score(
        _product(price_usd=price), "treatment", _ctx(category_prices={"treatment": prices}))
    assert score.value == expected


def test_price_value_stays_bounded_for_a_product_cheaper_than_every_peer():
    """A product absent from its own category price list would compute >1
    without the clamp, which would outrank a genuine best-in-category price."""
    score = PriceValueSignal().score(
        _product(price_usd=1.0), "treatment", _ctx(category_prices={"treatment": (10.0, 20.0)}))
    assert score.value == 1.0


@pytest.mark.parametrize("smoothed, n", [
    (0.0, 1), (0.0, 10_000), (1.0, 1), (1.0, 10_000), (0.5, 1),
])
def test_concern_efficacy_stays_within_bounds_at_extremes(smoothed, n):
    """Extreme help rates in tiny or huge cells must still land in 0..1, and a
    thin cell must be shrunk further toward neutral than a thick one."""
    store = {"products": {"p1": {"acne_inflammatory": {
        "all": {"n": n, "smoothed": smoothed, "help_rate": smoothed}}}}}
    score = ConcernEfficacySignal(store, {"version": "v1"}).score(
        _product(), "treatment", _ctx())
    assert 0.0 <= score.value <= 1.0


def test_concern_efficacy_shrinks_a_thin_cell_further_toward_neutral():
    def value(n):
        store = {"products": {"p1": {"acne_inflammatory": {
            "all": {"n": n, "smoothed": 1.0, "help_rate": 1.0}}}}}
        return ConcernEfficacySignal(store, {"version": "v1"}).score(
            _product(), "treatment", _ctx()).value

    thin, thick = value(1), value(10_000)
    assert 0.5 < thin < thick <= 1.0


def test_concern_efficacy_falls_back_to_general_acne_then_pooled():
    general = {"products": {"p1": {"acne_general": {
        "all": {"n": 50, "smoothed": 0.8, "help_rate": 0.8}}}}}
    score = ConcernEfficacySignal(general, {"version": "v1"}).score(
        _product(), "treatment", _ctx())
    assert score.details["matches"][0]["ladder"] == "acne_general"

    pooled = {"products": {"p1": {"n": 40, "smoothed": 4.0}}}
    score = ConcernEfficacySignal({"products": {}}, {"version": "v1"}, pooled).score(
        _product(), "treatment", _ctx())
    assert score.details["matches"][0]["ladder"] == "pooled"
    assert 0.0 <= score.value <= 1.0


def test_concern_fit_scores_zero_when_no_active_targets_the_concern():
    score = ConcernFitSignal().score(
        _product(actives=("glycerin",)), "treatment", _ctx())
    assert score.value == 0.0
    assert score.details["matched"] == {}
