from pathlib import Path

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.knowledge import load_knowledge
from recsys.scoring import score_products
from recsys.signals import ScoringContext, SignalScore, TargetConcern

K = load_knowledge(Path(__file__).parents[1] / "data" / "knowledge")


class FixedSignal:
    def __init__(self, name, value):
        self.name = name
        self.version = "test"
        self._value = value

    def score(self, product, slot, ctx):
        if self._value is None:
            return None
        return SignalScore(self.name, self._value, f"{self.name} evidence")


def ctx(targets=()):
    return ScoringContext(
        targets=tuple(targets), profile=Profile(pregnancy_status="not_pregnant"),
        knowledge=K, category_prices={},
    )


def product(pid="p1", actives=()):
    return CatalogProduct(
        product_id=pid, name=pid, brand="b", category="treatment",
        price_usd=None, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=tuple(actives),
    )


def test_final_score_is_decomposable_weighted_mean():
    providers = [FixedSignal("a", 0.8), FixedSignal("b", 0.4)]
    weights = {"a": 0.75, "b": 0.25}
    [scored] = score_products([product()], "treatment", providers, ctx(), weights)
    recomputed = sum(
        weights[s.name] * s.value for s in scored.signals
    ) / sum(weights[s.name] for s in scored.signals)
    assert scored.final == round(recomputed, 6) == 0.7


def test_missing_data_is_neutral_with_uncertainty_note():
    providers = [FixedSignal("a", 0.8), FixedSignal("b", None)]
    weights = {"a": 0.5, "b": 0.5}
    [scored] = score_products([product()], "treatment", providers, ctx(), weights)
    b = next(s for s in scored.signals if s.name == "b")
    assert b.value == 0.5 and b.details.get("missing")
    assert "no_b_data" in scored.uncertainty
    assert scored.final == 0.65


def test_zero_weight_provider_skipped():
    providers = [FixedSignal("a", 0.8), FixedSignal("b", 0.1)]
    [scored] = score_products([product()], "treatment", providers, ctx(), {"a": 1.0})
    assert [s.name for s in scored.signals] == ["a"]


def test_deterministic_tiebreak_by_product_id():
    providers = [FixedSignal("a", 0.5)]
    scored = score_products(
        [product("p2"), product("p1")], "treatment", providers, ctx(), {"a": 1.0}
    )
    assert [s.product.product_id for s in scored] == ["p1", "p2"]


def test_concern_fit_severity_weighting():
    from recsys.signals import ConcernFitSignal
    targets = [
        TargetConcern("acne_inflammatory", 4, 0.9),
        TargetConcern("hyperpigmentation", 2, 0.8),
    ]
    azelaic = product(actives=["azelaic_acid"])  # matches both concerns
    dryness_only = product("p2", actives=["squalane"])  # matches neither
    fit = ConcernFitSignal()
    both = fit.score(azelaic, "treatment", ctx(targets))
    none = fit.score(dryness_only, "treatment", ctx(targets))
    assert both.value == 1.0
    assert "azelaic acid" in both.evidence
    assert none.value == 0.0
    # partial match scales with severity: niacinamide targets both here too,
    # vitamin_c only hyperpigmentation (severity 2 of total 6)
    vitc = fit.score(product("p3", actives=["vitamin_c"]), "treatment", ctx(targets))
    assert vitc.value == round(2 / 6, 6)
