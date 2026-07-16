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


class PerProductSignal:
    name = "concern_fit"
    version = "test"

    def __init__(self, values):
        self.values = values

    def score(self, product, slot, ctx):
        value = self.values[product.product_id]
        return SignalScore(self.name, value, "synthetic fit")


def ctx(targets=()):
    return ScoringContext(
        targets=tuple(targets), profile=Profile(pregnancy_status="not_pregnant"),
        knowledge=K, category_prices={},
    )


def product(pid="p1", actives=(), roles=(), **facts):
    return CatalogProduct(
        product_id=pid, name=pid, brand="b", category="treatment",
        price_usd=None, size=None, format=facts.pop("format", None), spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=tuple(actives), routine_roles=tuple(roles),
        **facts,
    )


def test_final_score_is_decomposable_weighted_mean():
    providers = [FixedSignal("a", 0.8), FixedSignal("b", 0.4)]
    weights = {"a": 0.75, "b": 0.25}
    [scored] = score_products([product()], "treatment", providers, ctx(), weights)
    recomputed = sum(
        weights[s.name] * s.value for s in scored.signals
    ) / sum(weights[s.name] for s in scored.signals)
    assert scored.final == round(recomputed, 6) == 0.7


def test_missing_data_drops_its_weight_instead_of_scoring_a_neutral_value():
    """A signal with no data must not move `final` at all.

    This test previously asserted 0.65 and was named ..._is_neutral_..., blessing
    the bug: substituting 0.5 for the missing signal while leaving its 0.5 weight
    in the denominator dragged a product whose only real signal said 0.8 down to
    0.65. That is a 0.15 hidden penalty for data we simply do not have, which both
    signals.py and ARCHITECTURE.md promise cannot happen. Renormalising instead
    makes the product score exactly what signal `a` says.
    """
    providers = [FixedSignal("a", 0.8), FixedSignal("b", None)]
    weights = {"a": 0.5, "b": 0.5}
    [scored] = score_products([product()], "treatment", providers, ctx(), weights)
    assert scored.final == 0.8  # was 0.65 while `b`'s weight stayed in the denominator
    # no placeholder is emitted either: `final` may only cite signals it used
    assert [s.name for s in scored.signals] == ["a"]
    assert "no_b_data" in scored.uncertainty


def test_missing_signal_neither_penalises_a_good_product_nor_rescues_a_bad_one():
    """The substitution was distorting in both directions, at real weights.

    These are the real best_overall weights minus concern_efficacy, whose store is
    absent today — so its 0.25 is filtered at the provider level and the
    denominator renormalises to 0.75. That renormalisation was always correct; the
    None path was the inconsistency, keeping the weight for a value it invented.

    Before the fix, a product whose every signal said 1.0 scored 0.9 with
    popularity missing and 0.833333 with review_quality missing, while a product
    whose every signal said 0.0 was *lifted* to 0.1 / 0.166667. The magnitude
    tracked the missing signal's weight and the sign tracked whether the product
    sat above or below 0.5 — one hidden penalty and one hidden bonus from a single
    line. A missing signal must move `final` in neither direction.
    """
    weights = {"concern_fit": 0.25, "ingredient_analysis": 0.05,
               "review_quality": 0.25, "popularity": 0.15, "price_value": 0.05}
    for uniform in (1.0, 0.0):
        for missing in ("popularity", "review_quality"):
            providers = [
                FixedSignal(name, None if name == missing else uniform)
                for name in weights
            ]
            [scored] = score_products([product()], "treatment", providers, ctx(), weights)
            assert scored.final == uniform, (
                f"every signal says {uniform} but {missing} is missing: "
                f"final={scored.final} — the absence moved the score"
            )


def test_final_stays_decomposable_when_a_signal_is_missing():
    """The explanation must remain recomputable when data is absent.

    `final` is exactly the weighted mean of the signals actually emitted, so a
    reader can re-derive the score from the explanation alone. Emitting a
    placeholder for the missing signal would break precisely this: the emitted
    list would carry a 0.5 that `final` never used, which is the "separate
    marketing-copy path" the module docstring rules out.
    """
    providers = [FixedSignal("a", 0.8), FixedSignal("b", 0.4), FixedSignal("c", None)]
    weights = {"a": 0.5, "b": 0.25, "c": 0.25}
    [scored] = score_products([product()], "treatment", providers, ctx(), weights)
    recomputed = sum(
        weights[s.name] * s.value for s in scored.signals
    ) / sum(weights[s.name] for s in scored.signals)
    # c's 0.25 leaves the denominator entirely: (0.5*0.8 + 0.25*0.4) / 0.75
    assert scored.final == round(recomputed, 6) == 0.666667


def test_verification_status_is_explicit_and_sort_only():
    providers = [FixedSignal("a", 0.8), FixedSignal("b", None)]
    weights = {"a": 0.5, "b": 0.5}
    scored = score_products(
        # p2 has some evidence; p1 has none. Completeness is a tier, not a score.
        [product("p1"), product("p2", roles=("treatment",))],
        "treatment", providers, ctx(), weights,
    )
    assert [s.product.product_id for s in scored] == ["p2", "p1"]
    assert [s.final for s in scored] == [0.8, 0.8]
    assert [s.verification_status for s in scored] == ["partial", "unverified"]


def test_verified_candidate_precedes_partial_candidate():
    full = product(
        "full", actives=("azelaic_acid",), roles=("treatment",),
        intended_areas=("face",), format="cream", exposure="leave_on",
        cadence="daily", cadence_source="https://example.test/directions",
        contraindications_verified=True,
        comedogenic_claim="claimed_noncomedogenic",
        label_source="https://example.test/label", label_verified_at="2026-07-16",
        drug_actives=({
            "name": "azelaic_acid", "strength": "10%",
            "source": "https://example.test/label",
        },),
    )
    partial = product("partial", roles=("treatment",))
    scored = score_products(
        [partial, full], "treatment", [FixedSignal("a", 0.5)], ctx(), {"a": 1.0},
    )

    assert [item.verification_status for item in scored] == ["verified", "partial"]


def test_therapeutic_fit_precedes_verification_completeness():
    verified = product(
        "verified", actives=("azelaic_acid",), roles=("treatment",),
        intended_areas=("face",), format="cream", exposure="leave_on",
        cadence="daily", cadence_source="https://example.test/directions",
        contraindications_verified=True,
        comedogenic_claim="claimed_noncomedogenic",
        label_source="https://example.test/label", label_verified_at="2026-07-16",
        drug_actives=({
            "name": "azelaic_acid", "strength": "10%",
            "source": "https://example.test/label",
        },),
    )
    partial = product("partial", roles=("treatment",))
    scored = score_products(
        [verified, partial], "treatment",
        [PerProductSignal({"verified": 0.2, "partial": 0.8})],
        ctx(), {"concern_fit": 1.0},
    )

    assert [item.product.product_id for item in scored] == ["partial", "verified"]


def test_a_product_with_no_signals_at_all_scores_zero_rather_than_an_invented_mean():
    """Documents the one behaviour renormalising makes reachable.

    Previously every provider contributed weight even with no data, so
    total_weight could never hit 0 and the `else 0.0` guard was dead; such a
    product scored a phantom 0.5. Now that missing weights leave the denominator,
    a product nothing can speak to has no weighted mean to report. The
    pre-existing 0.0 convention is kept deliberately: it is uniform across such
    products (so their relative order is still the product_id tiebreak) and it
    stops an entirely unknown product from outranking one with real, merely
    mediocre, evidence. Every absence is still reported in `uncertainty`.
    """
    providers = [FixedSignal("a", None), FixedSignal("b", None)]
    weights = {"a": 0.5, "b": 0.5}
    [scored] = score_products([product()], "treatment", providers, ctx(), weights)
    assert scored.final == 0.0
    assert scored.signals == ()
    assert sorted(scored.uncertainty) == ["no_a_data", "no_b_data"]


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
