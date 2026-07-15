"""Multi-signal scorer. Every final score is decomposable: it is exactly the
weighted mean of the retained per-signal values, and the explanation builder
reads the same SignalScore objects — there is no separate marketing-copy path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import CatalogProduct
from .signals import ScoringContext, SignalScore

NEUTRAL_VALUE = 0.5
VERIFIED_BONUS = 0.05  # ranking nudge for evidence-verified products (sort-only)


@dataclass(frozen=True)
class ScoredCandidate:
    product: CatalogProduct
    final: float
    signals: tuple[SignalScore, ...]
    uncertainty: tuple[str, ...] = field(default=())


def score_products(
    products: list[CatalogProduct],
    slot: str,
    providers: list,
    ctx: ScoringContext,
    weights: dict[str, float],
) -> list[ScoredCandidate]:
    """Score and sort one slot's candidates (desc by final, ties by product_id
    for determinism). Providers whose name has no positive weight are skipped."""
    active = [p for p in providers if weights.get(p.name, 0) > 0]
    scored: list[ScoredCandidate] = []
    for product in products:
        signals: list[SignalScore] = []
        uncertainty: list[str] = []
        total_weight = 0.0
        acc = 0.0
        for provider in active:
            weight = weights[provider.name]
            result = provider.score(product, slot, ctx)
            if result is None:
                result = SignalScore(provider.name, NEUTRAL_VALUE, "no data", {"missing": True})
                uncertainty.append(f"no_{provider.name}_data")
            signals.append(result)
            acc += weight * result.value
            total_weight += weight
        final = round(acc / total_weight, 6) if total_weight else 0.0
        scored.append(ScoredCandidate(product, final, tuple(signals), tuple(uncertainty)))
    # Evidence-verified products (usage facts proven from a source) get a modest
    # ranking nudge over category-derived ones — applied only in the sort so the
    # stored `final` stays exactly the weighted mean. In strict mode every kept
    # product is verified, so the nudge is uniform and changes nothing.
    def rank_key(s: ScoredCandidate) -> tuple:
        nudge = VERIFIED_BONUS if s.product.routine_roles else 0.0
        return (-(s.final + nudge), s.product.product_id)
    return sorted(scored, key=rank_key)
