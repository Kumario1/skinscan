"""Multi-signal scorer. Every final score is decomposable: it is exactly the
weighted mean of the retained per-signal values, and the explanation builder
reads the same SignalScore objects — there is no separate marketing-copy path.

Missing data is never a hidden penalty or bonus. A provider that returns None
has no data for this product, so it is retained by nothing: not the numerator,
not the denominator, and not the emitted signal list. Its weight is redistributed
across the signals that do have something to say, exactly as an absent *store*
already behaves, and the product's `final` becomes precisely what its remaining
signals say. Only an uncertainty note records the absence.

Emitting a placeholder value here would break both invariants at once: it would
make `final` disagree with the mean of the emitted signals, and it would publish
a per-signal number the score never actually used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import CatalogProduct
from .signals import ScoringContext, SignalScore

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
                # No data for this product: drop the weight from the denominator
                # so `final` stays the weighted mean of what we actually know.
                # Substituting a "neutral" 0.5 while keeping the weight is not
                # neutral — it is only neutral when the product's other signals
                # already average 0.5, and otherwise drags good products down and
                # lifts bad ones up. This mirrors the missing-*store* path above,
                # where an absent provider's weight simply never enters the sum.
                uncertainty.append(f"no_{provider.name}_data")
                continue
            signals.append(result)
            acc += weight * result.value
            total_weight += weight
        # No signal had anything to say: there is no weighted mean to report.
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
