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

VERIFICATION_RANK = {"verified": 0, "partial": 1, "unverified": 2}


def verification_status(product: CatalogProduct, slot: str) -> str:
    """D-035 completeness tier. Safety eligibility remains in gates.py."""
    role = "sunscreen" if slot == "spf" else slot
    facts = [
        bool(product.intended_areas),
        role in product.routine_roles,
        product.format is not None,
        product.exposure is not None,
        product.cadence is not None and bool(product.cadence_source),
        product.contraindications_verified,
        product.comedogenic_claim not in (None, "unknown"),
    ]
    if slot in {"treatment", "spf"}:
        facts.extend([bool(product.label_source), bool(product.label_verified_at)])
    if slot == "treatment":
        facts.append(bool(product.drug_actives))
    if slot == "spf":
        facts.extend([product.spf_source == "verified", product.broad_spectrum is True])
    if all(facts):
        return "verified"
    # D-036: the catalog build derives routine_roles/cadence_source from the
    # dump taxonomy, so those fields no longer distinguish reviewed evidence
    # from a derived default. Only overlay-set markers count as evidence.
    has_evidence = bool(
        product.evidence_grade
        or product.daily_support_verified
        or product.label_source
        or product.contraindications_verified
    )
    return "partial" if has_evidence else "unverified"


@dataclass(frozen=True)
class ScoredCandidate:
    product: CatalogProduct
    final: float
    signals: tuple[SignalScore, ...]
    uncertainty: tuple[str, ...] = field(default=())
    verification_status: str = "unverified"


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
        scored.append(ScoredCandidate(
            product, final, tuple(signals), tuple(uncertainty),
            verification_status(product, slot),
        ))
    def rank_key(s: ScoredCandidate) -> tuple:
        fit = next((signal.value for signal in s.signals if signal.name == "concern_fit"), 0.0)
        return (-fit, VERIFICATION_RANK[s.verification_status], -s.final, s.product.product_id)
    return sorted(scored, key=rank_key)
