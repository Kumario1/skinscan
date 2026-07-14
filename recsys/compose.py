"""Routine composer: archetypes as data, AM/PM session assignment, greedy
selection with per-candidate backtracking.

Structural safety rules (from knowledge/safety_rules.json):
- SPF only in AM; retinoids only in PM (pinned sessions).
- A conflicting active pair (e.g. benzoyl peroxide x retinol) never shares a
  session; the composer splits them across AM/PM or rejects the candidate.
- One product per slot; cleanser/moisturizer serve both sessions.
Canonical step order: cleanser -> treatment -> serum -> moisturizer -> spf.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import CatalogProduct
from .contracts import SLOTS, Profile
from .gates import duplicate_active_reasons
from .knowledge import Knowledge
from .scoring import ScoredCandidate
from .signals import TargetConcern

CARRIER_ONLY_SLOTS = ("cleanser", "moisturizer", "spf")


@dataclass
class Step:
    slot: str
    scored: ScoredCandidate
    usage: str  # "AM" | "PM" | "AM_PM"
    notes: list[str] = field(default_factory=list)


@dataclass
class ComposedRoutine:
    archetype: dict
    steps: list[Step] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    compose_vetoes: list[dict] = field(default_factory=list)

    @property
    def product_ids(self) -> frozenset[str]:
        return frozenset(s.scored.product.product_id for s in self.steps)

    @property
    def total_price_usd(self) -> float | None:
        prices = [s.scored.product.price_usd for s in self.steps]
        known = [p for p in prices if p is not None]
        return round(sum(known), 2) if known else None


def _sessions(usage: str) -> frozenset[str]:
    return frozenset(("AM", "PM")) if usage == "AM_PM" else frozenset((usage,))


def preferred_usage(product: CatalogProduct, slot: str, k: Knowledge) -> tuple[str, bool]:
    """(usage, pinned). Pinned sessions may never flip."""
    if slot == "spf":
        return "AM", True
    if slot in ("cleanser", "moisturizer"):
        return "AM_PM", True
    actives = set(product.actives)
    if actives & k.pm_pinned_actives:
        return "PM", True
    if actives & k.am_preferred_actives:
        return "AM", False
    if actives & k.pm_preferred_actives:
        return "PM", False
    # spread actives across the day: treatments default AM, serums PM
    return ("AM" if slot == "treatment" else "PM"), False


def _conflict_between(a: CatalogProduct, b: CatalogProduct, k: Knowledge):
    for x in sorted(a.actives):
        for y in sorted(b.actives):
            if frozenset((x, y)) in k.active_conflicts:
                return x, y
    return None


def try_place(product: CatalogProduct, slot: str, steps: list[Step], k: Knowledge):
    """Find a session for the product that avoids every conflict pair, flipping
    away from the preferred session when allowed. Returns (usage, None) or
    (None, veto_reason)."""
    usage, pinned = preferred_usage(product, slot, k)
    options = [usage] if pinned else [usage, "PM" if usage == "AM" else "AM"]
    last_clash = None
    for option in options:
        clash = None
        for step in steps:
            if _sessions(option) & _sessions(step.usage):
                pair = _conflict_between(product, step.scored.product, k)
                if pair:
                    clash = pair
                    break
        if clash is None:
            return option, None
        last_clash = clash
    return None, f"conflicts_selected_active:{last_clash[0]}:{last_clash[1]}"


def _covered_concerns(
    products: list[CatalogProduct], targets: tuple[TargetConcern, ...], k: Knowledge
) -> set[str]:
    covered: set[str] = set()
    for t in targets:
        wanted = k.concern_actives.get(t.concern, frozenset())
        if any(set(p.actives) & wanted for p in products):
            covered.add(t.concern)
    return covered


def _filter_candidates(
    scored: list[ScoredCandidate],
    slot: str,
    constraints: dict,
    selected: list[CatalogProduct],
    targets: tuple[TargetConcern, ...],
    k: Knowledge,
    notes: list[str],
) -> list[ScoredCandidate]:
    out = scored
    if constraints.get("gentle"):
        out = [s for s in out if not (set(s.product.actives) & k.gentle_excluded_actives)]
        if slot in ("treatment", "serum"):
            out = [s for s in out if set(s.product.actives) & k.gentle_allowlist]
    if constraints.get("single_treatment_active") and slot != "treatment":
        out = [s for s in out if not (set(s.product.actives) & k.treatment_actives)]
    cap = constraints.get("max_item_price_usd")
    if cap is not None:
        out = [s for s in out
               if s.product.price_usd is not None and s.product.price_usd <= cap]
    if constraints.get("complementary_serum") and slot == "serum":
        covered = _covered_concerns(selected, targets, k)
        complementary = [
            s for s in out
            if _covered_concerns([s.product], targets, k) - covered
        ]
        if complementary:
            out = complementary
        elif out:
            notes.append("serum_not_complementary_no_uncovered_concern")
    return out


def _place_best(
    candidates: list[ScoredCandidate],
    slot: str,
    steps: list[Step],
    k: Knowledge,
    compose_vetoes: list[dict],
    exclude_ids: frozenset[str] = frozenset(),
    max_price: float | None = None,
) -> Step | None:
    selected = [s.scored.product for s in steps]
    for cand in candidates:
        if cand.product.product_id in exclude_ids:
            continue
        if max_price is not None and (
            cand.product.price_usd is None or cand.product.price_usd >= max_price
        ):
            continue
        reasons = duplicate_active_reasons(cand.product, selected, k)
        if not reasons:
            usage, conflict = try_place(cand.product, slot, steps, k)
            if conflict:
                reasons = [conflict]
            else:
                return Step(slot, cand, usage)
        compose_vetoes.extend(
            {"product_id": cand.product.product_id, "slot": slot, "reason": r}
            for r in reasons
        )
    return None


def compose_archetype(
    archetype: dict,
    scored_by_slot: dict[str, list[ScoredCandidate]],
    targets: tuple[TargetConcern, ...],
    profile: Profile,
    k: Knowledge,
) -> ComposedRoutine:
    constraints = archetype.get("constraints") or {}
    routine = ComposedRoutine(archetype=archetype)
    slots = list(archetype["slots"])
    if not targets:
        slots = [s for s in slots if s in CARRIER_ONLY_SLOTS]
        routine.notes.append("clear_skin_maintenance")

    filtered: dict[str, list[ScoredCandidate]] = {}
    for slot in slots:
        filtered[slot] = _filter_candidates(
            scored_by_slot.get(slot, []), slot, constraints,
            [s.scored.product for s in routine.steps], targets, k, routine.notes,
        )
        step = _place_best(filtered[slot], slot, routine.steps, k, routine.compose_vetoes)
        if step is None:
            routine.notes.append(f"slot_unfilled:{slot}")
        else:
            routine.steps.append(step)

    if constraints.get("spf_policy") == "fold_or_add" and "spf" not in slots:
        moisturizer = next((s for s in routine.steps if s.slot == "moisturizer"), None)
        if moisturizer and (moisturizer.scored.product.spf or 0) >= k.min_spf:
            moisturizer.notes.append("covers_spf")
            routine.notes.append("moisturizer_covers_spf")
        else:
            spf_candidates = _filter_candidates(
                scored_by_slot.get("spf", []), "spf", constraints,
                [s.scored.product for s in routine.steps], targets, k, routine.notes,
            )
            step = _place_best(spf_candidates, "spf", routine.steps, k, routine.compose_vetoes)
            if step is None:
                routine.notes.append("slot_unfilled:spf")
            else:
                routine.steps.append(step)
                routine.notes.append("spf_added_to_minimal")

    total_cap = constraints.get("max_total_price_usd")
    if total_cap is not None:
        _reduce_to_budget(routine, filtered, total_cap, k)

    routine.steps.sort(key=lambda s: SLOTS.index(s.slot))
    return routine


def _reduce_to_budget(
    routine: ComposedRoutine,
    filtered: dict[str, list[ScoredCandidate]],
    total_cap: float,
    k: Knowledge,
) -> None:
    """Swap the most expensive step for the next-cheaper candidate in its slot
    until the total fits (strictly-decreasing prices, so this terminates)."""
    while (routine.total_price_usd or 0) > total_cap:
        priced = [s for s in routine.steps if s.scored.product.price_usd is not None]
        if not priced:
            break
        replaced = False
        for step in sorted(priced, key=lambda s: -s.scored.product.price_usd):
            others = [s for s in routine.steps if s is not step]
            replacement = _place_best(
                filtered.get(step.slot, []), step.slot, others, k,
                routine.compose_vetoes,
                exclude_ids=frozenset({step.scored.product.product_id}),
                max_price=step.scored.product.price_usd,
            )
            if replacement is not None:
                routine.steps[routine.steps.index(step)] = replacement
                routine.notes.append(
                    f"budget_swap:{step.slot}:{step.scored.product.product_id}"
                    f"->{replacement.scored.product.product_id}"
                )
                replaced = True
                break
        if not replaced:
            routine.notes.append("over_total_budget")
            break


def compose_all(
    archetype_scored: list[tuple[dict, dict[str, list[ScoredCandidate]]]],
    targets: tuple[TargetConcern, ...],
    profile: Profile,
    k: Knowledge,
) -> list[ComposedRoutine]:
    """Compose every archetype in order (best_overall first) and enforce the
    diversity guarantee: each later archetype differs from best_overall by at
    least one product."""
    routines: list[ComposedRoutine] = []
    best_ids: frozenset[str] | None = None
    for archetype, scored_by_slot in archetype_scored:
        routine = compose_archetype(archetype, scored_by_slot, targets, profile, k)
        if best_ids is None:
            best_ids = routine.product_ids
        elif routine.steps and routine.product_ids == best_ids:
            _diversify(routine, scored_by_slot, targets, k)
        routines.append(routine)
    return routines


def _diversify(
    routine: ComposedRoutine,
    scored_by_slot: dict[str, list[ScoredCandidate]],
    targets: tuple[TargetConcern, ...],
    k: Knowledge,
) -> None:
    constraints = routine.archetype.get("constraints") or {}
    for step in routine.steps:
        candidates = _filter_candidates(
            scored_by_slot.get(step.slot, []), step.slot, constraints,
            [s.scored.product for s in routine.steps if s is not step],
            targets, k, routine.notes,
        )
        others = [s for s in routine.steps if s is not step]
        replacement = _place_best(
            candidates, step.slot, others, k, routine.compose_vetoes,
            exclude_ids=frozenset({step.scored.product.product_id}),
        )
        if replacement is not None:
            routine.steps[routine.steps.index(step)] = replacement
            routine.notes.append(f"diversified_from_best_overall:{step.slot}")
            routine.steps.sort(key=lambda s: SLOTS.index(s.slot))
            return
    routine.notes.append("identical_to_best_overall_no_alternative")
