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
from .gates import duplicate_active_reasons, profile_gate_reasons
from .knowledge import Knowledge
from .scoring import ScoredCandidate
from .signals import TargetLesion

CARRIER_ONLY_SLOTS = ("cleanser", "moisturizer", "spf")
STEP_ORDER = (*SLOTS, "scar_care")


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
    """The sessions a step may occupy.

    PER_LABEL means the label sets the schedule and the engine does not know
    which sessions that turns out to be. An unknown session fails SAFE as both:
    it might land anywhere, so it must conflict with everything. Returning an
    empty set here instead would make a PER_LABEL step conflict with *nothing*
    -- the one place this codebase's "unknowns fail safe" invariant was
    inverted, which let a PER_LABEL adapalene share a session with benzoyl
    peroxide unchecked.
    """
    if usage in ("AM_PM", "PER_LABEL"):
        return frozenset(("AM", "PM"))
    return frozenset((usage,))


_UNSCHEDULED_CADENCES = ("daily", "once_daily", "per_label")


def _occupied_sessions(product: CatalogProduct, usage: str) -> frozenset[str]:
    """The sessions a step may actually be used in, as opposed to the one the
    routine prints.

    `usage` is the engine's instruction; a label whose cadence leaves the
    session open ("daily", "per_label") is the authority the user will follow
    instead. The engine may still *print* PM for such a product -- the retinoid
    pin is an instruction and stays -- but it cannot guarantee compliance
    against the label's own directions, so for conflict purposes the product
    occupies both sessions no matter where it was placed. This is what keeps a
    per-label adapalene out of any routine containing benzoyl peroxide even
    though its printed session is PM and the peroxide's is AM.
    """
    if product.cadence in _UNSCHEDULED_CADENCES:
        return frozenset(("AM", "PM"))
    return _sessions(usage)


def preferred_usage(product: CatalogProduct, slot: str, k: Knowledge) -> tuple[str, bool]:
    """(usage, pinned). Pinned sessions may never flip."""
    if slot == "spf":
        return "AM", True
    actives = set(product.actives)
    # The PM pin is resolved BEFORE the label cadence, because "retinoids
    # PM-only" is a safety rule and a cadence is only a schedule. Read after the
    # cadence lookup, an approved `cadence: "am"` overlay fact would pin a
    # retinoid to the morning, and `cadence: "daily"` would leave its session
    # unknown -- either one silently overriding a pin the rules say may never
    # flip.
    if actives & k.pm_pinned_actives:
        return "PM", True
    verified = {
        "am": "AM", "pm": "PM", "am_pm": "AM_PM", "twice_daily": "AM_PM",
        "daily": "PER_LABEL", "once_daily": "PER_LABEL", "per_label": "PER_LABEL",
    }
    if product.cadence in verified:
        return verified[product.cadence], True
    if slot in ("cleanser", "moisturizer"):
        return "AM_PM", True
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


def _self_conflicts(product: CatalogProduct, k: Knowledge) -> list[tuple[str, str]]:
    """Conflicting pairs carried INSIDE one formulation, e.g. a fixed-dose
    adapalene + benzoyl peroxide gel, or the cosmetic serums that list both
    glycolic acid and retinol.

    The conflict table is a *layering* rule -- it says what not to apply
    together. A manufacturer's own combination is a different claim, and one the
    engine has no evidence for: all it knows is that its INCI parser found both
    names. So such a product is rejected as a candidate (see try_place) rather
    than exempted, and rejected at insertion so the greedy simply picks another
    one instead of losing the whole archetype to a post-hoc veto.
    """
    actives = set(product.actives)
    pairs: list[tuple[str, str]] = []
    for pair in sorted(k.active_conflicts, key=lambda value: sorted(value)):
        if pair <= actives:
            x, y = sorted(pair)
            pairs.append((x, y))
    return pairs


# The documented session rules, in published order. safety_checks reports these
# names verbatim, so the tuple is the contract.
SESSION_RULES = (
    "spf_am_only",
    "retinoids_pm_only",
    "no_conflicting_actives_in_same_session",
    "one_product_per_slot",
)


def session_rule_findings(routine: ComposedRoutine, k: Knowledge) -> list[dict]:
    """Evaluate the documented session rules once, for both of their consumers.

    `validate_routine` turns these into veto reasons and pipeline.run gates on
    that; `explain.safety_checks` turns the same findings into the attestation
    published alongside the routine. They read this one function on purpose.

    When they each implemented the rules separately they disagreed, and the
    disagreement ran both ways: the gate's conflict check ignored `usage`, so it
    threw away routines the composer had correctly split across AM and PM, while
    the two pinned-session rules were checked *only* by the attestation -- which
    nothing gated on. A routine could therefore ship carrying its own signed
    statement that it failed. Every rule below is expressed in the same session
    algebra (`_sessions`) so "unknown session" fails safe uniformly.
    """
    steps = routine.steps
    found: dict[str, list[str]] = {rule: [] for rule in SESSION_RULES}

    for step in steps:
        product_id = step.scored.product.product_id
        # "occupies a session other than the pinned one", not "usage != 'AM'":
        # AM_PM and PER_LABEL both reach into the forbidden session too.
        if step.slot == "spf" and _sessions(step.usage) - {"AM"}:
            found["spf_am_only"].append(f"spf_not_am_only:{product_id}:{step.usage}")
        if set(step.scored.product.actives) & k.retinoids and _sessions(step.usage) - {"PM"}:
            found["retinoids_pm_only"].append(
                f"retinoid_not_pm_only:{product_id}:{step.usage}"
            )

    # The conflict rule reads occupancy, not the printed session: an
    # unscheduled-cadence product may be used in either session whatever the
    # routine says, so it can never be "split away" from a conflicting active.
    # The two pin rules above deliberately keep reading the instruction --
    # widening them to occupancy would veto every per-label retinoid outright.
    for index, first in enumerate(steps):
        for second in steps[index + 1:]:
            if not (
                _occupied_sessions(first.scored.product, first.usage)
                & _occupied_sessions(second.scored.product, second.usage)
            ):
                continue  # split across AM/PM: the rule is honoured, not broken
            conflict = _conflict_between(first.scored.product, second.scored.product, k)
            if conflict:
                found["no_conflicting_actives_in_same_session"].append(
                    "routine_conflict:"
                    f"{first.scored.product.product_id}:"
                    f"{second.scored.product.product_id}:"
                    f"{conflict[0]}:{conflict[1]}"
                )

    if len({step.slot for step in steps}) != len(steps):
        found["one_product_per_slot"].append("more_than_one_product_per_role")

    return [
        {"rule": rule, "passed": not found[rule], "reasons": sorted(set(found[rule]))}
        for rule in SESSION_RULES
    ]


def validate_routine(
    routine: ComposedRoutine,
    profile: Profile,
    k: Knowledge,
    *,
    has_targets: bool,
    strict: bool = True,
    required_slots: set[str] | None = None,
) -> list[str]:
    """Final fail-closed validation over the complete selected regimen.
    `strict` remains for source compatibility; D-029 permits no soft gate."""
    reasons: list[str] = []
    slots_present = {step.slot for step in routine.steps}
    # A moisturizer folded as sun protection (fold_or_add) satisfies the spf
    # requirement — the fold path already held it to the spf-slot gates.
    if any("covers_spf" in step.notes for step in routine.steps):
        slots_present.add("spf")
    required = {"cleanser", "moisturizer", "spf"}
    if has_targets:
        required.add("treatment")
    required.update(required_slots or set())
    for slot in sorted(required - slots_present):
        reasons.append(f"required_role_missing:{slot}")
    for step in routine.steps:
        for reason in profile_gate_reasons(step.scored.product, step.slot, profile, k):
            reasons.append(f"product_ineligible:{step.scored.product.product_id}:{reason}")
        # Backstop only: try_place already refuses these at insertion, so a
        # composed routine never carries one. It stays here because this
        # function is the last fail-closed check over a *complete* regimen and
        # must not assume the composer built it.
        for x, y in _self_conflicts(step.scored.product, k):
            reasons.append(f"self_conflict:{step.scored.product.product_id}:{x}:{y}")
    # The session rules (SPF AM-only, retinoids PM-only, no conflicting pair in
    # a shared session, one product per slot) come from the single helper that
    # explain.safety_checks also reports, so the gate and the published
    # attestation cannot contradict each other.
    for finding in session_rule_findings(routine, k):
        reasons.extend(finding["reasons"])
    for index, first in enumerate(routine.steps):
        for second in routine.steps[index + 1:]:
            duplicates = (
                set(first.scored.product.actives)
                & set(second.scored.product.actives)
                & k.treatment_actives
            )
            for active in sorted(duplicates):
                reasons.append(f"routine_duplicate_active:{active}")
    return sorted(set(reasons))


def try_place(product: CatalogProduct, slot: str, steps: list[Step], k: Knowledge):
    """Find a session for the product that avoids every conflict pair, flipping
    away from the preferred session when allowed. Returns (usage, None) or
    (None, veto_reason)."""
    # A pair conflicting inside the formulation itself travels with the product
    # into every session, so no session assignment can resolve it. Rejecting it
    # here, at the one insertion chokepoint, lets the greedy fall through to the
    # next candidate and still fill the slot; left to validate_routine it would
    # instead drop the entire archetype after the fact, with nothing to recover.
    self_conflicts = _self_conflicts(product, k)
    if self_conflicts:
        x, y = self_conflicts[0]
        return None, f"self_conflicting_actives:{x}:{y}"
    usage, pinned = preferred_usage(product, slot, k)
    options = [usage] if pinned else [usage, "PM" if usage == "AM" else "AM"]
    last_clash = None
    for option in options:
        clash = None
        for step in steps:
            if _occupied_sessions(product, option) & _occupied_sessions(
                step.scored.product, step.usage
            ):
                pair = _conflict_between(product, step.scored.product, k)
                if pair:
                    clash = pair
                    break
        if clash is None:
            return option, None
        last_clash = clash
    return None, f"conflicts_selected_active:{last_clash[0]}:{last_clash[1]}"


def _covered_concerns(
    products: list[CatalogProduct], targets: tuple[TargetLesion, ...], k: Knowledge
) -> set[str]:
    covered: set[str] = set()
    for t in targets:
        wanted = frozenset(t.target_actives) or k.lesion_actives.get(
            t.lesion_type, frozenset()
        )
        if any(set(p.actives) & wanted for p in products):
            covered.add(t.lesion_type)
    return covered


def _filter_candidates(
    scored: list[ScoredCandidate],
    slot: str,
    constraints: dict,
    selected: list[CatalogProduct],
    targets: tuple[TargetLesion, ...],
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
    targets: tuple[TargetLesion, ...],
    profile: Profile,
    k: Knowledge,
    treatment_allowed: bool = True,
) -> ComposedRoutine:
    constraints = archetype.get("constraints") or {}
    routine = ComposedRoutine(archetype=archetype)
    slots = list(archetype["slots"])
    if not treatment_allowed:
        # Only the treatment slot is clinician-gated (D-029). Serums are
        # cosmetic: detected concerns still earn a serum even when therapy
        # is deferred.
        slots = [s for s in slots if s != "treatment"]
    if not targets:
        slots = [s for s in slots if s in CARRIER_ONLY_SLOTS]
        routine.notes.append("clear_skin_maintenance")

    # Fill the treatment slot first: the reviewed therapy is the routine's
    # anchor and support products are fungible. Greedy-filling in display order
    # lets a conflicting cleanser (vitamin C, am_pm — occupies both sessions)
    # land first and block the therapy from every session; the therapy then
    # reads as unfillable when only the cleanser choice was wrong. Display
    # order is unaffected — steps re-sort canonically below.
    fill_order = sorted(slots, key=lambda s: (s != "treatment", slots.index(s)))

    filtered: dict[str, list[ScoredCandidate]] = {}
    for slot in fill_order:
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
        # Folding makes the moisturizer the routine's sun protection, so it must
        # clear the same gates a dedicated sunscreen would (broad spectrum,
        # verified SPF) — an SPF-40 label alone is not sun-safety evidence.
        if moisturizer and (moisturizer.scored.product.spf or 0) >= k.min_spf and not profile_gate_reasons(
            moisturizer.scored.product, "spf", profile, k
        ):
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

    routine.steps.sort(key=lambda s: STEP_ORDER.index(s.slot))
    return routine


def _reduce_to_budget(
    routine: ComposedRoutine,
    filtered: dict[str, list[ScoredCandidate]],
    total_cap: float,
    k: Knowledge,
    protected_ids: frozenset[str] = frozenset(),
) -> None:
    """Swap the most expensive step for the next-cheaper candidate in its slot
    until the total fits (strictly-decreasing prices, so this terminates).

    protected_ids are never chosen as a replacement: _diversify passes the
    product it just swapped away from, so re-running the reducer cannot quietly
    restore it and undo the diversity guarantee.
    """
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
                exclude_ids=protected_ids | {step.scored.product.product_id},
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
    targets: tuple[TargetLesion, ...],
    profile: Profile,
    k: Knowledge,
    treatment_allowed: bool = True,
) -> list[ComposedRoutine]:
    """Compose every archetype in order (best_overall first) and enforce the
    diversity guarantee: each later archetype differs from best_overall by at
    least one product."""
    routines: list[ComposedRoutine] = []
    best_ids: frozenset[str] | None = None
    for archetype, scored_by_slot in archetype_scored:
        routine = compose_archetype(
            archetype, scored_by_slot, targets, profile, k,
            treatment_allowed=treatment_allowed,
        )
        if best_ids is None:
            best_ids = routine.product_ids
        elif routine.steps and routine.product_ids == best_ids:
            _diversify(routine, scored_by_slot, targets, k)
        routines.append(routine)
    return routines


def _diversify(
    routine: ComposedRoutine,
    scored_by_slot: dict[str, list[ScoredCandidate]],
    targets: tuple[TargetLesion, ...],
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
            original_id = step.scored.product.product_id
            routine.steps[routine.steps.index(step)] = replacement
            routine.notes.append(f"diversified_from_best_overall:{step.slot}")
            routine.steps.sort(key=lambda s: STEP_ORDER.index(s.slot))
            # The swap can push the total back over the archetype's cap, and the
            # reducer that enforced it already ran back in compose_archetype.
            # Re-run it, holding out the product we just diversified away from so
            # it cannot return as the cheapest replacement and silently undo the
            # diversity this call exists to create.
            total_cap = constraints.get("max_total_price_usd")
            if total_cap is not None:
                by_slot = {
                    s.slot: _filter_candidates(
                        scored_by_slot.get(s.slot, []), s.slot, constraints,
                        [o.scored.product for o in routine.steps if o is not s],
                        targets, k, routine.notes,
                    )
                    for s in routine.steps
                }
                _reduce_to_budget(
                    routine, by_slot, total_cap, k,
                    protected_ids=frozenset({original_id}),
                )
            return
    routine.notes.append("identical_to_best_overall_no_alternative")
