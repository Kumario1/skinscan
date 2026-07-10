"""Stage 3 recommender — rules over ingredients, then a catalog lookup.

This is intentionally buildable and testable with ZERO ML (D-007). Feed it a
hand-written ConcernReport, get AM/PM routines back. The concern->ingredient
rules below are a code mirror of RULES.md §1-2; keep the two in sync (or later,
load both from one YAML).

Engine v2 (issue #7): optional UserProfile (D-021), pregnancy rule, AM/PM slot
routines from interaction constraints, and an optional duck-typed ranker hook
(D-005). The engine still imports NO ML — the ranker only REORDERS rule-approved
candidates and can never add/remove products or touch flags. ranker=None ->
today's stable rules-only order exactly (D-019).
"""
from __future__ import annotations
from dataclasses import dataclass
from .schema import ConcernReport, Product, UserProfile, CATEGORIES

# RULES.md §1 — concern -> first-line actives (seed; expand from the doc)
CONCERN_ACTIVES: dict[str, list[str]] = {
    "acne_comedonal":    ["salicylic_acid", "adapalene", "azelaic_acid"],
    "acne_inflammatory": ["benzoyl_peroxide", "azelaic_acid", "niacinamide"],
    "acne_cystic":       ["centella"],  # soothing only; routes to professional
    "hyperpigmentation": ["niacinamide", "vitamin_c", "azelaic_acid"],
    "dryness":           ["ceramides", "hyaluronic_acid", "glycerin"],
}

# RULES.md §2 — pairs that must not share a routine step
INCOMPATIBLE = [
    {"benzoyl_peroxide", "retinol"},
    {"benzoyl_peroxide", "adapalene"},
    {"benzoyl_peroxide", "vitamin_c"},
    {"glycolic_acid", "retinol"},
]

# RULES.md §2 — retinoids are photosensitising: pin to PM.
RETINOIDS = {"retinol", "adapalene"}

# RULES.md §2 — preferred slot when an INCOMPATIBLE pair still shares a slot.
PREFERRED_SLOT = {
    "benzoyl_peroxide": "AM",
    "vitamin_c":        "AM",
    "glycolic_acid":    "PM",
    "lactic_acid":      "PM",
    "mandelic_acid":    "PM",
}

# RULES.md §2 — cap at one primary chemical exfoliant per routine.
CHEMICAL_EXFOLIANTS = {"glycolic_acid", "lactic_acid", "mandelic_acid", "salicylic_acid"}

SLOTS = ("AM", "PM")


@dataclass
class Recommendation:
    # slot -> category -> products, ordered
    routines: dict[str, dict[str, list[Product]]]
    slot_assignment: dict[str, set[str]]   # active -> {"AM"} / {"PM"} / {"AM","PM"}
    target_actives: list[str]
    flags: list[str]                       # e.g. "see a professional", "verify"

    @property
    def routine(self) -> dict[str, list[Product]]:
        """Backward-compat union of the AM+PM routines (order-preserving dedup)."""
        merged: dict[str, list[Product]] = {c: [] for c in CATEGORIES}
        for slot in SLOTS:
            for c in CATEGORIES:
                for p in self.routines[slot].get(c, []):
                    if p not in merged[c]:
                        merged[c].append(p)
        return merged

    def ordered_steps(self, slot: str | None = None):
        r = self.routines[slot] if slot else self.routine
        return [(c, r[c]) for c in CATEGORIES if r.get(c)]


def recommend(report: ConcernReport, catalog: list[Product],
              profile: UserProfile | None = None, ranker=None,
              conf_cutoff: float = 0.5) -> Recommendation:
    flags: list[str] = []
    concerns = [c.concern for c in report.concerns]

    # clear skin -> maintenance only (RULES.md §4, severity 0)
    if report.clear_skin or not report.concerns:
        target = ["ceramides", "hyaluronic_acid"]
        return _finish(target, catalog, True, profile, ranker,
                       flags + ["maintenance routine"], concerns=concerns)

    # cystic / severe -> soothe + escalate, do NOT pile on actives (RULES.md §4)
    if report.has_cystic or report.overall_severity >= 4:
        flags.append("see a dermatologist")
        target = ["centella", "ceramides", "hyaluronic_acid"]
        return _finish(target, catalog, True, profile, ranker, flags,
                       concerns=concerns)

    # collect actives from all sufficiently-confident concerns
    target: list[str] = []
    needs_spf = False
    for c in report.concerns:
        if c.concern == "hyperpigmentation":
            needs_spf = True  # RULES.md §3, non-negotiable
        actives = CONCERN_ACTIVES.get(c.concern, [])
        if c.confidence < conf_cutoff:
            flags.append(f"{c.concern}@{c.region}: possible — verify")
        for a in actives:
            if a not in target:
                target.append(a)

    if report.overall_severity == 3:
        flags.append("consider a professional")

    # RULES.md §2 — pregnancy/nursing: strip retinoids BEFORE conflict resolution.
    if profile and profile.pregnant_or_nursing:
        if any(a in RETINOIDS for a in target):
            flags.append("retinoids omitted (pregnancy/nursing) — cosmetic "
                         "guidance only, confirm with your doctor")
        target = [a for a in target if a not in RETINOIDS]

    kept, slots = _assign_slots(target, flags)
    return _finish(kept, catalog, needs_spf, profile, ranker, flags, slots,
                   concerns=concerns)


def _finish(target: list[str], catalog: list[Product], always_spf: bool,
            profile, ranker, flags: list[str],
            slots: dict[str, set[str]] | None = None,
            concerns: list[str] = ()) -> Recommendation:
    if slots is None:  # paths that skip conflict resolution: every active both slots
        slots = {a: {"AM", "PM"} for a in target}
    routines = _build_routines(target, catalog, always_spf, slots, profile,
                               ranker, concerns)
    return Recommendation(routines, slots, target, flags)


def _assign_slots(actives: list[str],
                  flags: list[str]) -> tuple[list[str], dict[str, set[str]]]:
    """RULES.md §2 — turn incompatible pairs into an AM/PM split.

    Every active defaults to both slots; retinoids pin to PM (photosensitivity);
    a second chemical exfoliant is dropped. For each INCOMPATIBLE pair still
    sharing a slot we shrink deterministically: if one member is already pinned
    to a single slot, the other takes the complement; if both are free, the
    LATER-listed active takes its preferred slot and the earlier takes the
    complement. Only a pair that still shares a slot afterwards falls back to
    dropping the later active with the legacy "held back" flag.
    """
    kept: list[str] = []
    slots: dict[str, set[str]] = {}
    exfoliant_seen = False
    for a in actives:
        if a in CHEMICAL_EXFOLIANTS:
            if exfoliant_seen:
                flags.append(f"{a}: held back (one chemical exfoliant per routine)")
                continue
            exfoliant_seen = True
        slots[a] = {"PM"} if a in RETINOIDS else {"AM", "PM"}
        kept.append(a)

    dropped: set[str] = set()
    for i, a in enumerate(kept):
        if a in dropped:
            continue
        for b in kept[i + 1:]:
            if b in dropped or {a, b} not in INCOMPATIBLE:
                continue
            if not (slots[a] & slots[b]):
                continue  # already separated (e.g. a retinoid pinned to PM)
            if not _split(a, b, slots):
                dropped.add(b)
                flags.append(f"{b}: held back (conflicts with earlier active)")

    kept = [a for a in kept if a not in dropped]
    return kept, {a: slots[a] for a in kept}


def _complement(slot: str) -> str:
    return "PM" if slot == "AM" else "AM"


def _split(a: str, b: str, slots: dict[str, set[str]]) -> bool:
    """Shrink one/both of an incompatible pair so they no longer share a slot.
    Returns False only when both are already pinned to the same single slot."""
    sa, sb = slots[a], slots[b]
    if len(sa) == 1 and len(sb) == 2:
        slots[b] = {_complement(next(iter(sa)))}
        return True
    if len(sb) == 1 and len(sa) == 2:
        slots[a] = {_complement(next(iter(sb)))}
        return True
    if len(sa) == 2 and len(sb) == 2:
        # both free -> later-listed active (b) claims its preferred slot.
        pref = PREFERRED_SLOT.get(b) or _complement(PREFERRED_SLOT.get(a, "PM"))
        slots[b] = {pref}
        slots[a] = {_complement(pref)}
        return True
    return sa != sb  # both single: only resolvable if they already differ


def _build_routines(target_actives: list[str], catalog: list[Product],
                    always_spf: bool, slots: dict[str, set[str]],
                    profile, ranker,
                    concerns: list[str] = ()) -> dict[str, dict[str, list[Product]]]:
    """Ingredient -> product lookup (D-006), per slot. A product lands in each
    slot where at least one of its matched target actives is assigned; SPF is
    pinned AM-only. Comedogenic products are down-ranked per slot (RULES.md §6);
    the optional ranker only reorders within that comedogenic partition (D-005).

    Tier-2 fallback (spec 2026-07-10-ingredient-kb): a slot x category is filled
    from review-backed tier-1 candidates; only when NONE exist do tier-2
    products (no_outcome_data=True) fill it. The ingredient-match score is a
    pure tiebreaker under the ranker — review-backed concern-stats dominate; the
    match score only orders products with equal/absent ranker scores.
    """
    routines = {s: {c: [] for c in CATEGORIES} for s in SLOTS}
    tset = set(target_actives)
    for p in catalog:
        if p.category == "spf":
            if always_spf:
                routines["AM"]["spf"].append(p)  # RULES.md §3 — SPF is AM only
            continue
        matched = set(p.actives) & tset
        if not matched:
            continue
        for slot in SLOTS:
            if any(slot in slots[a] for a in matched):
                routines[slot][p.category].append(p)

    def match_tiebreak(p: Product) -> float:
        if not concerns or not p.ingredient_match:
            return 0.0
        return max((p.ingredient_match.get(c, 0.0) for c in concerns), default=0.0)

    def sort_key(p: Product):
        # comedogenic partition ALWAYS dominates; the ranker (concern-stats)
        # breaks ties next; ingredient-match is the final tiebreaker only.
        if ranker is not None:
            return (len(p.comedogenic_flags), -ranker.score(p, profile),
                    -match_tiebreak(p))
        return (len(p.comedogenic_flags), -match_tiebreak(p))

    for slot in SLOTS:
        for c in CATEGORIES:
            candidates = routines[slot][c]
            tier1 = [p for p in candidates if p.tier == 1]
            chosen = tier1 if tier1 else candidates  # tier-2 fills empty slots only
            chosen.sort(key=sort_key)
            routines[slot][c] = chosen
    return routines
