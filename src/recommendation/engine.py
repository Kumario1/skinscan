"""Stage 3 recommender — rules over ingredients, then a catalog lookup.

This is intentionally buildable and testable with ZERO ML (D-007). Feed it a
hand-written ConcernReport, get a routine back. The concern->ingredient rules
below are a code mirror of RULES.md §1; keep the two in sync (or later, load
both from one YAML).

NOTE: this is a working skeleton, not the finished rules. Values are seeded from
RULES.md but the interaction/severity logic is deliberately minimal so we can
grow it test-first.
"""
from __future__ import annotations
from dataclasses import dataclass
from .schema import ConcernReport, Product, CATEGORIES

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


@dataclass
class Recommendation:
    routine: dict[str, list[Product]]   # category -> products, ordered
    target_actives: list[str]
    flags: list[str]                    # e.g. "see a professional", "verify"
    def ordered_steps(self):
        return [(c, self.routine[c]) for c in CATEGORIES if self.routine.get(c)]


def recommend(report: ConcernReport, catalog: list[Product],
              conf_cutoff: float = 0.5) -> Recommendation:
    flags: list[str] = []

    # clear skin -> maintenance only (RULES.md §4, severity 0)
    if report.clear_skin or not report.concerns:
        target = ["ceramides", "hyaluronic_acid"]
        routine = _build_routine(target, catalog, always_spf=True)
        return Recommendation(routine, target, ["maintenance routine"])

    # cystic / severe -> soothe + escalate, do NOT pile on actives (RULES.md §4)
    if report.has_cystic or report.overall_severity >= 4:
        flags.append("see a dermatologist")
        target = ["centella", "ceramides", "hyaluronic_acid"]
        routine = _build_routine(target, catalog, always_spf=True)
        return Recommendation(routine, target, flags)

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

    target = _resolve_conflicts(target, flags)
    routine = _build_routine(target, catalog, always_spf=needs_spf)
    return Recommendation(routine, target, flags)


def _resolve_conflicts(actives: list[str], flags: list[str]) -> list[str]:
    """RULES.md §2 — drop the later member of any incompatible pair (v1: simple
    priority = order of appearance). A real impl would split into AM/PM."""
    kept: list[str] = []
    for a in actives:
        clash = any({a, k} in INCOMPATIBLE for k in kept)
        if clash:
            flags.append(f"{a}: held back (conflicts with earlier active)")
            continue
        kept.append(a)
    return kept


def _build_routine(target_actives: list[str], catalog: list[Product],
                   always_spf: bool) -> dict[str, list[Product]]:
    """Ingredient -> product lookup (D-006). For each category, pick products
    whose actives intersect the target, down-ranking comedogenic ones."""
    routine: dict[str, list[Product]] = {c: [] for c in CATEGORIES}
    for p in catalog:
        if p.category == "spf":
            if always_spf:
                routine["spf"].append(p)
            continue
        if set(p.actives) & set(target_actives):
            routine[p.category].append(p)
    # down-rank comedogenic (RULES.md §6): stable sort, flagged last
    for c in CATEGORIES:
        routine[c].sort(key=lambda p: len(p.comedogenic_flags))
    return routine
