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
import re
from collections import Counter
from dataclasses import dataclass, replace
from .schema import (
    ConcernReport, Product, Recommendation as V3Recommendation,
    EligibilityDiagnostics, UserProfile, CATEGORIES,
)

# RULES.md §1 — concern -> first-line actives (seed; expand from the doc)
CONCERN_ACTIVES: dict[str, list[str]] = {
    "acne_comedonal":    ["salicylic_acid", "adapalene", "azelaic_acid"],
    "acne_inflammatory": ["benzoyl_peroxide", "azelaic_acid", "niacinamide"],
    "acne_cystic":       ["centella"],  # soothing only; routes to professional
    "acne_scarring":     ["ceramides"],
    "hyperpigmentation": ["azelaic_acid", "niacinamide"],
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

# RULES.md §4 — the soothe (cystic/severe) and maintenance paths must not
# recommend aggressive actives, even bundled inside an otherwise-matching
# product (an SA serum that also lists hyaluronic_acid stays out). Includes
# the PHA/botanical exfoliant sources the importer recognizes: they never
# qualify as first-line treatment, but they disqualify a product from soothe.
STRONG_ACTIVES = (CHEMICAL_EXFOLIANTS | RETINOIDS
                  | {"benzoyl_peroxide", "azelaic_acid", "vitamin_c",
                     "gluconolactone", "willow_bark"})

SLOTS = ("AM", "PM")

# ponytail: name-substring veto for exfoliants the INCI vocabulary can't see
# (citrus-juice "AHA" products etc.); extend the vocabulary if this grows.
_EXFOLIANT_NAME_HINTS = re.compile(
    r"\b(aha|bha|pha)\b|exfoliat|peel|resurfac|microdermabrasion|retinol|retinal"
    r"|clarif"   # "Clarifying Lotion" — high-alcohol exfoliating toners with clean INCI
    r"|scrub|polish")  # physical exfoliants carry clean INCI; only the name tells

# Exfoliant-source actives for the broad-inflammation leave-on cap: the
# classic chemical exfoliants plus the PHA/botanical sources the importer tags.
_EXFOLIANT_SOURCES = CHEMICAL_EXFOLIANTS | {"gluconolactone", "willow_bark"}

# RULES.md §3 extends to products that DECLARE sun protection in their name but
# sit in a non-spf category ("... Cream SPF 30"): photoprotection is AM-only.
_SPF_NAME_HINTS = re.compile(r"\bspf\b|sunscreen|sun screen|broad[- ]spectrum")

# Products named for nighttime use are PM-only. "sleep" as a substring also
# catches portmanteau names ("Sleepair Intensive Mask").
_NIGHT_NAME_HINTS = re.compile(r"overnight|\bnight\b|sleep")


def _gentle_only(catalog: list[Product]) -> list[Product]:
    """RULES.md §4 soothe/maintenance filter: no strong actives, and no
    products marketed as exfoliants even when their INCI parses clean."""
    return [p for p in catalog
            if not set(p.actives) & STRONG_ACTIVES
            and not _EXFOLIANT_NAME_HINTS.search(p.name.lower())]


@dataclass
class Recommendation:
    # slot -> category -> products, ordered
    routines: dict[str, dict[str, list[Product]]]
    slot_assignment: dict[str, set[str]]   # active -> {"AM"} / {"PM"} / {"AM","PM"}
    target_actives: list[str]
    flags: list[str]                       # e.g. "see a professional", "verify"
    # which RULES.md §4 path produced this routine — lets a consumer tell the
    # deliberate soothe/maintenance fallbacks apart from a matching bug.
    mode: str = "treatment"                # "treatment" | "soothe_escalation" | "maintenance"

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


def _concern_location(concern) -> str:
    return ",".join(concern.regions or [concern.region])


def _add_deep_tone_guidance(flags: list[str], profile,
                            concerns: set[str]) -> None:
    if (profile and profile.tone_bucket == "deep"
            and concerns & {"acne_inflammatory", "acne_scarring", "hyperpigmentation"}):
        flags.append("deeper tone: emphasize sunscreen and irritation avoidance to reduce "
                     "post-inflammatory hyperpigmentation risk")


def recommend_legacy(report: ConcernReport, catalog: list[Product],
                     profile: UserProfile | None = None, ranker=None,
                     conf_cutoff: float = 0.5) -> Recommendation:
    flags: list[str] = []
    concerns = [c.concern for c in report.concerns]
    reported_concerns = set(concerns)

    # clear skin -> maintenance only (RULES.md §4, severity 0)
    if report.clear_skin or not report.concerns:
        target = ["ceramides", "hyaluronic_acid"]
        return _finish(target, _gentle_only(catalog), True, profile, ranker,
                       flags + ["maintenance routine"], concerns=concerns,
                       mode="maintenance")

    # cystic / severe -> soothe + escalate, do NOT pile on actives (RULES.md §4)
    if report.has_cystic or report.overall_severity >= 4:
        flags.append("see a dermatologist")
        # RULES.md §5 — loud uncertainty survives the escalation short-circuit
        flags += [f"{c.concern}@{_concern_location(c)}: possible — verify"
                  for c in report.concerns if c.confidence < conf_cutoff]
        _add_deep_tone_guidance(flags, profile, reported_concerns)
        target = ["centella", "ceramides", "hyaluronic_acid"]
        return _finish(target, _gentle_only(catalog), True, profile, ranker,
                       flags, concerns=concerns, mode="soothe_escalation")

    # Active acne is handled before scar support, regardless of report order.
    concern_order = {"acne_inflammatory": 0, "acne_comedonal": 1,
                     "hyperpigmentation": 2, "dryness": 3, "acne_scarring": 4}
    ordered_concerns = sorted(enumerate(report.concerns),
                              key=lambda item: (concern_order.get(item[1].concern, 5), item[0]))

    target: list[str] = []
    needs_spf = False
    broad_inflammation = False
    for _, c in ordered_concerns:
        if c.concern in {"hyperpigmentation", "acne_scarring"}:
            needs_spf = True  # supportive SPF remains valid even under uncertainty
        if c.confidence < conf_cutoff:
            flags.append(f"{c.concern}@{_concern_location(c)}: possible — verify")
            continue
        actives = list(CONCERN_ACTIVES.get(c.concern, []))
        if c.concern == "acne_inflammatory" and c.evidence.affected_region_count >= 3:
            broad_inflammation = True
        for a in actives:
            if a not in target:
                target.append(a)
        if (c.concern == "acne_scarring"
                and (c.severity >= 3 or c.evidence.labels.get("hypertrophic_scar", 0))):
            flags.append("consider professional review for acne scarring")

    if any(a in STRONG_ACTIVES for a in target) and "ceramides" not in target:
        target.append("ceramides")

    _add_deep_tone_guidance(flags, profile, reported_concerns)

    if report.overall_severity == 3:
        flags.append("consider a professional")

    # RULES.md §2 — pregnancy/nursing: strip retinoids BEFORE conflict resolution.
    if profile and profile.pregnant_or_nursing:
        if any(a in RETINOIDS for a in target):
            flags.append("retinoids omitted (pregnancy/nursing) — cosmetic "
                         "guidance only, confirm with your doctor")
        target = [a for a in target if a not in RETINOIDS]

    # Decide broad-inflammation de-stacking against the fully assembled targets
    # through the same slot, product, and tier selection used for the final routine.
    if broad_inflammation:
        # e2e 2026-07-13 (run 262): the de-stack promise must gate the shown
        # PRODUCTS too — peels/scrubs/resurfacing formats are excluded and
        # leave-on exfoliant carriers capped inside _build_routines.
        flags.append("broad inflammation: exfoliating formats excluded")
    if broad_inflammation and "benzoyl_peroxide" in target and "azelaic_acid" in target:
        probe_flags: list[str] = []
        probe_kept, probe_slots = _assign_slots(target, probe_flags)
        probe_routines = _build_routines(probe_kept, catalog, needs_spf, probe_slots,
                                         profile, ranker, concerns,
                                         reduced_stacking=True)
        # SPF products are added to every routine unconditionally (RULES.md §3)
        # without matching against target actives, so a sunscreen that merely
        # lists azelaic_acid must not count as a surviving azelaic TREATMENT —
        # only actives-matched (non-SPF) slots prove azelaic_acid survived.
        if any("azelaic_acid" in product.actives
               for slot in SLOTS for category in CATEGORIES if category != "spf"
               for product in probe_routines[slot][category]):
            target.remove("benzoyl_peroxide")
            flags.append("broad inflammation: reduced strong-active stacking")

    kept, slots = _assign_slots(target, flags)
    return _finish(kept, catalog, needs_spf, profile, ranker, flags, slots,
                   concerns=concerns, reduced_stacking=broad_inflammation)


def recommend(
    report: ConcernReport,
    catalog: list[Product],
    profile: UserProfile | None = None,
    *,
    triage_policy=None,
    therapy_policy=None,
    concern_scorer=None,
    pooled_ranker=None,
    # Historical adapter inputs. New callers use pooled_ranker and both
    # explicit policies; no-policy calls are isolated to recommend_legacy.
    ranker=None,
    conf_cutoff: float = 0.5,
    collect_eligibility_details: bool = False,
) -> V3Recommendation | Recommendation:
    """Run v3 or explicitly isolate a historical v2 caller.

    Supplying neither policy selects the legacy adapter for repository callers
    that still consume category menus. Supplying one policy without the other
    is an error; the v3 path never constructs a silent default profile.
    """
    if triage_policy is None and therapy_policy is None:
        result = recommend_legacy(report, catalog, profile, ranker, conf_cutoff)
        # Machine-visible without changing historical serialized flags.
        result.legacy = True
        return result
    if triage_policy is None or therapy_policy is None:
        raise ValueError("v3 recommend requires both triage_policy and therapy_policy")
    if profile is None:
        raise ValueError("v3 recommend requires an explicit UserProfile")

    from .composer import compose_regimen, rank_equivalents
    from .decision import decide_care
    from .eligibility import check_eligibility
    from .therapy import plan_therapy

    decision = decide_care(report, triage_policy)
    therapy_plan = plan_therapy(decision, report, profile, therapy_policy)
    if decision.therapy_disposition == "active_treatment" and therapy_plan.primary is None:
        decision = replace(decision, therapy_disposition="defer")

    requested = list(therapy_plan.support_roles)
    if therapy_plan.primary is not None:
        requested.append(therapy_plan.primary.role)
    role_order = ("cleanser", "treatment", "moisturizer", "sunscreen")
    requested = [role for role in role_order if role in requested]

    diagnostics = EligibilityDiagnostics(requested, collect_eligibility_details)
    ranked_by_role: dict[str, list[Product]] = {}
    # Choose therapeutic intent first. Support products are filtered around an
    # available primary treatment before any scorer sees them.
    selection_order = [role for role in ("treatment", "cleanser", "moisturizer", "sunscreen")
                       if role in requested]
    selected_context: dict[str, Product] = {}
    for role in selection_order:
        therapy = therapy_plan.primary if role == "treatment" else None
        context_eligible: list[Product] = []
        for product in catalog:
            result = check_eligibility(product, role, therapy, profile, selected_context)
            if result.eligible:
                context_eligible.append(product)
            else:
                diagnostics.record(role, product.product_id, result.reasons)
            if result.eligible:
                diagnostics.record(role, product.product_id, [])
        ranked, _ = rank_equivalents(
            context_eligible,
            profile,
            concern_scorer=concern_scorer,
            pooled_ranker=pooled_ranker,
        )
        ranked_by_role[role] = ranked
        if ranked:
            selected_context[role] = ranked[0]

    if therapy_plan.primary is not None and "treatment" not in selected_context:
        decision = replace(decision, therapy_disposition="defer")
        # A deferred disposition must not carry a primary therapy: downstream
        # consumers (recsys contracts) reject the contradictory pair.
        therapy_plan = replace(
            therapy_plan,
            course_weeks=None,
            review_at_weeks=None,
            primary=None,
            alternatives=[],
            deferred_reasons=[*therapy_plan.deferred_reasons,
                              "no_eligible_treatment_product"],
        )

    # Alternatives must be valid replacements alongside every other selected
    # role. Keep the chosen product first so composition is stable.
    eligible_by_role: dict[str, list[Product]] = {}
    for role in requested:
        therapy = therapy_plan.primary if role == "treatment" else None
        other_selected = {
            key: value for key, value in selected_context.items() if key != role
        }
        safe = []
        for product in ranked_by_role.get(role, []):
            result = check_eligibility(product, role, therapy, profile, other_selected)
            if result.eligible:
                safe.append(product)
            else:
                diagnostics.reject_previously_eligible(
                    role, product.product_id, result.reasons
                )
        chosen = selected_context.get(role)
        if chosen in safe:
            safe.remove(chosen)
            safe.insert(0, chosen)
        eligible_by_role[role] = safe

    recommendation = compose_regimen(
        decision,
        therapy_plan,
        eligible_by_role,
        profile,
        eligibility_diagnostics=diagnostics,
        concern_scorer=concern_scorer,
        pooled_ranker=pooled_ranker,
    )
    recommendation.flags.extend([
        "release_blocked:clinician_policy_approval",
        "release_blocked:adequate_calibration_cohort",
        "release_blocked:external_clinical_review_set",
        "release_blocked:verified_real_catalog_overlay",
        "release_blocked:remote_detector_identity",
    ])
    return recommendation


def _finish(target: list[str], catalog: list[Product], always_spf: bool,
            profile, ranker, flags: list[str],
            slots: dict[str, set[str]] | None = None,
            concerns: list[str] = (), mode: str = "treatment",
            reduced_stacking: bool = False) -> Recommendation:
    if slots is None:  # paths that skip conflict resolution: every active both slots
        slots = {a: {"AM", "PM"} for a in target}
    routines = _build_routines(target, catalog, always_spf, slots, profile,
                               ranker, concerns, mode=mode,
                               reduced_stacking=reduced_stacking)
    return Recommendation(routines, slots, target, flags, mode)


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
                    profile, ranker, concerns: list[str] = (),
                    mode: str = "treatment",
                    reduced_stacking: bool = False) -> dict[str, dict[str, list[Product]]]:
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
        # Broad inflammation (reduced stacking): exclude products whose NAME
        # declares an exfoliating format (peel/scrub/resurfacing/...) — their
        # matched active may be gentle, the delivery format is not.
        if reduced_stacking and _EXFOLIANT_NAME_HINTS.search(p.name.lower()):
            continue
        # RULES.md §2a — the product must satisfy EVERY matched active's slot
        # pins (a PM-pinned retinoid keeps its product out of AM even when a
        # second matched active is AM-eligible). Empty intersection -> the
        # product can't be placed without violating a split; skip it.
        allowed = set.intersection(*(slots[a] for a in matched))
        # §2 photosensitivity is a property of what the product CARRIES, not of
        # why it matched: any retinoid on board pins it to PM even when the
        # match came through another active (niacinamide, ceramides).
        if set(p.actives) & RETINOIDS:
            allowed = allowed & {"PM"}
        name = p.name.lower()
        if _SPF_NAME_HINTS.search(name):        # SPF wins if a name says both
            allowed = allowed & {"AM"}
        elif _NIGHT_NAME_HINTS.search(name):
            allowed = allowed & {"PM"}
        for slot in SLOTS:
            if slot in allowed:
                routines[slot][p.category].append(p)

    def match_tiebreak(p: Product) -> float:
        if not concerns or not p.ingredient_match:
            return 0.0
        return max((p.ingredient_match.get(c, 0.0) for c in concerns), default=0.0)

    # HA/glycerin sit in nearly every product, so matching on them barely
    # filters; a product matching a RARE target (centella) is a far stronger
    # signal it delivers what the routine targets. Specificity = the catalog
    # count of the rarest matched target — lower is more distinctive. Gentle
    # paths ONLY: there the ranker is concern-blind popularity, so signature
    # actives must not be crowded out. On the treatment path the ranker's
    # per-profile ordering is the trained signal (D-005) and rarity would
    # arbitrarily invert first-line vs support actives — specificity stays out.
    active_counts = Counter(a for p in catalog for a in p.actives)
    def specificity(p: Product) -> int:
        if mode == "treatment":
            return 0
        matched = set(p.actives) & tset
        return min((active_counts[a] for a in matched), default=len(catalog) + 1)

    def sort_key(p: Product):
        # comedogenic partition ALWAYS dominates; target specificity next; the
        # ranker (concern-stats) breaks ties; ingredient-match is last.
        if ranker is not None:
            return (len(p.comedogenic_flags), specificity(p),
                    -ranker.score(p, profile), -match_tiebreak(p))
        return (len(p.comedogenic_flags), specificity(p), -match_tiebreak(p))

    for slot in SLOTS:
        for c in CATEGORIES:
            candidates = routines[slot][c]
            tier1 = [p for p in candidates if p.tier == 1]
            chosen = tier1 if tier1 else candidates  # tier-2 fills empty slots only
            chosen.sort(key=sort_key)
            routines[slot][c] = chosen

    # Broad inflammation: cap leave-on exfoliant carriers at ONE per slot,
    # swept in application order — cleansers are rinse-off and exempt, SPF
    # carries no exfoliants by vocabulary.
    if reduced_stacking:
        for slot in SLOTS:
            carriers_kept = 0
            for c in ("treatment", "serum", "moisturizer"):
                kept = []
                for p in routines[slot][c]:
                    if set(p.actives) & _EXFOLIANT_SOURCES:
                        carriers_kept += 1
                        if carriers_kept > 1:
                            continue
                    kept.append(p)
                routines[slot][c] = kept
    return routines
