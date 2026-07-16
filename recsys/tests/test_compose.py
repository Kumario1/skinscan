from pathlib import Path

from recsys.catalog import CatalogProduct
from recsys.compose import (
    ComposedRoutine,
    Step,
    _sessions,
    compose_all,
    compose_archetype,
    preferred_usage,
    try_place,
    validate_routine,
)
from recsys.contracts import Profile
from recsys.explain import safety_checks
from recsys.knowledge import load_knowledge
from recsys.scoring import ScoredCandidate
from recsys.signals import TargetConcern

K = load_knowledge(Path(__file__).parents[1] / "data" / "knowledge")

PROFILE = Profile(pregnancy_status="not_pregnant")
TARGETS = (TargetConcern("inflammatory_acne", 2, 0.9),)


def product(pid, category, actives=(), *, price_usd=10.0, cadence=None, spf=None):
    return CatalogProduct(
        product_id=pid, name=pid, brand="b", category=category,
        price_usd=price_usd, size=None, format=None, spf=spf, spf_source=None,
        inci=(), inci_sha256="", actives=tuple(actives), cadence=cadence,
    )


def step(prod, slot, usage):
    return Step(slot, ScoredCandidate(prod, 0.5, ()), usage)


def scored(prod, score=0.5):
    return ScoredCandidate(prod, score, ())


def routine_of(*steps, archetype_id="test"):
    return ComposedRoutine(archetype={"id": archetype_id}, steps=list(steps))


def carrier_steps():
    """A cleanser/moisturizer/SPF spine that draws no HARD gate reason, so a
    hybrid validate_routine reports only what the test is actually about."""
    return [
        step(product("cl", "cleanser"), "cleanser", "AM_PM"),
        step(product("mo", "moisturizer"), "moisturizer", "AM_PM"),
        step(product("sp", "spf", spf=50), "spf", "AM"),
    ]


def hybrid_reasons(routine, *, has_targets=True):
    return validate_routine(routine, PROFILE, K, has_targets=has_targets, strict=False)


def test_pinned_sessions():
    assert preferred_usage(product("p1", "spf"), "spf", K) == ("AM", True)
    assert preferred_usage(product("p2", "treatment", ["retinol"]), "treatment", K) == ("PM", True)
    assert preferred_usage(product("p3", "cleanser"), "cleanser", K) == ("AM_PM", True)


def test_session_preferences():
    assert preferred_usage(product("p1", "serum", ["vitamin_c"]), "serum", K)[0] == "AM"
    assert preferred_usage(product("p2", "serum", ["glycolic_acid"]), "serum", K)[0] == "PM"
    # slot defaults spread actives across the day
    assert preferred_usage(product("p3", "treatment", ["azelaic_acid"]), "treatment", K)[0] == "AM"
    assert preferred_usage(product("p4", "serum", ["niacinamide"]), "serum", K)[0] == "PM"


def test_conflict_splits_across_sessions():
    retinol_pm = step(product("p1", "treatment", ["retinol"]), "treatment", "PM")
    bp = product("p2", "serum", ["benzoyl_peroxide"])
    usage, reason = try_place(bp, "serum", [retinol_pm], K)
    assert reason is None
    assert usage == "AM"  # BP prefers AM anyway; never shares PM with the retinoid


def test_conflict_with_both_sessions_is_vetoed():
    bp_cleanser = step(product("p1", "cleanser", ["benzoyl_peroxide"]), "cleanser", "AM_PM")
    retinol = product("p2", "treatment", ["retinol"])
    usage, reason = try_place(retinol, "treatment", [bp_cleanser], K)
    assert usage is None
    # candidate's active first, then the already-selected product's active
    assert reason == "conflicts_selected_active:retinol:benzoyl_peroxide"


def test_vitamin_c_flips_when_am_conflicts():
    bp_am = step(product("p1", "treatment", ["benzoyl_peroxide"]), "treatment", "AM")
    vitc = product("p2", "serum", ["vitamin_c"])
    usage, reason = try_place(vitc, "serum", [bp_am], K)
    assert reason is None
    assert usage == "PM"  # AM preference flips away from the BP conflict


def test_whole_routine_validator_rejects_self_conflicts_and_missing_roles():
    conflicting = product(
        "p1", "treatment", ["retinol", "benzoyl_peroxide"]
    )
    routine = ComposedRoutine(
        archetype={"id": "test"},
        steps=[step(conflicting, "treatment", "PM")],
    )

    reasons = validate_routine(
        routine, Profile(pregnancy_status="not_pregnant"), K, has_targets=True
    )

    assert "self_conflict:p1:benzoyl_peroxide:retinol" in reasons
    assert "required_role_missing:cleanser" in reasons
    assert "required_role_missing:moisturizer" in reasons
    assert "required_role_missing:spf" in reasons


# --------------------------------------------------------------------------
# PER_LABEL means "unknown session", which must fail SAFE
# --------------------------------------------------------------------------

def test_per_label_occupies_every_session_because_its_schedule_is_unknown():
    """PER_LABEL defers the schedule to the label, so the engine does not know
    which sessions the product lands in. An unknown session has to conflict with
    everything; as the empty set it conflicted with nothing, which inverted the
    "unknowns fail safe" invariant the rest of the engine is built on."""
    assert _sessions("PER_LABEL") == frozenset(("AM", "PM"))
    assert _sessions("PER_LABEL") & _sessions("AM")
    assert _sessions("PER_LABEL") & _sessions("PM")
    assert _sessions("PER_LABEL") & _sessions("AM_PM")
    assert _sessions("PER_LABEL") & _sessions("PER_LABEL")


def test_per_label_candidate_is_vetoed_against_a_conflicting_selected_active():
    """A per-label benzoyl peroxide cannot be waved past a selected retinoid:
    its session is unknown, so it may well land in the retinoid's."""
    retinol_pm = step(product("ret", "serum", ["retinol"]), "serum", "PM")
    bp_per_label = product("bp", "treatment", ["benzoyl_peroxide"], cadence="per_label")

    assert preferred_usage(bp_per_label, "treatment", K) == ("PER_LABEL", True)
    usage, reason = try_place(bp_per_label, "treatment", [retinol_pm], K)

    assert usage is None
    assert reason == "conflicts_selected_active:benzoyl_peroxide:retinol"


def test_selected_per_label_product_vetoes_a_later_conflicting_candidate():
    """The blindness ran both ways: nothing was ever checked against an
    already-selected PER_LABEL step either."""
    bp_per_label = step(
        product("bp", "treatment", ["benzoyl_peroxide"], cadence="per_label"),
        "treatment", "PER_LABEL",
    )
    vitc = product("vc", "serum", ["vitamin_c"])

    usage, reason = try_place(vitc, "serum", [bp_per_label], K)

    assert usage is None  # vitamin C conflicts with BP and has no session to flip to
    assert reason == "conflicts_selected_active:vitamin_c:benzoyl_peroxide"


# --------------------------------------------------------------------------
# The composer's AM/PM split is a feature, and the validator must honour it
# --------------------------------------------------------------------------

def test_validator_accepts_a_conflicting_pair_the_composer_split_across_sessions():
    """Splitting BP into AM and a retinoid into PM is the documented remedy for a
    conflict pair, and the comprehensive archetype promises exactly it. The
    validator used to ignore `usage` and reject the very routines the composer
    had correctly built, so the feature was dead code failing as a silently
    missing archetype."""
    routine = routine_of(
        *carrier_steps(),
        step(product("bp-treat", "treatment", ["benzoyl_peroxide"]), "treatment", "AM"),
        step(product("ret-serum", "serum", ["retinol"]), "serum", "PM"),
    )

    assert _sessions("AM") & _sessions("PM") == frozenset()
    assert hybrid_reasons(routine) == []
    assert all(check["passed"] for check in safety_checks(routine, K))


def test_validator_still_rejects_a_conflicting_pair_inside_one_session():
    """The session-aware rule must not become a blanket exemption: a genuine
    same-session conflict is still a veto."""
    routine = routine_of(
        *carrier_steps(),
        step(product("bp-treat", "treatment", ["benzoyl_peroxide"]), "treatment", "AM"),
        step(product("vc-serum", "serum", ["vitamin_c"]), "serum", "AM"),
    )

    assert "routine_conflict:bp-treat:vc-serum:benzoyl_peroxide:vitamin_c" in hybrid_reasons(routine)


def test_validator_rejects_a_conflict_hidden_behind_an_am_pm_carrier():
    """AM_PM straddles both sessions, so it shares one with every step."""
    routine = routine_of(
        step(product("cl", "cleanser", ["benzoyl_peroxide"]), "cleanser", "AM_PM"),
        step(product("mo", "moisturizer"), "moisturizer", "AM_PM"),
        step(product("sp", "spf", spf=50), "spf", "AM"),
        step(product("ret", "serum", ["retinol"]), "serum", "PM"),
    )

    assert "routine_conflict:cl:ret:benzoyl_peroxide:retinol" in hybrid_reasons(routine)


# --------------------------------------------------------------------------
# THE INTERLOCK: the session-blind validator was the only thing catching the
# PER_LABEL hole. Making the validator session-aware without also fixing
# _sessions would have opened it.
# --------------------------------------------------------------------------

def test_per_label_retinoid_never_shares_a_routine_with_benzoyl_peroxide():
    """The interlock, end to end.

    A PER_LABEL adapalene beside benzoyl peroxide is the reachable hazard: the
    drug catalog's one OTC row is a fixed-dose adapalene + BP gel, fully
    eligible in hybrid + not_pregnant. It used to survive `try_place` (PER_LABEL
    intersected nothing) and be caught only by the session-BLIND validator --
    the accidental backstop that fixing the validator alone would have removed.

    Both the placement path and the final validator must reject it, so neither
    is load-bearing on its own.
    """
    bp = product("bp", "serum", ["benzoyl_peroxide"])
    adapalene_per_label = product(
        "ada", "treatment", ["adapalene"], cadence="per_label"
    )

    # 1. the placement path refuses it, in both insertion orders
    assert try_place(
        adapalene_per_label, "treatment", [step(bp, "serum", "AM")], K
    ) == (None, "conflicts_selected_active:adapalene:benzoyl_peroxide")
    assert try_place(
        bp, "serum", [step(adapalene_per_label, "treatment", "PER_LABEL")], K
    ) == (None, "conflicts_selected_active:benzoyl_peroxide:adapalene")

    # 2. and the validator refuses it too, if one is ever assembled anyway
    routine = routine_of(
        *carrier_steps(),
        step(adapalene_per_label, "treatment", "PER_LABEL"),
        step(bp, "serum", "AM"),
    )
    reasons = hybrid_reasons(routine)
    assert "routine_conflict:ada:bp:adapalene:benzoyl_peroxide" in reasons
    assert not all(check["passed"] for check in safety_checks(routine, K))


def test_fixed_dose_combination_product_draws_a_veto_rather_than_placing():
    """The real Differin Epiduo row: adapalene + benzoyl peroxide in one gel,
    cadence per_label, otc_drug so it is never dropped as a prescription. It is
    fully eligible and loses only on score, so nothing but this veto stands
    between it and a routine."""
    epiduo = product(
        "epiduo", "treatment", ["adapalene", "benzoyl_peroxide"], cadence="per_label"
    )

    usage, reason = try_place(epiduo, "treatment", [], K)

    assert usage is None
    assert reason == "self_conflicting_actives:adapalene:benzoyl_peroxide"


# --------------------------------------------------------------------------
# The attestation and the gate are the same computation
# --------------------------------------------------------------------------

def test_a_routine_failing_any_safety_check_is_always_rejected_by_the_validator():
    """explain.safety_checks is serialized into every routine as its own
    attestation, but pipeline.run gates on validate_routine. When the two
    computed the rules separately they disagreed, and a routine could ship
    carrying a signed statement that it had failed -- worse than no attestation
    at all for health-adjacent output. They now read one helper, so this holds
    for every routine, not just the ones we thought to check."""
    failing_routines = [
        # retinoid outside its pinned PM session (validate_routine never
        # checked this rule at all -- only the ignored attestation did)
        routine_of(
            *carrier_steps(),
            step(product("ada", "treatment", ["adapalene"], cadence="per_label"),
                 "treatment", "PER_LABEL"),
        ),
        routine_of(
            *carrier_steps(),
            step(product("ret", "serum", ["retinol"]), "serum", "AM"),
        ),
        # SPF outside AM
        routine_of(
            step(product("cl", "cleanser"), "cleanser", "AM_PM"),
            step(product("mo", "moisturizer"), "moisturizer", "AM_PM"),
            step(product("sp", "spf", spf=50), "spf", "AM_PM"),
        ),
        # conflicting pair sharing a session
        routine_of(
            *carrier_steps(),
            step(product("bp", "treatment", ["benzoyl_peroxide"]), "treatment", "AM"),
            step(product("vc", "serum", ["vitamin_c"]), "serum", "AM"),
        ),
        # two products in one slot
        routine_of(
            *carrier_steps(),
            step(product("s1", "serum", ["niacinamide"]), "serum", "PM"),
            step(product("s2", "serum", ["azelaic_acid"]), "serum", "PM"),
        ),
    ]

    for routine in failing_routines:
        checks = safety_checks(routine, K)
        failed = [c["rule"] for c in checks if not c["passed"]]
        assert failed, f"expected a failing attestation for {routine.product_ids}"
        assert hybrid_reasons(routine, has_targets=False), (
            f"{failed} failed the attestation but the validator emitted the routine"
        )


def test_safety_checks_reports_every_rule_in_its_published_order():
    """The rule names and their order are the document's contract."""
    routine = routine_of(*carrier_steps())

    assert [c["rule"] for c in safety_checks(routine, K)] == [
        "spf_am_only",
        "retinoids_pm_only",
        "no_conflicting_actives_in_same_session",
        "one_product_per_slot",
    ]
    assert all(set(c) == {"rule", "passed"} for c in safety_checks(routine, K))


# --------------------------------------------------------------------------
# A pinned session outranks a label cadence
# --------------------------------------------------------------------------

def test_label_cadence_never_moves_a_retinoid_out_of_its_pinned_pm_session():
    """`cadence` is an approvable overlay fact; "retinoids PM-only" is a safety
    rule the code itself says may never flip. The cadence lookup used to return
    before the pin was consulted, so one approved `cadence: "am"` fact was enough
    to ship a retinoid in am[]."""
    for cadence in ("am", "pm", "am_pm", "twice_daily", "daily", "once_daily",
                    "per_label", None):
        retinol = product("ret", "serum", ["retinol"], cadence=cadence)
        assert preferred_usage(retinol, "serum", K) == ("PM", True), (
            f"cadence={cadence!r} moved a retinoid off its PM pin"
        )


def test_label_cadence_still_sets_the_session_for_a_product_with_no_pin():
    """The pin takes precedence, but a verified cadence still governs everything
    it is not fighting."""
    assert preferred_usage(
        product("bp", "treatment", ["benzoyl_peroxide"], cadence="pm"), "treatment", K
    ) == ("PM", True)
    assert preferred_usage(
        product("aze", "treatment", ["azelaic_acid"], cadence="twice_daily"), "treatment", K
    ) == ("AM_PM", True)
    assert preferred_usage(
        product("nia", "serum", ["niacinamide"], cadence="per_label"), "serum", K
    ) == ("PER_LABEL", True)


# --------------------------------------------------------------------------
# An intra-product conflict costs one candidate, not the whole archetype
# --------------------------------------------------------------------------

def test_greedy_fills_the_slot_with_the_next_candidate_when_a_combination_is_rejected():
    """12 real cosmetic rows carry both glycolic acid and retinol. The pair is
    rejected either way, but rejecting it at insertion lets the greedy move to
    the next candidate; as a post-hoc validator veto it dropped the ENTIRE
    archetype, because _place_best had long since stopped looking."""
    archetype = {"id": "a", "title": "t", "rationale": "r",
                 "slots": ["cleanser", "serum", "moisturizer", "spf"], "constraints": {}}
    combo = product("combo", "serum", ["glycolic_acid", "retinol"])
    fallback = product("fallback", "serum", ["niacinamide"])
    scored_by_slot = {
        "cleanser": [scored(product("cl", "cleanser"))],
        # the combination outranks the fallback, so the greedy meets it first
        "serum": [scored(combo, 0.9), scored(fallback, 0.5)],
        "moisturizer": [scored(product("mo", "moisturizer"))],
        "spf": [scored(product("sp", "spf", spf=50))],
    }

    routine = compose_archetype(archetype, scored_by_slot, TARGETS, PROFILE, K)

    assert "fallback" in routine.product_ids  # the slot is filled, not lost
    assert "combo" not in routine.product_ids
    assert "slot_unfilled:serum" not in routine.notes
    assert hybrid_reasons(routine, has_targets=False) == []
    assert {"product_id": "combo", "slot": "serum",
            "reason": "self_conflicting_actives:glycolic_acid:retinol"} in routine.compose_vetoes


def test_validator_still_backstops_a_combination_product_it_never_placed():
    """try_place means the composer never emits one, but validate_routine is the
    last fail-closed check over a complete regimen and must not assume that."""
    routine = routine_of(
        *carrier_steps(),
        step(product("combo", "serum", ["glycolic_acid", "retinol"]), "serum", "PM"),
    )

    assert "self_conflict:combo:glycolic_acid:retinol" in hybrid_reasons(routine)


# --------------------------------------------------------------------------
# Diversity and the budget cap have to hold at the same time
# --------------------------------------------------------------------------

def _budget_archetypes():
    best = next(a for a in K.archetypes if a["id"] == "best_overall")
    budget = next(a for a in K.archetypes if a["id"] == "budget")
    return best, budget


def test_diversify_keeps_the_budget_archetype_inside_its_total_cap():
    """_diversify swaps a product in AFTER _reduce_to_budget has already run, so
    the swap can push the total straight back over the cap the archetype
    promises. It re-filters per item but never re-ran the reducer, so the budget
    routine silently shipped over its own cap."""
    best, budget = _budget_archetypes()
    cap = budget["constraints"]["max_total_price_usd"]
    # every slot $15 (total exactly the $75 cap); the cleanser has a $16
    # runner-up for _diversify to reach for, and a $14 one the reducer can use
    scored_by_slot = {
        "cleanser": [scored(product("cl-a", "cleanser", price_usd=15.0), 0.9),
                     scored(product("cl-b", "cleanser", price_usd=16.0), 0.8),
                     scored(product("cl-c", "cleanser", price_usd=14.0), 0.7)],
        "treatment": [scored(product("tr-a", "treatment", ["azelaic_acid"], price_usd=15.0), 0.9)],
        "serum": [scored(product("se-a", "serum", ["niacinamide"], price_usd=15.0), 0.9)],
        "moisturizer": [scored(product("mo-a", "moisturizer", price_usd=15.0), 0.9)],
        "spf": [scored(product("sp-a", "spf", price_usd=15.0, spf=50), 0.9)],
    }

    best_routine, budget_routine = compose_all(
        [(best, scored_by_slot), (budget, scored_by_slot)], TARGETS, PROFILE, K
    )

    assert budget_routine.total_price_usd <= cap
    # the diversity guarantee still holds -- the reducer did not simply put back
    # what _diversify swapped out
    assert budget_routine.product_ids != best_routine.product_ids
    assert any(n.startswith("diversified_from_best_overall") for n in budget_routine.notes)
    assert "over_total_budget" not in budget_routine.notes


def test_diversify_says_so_when_it_cannot_hold_both_the_cap_and_diversity():
    """With no cheaper alternative left, the cap cannot be met without undoing
    the diversification. The routine may go over, but it says `over_total_budget`
    rather than busting the cap silently -- the existing contract for "could not
    fit"."""
    best, budget = _budget_archetypes()
    scored_by_slot = {
        # cl-b at $16 is the only alternative, and it is the dearer one
        "cleanser": [scored(product("cl-a", "cleanser", price_usd=15.0), 0.9),
                     scored(product("cl-b", "cleanser", price_usd=16.0), 0.8)],
        "treatment": [scored(product("tr-a", "treatment", ["azelaic_acid"], price_usd=15.0), 0.9)],
        "serum": [scored(product("se-a", "serum", ["niacinamide"], price_usd=15.0), 0.9)],
        "moisturizer": [scored(product("mo-a", "moisturizer", price_usd=15.0), 0.9)],
        "spf": [scored(product("sp-a", "spf", price_usd=15.0, spf=50), 0.9)],
    }

    best_routine, budget_routine = compose_all(
        [(best, scored_by_slot), (budget, scored_by_slot)], TARGETS, PROFILE, K
    )

    assert budget_routine.product_ids != best_routine.product_ids
    assert "over_total_budget" in budget_routine.notes


def test_every_step_added_by_diversify_still_passes_the_conflict_check():
    """_place_best is the single chokepoint every step-adding path runs through,
    _diversify included, so a diversified routine is still a valid one."""
    best = next(a for a in K.archetypes if a["id"] == "best_overall")
    other = dict(best, id="comprehensive")
    scored_by_slot = {
        "cleanser": [scored(product("cl", "cleanser"))],
        "treatment": [scored(product("bp", "treatment", ["benzoyl_peroxide"]), 0.9)],
        # the only alternative serum conflicts with the treatment in AM, so
        # _diversify must place it in PM or not at all
        "serum": [scored(product("se-a", "serum", ["niacinamide"]), 0.9),
                  scored(product("vc", "serum", ["vitamin_c"]), 0.5)],
        "moisturizer": [scored(product("mo", "moisturizer"))],
        "spf": [scored(product("sp", "spf", spf=50))],
    }

    routines = compose_all(
        [(best, scored_by_slot), (other, scored_by_slot)], TARGETS, PROFILE, K
    )

    for routine in routines:
        assert hybrid_reasons(routine) == []
        assert all(c["passed"] for c in safety_checks(routine, K))
