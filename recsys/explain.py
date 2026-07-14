"""Explanation builder: per-product "why" from the same SignalScore objects the
ranker used (no separate marketing-copy path), plus the D-002 cosmetic framing
and the doctor-referral passthrough.
"""
from __future__ import annotations

from .compose import ComposedRoutine, Step, _conflict_between, _sessions
from .contracts import Profile
from .knowledge import Knowledge

FRAMING_TEXT = (
    "These are cosmetic, appearance-based suggestions only — not medical advice "
    "and not a diagnosis (SkinScan decision D-002). Concerns are described by how "
    "they look in a photo. Prescription-strength options exist for many of these "
    "concerns; a doctor or dermatologist can advise on and prescribe them."
)

_REFERRAL_PHRASES = {
    "high_count_or_severity_review": "the number or severity of detected lesions",
    "scarring_risk": "possible scarring",
    "persistent_pigment_concern": "persistent dark-spot concerns",
}


def see_doctor_note(
    triage_level: str,
    referral_reasons: tuple[str, ...],
    observations: tuple[dict, ...],
) -> str | None:
    parts: list[str] = []
    reasons = [_REFERRAL_PHRASES.get(r, r.replace("_", " ")) for r in referral_reasons]
    if triage_level in ("derm_first", "abstain"):
        parts.append(
            "Please see a dermatologist before starting new products"
            + (f" — the analysis flagged {', '.join(reasons)}." if reasons else ".")
        )
    elif triage_level == "routine_plus_review" or reasons:
        parts.append(
            "These routines are safe to start, but a professional review is "
            "recommended" + (f" for {', '.join(reasons)}." if reasons else ".")
        )
    flagged = [o["code"] for o in observations if o.get("professional_review")]
    if flagged:
        parts.append(
            "The analysis also noticed a spot it cannot assess "
            f"({', '.join(sorted(set(flagged)))}) — worth showing to a doctor."
        )
    return " ".join(parts) or None


def _step_summary(step: Step, k: Knowledge) -> str:
    concern_fit = next((s for s in step.scored.signals if s.name == "concern_fit"), None)
    matched = (concern_fit.details.get("matched") if concern_fit else None) or {}
    if matched:
        actives = sorted({a for overlap in matched.values() for a in overlap})
        phrases = []
        for concern in matched:
            phrase = k.phrasing.get(concern, concern)
            if phrase not in phrases:
                phrases.append(phrase)
        return (
            f"Contains {', '.join(a.replace('_', ' ') for a in actives)}, "
            f"which targets {'; '.join(phrases)}."
        )
    spf_summary = (
        "Daily verified broad-spectrum sun protection — also helps prevent new dark spots."
        if step.scored.product.broad_spectrum is True
        else "Daily sun protection; broad-spectrum status is not verified."
    )
    defaults = {
        "cleanser": "A daily cleanser to start each routine.",
        "moisturizer": "Keeps your skin barrier hydrated alongside the treatment steps.",
        "spf": spf_summary,
    }
    return defaults.get(step.slot, "Supports the routine.")


def _step_uncertainty(step: Step, profile: Profile) -> list[str]:
    notes = list(step.scored.uncertainty)
    if step.slot == "spf" and step.scored.product.spf_source == "name_parse":
        notes.append("spf_value_from_name_parse_not_verified")
    review = next((s for s in step.scored.signals if s.name == "review_quality"), None)
    if (
        review is not None
        and not review.details.get("missing")
        and profile.skin_type != "unknown"
        and review.details.get("cell") == "all"
    ):
        notes.append("no_outcome_data_for_your_skin_type")
    return sorted(set(notes))


def step_to_dict(step: Step, k: Knowledge, profile: Profile) -> dict:
    product = step.scored.product
    return {
        "slot": step.slot,
        "product_id": product.product_id,
        "name": product.name,
        "brand": product.brand,
        "price_usd": product.price_usd,
        "usage": step.usage,
        "directions": {
            "cadence": product.cadence,
            "cadence_source": product.cadence_source,
            "amount": product.amount,
            "amount_source": product.amount_source,
        },
        "notes": list(step.notes),
        "why": {
            "summary": _step_summary(step, k),
            "score": step.scored.final,
            "signals": [
                {"name": s.name, "value": s.value, "evidence": s.evidence}
                for s in step.scored.signals
            ],
            "uncertainty": _step_uncertainty(step, profile),
        },
    }


def safety_checks(routine: ComposedRoutine, k: Knowledge) -> list[dict]:
    steps = routine.steps
    spf_ok = all(s.usage == "AM" for s in steps if s.slot == "spf")
    retinoid_ok = all(
        s.usage == "PM" for s in steps
        if set(s.scored.product.actives) & k.retinoids
    )
    conflict_ok = True
    for i, a in enumerate(steps):
        for b in steps[i + 1:]:
            if _sessions(a.usage) & _sessions(b.usage):
                if _conflict_between(a.scored.product, b.scored.product, k):
                    conflict_ok = False
    slots_ok = len({s.slot for s in steps}) == len(steps)
    return [
        {"rule": "spf_am_only", "passed": spf_ok},
        {"rule": "retinoids_pm_only", "passed": retinoid_ok},
        {"rule": "no_conflicting_actives_in_same_session", "passed": conflict_ok},
        {"rule": "one_product_per_slot", "passed": slots_ok},
    ]


def routine_to_dict(routine: ComposedRoutine, k: Knowledge, profile: Profile) -> dict:
    steps = [step_to_dict(s, k, profile) for s in routine.steps]
    return {
        "archetype": routine.archetype["id"],
        "title": routine.archetype["title"],
        "rationale": routine.archetype["rationale"],
        "total_price_usd": routine.total_price_usd,
        "slot_count": len(routine.steps),
        "am": [s for s, raw in zip(steps, routine.steps) if raw.usage in ("AM", "AM_PM")],
        "pm": [s for s, raw in zip(steps, routine.steps) if raw.usage in ("PM", "AM_PM")],
        "per_label": [
            s for s, raw in zip(steps, routine.steps) if raw.usage == "PER_LABEL"
        ],
        "safety_checks": safety_checks(routine, k),
        "notes": sorted(set(routine.notes)),
    }
