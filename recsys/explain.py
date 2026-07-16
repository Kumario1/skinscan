"""Explanation builder: per-product "why" from the same SignalScore objects the
ranker used (no separate marketing-copy path), plus the D-002 cosmetic framing
and the doctor-referral passthrough.
"""
from __future__ import annotations

from .compose import ComposedRoutine, Step, session_rule_findings
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
    # A product whose usage facts came from the evidence overlay is "verified";
    # one placed by catalog category + safe defaults is "category_derived".
    verification = "verified" if product.routine_roles else "category_derived"
    notes = list(step.notes)
    if verification == "category_derived":
        notes.append("category-derived: role and usage inferred from the product "
                     "category, not individually evidence-verified")
    # The pipeline lists prescriptions rather than placing them, so this should
    # stay false; it holds the line if a drug row ever reaches a step.
    prescription = is_prescription(product)
    if prescription:
        notes.append("prescription — consult a doctor to get this prescribed")
    return {
        "slot": step.slot,
        "product_id": product.product_id,
        "name": product.name,
        "brand": product.brand,
        "price_usd": product.price_usd,
        "usage": step.usage,
        "verification": verification,
        "prescription": prescription,
        "directions": {
            "cadence": product.cadence,
            "cadence_source": product.cadence_source,
            "amount": product.amount,
            "amount_source": product.amount_source,
        },
        "notes": notes,
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


def is_prescription(product) -> bool:
    """A drug whose own label does not call it OTC (D-033).

    Both halves matter: a cosmetic carries no drug_actives, so it can never be
    mislabelled by otc_drug alone.
    """
    return bool(product.drug_actives) and product.otc_drug is False


def prescription_options(products, targets, k: Knowledge) -> list[dict]:
    """Prescription-strength products that fit the reported concerns.

    Surfaced for a doctor conversation rather than ranked into the routine
    (D-033: the app may surface prescription-strength options while advising the
    user to see a doctor to obtain them). Ranking them against cosmetics would
    need a claim about prescription-strength efficacy, and which therapies are
    indicated for which concern is D-029 clinician-gated -- so these are listed,
    never placed. Only products already past every safety gate reach here.
    """
    seen: set = set()
    options: list[dict] = []
    for product in products:
        if not is_prescription(product):
            continue
        actives = set(product.actives)
        targeted = sorted({
            t.concern for t in targets
            if actives & set(k.concern_actives.get(t.concern) or ())
        })
        if not targeted:
            continue
        strengths = tuple(sorted(
            (str(a.get("name")), str(a.get("strength"))) for a in product.drug_actives
        ))
        key = (product.name.strip().lower(), strengths)
        if key in seen:
            continue
        seen.add(key)
        options.append({
            "name": product.name,
            "format": product.format,
            "actives": [{"name": name, "strength": strength} for name, strength in strengths],
            "targets": targeted,
            # States only what the row proves -- the strengths its label gives,
            # and that the label does not mark it OTC -- which is exactly what
            # is_prescription() decided on. This said "at a strength only a
            # prescription can provide", which the engine's own drug catalog
            # refutes: Differin Epiduo is otc_drug=True at the same adapalene
            # 0.1% + benzoyl peroxide 2.5%. Rx status follows the molecule and
            # its label, not the strength (Acanya is Rx because clindamycin is
            # an antibiotic). The cause is left unstated rather than swapped for
            # another guess: "these strengths are sold OTC" would be as false
            # for clindamycin as the old sentence was for adapalene.
            "why": "Contains " + ", ".join(
                f"{name.replace('_', ' ')} {strength}" for name, strength in strengths
            ) + ", as stated on its FDA label, which does not list it as "
                "over-the-counter.",
            "label_source": product.label_source,
            "note": "prescription — a doctor or dermatologist can advise on and prescribe this",
        })
    options.sort(key=lambda item: (item["targets"], item["name"]))
    return options


def safety_checks(routine: ComposedRoutine, k: Knowledge) -> list[dict]:
    """The routine's published attestation that it honours the session rules.

    Projected from the very findings validate_routine vetoes on, so the two can
    never disagree. They used to compute the rules separately, and a routine
    could be emitted by the gate while carrying an attestation, right here in
    its own document, that it had failed -- an attestation nothing gated on.
    """
    return [
        {"rule": finding["rule"], "passed": finding["passed"]}
        for finding in session_rule_findings(routine, k)
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
