"""Pipeline orchestration: load -> targets -> candidates -> gates -> score ->
compose -> explain -> emit. Pure functions chained; every output document is
stamped with the sha256 of every input and data artifact it used."""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path

from . import ENGINE_VERSION
from .candidates import generate_candidates
from .catalog import load_catalog
from .compose import compose_all, validate_routine
from .contracts import (
    REFERRAL_ONLY_TRIAGE,
    SCHEMA_VERSION,
    AnalysisInput,
    load_analysis,
    resolve_profile,
    sha256_file,
)
from .explain import (
    FRAMING_TEXT,
    is_prescription,
    prescription_options,
    routine_to_dict,
    see_doctor_note,
)
from .gates import apply_profile_gates
from .knowledge import load_knowledge
from .scoring import score_products
from .signals import ScoringContext, TargetConcern, load_providers
from .verification import apply_verification, load_verification_overlay

DEFAULT_DATA_ROOT = Path(__file__).parent / "data"


def select_targets(analysis: AnalysisInput) -> tuple[TargetConcern, ...]:
    findings = [c for c in analysis.concerns if c.severity >= 1]
    findings.sort(key=lambda c: (-c.severity, -c.confidence, c.concern))
    return tuple(TargetConcern(c.concern, c.severity, c.confidence) for c in findings)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except OSError:
        return None


def run(
    analysis_path: str | Path,
    profile_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    data_root: str | Path | None = None,
    generated_at: str | None = None,
    eligibility_mode: str = "strict",
) -> dict:
    data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    if catalog_path:
        catalog_path = Path(catalog_path)
    elif (data_root / "catalog_full.json").exists():
        catalog_path = data_root / "catalog_full.json"
    else:
        catalog_path = data_root / "catalog" / "seed_catalog.json"
    static_root = data_root if (data_root / "knowledge").exists() else DEFAULT_DATA_ROOT

    knowledge = load_knowledge(static_root / "knowledge")
    analysis = load_analysis(analysis_path)
    profile = resolve_profile(profile_path, analysis)
    products, catalog_header = load_catalog(catalog_path)
    catalog_sha256 = sha256_file(catalog_path)
    verification_now = (
        _dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        if generated_at else _dt.datetime.now(_dt.timezone.utc)
    )
    verification_root = (
        data_root / "verification"
        if (data_root / "verification" / "approved.json").exists()
        else static_root / "verification"
    )
    overlay, verification_warnings, verification_meta = load_verification_overlay(
        verification_root, now=verification_now
    )
    # Drug rows ride in a catalog of their own: the signal stores are keyed by the
    # cosmetics catalog's sha256, so folding them into that file would strand
    # every store. Merge before the overlay so approved facts reach them too.
    drug_path = data_root / "catalog_drug.json"
    drug_meta = None
    if drug_path.exists():
        drug_products, _ = load_catalog(drug_path)
        products = products + drug_products
        drug_meta = {
            "path": str(drug_path),
            "sha256": sha256_file(drug_path),
            "products": len(drug_products),
            "prescription": sum(1 for p in drug_products if p.otc_drug is False),
        }
    products = apply_verification(products, overlay)
    providers, store_meta, signal_warnings = load_providers(data_root, catalog_sha256)
    warnings = verification_warnings + signal_warnings

    targets = select_targets(analysis)
    document: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": {"version": ENGINE_VERSION, "git_commit": _git_commit(),
                   "eligibility_mode": eligibility_mode},
        "inputs": {
            "analysis_sha256": analysis.analysis_sha256,
            "source_image_sha256": analysis.source_image_sha256,
            "profile_sha256": profile.profile_sha256,
            "profile_source": profile.source,
        },
        "profile_used": {
            "skin_type": profile.skin_type,
            "tone_bucket": profile.tone_bucket,
            "tone_source": profile.tone_source,
            "pregnancy_status": profile.pregnancy_status,
            "age_years": profile.age_years,
            "allergies": list(profile.allergies),
            "sensitivity_conditions": list(profile.sensitivity_conditions),
            "current_actives": list(profile.current_actives),
            "current_medications": list(profile.current_medications),
            "treatment_history": list(profile.treatment_history),
            "acne_duration_weeks": profile.acne_duration_weeks,
            "painful_or_deep_lesions": profile.painful_or_deep_lesions,
            "prior_scarring": profile.prior_scarring,
            "max_price_usd": profile.max_price_usd,
        },
        "data_versions": {
            "catalog": {
                "path": str(catalog_path),
                "sha256": catalog_sha256,
                "schema": catalog_header.get("schema_version"),
            },
            "drug_catalog": drug_meta,
            "signals": store_meta,
            "knowledge": [
                {"name": name, "sha256": digest}
                for name, digest in sorted(knowledge.file_sha256s.items())
            ],
            "verification": verification_meta,
        },
        "framing": {"cosmetic_only": True, "not_medical_advice": True, "text": FRAMING_TEXT},
        "triage": {
            "level": analysis.triage_level,
            "referral_reasons": list(analysis.referral_reasons),
            "professional_review_observations": [
                {"code": o["code"]} for o in analysis.safety_observations
                if o.get("professional_review")
            ],
            "see_doctor_note": see_doctor_note(
                analysis.triage_level, analysis.referral_reasons, analysis.safety_observations
            ),
        },
        "target_concerns": [
            {
                "concern": t.concern,
                "severity": t.severity,
                "selected_for_treatment": True,
                "referral_emphasis": t.concern in knowledge.referral_emphasis,
            }
            for t in targets
        ],
        "warnings": warnings,
    }

    if analysis.triage_level in REFERRAL_ONLY_TRIAGE:
        document["status"] = "referral_only"
        document["routines"] = []
        document["veto_log"] = {"profile": [], "compose": {}}
        return document

    strict_eligibility = eligibility_mode != "hybrid"
    candidates = generate_candidates(products, targets, knowledge, strict=strict_eligibility)
    gated, profile_vetoes, quality_flags = apply_profile_gates(
        candidates, profile, knowledge, strict=strict_eligibility
    )
    # Listed, never placed. Surfacing prescription-strength options with a
    # referral is D-033; ranking one into a routine would instead assert that it
    # beats the cosmetics, and which therapy suits which concern is D-029
    # clinician-gated. Reading the options out of the gated pool and then
    # dropping them makes that true by construction rather than by whichever way
    # the ranking happens to fall -- and keeps rows that carry no retail price
    # out of every routine total.
    rx_options = prescription_options(gated.get("treatment", []), targets, knowledge)
    gated = {
        slot: [p for p in items if not is_prescription(p)] for slot, items in gated.items()
    }

    category_prices = {
        category: tuple(sorted(
            p.price_usd for p in products
            if p.category == category and p.price_usd is not None
        ))
        for category in {p.category for p in products}
    }
    ctx = ScoringContext(
        targets=targets, profile=profile, knowledge=knowledge,
        category_prices=category_prices,
    )

    archetype_scored = []
    for archetype in knowledge.archetypes:
        weights = archetype.get("weights") or knowledge.default_weights
        scored_by_slot = {
            slot: score_products(slot_products, slot, providers, ctx, weights)
            for slot, slot_products in gated.items()
        }
        archetype_scored.append((archetype, scored_by_slot))

    routines = compose_all(archetype_scored, targets, profile, knowledge)

    valid_routines = []
    unavailable = []
    for routine in routines:
        reasons = validate_routine(
            routine, profile, knowledge, has_targets=bool(targets),
            strict=(eligibility_mode != "hybrid"),
        )
        if reasons:
            unavailable.append({
                "archetype": routine.archetype["id"],
                "reasons": reasons,
            })
        else:
            valid_routines.append(routine)
    document["status"] = (
        "ok" if len(valid_routines) == len(routines)
        else "partial" if valid_routines else "unavailable"
    )
    document["routines"] = [
        routine_to_dict(r, knowledge, profile) for r in valid_routines
    ]
    document["unavailable_archetypes"] = unavailable
    document["prescription_options"] = rx_options
    document["veto_log"] = {
        "profile": [v.to_dict() for v in profile_vetoes],
        "compose": {r.archetype["id"]: r.compose_vetoes for r in routines},
    }
    return document


def emit(document: dict, out_path: str | Path) -> Path:
    """Atomic write with stable key order."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, out_path)
    return out_path
