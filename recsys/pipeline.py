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
    SCHEMA_VERSION,
    AnalysisInput,
    ContractViolation,
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

# load_providers() reports a store it declined to load by ending the warning with
# SKIPPED_MARKER. Matching on the message is a seam, not a contract: signals.py
# should return this structurally (see the note in the review). Kept here as a
# named constant so the coupling is greppable from one place.
SKIPPED_MARKER = "— skipped"


def skipped_signal_warnings(warnings: list[str]) -> list[str]:
    """The warnings that mean a signal store did not load. A skipped store does
    not degrade the ranker gracefully -- it degenerates every store-backed signal
    to a neutral 0.5, so the ranker scores blind while the run still reports a
    status. Callers use this to refuse to publish such a run."""
    return [w for w in warnings if w.endswith(SKIPPED_MARKER)]


def select_targets(analysis: AnalysisInput) -> tuple[TargetConcern, ...]:
    findings = [c for c in analysis.concerns if c.severity >= 1]
    findings.sort(key=lambda c: (-c.severity, -c.confidence, c.concern))
    return tuple(TargetConcern(c.concern, c.severity, c.confidence) for c in findings)


def _primary_treatment_unknowns(profile) -> tuple[str, ...]:
    """Critical intake fields D-029 requires before primary treatment."""
    unknown = set(profile.unknown_fields)
    for name in ("age_years", "acne_duration_weeks", "painful_or_deep_lesions",
                 "prior_scarring"):
        if getattr(profile, name) is None:
            unknown.add(name)
    if profile.pregnancy_status == "unknown":
        unknown.add("pregnancy_status")
    return tuple(sorted(unknown))


def _mark_support_only(routines) -> None:
    for routine in routines:
        routine.notes = [
            note for note in routine.notes if note != "clear_skin_maintenance"
        ]
        if "support_only_treatment_deferred" not in routine.notes:
            routine.notes.append("support_only_treatment_deferred")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        # TimeoutExpired derives from SubprocessError, NOT OSError: catching only
        # OSError let a hung `git rev-parse` past timeout=10 kill the whole
        # recommendation over a provenance nicety. The commit is optional; the
        # recommendation is not.
        return None


def resolve_paths(
    data_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve (catalog_path, static_root, drug_path, verification_root) from a
    data root and an optional explicit catalog.

    The single source of truth for where the engine reads its data. Exported so
    that tools which check a run (tools/verify_e2e.py) resolve identically by
    construction rather than by a copy-paste that can drift out of step with
    this function -- the harness must never check a different catalog than the
    engine used.
    """
    data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    # The seed catalog is a 60-product fixture. It resolves only where it
    # actually sits, so a data root that carries no catalog of its own fails
    # here by name rather than degrading to the fixture and answering plausibly
    # from the wrong 60 products.
    full_path = data_root / "catalog_full.json"
    seed_path = data_root / "catalog" / "seed_catalog.json"
    if catalog_path:
        catalog_path = Path(catalog_path)
    elif full_path.exists():
        catalog_path = full_path
    elif seed_path.exists():
        catalog_path = seed_path
    else:
        raise ContractViolation(
            "catalog",
            f"no catalog under data root {data_root}: looked for "
            f"{full_path} and {seed_path}",
        )
    static_root = data_root if (data_root / "knowledge").exists() else DEFAULT_DATA_ROOT
    drug_path = data_root / "catalog_drug.json"
    verification_root = (
        data_root / "verification"
        if (data_root / "verification" / "approved.json").exists()
        else static_root / "verification"
    )
    return catalog_path, static_root, drug_path, verification_root


def run(
    analysis_path: str | Path,
    profile_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    data_root: str | Path | None = None,
    generated_at: str | None = None,
    eligibility_mode: str = "strict",
    allow_signal_catalog_mismatch: bool = False,
) -> dict:
    # The signal stores are keyed by the sha256 of the catalog they were built
    # against; scoring any other catalog with them strands every store, and a
    # stranded store means every store-backed signal degenerates to a neutral
    # 0.5 -- a blind ranking in a document that still looks complete. So
    # load_providers hard-fails on a mismatch unless
    # allow_signal_catalog_mismatch=True takes the blind ranking deliberately,
    # in which case the store is skipped with a warning that lands in this
    # document's warnings/data_versions.signals (and the CLI's exit code).
    data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    catalog_path, static_root, drug_path, verification_root = resolve_paths(
        data_root, catalog_path
    )

    knowledge = load_knowledge(static_root / "knowledge")
    analysis = load_analysis(analysis_path)
    profile = resolve_profile(profile_path, analysis)
    all_targets = select_targets(analysis)
    profile_unknowns = _primary_treatment_unknowns(profile)
    can_treat = (
        analysis.therapy_disposition == "active_treatment"
        and analysis.therapy_primary is not None
        and not profile_unknowns
    )
    therapy_primary = analysis.therapy_primary if can_treat else None
    effective_disposition = analysis.therapy_disposition
    deferred_reasons = list(analysis.therapy_deferred_reasons)
    if analysis.therapy_disposition == "active_treatment" and not can_treat:
        effective_disposition = "defer"
        deferred_reasons.extend(f"required_profile_unknown:{name}" for name in profile_unknowns)
        if analysis.therapy_primary is None:
            deferred_reasons.append("therapy_plan_primary_missing")
    products, catalog_header = load_catalog(catalog_path)
    catalog_sha256 = sha256_file(catalog_path)
    verification_now = (
        _dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        if generated_at else _dt.datetime.now(_dt.timezone.utc)
    )
    overlay, verification_warnings, verification_meta = load_verification_overlay(
        verification_root, now=verification_now
    )
    # Drug rows ride in a catalog of their own: the signal stores are keyed by the
    # cosmetics catalog's sha256, so folding them into that file would strand
    # every store. Merge before the overlay so approved facts reach them too.
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
    providers, store_meta, signal_warnings = load_providers(
        data_root, catalog_sha256,
        allow_catalog_mismatch=allow_signal_catalog_mismatch,
    )
    warnings = verification_warnings + signal_warnings
    if eligibility_mode == "hybrid":
        warnings.append("hybrid eligibility is disabled by D-029 — strict eligibility applied")

    fulfillment_status = (
        "pending" if can_treat
        else "deferred" if analysis.therapy_disposition == "active_treatment"
        else "not_requested"
    )
    document: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": {"version": ENGINE_VERSION, "git_commit": _git_commit(),
                   "eligibility_mode": "strict",
                   "requested_eligibility_mode": eligibility_mode},
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
            "unknown_fields": sorted(profile.unknown_fields),
        },
        "care_decision": {
            "triage_level": analysis.triage_level,
            "referral_reasons": list(analysis.referral_reasons),
            "therapy_disposition": analysis.therapy_disposition,
            "policy_reviewed": analysis.policy_reviewed,
            "therapy_policy_reviewed": analysis.therapy_policy_reviewed,
        },
        "therapy_plan": dict(analysis.therapy_plan),
        "treatment_fulfillment": {
            "status": fulfillment_status,
            "reasons": list(dict.fromkeys(deferred_reasons)),
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
                analysis.triage_level, analysis.referral_reasons,
                analysis.safety_observations, effective_disposition,
            ),
        },
        "target_concerns": [
            {
                "concern": t.concern,
                "severity": t.severity,
                "selected_for_treatment": can_treat,
                "referral_emphasis": t.concern in knowledge.referral_emphasis,
            }
            for t in all_targets
        ],
        "warnings": warnings,
    }

    candidates = generate_candidates(
        products, all_targets, knowledge, therapy_primary=therapy_primary,
    )
    gated, profile_vetoes, quality_flags = apply_profile_gates(
        candidates, profile, knowledge,
    )
    # Listed, never placed. Surfacing prescription-strength options with a
    # referral is D-033; ranking one into a routine would instead assert that it
    # beats the cosmetics, and which therapy suits which concern is D-029
    # clinician-gated. Reading the options out of the gated pool and then
    # dropping them makes that true by construction rather than by whichever way
    # the ranking happens to fall -- and keeps rows that carry no retail price
    # out of every routine total.
    rx_options = (
        prescription_options(gated.get("treatment", []), therapy_primary)
        if can_treat else []
    )
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
        targets=all_targets, profile=profile, knowledge=knowledge,
        category_prices=category_prices,
    )

    archetype_scored = []
    for archetype in knowledge.archetypes[:1]:
        weights = archetype.get("weights") or knowledge.default_weights
        scored_by_slot = {
            slot: score_products(slot_products, slot, providers, ctx, weights)
            for slot, slot_products in gated.items()
        }
        archetype_scored.append((archetype, scored_by_slot))

    routines = compose_all(archetype_scored, all_targets if can_treat else (), profile, knowledge)
    if not can_treat:
        _mark_support_only(routines)

    valid_routines = []
    unavailable = []
    for routine in routines:
        reasons = validate_routine(
            routine, profile, knowledge, has_targets=can_treat,
        )
        if reasons:
            unavailable.append({
                "archetype": routine.archetype["id"],
                "reasons": reasons,
            })
        else:
            valid_routines.append(routine)
    if can_treat and not valid_routines:
        document["treatment_fulfillment"] = {
            "status": "unfilled",
            "reasons": ["required_role_unfilled:treatment"],
        }
        for target in document["target_concerns"]:
            target["selected_for_treatment"] = False
        document["triage"]["see_doctor_note"] = see_doctor_note(
            analysis.triage_level, analysis.referral_reasons,
            analysis.safety_observations, "defer",
        )
        support_routines = compose_all(archetype_scored, (), profile, knowledge)
        _mark_support_only(support_routines)
        valid_routines = [
            routine for routine in support_routines
            if not validate_routine(routine, profile, knowledge, has_targets=False)
        ]
        routines.extend(support_routines)
        if valid_routines:
            unavailable = []
    elif can_treat:
        document["treatment_fulfillment"] = {
            "status": "included",
            "reasons": [],
        }
    document["status"] = "ok" if valid_routines else "unavailable"
    serialized = [
        routine_to_dict(r, knowledge, profile, quality_flags) for r in valid_routines[:1]
    ]
    document["routines"] = serialized
    document["selected_regimen"] = serialized[0] if serialized else None
    document["selected_products"] = (
        {step.slot: step.scored.product.product_id for step in valid_routines[0].steps}
        if valid_routines else {}
    )
    document["alternatives"] = {}
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
