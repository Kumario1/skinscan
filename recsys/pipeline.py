"""Pipeline orchestration: load -> targets -> candidates -> gates -> score ->
compose -> explain -> emit. Pure functions chained; every output document is
stamped with the sha256 of every input and data artifact it used."""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from dataclasses import dataclass, fields
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
from .signals import ScoringContext, TargetLesion, load_providers
from .verification import apply_verification, load_verification_overlay

DEFAULT_DATA_ROOT = Path(__file__).parent / "data"

# load_providers() reports a store it declined to load by ending the warning with
# SKIPPED_MARKER. Matching on the message is a seam, not a contract: signals.py
# should return this structurally (see the note in the review). Kept here as a
# named constant so the coupling is greppable from one place.
SKIPPED_MARKER = "— skipped"
SELECTABLE_SLOTS = (
    "cleanser", "treatment", "serum", "scar_care", "moisturizer", "spf",
)
CATALOG_SELECTOR_SLOTS = (
    "cleanser", "treatment", "serum", "moisturizer", "spf",
)


@dataclass(frozen=True)
class CandidateSelection:
    """Product IDs chosen by an optional post-gate catalog selector."""

    product_ids: dict[str, str | None]
    metadata: dict


class SelectionUnavailable(RuntimeError):
    """An optional selector could not return a usable catalog selection."""

    def __init__(self, reason: str, metadata: dict | None = None):
        super().__init__(reason)
        self.metadata = dict(metadata or {})


def skipped_signal_warnings(warnings: list[str]) -> list[str]:
    """The warnings that mean a signal store did not load. A skipped store does
    not degrade the ranker gracefully -- it degenerates every store-backed signal
    to a neutral 0.5, so the ranker scores blind while the run still reports a
    status. Callers use this to refuse to publish such a run."""
    return [w for w in warnings if w.endswith(SKIPPED_MARKER)]


def select_targets(analysis: AnalysisInput) -> tuple[TargetLesion, ...]:
    """Select exact labels whose audited pathway permits retail matching."""
    findings = {item.lesion_type: item for item in analysis.lesion_findings}
    targets: list[TargetLesion] = []
    for pathway in analysis.care_pathways:
        finding = findings[pathway.lesion_type]
        if finding.count < 1 or pathway.status != "retail_eligible":
            continue
        targets.append(TargetLesion(
            lesion_type=pathway.lesion_type,
            count=finding.count,
            confidence=finding.max_detector_confidence or 0.0,
            target_actives=pathway.retail_target_actives,
            required_roles=pathway.required_product_roles,
            target_specs=pathway.retail_target_specs,
        ))
    targets.sort(key=lambda item: (-item.count, -item.confidence, item.lesion_type))
    return tuple(targets)


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


def _profile_context(profile) -> dict:
    context = {}
    for item in fields(profile):
        if item.name in {"source", "profile_sha256"}:
            continue
        value = getattr(profile, item.name)
        if isinstance(value, (tuple, frozenset)):
            value = sorted(value)
        context[item.name] = value
    return context


def _selection_context(
    analysis: AnalysisInput, profile, document: dict, can_treat: bool
) -> dict:
    """The complete user context a selector needs, without raw/image artifacts."""
    return {
        "profile": _profile_context(profile),
        "lesion_findings": [
            {
                "lesion_type": finding.lesion_type,
                "count": finding.count,
            }
            for finding in analysis.lesion_findings if finding.count
        ],
        "care_pathways": [
            {
                "lesion_type": pathway.lesion_type,
                "status": pathway.status,
                "retail_target_actives": list(pathway.retail_target_actives),
                "required_product_roles": list(pathway.required_product_roles),
                "reason_codes": list(pathway.reason_codes),
                "policy_source_ids": list(pathway.policy_source_ids),
                "required_answers": list(pathway.required_answers),
            }
            for pathway in getattr(analysis, "care_pathways", ())
            if pathway.status != "not_detected"
        ],
        "decision": document["care_decision"],
        "therapy_plan": document["therapy_plan"],
        "safety_observations": [
            {
                "code": observation.get("code"),
                "professional_review": observation.get("professional_review", False),
            }
            for observation in analysis.safety_observations
        ],
        "required_slots": list(dict.fromkeys([
            "cleanser", "moisturizer", "spf",
            *(["treatment"] if can_treat else []),
        ])),
    }


def _selected_candidates(
    result: CandidateSelection,
    candidates: dict,
    required_slots: list[str],
    can_treat: bool,
) -> tuple[dict | None, str | None]:
    if not isinstance(result, CandidateSelection):
        return None, "invalid_result_type"
    if frozenset(result.product_ids) != frozenset(CATALOG_SELECTOR_SLOTS):
        return None, "invalid_slot_set"
    if not isinstance(result.metadata, dict):
        return None, "invalid_metadata"

    chosen = {slot: result.product_ids.get(slot) for slot in SELECTABLE_SLOTS}
    for slot in required_slots:
        if not isinstance(chosen.get(slot), str) or not chosen[slot]:
            return None, f"required_slot_missing:{slot}"
    if not can_treat and chosen.get("treatment") is not None:
        return None, "treatment_not_allowed"

    selected: dict = {}
    non_null_ids: list[str] = []
    for slot in SELECTABLE_SLOTS:
        product_id = chosen[slot]
        if product_id is None:
            selected[slot] = []
            continue
        if not isinstance(product_id, str) or not product_id:
            return None, f"invalid_product_id:{slot}"
        match = [item for item in candidates.get(slot, [])
                 if item.product.product_id == product_id]
        if len(match) != 1:
            return None, f"product_not_safe_for_slot:{slot}:{product_id}"
        selected[slot] = match
        non_null_ids.append(product_id)
    if len(non_null_ids) != len(set(non_null_ids)):
        return None, "duplicate_product"
    return selected, None


def _selection_unavailable(document: dict, reason: str, metadata: dict | None = None) -> dict:
    reason = f"llm_selection_unavailable:{reason}"
    selection = dict(metadata or {})
    selection.update(status="unavailable", reason=reason)
    document.update({
        "status": "unavailable",
        "reason": reason,
        "routines": [],
        "selected_regimen": None,
        "selected_products": {},
        "alternatives": {},
        "unavailable_archetypes": [
            {"archetype": "best_overall", "reasons": [reason]}
        ],
        "selection": selection,
    })
    for target in document["target_lesions"]:
        target["selected_for_treatment"] = False
    if document["treatment_fulfillment"]["status"] == "pending":
        document["treatment_fulfillment"] = {
            "status": "unfilled",
            "reasons": [reason],
        }
    return document


def _attach_selector_catalog_facts(routines: list[dict], products) -> None:
    """Enrich only experimental output; keep the default artifact unchanged."""
    by_id = {product.product_id: product for product in products}
    for routine in routines:
        for session in ("am", "pm", "per_label"):
            for step in routine.get(session, []):
                product = by_id[step["product_id"]]
                step["actives"] = list(product.actives)
                step["ingredients"] = list(product.inci)


def _lesion_coverage(analysis: AnalysisInput, routine, can_treat: bool) -> list[dict]:
    """Attribute selected SKUs to exact labels using active facts, never scores."""
    findings = {item.lesion_type: item for item in analysis.lesion_findings}
    selected_steps = list(routine.steps) if routine is not None else []
    role_slots = {"treatment": "treatment", "sunscreen": "spf", "scar_care": "scar_care"}
    coverage: list[dict] = []
    for pathway in analysis.care_pathways:
        finding = findings[pathway.lesion_type]
        if finding.count < 1:
            continue
        if pathway.status == "monitoring_only":
            status = "monitoring_only"
            matched = []
        elif pathway.status == "unsupported":
            status = "unsupported"
            matched = []
        elif pathway.status == "clinician_only":
            status = "clinician_only"
            matched = []
        elif pathway.status == "deferred" or (
            "treatment" in pathway.required_product_roles and not can_treat
        ):
            status = "deferred"
            matched = []
        else:
            target_actives = set(pathway.retail_target_actives)
            required_slots = {
                role_slots[role] for role in pathway.required_product_roles
                if role in role_slots
            }
            matched = []
            for step in selected_steps:
                product = step.scored.product
                active_overlap = sorted(set(product.actives) & target_actives)
                if step.slot in required_slots and active_overlap:
                    matched.append({
                        "product_id": product.product_id,
                        "slot": step.slot,
                        "matched_actives": active_overlap,
                    })
            status = "covered_by_product" if matched else "unfilled"
        coverage.append({
            "lesion_type": pathway.lesion_type,
            "status": status,
            "products": matched,
            "reason_codes": list(pathway.reason_codes),
        })
    return coverage


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
    eligibility_mode: str = "hybrid",
    allow_signal_catalog_mismatch: bool = False,
    allow_unreviewed_policy: bool = False,
    candidate_selector=None,
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
    analysis = load_analysis(analysis_path, allow_unreviewed=allow_unreviewed_policy)
    if analysis.schema_version == "4" and profile_path is not None:
        raise ContractViolation(
            "profile",
            "schema 4 binds the resolved synthetic fixture profile in analysis.json",
        )
    profile = resolve_profile(profile_path, analysis)
    all_targets = select_targets(analysis)
    profile_unknowns = _primary_treatment_unknowns(profile)
    treatment_requested = any("treatment" in target.required_roles for target in all_targets)
    can_treat = treatment_requested and not profile_unknowns
    if analysis.schema_version == "3":
        can_treat = can_treat and analysis.therapy_primary is not None
    therapy_primary = analysis.therapy_primary if analysis.schema_version == "3" and can_treat else None
    selection_targets = tuple(
        target for target in all_targets
        if can_treat or "treatment" not in target.required_roles
    )
    effective_disposition = analysis.therapy_disposition
    deferred_reasons = list(analysis.therapy_deferred_reasons)
    if treatment_requested and not can_treat:
        effective_disposition = "defer"
        deferred_reasons.extend(f"required_profile_unknown:{name}" for name in profile_unknowns)
        if analysis.schema_version == "3" and analysis.therapy_primary is None:
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
    if eligibility_mode == "strict":
        warnings.append("strict verification-only eligibility is retired by D-035 — hybrid applied")

    fulfillment_status = (
        "pending" if can_treat
        else "deferred" if treatment_requested
        else "not_requested"
    )
    document: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": {"version": ENGINE_VERSION, "git_commit": _git_commit(),
                   "eligibility_mode": "hybrid",
                   "requested_eligibility_mode": eligibility_mode},
        "inputs": {
            "analysis_sha256": analysis.analysis_sha256,
            "source_image_sha256": analysis.source_image_sha256,
            "profile_sha256": profile.profile_sha256,
            "profile_source": profile.source,
        },
        "profile_used": _profile_context(profile),
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
        "target_lesions": [
            {
                "lesion_type": t.lesion_type,
                "count": t.count,
                "retail_target_actives": list(t.target_actives),
                "required_product_roles": list(t.required_roles),
                "selected_for_treatment": can_treat and "treatment" in t.required_roles,
                "referral_emphasis": t.lesion_type in knowledge.referral_emphasis,
            }
            for t in all_targets
        ],
        "care_pathways": [
            {
                "lesion_type": pathway.lesion_type,
                "status": pathway.status,
                "clinician_options": list(pathway.clinician_options),
                "reason_codes": list(pathway.reason_codes),
                "policy_source_ids": list(pathway.policy_source_ids),
                "required_answers": list(pathway.required_answers),
            }
            for pathway in analysis.care_pathways
            if pathway.status != "not_detected"
        ],
        "lesion_coverage": _lesion_coverage(analysis, None, can_treat),
        "warnings": warnings,
    }

    candidates = generate_candidates(
        products, selection_targets, knowledge, therapy_primary=therapy_primary,
    )
    gated, profile_vetoes, quality_flags = apply_profile_gates(
        candidates, profile, knowledge, targets=selection_targets,
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
        if can_treat and therapy_primary is not None else []
    )
    document["prescription_options"] = rx_options
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
        targets=selection_targets, profile=profile, knowledge=knowledge,
        category_prices=category_prices,
    )

    if candidate_selector is None:
        archetypes = knowledge.archetypes
    else:
        best_overall = next(
            (archetype for archetype in knowledge.archetypes
             if archetype.get("id") == "best_overall"),
            None,
        )
        if best_overall is None:
            return _selection_unavailable(document, "best_overall_archetype_missing")
        archetypes = [best_overall]
    scar_care_required = any(
        "scar_care" in target.required_roles for target in selection_targets
    )
    if candidate_selector is not None and scar_care_required:
        return _selection_unavailable(document, "unsupported_required_slot:scar_care")
    required_active_slots = ({"scar_care"} if scar_care_required else set())
    archetype_scored = []
    selected_ids: set[str] | None = None
    for archetype in archetypes:
        if scar_care_required and "scar_care" not in archetype.get("slots", []):
            archetype = {**archetype, "slots": [*archetype["slots"], "scar_care"]}
        weights = archetype.get("weights") or knowledge.default_weights
        scored_by_slot = {
            slot: score_products(slot_products, slot, providers, ctx, weights)
            for slot, slot_products in gated.items()
        }
        if candidate_selector is not None:
            context = _selection_context(analysis, profile, document, can_treat)
            missing_pool = next(
                (slot for slot in context["required_slots"] if not scored_by_slot.get(slot)),
                None,
            )
            if missing_pool is not None:
                return _selection_unavailable(
                    document, f"required_candidate_pool_empty:{missing_pool}"
                )
            try:
                selector_versions = {
                    **document["data_versions"],
                    "policy": {
                        "identity": analysis.therapy_policy_identity,
                        "sha256": analysis.therapy_policy_sha256,
                    },
                }
                result = candidate_selector(
                    context, scored_by_slot, selector_versions
                )
            except SelectionUnavailable as exc:
                return _selection_unavailable(document, str(exc), exc.metadata)
            except Exception as exc:
                return _selection_unavailable(
                    document, f"selector_error:{type(exc).__name__}"
                )
            scored_by_slot, error = _selected_candidates(
                result, scored_by_slot, context["required_slots"], can_treat
            )
            if error:
                metadata = result.metadata if isinstance(result, CandidateSelection) else None
                return _selection_unavailable(document, error, metadata)
            selected_ids = {
                product_id for product_id in result.product_ids.values()
                if product_id is not None
            }
            document["selection"] = {
                **result.metadata,
                "status": "ok",
                "product_ids": result.product_ids,
            }
        archetype_scored.append((archetype, scored_by_slot))

    routines = compose_all(
        archetype_scored, all_targets, profile, knowledge, treatment_allowed=can_treat,
    )
    if not can_treat:
        _mark_support_only(routines)

    valid_routines = []
    unavailable = []
    for routine in routines:
        reasons = validate_routine(
            routine, profile, knowledge, has_targets=can_treat,
            required_slots=required_active_slots,
        )
        if reasons:
            unavailable.append({
                "archetype": routine.archetype["id"],
                "reasons": reasons,
            })
        else:
            valid_routines.append(routine)
    if candidate_selector is not None:
        if not valid_routines:
            return _selection_unavailable(
                document, "regimen_validation_failed", document.get("selection")
            )
        if valid_routines[0].product_ids != frozenset(selected_ids or ()):
            return _selection_unavailable(
                document, "selected_product_not_composed", document.get("selection")
            )
        commit = getattr(candidate_selector, "selection_validated", None)
        if commit is not None:
            try:
                commit(result)
            except Exception as exc:
                document["warnings"].append(
                    f"selection cache write failed: {type(exc).__name__}"
                )
    if can_treat and not valid_routines:
        document["treatment_fulfillment"] = {
            "status": "unfilled",
            "reasons": ["required_role_unfilled:treatment"],
        }
        for target in document["target_lesions"]:
            target["selected_for_treatment"] = False
        document["triage"]["see_doctor_note"] = see_doctor_note(
            analysis.triage_level, analysis.referral_reasons,
            analysis.safety_observations, "defer",
        )
        support_routines = compose_all(
            archetype_scored, all_targets, profile, knowledge, treatment_allowed=False,
        )
        _mark_support_only(support_routines)
        valid_routines = [
            routine for routine in support_routines
            if not validate_routine(
                routine, profile, knowledge, has_targets=False,
                required_slots=required_active_slots,
            )
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
    all_serialized = [
        routine_to_dict(r, knowledge, profile, quality_flags) for r in valid_routines
    ]
    serialized = all_serialized[:1]
    if candidate_selector is not None:
        _attach_selector_catalog_facts(serialized, products)
    document["routines"] = serialized
    document["selected_regimen"] = serialized[0] if serialized else None
    document["selected_products"] = (
        {step.slot: step.scored.product.product_id for step in valid_routines[0].steps}
        if valid_routines else {}
    )
    document["lesion_coverage"] = _lesion_coverage(
        analysis, valid_routines[0] if valid_routines else None, can_treat,
    )
    document["alternatives"] = {}
    document["unselected_archetypes"] = [
        {"archetype": routine["archetype"], "reason": "single_regimen_contract"}
        for routine in all_serialized[1:]
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
