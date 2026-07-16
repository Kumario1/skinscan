"""Production SA-RPN native-tile analysis and deterministic routine CLI."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence

from ..config import load_config
from ..recommendation.engine import Recommendation as LegacyRecommendation, recommend
from ..recommendation.decision import conservative_unreviewed_policy
from ..recommendation.import_catalog import load_catalog
from ..recommendation.schema import (
    ConcernReport, Product, Recommendation, UserProfile, SKIN_TYPES,
    PREGNANCY_STATUSES,
)
from ..recommendation.therapy import load_therapy_policy
from .provenance import build_provenance, catalog_bundle_identity, file_identity, sha256_file
from .regions import locate_regions
from .sarpn import (
    SarpnSettings,
    build_sarpn_concern_report,
    concern_to_dict,
    draw_detection_overlay,
    draw_lesion_sheet,
    draw_region_overlay,
    infer_native_tiles,
    load_rgb,
    observation_to_dict,
    sanitize_endpoint,
)
from .tone import ToneEstimate, estimate_tone


@dataclass(frozen=True)
class PipelineResult:
    analysis: dict[str, object]
    routine: dict[str, object] | None
    output_dir: Path


def load_optional_catalog(
    path: Path | None,
    tier2_path: Path | None = None,
    drug_path: Path | None = None,
) -> tuple[list[Product] | None, str | None]:
    if path is None:
        return None, "catalog path is missing"
    try:
        products = load_catalog(path)
        if tier2_path is None:
            tier2_path = path.with_name("catalog_tier2.json")
        for additional in (tier2_path, drug_path):
            if additional is not None and additional.exists():
                products += load_catalog(additional)
        return products, None
    except FileNotFoundError:
        return None, f"catalog is missing: {path}"
    except json.JSONDecodeError as exc:
        return None, f"catalog contains invalid JSON: {exc}"
    except (OSError, TypeError, ValueError, AssertionError) as exc:
        return None, f"catalog is unreadable or invalid: {exc}"


# The standalone engine is a subprocess at the end of the pipeline; a wedged one
# must not hold the whole run open. recsys/pipeline.py puts timeout=10 on its own
# git subprocess -- this is the same idiom at the integration seam, with room for
# a real catalog load.
RECSYS_TIMEOUT_SECONDS = 300
RECSYS_STDERR_TAIL_CHARS = 500

# Catalog gaps the drugstore can fill: named OTC pointers for targets no
# stocked product carries (e2e 2026-07-13: adapalene coverage is honestly 0).
_OTC_POINTERS = {
    "adapalene": ("adapalene 0.1% gel is available over the counter "
                  "(e.g., Differin) — ask a pharmacist"),
}


def _compose_notes(
    report: ConcernReport,
    recommendation: LegacyRecommendation,
    target_coverage: Mapping[str, int],
    safety: Sequence[object],
) -> str:
    """Human-readable rationale assembled from what actually happened —
    flags stay machine-terse; notes say why in sentences."""
    notes: list[str] = [report.notes] if report.notes else []
    if recommendation.mode == "soothe_escalation":
        notes.append("Routine held to soothing, barrier-supporting products "
                     "only: this presentation should be reviewed by a "
                     "professional before strong actives are layered on.")
    elif recommendation.mode == "maintenance":
        notes.append("No active concerns found: light maintenance routine only.")
    if "broad inflammation: exfoliating formats excluded" in recommendation.flags:
        notes.append("Inflammation spans several regions, so peel, scrub and "
                     "resurfacing formats were excluded and leave-on "
                     "exfoliants are capped at one per routine.")
    if "broad inflammation: reduced strong-active stacking" in recommendation.flags:
        notes.append("Benzoyl peroxide was set aside in favor of azelaic acid "
                     "to keep the strong-active load down.")
    for active, count in target_coverage.items():
        if count == 0:
            pointer = _OTC_POINTERS.get(active)
            notes.append(f"No product in the catalog carries {active}"
                         + (f"; {pointer}." if pointer else "."))
    nevus_review = any(getattr(item, "code", "") == "nevus_observation"
                       and getattr(item, "professional_review", False)
                       for item in safety)
    if nevus_review and any(c.concern == "hyperpigmentation"
                            for c in report.concerns):
        notes.append("Pigmented-spot caution: mole-like spots were flagged "
                     "for professional review — confirm dark spots are not "
                     "moles before treating them with acids.")
    return " ".join(notes)


def routine_payload(
    report: ConcernReport,
    tone: ToneEstimate,
    region_mapping: Mapping[str, object] | str,
    recommendation: LegacyRecommendation,
    top: int,
    safety: Sequence[object] = (),
) -> dict[str, object]:
    def product_payload(product: Product) -> dict[str, object]:
        return {
            "product_id": product.product_id,
            "brand": product.brand,
            "name": product.name,
            "actives": product.actives,
            "price_usd": product.price_usd,
            **({"comedogenic_flags": product.comedogenic_flags}
               if product.comedogenic_flags else {}),
            **({"tier": 2, "no_outcome_data": True}
               if product.no_outcome_data else {}),
        }

    method = (region_mapping.get("method", "unknown")
              if isinstance(region_mapping, Mapping) else region_mapping)
    shown = {
        slot: {
            category: products[:top]
            for category, products in recommendation.routines[slot].items()
            if products
        }
        for slot in ("AM", "PM")
    }

    # coverage over what the payload actually shows (post-truncation), so a
    # target active nothing carries — the old phantom-centella case — is loud.
    target_coverage = {
        active: len({product.product_id
                     for categories in shown.values()
                     for products in categories.values()
                     for product in products if active in product.actives})
        for active in recommendation.target_actives
    }
    return {
        "schema_version": "2.0",
        "image_id": report.image_id,
        "concerns": [concern_to_dict(concern) for concern in report.concerns],
        "clear_skin": report.clear_skin,
        "notes": _compose_notes(report, recommendation, target_coverage, safety),
        "tone": asdict(tone),
        "region_mapping": (dict(region_mapping)
                           if isinstance(region_mapping, Mapping)
                           else {"method": method}),
        "region_method": method,
        "routine_mode": recommendation.mode,
        "flags": recommendation.flags,
        "target_actives": recommendation.target_actives,
        "target_coverage": target_coverage,
        "slot_assignment": {
            active: sorted(slots)
            for active, slots in recommendation.slot_assignment.items()
        },
        "routines": {
            slot: {
                category: [product_payload(product) for product in products]
                for category, products in categories.items()
            }
            for slot, categories in shown.items()
        },
    }


def v3_routine_payload(
    recommendation: Recommendation,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Serialize one validated v3 regimen; category menus never enter it."""
    if recommendation.validation_errors:
        raise ValueError(
            "invalid recommendation: " + ", ".join(recommendation.validation_errors)
        )
    return {
        **dict(provenance),
        **recommendation.to_dict(),
        "validation_status": "valid",
    }


def recommendation_artifacts(
    recommendation: Recommendation,
    provenance: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object] | None]:
    """Derive compact analysis fields and optional public/debug artifacts."""
    summary = recommendation.eligibility_diagnostics.to_summary(
        list(recommendation.selected_products)
    )
    fields: dict[str, object] = {
        "decision": recommendation.decision.to_dict(),
        "therapy_plan": recommendation.therapy_plan.to_dict(),
        "recommendation_summary": summary,
    }
    if recommendation.validation_errors:
        fields.update({
            "recommendation_status": "invalid",
            "recommendation_reason": "regimen_validation_failed",
            "recommendation_errors": list(recommendation.validation_errors),
        })
        routine = None
    elif summary["missing_roles"]:
        fields.update({
            "recommendation_status": "unavailable",
            "recommendation_reason": "required_roles_unfilled",
        })
        routine = None
    else:
        fields["recommendation_status"] = "complete"
        routine = v3_routine_payload(recommendation, provenance)
    debug = recommendation.eligibility_diagnostics.debug_payload()
    if debug is not None:
        debug = {**debug, "replay_key": provenance.get("replay_key")}
    return fields, routine, debug


def _remove_path(path: Path) -> None:
    """rmtree a directory or unlink a file/symlink; no-op if it doesn't exist."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _check_output_dir_replaceable(output_dir: Path) -> None:
    """Refuse to touch a pre-existing --out path unless it is empty or looks
    like a prior pipeline output (contains analysis.json). Callers must run
    this before any HTTP inference so a doomed run never burns API budget.
    """
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(
            f"--out {output_dir} exists and is not a directory; "
            "pick a new or empty path"
        )
    if not any(output_dir.iterdir()):
        return
    if (output_dir / "analysis.json").exists():
        return
    raise ValueError(
        f"--out {output_dir} already exists, is non-empty, and doesn't look like a "
        "prior pipeline output (no analysis.json found in it); pick a new or empty "
        "directory"
    )


# A backup this fresh could still belong to a concurrent publish's in-flight
# rename (see the two-rename dance below) rather than a genuinely stranded
# crash artifact. Only adopt/clean backups older than this so process A never
# steals process B's live backup out from under it.
_BACKUP_ADOPTION_GRACE_SECONDS = 600


def _is_adoptable_backup(path: Path, *, now: float) -> bool:
    """True once a `.{name}.backup-*` sibling is old enough to be presumed
    stranded rather than belonging to a still-running concurrent publish."""
    try:
        age = now - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age >= _BACKUP_ADOPTION_GRACE_SECONDS


def _publish_staging(staging: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_glob = f".{output_dir.name}.backup-*"
    now = time.time()

    if not output_dir.exists():
        # A prior run may have been SIGKILLed between the two renames below,
        # stranding its result under a backup sibling. Restore the newest
        # one so it reappears as output_dir before we do anything else — but
        # only if it's old enough to rule out a still-running concurrent
        # publish (which also holds a `.name.backup-<pid>` mid-flight).
        stale_backups = sorted(
            (path for path in output_dir.parent.glob(backup_glob)
             if _is_adoptable_backup(path, now=now)),
            key=lambda path: path.stat().st_mtime,
        )
        if stale_backups:
            stale_backups[-1].rename(output_dir)

    # Any other leftover backups old enough to be safely presumed stranded
    # (from any pid, not just ours) are now safe to discard. Fresher ones
    # are left alone — they may be a live peer's in-flight backup.
    for stale in output_dir.parent.glob(backup_glob):
        if _is_adoptable_backup(stale, now=now):
            _remove_path(stale)

    backup = output_dir.with_name(f".{output_dir.name}.backup-{os.getpid()}")
    moved_existing = False
    try:
        if output_dir.exists():
            output_dir.rename(backup)
            # rename preserves the inode mtime — an hour-old output_dir would
            # make a seconds-old backup read as long-stranded to a concurrent
            # run's adoption guard above. Stamp the backup so its mtime
            # records its CREATION moment, which is what the guard measures.
            os.utime(backup)
            moved_existing = True
        staging.rename(output_dir)
    except Exception as exc:
        if moved_existing:
            if not output_dir.exists():
                backup.rename(output_dir)
            elif backup.exists():
                # A concurrent run may have already published its own fresh
                # result at output_dir — never destroy it. Leave our backup
                # on disk instead of silently losing the prior content.
                raise RuntimeError(
                    f"failed to publish results to {output_dir}: {exc}; a concurrent "
                    "run may have already published there. Previous contents were "
                    f"preserved at {backup}"
                ) from exc
            else:
                # Our own backup vanished too (e.g. reclaimed by a peer's
                # cleanup pass) — don't claim contents were preserved
                # somewhere they no longer exist.
                raise RuntimeError(
                    f"failed to publish results to {output_dir}: {exc}; a concurrent "
                    "run may have already published there, and the backup at "
                    f"{backup} is also gone, so the previous contents could not "
                    "be preserved"
                ) from exc
        raise
    else:
        _remove_path(backup)


def _read_git_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"git_commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "dirty": None}


def load_input_profile(
    profile_path: Path | None,
    *,
    skin_type: str | None = None,
    pregnancy_status: str | None = None,
    pregnant: bool = False,
) -> UserProfile:
    """Normalize JSON intake and narrow CLI overrides before detector work."""
    if profile_path is None:
        raw: dict[str, object] = {}
    else:
        try:
            raw_value = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"profile {profile_path}: invalid JSON: {exc}") from exc
        if not isinstance(raw_value, dict):
            raise ValueError("profile: expected a JSON object")
        raw = dict(raw_value)
    if skin_type is not None:
        existing = raw.get("skin_type")
        if existing not in (None, "unknown", skin_type):
            raise ValueError("--skin-type conflicts with profile skin_type")
        raw["skin_type"] = skin_type
    migrated = "pregnant" if pregnant else pregnancy_status
    if migrated is not None:
        existing = raw.get("pregnancy_status")
        if existing not in (None, "unknown", migrated):
            raise ValueError("pregnancy CLI input conflicts with profile pregnancy_status")
        raw["pregnancy_status"] = migrated
    if pregnant and pregnancy_status is not None and pregnancy_status != "pregnant":
        raise ValueError("--pregnant conflicts with --pregnancy-status")
    return UserProfile.from_dict(raw)


def run_pipeline(
    image_path: Path,
    output_dir: Path,
    *,
    settings: SarpnSettings,
    catalog_path: Path | None,
    catalog_tier2_path: Path | None = None,
    catalog_drug_path: Path | None = None,
    face_landmarker_path: Path | None,
    profile: UserProfile | None = None,
    # Historical direct-call compatibility. The CLI always passes profile.
    skin_type: str | None = None,
    pregnant_or_nursing: bool | None = None,
    top: int = 2,
    therapy_policy_path: Path | None = None,
    dataset_name: str = "unknown",
    sample_id: str | None = None,
    dataset_split: str = "unknown",
    split_proof: str | None = None,
    detector_sha256: str | None = None,
    oracle_annotations: Path | None = None,
    clock=lambda: datetime.now(timezone.utc),
    git_reader=_read_git_state,
    eligibility_debug: bool = False,
    recsys_enabled: bool = False,
    recsys_data_root: Path | None = None,
    recsys_catalog: Path | None = None,
    recsys_eligibility_mode: str | None = None,
) -> PipelineResult:
    if profile is None:
        profile = UserProfile(
            skin_type=skin_type or "unknown",
            pregnant_or_nursing=pregnant_or_nursing,
        )
    therapy_policy = load_therapy_policy(therapy_policy_path)
    triage_policy = conservative_unreviewed_policy()
    catalog, catalog_reason = load_optional_catalog(
        catalog_path, catalog_tier2_path, catalog_drug_path
    )
    if catalog_reason is None and not catalog:
        catalog_reason = "catalog is empty"

    rgb = load_rgb(image_path)
    # infer_native_tiles dedupes internally (dedupe=True default) — production
    # dedupe has exactly one owner there; do not dedupe a second time here
    # (Finding 13).
    evidence_source = "oracle" if oracle_annotations is not None else "prediction"
    if oracle_annotations is not None:
        from .oracle import load_voc_oracle_observations
        observations = load_voc_oracle_observations(oracle_annotations)
    else:
        observations = infer_native_tiles(rgb, settings)
    boxes = [observation.box for observation in observations]
    region_result = locate_regions(rgb, boxes, model_path=face_landmarker_path)
    tone = estimate_tone(rgb, region_result.polygons, boxes)
    report, observations, safety = build_sarpn_concern_report(
        image_path.name,
        observations,
        region_result.regions,
        settings.severity,
        low_light_flag=bool(tone.low_light),
        evidence_source=("annotation_oracle" if oracle_annotations else "prediction"),
    )

    from ..recommendation.decision import decide_care
    from ..recommendation.therapy import plan_therapy
    decision = decide_care(report, triage_policy)
    therapy_plan = plan_therapy(decision, report, profile, therapy_policy)
    if decision.therapy_disposition == "active_treatment" and therapy_plan.primary is None:
        decision = replace(decision, therapy_disposition="defer")

    runtime_config = load_config()
    provenance = build_provenance(
        {
            "source_image_sha256": sha256_file(image_path),
            "evidence_source": evidence_source,
            "oracle_annotations": file_identity(oracle_annotations),
            "dataset": {
                "name": dataset_name,
                "sample_id": sample_id or image_path.stem,
                "split": dataset_split,
                "split_proof": split_proof,
            },
            "input_profile": profile.to_dict(),
            "effective_config": {
                "pipeline": ("acnescu-voc-oracle" if oracle_annotations
                             else "sa-rpn-native-tiles"),
                "endpoint": sanitize_endpoint(settings.endpoint_url),
                "tile_size": settings.tile_size,
                "overlap": settings.tile_overlap,
                "minimum_score": settings.min_score,
                "class_min_scores": dict(settings.class_min_scores),
                "dedupe_threshold": settings.dedupe_threshold,
                "severity": settings.severity,
                "regions": runtime_config["regions"],
                "tone": runtime_config["tone"],
                "classification_crop_pad": runtime_config["classification"]["crop_pad"],
                "face_landmarker": file_identity(face_landmarker_path),
            },
            "models": {
                "detector": ({
                    "state": "not_applicable", "sha256": None,
                    "identity": "annotation_oracle",
                } if oracle_annotations else {
                    "sha256": detector_sha256,
                    "identity": "remote_sa_rpn" if detector_sha256 else "unknown",
                }),
                "classifier": {"state": "not_applicable", "sha256": None},
            },
            "catalog": catalog_bundle_identity(
                catalog_path, catalog_tier2_path, catalog_drug_path
            ),
            "ranker": {"state": "none", "sha256": None},
            "policies": {
                "triage": {
                    "identity": triage_policy.identifier,
                    "reviewed": triage_policy.approved,
                    "sha256": None,
                },
                "therapy": {
                    **file_identity(therapy_policy_path),
                    "identity": therapy_policy.identifier,
                    "reviewed": therapy_policy.reviewed,
                },
            },
        },
        clock=clock,
        git_reader=git_reader,
    )

    analysis: dict[str, object] = {
        **provenance,
        "image_id": image_path.name,
        "pipeline": {
            "identifier": ("acnescu-voc-oracle" if oracle_annotations
                           else "sa-rpn-native-tiles"),
            "endpoint": sanitize_endpoint(settings.endpoint_url),
            "tile_size": settings.tile_size,
            "overlap": settings.tile_overlap,
            "minimum_score": settings.min_score,
            "dedupe_threshold": settings.dedupe_threshold,
        },
        "detections": [observation_to_dict(item) for item in observations],
        "concerns": [concern_to_dict(concern) for concern in report.concerns],
        "clear_skin": report.clear_skin,
        "skin_tone": asdict(tone),
        "region_mapping": dict(region_result.metadata),
        "safety_observations": [asdict(item) for item in safety],
        "decision": decision.to_dict(),
        "therapy_plan": therapy_plan.to_dict(),
        "recommendation_status": "unavailable",
    }

    routine: dict[str, object] | None = None
    debug_rejections: dict[str, object] | None = None
    if catalog_reason:
        analysis["recommendation_reason"] = catalog_reason
    else:
        try:
            recommendation = recommend(
                report, catalog or [], profile,
                triage_policy=triage_policy,
                therapy_policy=therapy_policy,
                concern_scorer=None,
                pooled_ranker=None,
                collect_eligibility_details=eligibility_debug,
            )
            fields, routine, debug_rejections = recommendation_artifacts(
                recommendation, provenance
            )
            analysis.update(fields)
        except Exception as exc:
            analysis["recommendation_reason"] = f"recommendation unavailable: {exc}"

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        draw_detection_overlay(rgb, observations, staging / "detections.jpg")
        draw_region_overlay(rgb, observations, region_result, staging / "region_overlay.jpg")
        draw_lesion_sheet(rgb, observations, staging / "lesion_sheet.jpg")
        (staging / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
        if routine is not None:
            (staging / "routine.json").write_text(json.dumps(routine, indent=2) + "\n")
        if debug_rejections is not None:
            (staging / "eligibility_rejections.json").write_text(
                json.dumps(debug_rejections, indent=2) + "\n"
            )
        if recsys_enabled:
            recommendations_path = staging / "recommendations.json"
            if recsys_data_root is None and recsys_catalog is None:
                _write_recsys_unavailable(
                    recommendations_path,
                    "standalone catalog not configured; pass --recsys-catalog or "
                    "--recsys-data-root",
                )
            elif recsys_data_root is None:
                # A catalog without its data root strands every signal store: the
                # stores are keyed by the sha256 of the catalog they were built
                # against, so they load from the default data root, mismatch, and
                # are skipped with only a warning -- leaving the ranker to score a
                # neutral 0.5 on every store-backed signal while still reporting
                # 'partial' with priced routines, exactly what a healthy run
                # reports. Refusing here is the same fail-closed treatment the
                # neither-configured case above already gets, and the worse of the
                # two cases: that one fails loudly, this one does not fail at all.
                _write_recsys_unavailable(
                    recommendations_path,
                    "standalone signal stores are keyed to a catalog; pass "
                    "--recsys-data-root alongside --recsys-catalog",
                )
            else:
                command = [
                    sys.executable, "-m", "recsys", "recommend",
                    "--analysis", str(staging / "analysis.json"),
                    "--out", str(recommendations_path),
                ]
                if recsys_data_root is not None:
                    command += ["--data-root", str(recsys_data_root)]
                if recsys_catalog is not None:
                    command += ["--catalog", str(recsys_catalog)]
                if recsys_eligibility_mode is not None:
                    command += ["--eligibility-mode", recsys_eligibility_mode]
                try:
                    completed = subprocess.run(
                        command,
                        cwd=Path(__file__).resolve().parents[2],
                        capture_output=True,
                        text=True,
                        timeout=RECSYS_TIMEOUT_SECONDS,
                    )
                except subprocess.TimeoutExpired:
                    # Without a timeout a wedged child hangs run_pipeline forever
                    # with no diagnostic at all. Unavailable-with-a-reason is the
                    # contract for every other way this step can fail.
                    _write_recsys_unavailable(
                        recommendations_path,
                        "standalone recommendation process did not finish within "
                        f"{RECSYS_TIMEOUT_SECONDS}s",
                    )
                else:
                    if completed.returncode:
                        _write_recsys_unavailable(
                            recommendations_path,
                            "standalone recommendation process exited with status "
                            f"{completed.returncode}"
                            + _stderr_tail(completed.stderr),
                        )
        _publish_staging(staging, output_dir)
    finally:
        _remove_path(staging)
    return PipelineResult(analysis, routine, output_dir)


def _stderr_tail(stderr: str | None, limit: int = RECSYS_STDERR_TAIL_CHARS) -> str:
    """The child's own diagnostic, appended to the reason.

    capture_output=True puts the reason the run failed in stderr and then only the
    returncode reached the operator: a missing catalog arrived as 'exited with
    status 1', and a contract_violation:<field> would have been just as invisible.
    The bytes are already in memory; the tail is the part that names the cause.
    """
    text = (stderr or "").strip()
    if not text:
        return ""
    if len(text) > limit:
        text = "..." + text[-limit:]
    return f": {text}"


def _write_recsys_unavailable(path: Path, reason: str) -> None:
    path.write_text(json.dumps({
        "schema_version": "recsys-1",
        "status": "unavailable",
        "reason": reason,
        "routines": [],
    }, indent=2) + "\n")


def _parser(config: dict[str, object]) -> argparse.ArgumentParser:
    sa_rpn = config["sa_rpn"]
    paths = config["paths"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default runs/e2e/<image stem>)")
    parser.add_argument("--api", default=sa_rpn["endpoint_url"])
    parser.add_argument("--catalog", type=Path, default=Path(paths["catalog_processed"]))
    parser.add_argument("--catalog-tier2", type=Path, default=None)
    parser.add_argument("--catalog-drug", type=Path, default=None)
    parser.add_argument("--eligibility-debug", action="store_true")
    parser.add_argument("--recsys", action="store_true",
                        help="also write standalone recsys recommendations.json")
    parser.add_argument("--recsys-data-root", type=Path, default=None)
    parser.add_argument("--recsys-catalog", type=Path, default=None)
    parser.add_argument("--recsys-eligibility-mode", default=None,
                        choices=("strict", "hybrid"),
                        help="passed through to recsys; hybrid opens the whole "
                             "catalog by category and lists prescription options")
    parser.add_argument("--face-landmarker", type=Path, default=Path(paths["face_landmarker"]))
    parser.add_argument("--tile-size", type=int, default=sa_rpn["tile_size"])
    parser.add_argument("--overlap", type=int, default=sa_rpn["tile_overlap"])
    parser.add_argument("--connect-timeout", type=float,
                        default=sa_rpn["connect_timeout_seconds"])
    parser.add_argument("--read-timeout", type=float,
                        default=sa_rpn["read_timeout_seconds"])
    parser.add_argument("--request-batch-size", type=int,
                        default=sa_rpn["request_batch_size"])
    parser.add_argument("--min-score", type=float, default=sa_rpn["min_score"])
    parser.add_argument("--dedupe-threshold", type=float,
                        default=sa_rpn["dedupe_threshold"])
    parser.add_argument("--profile", type=Path, default=None,
                        help="full explicit safety-profile JSON")
    parser.add_argument("--skin-type", choices=sorted(SKIN_TYPES), default=None,
                        help="narrow override; default remains unknown")
    parser.add_argument("--pregnancy-status", choices=sorted(PREGNANCY_STATUSES),
                        default=None)
    parser.add_argument("--pregnant", action="store_true")
    parser.add_argument("--therapy-policy", type=Path, default=None,
                        help="clinician-reviewed therapy policy JSON; missing defers therapy")
    parser.add_argument("--dataset-name", default="unknown")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "valid", "test", "external", "smoke", "unknown"),
        default="unknown",
    )
    parser.add_argument("--split-proof", default=None)
    parser.add_argument("--detector-sha256", default=None,
                        help="immutable remote detector artifact hash")
    parser.add_argument("--oracle-annotations", type=Path, default=None,
                        help="evaluation-only AcneSCU VOC XML; derives oracle evidence")
    parser.add_argument("--top", type=int, default=2,
                        help="legacy display compatibility; v3 selects one product per role")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config()
    args = _parser(config).parse_args(argv)
    output_dir = args.out or Path("runs/e2e") / args.image.stem
    default_catalog = Path(config["paths"]["catalog_processed"])
    catalog_tier2 = args.catalog_tier2
    catalog_drug = args.catalog_drug
    if args.catalog == default_catalog:
        catalog_tier2 = catalog_tier2 or Path(config["paths"]["catalog_tier2"])
        catalog_drug = catalog_drug or Path(config["paths"]["catalog_drug"])
    try:
        profile = load_input_profile(
            args.profile,
            skin_type=args.skin_type,
            pregnancy_status=args.pregnancy_status,
            pregnant=args.pregnant,
        )
        _check_output_dir_replaceable(output_dir)
        base = SarpnSettings.from_config(config)
        settings = replace(
            base,
            endpoint_url=args.api,
            tile_size=args.tile_size,
            tile_overlap=args.overlap,
            connect_timeout_seconds=args.connect_timeout,
            read_timeout_seconds=args.read_timeout,
            request_batch_size=args.request_batch_size,
            min_score=args.min_score,
            dedupe_threshold=args.dedupe_threshold,
        )
        settings._validate()
        result = run_pipeline(
            args.image,
            output_dir,
            settings=settings,
            catalog_path=args.catalog,
            catalog_tier2_path=catalog_tier2,
            catalog_drug_path=catalog_drug,
            face_landmarker_path=args.face_landmarker,
            profile=profile,
            top=args.top,
            therapy_policy_path=args.therapy_policy,
            dataset_name=args.dataset_name,
            sample_id=args.sample_id,
            dataset_split=args.dataset_split,
            split_proof=args.split_proof,
            detector_sha256=args.detector_sha256,
            oracle_annotations=args.oracle_annotations,
            eligibility_debug=args.eligibility_debug,
            recsys_enabled=args.recsys,
            recsys_data_root=args.recsys_data_root,
            recsys_catalog=args.recsys_catalog,
            recsys_eligibility_mode=args.recsys_eligibility_mode,
        )
    except Exception as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1

    count = len(result.analysis["detections"])
    status = result.analysis["recommendation_status"]
    decision = result.analysis["decision"]
    release = result.analysis["release_eligibility"]
    print(
        f"wrote {result.output_dir}: {count} detections, "
        f"triage {decision['triage_level']}, therapy {decision['therapy_disposition']}, "
        f"recommendation {status}, release eligible={release['eligible']}"
    )
    for reason in decision["referral_reasons"]:
        print(f"  ⚑ referral: {reason}")
    _print_safety_escalations(result.analysis)
    if result.routine is not None:
        for flag in result.routine["flags"]:
            print(f"  ⚑ {flag}")
    return 0


def _print_safety_escalations(analysis: Mapping[str, object]) -> None:
    """Surface derm-escalation signals straight from the analysis, so they
    show up even when no catalog/routine is available (Findings 6+7)."""
    for concern in analysis["concerns"]:
        if concern["severity"] >= 4 or concern["concern"] == "acne_cystic":
            print(f"  ⚑ severe {concern['concern']} — see a dermatologist")
    for observation in analysis["safety_observations"]:
        if observation["professional_review"]:
            print(f"  ⚑ {observation['code']}: professional review recommended")


if __name__ == "__main__":
    raise SystemExit(main())
