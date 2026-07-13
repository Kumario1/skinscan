"""Production SA-RPN native-tile analysis and deterministic routine CLI."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence

from ..config import load_config
from ..recommendation.engine import Recommendation, recommend
from ..recommendation.import_catalog import load_catalog
from ..recommendation.schema import ConcernReport, Product, UserProfile, SKIN_TYPES
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


def load_optional_catalog(path: Path | None) -> tuple[list[Product] | None, str | None]:
    if path is None:
        return None, "catalog path is missing"
    try:
        products = load_catalog(path)
        tier2_path = path.with_name("catalog_tier2.json")
        if tier2_path.exists():
            products = products + load_catalog(tier2_path)
        return products, None
    except FileNotFoundError:
        return None, f"catalog is missing: {path}"
    except json.JSONDecodeError as exc:
        return None, f"catalog contains invalid JSON: {exc}"
    except (OSError, TypeError, ValueError, AssertionError) as exc:
        return None, f"catalog is unreadable or invalid: {exc}"


# Catalog gaps the drugstore can fill: named OTC pointers for targets no
# stocked product carries (e2e 2026-07-13: adapalene coverage is honestly 0).
_OTC_POINTERS = {
    "adapalene": ("adapalene 0.1% gel is available over the counter "
                  "(e.g., Differin) — ask a pharmacist"),
}


def _compose_notes(
    report: ConcernReport,
    recommendation: Recommendation,
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
    recommendation: Recommendation,
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


def run_pipeline(
    image_path: Path,
    output_dir: Path,
    *,
    settings: SarpnSettings,
    catalog_path: Path | None,
    face_landmarker_path: Path | None,
    skin_type: str,
    pregnant_or_nursing: bool,
    top: int,
) -> PipelineResult:
    rgb = load_rgb(image_path)
    # infer_native_tiles dedupes internally (dedupe=True default) — production
    # dedupe has exactly one owner there; do not dedupe a second time here
    # (Finding 13).
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
    )

    analysis: dict[str, object] = {
        "schema_version": "2.0",
        "image_id": image_path.name,
        "pipeline": {
            "identifier": "sa-rpn-native-tiles",
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
        "recommendation_status": "unavailable",
    }

    routine: dict[str, object] | None = None
    catalog, catalog_reason = load_optional_catalog(catalog_path)
    if catalog_reason is None and not catalog:
        catalog_reason = "catalog is empty"
    if catalog_reason:
        analysis["recommendation_reason"] = catalog_reason
    else:
        try:
            profile = UserProfile(
                skin_type=skin_type,
                tone_bucket=tone.bucket,
                tone_source="photo",
                pregnant_or_nursing=pregnant_or_nursing,
            )
            # Lazy import: e2e must not load the ranker module (pandas/sklearn)
            # unless a recommendation is actually produced (see the forbidden-
            # imports test). load_ranker resolves Ranker/StatsRanker/None per
            # D-022; a load failure degrades to rules-only order (D-019).
            try:
                from ..recommendation.ranker import load_ranker
                ranker = load_ranker()
            except Exception:
                ranker = None
            recommendation = recommend(report, catalog or [], profile=profile, ranker=ranker)
            routine = routine_payload(report, tone, region_result.metadata,
                                      recommendation, top, safety=safety)
            analysis["recommendation_status"] = "complete"
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
        _publish_staging(staging, output_dir)
    finally:
        _remove_path(staging)
    return PipelineResult(analysis, routine, output_dir)


def _parser(config: dict[str, object]) -> argparse.ArgumentParser:
    sa_rpn = config["sa_rpn"]
    paths = config["paths"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default runs/e2e/<image stem>)")
    parser.add_argument("--api", default=sa_rpn["endpoint_url"])
    parser.add_argument("--catalog", type=Path, default=Path(paths["catalog_processed"]))
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
    parser.add_argument("--skin-type", choices=sorted(SKIN_TYPES), default="combination")
    parser.add_argument("--pregnant", action="store_true")
    parser.add_argument("--top", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config()
    args = _parser(config).parse_args(argv)
    output_dir = args.out or Path("runs/e2e") / args.image.stem
    try:
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
            face_landmarker_path=args.face_landmarker,
            skin_type=args.skin_type,
            pregnant_or_nursing=args.pregnant,
            top=args.top,
        )
    except Exception as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1

    count = len(result.analysis["detections"])
    status = result.analysis["recommendation_status"]
    print(f"wrote {result.output_dir}: {count} detections, recommendation {status}")
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
