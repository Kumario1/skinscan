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
    dedupe_observations,
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
        return load_catalog(path), None
    except FileNotFoundError:
        return None, f"catalog is missing: {path}"
    except json.JSONDecodeError as exc:
        return None, f"catalog contains invalid JSON: {exc}"
    except (OSError, TypeError, ValueError, AssertionError) as exc:
        return None, f"catalog is unreadable or invalid: {exc}"


def routine_payload(
    report: ConcernReport,
    tone: ToneEstimate,
    region_mapping: Mapping[str, object] | str,
    recommendation: Recommendation,
    top: int,
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
    return {
        "schema_version": "2.0",
        "image_id": report.image_id,
        "concerns": [concern_to_dict(concern) for concern in report.concerns],
        "clear_skin": report.clear_skin,
        "notes": report.notes,
        "tone": asdict(tone),
        "region_mapping": (dict(region_mapping)
                           if isinstance(region_mapping, Mapping)
                           else {"method": method}),
        "region_method": method,
        "flags": recommendation.flags,
        "target_actives": recommendation.target_actives,
        "slot_assignment": {
            active: sorted(slots)
            for active, slots in recommendation.slot_assignment.items()
        },
        "routines": {
            slot: {
                category: [product_payload(product) for product in products[:top]]
                for category, products in recommendation.routines[slot].items()
                if products
            }
            for slot in ("AM", "PM")
        },
    }


def _publish_staging(staging: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup = output_dir.with_name(f".{output_dir.name}.backup-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    moved_existing = False
    try:
        if output_dir.exists():
            output_dir.rename(backup)
            moved_existing = True
        staging.rename(output_dir)
    except Exception:
        if output_dir.exists() and moved_existing:
            shutil.rmtree(output_dir)
        if moved_existing and backup.exists():
            backup.rename(output_dir)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


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
    observations = infer_native_tiles(rgb, settings)
    observations = dedupe_observations(
        observations, threshold=settings.dedupe_threshold, preserve_tile_order=True,
    )
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
            recommendation = recommend(report, catalog or [], profile=profile, ranker=None)
            routine = routine_payload(report, tone, region_result.metadata, recommendation, top)
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
        if staging.exists():
            shutil.rmtree(staging)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
