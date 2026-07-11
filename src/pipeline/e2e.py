"""Full-pipeline CLI: face image -> detector -> type classifier -> regions ->
tone -> ConcernReport -> ranked AM/PM routine (issue #1 recommender over the
issue #6 geometry).

    python -m src.pipeline.e2e --image photo.jpg [--top 5] [--skin-type dry]

Writes <out>/routine.json (plus the detector/classifier artifacts from
run_acne04_pipeline) and prints a readable summary. The concern-stats ranker
and tier-2 catalog are used when their processed files exist, else skipped —
same graceful degradation as the importer (D-024).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..recommendation.bridge import build_concern_report
from ..recommendation.concern_stats import ConcernStatsRanker
from ..recommendation.engine import recommend
from ..recommendation.schema import Product, UserProfile


def load_catalog(cfg) -> list[Product]:
    catalog_path = Path(cfg["paths"]["catalog_processed"])
    products = [Product(**p) for p in json.loads(catalog_path.read_text())]
    tier2_path = catalog_path.with_name("catalog_tier2.json")
    if tier2_path.exists():
        products += [Product(**p) for p in json.loads(tier2_path.read_text())]
    return products


def routine_payload(report, tone, region_method, rec, top: int) -> dict:
    def fmt(p: Product) -> dict:
        return {
            "product_id": p.product_id, "brand": p.brand, "name": p.name,
            "actives": p.actives, "price_usd": p.price_usd,
            **({"comedogenic_flags": p.comedogenic_flags} if p.comedogenic_flags else {}),
            **({"tier": 2, "no_outcome_data": True} if p.no_outcome_data else {}),
        }

    return {
        "image_id": report.image_id,
        "concerns": [{"concern": c.concern, "region": c.region, "severity": c.severity,
                      "lesion_count": c.lesion_count, "confidence": round(c.confidence, 3)}
                     for c in report.concerns],
        "clear_skin": report.clear_skin,
        "notes": report.notes,
        "tone": {"bucket": tone.bucket, "ita": tone.ita, "low_light": tone.low_light},
        "region_method": region_method,
        "flags": rec.flags,
        "target_actives": rec.target_actives,
        "slot_assignment": {a: sorted(s) for a, s in rec.slot_assignment.items()},
        "routines": {slot: {cat: [fmt(p) for p in prods[:top]]
                            for cat, prods in rec.routines[slot].items() if prods}
                     for slot in ("AM", "PM")},
    }


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default runs/e2e/<image stem>)")
    ap.add_argument("--detector", type=Path, default=Path(cfg["detection"]["weights"]))
    ap.add_argument("--classifier", type=Path, default=Path(cfg["classification"]["weights"]))
    ap.add_argument("--skin-type", default="combination",
                    help="review-vocabulary skin type for the ranker sub-cells")
    ap.add_argument("--pregnant", action="store_true")
    ap.add_argument("--top", type=int, default=5, help="products per category in the output")
    args = ap.parse_args(argv)
    out = args.out or Path("runs/e2e") / args.image.stem

    # heavy imports stay lazy so --help works without TF/ultralytics
    from ultralytics import YOLO

    from ..classification.classifier import AcneTypeClassifier
    from ..classification.run_acne04_pipeline import analyze_image, load_rgb
    from .regions import locate_regions
    from .tone import estimate_tone

    out.mkdir(parents=True, exist_ok=True)
    clf = AcneTypeClassifier(args.classifier)
    record = analyze_image(
        args.image, YOLO(str(args.detector)), clf, out,
        crop_size=clf.image_size, crop_pad=cfg["classification"]["crop_pad"],
        max_boxes=16, conf=cfg["detection"]["conf_threshold"],
        iou=cfg["detection"]["iou_threshold"], imgsz=cfg["detection"]["img_size"],
        collage_tiles=9,
    )
    (out / "predictions.json").write_text(json.dumps([record], indent=2) + "\n")

    rgb = load_rgb(args.image)
    boxes = [tuple(d["box"]) for d in record["detections"]]
    region_result = locate_regions(rgb, boxes)
    tone = estimate_tone(rgb, region_result.polygons, boxes)

    report = build_concern_report(args.image.name,
                                  [d["probs"] for d in record["detections"]],
                                  list(region_result.regions),
                                  low_light_flag=bool(tone.low_light))
    profile = UserProfile(skin_type=args.skin_type, tone_bucket=tone.bucket,
                          tone_source="photo", pregnant_or_nursing=args.pregnant)

    stats_path = Path(cfg["concern"]["stats_path"])
    ranker = (ConcernStatsRanker.from_file(stats_path, [c.concern for c in report.concerns])
              if stats_path.exists() else None)
    rec = recommend(report, load_catalog(cfg), profile=profile, ranker=ranker)

    payload = routine_payload(report, tone, region_result.metadata["method"], rec, args.top)
    (out / "routine.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n{args.image.name}: {len(boxes)} detections | regions: {payload['region_method']}"
          f" | tone: {tone.bucket} | ranker: {'concern-stats' if ranker else 'rules-only'}")
    for c in report.concerns:
        print(f"  {c.concern}@{c.region}: severity {c.severity}, "
              f"{c.lesion_count} lesion(s), conf {c.confidence:.2f}")
    for flag in rec.flags:
        print(f"  ⚑ {flag}")
    print(f"  target actives: {', '.join(rec.target_actives)}")
    for slot in ("AM", "PM"):
        print(f"  {slot}:")
        for cat, prods in payload["routines"][slot].items():
            names = "; ".join(f"{p['brand']} {p['name']}" for p in prods[:3])
            more = f" (+{len(prods) - 3} more)" if len(prods) > 3 else ""
            print(f"    {cat}: {names}{more}")
    print("wrote", out / "routine.json")


if __name__ == "__main__":
    main()
