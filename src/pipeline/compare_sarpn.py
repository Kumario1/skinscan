"""A/B harness for the SA-RPN detector (sa-rpn/serve.py) on one face photo.

Two ways to get consumer-photo lesions in front of a model trained on
1024px clinical tiles:

  zoom  — Stage-1 YOLO finds the acne areas, each area is cropped with
          context and upscaled to the model's 1024px input (lesions get big,
          but only where YOLO already looked);
  tile  — the original image is chunked into native-resolution 1024px tiles
          with overlap and every chunk goes through the model (full coverage,
          but lesions stay small).

Both funnels end in the SAME downstream: detections -> D-020 regions ->
ConcernReport (production SA-RPN severity rules, shared with the e2e
pipeline) -> issue #1 recommender, so the routine.json outputs are directly
comparable. The tile path itself — HTTP calls, response validation,
coordinate restoration, label mapping, and dedupe — is entirely production
code (src.pipeline.sarpn); only the zoom path (a historical alternative,
never shipped) keeps its own crop/upscale geometry.

    python -m src.pipeline.compare_sarpn --image face.jpg \
        [--api http://localhost:8000/predict] [--pipeline both|zoom|tile]

The SA-RPN API runs where the torch1.9/mmcv env lives (Lightning studio);
tunnel it here with:  ssh -L 8000:localhost:8000 <studio>
Catalog/ranker artifacts are optional — without them the run stops at the
concern reports (identification comparison only), same degradation as e2e.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, replace
import functools
import io
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..config import load_config
from ..recommendation.import_catalog import load_catalog
from ..recommendation.schema import UserProfile
from .sarpn import (
    LesionObservation,
    SarpnSettings,
    build_sarpn_concern_report,
    concern_to_dict,
    dedupe_observations,
    infer_native_tiles,
    load_rgb,
    make_tiles,
)

CONCERN_COLORS = {
    "acne_comedonal": (255, 200, 0),
    "acne_inflammatory": (255, 40, 40),
    "acne_cystic": (200, 0, 255),
    "acne_scarring": (120, 80, 40),
    "hyperpigmentation": (40, 120, 255),
    None: (160, 160, 160),
}


def api_detect(rgb, url, timeout=300):
    """One image -> SA-RPN detections [{label, score, bbox}] via the REST API.

    Zoom-only: each crop is a standalone image (no tile bookkeeping), so this
    stays a plain, unvalidated POST rather than the production tile client.
    """
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
    payload = json.dumps({"image": base64.b64encode(buf.getvalue()).decode()}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))["detections"]


# --- pipeline A: zoom -------------------------------------------------------

def _square(box, pad, min_side, shape):
    """Square crop rect around an xyxy box: side = max-dim * pad, clamped
    to [min_side, image min-dim] and shifted fully inside the image."""
    x0, y0, x1, y1 = box
    h, w = shape[:2]
    side = min(max(max(x1 - x0, y1 - y0) * pad, min_side), w, h)
    left = min(max((x0 + x1 - side) / 2, 0), w - side)
    top = min(max((y0 + y1 - side) / 2, 0), h - side)
    return (left, top, left + side, top + side)


def _overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def zoom_crops(boxes, pad, min_side, shape):
    """Merge boxes whose padded squares overlap; one square crop per cluster.
    ponytail: O(N^2) fixpoint merge — N is YOLO boxes on one face, tens at most."""
    clusters = [[list(b)] for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(len(clusters) - 1, i, -1):
                bi = _cluster_bbox(clusters[i])
                bj = _cluster_bbox(clusters[j])
                if _overlaps(_square(bi, pad, min_side, shape),
                             _square(bj, pad, min_side, shape)):
                    clusters[i] += clusters.pop(j)
                    merged = True
    return [_square(_cluster_bbox(c), pad, min_side, shape) for c in clusters]


def _cluster_bbox(boxes):
    xs0, ys0, xs1, ys1 = zip(*boxes)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def zoom_pipeline(rgb, boxes, detect, *, out_size=1024, pad=4.0, min_side=192):
    """YOLO boxes -> clustered square crops, upscaled to out_size -> detect ->
    detections mapped back to original coordinates."""
    crops = zoom_crops(boxes, pad, min_side, rgb.shape)
    dets = []
    for left, top, right, _bottom in crops:
        side = right - left
        crop = rgb[int(round(top)):int(round(top + side)),
                   int(round(left)):int(round(left + side))]
        scaled = np.asarray(Image.fromarray(crop).resize((out_size, out_size),
                                                         Image.LANCZOS))
        scale = side / out_size
        for d in detect(scaled):
            x0, y0, x1, y1 = d["bbox"]
            dets.append({**d, "bbox": [x0 * scale + left, y0 * scale + top,
                                       x1 * scale + left, y1 * scale + top]})
    return dets, crops


def _observations_from_zoom_dicts(dets, rgb_shape, settings):
    """Zoom crops call the API directly and return plain dicts (no tile
    bookkeeping). Wrap them in the same LesionObservation shape the tile path
    uses so dedupe/region/concern-bridge downstream is identical either way.

    The tile path gets score/bbox validation and min_score filtering for
    free from src.pipeline.sarpn (_validated_detections); this path calls
    the raw API directly (see api_detect's docstring) so it must apply the
    same client-side checks itself for the A/B comparison to be symmetric.
    This is the debug harness, so invalid entries are dropped rather than
    aborting the whole comparison."""
    height, width = rgb_shape[:2]
    whole_image = (0, 0, width, height)
    observations = []
    for d in dets:
        score = d.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            continue
        if score < settings.min_score:
            continue
        bbox = d.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in bbox):
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if x2 <= x1 or y2 <= y1:
            continue
        observations.append(
            LesionObservation(d["label"], d["label"], float(score),
                              (x1, y1, x2, y2), 0, whole_image)
        )
    return observations


def _load_optional_catalog(path):
    """Mirrors src.pipeline.e2e.load_optional_catalog's tier-2 merge and
    degrade-on-error behavior (Finding 1) without importing e2e — this
    harness must not depend on the user-facing e2e module by design.
    Returns the merged catalog, or None (with a printed note) when the
    catalog is absent/corrupt, matching this module's docstring promise:
    "same degradation as e2e"."""
    if not path.exists():
        print(f"no catalog at {path} — stopping at concern reports")
        return None
    try:
        products = load_catalog(path)
        tier2_path = path.with_name("catalog_tier2.json")
        if tier2_path.exists():
            products = products + load_catalog(tier2_path)
        return products
    except json.JSONDecodeError as exc:
        print(f"catalog at {path} contains invalid JSON ({exc}) — stopping at concern reports")
        return None
    except (OSError, TypeError, ValueError, AssertionError) as exc:
        print(f"catalog at {path} is unreadable or invalid ({exc}) — stopping at concern reports")
        return None


def yolo_boxes(image_path, weights, cfg):
    from ultralytics import YOLO

    result = YOLO(str(weights)).predict(
        str(image_path), conf=cfg["detection"]["conf_threshold"],
        iou=cfg["detection"]["iou_threshold"], imgsz=cfg["detection"]["img_size"],
        verbose=False)[0]
    return [b.xyxy[0].tolist() for b in result.boxes]


# --- pipeline B: tile --------------------------------------------------------

def run_tile_comparison(rgb, settings):
    """Native-tile detection for the tile pipeline. Tiling, HTTP calls,
    response validation, coordinate restoration, and dedupe are all
    production code (src.pipeline.sarpn) — nothing here duplicates it.

    dedupe=False: infer_native_tiles normally dedupes internally (the
    production contract), which would make raw_detections silently equal
    detections_after_dedupe for this pipeline. Requesting raw observations
    here lets run_downstream's single dedupe_observations() pass produce a
    genuine raw -> deduped delta, same as the zoom pipeline already gets.
    """
    observations = infer_native_tiles(rgb, settings, dedupe=False)
    tiles = make_tiles(rgb.shape, tile_size=settings.tile_size, overlap=settings.tile_overlap)
    rects = [(t.x, t.y, t.x + t.width, t.y + t.height) for t in tiles]
    return observations, rects


# --- shared downstream ------------------------------------------------------

def draw_overlay(rgb, dets, rects, path):
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for rect in rects:
        draw.rectangle(rect, outline=(0, 255, 0), width=1)
    for d in dets:
        color = CONCERN_COLORS.get(d.mapped_concern, CONCERN_COLORS[None])
        draw.rectangle(d.box, outline=color, width=3)
        draw.text((d.box[0], max(d.box[1] - 12, 0)),
                  f"{d.label} {d.score:.2f} {d.region}", fill=color)
    img.save(path, quality=92)


def compare(dets_a, dets_b, iou_thr=0.3):
    """Greedy IoU matching between the two pipelines' detections."""
    def iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
        return inter / (area(a) + area(b) - inter or 1.0)

    pairs = sorted(((iou(a["bbox"], b["bbox"]), i, j)
                    for i, a in enumerate(dets_a) for j, b in enumerate(dets_b)),
                   reverse=True)
    used_a, used_b, same, diff = set(), set(), [], []
    for score, i, j in pairs:
        if score < iou_thr or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        entry = {"iou": round(score, 3), "zoom": dets_a[i], "tile": dets_b[j]}
        (same if dets_a[i]["label"] == dets_b[j]["label"] else diff).append(entry)
    return {
        "matched_same_label": len(same),
        "matched_different_label": len(diff),
        "zoom_only": len(dets_a) - len(used_a),
        "tile_only": len(dets_b) - len(used_b),
        "label_disagreements": diff,
    }


def label_counts(dets):
    counts: dict[str, int] = {}
    for d in dets:
        counts[d.label] = counts.get(d.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _routine_payload(report, tone, region_mapping, recommendation, top):
    """Debug-comparison routine.json — deliberately not imported from e2e
    (this harness must not depend on the user-facing e2e module), but close
    enough in shape to eyeball a diff against a real e2e run."""
    def product_payload(product):
        return {
            "product_id": product.product_id,
            "brand": product.brand,
            "name": product.name,
            "actives": product.actives,
            "price_usd": product.price_usd,
        }

    method = (region_mapping.get("method", "unknown")
              if isinstance(region_mapping, dict) else region_mapping)
    return {
        "schema_version": "2.0",
        "image_id": report.image_id,
        "concerns": [concern_to_dict(c) for c in report.concerns],
        "clear_skin": report.clear_skin,
        "tone": asdict(tone),
        "region_method": method,
        "flags": recommendation.flags,
        "target_actives": recommendation.target_actives,
        "routines": {
            slot: {
                category: [product_payload(p) for p in products[:top]]
                for category, products in recommendation.routines[slot].items()
                if products
            }
            for slot in ("AM", "PM")
        },
    }


def run_downstream(name, rgb, observations, rects, image_name, settings, out, *,
                   catalog, ranker, skin_type, pregnant, seconds, top):
    """Everything after detection is identical for both pipelines: dedupe,
    region mapping, and the SA-RPN concern bridge are all production code."""
    from ..recommendation.engine import recommend
    from .regions import locate_regions
    from .tone import estimate_tone

    dets = dedupe_observations(observations, threshold=settings.dedupe_threshold,
                               preserve_tile_order=True)
    boxes = [d.box for d in dets]
    region_result = locate_regions(rgb, boxes)
    report, dets, safety = build_sarpn_concern_report(
        image_name, dets, region_result.regions, settings.severity)
    draw_overlay(rgb, dets, rects, out / f"{name}_overlay.jpg")

    result = {
        "pipeline": name,
        "api_calls": len(rects),
        "seconds": round(seconds, 1),
        "raw_detections": len(observations),
        "detections_after_dedupe": len(dets),
        "label_counts": label_counts(dets),
        "detections": [{"label": d.label, "score": d.score, "region": d.region,
                        "bbox": [round(v, 1) for v in d.box]} for d in dets],
        "region_method": region_result.metadata["method"],
        "concerns": [{"concern": c.concern, "region": c.region,
                      "severity": c.severity, "lesion_count": c.lesion_count,
                      "confidence": round(c.confidence, 3)} for c in report.concerns],
        "safety_observations": [asdict(item) for item in safety],
    }
    if catalog is not None:
        tone = estimate_tone(rgb, region_result.polygons, boxes)
        profile = UserProfile(skin_type=skin_type, tone_bucket=tone.bucket,
                              tone_source="photo", pregnant_or_nursing=pregnant)
        rec = recommend(report, catalog, profile=profile, ranker=ranker)
        payload = _routine_payload(report, tone, region_result.metadata["method"],
                                   rec, top)
        (out / f"routine_{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        result["routine"] = payload
    return result


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--api", default="http://localhost:8000/predict")
    ap.add_argument("--pipeline", choices=("both", "zoom", "tile"), default="both")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--detector", type=Path, default=Path(cfg["detection"]["weights"]),
                    help="Stage-1 YOLO weights (zoom pipeline only)")
    ap.add_argument("--tile", type=int, default=1024, help="tile size, px")
    ap.add_argument("--overlap", type=int, default=128, help="tile overlap, px")
    ap.add_argument("--zoom-pad", type=float, default=4.0,
                    help="crop side = YOLO cluster max-dim x this")
    ap.add_argument("--zoom-min", type=int, default=192,
                    help="minimum crop side before upscaling, px")
    ap.add_argument("--skin-type", default="combination")
    ap.add_argument("--pregnant", action="store_true")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args(argv)

    out = args.out or Path("runs/sarpn_compare") / args.image.stem
    out.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(args.image)
    detect = functools.partial(api_detect, url=args.api)

    settings = replace(SarpnSettings.from_config(cfg), endpoint_url=args.api,
                       tile_size=args.tile, tile_overlap=args.overlap)
    settings._validate()

    catalog_path = Path(cfg["paths"]["catalog_processed"])
    catalog = _load_optional_catalog(catalog_path)
    # Historical ConcernStatsRanker comparison is not wired up here; it would
    # need an explicit future opt-in rather than loading automatically.
    ranker = None

    results = {}
    if args.pipeline in ("both", "zoom"):
        if not args.detector.exists():
            print(f"SKIP zoom: no YOLO weights at {args.detector} "
                  "(pass --detector or run where the weights live)")
        else:
            start = time.perf_counter()
            boxes = yolo_boxes(args.image, args.detector, cfg)
            dets, rects = zoom_pipeline(rgb, boxes, detect, pad=args.zoom_pad,
                                        min_side=args.zoom_min)
            observations = _observations_from_zoom_dicts(dets, rgb.shape, settings)
            results["zoom"] = run_downstream(
                "zoom", rgb, observations, rects, args.image.name, settings, out,
                catalog=catalog, ranker=ranker, skin_type=args.skin_type,
                pregnant=args.pregnant,
                seconds=time.perf_counter() - start, top=args.top)
            results["zoom"]["stage1_boxes"] = len(boxes)

    if args.pipeline in ("both", "tile"):
        start = time.perf_counter()
        observations, rects = run_tile_comparison(rgb, settings)
        results["tile"] = run_downstream(
            "tile", rgb, observations, rects, args.image.name, settings, out,
            catalog=catalog, ranker=ranker, skin_type=args.skin_type,
            pregnant=args.pregnant,
            seconds=time.perf_counter() - start, top=args.top)

    summary = {"image": str(args.image), "api": args.api, "pipelines": results}
    if len(results) == 2:
        summary["agreement"] = compare(results["zoom"]["detections"],
                                       results["tile"]["detections"])
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    for name, r in results.items():
        print(f"\n[{name}] {r['detections_after_dedupe']} lesions "
              f"({r['raw_detections']} raw, {r['api_calls']} API calls, "
              f"{r['seconds']}s) regions: {r['region_method']}")
        print(f"  types: {r['label_counts']}")
        for c in r["concerns"]:
            print(f"  {c['concern']}@{c['region']}: severity {c['severity']}, "
                  f"{c['lesion_count']} lesion(s), conf {c['confidence']}")
    if "agreement" in summary:
        a = summary["agreement"]
        print(f"\nagreement: {a['matched_same_label']} same-label, "
              f"{a['matched_different_label']} different-label, "
              f"{a['zoom_only']} zoom-only, {a['tile_only']} tile-only")
    print("\nwrote", out / "comparison.json")


if __name__ == "__main__":
    main()
