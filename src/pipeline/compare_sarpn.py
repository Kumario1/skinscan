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
ConcernReport (severity from lesion counts, same thresholds as the bridge) ->
issue #1 recommender, so the routine.json outputs are directly comparable.

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
import functools
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..config import load_config
from ..recommendation.bridge import severity_from_count
from ..recommendation.schema import Concern, ConcernReport, UserProfile
from .e2e import load_catalog, routine_payload

# SA-RPN's 10 AcneSCU classes -> the closed concern vocabulary. nevus/other
# are real detections but not treatable concerns — dropped like Not_acne.
SARPN_TO_CONCERN = {
    "closed_comedo": "acne_comedonal",
    "open_comedo": "acne_comedonal",
    "papule": "acne_inflammatory",
    "pustule": "acne_inflammatory",
    "nodule": "acne_cystic",
    "atrophic_scar": "hyperpigmentation",
    "hypertrophic_scar": "hyperpigmentation",
    "melasma": "hyperpigmentation",
}

CONCERN_COLORS = {
    "acne_comedonal": (255, 200, 0),
    "acne_inflammatory": (255, 40, 40),
    "acne_cystic": (200, 0, 255),
    "hyperpigmentation": (40, 120, 255),
    None: (160, 160, 160),
}


def load_rgb(path):
    """EXIF-corrected pixels, same as run_acne04_pipeline (no TF import here)."""
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def api_detect(rgb, url, timeout=300):
    """One image -> SA-RPN detections [{label, score, bbox}] via the REST API."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
    payload = json.dumps({"image": base64.b64encode(buf.getvalue()).decode()}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))["detections"]


def dedupe(dets, thr=0.5):
    # Same suppression as sa-rpn/serve.py, reapplied here because zoom crops
    # and tile overlaps re-introduce duplicates the server can't see across
    # requests. Intersection over the SMALLER box; dets sorted by score desc.
    keep = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        suppressed = False
        for k in keep:
            kx1, ky1, kx2, ky2 = k["bbox"]
            iw = min(x2, kx2) - max(x1, kx1)
            ih = min(y2, ky2) - max(y1, ky1)
            if iw <= 0 or ih <= 0:
                continue
            smaller = min(area, (kx2 - kx1) * (ky2 - ky1)) or 1.0
            if (iw * ih) / smaller > thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
    return keep


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


# --- pipeline B: tile -------------------------------------------------------

def tile_origins(length, tile, stride):
    """Minimal tile count reaching the far edge, evenly spaced so the overlap
    never drops below the requested one (a seam lesion smaller than the
    overlap is always fully inside some tile)."""
    if length <= tile:
        return [0]
    count = -(-(length - tile) // stride) + 1  # ceil division
    return [round(i * (length - tile) / (count - 1)) for i in range(count)]


def tile_pipeline(rgb, detect, *, tile=1024, overlap=128):
    """Native-resolution overlapping chunks of the whole image -> detect ->
    detections offset back to original coordinates."""
    h, w = rgb.shape[:2]
    stride = tile - overlap
    tiles = [(x, y, min(tile, w - x), min(tile, h - y))
             for y in tile_origins(h, tile, stride)
             for x in tile_origins(w, tile, stride)]
    dets = []
    for x, y, tw, th in tiles:
        for d in detect(rgb[y:y + th, x:x + tw]):
            x0, y0, x1, y1 = d["bbox"]
            dets.append({**d, "bbox": [x0 + x, y0 + y, x1 + x, y1 + y]})
    return dets, [(x, y, x + tw, y + th) for x, y, tw, th in tiles]


# --- shared downstream ------------------------------------------------------

def report_from_detections(image_id, dets, regions, thresholds):
    """SA-RPN labels -> ConcernReport; mirrors bridge.build_concern_report but
    takes hard labels+scores instead of classifier prob vectors."""
    groups: dict[tuple[str, str], list[float]] = {}
    dropped = 0
    for d, region in zip(dets, regions):
        concern = SARPN_TO_CONCERN.get(d["label"])
        if concern is None:
            dropped += 1
            continue
        groups.setdefault((concern, region), []).append(d["score"])
    concerns = [Concern(concern=concern, region=region,
                        severity=severity_from_count(len(scores), thresholds),
                        confidence=sum(scores) / len(scores),
                        lesion_count=len(scores))
                for (concern, region), scores in sorted(groups.items())]
    return ConcernReport(image_id=image_id, concerns=concerns,
                         clear_skin=not concerns,
                         notes=f"dropped {dropped} non-concern detection(s) "
                               "(nevus/other)" if dropped else "")


def draw_overlay(rgb, dets, regions, rects, path):
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for rect in rects:
        draw.rectangle(rect, outline=(0, 255, 0), width=1)
    for d, region in zip(dets, regions):
        color = CONCERN_COLORS[SARPN_TO_CONCERN.get(d["label"])]
        draw.rectangle(d["bbox"], outline=color, width=3)
        draw.text((d["bbox"][0], max(d["bbox"][1] - 12, 0)),
                  f"{d['label']} {d['score']:.2f} {region}", fill=color)
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
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def run_downstream(name, rgb, raw_dets, rects, image_name, cfg, out, *,
                   catalog, ranker, skin_type, pregnant, seconds, top):
    """Everything after detection is identical for both pipelines."""
    from ..recommendation.engine import recommend
    from .regions import locate_regions
    from .tone import estimate_tone

    dets = dedupe(sorted(raw_dets, key=lambda d: d["score"], reverse=True))
    boxes = [tuple(d["bbox"]) for d in dets]
    region_result = locate_regions(rgb, boxes)
    thresholds = cfg["concern_report"]["severity_count_thresholds"]
    report = report_from_detections(image_name, dets, region_result.regions,
                                    thresholds)
    draw_overlay(rgb, dets, region_result.regions, rects,
                 out / f"{name}_overlay.jpg")

    result = {
        "pipeline": name,
        "api_calls": len(rects),
        "seconds": round(seconds, 1),
        "raw_detections": len(raw_dets),
        "detections_after_dedupe": len(dets),
        "label_counts": label_counts(dets),
        "detections": [{**d, "region": r, "bbox": [round(v, 1) for v in d["bbox"]]}
                       for d, r in zip(dets, region_result.regions)],
        "region_method": region_result.metadata["method"],
        "concerns": [{"concern": c.concern, "region": c.region,
                      "severity": c.severity, "lesion_count": c.lesion_count,
                      "confidence": round(c.confidence, 3)} for c in report.concerns],
        "notes": report.notes,
    }
    if catalog is not None:
        tone = estimate_tone(rgb, region_result.polygons, boxes)
        profile = UserProfile(skin_type=skin_type, tone_bucket=tone.bucket,
                              tone_source="photo", pregnant_or_nursing=pregnant)
        rec = recommend(report, catalog, profile=profile, ranker=ranker)
        payload = routine_payload(report, tone, region_result.metadata["method"],
                                  rec, top)
        (out / f"routine_{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        result["routine"] = payload
    return result


def yolo_boxes(image_path, weights, cfg):
    from ultralytics import YOLO

    result = YOLO(str(weights)).predict(
        str(image_path), conf=cfg["detection"]["conf_threshold"],
        iou=cfg["detection"]["iou_threshold"], imgsz=cfg["detection"]["img_size"],
        verbose=False)[0]
    return [b.xyxy[0].tolist() for b in result.boxes]


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

    catalog_path = Path(cfg["paths"]["catalog_processed"])
    catalog = load_catalog(cfg) if catalog_path.exists() else None
    if catalog is None:
        print(f"no catalog at {catalog_path} — stopping at concern reports")
    ranker = None
    if catalog is not None:
        from ..recommendation.concern_stats import ConcernStatsRanker
        stats_path = Path(cfg["concern"]["stats_path"])
        if stats_path.exists():
            ranker = ConcernStatsRanker.from_file(stats_path, list(SARPN_TO_CONCERN.values()))

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
            results["zoom"] = run_downstream(
                "zoom", rgb, dets, rects, args.image.name, cfg, out,
                catalog=catalog, ranker=ranker, skin_type=args.skin_type,
                pregnant=args.pregnant,
                seconds=time.perf_counter() - start, top=args.top)
            results["zoom"]["stage1_boxes"] = len(boxes)

    if args.pipeline in ("both", "tile"):
        start = time.perf_counter()
        dets, rects = tile_pipeline(rgb, detect, tile=args.tile,
                                    overlap=args.overlap)
        results["tile"] = run_downstream(
            "tile", rgb, dets, rects, args.image.name, cfg, out,
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
