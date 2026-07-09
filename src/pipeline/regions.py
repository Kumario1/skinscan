"""Assign detector boxes to the closed face-region vocabulary (D-020)."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path as FilePath

import numpy as np
from matplotlib.path import Path

from ..config import load_config


Box = tuple[float, float, float, float]
Polygon = Sequence[tuple[float, float]]

# Candidate table rendered in the issue #6 self-collected-photo overlay.
# Each sequence follows the visible boundary of one semantic region; the table
# remains subject to the issue's human visual-approval gate.
LANDMARK_INDEX_TABLE = {
    "forehead": (21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251,
                 300, 293, 334, 296, 336, 107, 66, 105, 63, 70, 46),
    "nose": (168, 6, 197, 195, 5, 4, 275, 440, 344, 278, 294, 327,
             326, 94, 97, 98, 64, 48, 115, 220, 45),
    # MediaPipe uses anatomical left/right; image-left is the subject's right.
    "right_cheek": (21, 162, 127, 234, 93, 132, 58, 61, 205, 50, 101,
                    123, 116, 118, 119, 120, 121, 133, 155, 154, 153,
                    145, 144, 163, 7, 33, 70),
    "left_cheek": (251, 389, 356, 454, 323, 361, 288, 291, 425, 280,
                   330, 352, 345, 347, 348, 349, 350, 362, 382, 381, 380,
                   374, 373, 390, 249, 263, 300),
    "chin_jaw": (58, 172, 136, 150, 149, 176, 148, 152, 377, 400,
                 378, 379, 365, 397, 288, 291, 17, 61),
}
LIP_LANDMARKS = (0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84,
                 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270,
                 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375,
                 402, 405, 409, 415)


@dataclass
class RegionResult:
    regions: list[str]
    polygons: dict[str, Polygon]
    metadata: dict[str, object]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Small monotonic-chain hull; enough for the lip outline."""
    ordered = sorted(map(tuple, points))

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def landmark_polygons(landmarks_xy: np.ndarray) -> dict[str, Polygon]:
    """Build the six D-020 polygons from pixel-space FaceLandmarker points."""
    if len(landmarks_xy) < 468:
        raise ValueError("FaceLandmarker must return at least 468 landmarks")
    polygons = {
        region: landmarks_xy[list(indices)]
        for region, indices in LANDMARK_INDEX_TABLE.items()
    }
    lips = _convex_hull(landmarks_xy[list(LIP_LANDMARKS)])
    center = lips.mean(axis=0)
    polygons["perioral"] = center + (lips - center) * 1.55
    # Insertion order resolves the small nose/far-cheek and mouth/jaw overlaps.
    return {region: polygons[region] for region in (
        "forehead", "nose", "right_cheek", "left_cheek", "perioral", "chin_jaw"
    )}


def grid_polygons(image_shape: Sequence[int], face_box: Box | None = None) -> dict[str, Polygon]:
    """Return the deterministic D-020 thirds grid, optionally inside a face box."""
    height, width = image_shape[:2]
    x0, y0, x1, y1 = face_box or (0, 0, width, height)
    third_x = (x1 - x0) / 3
    third_y = (y1 - y0) / 3
    xl, xr = x0 + third_x, x1 - third_x
    yt, yb = y0 + third_y, y1 - third_y

    def rectangle(left: float, top: float, right: float, bottom: float) -> Polygon:
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    # perioral precedes chin_jaw because the latter spans the full lower third.
    return {
        "forehead": rectangle(x0, y0, x1, yt),
        "nose": rectangle(xl, yt, xr, yb),
        "right_cheek": rectangle(x0, yt, xl, yb),
        "left_cheek": rectangle(xr, yt, x1, yb),
        "perioral": rectangle(xl, yb, xr, y1),
        "chin_jaw": rectangle(x0, yb, x1, y1),
    }


def _distance_to_polygon(point: np.ndarray, polygon: Polygon) -> float:
    vertices = np.asarray(polygon, dtype=float)
    starts, ends = vertices, np.roll(vertices, -1, axis=0)
    segments = ends - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    offsets = point - starts
    fractions = np.divide(
        np.sum(offsets * segments, axis=1),
        lengths_squared,
        out=np.zeros_like(lengths_squared),
        where=lengths_squared != 0,
    )
    closest = starts + np.clip(fractions, 0, 1)[:, None] * segments
    return float(np.min(np.linalg.norm(point - closest, axis=1)))


def assign_regions(boxes: Sequence[Box], polygons: Mapping[str, Polygon]) -> list[str]:
    """Assign each xyxy box by centroid point-in-polygon."""
    paths = {region: Path(points) for region, points in polygons.items()}
    assigned = []
    for x0, y0, x1, y1 in boxes:
        centroid = np.array(((x0 + x1) / 2, (y0 + y1) / 2))
        inside = next((region for region, path in paths.items()
                       if path.contains_point(centroid)), None)
        assigned.append(inside or min(polygons, key=lambda region:
                                      _distance_to_polygon(centroid, polygons[region])))
    return assigned


def fallback_regions(image_shape: Sequence[int], boxes: Sequence[Box], *,
                     face_box: Box | None = None, reason: str) -> RegionResult:
    """Assign through the grid and return loud fallback metadata."""
    polygons = grid_polygons(image_shape, face_box)
    return RegionResult(
        regions=assign_regions(boxes, polygons),
        polygons=polygons,
        metadata={
            "method": "grid_fallback",
            "fallback": True,
            "reason": reason,
            "face_detected": face_box is not None,
            "face_box": list(face_box) if face_box is not None else None,
        },
    )


def _detect_landmarks(image_rgb: np.ndarray, model_path: FilePath):
    """Run current MediaPipe Tasks lazily; imports stay out of fast tests."""
    if not model_path.exists():
        return None, f"missing FaceLandmarker model artifact: {model_path}"
    try:
        import mediapipe as mp
    except ImportError:
        return None, "MediaPipe unavailable"

    try:
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        with mp.tasks.vision.FaceLandmarker.create_from_model_path(str(model_path)) as landmarker:
            faces = landmarker.detect(image).face_landmarks
    except (OSError, RuntimeError, ValueError) as error:
        return None, f"FaceLandmarker failed: {error}"
    if not faces:
        return None, "FaceLandmarker found no face"

    height, width = image_rgb.shape[:2]
    return np.asarray([(point.x * width, point.y * height) for point in faces[0]]), None


def locate_regions(image_rgb: np.ndarray, boxes: Sequence[Box], *,
                   face_box: Box | None = None,
                   model_path: FilePath | None = None) -> RegionResult:
    """Assign regions with landmarks, degrading loudly to the deterministic grid."""
    if face_box is not None:
        return fallback_regions(
            image_rgb.shape, boxes, face_box=face_box,
            reason="manual face box requested; using grid fallback",
        )

    model_path = model_path or FilePath(load_config()["paths"]["face_landmarker"])
    landmarks, reason = _detect_landmarks(image_rgb, model_path)
    if landmarks is None:
        return fallback_regions(image_rgb.shape, boxes, reason=reason)

    polygons = landmark_polygons(landmarks)
    minimum = landmarks.min(axis=0)
    maximum = landmarks.max(axis=0)
    return RegionResult(
        regions=assign_regions(boxes, polygons),
        polygons=polygons,
        metadata={
            "method": "mediapipe_face_landmarker",
            "fallback": False,
            "reason": None,
            "face_detected": True,
            "face_box": [float(minimum[0]), float(minimum[1]),
                         float(maximum[0]), float(maximum[1])],
            "model_path": str(model_path),
        },
    )


def render_debug_overlay(image_rgb: np.ndarray, boxes: Sequence[Box], output: FilePath, *,
                         face_box: Box | None = None,
                         model_path: FilePath | None = None) -> RegionResult:
    """Render the issue #6 eyeball gate: regions plus labeled detector boxes."""
    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch, Rectangle

    result = locate_regions(image_rgb, boxes, face_box=face_box, model_path=model_path)
    figure, axes = plt.subplots(figsize=(9, 9))
    axes.imshow(image_rgb)
    colors = plt.colormaps["tab10"]
    for index, (region, polygon) in enumerate(result.polygons.items()):
        points = np.asarray(polygon)
        axes.add_patch(PolygonPatch(points, fill=False, linewidth=2,
                                    color=colors(index), label=region))
        axes.text(*points.mean(axis=0), region, color="white", fontsize=8,
                  ha="center", va="center",
                  bbox={"facecolor": colors(index), "alpha": 0.7, "pad": 2})
    for box, region in zip(boxes, result.regions):
        x0, y0, x1, y1 = box
        axes.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 fill=False, linewidth=2, color="lime"))
        axes.text(x0, y0, region, color="black", fontsize=8,
                  bbox={"facecolor": "lime", "alpha": 0.8, "pad": 1})
    axes.axis("off")
    axes.legend(loc="lower center", ncols=3, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=160)
    plt.close(figure)
    return result


def load_boxes(path: FilePath | None) -> list[Box]:
    """Read plain boxes or the existing pipeline's predictions.json shape."""
    if path is None:
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict) and "detections" in data:
        data = [detection["box"] for detection in data["detections"]]
    elif isinstance(data, dict):
        data = data.get("boxes", [])
    return [tuple(box) for box in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render face regions for visual approval")
    parser.add_argument("image", type=FilePath)
    parser.add_argument("--boxes", type=FilePath,
                        help="JSON list of detector xyxy boxes (or an object with a boxes key)")
    parser.add_argument("--face-box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="manual face box for profile images the fallback detector misses")
    parser.add_argument("--model", type=FilePath,
                        default=FilePath(load_config()["paths"]["face_landmarker"]))
    parser.add_argument("--output", type=FilePath,
                        default=FilePath("runs/face_regions_overlay.png"))
    args = parser.parse_args()

    from ..classification.run_acne04_pipeline import load_rgb

    result = render_debug_overlay(
        load_rgb(args.image), load_boxes(args.boxes), args.output,
        face_box=tuple(args.face_box) if args.face_box else None,
        model_path=args.model,
    )
    print(json.dumps({"output": str(args.output), **result.metadata}, indent=2))


if __name__ == "__main__":
    main()
