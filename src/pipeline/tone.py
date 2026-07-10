"""Skin-tone triage from non-lesional pixels (D-021); never a diagnosis."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path as FilePath

import numpy as np
from matplotlib.path import Path

from ..config import load_config


SKIN_SAMPLE_REGIONS = {"forehead", "left_cheek", "right_cheek"}


@lru_cache(maxsize=1)
def _sephora_tone_buckets() -> dict[str, frozenset[str]]:
    """Config carries the vocabulary (D-021); cached — called per review row."""
    return {bucket: frozenset(values) for bucket, values
            in load_config()["tone"]["sephora_tone_buckets"].items()}


@dataclass(frozen=True)
class ToneEstimate:
    bucket: str
    ita: float | None
    median_l: float | None
    low_light: bool
    sample_count: int
    source: str = "photo_estimate"


def sephora_tone_bucket(value: str | None) -> str:
    """Map the review dataset vocabulary; unrecognized values remain unknown."""
    normalized = "".join(character for character in (value or "").casefold()
                         if character.isalnum())
    return next((bucket for bucket, values in _sephora_tone_buckets().items()
                 if normalized in values), "unknown")


def tone_bucket(ita: float | None, *, light_min: float, medium_min: float) -> str:
    if ita is None or not np.isfinite(ita):
        return "unknown"
    if ita >= light_min:
        return "light"
    if ita >= medium_min:
        return "medium"
    return "deep"


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 or [0, 1] sRGB values to CIELAB (D65) in plain NumPy."""
    array = np.asarray(rgb)
    values = array.astype(float)
    if np.issubdtype(array.dtype, np.integer) or (values.size and values.max() > 1):
        values /= 255
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    transformed = np.where(
        xyz > delta ** 3,
        np.cbrt(xyz),
        xyz / (3 * delta ** 2) + 4 / 29,
    )
    x, y, z = np.moveaxis(transformed, -1, 0)
    return np.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), axis=-1)


def _sample_mask(image_shape: Sequence[int],
                 polygons: Mapping[str, Sequence[tuple[float, float]]],
                 profile_cheek_ratio: float) -> np.ndarray:
    height, width = image_shape[:2]
    y, x = np.mgrid[:height, :width]
    pixel_centers = np.column_stack((x.ravel() + 0.5, y.ravel() + 0.5))
    mask = np.zeros(height * width, dtype=bool)
    sample_regions = set(SKIN_SAMPLE_REGIONS)
    if {"left_cheek", "right_cheek"} <= polygons.keys():
        cheek_areas = {}
        for region in ("left_cheek", "right_cheek"):
            polygon_points = np.asarray(polygons[region])
            cheek_areas[region] = 0.5 * abs(
                np.dot(polygon_points[:, 0], np.roll(polygon_points[:, 1], 1))
                - np.dot(polygon_points[:, 1], np.roll(polygon_points[:, 0], 1))
            )
        smaller = min(cheek_areas, key=cheek_areas.get)
        larger = max(cheek_areas.values())
        if larger and cheek_areas[smaller] / larger < profile_cheek_ratio:
            sample_regions.remove(smaller)

    for region, polygon in polygons.items():
        if region in sample_regions:
            mask |= Path(polygon).contains_points(pixel_centers)
    return mask.reshape(height, width)


def _exclude_lesions(mask: np.ndarray, boxes: Sequence[Sequence[float]], pad: float) -> None:
    height, width = mask.shape
    for x0, y0, x1, y1 in boxes:
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        half_width, half_height = (x1 - x0) * pad / 2, (y1 - y0) * pad / 2
        left, right = max(0, int(np.floor(center_x - half_width))), min(width, int(np.ceil(center_x + half_width)))
        top, bottom = max(0, int(np.floor(center_y - half_height))), min(height, int(np.ceil(center_y + half_height)))
        mask[top:bottom, left:right] = False


def _valid_sample_mask(image_rgb: np.ndarray,
                       polygons: Mapping[str, Sequence[tuple[float, float]]],
                       lesion_boxes: Sequence[Sequence[float]],
                       crop_pad: float,
                       tone_config: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact image mask and Lab pixels used by the ITA estimate."""
    mask = _sample_mask(image_rgb.shape, polygons, tone_config["profile_cheek_area_ratio"])
    _exclude_lesions(mask, lesion_boxes, crop_pad)
    pixels = np.asarray(image_rgb)[mask]
    if not len(pixels):
        return mask, np.empty((0, 3))

    channel_range = np.ptp(pixels, axis=1)
    specular = ((pixels.max(axis=1) >= tone_config["specular_rgb_min"])
                & (channel_range <= tone_config["specular_rgb_range"]))
    lab = srgb_to_lab(pixels)
    gate = tone_config["skin_lab_gate"]
    skin_colored = (lab[:, 1] > gate["a_min"]) & (lab[:, 2] > gate["b_min"])
    valid = ~specular & (lab[:, 0] > gate["l_min"]) & skin_colored
    candidate_locations = np.flatnonzero(mask)
    mask.flat[candidate_locations[~valid]] = False
    return mask, lab[valid]


def estimate_tone(image_rgb: np.ndarray,
                  polygons: Mapping[str, Sequence[tuple[float, float]]] | None = None,
                  lesion_boxes: Sequence[Sequence[float]] = (), *, min_pixels: int | None = None,
                  light_min: float | None = None, medium_min: float | None = None,
                  low_light_l: float | None = None, crop_pad: float | None = None) -> ToneEstimate:
    """Estimate a coarse photo tone bucket from non-lesional facial pixels."""
    config = load_config()
    tone_config = config["tone"]
    min_pixels = tone_config["min_sample_pixels"] if min_pixels is None else min_pixels
    light_min = tone_config["ita_light_min"] if light_min is None else light_min
    medium_min = tone_config["ita_medium_min"] if medium_min is None else medium_min
    low_light_l = tone_config["low_light_l_threshold"] if low_light_l is None else low_light_l
    crop_pad = config["classification"]["crop_pad"] if crop_pad is None else crop_pad
    if polygons is None:
        from .regions import grid_polygons
        polygons = grid_polygons(image_rgb.shape)

    _, lab = _valid_sample_mask(image_rgb, polygons, lesion_boxes, crop_pad, tone_config)
    sample_count = len(lab)
    median_l = float(np.median(lab[:, 0])) if sample_count else None
    low_light = median_l is not None and median_l < low_light_l
    if sample_count < min_pixels:
        return ToneEstimate("unknown", None, median_l, low_light, sample_count)

    median_b = float(np.median(lab[:, 2]))
    ita = float(np.degrees(np.arctan(np.divide(median_l - 50, median_b))))
    return ToneEstimate(
        tone_bucket(ita, light_min=light_min, medium_min=medium_min),
        ita,
        median_l,
        low_light,
        sample_count,
    )


def render_debug_overlay(image_rgb: np.ndarray, lesion_boxes: Sequence[Sequence[float]],
                         output: FilePath, *, face_box=None,
                         model_path: FilePath | None = None) -> tuple[ToneEstimate, dict[str, object]]:
    """Render the exact non-lesional sampling pixels used for ITA."""
    from matplotlib import pyplot as plt
    from .regions import locate_regions

    region_result = locate_regions(
        image_rgb, lesion_boxes, face_box=face_box, model_path=model_path
    )
    config = load_config()
    candidate_mask, _ = _valid_sample_mask(
        image_rgb,
        region_result.polygons,
        lesion_boxes,
        config["classification"]["crop_pad"],
        config["tone"],
    )
    estimate = estimate_tone(image_rgb, region_result.polygons, lesion_boxes)

    overlay = np.zeros((*candidate_mask.shape, 4), dtype=float)
    overlay[candidate_mask] = (0.0, 1.0, 0.3, 0.35)
    figure, axes = plt.subplots(figsize=(9, 9))
    axes.imshow(image_rgb)
    axes.imshow(overlay)
    axes.set_title(
        f"Tone triage: {estimate.bucket} | ITA={estimate.ita} | "
        f"L*={estimate.median_l} | N={estimate.sample_count}"
    )
    axes.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=160)
    plt.close(figure)
    return estimate, region_result.metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ITA skin-sampling pixels for visual approval")
    parser.add_argument("image", type=FilePath)
    parser.add_argument("--boxes", type=FilePath,
                        help="JSON list of detector xyxy boxes (or an object with a boxes key)")
    parser.add_argument("--face-box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="manual face box for profile images the fallback detector misses")
    parser.add_argument("--model", type=FilePath,
                        default=FilePath(load_config()["paths"]["face_landmarker"]))
    parser.add_argument("--output", type=FilePath,
                        default=FilePath("runs/tone_samples_overlay.png"))
    args = parser.parse_args()

    from ..classification.run_acne04_pipeline import load_rgb

    from .regions import load_boxes

    estimate, metadata = render_debug_overlay(
        load_rgb(args.image), load_boxes(args.boxes), args.output,
        face_box=tuple(args.face_box) if args.face_box else None,
        model_path=args.model,
    )
    print(json.dumps({"output": str(args.output), "tone": asdict(estimate),
                      "regions": metadata}, indent=2))


if __name__ == "__main__":
    main()
