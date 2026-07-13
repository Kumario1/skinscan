"""Validated native-tile HTTP client for the production SA-RPN service."""

import base64
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests

from src.recommendation.schema import Concern, ConcernEvidence, ConcernReport


SARPN_LABEL_TO_CONCERN = {
    "closed_comedo": "acne_comedonal", "open_comedo": "acne_comedonal",
    "papule": "acne_inflammatory", "pustule": "acne_inflammatory",
    "nodule": "acne_cystic", "atrophic_scar": "acne_scarring",
    "hypertrophic_scar": "acne_scarring", "melasma": "hyperpigmentation",
}
SARPN_NON_ACTIONABLE_LABELS = {"nevus", "other"}
LABEL_COLORS = {
    "closed_comedo": "#2E86AB", "open_comedo": "#3C91E6",
    "papule": "#E45756", "pustule": "#F3A712", "nodule": "#8F2D56",
    "atrophic_scar": "#6A4C93", "hypertrophic_scar": "#A23B72",
    "melasma": "#7A5195", "nevus": "#4D908E", "other": "#577590",
}
REGION_COLORS = {
    "forehead": "#277DA1", "nose": "#F9844A", "right_cheek": "#43AA8B",
    "left_cheek": "#90BE6D", "perioral": "#F94144", "chin_jaw": "#9B5DE5",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class SarpnSettings:
    endpoint_url: str
    tile_size: int
    tile_overlap: int
    connect_timeout_seconds: float
    read_timeout_seconds: float
    request_batch_size: int
    min_score: float
    class_min_scores: Mapping[str, float]
    dedupe_threshold: float
    severity: Mapping[str, Any]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SarpnSettings":
        values = config["sa_rpn"]
        settings = cls(
            endpoint_url=values["endpoint_url"],
            tile_size=values["tile_size"],
            tile_overlap=values["tile_overlap"],
            connect_timeout_seconds=values["connect_timeout_seconds"],
            read_timeout_seconds=values["read_timeout_seconds"],
            request_batch_size=values["request_batch_size"],
            min_score=values["min_score"],
            class_min_scores=_freeze({
                normalize_sarpn_label(label): score
                for label, score in (values.get("class_min_scores") or {}).items()
            }),
            dedupe_threshold=values["dedupe_threshold"],
            severity=_freeze(values["severity"]),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.tile_overlap < 0 or self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must satisfy 0 <= tile_overlap < tile_size")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if self.read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive")
        if self.request_batch_size <= 0:
            raise ValueError("request_batch_size must be positive")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if any(not 0 <= score <= 1 for score in self.class_min_scores.values()):
            raise ValueError("class_min_scores values must be between 0 and 1")
        if not 0 <= self.dedupe_threshold <= 1:
            raise ValueError("dedupe_threshold must be between 0 and 1")


@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class LesionObservation:
    label: str
    label_name: str
    score: float
    box: tuple[float, float, float, float]
    tile_index: int
    tile_box: tuple[int, int, int, int]
    region: str | None = None
    mapped_concern: str | None = None
    observation_status: str = "actionable"

    @property
    def normalized_label(self) -> str:
        return self.label

    @property
    def original_label(self) -> str:
        return self.label_name

    @property
    def confidence(self) -> float:
        return self.score


@dataclass(frozen=True)
class SafetyObservation:
    code: str
    message: str
    labels: dict[str, int]
    count: int
    max_confidence: float
    professional_review: bool


class SarpnAnalysisError(RuntimeError):
    """Base failure for the SA-RPN identification stage."""


class SarpnTransportError(SarpnAnalysisError):
    """The SA-RPN endpoint could not successfully serve a tile."""


class SarpnResponseError(SarpnAnalysisError):
    """The SA-RPN endpoint returned a response outside its strict contract."""


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image after EXIF orientation correction and convert it to RGB."""
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Return evenly spaced origins whose final tile reaches the far edge."""
    if length <= tile:
        return [0]
    count = -(-(length - tile) // stride) + 1
    return [round(index * (length - tile) / (count - 1)) for index in range(count)]


def make_tiles(
    image_shape: Sequence[int], *, tile_size: int, overlap: int
) -> list[Tile]:
    height, width = image_shape[:2]
    stride = tile_size - overlap
    tiles = []
    for y in tile_origins(height, tile_size, stride):
        for x in tile_origins(width, tile_size, stride):
            tiles.append(
                Tile(
                    index=len(tiles),
                    x=x,
                    y=y,
                    width=min(tile_size, width - x),
                    height=min(tile_size, height - y),
                )
            )
    return tiles


def _response_error(tile: Tile, endpoint: str, field: str, detail: str) -> SarpnResponseError:
    return SarpnResponseError(
        f"tile {tile.index} response field {field} from {sanitize_endpoint(endpoint)}: {detail}"
    )


def _jpeg_base64(rgb: np.ndarray) -> str:
    output = BytesIO()
    Image.fromarray(rgb).save(output, format="JPEG", quality=92)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _validated_detections(
    payload: Any, tile: Tile, endpoint: str, min_score: float, image_size: tuple[int, int],
    class_min_scores: Mapping[str, float] = MappingProxyType({}),
) -> list[LesionObservation]:
    if not isinstance(payload, Mapping) or "detections" not in payload:
        raise _response_error(tile, endpoint, "detections", "is missing")
    detections = payload["detections"]
    if not isinstance(detections, list):
        raise _response_error(tile, endpoint, "detections", "must be a list")
    count = payload.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise _response_error(tile, endpoint, "count", "must be a non-negative integer")
    if count != len(detections):
        raise _response_error(tile, endpoint, "count", "must match detections length")

    image_width, image_height = image_size
    observations = []
    for detection in detections:
        if not isinstance(detection, Mapping):
            raise _response_error(tile, endpoint, "detections", "entry must be an object")
        label = detection.get("label")
        if not isinstance(label, str) or not label.strip():
            raise _response_error(tile, endpoint, "label", "must be a non-blank string")
        # Normalize once (D-... plan): retain the exact raw server string as
        # the observation's original_label end-to-end. Title-casing is a
        # display concern only — do it in drawing/rendering code, not here.
        label_name = label.strip()
        score = detection.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise _response_error(tile, endpoint, "score", "must be a finite number between 0 and 1")
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise _response_error(tile, endpoint, "bbox", "must contain four coordinates")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox):
            raise _response_error(tile, endpoint, "bbox", "coordinates must be finite numbers")
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if x2 <= x1 or y2 <= y1:
            raise _response_error(tile, endpoint, "bbox", "must have positive area")
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(tile.width), x2), min(float(tile.height), y2)
        if x2 <= x1 or y2 <= y1:
            raise _response_error(tile, endpoint, "bbox", "is empty after clipping to tile")
        if score < max(min_score,
                       class_min_scores.get(normalize_sarpn_label(label_name), 0.0)):
            continue
        full_box = (
            max(0.0, min(float(image_width), x1 + tile.x)),
            max(0.0, min(float(image_height), y1 + tile.y)),
            max(0.0, min(float(image_width), x2 + tile.x)),
            max(0.0, min(float(image_height), y2 + tile.y)),
        )
        if full_box[2] <= full_box[0] or full_box[3] <= full_box[1]:
            raise _response_error(tile, endpoint, "bbox", "is empty after restoration")
        observations.append(
            LesionObservation(
                label.strip(), label_name, float(score), full_box, tile.index,
                (tile.x, tile.y, tile.x + tile.width, tile.y + tile.height),
            )
        )
    return observations


def _scrub_error_text(text: str, settings: "SarpnSettings") -> str:
    """Best-effort redaction of credentials/queries embedded in exception
    text. requests/urllib3 surface the request URL in several shapes that a
    single literal replace against the configured endpoint_url can miss:

      - relative-form URLs in ConnectionError messages, e.g. "Max retries
        exceeded with url: /predict?token=SECRET" (no scheme/host at all);
      - urllib3-normalized URLs (e.g. lowercased host) in HTTPError messages,
        which no longer match the configured endpoint_url verbatim.

    So this applies every mitigation unconditionally rather than relying on
    one exact match.
    """
    safe_endpoint = sanitize_endpoint(settings.endpoint_url)
    text = text.replace(settings.endpoint_url, safe_endpoint)
    # Strip basic-auth userinfo after any scheme, regardless of host casing.
    text = re.sub(r"(?<=://)[^/@\s]+@", "", text)
    parsed = urlsplit(settings.endpoint_url)
    if parsed.query:
        path = parsed.path or "/"
        if path:
            text = re.sub(re.escape(path) + r"\?[^\s)\"']*", path, text)
        text = text.replace(f"?{parsed.query}", "")
    return text


def _infer_tile(
    rgb: np.ndarray,
    tile: Tile,
    settings: SarpnSettings,
    session_factory: Callable[[], requests.Session],
) -> list[LesionObservation]:
    encoded = _jpeg_base64(rgb[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width])
    session = session_factory()
    try:
        try:
            response = session.post(
                settings.endpoint_url,
                json={"image": encoded},
                timeout=(settings.connect_timeout_seconds, settings.read_timeout_seconds),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            # requests embeds the literal request URL (including any basic-auth
            # userinfo) in some exception messages, e.g. HTTPError from
            # raise_for_status(); scrub it before it reaches logs or analysis.json.
            safe_endpoint = sanitize_endpoint(settings.endpoint_url)
            detail = _scrub_error_text(str(exc), settings)
            raise SarpnTransportError(
                f"tile {tile.index} request to {safe_endpoint} failed: {detail}"
            ) from exc
        try:
            payload = response.json()
        except (requests.JSONDecodeError, ValueError) as exc:
            raise _response_error(tile, settings.endpoint_url, "response", "must be valid JSON") from exc
    finally:
        session.close()
    return _validated_detections(
        payload, tile, settings.endpoint_url, settings.min_score, (rgb.shape[1], rgb.shape[0]),
        settings.class_min_scores,
    )


def infer_native_tiles(
    rgb: np.ndarray,
    settings: SarpnSettings,
    *,
    session_factory: Callable[[], requests.Session] = requests.Session,
    dedupe: bool = True,
) -> list[LesionObservation]:
    """Infer every native-resolution tile; any tile failure aborts the analysis.

    dedupe=True (default) preserves the production contract: cross-tile
    duplicate suppression happens exactly once, here. Callers that need the
    genuine raw (pre-dedupe) detections — e.g. the compare_sarpn A/B harness,
    which reports raw_detections vs. detections_after_dedupe — pass
    dedupe=False and own the single dedupe_observations() call downstream.
    """
    tiles = make_tiles(rgb.shape, tile_size=settings.tile_size, overlap=settings.tile_overlap)
    results: dict[int, list[LesionObservation]] = {}
    with ThreadPoolExecutor(max_workers=settings.request_batch_size) as executor:
        futures = {
            executor.submit(_infer_tile, rgb, tile, settings, session_factory): tile
            for tile in tiles
        }
        try:
            for future in as_completed(futures):
                tile = futures[future]
                results[tile.index] = future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise
    ordered = [observation for index in sorted(results) for observation in results[index]]
    if not dedupe:
        return ordered
    return dedupe_observations(
        ordered, threshold=settings.dedupe_threshold, preserve_tile_order=True
    )


def normalize_sarpn_label(label: str) -> str:
    return "_".join(label.casefold().strip().replace("-", " ").replace("_", " ").split())


def _severity(labels: Counter, scores: list[float], region_count: int, config: Mapping[str, Any], concern: str) -> int:
    if labels["nodule"]:
        return config["nodule_severity"]
    severity = bisect_right(tuple(config["count_thresholds"][concern]), sum(labels.values()))
    # Check the broad floor first so the floor is monotonic in region_count
    # regardless of how broad_region_count is configured (Finding 9): a
    # region_count that already qualifies as "broad" must never fall through
    # to a lower floor than a smaller region_count would get.
    if region_count >= config["broad_region_count"]:
        severity = max(severity, 3)
    elif region_count >= 2:
        severity = max(severity, 2)
    if labels["hypertrophic_scar"]:
        severity = max(severity, config["hypertrophic_scar_min_severity"])
    if scores and max(scores) < config["confidence_cutoff"]:
        severity = min(severity, 1)
    return severity


def build_sarpn_concern_report(image_id: str, observations: Sequence[LesionObservation],
                               regions: Sequence[str], severity_config: Mapping[str, Any], *,
                               low_light_flag: bool = False):
    if len(observations) != len(regions):
        raise ValueError("observations and regions must have the same length")
    grouped = defaultdict(list)
    safety_groups = defaultdict(list)
    updated = []
    for observation, region in zip(observations, regions):
        label = normalize_sarpn_label(observation.label)
        concern = SARPN_LABEL_TO_CONCERN.get(label)
        if concern:
            status = "actionable"
            grouped[concern].append((label, observation.score, region))
        elif label in SARPN_NON_ACTIONABLE_LABELS:
            status = "non_actionable"
            safety_groups[label].append(observation.score)
        else:
            status = "unsupported"
            safety_groups["unsupported"].append((label, observation.score))
        updated.append(LesionObservation(label, observation.label_name, observation.score,
                                          observation.box, observation.tile_index, observation.tile_box,
                                          region, concern, status))
    concerns = []
    for concern_name, members in sorted(grouped.items()):
        labels = Counter(label for label, _, _ in members)
        scores = [score for _, score, _ in members]
        concern_regions = sorted({region for _, _, region in members})
        evidence = ConcernEvidence(dict(labels), max(scores), len(concern_regions))
        concerns.append(Concern(concern_name, concern_regions[0],
                                _severity(labels, scores, len(concern_regions), severity_config, concern_name),
                                sum(scores) / len(scores), len(members), concern_regions, evidence))
    safety = []
    for label in ("nevus", "other"):
        scores = safety_groups.get(label, [])
        if scores:
            policy = severity_config["professional_review"][label]
            review = len(scores) >= policy["min_count"] or max(scores) >= policy["min_confidence"]
            safety.append(SafetyObservation(f"{label}_observation", f"Non-actionable {label} observation",
                                             {label: len(scores)}, len(scores), max(scores), review))
    unsupported = safety_groups.get("unsupported", [])
    if unsupported:
        counts = Counter(label for label, _ in unsupported)
        safety.append(SafetyObservation("unsupported_label", "Unsupported SA-RPN label",
                                         dict(counts), len(unsupported), max(score for _, score in unsupported), False))
    return ConcernReport(image_id, concerns, not concerns, low_light_flag), updated, safety


def concern_to_dict(concern: Concern) -> dict[str, object]:
    return {"concern": concern.concern, "regions": list(concern.regions), "severity": concern.severity,
            "confidence": concern.confidence, "lesion_count": concern.lesion_count,
            "evidence": {"labels": dict(concern.evidence.labels),
                         "max_confidence": concern.evidence.max_confidence,
                         "affected_region_count": concern.evidence.affected_region_count}}


def sanitize_endpoint(url: str) -> str:
    """Return a publish-safe endpoint without credentials, query, or fragment."""
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    try:
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
    except ValueError:
        pass
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def observation_to_dict(observation: LesionObservation) -> dict[str, object]:
    return {
        "normalized_label": observation.normalized_label,
        "original_label": observation.original_label,
        "confidence": observation.confidence,
        "box": list(observation.box),
        "region": observation.region,
        "mapped_concern": observation.mapped_concern,
        "observation_status": observation.observation_status,
        "source_tile": {
            "index": observation.tile_index,
            "box": list(observation.tile_box),
        },
    }


def _label_color(label: str) -> str:
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    value = sum((index + 1) * ord(char) for index, char in enumerate(label))
    return f"#{64 + value % 160:02x}{64 + (value // 5) % 160:02x}{64 + (value // 11) % 160:02x}"


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_detection_overlay(
    image_rgb: np.ndarray, observations: Sequence[LesionObservation], output: Path,
) -> None:
    image = Image.fromarray(image_rgb).copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width = max(2, round(min(image.size) / 250))
    for observation in observations:
        color = _label_color(observation.normalized_label)
        draw.rectangle(observation.box, outline=color, width=width)
        label = f"{observation.normalized_label} {observation.confidence:.2f}"
        text_width, text_height = _text_size(draw, label, font)
        x1, y1 = observation.box[:2]
        top = max(0, int(y1) - text_height - 5)
        draw.rectangle((int(x1), top, int(x1) + text_width + 6, top + text_height + 5), fill=color)
        draw.text((int(x1) + 3, top + 2), label, fill="white", font=font)
    labels = sorted({item.normalized_label for item in observations})
    if labels:
        x = 8
        y = 8
        for label in labels:
            text_width, text_height = _text_size(draw, label, font)
            draw.rectangle((x, y, x + 12, y + 12), fill=_label_color(label))
            draw.text((x + 17, y), label, fill="white", stroke_width=2,
                      stroke_fill="black", font=font)
            y += max(16, text_height + 4)
    image.save(output, format="JPEG", quality=92)


def draw_region_overlay(
    image_rgb: np.ndarray, observations: Sequence[LesionObservation], region_result: Any,
    output: Path,
) -> None:
    image = Image.fromarray(image_rgb).copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 300))
    for region, polygon in region_result.polygons.items():
        points = [(round(x), round(y)) for x, y in polygon]
        if len(points) >= 2:
            color = REGION_COLORS.get(region, "#FFFFFF")
            draw.line(points + [points[0]], fill=color, width=line_width)
            draw.text(points[0], region, fill=color, stroke_width=2, stroke_fill="black", font=font)
    for observation in observations:
        x1, y1, x2, y2 = observation.box
        center = (round((x1 + x2) / 2), round((y1 + y2) / 2))
        color = REGION_COLORS.get(observation.region or "", "#FFFFFF")
        radius = max(3, line_width + 1)
        draw.ellipse((center[0] - radius, center[1] - radius,
                      center[0] + radius, center[1] + radius), fill=color, outline="black")
        draw.text((center[0] + radius + 2, center[1] - radius), observation.region or "unknown",
                  fill=color, stroke_width=2, stroke_fill="black", font=font)
    title = f"region mapping: {region_result.metadata.get('method', 'unknown')}"
    text_width, text_height = _text_size(draw, title, font)
    draw.rectangle((0, 0, text_width + 12, text_height + 8), fill="black")
    draw.text((6, 4), title, fill="white", font=font)
    image.save(output, format="JPEG", quality=92)


def draw_lesion_sheet(
    image_rgb: np.ndarray, observations: Sequence[LesionObservation], output: Path,
) -> None:
    font = ImageFont.load_default()
    if not observations:
        sheet = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(sheet).text((24, 64), "No retained SA-RPN detections.", fill="black", font=font)
        sheet.save(output, format="JPEG", quality=92)
        return

    tile_width, tile_height, caption_height = 240, 190, 58
    columns = min(4, len(observations))
    rows = math.ceil(len(observations) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + caption_height)), "white")
    source = Image.fromarray(image_rgb)
    for index, observation in enumerate(observations):
        x1, y1, x2, y2 = observation.box
        box_width, box_height = x2 - x1, y2 - y1
        pad_x, pad_y = box_width * 0.25, box_height * 0.25
        crop_box = (
            max(0, math.floor(x1 - pad_x)), max(0, math.floor(y1 - pad_y)),
            min(source.width, math.ceil(x2 + pad_x)), min(source.height, math.ceil(y2 + pad_y)),
        )
        crop = ImageOps.contain(source.crop(crop_box), (tile_width, tile_height))
        cell_x = (index % columns) * tile_width
        cell_y = (index // columns) * (tile_height + caption_height)
        sheet.paste(crop, (cell_x + (tile_width - crop.width) // 2, cell_y))
        status = observation.mapped_concern or (
            "safety" if observation.observation_status == "non_actionable"
            else observation.observation_status
        )
        caption = (
            f"{observation.normalized_label} {observation.confidence:.2f}\n"
            f"{observation.region or 'unknown'} | {status}"
        )
        ImageDraw.Draw(sheet).text((cell_x + 6, cell_y + tile_height + 5), caption,
                                   fill="black", font=font)
    sheet.save(output, format="JPEG", quality=92)


def dedupe_observations(
    observations: Sequence[LesionObservation],
    *,
    threshold: float,
    preserve_tile_order: bool = False,
) -> list[LesionObservation]:
    """Greedily suppress boxes by overlap over smaller area, regardless of class."""
    kept: list[LesionObservation] = []
    for observation in sorted(observations, key=lambda item: item.score, reverse=True):
        x1, y1, x2, y2 = observation.box
        area = (x2 - x1) * (y2 - y1)
        for accepted in kept:
            ax1, ay1, ax2, ay2 = accepted.box
            intersection_width = min(x2, ax2) - max(x1, ax1)
            intersection_height = min(y2, ay2) - max(y1, ay1)
            if intersection_width <= 0 or intersection_height <= 0:
                continue
            accepted_area = (ax2 - ax1) * (ay2 - ay1)
            if (intersection_width * intersection_height) / min(area, accepted_area) > threshold:
                break
        else:
            kept.append(observation)
    if preserve_tile_order:
        kept.sort(key=lambda item: item.tile_index)
    return kept
