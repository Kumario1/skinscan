"""Evaluation-only AcneSCU VOC annotation reader for oracle counterfactuals."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from .sarpn import LesionObservation, normalize_sarpn_label


def load_voc_oracle_observations(path: str | Path) -> list[LesionObservation]:
    """Return annotation-derived observations; values are labels, not probabilities."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"oracle annotations invalid: {path}: {exc}") from exc
    size = root.find("size")
    if size is None:
        raise ValueError(f"oracle annotations missing size: {path}")
    try:
        width = int(size.findtext("width", ""))
        height = int(size.findtext("height", ""))
    except ValueError as exc:
        raise ValueError(f"oracle annotations invalid size: {path}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"oracle annotations invalid size: {path}")

    observations: list[LesionObservation] = []
    for index, node in enumerate(root.findall("object")):
        raw_label = (node.findtext("name") or "").strip()
        box = node.find("bndbox")
        if not raw_label or box is None:
            raise ValueError(f"oracle annotations object[{index}] missing label/bndbox: {path}")
        try:
            x1 = float(box.findtext("xmin", ""))
            y1 = float(box.findtext("ymin", ""))
            x2 = float(box.findtext("xmax", ""))
            y2 = float(box.findtext("ymax", ""))
        except ValueError as exc:
            raise ValueError(f"oracle annotations object[{index}] invalid bbox: {path}") from exc
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"oracle annotations object[{index}] empty bbox: {path}")
        observations.append(LesionObservation(
            normalize_sarpn_label(raw_label), raw_label, 1.0, (x1, y1, x2, y2),
            -1, (0, 0, width, height),
        ))
    return observations
