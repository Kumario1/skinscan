"""Tests for concern-stats aggregation (plan 015, D-023). Pure-Python."""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendation.concern_stats import (
    build_concern_stats, labels_frame, main,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["product_id", "skin_type",
                                       "concern", "outcome"])


def test_smoothing_math():
    rows = ([("PA", "oily", "acne_general", "helped")] * 8
            + [("PA", "oily", "acne_general", "worsened")] * 2
            + [("PB", "dry", "acne_general", "helped")] * 2
            + [("PB", "dry", "acne_general", "worsened")] * 8)
    stats = build_concern_stats(_df(rows), m=20, sub_cell_min_n=5)
    # prior = 10 helped / 20 outcomes = 0.5; PA = (8 + 20*0.5) / (10 + 20) = 0.6
    cell = stats["cells"]["PA"]["acne_general"]["__all__"]
    assert cell["n"] == 10 and cell["helped"] == 8 and cell["worsened"] == 2
    assert abs(cell["smoothed"] - 0.6) < 1e-9
    assert abs(stats["priors"]["acne_general"] - 0.5) < 1e-9
    # PB = (2 + 10) / 30 = 0.4 -> ordering reflects evidence
    assert stats["cells"]["PB"]["acne_general"]["__all__"]["smoothed"] < 0.5


def test_unclear_counted_but_excluded_from_n():
    rows = [("PA", "oily", "dryness", "helped"),
            ("PA", "oily", "dryness", "unclear")]
    stats = build_concern_stats(_df(rows), m=20, sub_cell_min_n=5)
    cell = stats["cells"]["PA"]["dryness"]["__all__"]
    assert cell["n"] == 1 and cell["n_unclear"] == 1


def test_skin_type_subcells_respect_min_n():
    rows = ([("PA", "oily", "acne_general", "helped")] * 5
            + [("PA", "dry", "acne_general", "helped")] * 2)
    stats = build_concern_stats(_df(rows), m=20, sub_cell_min_n=5)
    concern_cell = stats["cells"]["PA"]["acne_general"]
    assert "oily" in concern_cell and "dry" not in concern_cell
    assert concern_cell["__all__"]["n"] == 7


def test_labels_frame_ignores_non_ok_records():
    recs = [
        {"uid": "u1", "product_id": "PA", "skin_type": "oily", "status": "ok",
         "labels": [{"concern": "acne_general", "outcome": "helped",
                     "reviewer_has_condition": True}]},
        {"uid": "u2", "product_id": "PB", "skin_type": "dry",
         "status": "parse_error", "labels": []},
        {"uid": "u3", "product_id": "PC", "skin_type": "dry", "status": "ok",
         "labels": []},                      # ok but nothing mentioned
    ]
    df = labels_frame(recs)
    assert list(df["product_id"]) == ["PA"]


def test_cli_end_to_end():
    recs = [{"uid": f"u{i}", "product_id": "PA", "skin_type": "oily",
             "status": "ok",
             "labels": [{"concern": "acne_general", "outcome": "helped",
                         "reviewer_has_condition": True}]}
            for i in range(3)]
    with tempfile.TemporaryDirectory() as td:
        labels = Path(td) / "labels.jsonl"
        out = Path(td) / "concern_stats.json"
        labels.write_text("".join(json.dumps(r) + "\n" for r in recs))
        main(["--labels", str(labels), "--out", str(out)])
        stats = json.loads(out.read_text())
        assert stats["cells"]["PA"]["acne_general"]["__all__"]["n"] == 3
        assert stats["smoothing_m"] == 20


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
