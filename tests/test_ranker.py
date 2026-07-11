"""Tests for the learned re-ranker (src/recommendation/ranker.py).

Pure Python (sklearn/pandas/joblib allowed in the default suite, no TF/YOLO/
mediapipe). Trains on the committed fixture `tests/fixtures/reviews_sample.csv`
against an in-code two-product catalog written to a tmp catalog.json, so the
whole aggregate -> train -> eval -> load -> score path is exercised end-to-end
without touching the real data tree.

The fixture is engineered so a model that uses the reviewer's skin_type x product
interaction beats a product-only popularity/rating baseline on a held-out
reviewer split. Standalone via __main__ (pytest not required) but named test_*
so `pytest tests/` also works.
"""
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_config
from src.recommendation import ranker as R
from src.recommendation.engine import Recommendation, recommend
from src.recommendation.schema import Concern, ConcernReport, Product, UserProfile

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "reviews_sample.csv"

# both products are `treatment` so they compete in ONE routine step; both carry a
# real price so f_price is never an all-NaN training column (crashes HGB 1.9).
TRAIN_CATALOG = [
    Product("PA", "SA Treatment", "BrandX", "treatment",
            actives=["salicylic_acid"], price_usd=24.0),
    Product("PB", "Ceramide Treatment", "BrandY", "treatment",
            actives=["ceramides"], price_usd=18.0),
]


def _train(tmp):
    """Train against the fixture into a tmp dir; return (eval_result, paths)."""
    tmp = Path(tmp)
    catalog_path = tmp / "catalog.json"
    catalog_path.write_text(json.dumps([asdict(p) for p in TRAIN_CATALOG]))
    model_path = tmp / "ranker.joblib"
    stats_path = tmp / "review_stats.json"
    eval_path = tmp / "eval.json"
    result = R.train_pipeline(
        str(FIXTURE), str(catalog_path), str(model_path), str(stats_path),
        str(eval_path), load_config(), verbose=False,
    )
    return result, model_path, stats_path, eval_path


def _config_at(model_path, stats_path, min_cell_size=5):
    return {"ranker": {"model_path": str(model_path),
                       "review_stats_path": str(stats_path),
                       "min_cell_size": min_cell_size}}


def test_end_to_end_fixture_gate_passes():
    with tempfile.TemporaryDirectory() as d:
        result, model_path, stats_path, eval_path = _train(d)
        assert result["gate_passed"] is True, result["pooled"]
        # gate passed -> all three artifacts on disk
        assert model_path.exists()
        assert stats_path.exists()
        assert eval_path.exists()
        # eval.json is valid JSON (NaN cells sanitized to null) and mirrors the gate
        loaded = json.loads(eval_path.read_text())
        assert loaded["gate_passed"] is True
        assert loaded["by_tone"]["unknown"]["model"]["roc_auc"] is None  # empty bucket -> null


def test_model_beats_baselines():
    with tempfile.TemporaryDirectory() as d:
        result, *_ = _train(d)
        pooled = result["pooled"]
        assert pooled["model"]["roc_auc"] > pooled["popularity"]["roc_auc"]
        assert pooled["model"]["roc_auc"] > pooled["bayesian"]["roc_auc"]
        assert pooled["model"]["pairwise"] > pooled["popularity"]["pairwise"]
        assert pooled["model"]["pairwise"] > pooled["bayesian"]["pairwise"]


def test_disaggregation_shape():
    with tempfile.TemporaryDirectory() as d:
        result, *_ = _train(d)
        by_tone = result["by_tone"]
        for bucket in ("light", "medium", "deep"):
            assert bucket in by_tone, sorted(by_tone)
            row = by_tone[bucket]
            assert "n" in row and "low_n" in row
            for method in ("model", "popularity", "bayesian"):
                assert "roc_auc" in row[method] and "pairwise" in row[method]
        # 'unknown' is representable, never silently dropped (present, explicitly empty)
        assert "unknown" in by_tone
        assert by_tone["unknown"]["n"] == 0


def test_loaded_ranker_reorders_by_skin_type():
    with tempfile.TemporaryDirectory() as d:
        _result, model_path, stats_path, _eval_path = _train(d)
        ranker = R.load_ranker(config=_config_at(model_path, stats_path))
        assert ranker is not None
        # PC is a comedogenic salicylic clone: must sort LAST regardless of score.
        catalog = [
            Product("PB", "Ceramide Treatment", "BrandY", "treatment",
                    actives=["ceramides"], price_usd=18.0),
            Product("PC", "SA Balm", "BrandZ", "treatment",
                    actives=["salicylic_acid"], comedogenic_flags=["coconut_oil"],
                    price_usd=9.0),
            Product("PA", "SA Treatment", "BrandX", "treatment",
                    actives=["salicylic_acid"], price_usd=24.0),
        ]
        report = ConcernReport("img", concerns=[
            Concern("acne_comedonal", "nose", 2, 0.9),
            Concern("dryness", "left_cheek", 1, 0.9),
        ])
        oily = recommend(report, catalog,
                         profile=UserProfile(skin_type="oily"), ranker=ranker)
        dry = recommend(report, catalog,
                        profile=UserProfile(skin_type="dry"), ranker=ranker)
        oily_ids = [p.product_id for p in oily.routines["AM"]["treatment"]]
        dry_ids = [p.product_id for p in dry.routines["AM"]["treatment"]]
        # oily favors PA (salicylic); dry favors PB (ceramides); PC comedogenic last.
        assert oily_ids == ["PA", "PB", "PC"], oily_ids
        assert dry_ids == ["PB", "PA", "PC"], dry_ids


def test_evidence_fallback_below_min_cell_size():
    with tempfile.TemporaryDirectory() as d:
        _result, model_path, stats_path, _eval_path = _train(d)
        ranker = R.load_ranker(config=_config_at(model_path, stats_path))
        # skin_type absent from the fixture -> the __all__ cell tagged fallback.
        normal = ranker.evidence("PA", "normal")
        assert normal is not None and normal["fallback"] is True
        assert normal["cell"] == "all_reviewers"
        # a present skin_type well above min_cell_size -> its own cell, no fallback.
        oily = ranker.evidence("PA", "oily")
        assert oily is not None and not oily["fallback"]
        assert oily["cell"] == "oily"
        assert oily["n"] >= 5
        # absent product -> None.
        assert ranker.evidence("NOPE", "oily") is None


def test_degrades_to_rules_only_when_model_absent():
    with tempfile.TemporaryDirectory() as d:
        missing_model = Path(d) / "does_not_exist.joblib"
        ranker = R.load_ranker(config=_config_at(missing_model, Path(d) / "s.json"))
        assert ranker is None
        catalog = [
            Product("PB", "Ceramide Treatment", "BrandY", "treatment",
                    actives=["ceramides"], price_usd=18.0),
            Product("PA", "SA Treatment", "BrandX", "treatment",
                    actives=["salicylic_acid"], price_usd=24.0),
        ]
        report = ConcernReport("img", concerns=[Concern("dryness", "left_cheek", 1, 0.9)])
        rec = recommend(report, catalog, ranker=None)
        assert isinstance(rec, Recommendation)


def test_deterministic_split_is_stable():
    ids = pd.Series(["a00", "a01", "a02", "a03", "a17", "a42"])
    first = R.deterministic_test_mask(ids, 0.25)
    second = R.deterministic_test_mask(ids, 0.25)
    # same author -> same side across calls, no state file, no builtin hash().
    assert list(first) == list(second)


if __name__ == "__main__":
    test_end_to_end_fixture_gate_passes()
    test_model_beats_baselines()
    test_disaggregation_shape()
    test_loaded_ranker_reorders_by_skin_type()
    test_evidence_fallback_below_min_cell_size()
    test_degrades_to_rules_only_when_model_absent()
    test_deterministic_split_is_stable()
    print("ok")
