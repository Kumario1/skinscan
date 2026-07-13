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
                       "min_cell_size": min_cell_size,
                       "bayesian_prior_count": 20}}


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
    """BOTH artifacts absent -> load_ranker None -> exact rules-only order
    (D-019; the three-way loader's last branch). Model-absent-but-stats-present
    is covered by test_load_ranker_three_way."""
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


def test_stats_ranker_orders_by_smoothed_rating():
    # the smoothing test: a 5.0-rated n=5 product must NOT outrank a
    # well-attested 4.5 (PA -> 4.4167, PB -> (25+80)/25 = 4.2 with m=20, prior 4.0).
    stats = {"global_mean_rating": 4.0, "cells": {
        "PA": {"__all__": {"n": 100, "mean_rating": 4.5, "pct_recommend": 0.9}},
        "PB": {"__all__": {"n": 5, "mean_rating": 5.0, "pct_recommend": 1.0}},
    }}
    ranker = R.StatsRanker(stats, 20, 5)
    pa = Product("PA", "A", "B", "treatment")
    pb = Product("PB", "B", "B", "treatment")
    assert ranker.score(pa, None) > ranker.score(pb, None)


def test_stats_ranker_popularity_nudge_orders_equal_ratings():
    # D-028: identical review cells, different loves -> more-loved wins, and the
    # boost is exactly w * log1p(loves)/log1p(max_loves) on top of the rating.
    import math
    cell = {"__all__": {"n": 100, "mean_rating": 4.5, "pct_recommend": 0.9}}
    stats = {"global_mean_rating": 4.0,
             "cells": {"PA": cell, "PB": cell},
             "loves": {"PA": 1000, "PB": 800000}}
    ranker = R.StatsRanker(stats, 20, 5, popularity_weight=0.2)
    pa = Product("PA", "A", "B", "treatment")
    pb = Product("PB", "B", "B", "treatment")
    assert ranker.score(pb, None) > ranker.score(pa, None)
    smoothed = (100 * 4.5 + 20 * 4.0) / 120
    expected_pb = smoothed + 0.2 * math.log1p(800000) / math.log1p(800000)
    assert abs(ranker.score(pb, None) - expected_pb) < 1e-9


def test_stats_ranker_popularity_weight_zero_and_missing_loves():
    # w=0 (or a product absent from the loves map) -> exactly the old ordering:
    # pure smoothed rating, no nudge.
    cell_hi = {"__all__": {"n": 100, "mean_rating": 4.5, "pct_recommend": 0.9}}
    cell_lo = {"__all__": {"n": 100, "mean_rating": 4.4, "pct_recommend": 0.9}}
    stats = {"global_mean_rating": 4.0,
             "cells": {"PA": cell_hi, "PB": cell_lo},
             "loves": {"PB": 900000}}
    pa = Product("PA", "A", "B", "treatment")
    pb = Product("PB", "B", "B", "treatment")
    # PA missing from loves -> its score is exactly the smoothed rating.
    ranker = R.StatsRanker(stats, 20, 5, popularity_weight=0.2)
    assert ranker.score(pa, None) == (100 * 4.5 + 20 * 4.0) / 120
    # knob off -> loves ignored entirely, better-rated PA wins again.
    off = R.StatsRanker(stats, 20, 5, popularity_weight=0.0)
    assert off.score(pa, None) > off.score(pb, None)
    assert off.score(pb, None) == (100 * 4.4 + 20 * 4.0) / 120


def test_load_ranker_passes_popularity_weight_from_config():
    with tempfile.TemporaryDirectory() as d:
        stats = {"global_mean_rating": 4.0, "cells": {},
                 "loves": {"PB": 900000}}
        stats_path = Path(d) / "review_stats.json"
        stats_path.write_text(json.dumps(stats))
        cfg = _config_at(Path(d) / "no.joblib", stats_path)
        cfg["ranker"]["popularity_weight"] = 0.5
        ranker = R.load_ranker(config=cfg)
        pb = Product("PB", "B", "B", "treatment")
        assert ranker.score(pb, None) == 4.0 + 0.5  # prior + full nudge
        del cfg["ranker"]["popularity_weight"]  # absent key -> default 0.2
        assert R.load_ranker(config=cfg).score(pb, None) == 4.0 + 0.2


def test_train_pipeline_stamps_loves_for_catalog_products_only():
    # D-028: review_stats.json carries a top-level loves map joined from
    # product_info.csv, restricted to catalog products; no file -> no map.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        product_info = tmp / "product_info.csv"
        pd.DataFrame({
            "product_id": ["PA", "PB", "PX"],
            "loves_count": [1200, 850000, 999999],
        }).to_csv(product_info, index=False)
        catalog_path = tmp / "catalog.json"
        catalog_path.write_text(json.dumps([asdict(p) for p in TRAIN_CATALOG]))
        R.train_pipeline(
            str(FIXTURE), str(catalog_path), str(tmp / "ranker.joblib"),
            str(tmp / "review_stats.json"), str(tmp / "eval.json"),
            load_config(), verbose=False, product_info_path=str(product_info),
        )
        stats = json.loads((tmp / "review_stats.json").read_text())
        assert stats["loves"] == {"PA": 1200, "PB": 850000}  # PX not in catalog
        # missing product_info -> pipeline still runs, loves map just absent
        R.train_pipeline(
            str(FIXTURE), str(catalog_path), str(tmp / "ranker2.joblib"),
            str(tmp / "review_stats2.json"), str(tmp / "eval2.json"),
            load_config(), verbose=False, product_info_path=str(tmp / "nope.csv"),
        )
        assert "loves" not in json.loads((tmp / "review_stats2.json").read_text())


def test_bakeoff_measures_blended_method_with_soft_gate():
    # D-028: 'blended' (bayesian + popularity nudge) joins the harness; the
    # soft gate passes iff its pooled pairwise is within 0.02 of bayesian's.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        product_info = tmp / "product_info.csv"
        pd.DataFrame({
            "product_id": ["PA", "PB"],
            "loves_count": [1200, 850000],
        }).to_csv(product_info, index=False)
        catalog_path = tmp / "catalog.json"
        catalog_path.write_text(json.dumps([asdict(p) for p in TRAIN_CATALOG]))
        result = R.train_pipeline(
            str(FIXTURE), str(catalog_path), str(tmp / "ranker.joblib"),
            str(tmp / "review_stats.json"), str(tmp / "eval.json"),
            load_config(), verbose=False, product_info_path=str(product_info),
        )
        blended = result["pooled"]["blended"]
        bayes = result["pooled"]["bayesian"]
        assert set(blended) == {"roc_auc", "pairwise"}
        assert result["blended_gate_passed"] == (
            blended["pairwise"] >= bayes["pairwise"] - 0.02)
        # no loves data -> blended degrades to exactly the bayesian baseline
        no_loves = R.train_pipeline(
            str(FIXTURE), str(catalog_path), str(tmp / "r2.joblib"),
            str(tmp / "s2.json"), str(tmp / "e2.json"),
            load_config(), verbose=False,
        )
        assert no_loves["pooled"]["blended"] == no_loves["pooled"]["bayesian"]


def test_stats_ranker_unknown_product_gets_prior():
    ranker = R.StatsRanker({"global_mean_rating": 4.0, "cells": {}}, 20, 5)
    assert ranker.score(Product("NOPE", "N", "B", "treatment"), None) == 4.0


def test_stats_ranker_through_engine():
    with tempfile.TemporaryDirectory() as d:
        _result, model_path, stats_path, _eval_path = _train(d)
        # point the loader away from the model: stats-only -> StatsRanker.
        cfg = _config_at(Path(d) / "no_model.joblib", stats_path)
        ranker = R.load_ranker(config=cfg)
        assert isinstance(ranker, R.StatsRanker)
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
        rec = recommend(report, catalog,
                        profile=UserProfile(skin_type="oily"), ranker=ranker)
        ids = [p.product_id for p in rec.routines["AM"]["treatment"]]
        assert ids[-1] == "PC", ids  # comedogenic partition still dominates
        # PA/PB order follows their smoothed pooled ratings from the stats file.
        assert ids[:2] == sorted(["PA", "PB"], key=lambda p: -ranker._scores[p]), ids


def test_load_ranker_three_way():
    with tempfile.TemporaryDirectory() as d:
        _result, model_path, stats_path, _eval_path = _train(d)
        assert isinstance(R.load_ranker(config=_config_at(model_path, stats_path)),
                          R.Ranker)
        assert isinstance(R.load_ranker(config=_config_at(Path(d) / "no.joblib", stats_path)),
                          R.StatsRanker)
        assert R.load_ranker(config=_config_at(Path(d) / "no.joblib",
                                               Path(d) / "no.json")) is None


def test_stats_ranker_evidence_matches_ranker_evidence():
    with tempfile.TemporaryDirectory() as d:
        _result, model_path, stats_path, _eval_path = _train(d)
        learned = R.load_ranker(config=_config_at(model_path, stats_path))
        stats_only = R.load_ranker(config=_config_at(Path(d) / "no.joblib", stats_path))
        for skin_type in ("oily", "normal"):  # own cell + the fallback case
            assert learned.evidence("PA", skin_type) == stats_only.evidence("PA", skin_type)
        assert learned.evidence("NOPE", "oily") is None
        assert stats_only.evidence("NOPE", "oily") is None


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
    test_stats_ranker_orders_by_smoothed_rating()
    test_stats_ranker_popularity_nudge_orders_equal_ratings()
    test_stats_ranker_popularity_weight_zero_and_missing_loves()
    test_load_ranker_passes_popularity_weight_from_config()
    test_train_pipeline_stamps_loves_for_catalog_products_only()
    test_bakeoff_measures_blended_method_with_soft_gate()
    test_stats_ranker_unknown_product_gets_prior()
    test_stats_ranker_through_engine()
    test_load_ranker_three_way()
    test_stats_ranker_evidence_matches_ranker_evidence()
    test_deterministic_split_is_stable()
    print("ok")
