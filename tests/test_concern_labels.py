"""Tests for the concern-efficacy labeling pipeline (plan 015, D-023).

Pure-Python: the LLM sits behind a duck-typed labeler seam and is stubbed;
no network, no anthropic import needed for this suite.
"""
import json
import tempfile
from pathlib import Path

from src.config import load_config
from src.recommendation.concern_labels import (
    CONCERNS,
    compile_prefilter,
    load_review_rows,
    review_uid,
)


def _patterns():
    return compile_prefilter(load_config()["concern"]["prefilter"])


def test_prefilter_flags_concerns():
    p = _patterns()
    assert p["acne_comedonal"].search("this cleared my blackheads fast")
    assert p["acne_cystic"].search("my hormonal acne is gone")
    assert p["acne_general"].search("it broke me out badly")
    assert p["hyperpigmentation"].search("faded my dark spots in weeks")
    assert p["dryness"].search("no more flaky patches")
    assert not p["acne_comedonal"].search("a blackheadless routine")  # word boundary
    assert not any(rx.search("lovely texture and smell") for rx in p.values())


def test_review_uid_stable_and_distinct():
    a = review_uid("123", "P1", "great product " * 50)
    b = review_uid("123", "P1", "great product " * 50)
    assert a == b and len(a) == 32
    assert review_uid("124", "P1", "great product") != a


def test_load_review_rows_prefilters_joins_and_truncates():
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "reviews_test.csv"
        csv.write_text(
            "author_id,rating,is_recommended,skin_tone,skin_type,"
            "product_id,review_text,review_title\n"
            'a1,5,1.0,fair,oily,PA,"cleared my blackheads ' + "x" * 100 + '",great\n'
            'a2,4,1.0,fair,dry,PA,"smells lovely",nice\n'          # no concern
            'a3,2,0.0,fair,dry,PX,"broke me out",bad\n'            # not in catalog
        )
        rows = load_review_rows(td, {"PA"}, _patterns(), truncate_chars=40)
        assert len(rows) == 1
        assert rows[0]["product_id"] == "PA" and rows[0]["skin_type"] == "oily"
        assert len(rows[0]["text"]) == 40
