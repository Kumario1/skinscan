"""Tests for the catalog importer (src/recommendation/import_catalog.py).

Runs the importer against a committed fixture CSV that exercises every rule:
active matching, parenthetical aliases, synonyms, comedogenic flags, category
dropping, punctuation/number tolerance, and price parsing. Test 7 is the
integration proof — the produced catalog feeds engine.recommend() unchanged.

Pure Python, no ML. Standalone via __main__ (pytest not required) but named
test_* so `pytest tests/` also works.
"""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendation.import_catalog import import_csv, load_catalog
from src.recommendation.engine import Recommendation, recommend
from src.recommendation.schema import ConcernReport, Product

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "catalog_sample.csv"


def _import(tmpdir) -> tuple[dict, list[Product]]:
    out = Path(tmpdir) / "catalog.json"
    log = import_csv(FIXTURE, out)
    return log, load_catalog(out)


def _by_name(products, name):
    matches = [p for p in products if p.name == name]
    assert len(matches) == 1, f"expected one {name!r}, got {len(matches)}"
    return matches[0]


def test_import_counts():
    with tempfile.TemporaryDirectory() as d:
        log, _ = _import(d)
        assert log == {
            "rows": 8,
            "kept": 7,
            "dropped_category": 1,
            "with_actives": 5,
            "zero_actives": 2,
        }, log


def test_parenthetical_alias():
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        assert _by_name(products, "Vitamin C Serum").actives == ["vitamin_c"]


def test_synonym_and_multi():
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        assert set(_by_name(products, "HA Moisturizer").actives) == {
            "glycerin",
            "hyaluronic_acid",
        }


def test_comedogenic_flagged():
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        p = _by_name(products, "Coconut Moisturizer")
        assert "coconut_oil" in p.comedogenic_flags
        assert "ceramides" in p.actives


def test_dropped_category():
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        assert all(p.name != "Tinted Foundation" for p in products)


def test_price_handling():
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        # garbage price ("?") -> None
        assert _by_name(products, "BP Spot Treatment").price_usd is None
        # clean float parses
        assert _by_name(products, "SA Cleanser").price_usd == 15.99
        # prices are never trusted
        assert all(p.price_is_stale is True for p in products)


def test_load_catalog_round_trip():
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        assert products and all(isinstance(p, Product) for p in products)
        # the whole point: this catalog is engine-compatible
        rec = recommend(ConcernReport("img", clear_skin=True), products)
        assert isinstance(rec, Recommendation)


def test_idempotent():
    with tempfile.TemporaryDirectory() as d:
        out1 = Path(d) / "a.json"
        out2 = Path(d) / "b.json"
        import_csv(FIXTURE, out1)
        import_csv(FIXTURE, out2)
        assert out1.read_bytes() == out2.read_bytes()


if __name__ == "__main__":
    test_import_counts()
    test_parenthetical_alias()
    test_synonym_and_multi()
    test_comedogenic_flagged()
    test_dropped_category()
    test_price_handling()
    test_load_catalog_round_trip()
    test_idempotent()
    print("ok")
