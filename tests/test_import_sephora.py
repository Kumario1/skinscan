"""Tests for the Sephora import adapter (import_catalog.py, fmt="sephora").

Runs the importer in sephora format against a committed fixture of 12 real rows
pulled verbatim from the Kaggle `product_info.csv`, covering every mapping
branch: each of the five target categories, the non-obvious cross-secondary
mappings (Toners->cleanser, Exfoliators->treatment, Masks->treatment), and both
drop kinds (non-skincare primary + unmapped skincare pair). The point of the
adapter is that everything downstream of the per-row seam is untouched, so the
produced catalog still feeds engine.recommend() (D-009 schema unchanged).

Pure Python, no ML. Standalone via __main__ (pytest not required).
"""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendation.import_catalog import (
    import_csv,
    load_catalog,
    sephora_row_to_simple,
)
from src.recommendation.engine import Recommendation, recommend
from src.recommendation.schema import CATEGORIES, ConcernReport, Product

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sephora_sample.csv"


def _import(tmpdir) -> tuple[dict, list[Product]]:
    out = Path(tmpdir) / "catalog.json"
    log = import_csv(FIXTURE, out, fmt="sephora")
    return log, load_catalog(out)


def _by_id(products, pid):
    matches = [p for p in products if p.product_id == pid]
    assert len(matches) == 1, f"expected one {pid!r}, got {len(matches)}"
    return matches[0]


def test_counts_and_drop_breakdown():
    with tempfile.TemporaryDirectory() as d:
        log, _ = _import(d)
        assert log["rows"] == 15, log
        assert log["kept"] == 12, log
        assert log["dropped_category"] == 3, log
        assert log["with_actives"] == 12, log
        assert log["zero_actives"] == 0, log
        # the auditable dropped-by-category breakdown (sums to dropped_category)
        assert log["dropped_by_category"] == {
            "Makeup": 1,
            "Skincare / Eye Care / Eye Creams & Treatments": 1,
            "Skincare / Value & Gift Sets / ": 1,
        }, log
        assert sum(log["dropped_by_category"].values()) == log["dropped_category"]


def test_all_five_categories_nonempty():
    with tempfile.TemporaryDirectory() as d:
        log, products = _import(d)
        got = {p.category for p in products}
        assert got == set(CATEGORIES), got
        assert log["kept_by_category"] == {
            "cleanser": 3,
            "treatment": 3,
            "serum": 1,
            "moisturizer": 3,
            "spf": 2,
        }, log


def test_sephora_ids_preserved():
    # load-bearing for joining reviews later — the original Sephora id survives.
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        assert all(p.product_id.startswith("P") for p in products), \
            [p.product_id for p in products]
        toner = _by_id(products, "P480274")
        assert toner.category == "cleanser"          # Toners -> cleanser
        assert "salicylic_acid" in toner.actives     # reused parser, unchanged


def test_cross_secondary_mappings():
    # tertiary wins over the Sephora secondary grouping where they disagree.
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        assert _by_id(products, "P505338").category == "treatment"   # Cleansers/Exfoliators
        assert _by_id(products, "P442859").category == "treatment"   # Masks/Face Masks
        assert _by_id(products, "P311143").category == "spf"         # Sunscreen/Face Sunscreen


def test_empty_tertiary_fallback():
    # exact-string match on an empty tertiary — the branch a lookup bug hides in.
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        assert _by_id(products, "P504988").category == "cleanser"    # Cleansers/""
        assert _by_id(products, "P505209").category == "moisturizer"  # Moisturizers/""
        assert _by_id(products, "P504987").category == "spf"         # Sunscreen/""


def test_engine_compatible():
    # downstream is untouched: the catalog still shops through the engine.
    with tempfile.TemporaryDirectory() as d:
        _, products = _import(d)
        rec = recommend(ConcernReport("img", clear_skin=True), products)
        assert isinstance(rec, Recommendation)


def test_adapter_drops_and_maps():
    # unit-level seam checks, no file IO.
    assert sephora_row_to_simple({"primary_category": "Makeup"}) is None
    assert sephora_row_to_simple({
        "primary_category": "Skincare",
        "secondary_category": "Eye Care",
        "tertiary_category": "Eye Creams & Treatments",
    }) is None
    simple = sephora_row_to_simple({
        "product_id": "P1", "product_name": "X", "brand_name": "B",
        "ingredients": "Water, Niacinamide", "price_usd": "12",
        "primary_category": "Skincare",
        "secondary_category": "Moisturizers", "tertiary_category": "Face Oils",
    })
    assert simple["category"] == "moisturizer"
    assert simple["product_id"] == "P1"
    assert simple["name"] == "X" and simple["brand"] == "B"


def test_idempotent():
    with tempfile.TemporaryDirectory() as d:
        out1, out2 = Path(d) / "a.json", Path(d) / "b.json"
        import_csv(FIXTURE, out1, fmt="sephora")
        import_csv(FIXTURE, out2, fmt="sephora")
        assert out1.read_bytes() == out2.read_bytes()


if __name__ == "__main__":
    test_counts_and_drop_breakdown()
    test_all_five_categories_nonempty()
    test_sephora_ids_preserved()
    test_cross_secondary_mappings()
    test_empty_tertiary_fallback()
    test_engine_compatible()
    test_adapter_drops_and_maps()
    test_idempotent()
    print("ok")
