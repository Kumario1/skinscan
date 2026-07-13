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

from src.recommendation.import_catalog import (
    import_csv, load_catalog, parse_ingredients, product_from_row,
    sephora_row_to_simple,
)
from src.recommendation.engine import Recommendation, recommend
from src.recommendation.schema import ConcernReport, Product

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "catalog_sample.csv"
SEPHORA_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sephora_sample.csv"
VERIFICATION = Path(__file__).resolve().parent / "fixtures" / "catalog_verification_sample.json"


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


def test_centella_inci_variants_all_map_to_centella():
    """e2e finding (2026-07-13): the Sephora dump never writes bare "centella" —
    it writes the INCI forms below. Every one must map to the canonical id, or
    the soothe path's signature active can never appear in a routine."""
    variants = [
        "Centella Asiatica Extract",
        "Centella Asiatica Leaf Extract",
        "Centella Asiatica Leaf Water",
        "Centella Asiatica Leaf Cell Extract",
        "Centella Asiatica Meristem Cell Culture",
        "Centella Asiatica (Hydrocotyl) Extract",
        "Centella asiatica (Gotu Kola) Extract*",
        "Centella Asiatica Flower/Leaf/Stem Extract",
    ]
    for raw in variants:
        actives, _ = parse_ingredients(raw)
        assert actives == ["centella"], f"{raw!r} -> {actives}"


def test_ceramide_inci_variants_all_map_to_ceramides():
    """e2e finding (2026-07-13, random-147 run): Sephora INCI writes suffixed
    ceramide codes ("Ceramide NP" x250) far more often than plain "Ceramides"
    (x11) — without these, ceramide barrier products are mostly untagged and
    target_coverage for ceramides collapses to 0."""
    variants = [
        "Ceramide NP", "Ceramide AP", "Ceramide EOP",
        "Ceramide NG", "Ceramide NS", "Ceramide EOS",
    ]
    for raw in variants:
        actives, _ = parse_ingredients(raw)
        assert actives == ["ceramides"], f"{raw!r} -> {actives}"


def test_toner_names_map_to_treatment_not_cleanser():
    """e2e finding (2026-07-13, run 262): toners are leave-on, not rinse-off —
    a BHA/azelaic toner in the cleanser slot misstates delivery. Both the
    Sephora taxonomy and the beautyapi name rules must agree."""
    from src.recommendation.import_catalog import infer_beautyapi_category
    assert infer_beautyapi_category("Hydrating Rose Toner", "skincare") == "treatment"


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


def test_exfoliant_source_vocabulary():
    """Marketing-named exfoliants must be caught from INCI truth: betaine
    salicylate IS a BHA; gluconolactone (PHA) and willow bark are exfoliant
    sources the soothe path must be able to veto (RULES.md §4)."""
    actives, _ = parse_ingredients(
        "Water, Betaine Salicylate, Gluconolactone, "
        "Salix Nigra (Willow) Bark Extract, Sodium Hyaluronate")
    assert "salicylic_acid" in actives, actives
    assert "gluconolactone" in actives, actives
    assert "willow_bark" in actives, actives


def test_verification_overlay_enriches_treatment_and_writes_quarantine(tmp_path):
    out = tmp_path / "catalog.json"
    quarantine = tmp_path / "quarantine.json"
    import_csv(SEPHORA_FIXTURE, out, fmt="sephora", verification=VERIFICATION,
               quarantine_out=quarantine)
    products = load_catalog(out)
    azelaic = next(product for product in products if product.product_id == "P480274")
    assert azelaic.routine_roles == ["treatment"]
    assert azelaic.intended_areas == ["face"]
    assert azelaic.format == "solution" and azelaic.exposure == "leave_on"
    assert azelaic.drug_actives[0].strength == "10%"
    assert azelaic.drug_actives[0].source == "synthetic://aza-label"
    report = __import__("json").loads(quarantine.read_text())
    assert report["products"]["P480274"]["quarantined_roles"] == {}


def test_product_without_overlay_stays_stored_but_quarantined(tmp_path):
    out = tmp_path / "catalog.json"
    quarantine = tmp_path / "quarantine.json"
    import_csv(SEPHORA_FIXTURE, out, fmt="sephora", quarantine_out=quarantine)
    products = load_catalog(out)
    product = next(product for product in products if product.product_id == "P480274")
    assert product.drug_actives == []
    report = __import__("json").loads(quarantine.read_text())
    reasons = report["products"]["P480274"]["quarantined_roles"]["treatment"]
    assert "drug_active_not_verified" in reasons
    assert "label_source_missing" in reasons


def test_raw_source_taxonomy_never_grants_a_verified_routine_role(tmp_path):
    out = tmp_path / "catalog.json"
    import_csv(SEPHORA_FIXTURE, out, fmt="sephora")
    products = load_catalog(out)
    assert all(product.routine_roles == [] for product in products)


def test_neck_source_taxonomy_never_becomes_face_moisturizer():
    simple = sephora_row_to_simple({
        "product_id": "neck", "product_name": "Neck Serum", "brand_name": "B",
        "primary_category": "Skincare", "secondary_category": "Moisturizers",
        "tertiary_category": "Decollete & Neck Creams", "ingredients": "Glycerin",
    })
    product = product_from_row(simple, 0)
    assert product.intended_areas == ["neck"]
    assert "moisturizer" not in product.routine_roles


def test_mask_and_scrub_source_formats_remain_quarantined(tmp_path):
    out = tmp_path / "catalog.json"
    quarantine = tmp_path / "quarantine.json"
    import_csv(SEPHORA_FIXTURE, out, fmt="sephora", quarantine_out=quarantine)
    report = __import__("json").loads(quarantine.read_text())
    mask = next(product for product in load_catalog(out) if product.product_id == "P442859")
    assert mask.exposure == "mask"
    assert "non_daily_format" in report["products"]["P442859"]["quarantined_roles"]["treatment"]


def test_spf_requires_broad_spectrum_and_verified_spf30(tmp_path):
    out = tmp_path / "catalog.json"
    quarantine = tmp_path / "quarantine.json"
    import_csv(SEPHORA_FIXTURE, out, fmt="sephora", verification=VERIFICATION,
               quarantine_out=quarantine)
    report = __import__("json").loads(quarantine.read_text())
    assert "broad_spectrum_not_verified" in report["products"]["P311143"][
        "quarantined_roles"
    ]["sunscreen"]
    assert report["products"]["P504987"]["quarantined_roles"] == {}


def test_unknown_verification_values_do_not_turn_into_false(tmp_path):
    out = tmp_path / "catalog.json"
    import_csv(SEPHORA_FIXTURE, out, fmt="sephora")
    product = next(product for product in load_catalog(out) if product.product_id == "P505209")
    assert product.comedogenic_claim == "unknown"
    assert product.contraindications == []
    assert product.broad_spectrum is None


def test_malformed_overlay_names_product_and_field(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"products":[{"product_id":"P480274","drug_actives":"no"}]}')
    try:
        import_csv(SEPHORA_FIXTURE, tmp_path / "out.json", fmt="sephora", verification=bad)
    except ValueError as exc:
        assert "P480274" in str(exc)
        assert "drug_actives" in str(exc)
    else:
        raise AssertionError("malformed overlay should fail")


def test_wrong_typed_spf_overlay_names_product_and_field(tmp_path):
    bad = tmp_path / "bad-spf.json"
    bad.write_text('{"products":[{"product_id":"P504987","spf":"50"}]}')
    try:
        import_csv(SEPHORA_FIXTURE, tmp_path / "out.json", fmt="sephora", verification=bad)
    except ValueError as exc:
        assert "P504987" in str(exc)
        assert "spf" in str(exc)
    else:
        raise AssertionError("wrong-typed SPF should fail")


def test_verified_import_and_quarantine_are_byte_identical(tmp_path):
    first_catalog, second_catalog = tmp_path / "a.json", tmp_path / "b.json"
    first_q, second_q = tmp_path / "aq.json", tmp_path / "bq.json"
    import_csv(SEPHORA_FIXTURE, first_catalog, fmt="sephora", verification=VERIFICATION,
               quarantine_out=first_q)
    import_csv(SEPHORA_FIXTURE, second_catalog, fmt="sephora", verification=VERIFICATION,
               quarantine_out=second_q)
    assert first_catalog.read_bytes() == second_catalog.read_bytes()
    assert first_q.read_bytes() == second_q.read_bytes()


if __name__ == "__main__":
    test_exfoliant_source_vocabulary()
    test_import_counts()
    test_parenthetical_alias()
    test_synonym_and_multi()
    test_comedogenic_flagged()
    test_dropped_category()
    test_price_handling()
    test_load_catalog_round_trip()
    test_idempotent()
    print("ok")
