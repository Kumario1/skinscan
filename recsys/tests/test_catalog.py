from pathlib import Path

import pytest

from recsys.catalog import load_catalog
from recsys.contracts import SLOTS
from recsys.inci import parse_ingredients

DATA = Path(__file__).parents[1] / "data"
RAW_DIR = Path("/Users/princekumar/Documents/skinscan/data/raw/sephora")


def test_parse_ingredients_real_dump_forms():
    # cases carried over from the src/recommendation importer's hard-won fixes
    actives, _ = parse_ingredients("Water, Ceramide NP, Ceramide AP")
    assert actives == ["ceramides"]
    actives, _ = parse_ingredients("Centella Asiatica (Gotu Kola) Extract")
    assert actives == ["centella"]
    actives, _ = parse_ingredients("Ascorbic Acid (Vitamin C), Sodium Hyaluronate")
    assert actives == ["hyaluronic_acid", "vitamin_c"]
    actives, _ = parse_ingredients("Benzoyl Peroxide 2.5%, Glycerin")
    assert "benzoyl_peroxide" in actives and "glycerin" in actives
    _, flags = parse_ingredients("Cocos Nucifera (Coconut Oil), Water")
    assert flags == ["coconut_oil"]


def test_seed_catalog_valid_and_covering():
    products, header = load_catalog(DATA / "catalog" / "seed_catalog.json")
    assert header["source"]["dataset"] == "kaggle-sephora"
    by_category = {slot: [p for p in products if p.category == slot] for slot in SLOTS}
    for slot in SLOTS:
        assert by_category[slot], f"no seed products for {slot}"
    assert len({p.product_id for p in products}) == len(products)
    # SPF slot must have usable (>= 30) options for the gates to pass
    assert any((p.spf or 0) >= 30 for p in by_category["spf"])
    # every product with a parsed SPF is marked name_parse until Phase 3 verifies
    assert all(p.spf_source == "name_parse" for p in products if p.spf is not None)
    # gentle + budget coverage that the archetypes depend on
    assert any(p.price_usd is not None and p.price_usd <= 20 for p in by_category["spf"])
    assert any("retinol" in p.actives for p in products)
    assert any("benzoyl_peroxide" in p.actives for p in products)


@pytest.mark.raw_dump
@pytest.mark.skipif(not RAW_DIR.exists(), reason="Kaggle dump not present")
def test_seed_rebuild_is_byte_identical(tmp_path):
    from recsys.tools.build_catalog import build

    out = tmp_path / "seed_catalog.json"
    build(RAW_DIR, out, DATA / "catalog" / "seed_ids.txt")
    assert out.read_bytes() == (DATA / "catalog" / "seed_catalog.json").read_bytes()
