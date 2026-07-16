import json
import os
from pathlib import Path

import pytest

from recsys.catalog import load_catalog
from recsys.contracts import ContractViolation, SLOTS
from recsys.inci import parse_ingredients
from recsys.tools.common import DEFAULT_RAW_DIR

DATA = Path(__file__).parents[1] / "data"

# The dump is gitignored and lives at the repo root, so resolve it the same way
# the build tools do -- relative to the checkout -- rather than hardcoding one
# machine's home directory, which made the byte-identical claim untestable
# anywhere else (CI green said nothing). SKINSCAN_RAW_DIR points the checks at a
# dump outside the checkout: a git worktree has no data/raw/ of its own, so
# without it these skip here and only ever run in the main checkout.
RAW_DIR = Path(os.environ.get("SKINSCAN_RAW_DIR")
               or Path(__file__).parents[2] / DEFAULT_RAW_DIR)
needs_raw_dump = pytest.mark.skipif(
    not RAW_DIR.exists(),
    reason=f"Kaggle dump not present at {RAW_DIR} (set SKINSCAN_RAW_DIR)",
)


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


SPL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/set.xml"
EMPTY_INCI_SHA = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"


def _drug_row(**over):
    row = {
        "product_id": "dailymed:set:0-1:tretinoin-0.05%",
        "name": "Retin-A", "brand": "DailyMed SPL", "category": "treatment",
        "price_usd": None, "inci": [], "inci_sha256": EMPTY_INCI_SHA,
        "actives": ["tretinoin"], "otc_drug": False, "label_source": SPL,
        "drug_actives": [{"name": "tretinoin", "strength": "0.05%", "source": SPL}],
    }
    row.update(over)
    return row


def test_drug_row_derives_actives_from_the_label_not_an_inci_list():
    # A drug label publishes no INCI, but names each active with an exact
    # strength and cites the label it came from -- stronger than a parsed string.
    from recsys.catalog import CatalogProduct
    product = CatalogProduct.from_dict(_drug_row())
    assert product.actives == ("tretinoin",)
    assert product.otc_drug is False and product.price_usd is None


def test_label_exemption_does_not_leak_to_a_product_citing_another_source():
    # Anything not citing the regulatory label falls back to the INCI rule, so
    # actives=['tretinoin'] with an empty INCI must fail closed.
    from recsys.catalog import CatalogProduct
    for bad in (
        {"label_source": "https://brand.example.com/product"},   # not the label host
        {"label_source": "http://dailymed.nlm.nih.gov/spl.xml"},  # not https
        {"drug_actives": [{"name": "tretinoin", "source": SPL}]},  # no strength
        {"drug_actives": [{"name": "tretinoin", "strength": "0.05%",
                           "source": "https://brand.example.com/x"}]},  # unsourced active
        {"drug_actives": []},                                      # asserted, not stated
    ):
        with pytest.raises(ContractViolation):
            CatalogProduct.from_dict(_drug_row(**bad))


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


@pytest.mark.parametrize("field", ["inci_sha256", "actives"])
def test_catalog_rejects_stale_derived_inci_fields(tmp_path, field):
    data = json.loads((DATA / "catalog" / "seed_catalog.json").read_text())
    data["products"][0][field] = "bad" if field == "inci_sha256" else ["retinol"]
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ContractViolation, match=field):
        load_catalog(path)


@pytest.mark.raw_dump
@needs_raw_dump
def test_seed_rebuild_is_byte_identical(tmp_path):
    from recsys.tools.build_catalog import build

    out = tmp_path / "seed_catalog.json"
    build(RAW_DIR, out, DATA / "catalog" / "seed_ids.txt")
    assert out.read_bytes() == (DATA / "catalog" / "seed_catalog.json").read_bytes()


@pytest.mark.raw_dump
@needs_raw_dump
def test_review_stats_rebuild_is_byte_identical(tmp_path):
    # Committed, engine-read, and derived from 1.09M review rows -- but nothing
    # pinned it, so a change in the aggregation could land invisibly. The full
    # rebuild takes seconds, which is cheap enough to assert on every raw_dump run.
    from recsys.tools.build_review_stats import build

    out = tmp_path / "signals" / "review_stats.v1.json"
    build(RAW_DIR, DATA / "catalog" / "seed_catalog.json", out, tmp_path)
    assert out.read_bytes() == (DATA / "signals" / "review_stats.v1.json").read_bytes()


@pytest.mark.raw_dump
@needs_raw_dump
def test_popularity_rebuild_is_byte_identical(tmp_path):
    from recsys.tools.build_popularity import build

    out = tmp_path / "signals" / "popularity.v1.json"
    build(RAW_DIR, DATA / "catalog" / "seed_catalog.json", out, tmp_path)
    assert out.read_bytes() == (DATA / "signals" / "popularity.v1.json").read_bytes()
