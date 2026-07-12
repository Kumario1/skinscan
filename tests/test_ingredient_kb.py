"""Tests for the ingredient KB + match score + tier-2 catalog (spec
2026-07-10-ingredient-kb-design.md).

All fixture-driven — no network, no real data. A ~5-row handwritten
beautyproducts-style JSONL exercises KB aggregation (max-on-conflict, range
parsing, direct>supporting rating, function union, aliases), the pure
match_score (ordering, acne-only comedogenic penalty, INCI position discount),
the tier-2 importer (category inference + drops, tier/no_outcome_data), the
KB-less importer regression, and the engine's tier-2 fallback.

Pure Python, no ML. Standalone via __main__ (pytest not required) but named
test_* so `pytest tests/` also works.
"""
from pathlib import Path
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendation.ingredient_kb import (
    CONCERN_ACTIVES,
    build_alias_index,
    build_kb,
    kb_comedogenic_flags,
    match_score,
    product_matches,
)
from src.recommendation.import_catalog import (
    import_beautyapi,
    import_csv,
    load_catalog,
)
from src.recommendation.engine import recommend
from src.recommendation.schema import Concern, ConcernReport, Product

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "beautyapi_sample.jsonl"
SEPHORA_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sephora_sample.csv"


def _rows():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()
            if line.strip()]


def _kb():
    return build_kb(_rows())


# --- KB aggregation --------------------------------------------------------
def test_kb_max_on_conflict_and_range_parse():
    kb = _kb()
    sa = kb["salicylic acid"]
    # comedogenicity: row1 "1" vs row2 range "0-2" (->2) -> max 2
    assert sa["comedogenicity"] == 2, sa
    # irritancy: row1 "2" vs row2 range "1-3" (->3) -> max 3
    assert sa["irritancy"] == 3, sa


def test_kb_direct_beats_supporting_rating():
    kb = _kb()
    # salicylic acid is "supporting actives" in row1, "direct actives" in row2
    assert kb["salicylic acid"]["rating"] == "direct actives"


def test_kb_functions_are_unioned_and_sorted():
    kb = _kb()
    assert kb["salicylic acid"]["functions"] == [
        "exfoliating", "keratolytic", "skin conditioning",
    ]


def test_kb_collects_aliases():
    kb = _kb()
    aliases = kb["salicylic acid"]["aliases"]
    assert "acidum salicylicum" in aliases       # ph_eur_name
    assert "hydroxybenzoic acid" in aliases       # other_names (digit stripped)
    assert "salicylic acid" not in aliases        # the key never aliases itself


def test_alias_index_resolves_to_canonical_key():
    idx = build_alias_index(_kb())
    assert idx["acidum salicylicum"] == "salicylic acid"
    assert idx["cocos nucifera oil"] == "coconut oil"
    assert idx["salicylic acid"] == "salicylic acid"


def test_kb_is_deterministic_sorted():
    kb1, kb2 = build_kb(_rows()), build_kb(_rows())
    assert kb1 == kb2
    assert list(kb1) == sorted(kb1)


# --- match_score -----------------------------------------------------------
def test_match_score_ranks_better_product_higher():
    kb = _kb()
    good = "Water, Salicylic Acid, Niacinamide"     # two acne actives
    worse = "Water, Coconut Oil"                     # comedogenic, no active
    assert match_score(good, "acne_comedonal", kb) > \
        match_score(worse, "acne_comedonal", kb)
    assert match_score(good, "acne_comedonal", kb) > 0.5


def test_comedogenic_penalty_is_acne_only():
    kb = _kb()
    raw = "Water, Cocos Nucifera Oil"       # comedo 4 via alias, no active
    acne = match_score(raw, "acne_comedonal", kb)
    dry = match_score(raw, "dryness", kb)
    assert acne < 0.5           # penalised for an acne concern
    assert dry == 0.5           # no penalty, no active -> neutral
    assert acne < dry


def test_match_score_discounts_by_inci_position():
    kb = _kb()
    early = "Water, Salicylic Acid"
    late = "Water, X, X, X, X, Salicylic Acid"
    assert match_score(early, "acne_comedonal", kb) > \
        match_score(late, "acne_comedonal", kb)


def test_match_score_resolves_beneficial_via_alias():
    kb = _kb()
    # "Acidum Salicylicum" resolves to the canonical salicylic acid active
    assert match_score("Water, Acidum Salicylicum", "acne_comedonal", kb) > 0.5


def test_acne_scarring_match_metadata_includes_barrier_and_pigment_safe_ingredients():
    assert CONCERN_ACTIVES["acne_scarring"] == {
        "ceramide", "ceramides", "panthenol", "niacinamide", "azelaic acid",
        "centella",
    }


def test_product_matches_covers_every_concern():
    kb = _kb()
    scores = product_matches("Water, Salicylic Acid", kb)
    assert set(scores) == {
        "acne_comedonal", "acne_inflammatory", "acne_cystic",
        "acne_general", "acne_scarring", "hyperpigmentation", "dryness",
    }
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_kb_comedogenic_flags_uses_aliases():
    kb = _kb()
    flags = kb_comedogenic_flags("Water, Cocos Nucifera Oil, Glycerin", kb)
    assert "coconut_oil" in flags          # comedo 4 -> flagged (snake_cased key)
    assert "glycerin" not in flags         # comedo 0 -> not flagged


# --- tier-2 importer -------------------------------------------------------
def _import_tier2(tmpdir, kb=None):
    out = Path(tmpdir) / "catalog_tier2.json"
    log = import_beautyapi(FIXTURE, out, kb=kb)
    return log, load_catalog(out)


def test_tier2_infers_categories_and_drops_unmappable():
    with tempfile.TemporaryDirectory() as d:
        log, products = _import_tier2(d)
        cats = {p.product_id: p.category for p in products}
        assert cats["b101"] == "serum"        # "SA Renewing Serum"
        assert cats["b102"] == "cleanser"     # "Gentle Salicylic Toner"
        assert cats["b103"] == "moisturizer"  # "Coconut Rich Cream"
        assert cats["b104"] == "spf"          # suncare -> spf
        assert "b105" not in cats             # haircare shampoo -> dropped
        assert log["kept"] == 4 and log["dropped_category"] == 1


def test_tier2_products_flagged_no_outcome_data():
    with tempfile.TemporaryDirectory() as d:
        _, products = _import_tier2(d)
        assert products and all(p.tier == 2 for p in products)
        assert all(p.no_outcome_data for p in products)


def test_tier2_with_kb_enriches_flags_and_match():
    with tempfile.TemporaryDirectory() as d:
        _, products = _import_tier2(d, kb=_kb())
        cream = next(p for p in products if p.product_id == "b103")
        assert "coconut_oil" in cream.comedogenic_flags   # KB-derived
        assert cream.ingredient_match                      # per-concern scores present
        assert set(cream.ingredient_match) >= {"acne_comedonal", "dryness"}


# --- KB-less importer regression (byte-identical to before) ----------------
def test_importer_without_kb_is_byte_identical():
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.json"
        b = Path(d) / "b.json"
        import_csv(SEPHORA_FIXTURE, a, fmt="sephora")           # no kb
        import_csv(SEPHORA_FIXTURE, b, fmt="sephora", kb=None)  # explicit None
        assert a.read_bytes() == b.read_bytes()
        # and none of the new keys leak into the serialized output
        data = json.loads(a.read_text())
        assert all("ingredient_match" not in p for p in data)
        assert all("tier" not in p for p in data)
        assert all("no_outcome_data" not in p for p in data)


# --- engine tier-2 fallback ------------------------------------------------
def _report():
    return ConcernReport("img", concerns=[Concern("dryness", "left_cheek", 1, 0.9)])


def test_tier2_fills_only_empty_tier1_slots():
    # tier-1 moisturizer present -> tier-2 moisturizer must NOT appear;
    # no tier-1 serum -> tier-2 serum fills the slot with its flag intact.
    catalog = [
        Product("t1", "Tier1 Cream", "b", "moisturizer", actives=["ceramides"]),
        Product("t2", "Tier2 Cream", "b", "moisturizer", actives=["ceramides"],
                tier=2, no_outcome_data=True),
        Product("t3", "Tier2 Serum", "b", "serum", actives=["hyaluronic_acid"],
                tier=2, no_outcome_data=True),
    ]
    rec = recommend(_report(), catalog)
    moist = [p.product_id for p in rec.routine["moisturizer"]]
    serum = rec.routine["serum"]
    assert moist == ["t1"], moist               # tier-2 cream suppressed
    assert [p.product_id for p in serum] == ["t3"]
    assert serum[0].no_outcome_data is True     # flag carries through


def test_tier1_only_when_present_ignores_tier2_entirely():
    catalog = [
        Product("t1", "Tier1 Cream", "b", "moisturizer", actives=["ceramides"]),
        Product("t2", "Tier2 Cream", "b", "moisturizer", actives=["ceramides"],
                tier=2, no_outcome_data=True),
    ]
    rec = recommend(_report(), catalog)
    assert all(not p.no_outcome_data for p in rec.routine["moisturizer"])


def test_ingredient_match_is_tiebreaker_under_equal_comedogenic():
    # two clean tier-1 moisturizers, no ranker: higher dryness match sorts first.
    catalog = [
        Product("lo", "Low Match", "b", "moisturizer", actives=["ceramides"],
                ingredient_match={"dryness": 0.2}),
        Product("hi", "High Match", "b", "moisturizer", actives=["ceramides"],
                ingredient_match={"dryness": 0.9}),
    ]
    rec = recommend(_report(), catalog)
    assert [p.product_id for p in rec.routine["moisturizer"]] == ["hi", "lo"]


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
    print("ok")
