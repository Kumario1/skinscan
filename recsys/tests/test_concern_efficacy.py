import json
from pathlib import Path

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.knowledge import load_knowledge
from recsys.signals import ConcernEfficacySignal, ScoringContext, TargetConcern
from recsys.tools.build_concern_efficacy import build


DATA = Path(__file__).parents[1] / "data"


def _record(uid, outcome, skin_type="oily", product_id="p1"):
    return {
        "uid": uid,
        "product_id": product_id,
        "skin_type": skin_type,
        "prompt_version": "p7",
        "status": "ok",
        "labels": [{
            "concern": "acne_comedonal",
            "outcome": outcome,
            "reviewer_has_condition": True,
        }],
    }


def test_cached_labels_build_registered_concern_signal(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(_record(str(i), "helped" if i < 8 else "worsened"))
                                 for i in range(10)) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    build(labels, out, data_root, catalog_products=1, smoothing_m=20, sub_cell_min_n=5)

    store = json.loads(out.read_text())
    registry = json.loads((data_root / "signals" / "registry.json").read_text())
    assert registry["stores"][0]["kind"] == "concern_efficacy"
    assert registry["stores"][0]["coverage"] == {
        "catalog_products": 1,
        "products": 1,
        "products_with_acne_cell_n15": 0,
    }

    provider = ConcernEfficacySignal(store, {"version": "v1"})
    product = CatalogProduct(
        product_id="p1", name="Test", brand="Test", category="treatment",
        price_usd=10, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=("salicylic_acid",),
    )
    score = provider.score(product, "treatment", ScoringContext(
        targets=(TargetConcern("acne_comedonal", 3, 0.9),),
        profile=Profile(skin_type="oily"),
        knowledge=load_knowledge(DATA / "knowledge"),
        category_prices={},
    ))
    assert 0.5 < score.value < store["products"]["p1"]["acne_comedonal"]["by_skin_type"]["oily"]["smoothed"]
    assert "80% of 10 reviewers" in score.evidence
    assert score.details["matches"][0]["ladder"] == "exact"


def test_build_only_includes_products_in_selected_catalog(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join((
        json.dumps(_record("in", "helped")),
        json.dumps(_record("out", "helped", product_id="p2")),
    )) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    coverage = build(
        labels,
        out,
        data_root,
        catalog_products=1,
        catalog_product_ids=frozenset({"p1"}),
    )

    store = json.loads(out.read_text())
    assert set(store["products"]) == {"p1"}
    assert coverage["products"] == 1
