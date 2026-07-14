import json
from pathlib import Path

import pytest

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.knowledge import load_knowledge
from recsys.signals import (
    ConcernEfficacySignal, ScoringContext, TargetConcern, load_providers,
)
from recsys.tools.build_concern_efficacy import build


DATA = Path(__file__).parents[1] / "data"


def _record(uid, outcome, skin_type="oily", product_id="p1", has_condition=True):
    return {
        "uid": uid,
        "product_id": product_id,
        "skin_type": skin_type,
        "prompt_version": "p7",
        "status": "ok",
        "labels": [{
            "concern": "acne_comedonal",
            "outcome": outcome,
            "reviewer_has_condition": has_condition,
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


def test_build_ignores_labels_from_reviewers_without_the_condition(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join((
        json.dumps(_record("condition", "helped", has_condition=True)),
        json.dumps(_record("generic", "worsened", has_condition=False)),
    )) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    build(labels, out, data_root, catalog_products=1)

    cell = json.loads(out.read_text())["products"]["p1"]["acne_comedonal"]["all"]
    assert cell["n"] == 1
    assert cell["helped"] == 1
    assert cell["worsened"] == 0


def test_concern_signal_falls_back_to_pooled_review_evidence():
    provider = ConcernEfficacySignal(
        {"products": {}}, {"version": "v1"},
        pooled_store={"products": {"p1": {
            "n": 100, "mean": 4.5, "smoothed": 4.4,
        }}},
    )
    product = CatalogProduct(
        product_id="p1", name="Test", brand="Test", category="treatment",
        price_usd=10, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=(),
    )
    score = provider.score(product, "treatment", ScoringContext(
        targets=(TargetConcern("acne_cystic", 3, 0.9),),
        profile=Profile(skin_type="oily"),
        knowledge=load_knowledge(DATA / "knowledge"),
        category_prices={},
    ))

    assert score is not None
    assert score.details["matches"][0]["ladder"] == "pooled"
    assert score.details["matches"][0]["n"] == 100
    assert "4.4★ across 100 pooled reviews" in score.evidence


def test_concern_signal_uses_general_acne_before_pooled_review_evidence():
    provider = ConcernEfficacySignal(
        {"products": {"p1": {
            "acne_comedonal": {"all": {"n": 0}},
            "acne_general": {"all": {
                "n": 10, "helped": 8, "worsened": 2, "help_rate": 0.8,
                "smoothed": 0.7,
            }},
        }}},
        {"version": "v1"},
        pooled_store={"products": {"p1": {
            "n": 100, "mean": 4.5, "smoothed": 4.4,
        }}},
    )
    product = CatalogProduct(
        product_id="p1", name="Test", brand="Test", category="treatment",
        price_usd=10, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=(),
    )
    score = provider.score(product, "treatment", ScoringContext(
        targets=(TargetConcern("acne_comedonal", 3, 0.9),),
        profile=Profile(skin_type="oily"),
        knowledge=load_knowledge(DATA / "knowledge"),
        category_prices={},
    ))

    assert score.details["matches"][0]["ladder"] == "acne_general"
    assert score.details["matches"][0]["cell_concern"] == "acne_general"
    assert "80% of 10 reviewers" in score.evidence


def test_loaded_concern_signal_receives_pooled_review_store(tmp_path):
    signals = tmp_path / "signals"
    signals.mkdir()
    concern_path = signals / "concern.json"
    review_path = signals / "review.json"
    concern_path.write_text(json.dumps({"products": {}}))
    review_path.write_text(json.dumps({"products": {"p1": {
        "n": 100, "mean": 4.5, "smoothed": 4.4,
    }}}))
    from recsys.contracts import sha256_file
    registry = {
        "schema_version": "recsys-registry-1",
        "stores": [
            {"name": "concern", "kind": "concern_efficacy", "version": "v1",
             "path": "signals/concern.json", "sha256": sha256_file(concern_path),
             "status": "active"},
            {"name": "review", "kind": "review_stats", "version": "v1",
             "path": "signals/review.json", "sha256": sha256_file(review_path),
             "status": "active"},
        ],
    }
    (signals / "registry.json").write_text(json.dumps(registry))

    providers, _, _ = load_providers(tmp_path)
    concern = next(p for p in providers if isinstance(p, ConcernEfficacySignal))
    product = CatalogProduct(
        product_id="p1", name="Test", brand="Test", category="treatment",
        price_usd=10, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=(),
    )
    score = concern.score(product, "treatment", ScoringContext(
        targets=(TargetConcern("acne_cystic", 3, 0.9),),
        profile=Profile(skin_type="oily"),
        knowledge=load_knowledge(DATA / "knowledge"),
        category_prices={},
    ))

    assert score.details["matches"][0]["ladder"] == "pooled"


def test_failed_p3_bakeoff_does_not_register_concern_store(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(_record("one", "helped")) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"
    p3 = {
        "pooled": {
            "champion": {"roc_auc": 0.70, "pairwise": 0.60},
            "concern_conditioned": {"roc_auc": 0.71, "pairwise": 0.59},
        },
    }

    with pytest.raises(RuntimeError, match="P3 bake-off failed"):
        build(labels, out, data_root, catalog_products=1, p3_evaluation=p3)

    assert not out.exists()
    assert not (data_root / "signals" / "registry.json").exists()


def test_passing_p3_bakeoff_is_recorded_before_registration(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(_record("one", "helped")) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"
    p3 = {
        "pooled": {
            "champion": {"roc_auc": 0.70, "pairwise": 0.60},
            "concern_conditioned": {"roc_auc": 0.71, "pairwise": 0.61},
        },
    }

    coverage = build(labels, out, data_root, catalog_products=1, p3_evaluation=p3)

    store = json.loads(out.read_text())
    registry = json.loads((data_root / "signals" / "registry.json").read_text())
    assert coverage["p3_gate_passed"] is True
    assert store["p3"]["pooled"]["concern_conditioned"]["pairwise"] == 0.61
    assert registry["stores"][0]["source"]["p3"] == p3


def test_evaluable_labels_auto_run_the_p3_gate(tmp_path):
    record = _record("one", "helped")
    record.update(author_id="one", rating=5.0)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(record) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    with pytest.raises(RuntimeError, match="P3 bake-off failed"):
        build(labels, out, data_root, catalog_products=1)

    assert not (data_root / "signals" / "registry.json").exists()
