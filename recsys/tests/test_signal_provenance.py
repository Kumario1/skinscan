import hashlib
import json
from pathlib import Path

import pytest

from recsys.contracts import ContractViolation, sha256_file
from recsys.pipeline import run
from recsys.signals import ReviewQualitySignal, load_providers
from recsys.tools.build_popularity import build as build_popularity
from recsys.tools.build_review_stats import build as build_review_stats


DATA = Path(__file__).parents[1] / "data"
FIXTURES = Path(__file__).parent / "fixtures"
PASSING_P3 = {
    "pooled": {
        "champion": {"roc_auc": 0.70, "pairwise": 0.60},
        "concern_conditioned": {"roc_auc": 0.71, "pairwise": 0.61},
    },
}


def _registered_store(tmp_path, *, source=None, name="review_stats", kind="review_stats"):
    signals = tmp_path / "signals"
    signals.mkdir()
    store_path = signals / "review.json"
    store_path.write_text(json.dumps({"products": {}}))
    registry = {
        "schema_version": "recsys-registry-1",
        "stores": [{
            "name": name,
            "kind": kind,
            "version": "v1",
            "path": "signals/review.json",
            "sha256": hashlib.sha256(store_path.read_bytes()).hexdigest(),
            "status": "active",
        }],
    }
    if source is not None:
        registry["stores"][0]["source"] = source
    (signals / "registry.json").write_text(json.dumps(registry))


def test_catalog_bound_store_loads_for_matching_catalog(tmp_path):
    _registered_store(tmp_path, source={"catalog_sha256": "catalog-1"})

    providers, _meta, warnings = load_providers(tmp_path, "catalog-1")

    assert any(isinstance(provider, ReviewQualitySignal) for provider in providers)
    assert warnings == []


@pytest.mark.parametrize("name, kind", [
    ("review_stats", "review_stats"),
    ("ingredient_analysis", "ingredient_analysis"),
])
def test_catalog_mismatch_refuses_to_load_the_store(tmp_path, name, kind):
    """A store keyed by another catalog's product ids has nothing to say about
    the products being scored. Loading must fail loudly: dropping the store
    instead leaves every product at neutral 0.5 in a document that still reads
    like a real recommendation."""
    _registered_store(
        tmp_path, name=name, kind=kind, source={"catalog_sha256": "catalog-old"}
    )

    with pytest.raises(ContractViolation):
        load_providers(tmp_path, "catalog-new")


def test_catalog_mismatch_can_be_skipped_only_when_explicitly_allowed(tmp_path):
    _registered_store(tmp_path, source={"catalog_sha256": "catalog-old"})

    providers, meta, warnings = load_providers(
        tmp_path, "catalog-new", allow_catalog_mismatch=True
    )

    assert not any(isinstance(provider, ReviewQualitySignal) for provider in providers)
    assert meta == []
    assert warnings == [
        "catalog_sha256 mismatch for store 'review_stats': expected "
        "catalog-new, got catalog-old — skipped"
    ]


def test_bound_store_refuses_to_load_for_an_unnamed_catalog(tmp_path):
    """A caller that never names its catalog cannot have its stores checked."""
    _registered_store(tmp_path, source={"catalog_sha256": "catalog-1"})

    with pytest.raises(ContractViolation):
        load_providers(tmp_path)


def test_store_without_catalog_provenance_is_skipped(tmp_path):
    _registered_store(tmp_path)

    providers, _meta, warnings = load_providers(tmp_path, "catalog-1")

    assert not any(isinstance(provider, ReviewQualitySignal) for provider in providers)
    assert any("no catalog_sha256 provenance" in warning for warning in warnings)


def test_review_stats_builder_records_catalog_sha(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"products": [{"product_id": "p1"}]}))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reviews_1.csv").write_text(
        "rating,product_id,skin_type\n5,p1,oily\n"
    )
    data_root = tmp_path / "data"
    out = data_root / "signals" / "review.json"

    build_review_stats(raw, catalog, out, data_root)

    registry = json.loads((data_root / "signals" / "registry.json").read_text())
    assert registry["stores"][0]["source"]["catalog_sha256"] == sha256_file(catalog)


def test_popularity_builder_records_catalog_sha(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"products": [{"product_id": "p1"}]}))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "product_info.csv").write_text(
        "primary_category,secondary_category,tertiary_category,product_id,loves_count\n"
        "Skincare,Cleansers,Face Wash & Cleansers,p1,10\n"
    )
    data_root = tmp_path / "data"
    out = data_root / "signals" / "popularity.json"

    build_popularity(raw, catalog, out, data_root)

    registry = json.loads((data_root / "signals" / "registry.json").read_text())
    assert registry["stores"][0]["source"]["catalog_sha256"] == sha256_file(catalog)


def test_concern_efficacy_builder_records_catalog_sha(tmp_path):
    from recsys.signals import ConcernEfficacySignal
    from recsys.tools.build_concern_efficacy import PROMPT_VERSION
    from recsys.tools.build_concern_efficacy import build as build_concern

    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"products": [{"product_id": "p1"}]}))
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({
        "uid": "1", "product_id": "p1", "skin_type": "oily",
        "prompt_version": PROMPT_VERSION, "status": "ok",
        "labels": [{"concern": "acne_comedonal", "outcome": "helped",
                    "reviewer_has_condition": True}],
    }) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    build_concern(labels, out, data_root, catalog_products=1,
                  catalog_product_ids=frozenset({"p1"}), catalog_path=catalog,
                  p3_evaluation=PASSING_P3)

    registry = json.loads((data_root / "signals" / "registry.json").read_text())
    assert registry["stores"][0]["source"]["catalog_sha256"] == sha256_file(catalog)

    # And without that provenance load_providers would skip the catalog-bound
    # concern store; with it, the signal is present at inference.
    providers, _meta, warnings = load_providers(data_root, sha256_file(catalog))
    assert any(isinstance(provider, ConcernEfficacySignal) for provider in providers)
    assert not any("concern_efficacy" in warning for warning in warnings)


def test_pipeline_binds_runtime_stores_to_selected_catalog(tmp_path):
    catalog = tmp_path / "catalog_full.json"
    catalog.write_bytes((DATA / "catalog" / "seed_catalog.json").read_bytes())
    catalog_sha = sha256_file(catalog)
    signals = tmp_path / "signals"
    signals.mkdir()
    store = signals / "review.json"
    store.write_text(json.dumps({"products": {}}))
    (signals / "registry.json").write_text(json.dumps({
        "schema_version": "recsys-registry-1",
        "stores": [{
            "name": "review_stats",
            "kind": "review_stats",
            "version": "v1",
            "path": "signals/review.json",
            "sha256": hashlib.sha256(store.read_bytes()).hexdigest(),
            "source": {"catalog_sha256": catalog_sha},
            "status": "active",
        }],
    }))

    document = run(
        FIXTURES / "analysis_v3_sample.json",
        FIXTURES / "profile_complete.json",
        data_root=tmp_path,
        generated_at="2026-07-14T00:00:00+00:00",
    )

    assert [entry["name"] for entry in document["data_versions"]["signals"]] == [
        "review_stats"
    ]


def test_cli_writes_but_returns_three_when_catalog_mismatch_is_allowed(tmp_path):
    from recsys.__main__ import main

    catalog = tmp_path / "catalog_full.json"
    catalog.write_bytes((DATA / "catalog" / "seed_catalog.json").read_bytes())
    _registered_store(tmp_path, source={"catalog_sha256": "catalog-old"})
    out = tmp_path / "recommendations.json"

    code = main([
        "recommend",
        "--analysis", str(FIXTURES / "analysis_v3_sample.json"),
        "--profile", str(FIXTURES / "profile_complete.json"),
        "--data-root", str(tmp_path),
        "--generated-at", "2026-07-14T00:00:00+00:00",
        "--allow-signal-catalog-mismatch",
        "--out", str(out),
    ])

    document = json.loads(out.read_text())
    assert code == 3
    assert document["data_versions"]["signals"] == []
    assert any("catalog_sha256 mismatch" in warning for warning in document["warnings"])
