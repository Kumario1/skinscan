import json
from pathlib import Path

import pytest

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.knowledge import load_knowledge
from recsys.signals import IngredientAnalysisSignal, ScoringContext
from recsys.tools.build_ingredient_analysis import (
    PROMPT_VERSION,
    append_cache,
    build,
    label_product,
)


PRODUCT = {
    "product_id": "p1",
    "name": "Test Serum",
    "brand": "Test",
    "category": "serum",
    "price_usd": 12.0,
    "size": None,
    "format": "serum",
    "spf": None,
    "spf_source": None,
    "inci": ["Water", "Cocos Nucifera Oil"],
    "inci_sha256": "abc123",
    "actives": [],
}


def analysis_entry():
    return {
        "product_id": "p1",
        "inci_sha256": "abc123",
        "prompt_version": PROMPT_VERSION,
        "model_id": "test/model",
        "actives_beyond_table": ["Beta Glucan"],
        "comedogenic_ingredients": ["Coconut Oil"],
        "irritancy_tier": "low",
        "fragrance_or_essential_oils": False,
        "concern_fit_notes": {"dryness": "Beta glucan may support hydration."},
    }


def test_openrouter_structured_output(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            content = {k: v for k, v in analysis_entry().items()
                       if k not in {"product_id", "inci_sha256", "prompt_version", "model_id"}}
            return {"choices": [{"message": {"content": json.dumps(content)}}]}

    class Session:
        def post(self, _url, **kwargs):
            self.request = kwargs
            return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session = Session()
    entry = label_product(PRODUCT, "test/model", session)
    assert entry["comedogenic_ingredients"] == ["coconut oil"]
    assert session.request["json"]["response_format"]["type"] == "json_schema"
    assert session.request["json"]["provider"] == {"require_parameters": True}


def test_free_endpoint_malformed_reply_is_retried(monkeypatch):
    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    class Session:
        calls = 0

        def post(self, _url, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return Response('{"irritancy_tier":')
            content = {k: v for k, v in analysis_entry().items()
                       if k not in {"product_id", "inci_sha256", "prompt_version", "model_id"}}
            return Response(json.dumps(content))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session = Session()
    entry = label_product(PRODUCT, "test/model", session, sleep=lambda _seconds: None)
    assert entry["irritancy_tier"] == "low"
    assert session.calls == 2


def test_cached_build_registers_provider(tmp_path):
    data_root = tmp_path / "data"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "schema_version": "recsys-catalog-1",
        "source": {},
        "products": [PRODUCT],
    }))
    cache_path = data_root / "cache" / "ingredient_analysis.jsonl"
    append_cache(cache_path, analysis_entry())
    out_path = data_root / "signals" / "ingredient_analysis.v1.json"

    build(catalog_path, out_path, data_root, cache_path, "unused/model")
    store = json.loads(out_path.read_text())
    registry = json.loads((data_root / "signals" / "registry.json").read_text())
    assert registry["stores"][0]["source"]["model_ids"] == ["test/model"]

    provider = IngredientAnalysisSignal(store, {"version": "v1"})
    product = CatalogProduct.from_dict(PRODUCT)
    knowledge = load_knowledge(Path(__file__).parents[1] / "data" / "knowledge")
    score = provider.score(product, "serum", ScoringContext(
        targets=(), profile=Profile(), knowledge=knowledge, category_prices={},
    ))
    assert score.value == 1.0
    assert "coconut oil" in score.evidence


def test_build_caps_paid_calls(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "schema_version": "recsys-catalog-1", "source": {}, "products": [PRODUCT],
    }))
    with pytest.raises(SystemExit, match="refusing 1 paid labels"):
        build(catalog_path, tmp_path / "data/signals/out.json", tmp_path / "data",
              tmp_path / "cache.jsonl", "test/model", max_new_labels=0)
