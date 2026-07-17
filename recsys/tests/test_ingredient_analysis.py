import json
from pathlib import Path

import pytest

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.knowledge import load_knowledge
from recsys.signals import REGISTRY_SCHEMA_VERSION, IngredientAnalysisSignal, ScoringContext
from recsys.tools.build_ingredient_analysis import (
    DEFAULT_MODEL,
    OUTPUT_SCHEMA,
    PROMPT_FINGERPRINT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    append_cache,
    build,
    check_prompt_fingerprint,
    label_product,
    label_products,
    prompt_fingerprint,
)
from recsys.tools.common import update_registry


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
    "inci_sha256": "42bdd5aaf9ef28381fc9ed2011322686e0ffedc432f56a716c15cf8731fd5fa0",
    "actives": [],
}


def analysis_entry():
    return {
        "product_id": "p1",
        "inci_sha256": PRODUCT["inci_sha256"],
        "prompt_version": PROMPT_VERSION,
        "model_id": "test/model",
        "actives_beyond_table": ["Beta Glucan"],
        "comedogenic_ingredients": ["Coconut Oil"],
        "irritancy_tier": "low",
        "fragrance_or_essential_oils": False,
        "concern_fit_notes": {"dryness": "Beta glucan may support hydration."},
    }


def batch_content(*product_ids):
    content = {k: v for k, v in analysis_entry().items()
               if k not in {"product_id", "inci_sha256", "prompt_version", "model_id"}}
    return json.dumps({"results": [
        {"product_id": product_id, "analysis": content}
        for product_id in product_ids
    ]})


def test_openrouter_structured_output(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": batch_content("p1")}}]}

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
            return Response(batch_content("p1"))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session = Session()
    entry = label_product(PRODUCT, "test/model", session, sleep=lambda _seconds: None)
    assert entry["irritancy_tier"] == "low"
    assert session.calls == 2


def test_openrouter_batches_multiple_products_in_one_request(monkeypatch):
    second = {**PRODUCT, "product_id": "p2", "inci_sha256": "def456"}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": batch_content("p1", "p2")}}]}

    class Session:
        calls = 0

        def post(self, _url, **kwargs):
            self.calls += 1
            self.request = kwargs
            return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session = Session()
    entries = label_products([PRODUCT, second], "test/model", session)

    assert [entry["product_id"] for entry in entries] == ["p1", "p2"]
    assert session.calls == 1
    request = session.request["json"]
    assert len(json.loads(request["messages"][1]["content"])) == 2
    item_schema = request["response_format"]["json_schema"]["schema"]["properties"]["results"]["items"]
    assert item_schema["properties"]["product_id"]["enum"] == ["p1", "p2"]


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

    build(catalog_path, out_path, data_root, cache_path, "test/model")
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


def _stale_entry_setup(tmp_path):
    """One product, fully cached, whose cached label names a dead concern id."""
    data_root = tmp_path / "data"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "schema_version": "recsys-catalog-1", "source": {}, "products": [PRODUCT],
    }))
    cache_path = data_root / "cache" / "ingredient_analysis.jsonl"
    stale = {**analysis_entry(),
             "concern_fit_notes": {"not_a_real_concern": "left over from an older CONCERNS set"}}
    append_cache(cache_path, stale)
    return catalog_path, data_root, cache_path


def test_resume_drops_cached_entry_with_now_invalid_concern_without_aborting(tmp_path):
    # The leniency this has always had, kept: a cached entry that no longer
    # validates against CONCERNS is dropped rather than raised through, so an
    # otherwise no-op rebuild still completes. Floored at 0 because here the
    # drop is the entire catalog -- what a drop must not do is pass a floor it
    # never cleared, which is the case below.
    catalog_path, data_root, cache_path = _stale_entry_setup(tmp_path)
    out_path = data_root / "signals" / "ingredient_analysis.v1.json"

    log = build(catalog_path, out_path, data_root, cache_path, "test/model",
                min_coverage=0.0)

    assert json.loads(out_path.read_text())["products"] == {}
    assert log["dropped_invalid"] == 1 and log["unlabeled"] == 0


def test_read_path_drop_counts_against_coverage_rather_than_the_cache(tmp_path):
    """A dropped entry is absent from the store, so it must fail the floor.

    This case previously asserted that build() writes an EMPTY store at the
    default 0.95 floor and exits 0 -- it pinned the bug. Coverage was counted
    from cache membership, so every entry the read path dropped still scored as
    "covered": a 10-product catalog reaching the store with 1 product logged
    coverage 1.0, and register_store then pointed the registry sha at that thin
    store. The engine loaded it with a matching sha and no warning, and scored
    the 9 absent products at a neutral 0.5. Coverage has to mean the fraction of
    the catalog that reaches the store -- an unlabeled product and a dropped one
    are equally missing from it.
    """
    catalog_path, data_root, cache_path = _stale_entry_setup(tmp_path)
    out_path = data_root / "signals" / "ingredient_analysis.v1.json"

    with pytest.raises(RuntimeError, match="coverage 0.0% below 95% floor"):
        build(catalog_path, out_path, data_root, cache_path, "test/model")
    assert not out_path.exists(), "a store below the floor must not be written"


def test_coverage_floor_is_enforced_when_no_request_failed(tmp_path):
    # The floor check used to be nested under `if failures:`, so a run whose
    # requests all succeeded -- or that issued none at all, as here, since the
    # catalog is fully cached -- skipped the check entirely. Coverage can be
    # zero with zero failures, so the floor must not depend on a failed request.
    catalog_path, data_root, cache_path = _stale_entry_setup(tmp_path)

    with pytest.raises(RuntimeError, match=r"0 unlabeled, 1 dropped as invalid"):
        build(catalog_path, data_root / "signals" / "out.json", data_root,
              cache_path, "test/model")


def test_build_caps_paid_calls(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "schema_version": "recsys-catalog-1", "source": {}, "products": [PRODUCT],
    }))
    with pytest.raises(SystemExit, match="refusing 1 paid labels"):
        build(catalog_path, tmp_path / "data/signals/out.json", tmp_path / "data",
              tmp_path / "cache.jsonl", "test/model", max_new_labels=0)


def _two_product_setup(tmp_path):
    # p2 shares p1's INCI (so it passes catalog validation) but a distinct
    # product_id, so its cache key is absent and it must be (re)labeled.
    p2 = {**PRODUCT, "product_id": "p2"}
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "schema_version": "recsys-catalog-1", "source": {}, "products": [PRODUCT, p2],
    }))
    cache_path = tmp_path / "data" / "cache" / "ing.jsonl"
    append_cache(cache_path, analysis_entry())          # p1 cached, p2 will fail
    return catalog_path, cache_path


def test_build_writes_partial_store_above_coverage_floor(tmp_path, monkeypatch):
    catalog_path, cache_path = _two_product_setup(tmp_path)
    out_path = tmp_path / "data" / "signals" / "out.json"

    def boom(*a, **k):
        raise RuntimeError("429 rate limited")   # p2 always fails, like the free tier
    monkeypatch.setattr("recsys.tools.build_ingredient_analysis.label_products", boom)

    # 1 of 2 cached = 50% coverage: clears a 0.5 floor -> store written from cache
    log = build(catalog_path, out_path, tmp_path / "data", cache_path, "test/model",
                max_new_labels=1, min_coverage=0.5)
    assert log["unlabeled"] == 1 and log["coverage"] == 0.5
    assert set(json.loads(out_path.read_text())["products"]) == {"p1"}


def test_build_aborts_below_coverage_floor(tmp_path, monkeypatch):
    catalog_path, cache_path = _two_product_setup(tmp_path)
    monkeypatch.setattr("recsys.tools.build_ingredient_analysis.label_products",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429")))
    # 50% coverage is below a 0.9 floor -> genuine failure surfaces
    with pytest.raises(RuntimeError, match="below 90% floor"):
        build(catalog_path, tmp_path / "data/signals/out.json", tmp_path / "data",
              cache_path, "test/model", max_new_labels=1, min_coverage=0.9)


def test_default_model_is_the_model_the_committed_store_was_built_with():
    """The documented rebuild passes no --model, so the default IS the contract.

    A commit whose subject was "batch ingredient analysis requests" also moved
    DEFAULT_MODEL to a model no cache or store entry has ever carried. model_id
    is part of the cache key, so the README's command scored 0 of 60 cache hits
    and -- since 60 is under the default --max-new-labels 100 -- would not have
    refused: it would have bought 60 fresh labels and overwritten the committed
    artifact plus its registry sha. Whatever model this names must be the one
    whose labels are actually committed.
    """
    store = json.loads((Path(__file__).parents[1] / "data" / "signals" /
                        "ingredient_analysis.v1.json").read_text())
    assert {entry["model_id"] for entry in store["products"].values()} == {DEFAULT_MODEL}
    assert store["prompt_version"] == PROMPT_VERSION


def _built_store(tmp_path, model="test/model"):
    """A store on disk, built from cache, as a committed artifact would be."""
    data_root = tmp_path / "data"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "schema_version": "recsys-catalog-1", "source": {}, "products": [PRODUCT],
    }))
    cache_path = data_root / "cache" / "ingredient_analysis.jsonl"
    append_cache(cache_path, {**analysis_entry(), "model_id": model})
    out_path = data_root / "signals" / "ingredient_analysis.v1.json"
    build(catalog_path, out_path, data_root, cache_path, model)
    return catalog_path, data_root, cache_path, out_path


def test_build_refuses_to_replace_a_store_whose_labels_came_from_another_model(tmp_path):
    # This store cannot be rederived from the dump -- it is model output, and
    # the provider is not deterministic even at temperature 0. Running it under
    # a different model does not rebuild the artifact, it relabels the catalog
    # with unrelated answers and moves the registry sha to match, so the engine
    # loads them silently. Refuse before spending anything.
    catalog_path, data_root, cache_path, out_path = _built_store(tmp_path)
    before = out_path.read_bytes()

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        build(catalog_path, out_path, data_root, cache_path, "other/model")

    assert out_path.read_bytes() == before, "the committed store must survive a refusal"


def test_build_refuses_a_store_built_under_a_different_prompt_version(tmp_path):
    # Same reasoning as the model guard: the prompt is half of what produced
    # these labels, so a prompt bump makes the existing store's labels answers
    # to a question no longer being asked.
    catalog_path, data_root, cache_path, out_path = _built_store(tmp_path)
    store = json.loads(out_path.read_text())
    store["prompt_version"] = "p0"
    out_path.write_text(json.dumps(store))

    with pytest.raises(SystemExit, match=r"prompt_version 'p0' -> 'p1'"):
        build(catalog_path, out_path, data_root, cache_path, "test/model")


def test_allow_model_change_relabels_a_store_on_purpose(tmp_path):
    # The guard refuses a drifted model; it must not make a deliberate relabel
    # impossible, only typed. With the new model's label cached, the rebuild is
    # a no-op that swaps the store's provenance to the requested model.
    catalog_path, data_root, cache_path, out_path = _built_store(tmp_path)
    append_cache(cache_path, {**analysis_entry(), "model_id": "other/model"})

    build(catalog_path, out_path, data_root, cache_path, "other/model",
          allow_model_change=True)

    store = json.loads(out_path.read_text())
    assert store["products"]["p1"]["model_id"] == "other/model"


def test_prompt_version_is_pinned_to_the_text_the_model_actually_sees():
    # PROMPT_VERSION is hand-maintained but SYSTEM_PROMPT and OUTPUT_SCHEMA are
    # free to change under it, and the cache key trusts the version alone. This
    # pin is the linkage: it fails the moment the prompt text drifts from the
    # version claiming to describe it.
    assert prompt_fingerprint() == PROMPT_FINGERPRINT, (
        "SYSTEM_PROMPT/OUTPUT_SCHEMA changed: bump PROMPT_VERSION and repin "
        f"PROMPT_FINGERPRINT to {prompt_fingerprint()!r}"
    )
    check_prompt_fingerprint()


@pytest.mark.parametrize("attr, value", [
    ("SYSTEM_PROMPT", SYSTEM_PROMPT + "\nAlso list every humectant.\n"),
    ("OUTPUT_SCHEMA", {**OUTPUT_SCHEMA, "properties": {}}),
])
def test_editing_the_prompt_without_bumping_the_version_is_refused(monkeypatch, attr, value):
    # Without this, editing the prompt silently reuses every label cached under
    # the old one: the key never moves, so the answers to the previous question
    # are served as answers to the new one.
    monkeypatch.setattr(f"recsys.tools.build_ingredient_analysis.{attr}", value)

    with pytest.raises(SystemExit, match="PROMPT_VERSION is still 'p1'"):
        check_prompt_fingerprint()


def test_update_registry_stamps_the_schema_version_it_writes(tmp_path):
    # The entries written here are this schema's, but the old version was read
    # back and written straight out again, so a rebuild left a stale registry
    # that load_providers rejects -- with no remedy but deleting the file the
    # builder had just written.
    registry_path = tmp_path / "signals" / "registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps({"schema_version": "recsys-registry-0", "stores": []}))

    update_registry(tmp_path, {"name": "ingredient_analysis", "kind": "ingredient_analysis"})

    assert json.loads(registry_path.read_text())["schema_version"] == REGISTRY_SCHEMA_VERSION
