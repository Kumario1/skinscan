"""Optional catalog selector: safe candidates in, one validated regimen out."""
import json
from pathlib import Path

import pytest

from recsys.pipeline import CandidateSelection, run
from recsys.tools.llm_recommend import AzureCatalogSelector, main
from recsys.tools.render_html import render


FIXTURES = Path(__file__).parent / "fixtures"
ANALYSIS = FIXTURES / "analysis_v3_sample.json"
PROFILE = FIXTURES / "profile_complete.json"


def _active_treatment_analysis(tmp_path):
    data = json.loads(ANALYSIS.read_text())
    data["decision"].update(
        therapy_disposition="active_treatment",
        policy_reviewed=True,
        policy_version="test-reviewed:1",
    )
    data["policies"]["therapy"].update(
        reviewed=True, identity="test-reviewed:1", sha256="a" * 64,
    )
    data["therapy_plan"].update(
        primary={
            "therapy": "benzoyl_peroxide", "strength_band": "2.5%",
            "exposure": "leave_on", "cadence": "per_label",
            "role": "treatment", "cadence_source": "test-reviewed:1",
        },
        deferred_reasons=[],
        policy_version="test-reviewed:1",
    )
    path = tmp_path / "active-treatment-analysis.json"
    path.write_text(json.dumps(data))
    return path


def test_selector_chooses_one_regimen_from_post_gate_candidates():
    baseline = run(ANALYSIS, PROFILE, generated_at="2026-07-14T00:00:00+00:00")
    observed = {}

    def select(context, candidates, versions):
        observed.update(context=context, candidates=candidates, versions=versions)
        return CandidateSelection(
            product_ids={
                slot: baseline["selected_products"].get(slot)
                for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
            },
            metadata={"source": "test_selector", "cache_hit": False},
        )

    document = run(
        ANALYSIS,
        PROFILE,
        generated_at="2026-07-14T00:00:00+00:00",
        candidate_selector=select,
    )

    assert document["status"] == "ok"
    assert len(document["routines"]) == 1
    assert document["routines"][0]["archetype"] == "best_overall"
    assert document["selected_products"] == baseline["selected_products"]
    assert document["selection"]["source"] == "test_selector"
    assert set(observed["context"]) == {
        "profile", "lesion_findings", "care_pathways", "decision",
        "therapy_plan", "safety_observations", "required_slots",
    }
    assert observed["context"]["required_slots"] == ["cleanser", "moisturizer", "spf"]
    assert "source_image_sha256" not in str(observed["context"])
    assert all(
        pathway["status"] != "not_detected"
        for pathway in observed["context"]["care_pathways"]
    )
    assert all(
        not candidate.product.drug_actives or candidate.product.otc_drug is True
        for items in observed["candidates"].values()
        for candidate in items
    )


def test_out_of_pool_product_returns_no_result_without_fallback():
    def select(_context, _candidates, _versions):
        return CandidateSelection(
            product_ids={
                "cleanser": "not-in-catalog",
                "treatment": None,
                "serum": None,
                "moisturizer": "not-in-catalog",
                "spf": "not-in-catalog",
            },
            metadata={"source": "test_selector"},
        )

    document = run(ANALYSIS, PROFILE, candidate_selector=select)

    assert document["status"] == "unavailable"
    assert document["routines"] == []
    assert document["selected_regimen"] is None
    assert document["selected_products"] == {}
    assert document["reason"].startswith(
        "llm_selection_unavailable:product_not_safe_for_slot:cleanser"
    )


def test_active_treatment_selection_requires_a_treatment_product(tmp_path):
    analysis = _active_treatment_analysis(tmp_path)
    baseline = run(analysis, PROFILE)

    def select(context, _candidates, _versions):
        assert "treatment" in context["required_slots"]
        return CandidateSelection(
            {
                slot: baseline["selected_products"].get(slot)
                for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
            } | {"treatment": None},
            {"source": "test_selector"},
        )

    document = run(analysis, PROFILE, candidate_selector=select)

    assert document["status"] == "unavailable"
    assert document["reason"] == (
        "llm_selection_unavailable:required_slot_missing:treatment"
    )
    assert document["routines"] == []


def test_valid_active_treatment_selection_keeps_reviewed_therapy(tmp_path):
    analysis = _active_treatment_analysis(tmp_path)
    baseline = run(analysis, PROFILE)

    def select(_context, _candidates, _versions):
        return CandidateSelection(
            {
                slot: baseline["selected_products"].get(slot)
                for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
            },
            {"source": "test_selector"},
        )

    document = run(analysis, PROFILE, candidate_selector=select)

    assert document["status"] == "ok"
    assert document["selected_products"]["treatment"] == "P188306"
    assert document["treatment_fulfillment"] == {"status": "included", "reasons": []}


def test_selector_is_not_called_when_a_required_safe_pool_is_empty(tmp_path):
    analysis = _active_treatment_analysis(tmp_path)
    data = json.loads(analysis.read_text())
    data["therapy_plan"]["primary"].update(
        therapy="azelaic_acid", strength_band="20%",
    )
    analysis.write_text(json.dumps(data))
    calls = 0

    def select(_context, _candidates, _versions):
        nonlocal calls
        calls += 1
        raise AssertionError("an empty treatment pool must not trigger a paid call")

    document = run(analysis, PROFILE, candidate_selector=select)

    assert calls == 0
    assert document["status"] == "unavailable"
    assert document["reason"] == (
        "llm_selection_unavailable:required_candidate_pool_empty:treatment"
    )


def test_failed_whole_regimen_validation_returns_no_result_instead_of_substituting(
    monkeypatch,
):
    import recsys.pipeline as pipeline

    baseline = run(ANALYSIS, PROFILE)

    def select(_context, _candidates, _versions):
        return CandidateSelection(
            {
                slot: baseline["selected_products"].get(slot)
                for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
            },
            {"source": "test_selector"},
        )

    monkeypatch.setattr(
        pipeline,
        "validate_routine",
        lambda *_args, **_kwargs: ["routine_conflict:test-fixture"],
    )

    document = run(ANALYSIS, PROFILE, candidate_selector=select)

    assert document["status"] == "unavailable"
    assert document["routines"] == []
    assert document["reason"] == (
        "llm_selection_unavailable:regimen_validation_failed"
    )


def test_azure_selector_sends_compact_safe_catalog_and_caches_valid_result(tmp_path):
    baseline = run(ANALYSIS, PROFILE)
    selected = {
        slot: baseline["selected_products"].get(slot)
        for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
    }
    expected_safe_ids = set()

    def capture_safe_pool(_context, candidates, _versions):
        expected_safe_ids.update(
            (slot, candidate.product.product_id)
            for slot, items in candidates.items()
            for candidate in items
        )
        return CandidateSelection(selected, {"source": "capture"})

    assert run(
        ANALYSIS, PROFILE, candidate_selector=capture_safe_pool,
    )["status"] == "ok"

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "response-1",
                "output_text": json.dumps(selected),
                "usage": {"input_tokens": 321, "output_tokens": 17},
            }

    class Session:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    session = Session()
    cache = tmp_path / "selection-cache.jsonl"
    usage = tmp_path / "usage.jsonl"
    selector = AzureCatalogSelector(
        endpoint="https://azure.example/responses",
        api_key="secret",
        deployment="gpt-5-test",
        model_identity="gpt-5-test:fixture",
        cache_secret="cache-secret",
        cache_path=cache,
        usage_path=usage,
        session=session,
        input_price_per_million=2.0,
        output_price_per_million=8.0,
    )

    document = run(ANALYSIS, PROFILE, candidate_selector=selector)

    assert document["status"] == "ok"
    assert document["selected_products"] == baseline["selected_products"]
    assert document["selection"]["cache_hit"] is False
    assert document["selection"]["usage"] == {
        "input_tokens": 321,
        "output_tokens": 17,
        "estimated_cost_usd": 0.000778,
    }
    assert len(session.calls) == 1
    _, request = session.calls[0]
    body = request["json"]
    assert body["store"] is False
    assert body["text"]["format"]["strict"] is True
    payload = json.loads(body["input"])
    assert set(payload) == {"user", "candidates"}
    assert len(payload["candidates"]) == document["selection"]["candidate_count"]
    assert {
        (candidate["role"], candidate["product_id"])
        for candidate in payload["candidates"]
    } == expected_safe_ids
    assert sum(document["selection"]["candidate_counts"].values()) == len(
        expected_safe_ids
    )
    serialized = json.dumps(payload)
    for forbidden in (
        "source_image_sha256", "analysis_sha256", "veto_log",
        "detections", "lesion_sheet", "inci", "regions",
        "mean_detector_confidence", "max_detector_confidence", "evidence_source",
        "sha256",
    ):
        assert forbidden not in serialized
    assert cache.exists(), "validated selections are cached"
    cache_entry = json.loads(cache.read_text().strip())
    assert set(cache_entry) == {
        "cache_key", "product_ids", "provider", "model", "prompt_version",
    }
    assert len(cache_entry["cache_key"]) == 64
    assert "oily" not in cache.read_text()
    assert "pregnancy" not in cache.read_text()
    assert usage.exists()


def test_guarded_similarity_cache_reuses_valid_selection_without_azure(tmp_path):
    baseline = run(ANALYSIS, PROFILE)
    selected = {
        slot: baseline["selected_products"].get(slot)
        for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": json.dumps(selected), "usage": {}}

    class Session:
        def __init__(self):
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

    session = Session()
    selector = AzureCatalogSelector(
        endpoint="https://azure.example/responses", api_key="secret",
        deployment="gpt-5-test", model_identity="gpt-5-test:fixture",
        cache_secret="cache-secret",
        cache_path=tmp_path / "cache.jsonl", usage_path=tmp_path / "usage.jsonl",
        session=session,
    )

    first = run(ANALYSIS, PROFILE, candidate_selector=selector)
    second = run(ANALYSIS, PROFILE, candidate_selector=selector)

    assert first["selection"]["cache_hit"] is False
    assert second["selection"]["cache_hit"] is True
    assert second["selected_products"] == baseline["selected_products"]
    assert session.calls == 1


def test_invalid_cached_ids_become_a_miss_and_are_reselected(tmp_path):
    baseline = run(ANALYSIS, PROFILE)
    selected = {
        slot: baseline["selected_products"].get(slot)
        for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": json.dumps(selected), "usage": {}}

    class Session:
        def __init__(self):
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

    cache = tmp_path / "cache.jsonl"
    session = Session()
    selector = AzureCatalogSelector(
        endpoint="https://azure.example/responses", api_key="secret",
        deployment="gpt-5-test", model_identity="gpt-5-test:fixture",
        cache_secret="cache-secret",
        cache_path=cache, usage_path=tmp_path / "usage.jsonl", session=session,
    )
    first = run(ANALYSIS, PROFILE, candidate_selector=selector)
    corrupt = json.loads(cache.read_text().splitlines()[-1])
    corrupt["product_ids"]["cleanser"] = "no-longer-safe"
    cache.write_text(cache.read_text() + json.dumps(corrupt) + "\n")

    second = run(ANALYSIS, PROFILE, candidate_selector=selector)

    assert first["status"] == second["status"] == "ok"
    assert second["selection"]["cache_hit"] is False
    assert second["selection"]["cache_status"] == "invalidated"
    assert second["selected_products"] == baseline["selected_products"]
    assert session.calls == 2


def test_guarded_cache_misses_when_an_exact_safety_field_changes(tmp_path):
    baseline = run(ANALYSIS, PROFILE)
    selected = {
        slot: baseline["selected_products"].get(slot)
        for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": json.dumps(selected), "usage": {}}

    class Session:
        def __init__(self):
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

    changed = json.loads(PROFILE.read_text())
    changed["prior_scarring"] = True
    changed_profile = tmp_path / "changed-profile.json"
    changed_profile.write_text(json.dumps(changed))
    session = Session()
    selector = AzureCatalogSelector(
        endpoint="https://azure.example/responses", api_key="secret",
        deployment="gpt-5-test", model_identity="gpt-5-test:fixture",
        cache_secret="cache-secret",
        cache_path=tmp_path / "cache.jsonl", usage_path=tmp_path / "usage.jsonl",
        session=session,
    )

    first = run(ANALYSIS, PROFILE, candidate_selector=selector)
    second = run(ANALYSIS, changed_profile, candidate_selector=selector)

    assert first["status"] == second["status"] == "ok"
    assert first["selection"]["cache_key"] != second["selection"]["cache_key"]
    assert session.calls == 2


def test_cache_is_invalidated_when_model_identity_or_reasoning_changes(tmp_path):
    baseline = run(ANALYSIS, PROFILE)
    selected = {
        slot: baseline["selected_products"].get(slot)
        for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": json.dumps(selected), "usage": {}}

    class Session:
        def __init__(self):
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

    session = Session()
    common = dict(
        endpoint="https://azure.example/responses", api_key="secret",
        deployment="production-alias", cache_secret="cache-secret",
        cache_path=tmp_path / "cache.jsonl",
        usage_path=tmp_path / "usage.jsonl", session=session,
    )
    first = run(
        ANALYSIS, PROFILE,
        candidate_selector=AzureCatalogSelector(
            model_identity="model-a:2026-01", reasoning_effort="minimal", **common,
        ),
    )
    second = run(
        ANALYSIS, PROFILE,
        candidate_selector=AzureCatalogSelector(
            model_identity="model-b:2026-07", reasoning_effort="minimal", **common,
        ),
    )
    third = run(
        ANALYSIS, PROFILE,
        candidate_selector=AzureCatalogSelector(
            model_identity="model-b:2026-07", reasoning_effort="medium", **common,
        ),
    )

    assert len({
        first["selection"]["cache_key"],
        second["selection"]["cache_key"],
        third["selection"]["cache_key"],
    }) == 3
    assert session.calls == 3


def test_cache_key_changes_with_authoritative_care_policy_identity(tmp_path):
    baseline = run(ANALYSIS, PROFILE)
    selected = {
        slot: baseline["selected_products"].get(slot)
        for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
    }
    observed = {}

    def capture(context, candidates, versions):
        observed.update(context=context, candidates=candidates, versions=versions)
        return CandidateSelection(selected, {"source": "capture"})

    assert run(ANALYSIS, PROFILE, candidate_selector=capture)["status"] == "ok"
    selector = AzureCatalogSelector(
        endpoint="https://azure.example/responses", api_key="secret",
        deployment="gpt-5-test", model_identity="gpt-5-test:fixture",
        cache_secret="cache-secret",
        cache_path=tmp_path / "cache.jsonl", usage_path=tmp_path / "usage.jsonl",
        session=object(),
    )
    original = selector._cache_key(
        observed["context"], observed["candidates"], observed["versions"]
    )
    changed_versions = json.loads(json.dumps(observed["versions"]))
    changed_versions["policy"]["identity"] = "replacement-policy:2"

    assert selector._cache_key(
        observed["context"], observed["candidates"], changed_versions
    ) != original


def test_invalid_azure_product_id_is_not_cached_or_replaced(tmp_path):
    invalid = {
        "cleanser": "invented-product", "treatment": None, "serum": None,
        "moisturizer": "invented-product", "spf": "invented-product",
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": json.dumps(invalid), "usage": {}}

    class Session:
        @staticmethod
        def post(*_args, **_kwargs):
            return Response()

    cache = tmp_path / "cache.jsonl"
    selector = AzureCatalogSelector(
        endpoint="https://azure.example/responses", api_key="secret",
        deployment="gpt-5-test", model_identity="gpt-5-test:fixture",
        cache_secret="cache-secret",
        cache_path=cache, usage_path=tmp_path / "usage.jsonl", session=Session(),
    )

    document = run(ANALYSIS, PROFILE, candidate_selector=selector)

    assert document["status"] == "unavailable"
    assert document["routines"] == []
    assert document["reason"].startswith(
        "llm_selection_unavailable:product_not_safe_for_slot"
    )
    assert not cache.exists()


@pytest.mark.parametrize("failure,reason", [
    ("malformed", "malformed_response"),
    ("timeout", "azure_error:TimeoutError"),
    ("incomplete", "azure_response_incomplete"),
])
def test_azure_failure_returns_no_result(tmp_path, failure, reason):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            if failure == "incomplete":
                baseline = run(ANALYSIS, PROFILE)
                selected = {
                    slot: baseline["selected_products"].get(slot)
                    for slot in (
                        "cleanser", "treatment", "serum", "moisturizer", "spf",
                    )
                }
                return {
                    "status": "incomplete",
                    "output_text": json.dumps(selected),
                    "usage": {},
                }
            return {"output_text": "not-json", "usage": {}}

    class Session:
        @staticmethod
        def post(*_args, **_kwargs):
            if failure == "timeout":
                raise TimeoutError("test timeout")
            return Response()

    selector = AzureCatalogSelector(
        endpoint="https://azure.example/responses", api_key="secret",
        deployment="gpt-5-test", model_identity="gpt-5-test:fixture",
        cache_secret="cache-secret",
        cache_path=tmp_path / "cache.jsonl",
        usage_path=tmp_path / "usage.jsonl", session=Session(),
    )

    document = run(ANALYSIS, PROFILE, candidate_selector=selector)

    assert document["status"] == "unavailable"
    assert document["routines"] == []
    assert document["reason"] == f"llm_selection_unavailable:{reason}"


def test_optional_cli_writes_unavailable_artifact_when_configuration_is_missing(
    tmp_path, monkeypatch,
):
    for name in (
        "TARGET_URL", "AZURE_OPENAI_ENDPOINT", "AZURE_KEY",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_MODEL_IDENTITY",
        "SKINSCAN_CACHE_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "llm-recommendations.json"

    status = main([
        "--analysis", str(ANALYSIS), "--profile", str(PROFILE),
        "--out", str(output), "--cache", str(tmp_path / "cache.jsonl"),
        "--usage-log", str(tmp_path / "usage.jsonl"),
    ])

    document = json.loads(output.read_text())
    assert status == 4
    assert document["status"] == "unavailable"
    assert document["reason"] == "llm_selection_unavailable:configuration_missing"
    assert document["routines"] == []


def test_optional_cli_rejects_legacy_analysis_before_azure(tmp_path, monkeypatch):
    import requests

    for name, value in {
        "TARGET_URL": "https://azure.example/responses",
        "AZURE_KEY": "secret",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-5-test",
        "AZURE_OPENAI_MODEL_IDENTITY": "gpt-5-test:fixture",
        "SKINSCAN_CACHE_SECRET": "cache-secret",
    }.items():
        monkeypatch.setenv(name, value)
    calls = 0

    def unexpected_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("legacy analysis must not trigger Azure")

    monkeypatch.setattr(requests, "post", unexpected_call)
    output = tmp_path / "llm-recommendations.json"

    status = main([
        "--analysis", str(ANALYSIS), "--profile", str(PROFILE),
        "--out", str(output), "--cache", str(tmp_path / "cache.jsonl"),
        "--usage-log", str(tmp_path / "usage.jsonl"),
    ])

    document = json.loads(output.read_text())
    assert status == 4
    assert document["status"] == "unavailable"
    assert document["reason"] == (
        "llm_selection_unavailable:schema4_analysis_required"
    )
    assert document["routines"] == []
    assert calls == 0


def test_selected_product_ingredients_are_joined_from_the_catalog():
    baseline = run(ANALYSIS, PROFILE)
    catalog = json.loads(
        (Path(__file__).parents[1] / "data" / "catalog" / "seed_catalog.json").read_text()
    )
    by_id = {item["product_id"]: item for item in catalog["products"]}

    def select(_context, _candidates, _versions):
        return CandidateSelection(
            {
                slot: baseline["selected_products"].get(slot)
                for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
            },
            {"source": "test_selector"},
        )

    document = run(ANALYSIS, PROFILE, candidate_selector=select)

    for step in (
        document["selected_regimen"]["am"]
        + document["selected_regimen"]["pm"]
        + document["selected_regimen"]["per_label"]
    ):
        product = by_id[step["product_id"]]
        assert step["actives"] == product["actives"]
        assert step["ingredients"] == product["inci"]


def test_renderer_shows_catalog_ingredients_and_selector_provenance():
    baseline = run(ANALYSIS, PROFILE)

    def select(_context, _candidates, _versions):
        return CandidateSelection(
            {
                slot: baseline["selected_products"].get(slot)
                for slot in ("cleanser", "treatment", "serum", "moisturizer", "spf")
            },
            {
                "source": "azure_catalog_selector", "provider": "azure",
                "model": "gpt-5-test", "prompt_version": "test-v1",
                "cache_hit": False, "candidate_count": 34,
                "usage": {
                    "input_tokens": 321, "output_tokens": 17,
                    "estimated_cost_usd": 0.000778,
                },
            },
        )

    html = render(run(ANALYSIS, PROFILE, candidate_selector=select))

    assert "Active ingredients" in html
    assert "Full ingredient list" in html
    assert "azure_catalog_selector" in html
    assert "gpt-5-test" in html
    assert "cache miss" in html
    assert "321 input tokens" in html
    assert "$0.000778" in html
