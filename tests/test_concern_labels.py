"""Tests for the concern-efficacy labeling pipeline (plan 015, D-023).

Pure-Python: the LLM sits behind a duck-typed labeler seam and is stubbed;
no network or provider credentials needed for this suite.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.recommendation.concern_labels import (
    AzureResponsesLabeler,
    CONCERNS,
    OpenRouterLabeler,
    cmd_label,
    compile_prefilter,
    estimate_cost,
    enforce_literal_policy,
    load_cache,
    load_review_rows,
    review_uid,
    run_labeling,
    _labeler,
)


def _patterns():
    return compile_prefilter(load_config()["concern"]["prefilter"])


def test_prefilter_flags_concerns():
    p = _patterns()
    assert p["acne_comedonal"].search("this cleared my blackheads fast")
    assert p["acne_cystic"].search("my hormonal acne is gone")
    assert p["acne_general"].search("it broke me out badly")
    assert p["hyperpigmentation"].search("faded my dark spots in weeks")
    assert p["dryness"].search("no more flaky patches")
    assert not p["acne_comedonal"].search("a blackheadless routine")  # word boundary
    assert not any(rx.search("lovely texture and smell") for rx in p.values())


def test_review_uid_stable_and_distinct():
    a = review_uid("123", "P1", "great product " * 50)
    b = review_uid("123", "P1", "great product " * 50)
    assert a == b and len(a) == 32
    assert review_uid("124", "P1", "great product") != a


def test_load_review_rows_prefilters_joins_and_truncates():
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "reviews_test.csv"
        csv.write_text(
            "author_id,rating,is_recommended,skin_tone,skin_type,"
            "product_id,review_text,review_title\n"
            'a1,5,1.0,fair,oily,PA,"cleared my blackheads ' + "x" * 100 + '",great\n'
            'a2,4,1.0,fair,dry,PA,"smells lovely",nice\n'          # no concern
            'a3,2,0.0,fair,dry,PX,"broke me out",bad\n'            # not in catalog
        )
        rows = load_review_rows(td, {"PA"}, _patterns(), truncate_chars=40)
        assert len(rows) == 1
        assert rows[0]["product_id"] == "PA" and rows[0]["skin_type"] == "oily"
        assert len(rows[0]["text"]) == 40


LABEL_OK = json.dumps({"labels": [{"concern": "acne_comedonal",
                                   "outcome": "helped",
                                   "reviewer_has_condition": True}]})


def _row(author, pid, text, skin_type="oily"):
    return {"uid": review_uid(author, pid, text), "author_id": author,
            "product_id": pid, "skin_type": skin_type, "skin_tone": "fair",
            "rating": 5.0, "is_recommended": 1.0, "text": text}


class StubLabeler:
    """Duck-typed batch labeler (same 3-method seam as AnthropicBatchLabeler).

    replies: uid -> json text | None (None simulates an API-level error).
    """

    def __init__(self, replies):
        self.replies = replies
        self.submitted = []      # chunks passed to submit() this run
        self._batches = {}

    def submit(self, rows):
        self.submitted.append(list(rows))
        bid = f"batch_{len(self._batches)}"
        self._batches[bid] = list(rows)
        return bid

    def status(self, batch_id):
        return "ended"

    def fetch(self, batch_id):
        out = []
        for row in self._batches[batch_id]:
            text = self.replies.get(row["uid"], '{"labels": []}')
            if text is None:
                out.append((row["uid"], None, "errored"))
            else:
                out.append((row["uid"], text, None))
        return out


def test_run_labeling_writes_cache_and_skips_on_rerun():
    r1 = _row("a1", "PA", "cleared my blackheads")
    r2 = _row("a2", "PB", "it broke me out", skin_type="dry")
    stub = StubLabeler({
        r1["uid"]: LABEL_OK,
        r2["uid"]: json.dumps({"labels": [{"concern": "acne_general",
                                           "outcome": "worsened",
                                           "reviewer_has_condition": False}]}),
    })
    with tempfile.TemporaryDirectory() as td:
        cache, state = Path(td) / "labels.jsonl", Path(td) / "state.json"
        s = run_labeling([r1, r2], stub, cache, state, chunk_size=10,
                         poll_seconds=0)
        assert s["ok"] == 2 and s["failed"] == 0 and s["submitted"] == 2
        recs = load_cache(cache)
        assert recs[r1["uid"]]["labels"][0]["concern"] == "acne_comedonal"
        assert recs[r2["uid"]]["product_id"] == "PB"
        assert recs[r2["uid"]]["skin_type"] == "dry"
        assert recs[r2["uid"]]["rating"] == 5.0
        s2 = run_labeling([r1, r2], stub, cache, state, chunk_size=10,
                          poll_seconds=0)
        assert s2["submitted"] == 0 and s2["cached_before"] == 2


def test_malformed_reply_cached_as_parse_error_not_rebilled():
    r1 = _row("a1", "PA", "cleared my blackheads")
    stub = StubLabeler({r1["uid"]: "not json {{"})
    with tempfile.TemporaryDirectory() as td:
        cache, state = Path(td) / "labels.jsonl", Path(td) / "state.json"
        s = run_labeling([r1], stub, cache, state, chunk_size=10, poll_seconds=0)
        assert s["parse_error"] == 1
        rec = load_cache(cache)[r1["uid"]]
        assert rec["status"] == "parse_error" and rec["labels"] == []
        s2 = run_labeling([r1], stub, cache, state, chunk_size=10, poll_seconds=0)
        assert s2["submitted"] == 0   # billed once, never again


def test_api_error_rows_not_cached_and_retried_next_run():
    r1 = _row("a1", "PA", "cleared my blackheads")
    stub = StubLabeler({r1["uid"]: None})
    with tempfile.TemporaryDirectory() as td:
        cache, state = Path(td) / "labels.jsonl", Path(td) / "state.json"
        s = run_labeling([r1], stub, cache, state, chunk_size=10, poll_seconds=0)
        assert s["failed"] == 1
        assert r1["uid"] not in load_cache(cache)
        stub.replies[r1["uid"]] = LABEL_OK        # API recovered
        s2 = run_labeling([r1], stub, cache, state, chunk_size=10, poll_seconds=0)
        assert s2["ok"] == 1


def test_invalid_label_entries_filtered_but_row_ok():
    r1 = _row("a1", "PA", "cleared my blackheads")
    reply = json.dumps({"labels": [
        {"concern": "acne_comedonal", "outcome": "helped",
         "reviewer_has_condition": True},
        {"concern": "wrinkles", "outcome": "helped",
         "reviewer_has_condition": True},          # not in vocab -> dropped
    ]})
    stub = StubLabeler({r1["uid"]: reply})
    with tempfile.TemporaryDirectory() as td:
        cache, state = Path(td) / "labels.jsonl", Path(td) / "state.json"
        run_labeling([r1], stub, cache, state, chunk_size=10, poll_seconds=0)
        rec = load_cache(cache)[r1["uid"]]
        assert rec["status"] == "ok" and len(rec["labels"]) == 1


def _label(concern, outcome, condition):
    return {"concern": concern, "outcome": outcome,
            "reviewer_has_condition": condition}


def test_literal_policy_adds_missed_explicit_subtypes():
    labels = [_label("acne_general", "helped", True)]
    text = ("My pimples and breakouts got much smaller after this serum. "
            "It also decreased the blackheads on my nose.")
    actual = enforce_literal_policy(text, labels)
    assert {tuple(item.values()) for item in actual} == {
        ("acne_comedonal", "helped", True),
        ("acne_general", "helped", True),
        ("acne_inflammatory", "helped", True),
    }


def test_literal_policy_keeps_prevention_and_generic_claims_nonpersonal():
    preventive = enforce_literal_policy(
        "This did not give me breakouts.",
        [_label("acne_general", "helped", True)],
    )
    generic = enforce_literal_policy(
        "This product is amazing for cystic acne.",
        [_label("acne_cystic", "unclear", True)],
    )
    assert preventive == [_label("acne_general", "helped", False)]
    assert generic == [_label("acne_cystic", "helped", False)]


def test_literal_policy_preserves_context_as_unclear():
    actual = enforce_literal_policy(
        "I use harsh acne treatments and this moisturizer keeps my dry skin hydrated.",
        [_label("acne_general", "helped", True), _label("dryness", "helped", True)],
    )
    assert actual == [
        _label("acne_general", "unclear", True),
        _label("dryness", "helped", True),
    ]


def test_literal_policy_does_not_spread_clogged_pore_worsening_to_acne():
    actual = enforce_literal_policy(
        "My pores clogged up. I had my breakouts in check before this.",
        [_label("acne_comedonal", "worsened", True),
         _label("acne_general", "worsened", True)],
    )
    assert actual == [
        _label("acne_comedonal", "worsened", True),
        _label("acne_general", "unclear", True),
    ]


def test_literal_policy_adds_context_only_dark_spots_and_dryness():
    actual = enforce_literal_policy(
        "I am concerned with my dark spots. A few hours later my skin feels so dry.",
        [],
    )
    assert actual == [
        _label("hyperpigmentation", "unclear", True),
        _label("dryness", "unclear", True),
    ]


def test_literal_policy_marks_multi_product_result_unclear():
    actual = enforce_literal_policy(
        "I purchased this, a serum, and an oil. My dry flaky skin is now soft.",
        [_label("dryness", "helped", True)],
    )
    assert actual == [_label("dryness", "unclear", True)]


def test_literal_policy_recognizes_explicit_dryness_benefit():
    actual = enforce_literal_policy(
        "My cheeks get really dry but this moisturizer works great.",
        [_label("dryness", "unclear", True)],
    )
    assert actual == [_label("dryness", "helped", True)]


def test_literal_policy_keeps_negated_subtype_separate_from_personal_breakouts():
    actual = enforce_literal_policy(
        "I don't really get blackheads, but I do break out on my chin. This saved me.",
        [_label("acne_general", "helped", True)],
    )
    assert actual == [
        _label("acne_comedonal", "unclear", False),
        _label("acne_general", "helped", True),
    ]


def test_literal_policy_prioritizes_explicit_causation_over_unrelated_negation():
    actual = enforce_literal_policy(
        "I didn't want to believe it, but this caused itchy breakouts.",
        [_label("acne_general", "helped", False)],
    )
    assert actual == [_label("acne_general", "worsened", True)]


def test_literal_policy_does_not_spread_prevention_to_existing_subtypes():
    actual = enforce_literal_policy(
        "I have had acne for years and have a few pimples. This did not cause additional breakouts. If a product is too drying I break out.",
        [_label("acne_general", "helped", True),
         _label("acne_inflammatory", "helped", False),
         _label("dryness", "unclear", True)],
    )
    assert actual == [
        _label("acne_inflammatory", "unclear", True),
        _label("acne_general", "helped", True),
        _label("dryness", "unclear", False),
    ]


def test_literal_policy_does_not_infer_acne_from_acne_scarring():
    actual = enforce_literal_policy(
        "My dark spots come from acne scarring and this faded them.",
        [_label("acne_general", "helped", True),
         _label("hyperpigmentation", "helped", True)],
    )
    assert actual == [_label("hyperpigmentation", "helped", True)]


def test_literal_policy_handles_unchanged_and_hypothetical_outcomes():
    unchanged = enforce_literal_policy(
        "My acne is the same as it has always been.",
        [_label("acne_general", "helped", True)],
    )
    future = enforce_literal_policy(
        "I plan to have my daughter try it to see if it helps her acne.",
        [_label("acne_general", "helped", False)],
    )
    assert unchanged == [_label("acne_general", "unclear", True)]
    assert future == [_label("acne_general", "unclear", False)]


def test_literal_policy_distinguishes_product_finish_from_skin_dryness():
    actual = enforce_literal_policy(
        "It stops blemishes in their tracks and has a smooth finish once dry.",
        [_label("acne_general", "worsened", True),
         _label("dryness", "unclear", False)],
    )
    assert actual == [_label("acne_general", "helped", False)]


def test_literal_policy_recognizes_product_associated_dryness():
    actual = enforce_literal_policy(
        "This oil falls short; within a couple hours my skin feels so dry.",
        [_label("dryness", "unclear", True)],
    )
    assert actual == [_label("dryness", "worsened", True)]


def test_chunking_respects_chunk_size():
    rows = [_row(f"a{i}", "PA", f"cleared my blackheads {i}") for i in range(5)]
    stub = StubLabeler({r["uid"]: LABEL_OK for r in rows})
    with tempfile.TemporaryDirectory() as td:
        cache, state = Path(td) / "labels.jsonl", Path(td) / "state.json"
        run_labeling(rows, stub, cache, state, chunk_size=2, poll_seconds=0)
        assert [len(c) for c in stub.submitted] == [2, 2, 1]


def test_resume_drains_submitted_batch_without_resubmitting():
    r1 = _row("a1", "PA", "cleared my blackheads")
    stub = StubLabeler({r1["uid"]: LABEL_OK})
    with tempfile.TemporaryDirectory() as td:
        cache, state = Path(td) / "labels.jsonl", Path(td) / "state.json"
        bid = stub.submit([r1])            # a prior run crashed after submit
        stub.submitted.clear()
        state.write_text(json.dumps({"batches": {bid: {"fetched": False}}}))
        s = run_labeling([r1], stub, cache, state, chunk_size=10, poll_seconds=0)
        assert stub.submitted == []        # nothing resubmitted, nothing re-billed
        assert load_cache(cache)[r1["uid"]]["status"] == "ok"
        assert s["ok"] == 1 and s["submitted"] == 0


def test_openrouter_grouped_results_are_spooled_and_reused():
    rows = [_row(f"a{i}", "PA", f"cleared my blackheads {i}") for i in range(2)]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            results = [{"uid": r["uid"], "labels": json.loads(LABEL_OK)["labels"]}
                       for r in rows]
            return {"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps({"results": results})}}]}

    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return Response()

    with tempfile.TemporaryDirectory() as td:
        old = __import__("os").environ.get("OPENROUTER_API_KEY")
        __import__("os").environ["OPENROUTER_API_KEY"] = "test-key"
        try:
            session = Session()
            labeler = OpenRouterLabeler("test/model", td, 10, 1, session)
            bid = labeler.submit(rows)
            assert session.calls == 1 and len(labeler.fetch(bid)) == 2
            assert labeler.submit(rows) == bid and session.calls == 1
        finally:
            if old is None:
                __import__("os").environ.pop("OPENROUTER_API_KEY", None)
            else:
                __import__("os").environ["OPENROUTER_API_KEY"] = old


def test_azure_responses_grouped_results_are_spooled(monkeypatch, tmp_path):
    rows = [_row(f"a{i}", "PA", f"cleared my blackheads {i}") for i in range(2)]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            results = [{"uid": row["uid"], "labels": json.loads(LABEL_OK)["labels"]}
                       for row in rows]
            return {
                "output_text": json.dumps({"results": results}),
                "usage": {"input_tokens": 100, "output_tokens": 40},
            }

    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            self.request = kwargs
            return Response()

    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses?api-version=test")
    session = Session()
    labeler = AzureResponsesLabeler(
        "cheap-deployment", tmp_path, 250, 1, session,
        usage_path=tmp_path / "usage.jsonl",
    )

    bid = labeler.submit(rows)

    assert session.calls == 1
    assert len(labeler.fetch(bid)) == 2
    body = session.request["json"]
    assert body["model"] == "cheap-deployment"
    assert body["text"]["format"]["type"] == "json_schema"
    assert json.loads((tmp_path / "usage.jsonl").read_text())["output_tokens"] == 40


def test_labeler_selects_complete_azure_configuration(monkeypatch, tmp_path):
    cfg = {**load_config()["concern"], "batch_spool_dir": str(tmp_path)}
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "cheap-deployment")

    assert isinstance(_labeler(cfg), AzureResponsesLabeler)


def test_labeler_refuses_partial_azure_configuration(monkeypatch, tmp_path):
    cfg = {**load_config()["concern"], "batch_spool_dir": str(tmp_path)}
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_DEPLOYMENT"):
        _labeler(cfg)


def test_azure_cost_preflight_requires_explicit_prices(monkeypatch):
    cfg = load_config()["concern"]
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "cheap-deployment")
    monkeypatch.delenv("AZURE_INPUT_PRICE_PER_MILLION", raising=False)
    monkeypatch.delenv("AZURE_OUTPUT_PRICE_PER_MILLION", raising=False)

    with pytest.raises(RuntimeError, match="preflight requires"):
        estimate_cost([{"text": "x" * 400}], cfg)


def test_azure_cost_preflight_uses_conservative_output_allowance(monkeypatch):
    cfg = {**load_config()["concern"], "reviews_per_request": 10}
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "cheap-deployment")
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_MILLION", "1")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_MILLION", "2")

    cost = estimate_cost([{"text": "x" * 400}], cfg)

    assert cost == pytest.approx((100 + 450) / 1e6 + 120 / 1e6 * 2)


def test_full_run_is_pinned_to_zero_cost_endpoint():
    cfg = load_config()["concern"]
    rows = [{"text": "x" * 1200}] * 202_000
    assert estimate_cost(rows, cfg) == 0
    assert cfg["max_budget_usd"] <= 20
    assert cfg["labeling_model"].endswith(":free")


def test_full_run_fits_free_daily_request_budget():
    cfg = load_config()["concern"]
    row_count = 202_000
    groups = (row_count + cfg["reviews_per_request"] - 1) // cfg["reviews_per_request"]
    assert groups <= 900  # preserve headroom under OpenRouter's 1,000 free requests/day


def test_full_label_requires_p2_signoff():
    try:
        cmd_label([], load_config()["concern"], yes=True)
    except RuntimeError as exc:
        assert "P2 sign-off" in str(exc)
    else:
        raise AssertionError("full labeling ran without P2 sign-off")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
