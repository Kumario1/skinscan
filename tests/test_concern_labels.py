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
    PROMPT_VERSION,
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
import src.recommendation.concern_labels as concern_labels


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


def test_review_uid_uses_text_after_first_300_chars():
    prefix = "x" * 300
    assert review_uid("123", "P1", prefix + " first") != review_uid(
        "123", "P1", prefix + " second"
    )


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


def test_cache_lookup_is_bound_to_provider_model_and_prompt():
    class BoundStubLabeler(StubLabeler):
        provider = "test-provider"
        model = "model-a"
        prompt_version = "prompt-a"

    row = _row("a1", "PA", "cleared my blackheads")
    with tempfile.TemporaryDirectory() as td:
        cache, state = Path(td) / "labels.jsonl", Path(td) / "state.json"
        first = BoundStubLabeler({row["uid"]: LABEL_OK})
        run_labeling([row], first, cache, state, chunk_size=10, poll_seconds=0)
        record = load_cache(cache, prompt_version="prompt-a",
                            provider="test-provider", model="model-a")[row["uid"]]
        assert record["provider"] == "test-provider"
        assert record["model"] == "model-a"
        assert record["prompt_version"] == "prompt-a"

        changed = BoundStubLabeler({row["uid"]: LABEL_OK})
        changed.model = "model-b"
        summary = run_labeling([row], changed, cache, state,
                               chunk_size=10, poll_seconds=0)
        assert summary["submitted"] == 1

        changed_prompt = BoundStubLabeler({row["uid"]: LABEL_OK})
        changed_prompt.prompt_version = "prompt-b"
        summary = run_labeling([row], changed_prompt, cache, state,
                               chunk_size=10, poll_seconds=0)
        assert summary["submitted"] == 1


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


def test_literal_policy_treats_post_discontinuation_breakout_as_helped():
    actual = enforce_literal_policy(
        "No breakouts whatsoever while I used it. A week after I finished the "
        "bottle my face has broken out badly.",
        [_label("acne_general", "worsened", True)],
    )
    assert actual == [_label("acne_general", "helped", True)]


def test_literal_policy_marks_purging_unclear_unless_worse():
    purging = enforce_literal_policy(
        "A decent amount of pimples popped up after, probably skin purging.",
        [_label("acne_inflammatory", "worsened", True)],
    )
    rejected = enforce_literal_policy(
        "It caused the worst cystic acne of my life. I thought my skin was "
        "purging but that stage never ended.",
        [_label("acne_cystic", "worsened", True)],
    )
    assert purging == [_label("acne_inflammatory", "unclear", True)]
    assert rejected == [_label("acne_cystic", "worsened", True)]


def test_literal_policy_marks_multi_product_set_credit_unclear():
    actual = enforce_literal_policy(
        "I used to flare up with breakouts but these two items are perfect "
        "for me.",
        [_label("acne_general", "helped", True)],
    )
    assert actual == [_label("acne_general", "unclear", True)]


def test_literal_policy_keeps_goal_statements_out_of_helped():
    actual = enforce_literal_policy(
        "I bought this with an emphasis on clearing up the blackheads on my "
        "nose.",
        [_label("acne_comedonal", "helped", True)],
    )
    assert actual == [_label("acne_comedonal", "unclear", True)]


def test_literal_policy_hypothetical_worry_is_not_a_condition():
    actual = enforce_literal_policy(
        "I was concerned this rich cream might break me out.",
        [_label("acne_general", "unclear", True)],
    )
    assert actual == [_label("acne_general", "unclear", False)]


def test_literal_policy_as_sometimes_i_do_admits_condition():
    actual = enforce_literal_policy(
        "It didn't break me out or irritate me (as sometimes I do from new "
        "products).",
        [_label("acne_general", "helped", False)],
    )
    assert actual == [_label("acne_general", "helped", True)]


def test_literal_policy_spreads_caused_breakout_to_enumerated_lesions():
    actual = enforce_literal_policy(
        "This caused crazy break outs. I have red bumps across my forehead as "
        "well as multiple white head zits.",
        [_label("acne_general", "worsened", True),
         _label("acne_inflammatory", "worsened", True)],
    )
    assert actual == [
        _label("acne_comedonal", "worsened", True),
        _label("acne_inflammatory", "worsened", True),
        _label("acne_general", "worsened", True),
    ]


def test_literal_policy_recognizes_bare_scars_pigmentation_and_drier():
    pigment = enforce_literal_policy(
        "My pigmentation from old breakouts did not diminish at all.",
        [_label("acne_general", "unclear", True)],
    )
    assert _label("hyperpigmentation", "unclear", True) in pigment
    scares = enforce_literal_policy(
        "I have to admit my acne scares are fading fast.", [],
    )
    assert scares == [_label("hyperpigmentation", "helped", True)]
    drier = enforce_literal_policy(
        "Now that I have drier skin I was wary. Great results and NO flaking.",
        [_label("dryness", "unclear", False)],
    )
    assert drier == [_label("dryness", "unclear", True)]


def test_literal_policy_bare_modal_does_not_downgrade_comedonal_helped():
    # "may buy again" is a repurchase modal, not a no-effect signal; a genuine
    # comedonal benefit must survive it (previously the comedonal special-case
    # let any _NO_EFFECT word downgrade even a real "helped").
    helped = enforce_literal_policy(
        "Blackheads are much smaller, may buy again.",
        [_label("acne_comedonal", "helped", True)],
    )
    assert helped == [_label("acne_comedonal", "helped", True)]


def test_literal_policy_lesion_spread_ignores_habitual_mentions():
    # An incidental/habitual "blackhead on my nose" must NOT inherit worsening
    # from a co-occurring product-caused breakout.
    incidental = enforce_literal_policy(
        "This caused breakouts! I always get a blackhead on my nose.",
        [_label("acne_general", "worsened", True),
         _label("acne_comedonal", "unclear", True)],
    )
    assert _label("acne_comedonal", "unclear", True) in incidental
    # A lesion enumerated across the active outbreak still inherits worsening.
    enumerated = enforce_literal_policy(
        "Caused crazy break outs. I have red bumps all over my forehead as "
        "well as whiteheads.",
        [_label("acne_general", "worsened", True),
         _label("acne_comedonal", "unclear", True)],
    )
    assert _label("acne_comedonal", "worsened", True) in enumerated


def test_literal_policy_ignores_product_texture_drying():
    assert enforce_literal_policy(
        "This quick drying formula is lightweight and never greasy.", [],
    ) == []
    # a real skin-dryness mention is still captured
    assert enforce_literal_policy(
        "This quick-drying formula still left my skin dry.", [],
    ) == [_label("dryness", "worsened", True)]


def test_literal_policy_discontinuation_does_not_mask_explicit_product_cause():
    # The "since I finished my antibiotics" discontinuation clause must not
    # suppress worsening the reviewer explicitly attributes to this product.
    actual = enforce_literal_policy(
        "Since I finished my antibiotics my acne flared, but this cream also "
        "clogged my pores.",
        [_label("acne_comedonal", "worsened", True)],
    )
    by = {l["concern"]: l["outcome"] for l in actual}
    assert by.get("acne_comedonal") == "worsened"


def test_literal_policy_does_not_propagate_acne_outcome_to_pimples():
    actual = enforce_literal_policy(
        "This cleared my acne. I still have pimples.",
        [_label("acne_inflammatory", "helped", True),
         _label("acne_general", "helped", True)],
    )
    assert actual == [
        _label("acne_inflammatory", "unclear", True),
        _label("acne_general", "helped", True),
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
            results = [{"i": i, "c": ["001"]} for i, _row_value in enumerate(rows)]
            return {
                "output_text": json.dumps({"r": results}),
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
    fetched = labeler.fetch(bid)
    assert len(fetched) == 2
    assert all(error is None for _uid, _payload, error in fetched)
    assert [json.loads(payload)["labels"][0] for _uid, payload, _error in fetched] == [
        {
            "concern": "acne_comedonal",
            "outcome": "helped",
            "reviewer_has_condition": True,
        },
    ] * 2
    body = session.request["json"]
    assert body["model"] == "cheap-deployment"
    assert body["text"]["format"]["type"] == "json_schema"
    assert "r" in body["text"]["format"]["schema"]["properties"]
    result_schema = body["text"]["format"]["schema"]["properties"]["r"]
    assert result_schema["minItems"] == result_schema["maxItems"] == 2
    item_props = result_schema["items"]["properties"]
    assert item_props["i"]["enum"] == [0, 1]
    assert "001" in item_props["c"]["items"]["enum"]
    assert [json.loads(line) for line in body["input"].splitlines()] == [
        {"i": 0, "text": rows[0]["text"]},
        {"i": 1, "text": rows[1]["text"]},
    ]
    assert "0=acne_comedonal" in body["instructions"]
    assert "0=helped" in body["instructions"]
    assert "three-digit COH" in body["instructions"]
    assert json.loads((tmp_path / "usage.jsonl").read_text())["output_tokens"] == 40


def test_azure_decodes_by_index_and_survives_reordering(monkeypatch, tmp_path):
    rows = [_row("a0", "PA", "cleared my blackheads"),
            _row("a1", "PA", "this broke me out")]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            # returned out of input order: index 1 first, then index 0
            return {"output_text": json.dumps({"r": [
                {"i": 1, "c": ["301"]},   # acne_general worsened
                {"i": 0, "c": ["001"]},   # acne_comedonal helped
            ]}), "usage": {"input_tokens": 10, "output_tokens": 8}}

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setenv("AZURE_KEY", "k")
    monkeypatch.setenv("TARGET_URL", "https://x.openai.azure.com/openai/responses")
    labeler = AzureResponsesLabeler("gpt-5-mini", tmp_path / "s", 250, 1, Session(),
                                    usage_path=tmp_path / "u.jsonl")
    bid = labeler.submit(rows)
    by_uid = {uid: json.loads(payload)["labels"][0]["concern"]
              for uid, payload, err in labeler.fetch(bid) if err is None}
    # each uid gets ITS OWN label despite the model's reordering
    assert by_uid[rows[0]["uid"]] == "acne_comedonal"
    assert by_uid[rows[1]["uid"]] == "acne_general"


def test_azure_rejects_mismatched_index_set(monkeypatch, tmp_path):
    rows = [_row("a0", "PA", "cleared my blackheads"),
            _row("a1", "PA", "this broke me out")]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            # duplicate index 0, missing index 1
            return {"output_text": json.dumps({"r": [
                {"i": 0, "c": ["001"]}, {"i": 0, "c": ["301"]},
            ]}), "usage": {"input_tokens": 10, "output_tokens": 8}}

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setenv("AZURE_KEY", "k")
    monkeypatch.setenv("TARGET_URL", "https://x.openai.azure.com/openai/responses")
    labeler = AzureResponsesLabeler("gpt-5-mini", tmp_path / "s", 250, 1, Session(),
                                    usage_path=tmp_path / "u.jsonl")
    bid = labeler.submit(rows)
    # a corrupt index set must NOT be cached as labels — every row errors
    assert all(err is not None for _uid, _payload, err in labeler.fetch(bid))


def test_azure_timeout_records_ceiling_not_zero(monkeypatch, tmp_path):
    row = _row("a0", "PA", "cleared my blackheads")

    class Boom:
        def post(self, *args, **kwargs):
            raise RuntimeError("timed out")   # no response body, no usage

    monkeypatch.setenv("AZURE_KEY", "k")
    monkeypatch.setenv("TARGET_URL", "https://x.openai.azure.com/openai/responses")
    labeler = AzureResponsesLabeler(
        "gpt-5-mini", tmp_path / "s", 250, 1, Boom(),
        usage_path=tmp_path / "u.jsonl", max_budget_usd=90.0,
        input_price_per_million=0.25, output_price_per_million=2.0,
        max_requests=1400)
    labeler.submit([row])
    rec = json.loads((tmp_path / "u.jsonl").read_text())
    assert rec["status"] == "failed"
    # a usage-less failure is charged the conservative ceiling, never $0
    assert rec["output_tokens"] > 0 and rec["input_tokens"] > 0
    # and the in-flight reservation was released (no leak)
    assert labeler._reservations == {} and labeler._ceilings == {}


def test_azure_http_error_is_not_charged_the_ceiling(monkeypatch, tmp_path):
    row = _row("a0", "PA", "cleared my blackheads")

    class RateLimited:
        def json(self):
            return {"error": {"code": "429"}}   # error body, no usage block

        def raise_for_status(self):
            raise RuntimeError("429 Too Many Requests")

    class Session:
        def post(self, *args, **kwargs):
            return RateLimited()   # a response DID come back (just an error)

    monkeypatch.setenv("AZURE_KEY", "k")
    monkeypatch.setenv("TARGET_URL", "https://x.openai.azure.com/openai/responses")
    labeler = AzureResponsesLabeler(
        "gpt-5-mini", tmp_path / "s", 250, 1, Session(),
        usage_path=tmp_path / "u.jsonl", max_budget_usd=90.0,
        input_price_per_million=0.25, output_price_per_million=2.0, max_requests=1400)
    labeler.submit([row])
    rec = json.loads((tmp_path / "u.jsonl").read_text())
    assert rec["status"] == "failed"
    # a 429 was rejected unbilled by Azure -> $0, not the ceiling
    assert rec["input_tokens"] == 0 and rec["output_tokens"] == 0
    assert labeler._reservations == {} and labeler._ceilings == {}


def test_azure_reasoning_effort_configurable_default_medium(monkeypatch, tmp_path):
    row = _row("a0", "PA", "cleared my blackheads")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"output_text": json.dumps({"r": [{"i": 0, "c": ["001"]}]}),
                    "usage": {"input_tokens": 10, "output_tokens": 4}}

    class Session:
        def post(self, *args, **kwargs):
            self.request = kwargs
            return Response()

    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.delenv("AZURE_REASONING_EFFORT", raising=False)

    session = Session()
    labeler = AzureResponsesLabeler("gpt-5-mini", tmp_path / "s1", 250, 1, session,
                                    usage_path=tmp_path / "u1.jsonl")
    labeler.submit([row])
    assert session.request["json"]["reasoning"] == {"effort": "medium"}
    assert session.request["json"]["max_output_tokens"] == 120 + 16_000

    monkeypatch.setenv("AZURE_REASONING_EFFORT", "minimal")
    session = Session()
    labeler = AzureResponsesLabeler("gpt-5-mini", tmp_path / "s0", 250, 1, session,
                                    usage_path=tmp_path / "u0.jsonl")
    labeler.submit([row])
    assert session.request["json"]["max_output_tokens"] == 120

    monkeypatch.setenv("AZURE_REASONING_EFFORT", "high")
    session = Session()
    labeler = AzureResponsesLabeler("gpt-5-mini", tmp_path / "s2", 250, 1, session,
                                    usage_path=tmp_path / "u2.jsonl")
    labeler.submit([row])
    assert session.request["json"]["reasoning"] == {"effort": "high"}

    monkeypatch.delenv("AZURE_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_MILLION", "0.25")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_MILLION", "2")
    cfg = {**load_config()["concern"], "batch_spool_dir": str(tmp_path / "s3")}
    assert _labeler(cfg).reasoning_effort == cfg["azure_reasoning_effort"] == "medium"


def test_azure_timeout_configurable_default_600(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.delenv("AZURE_TIMEOUT_SECONDS", raising=False)
    assert AzureResponsesLabeler("gpt-5-mini", tmp_path / "d").timeout == 600
    assert AzureResponsesLabeler("gpt-5-mini", tmp_path / "e", timeout=900).timeout == 900
    monkeypatch.setenv("AZURE_TIMEOUT_SECONDS", "300")
    assert AzureResponsesLabeler("gpt-5-mini", tmp_path / "f").timeout == 300
    monkeypatch.delenv("AZURE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_MILLION", "0.25")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_MILLION", "2")
    cfg = {**load_config()["concern"], "batch_spool_dir": str(tmp_path / "g")}
    assert _labeler(cfg).timeout == cfg["azure_timeout_seconds"] == 600


def test_azure_records_failed_request_usage_with_identity(monkeypatch, tmp_path):
    row = _row("a0", "PA", "cleared my blackheads")

    class Response:
        def json(self):
            return {"id": "azure-request-1", "usage": {
                "input_tokens": 100, "output_tokens": 40,
            }}

        def raise_for_status(self):
            raise RuntimeError("provider rejected request")

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    labeler = AzureResponsesLabeler(
        "deployment-a", tmp_path / "spool", 250, 1, Session(),
        usage_path=tmp_path / "usage.jsonl",
    )

    labeler.submit([row])

    record = json.loads((tmp_path / "usage.jsonl").read_text())
    assert record["provider"] == "azure"
    assert record["model"] == "deployment-a"
    assert record["prompt_version"] == PROMPT_VERSION
    assert record["request_id"] == "azure-request-1"
    assert record["status"] == "failed"
    assert record["input_tokens"] == 100 and record["output_tokens"] == 40


def test_labeler_selects_complete_azure_configuration(monkeypatch, tmp_path):
    cfg = {**load_config()["concern"], "batch_spool_dir": str(tmp_path)}
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "cheap-deployment")
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_MILLION", "0.25")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_MILLION", "2")

    labeler = _labeler(cfg)
    assert isinstance(labeler, AzureResponsesLabeler)
    assert labeler.max_budget_usd == cfg["max_budget_usd"]


def test_azure_labeler_reserves_budget_before_http(monkeypatch, tmp_path):
    row = _row("a0", "PA", "cleared my blackheads")

    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("HTTP must not run after reservation refusal")

    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    session = Session()
    labeler = AzureResponsesLabeler(
        "deployment-a", tmp_path / "spool", 250, 1, session,
        usage_path=tmp_path / "usage.jsonl",
        max_budget_usd=0.000001,
        input_price_per_million=1,
        output_price_per_million=1,
        max_requests=10,
    )

    batch_id = labeler.submit([row])

    assert session.calls == 0
    assert labeler.fetch(batch_id)[0][2] == "budget_ceiling"
    assert not (tmp_path / "usage.jsonl").exists()


def test_azure_labeler_reserves_request_count_before_http(monkeypatch, tmp_path):
    row = _row("a0", "PA", "cleared my blackheads")

    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("HTTP must not run after request-limit refusal")

    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    session = Session()
    labeler = AzureResponsesLabeler(
        "deployment-a", tmp_path / "spool", 250, 1, session,
        usage_path=tmp_path / "usage.jsonl",
        max_budget_usd=10,
        input_price_per_million=1,
        output_price_per_million=1,
        max_requests=0,
    )

    batch_id = labeler.submit([row])

    assert session.calls == 0
    assert labeler.fetch(batch_id)[0][2] == "request_ceiling"


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


def test_azure_cost_preflight_uses_calibration_usage_with_margin(monkeypatch, tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text(json.dumps({
        "model": "cheap-deployment",
        "prompt_version": PROMPT_VERSION,
        "rows": 100,
        "input_tokens": 20_000,
        "output_tokens": 2_000,
    }) + "\n")
    cfg = {
        **load_config()["concern"],
        "reviews_per_request": 10,
        "azure_usage_path": str(usage),
    }
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "cheap-deployment")
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_MILLION", "1")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_MILLION", "2")

    cost = estimate_cost([{"text": "x" * 400}], cfg)

    assert cost == pytest.approx((100 + 450) / 1e6 + 25 / 1e6 * 2)


def _azure_gate_cfg(tmp_path, usage_path, **overrides):
    cfg = {**load_config()["concern"],
           "labels_path": str(tmp_path / "labels.jsonl"),
           "batch_state_path": str(tmp_path / "batches.json"),
           "azure_usage_path": str(usage_path),
           "max_budget_usd": 100.0}
    cfg.update(overrides)
    (tmp_path / "calibration_report.json").write_text(json.dumps({
        "yield": 0.40,
        "gate_p2_yield": "PASS",
        "measured_agreement": 0.90,
        "audited_rows": 50,
        "prompt_version": PROMPT_VERSION,
    }))
    return cfg


def test_azure_preflight_debits_matching_cumulative_usage(monkeypatch, tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text("\n".join(json.dumps(record) for record in [
        {"provider": "azure", "model": "deployment-a",
         "prompt_version": "p7", "request_id": "1",
         "status": "failed", "rows": 1, "input_tokens": 1000,
         "output_tokens": 100},
        {"provider": "azure", "model": "other-deployment",
         "prompt_version": PROMPT_VERSION, "request_id": "2",
         "status": "succeeded", "rows": 1, "input_tokens": 9000,
         "output_tokens": 9000},
    ]) + "\n")
    cfg = _azure_gate_cfg(tmp_path, usage, max_budget_usd=0.001)
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment-a")
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_MILLION", "1")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_MILLION", "1")

    with pytest.raises(RuntimeError, match="cumulative"):
        cmd_label([], cfg, yes=True, p2_approved=False)


def test_azure_preflight_enforces_cumulative_request_limit(monkeypatch, tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text("\n".join(json.dumps(record) for record in [
        {"provider": "azure", "model": "deployment-a",
         "prompt_version": PROMPT_VERSION, "request_id": "1",
         "status": "failed", "rows": 1},
        {"provider": "azure", "model": "deployment-a",
         "prompt_version": PROMPT_VERSION, "request_id": "2",
         "status": "succeeded", "rows": 1},
    ]) + "\n")
    cfg = _azure_gate_cfg(tmp_path, usage, azure_max_requests=1)
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment-a")
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_MILLION", "1")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_MILLION", "1")

    with pytest.raises(RuntimeError, match="request"):
        cmd_label([], cfg, yes=True, p2_approved=False)


def test_azure_label_summary_reports_cumulative_budget_and_requests(monkeypatch, tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text(json.dumps({
        "provider": "azure", "model": "deployment-a",
        "prompt_version": PROMPT_VERSION, "request_id": "1",
        "status": "succeeded", "rows": 1,
        "input_tokens": 100, "output_tokens": 40,
    }) + "\n")
    cfg = _azure_gate_cfg(tmp_path, usage, azure_max_requests=10)
    monkeypatch.setenv("AZURE_KEY", "test-key")
    monkeypatch.setenv("TARGET_URL", "https://example.openai.azure.com/openai/responses")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment-a")
    monkeypatch.setenv("AZURE_INPUT_PRICE_PER_MILLION", "1")
    monkeypatch.setenv("AZURE_OUTPUT_PRICE_PER_MILLION", "1")
    monkeypatch.setattr(concern_labels, "_labeler", lambda _cfg: StubLabeler({}))

    summary = cmd_label([], cfg, yes=True, p2_approved=False)

    assert summary["azure_request_count"] == 1
    assert summary["azure_max_request_count"] == 10
    assert summary["azure_historical_cost_usd"] == pytest.approx(0.00014)


def test_free_fallback_endpoint_stays_zero_cost():
    # Without Azure env configured, the labeler falls back to the free model
    # and estimate_cost must be exactly zero.
    cfg = load_config()["concern"]
    rows = [{"text": "x" * 1200}] * 202_000
    assert estimate_cost(rows, cfg) == 0
    assert cfg["labeling_model"].endswith(":free")


def test_budget_ceiling_stays_within_azure_credit():
    cfg = load_config()["concern"]
    assert cfg["max_budget_usd"] <= 100   # hard stop under the ~$100 Azure credit


def test_full_run_fits_azure_request_ceiling():
    cfg = load_config()["concern"]
    row_count = 202_000
    groups = (row_count + cfg["reviews_per_request"] - 1) // cfg["reviews_per_request"]
    assert groups <= cfg["azure_max_requests"]  # full corpus fits the request ceiling


def test_full_label_requires_p2_signoff(tmp_path):
    # Isolate the report path so a real certified report on disk can't satisfy
    # the gate — this asserts the no-report case genuinely refuses.
    cfg = {**load_config()["concern"],
           "batch_state_path": str(tmp_path / "batches.json"),
           "labels_path": str(tmp_path / "labels.jsonl")}
    try:
        cmd_label([], cfg, yes=True)
    except RuntimeError as exc:
        assert "sign-off" in str(exc)
    else:
        raise AssertionError("full labeling ran without P2 sign-off")


def test_full_label_rejects_stale_prompt_version_signoff(monkeypatch, tmp_path):
    # A passing report from an earlier prompt version must not certify a run of
    # the current policy version.
    cfg = {**load_config()["concern"],
           "labels_path": str(tmp_path / "labels.jsonl"),
           "batch_state_path": str(tmp_path / "batches.json")}
    (tmp_path / "calibration_report.json").write_text(json.dumps({
        "yield": 0.40, "gate_p2_yield": "PASS",
        "measured_agreement": 0.94, "audited_rows": 50,
        "prompt_version": "p0-stale",
    }))
    monkeypatch.setattr(concern_labels, "_labeler", lambda _cfg: StubLabeler({}))
    with pytest.raises(RuntimeError, match="prompt_version"):
        cmd_label([], cfg, yes=True, p2_approved=False)


def test_full_label_requires_persisted_calibration_report(monkeypatch, tmp_path):
    cfg = {**load_config()["concern"],
           "labels_path": str(tmp_path / "labels.jsonl"),
           "batch_state_path": str(tmp_path / "batches.json")}
    report_path = tmp_path / "calibration_report.json"
    report_path.write_text(json.dumps({
        "yield": 0.40,
        "gate_p2_yield": "PASS",
        "measured_agreement": 0.90,
        "audited_rows": 50,
        "prompt_version": PROMPT_VERSION,
    }))
    monkeypatch.setattr(concern_labels, "_labeler", lambda _cfg: StubLabeler({}))

    summary = cmd_label([], cfg, yes=True, p2_approved=False)

    assert summary["submitted"] == 0


def test_p2_boolean_cannot_bypass_calibration_report(tmp_path):
    cfg = {**load_config()["concern"],
           "labels_path": str(tmp_path / "labels.jsonl"),
           "batch_state_path": str(tmp_path / "batches.json")}
    (tmp_path / "calibration_report.json").write_text(json.dumps({
        "yield": 0.40,
        "gate_p2_yield": "PASS",
        "measured_agreement": 0.84,
        "audited_rows": 50,
    }))

    with pytest.raises(RuntimeError, match="agreement"):
        cmd_label([], cfg, yes=True, p2_approved=True)


def test_calibration_audit_is_sample_bound_and_recomputed(tmp_path):
    sample = tmp_path / "calibration_sample.csv"
    sample.write_text("uid,text,labels\n" + "\n".join(
        f'u{i},text {i},"[]"' for i in range(50)
    ) + "\n")
    audits = [{"uid": f"u{i}", "exact_match": i < 44} for i in range(50)]
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "schema_version": "concern-calibration-audit-1",
        "reviewer_model": "gpt-5.6-luna",
        "reasoning_effort": "xhigh",
        "policy_prompt_version": PROMPT_VERSION,
        "sample_sha256": __import__("hashlib").sha256(sample.read_bytes()).hexdigest(),
        "audited_rows": 50,
        "exact_matches": 44,
        "measured_agreement": 0.88,
        "audits": audits,
    }))

    result = concern_labels._validated_calibration_audit(
        audit, sample, [f"u{i}" for i in range(50)],
    )

    assert result["audited_rows"] == 50
    assert result["exact_matches"] == 44
    assert result["measured_agreement"] == 0.88
    assert result["reviewer_model"] == "gpt-5.6-luna"


def test_calibration_audit_rejects_declared_agreement_mismatch(tmp_path):
    sample = tmp_path / "calibration_sample.csv"
    sample.write_text("uid,text,labels\nu0,text,[]\n")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "schema_version": "concern-calibration-audit-1",
        "policy_prompt_version": PROMPT_VERSION,
        "sample_sha256": __import__("hashlib").sha256(sample.read_bytes()).hexdigest(),
        "audited_rows": 1,
        "exact_matches": 1,
        "measured_agreement": 0.0,
        "audits": [{"uid": "u0", "exact_match": True}],
    }))

    with pytest.raises(RuntimeError, match="agreement"):
        concern_labels._validated_calibration_audit(audit, sample, ["u0"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
