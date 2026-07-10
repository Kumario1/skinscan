"""Tests for the concern-efficacy labeling pipeline (plan 015, D-023).

Pure-Python: the LLM sits behind a duck-typed labeler seam and is stubbed;
no network or provider credentials needed for this suite.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.recommendation.concern_labels import (
    CONCERNS,
    OpenRouterLabeler,
    cmd_label,
    compile_prefilter,
    estimate_cost,
    load_cache,
    load_review_rows,
    review_uid,
    run_labeling,
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


def test_full_run_estimate_fits_configured_balance():
    cfg = load_config()["concern"]
    rows = [{"text": "x" * 1200}] * 202_000
    assert estimate_cost(rows, cfg) < cfg["max_budget_usd"] <= 9


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
