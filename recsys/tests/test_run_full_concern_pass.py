from recsys.tools.run_full_concern_pass import select_corpus


def _rows(*uids):
    return [{"uid": uid, "product_id": "p", "text": uid} for uid in uids]


def test_full_mode_uid_sorts_the_whole_corpus():
    rows = _rows("c", "a", "b")
    corpus, todo = select_corpus(rows, cache={}, limit=None)
    assert [r["uid"] for r in corpus] == ["a", "b", "c"]
    assert [r["uid"] for r in todo] == ["a", "b", "c"]


def test_limit_keeps_every_cached_uid_for_draining():
    rows = _rows("d", "a", "c", "b")          # a, b cached; c, d uncached
    cache = {"a": {}, "b": {}}

    corpus, todo = select_corpus(rows, cache, limit=1)

    # Only one NEW (uncached) row is labeled — the first uncached uid, uid-sorted.
    assert [r["uid"] for r in todo] == ["c"]
    # Every cached uid survives so run_labeling can drain leftover batches
    # against a complete by_uid without discarding already-paid rows (Finding 9).
    assert {"a", "b"} <= {r["uid"] for r in corpus}
    # The second uncached uid is not labeled this run.
    assert "d" not in {r["uid"] for r in corpus}
    # Uid-sorted, matching full-mode chunk order (Finding 20).
    assert [r["uid"] for r in corpus] == ["a", "b", "c"]


def test_limit_and_full_mode_share_uid_sorted_chunk_order():
    rows = _rows("d", "a", "c", "b")

    full, _ = select_corpus(rows, cache={}, limit=None)
    limited, _ = select_corpus(rows, cache={}, limit=2)

    full_order = [r["uid"] for r in full]
    limited_order = [r["uid"] for r in limited]
    assert limited_order == sorted(limited_order)
    # Limited corpus is a uid-sorted subsequence of the full corpus, so both
    # modes chunk identically.
    assert limited_order == [uid for uid in full_order if uid in set(limited_order)]
