# Plan 015: Concern-efficacy labels — prefilter, LLM labeling CLI, concern-stats aggregation

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in "STOP conditions" occurs, stop and report — do not
> improvise. Your reviewer maintains `plans/README.md` — do not edit it.
>
> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:executing-plans` (or subagent-driven-development) to implement
> task-by-task; steps use checkbox syntax for tracking. TDD: for every module
> step, write the failing test first, watch it fail, implement, watch it pass,
> commit.
>
> **Drift check (run first)**:
> `git diff --stat 509ab60 -- src/recommendation/ src/config.py configs/default.yaml requirements.txt`
> must be empty (you start from `main` @ `509ab60`; this plan does NOT depend
> on plan 013/014's branches — it creates new modules only). On a mismatch,
> STOP.

**Goal:** One-time LLM labeling of Sephora review texts into per-review
(concern, outcome) labels, plus the `concern_stats.json` aggregation — the
data layer of the concern-efficacy recommender spec.

**Architecture:** regex prefilter (config-driven) → Anthropic Message Batches
(Haiku, structured outputs) → append-only JSONL label cache (resumable, never
re-bills) → pure aggregation into Bayesian-smoothed per-product × concern
cells. The LLM is behind a 3-method duck-typed seam so the fast test suite
stubs it entirely.

**Tech stack:** pandas (installed), `anthropic` SDK (added by this plan),
stdlib `re`/`json`/`hashlib`. No pyarrow — the cache is JSONL (append-safe
resume, no new binary dep).

## Status

- **Priority**: P1 (data layer of the approved redirect; gates P2/P3 depend on it)
- **Effort**: L
- **Risk**: MED — external API dependency (operator-gated); calibration gate P2 is empirical
- **Depends on**: `main` @ `509ab60` only (spec committed). Plans 013/014 NOT required.
- **Spec**: `docs/superpowers/specs/2026-07-10-concern-efficacy-recommender-design.md` (approved 2026-07-10)
- **Planned at**: `509ab60`, 2026-07-10
- **Issue**: Kumario1/skinscan#1 (PRD milestone; the redirect supersedes #8's learned-ranker scope)

## Global constraints

- The fast `pytest` suite makes NO network calls and must pass with `anthropic` uninstalled → `import anthropic` happens ONLY inside `AnthropicBatchLabeler.__init__` (lazy).
- Deterministic: `hashlib.md5` uids (never builtin `hash()`), uid-sorted calibration sampling, no RNG.
- Idempotent + never re-bills: cache is checked before submit; a submitted-but-unfetched batch is drained from the state file before anything new is submitted.
- Config is the single source of knobs (`from ..config import load_config`); CLI flags override.
- Paid steps (calibrate ≈ $1–2, full label ≈ $60–90) are OPERATOR-GATED (Steps 7–8). Without API credentials they are a STOP, not an improvisation.
- Conventional commits referencing #1.

## Why this matters

Plan 013's gate failure proved the existing feature columns cannot beat pooled
review statistics (`plans/ranker-v2-probe-evidence.md`). The one untapped
signal is review TEXT — the only place *product × acne-type outcome* exists
("cleared my blackheads", "broke me out"). Gate P1 (spec) was executed
2026-07-10 and **PASSED decisively**: 970 catalog products have an n≥15 raw
mention cell for at least one acne concern (floor: 300). This plan turns that
signal into two artifacts: `review_concern_labels.jsonl` (per-review labels)
and `concern_stats.json` (smoothed per-product × concern efficacy cells).
Plan 016 runs the bake-off (gate P3) on them; plan 017 inverts the engine only
if P3 passes.

## Current state (verified 2026-07-10, this session)

- **Reviews**: `data/raw/sephora/reviews_*.csv` — 5 files, 1,094,411 rows,
  1,092,967 with text. Header (leading empty-name index column, ignore via
  `usecols`):
  `,author_id,rating,is_recommended,helpfulness,total_feedback_count,total_neg_feedback_count,total_pos_feedback_count,submission_time,review_text,review_title,skin_tone,eye_color,skin_type,hair_color,product_id,product_name,brand_name,price_usd`
  Read `author_id`/`product_id` as `str` (numeric ids otherwise come back int
  and `.encode()` in the md5 uid crashes — same landmine as plan 013).
  `skin_type` has NaN → fillna("unknown"); `skin_tone` NaN → fillna("").
- **Catalog**: `data/processed/catalog.json` is a **bare JSON list** of 1,634
  products. Rebuild if absent:
  `.venv/bin/python -m src.recommendation.import_catalog --csv data/raw/sephora/product_info.csv --format sephora`
  Use `from .import_catalog import load_catalog` → `list[Product]`.
- **Probe numbers** (raw lexicon matches, all-rows basis; cells are
  catalog-joinable): comedonal 14,972 / inflammatory 23,927 / cystic 13,766 /
  acne_general 170,381 / hyperpigmentation 37,777 / dryness 38,573. Any-match
  233,506 rows; **201,936 joinable to the catalog** (only those get labeled).
  Products with n≥15 cells: comedonal 174, inflammatory 249, cystic 176,
  general 968, hyperpig 349, dryness 455 → **gate P1: 970 ≥ 300, PASS**.
  Input-token estimate at 1,200-char truncation: ~22M tokens for joinable rows.
- **Environment**: `anthropic` NOT installed, `pyarrow` NOT installed (that is
  why the cache is JSONL), `ANTHROPIC_API_KEY` NOT set, no `ant` CLI. The
  paid steps therefore STOP for the operator.
- **Config**: `src.config.load_config()` returns the merged
  `configs/default.yaml`; top-level keys today:
  `classification, concern_report, detection, evaluation, paths, profile, ranker, recommendation, regions, tone`. `paths.reviews_raw: data/raw/sephora`,
  `paths.catalog_processed: data/processed/catalog.json`.
- **Worktrees have no `data/`** (gitignored) — run data-touching commands with
  the explicit flags pointing at the main checkout (commands below do).
- **Anthropic Batch API facts** (from the claude-api reference, current):
  `client.messages.batches.create(requests=[{custom_id, params}, ...])` (max
  100,000 requests / 256MB per batch; 50% price) → poll
  `batches.retrieve(id).processing_status == "ended"` → iterate
  `batches.results(id)`; results arrive in ANY order — key by `custom_id`.
  Haiku 4.5 (`claude-haiku-4-5`) supports **structured outputs on batches**:
  `params["output_config"] = {"format": {"type": "json_schema", "schema": ...}}`
  guarantees schema-valid JSON on success. Check `message.stop_reason ==
  "refusal"` before reading content. Requests are plain dicts (the SDK's
  `Request`/`MessageCreateParamsNonStreaming` are TypedDicts).

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| New tests | `.venv/bin/python -m pytest tests/test_concern_labels.py tests/test_concern_stats.py -q` | all pass |
| Full fast suite | `.venv/bin/python -m pytest -q` | all pass, no network |
| Standalone | `.venv/bin/python tests/test_concern_labels.py` | prints `ok` |
| Free probe (real data) | `.venv/bin/python -m src.recommendation.concern_labels probe --reviews-dir /Users/princekumar/Documents/skinscan/data/raw/sephora --catalog /Users/princekumar/Documents/skinscan/data/processed/catalog.json` | per-concern table + `gate_p1: PASS (970 >= 300)` (±1% after dedup) |
| Calibrate (PAID, operator) | `.venv/bin/python -m src.recommendation.concern_labels calibrate` | yield stats + `runs/concern/calibration_sample.csv` |
| Full label (PAID, operator) | `.venv/bin/python -m src.recommendation.concern_labels label --yes` | JSONL cache complete, summary printed |
| Aggregate | `.venv/bin/python -m src.recommendation.concern_stats` | writes `data/processed/concern_stats.json`, prints summary |

## Scope

**In scope** (only files you create/modify):
- `src/recommendation/concern_labels.py` — create
- `src/recommendation/concern_stats.py` — create
- `tests/test_concern_labels.py`, `tests/test_concern_stats.py` — create
- `configs/default.yaml` — add the `concern:` block (Step 1, nothing else)
- `requirements.txt` — add `anthropic`
- `docs/DECISIONS.md` — append D-023 (Step 6, verbatim text)
- `CONTEXT.md` — one glossary entry (Step 6)
- `README.md` — run-it lines (Step 6)

**Out of scope** (do NOT touch):
- `src/recommendation/engine.py`, `ranker.py`, `schema.py`, `src/pipeline/tone.py` — the inversion is plan 017; the bake-off is plan 016.
- `min_cell_n` / `admit_top_k` / `eval_path` config keys — consumed by plans 016/017, added there (repo rule: every key added must be consumed by the plan that adds it).
- `plans/README.md` — reviewer maintains it.

## Git workflow

Branch off `main`: `git switch -c issue-1-concern-labels` (worktree executors:
your worktree already starts at `main` — no merge needed, unlike plan 014).
Commit per step. Do NOT push, PR, or merge to `main`.

## Steps

### Step 1: Config keys + dependency

- [ ] Add to `configs/default.yaml` (top level, after `ranker:`):

```yaml
concern:                              # concern-efficacy labeling (D-023, spec 2026-07-10)
  labels_path: data/processed/review_concern_labels.jsonl   # append-only label cache
  stats_path: data/processed/concern_stats.json             # smoothed product x concern cells
  batch_state_path: runs/concern/batches.json               # batch-id resume state
  labeling_model: claude-haiku-4-5    # Anthropic Batch API model for the one-time pass
  text_truncate_chars: 1200           # review text sent to the labeler is capped here
  batch_chunk_size: 40000             # requests per batch (API cap 100k / 256MB)
  calibration_sample_size: 2000       # gate P2 sample
  smoothing_m: 20                     # Bayesian m for concern cells (same pattern as ranker)
  sub_cell_min_n: 5                   # min n for a skin-type sub-cell to be emitted
  prefilter:                          # word-boundary regex fragments, per concern
    acne_comedonal: ['blackheads?', 'whiteheads?', 'clogged pores?', 'comedon\w*']
    acne_inflammatory: ['pimples?', 'zits?', 'pustules?', 'papules?', 'inflammatory acne', 'inflamed acne']
    acne_cystic: ['cystic', 'cysts?', 'hormonal acne', 'hormonal breakouts?']
    acne_general: ['acne', 'break\s?outs?', 'broke me out', 'breaking out', 'break out', 'blemish(?:es)?']
    hyperpigmentation: ['dark spots?', 'hyperpigmentation', 'acne scar(?:s|ring)?', 'acne marks?', 'dark marks?', 'discoloration', 'melasma', 'sun spots?', 'post[- ]inflammatory']
    dryness: ['dryness', 'dry patch(?:es)?', 'flak(?:y|iness|ing)', 'dehydrated']
```

- [ ] Append `anthropic` on its own line to `requirements.txt`, then
  `.venv/bin/pip install anthropic`.
- [ ] **Verify**:
  `.venv/bin/python -c "from src.config import load_config; c=load_config()['concern']; print(sorted(c)); print(len(c['prefilter']))"`
  → `['batch_chunk_size', 'batch_state_path', 'calibration_sample_size', 'labeling_model', 'labels_path', 'prefilter', 'smoothing_m', 'stats_path', 'sub_cell_min_n', 'text_truncate_chars']` and `6`.
  `.venv/bin/python -c "import anthropic; print(anthropic.__version__)"` → a version string.
- [ ] Commit: `chore: concern labeling config block + anthropic dependency (#1)`

### Step 2: Prefilter + uid (TDD)

- [ ] Create `tests/test_concern_labels.py` with the header + first two tests:

```python
"""Tests for the concern-efficacy labeling pipeline (plan 015, D-023).

Pure-Python: the LLM sits behind a duck-typed labeler seam and is stubbed;
no network, no anthropic import needed for this suite.
"""
import json
import tempfile
from pathlib import Path

from src.config import load_config
from src.recommendation.concern_labels import (
    CONCERNS,
    compile_prefilter,
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
```

- [ ] Run: `.venv/bin/python -m pytest tests/test_concern_labels.py -q` →
  FAIL with `ModuleNotFoundError`/`ImportError` (module doesn't exist yet).
- [ ] Create `src/recommendation/concern_labels.py` with the docstring,
  constants, prefilter, uid, and row loading:

```python
"""Concern-efficacy labeling (D-023): prefilter -> LLM batch labels -> JSONL cache.

Implements the offline labeling pass of the concern-efficacy recommender spec
(docs/superpowers/specs/2026-07-10-concern-efficacy-recommender-design.md).
Review text is the only place product x acne-type outcomes exist in the data;
this module extracts them ONCE via the Anthropic Batch API into a local
append-only JSONL cache. Everything downstream reads the cache; inference and
the test suite never touch the API. Subcommands: probe (free, gate P1),
calibrate (gate P2 sample), label (the full pass).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

from ..config import load_config
from .import_catalog import load_catalog

CONCERNS = [
    "acne_comedonal", "acne_inflammatory", "acne_cystic", "acne_general",
    "hyperpigmentation", "dryness",
]
ACNE_CONCERNS = CONCERNS[:4]
VALID_OUTCOMES = {"helped", "worsened", "unclear"}

USECOLS = ["author_id", "rating", "is_recommended", "skin_tone", "skin_type",
           "product_id", "review_text", "review_title"]

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concern": {"type": "string", "enum": CONCERNS},
                    "outcome": {"type": "string",
                                "enum": ["helped", "worsened", "unclear"]},
                    "reviewer_has_condition": {"type": "boolean"},
                },
                "required": ["concern", "outcome", "reviewer_has_condition"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You label skincare product reviews for mentions of skin concerns and whether \
THIS product helped or worsened each one.

Concern ids (use exactly these):
- acne_comedonal: blackheads, whiteheads, clogged pores, comedones
- acne_inflammatory: pimples, zits, pustules, papules
- acne_cystic: cystic acne, hormonal acne, deep painful bumps
- acne_general: acne, breakouts, blemishes when the type is unspecified
- hyperpigmentation: dark spots, acne scars/marks, discoloration, melasma
- dryness: dryness, flaking, dry patches, dehydrated skin

For each concern mentioned, output one label:
- outcome "helped": this product improved it ("cleared my blackheads",
  "faded my dark spots"; "did not break me out" is acne_general helped).
- outcome "worsened": this product caused or worsened it ("broke me out",
  "made my acne worse", "clogged my pores").
- outcome "unclear": mentioned without a clear product effect
  ("I have acne-prone skin").
- reviewer_has_condition: true if the reviewer has (or had) the concern.

Rules: negation flips the outcome. Attribute outcomes to this product only.
"Bought it for wrinkles but it cleared my acne" -> acne_general helped.
No concern mentioned -> empty labels list.
"""


def compile_prefilter(prefilter_cfg: dict) -> dict[str, re.Pattern]:
    """Concern -> compiled word-boundary regex over the config term lists."""
    return {c: re.compile(r"\b(?:" + "|".join(terms) + r")\b")
            for c, terms in prefilter_cfg.items()}


def review_uid(author_id: str, product_id: str, text: str) -> str:
    """Stable review identity: md5 (NOT builtin hash) of author|product|text."""
    key = f"{author_id}|{product_id}|{text[:300]}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def load_review_rows(reviews_dir, catalog_ids: set, patterns: dict,
                     truncate_chars: int) -> list[dict]:
    """Prefilter-matching, catalog-joinable review rows, deduped by uid.

    Text = review_text + ' ' + review_title (matching is case-insensitive via
    lowercasing; the payload keeps original case, truncated).
    """
    files = sorted(glob.glob(str(Path(reviews_dir) / "reviews_*.csv")))
    frames = [pd.read_csv(f, usecols=USECOLS,
                          dtype={"author_id": str, "product_id": str})
              for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["product_id"].isin(catalog_ids)]
    text = (df["review_text"].fillna("") + " "
            + df["review_title"].fillna("")).str.strip()
    lower = text.str.lower()
    mask = None
    for rx in patterns.values():
        m = lower.str.contains(rx)
        mask = m if mask is None else (mask | m)
    df = df.assign(text_joined=text)[mask]
    df["skin_type"] = df["skin_type"].fillna("unknown")
    df["skin_tone"] = df["skin_tone"].fillna("")
    rows, seen = [], set()
    for r in df.itertuples(index=False):
        if not r.text_joined:
            continue
        uid = review_uid(r.author_id, r.product_id, r.text_joined)
        if uid in seen:
            continue
        seen.add(uid)
        rows.append({
            "uid": uid, "author_id": r.author_id, "product_id": r.product_id,
            "skin_type": r.skin_type, "skin_tone": r.skin_tone,
            "rating": float(r.rating) if pd.notna(r.rating) else None,
            "is_recommended": (float(r.is_recommended)
                               if pd.notna(r.is_recommended) else None),
            "text": r.text_joined[:truncate_chars],
        })
    return rows
```

  (The joined-text column is deliberately named `text_joined`, not `_text` —
  `itertuples` mangles leading-underscore column names.)
- [ ] Run the two tests → PASS.
- [ ] Add the row-loading test (tmp CSV; needs no real data):

```python
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
```

- [ ] Run → PASS. Commit:
  `feat: concern prefilter, review uid, prefiltered row loader (#1)`

### Step 3: Labeling orchestration — cache, state, resume (TDD)

- [ ] Append the stub + orchestration tests to `tests/test_concern_labels.py`:

```python
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
```

- [ ] Run → the new tests FAIL (`run_labeling`/`load_cache` missing).
- [ ] Implement in `concern_labels.py`:

```python
def load_cache(path) -> dict[str, dict]:
    """uid -> cached record. Missing file -> empty (first run)."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["uid"]] = rec
    return out


def append_cache(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _load_state(path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {"batches": {}}


def _save_state(path, state) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


def _record(row: dict, status: str, labels: list) -> dict:
    return {"uid": row["uid"], "author_id": row["author_id"],
            "product_id": row["product_id"], "skin_type": row["skin_type"],
            "skin_tone": row["skin_tone"], "rating": row["rating"],
            "is_recommended": row["is_recommended"],
            "status": status, "labels": labels}


def _parse_labels(text: str) -> list[dict]:
    data = json.loads(text)
    return [l for l in data["labels"]
            if isinstance(l, dict) and l.get("concern") in CONCERNS
            and l.get("outcome") in VALID_OUTCOMES]


def run_labeling(rows, labeler, cache_path, state_path, chunk_size,
                 poll_seconds=60, sleep=time.sleep) -> dict:
    """Label every row not yet cached. Idempotent and crash-safe:

    - already-cached uids are never resubmitted (never re-billed);
    - batches submitted by a crashed run are drained from the state file
      BEFORE anything new is submitted;
    - unparseable/refused replies are cached (billed once, never retried);
      API-level failures (errored/expired) are NOT cached -> retried next run.
    """
    cache = load_cache(cache_path)
    by_uid = {r["uid"]: r for r in rows}
    state = _load_state(state_path)
    summary = {"cached_before": 0, "submitted": 0, "ok": 0,
               "parse_error": 0, "refusal": 0, "failed": 0}

    def drain(batch_id):
        while labeler.status(batch_id) != "ended":
            sleep(poll_seconds)
        new = []
        for uid, text, failure in labeler.fetch(batch_id):
            row = by_uid.get(uid)
            if row is None or uid in cache:
                continue
            if failure == "refusal":
                new.append(_record(row, "refusal", []))
                summary["refusal"] += 1
            elif failure is not None:
                summary["failed"] += 1     # not cached -> retryable
            else:
                try:
                    labels = _parse_labels(text)
                    new.append(_record(row, "ok", labels))
                    summary["ok"] += 1
                except (ValueError, KeyError, TypeError, AttributeError):
                    new.append(_record(row, "parse_error", []))
                    summary["parse_error"] += 1
        append_cache(cache_path, new)
        cache.update({r["uid"]: r for r in new})
        state["batches"][batch_id] = {"fetched": True}
        _save_state(state_path, state)

    # 1) drain leftovers from a crashed run
    for bid, meta in list(state["batches"].items()):
        if not meta.get("fetched"):
            drain(bid)

    # 2) submit what is still unlabeled, then drain each batch
    todo = [r for r in rows if r["uid"] not in cache]
    summary["cached_before"] = len(rows) - len(todo)
    pending = []
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        bid = labeler.submit(chunk)
        summary["submitted"] += len(chunk)
        state["batches"][bid] = {"fetched": False}
        _save_state(state_path, state)
        pending.append(bid)
    for bid in pending:
        drain(bid)
    return summary
```

- [ ] Run: `.venv/bin/python -m pytest tests/test_concern_labels.py -q` → all
  pass; `.venv/bin/python tests/test_concern_labels.py` → `ok`.
- [ ] Commit: `feat: labeling orchestration — JSONL cache, batch state, crash-safe resume (#1)`

### Step 4: Real batch labeler + CLI (probe / calibrate / label)

No new fast tests (the seam is already covered); the free `probe` run against
real data is this step's verification.

- [ ] Append to `concern_labels.py`:

```python
class AnthropicBatchLabeler:
    """Thin wrapper over the Anthropic Message Batches API (50% price).

    Lazy-imports anthropic so the fast suite never needs the SDK or network.
    Zero-arg client: resolves ANTHROPIC_API_KEY (or an `ant auth` profile).
    """

    def __init__(self, model: str):
        import anthropic  # lazy: only the paid CLI paths construct this
        self.client = anthropic.Anthropic()
        self.model = model

    def submit(self, rows) -> str:
        requests = [{
            "custom_id": r["uid"],
            "params": {
                "model": self.model,
                "max_tokens": 500,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": r["text"]}],
                "output_config": {"format": {"type": "json_schema",
                                             "schema": LABEL_SCHEMA}},
            },
        } for r in rows]
        return self.client.messages.batches.create(requests=requests).id

    def status(self, batch_id: str) -> str:
        return self.client.messages.batches.retrieve(batch_id).processing_status

    def fetch(self, batch_id: str):
        out = []
        for res in self.client.messages.batches.results(batch_id):
            r = res.result
            if r.type == "succeeded":
                msg = r.message
                if msg.stop_reason == "refusal":
                    out.append((res.custom_id, None, "refusal"))
                    continue
                text = next((b.text for b in msg.content if b.type == "text"), "")
                out.append((res.custom_id, text, None))
            else:
                out.append((res.custom_id, None, r.type))  # errored/canceled/expired
        return out


def _match_counts(rows, patterns):
    """Per-concern joinable match counts + per-product cell sizes."""
    counts = {c: 0 for c in patterns}
    cells = {c: {} for c in patterns}
    for row in rows:
        low = row["text"].lower()
        for concern, rx in patterns.items():
            if rx.search(low):
                counts[concern] += 1
                cells[concern][row["product_id"]] = (
                    cells[concern].get(row["product_id"], 0) + 1)
    return counts, cells


def cmd_probe(rows, patterns) -> bool:
    counts, cells = _match_counts(rows, patterns)
    gate_products = set()
    print(f"joinable prefiltered rows: {len(rows)}")
    for concern in CONCERNS:
        n15 = [p for p, n in cells[concern].items() if n >= 15]
        print(f"{concern}: rows {counts[concern]}, products n>=15: {len(n15)}")
        if concern in ACNE_CONCERNS:
            gate_products.update(n15)
    passed = len(gate_products) >= 300
    print(f"gate_p1: {'PASS' if passed else 'FAIL'} "
          f"({len(gate_products)} >= 300 acne-concern products with n>=15)")
    return passed


def cmd_calibrate(rows, ccfg, n) -> dict:
    sample = sorted(rows, key=lambda r: r["uid"])[:n]   # deterministic
    labeler = AnthropicBatchLabeler(ccfg["labeling_model"])
    summary = run_labeling(sample, labeler, ccfg["labels_path"],
                           ccfg["batch_state_path"], ccfg["batch_chunk_size"])
    cache = load_cache(ccfg["labels_path"])
    sample_recs = [cache[r["uid"]] for r in sample if r["uid"] in cache]
    ok = [r for r in sample_recs if r["status"] == "ok"]
    outcome_bearing = [r for r in ok if any(
        l["outcome"] in ("helped", "worsened") for l in r["labels"])]
    yield_rate = len(outcome_bearing) / max(len(sample), 1)
    report = {"sample_size": len(sample), "labeled_ok": len(ok),
              "outcome_bearing": len(outcome_bearing),
              "yield": round(yield_rate, 4), "run_summary": summary,
              "gate_p2_yield": "PASS" if yield_rate >= 0.30 else "FAIL"}
    out_dir = Path(ccfg["batch_state_path"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
    by_uid = {r["uid"]: r for r in sample}
    hand = pd.DataFrame([{"uid": r["uid"], "text": by_uid[r["uid"]]["text"],
                          "labels": json.dumps(r["labels"])}
                         for r in sample_recs[:50]])
    hand.to_csv(out_dir / "calibration_sample.csv", index=False)
    print(json.dumps(report, indent=2))
    print("hand-check 50 rows in runs/concern/calibration_sample.csv "
          "(gate P2 agreement >= 85% is the maintainer's call)")
    return report


def cmd_label(rows, ccfg, yes: bool) -> dict | None:
    cache = load_cache(ccfg["labels_path"])
    todo = [r for r in rows if r["uid"] not in cache]
    est_in = sum(len(r["text"]) for r in todo) / 4 + 450 * len(todo)
    est_usd = est_in / 1e6 * 0.50 + len(todo) * 80 / 1e6 * 2.50
    print(f"to label: {len(todo)} of {len(rows)} "
          f"(est input {est_in/1e6:.0f}M tok, est cost ~${est_usd:.0f} on batch Haiku)")
    if not yes:
        print("dry run — pass --yes to submit")
        return None
    labeler = AnthropicBatchLabeler(ccfg["labeling_model"])
    summary = run_labeling(rows, labeler, ccfg["labels_path"],
                           ccfg["batch_state_path"], ccfg["batch_chunk_size"])
    print(json.dumps(summary, indent=2))
    print("next: .venv/bin/python -m src.recommendation.concern_stats")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "calibrate", "label"):
        sp = sub.add_parser(name)
        sp.add_argument("--reviews-dir")
        sp.add_argument("--catalog")
        if name == "calibrate":
            sp.add_argument("--n", type=int)
        if name == "label":
            sp.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    ccfg = cfg["concern"]
    patterns = compile_prefilter(ccfg["prefilter"])
    catalog = load_catalog(args.catalog or cfg["paths"]["catalog_processed"])
    catalog_ids = {p.product_id for p in catalog}
    rows = load_review_rows(args.reviews_dir or cfg["paths"]["reviews_raw"],
                            catalog_ids, patterns, ccfg["text_truncate_chars"])
    if args.cmd == "probe":
        cmd_probe(rows, patterns)
    elif args.cmd == "calibrate":
        cmd_calibrate(rows, ccfg, args.n or ccfg["calibration_sample_size"])
    else:
        cmd_label(rows, ccfg, args.yes)


if __name__ == "__main__":
    main()
```

- [ ] **Verify (import, no API)**:
  `.venv/bin/python -c "from src.recommendation.concern_labels import AnthropicBatchLabeler, cmd_probe, main; print('ok')"` → `ok`.
- [ ] **Verify (free real-data probe)** — run the probe command from the
  Commands table (with the absolute `--reviews-dir`/`--catalog` flags if in a
  worktree). Expected: `joinable prefiltered rows:` ≈ 201,936 (±1% after
  dedup), per-concern `products n>=15` ≈ 174/249/176/968/349/455, and
  `gate_p1: PASS (≈970 >= 300 ...)`. If FAIL, STOP.
- [ ] Commit: `feat: Anthropic batch labeler + probe/calibrate/label CLI (#1)`

### Step 5: concern_stats aggregation (TDD)

- [ ] Create `tests/test_concern_stats.py`:

```python
"""Tests for concern-stats aggregation (plan 015, D-023). Pure-Python."""
import json
import tempfile
from pathlib import Path

import pandas as pd

from src.recommendation.concern_stats import (
    build_concern_stats, labels_frame, main,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["product_id", "skin_type",
                                       "concern", "outcome"])


def test_smoothing_math():
    rows = ([("PA", "oily", "acne_general", "helped")] * 8
            + [("PA", "oily", "acne_general", "worsened")] * 2
            + [("PB", "dry", "acne_general", "helped")] * 2
            + [("PB", "dry", "acne_general", "worsened")] * 8)
    stats = build_concern_stats(_df(rows), m=20, sub_cell_min_n=5)
    # prior = 10 helped / 20 outcomes = 0.5; PA = (8 + 20*0.5) / (10 + 20) = 0.6
    cell = stats["cells"]["PA"]["acne_general"]["__all__"]
    assert cell["n"] == 10 and cell["helped"] == 8 and cell["worsened"] == 2
    assert abs(cell["smoothed"] - 0.6) < 1e-9
    assert abs(stats["priors"]["acne_general"] - 0.5) < 1e-9
    # PB = (2 + 10) / 30 = 0.4 -> ordering reflects evidence
    assert stats["cells"]["PB"]["acne_general"]["__all__"]["smoothed"] < 0.5


def test_unclear_counted_but_excluded_from_n():
    rows = [("PA", "oily", "dryness", "helped"),
            ("PA", "oily", "dryness", "unclear")]
    stats = build_concern_stats(_df(rows), m=20, sub_cell_min_n=5)
    cell = stats["cells"]["PA"]["dryness"]["__all__"]
    assert cell["n"] == 1 and cell["n_unclear"] == 1


def test_skin_type_subcells_respect_min_n():
    rows = ([("PA", "oily", "acne_general", "helped")] * 5
            + [("PA", "dry", "acne_general", "helped")] * 2)
    stats = build_concern_stats(_df(rows), m=20, sub_cell_min_n=5)
    concern_cell = stats["cells"]["PA"]["acne_general"]
    assert "oily" in concern_cell and "dry" not in concern_cell
    assert concern_cell["__all__"]["n"] == 7


def test_labels_frame_ignores_non_ok_records():
    recs = [
        {"uid": "u1", "product_id": "PA", "skin_type": "oily", "status": "ok",
         "labels": [{"concern": "acne_general", "outcome": "helped",
                     "reviewer_has_condition": True}]},
        {"uid": "u2", "product_id": "PB", "skin_type": "dry",
         "status": "parse_error", "labels": []},
        {"uid": "u3", "product_id": "PC", "skin_type": "dry", "status": "ok",
         "labels": []},                      # ok but nothing mentioned
    ]
    df = labels_frame(recs)
    assert list(df["product_id"]) == ["PA"]


def test_cli_end_to_end():
    recs = [{"uid": f"u{i}", "product_id": "PA", "skin_type": "oily",
             "status": "ok",
             "labels": [{"concern": "acne_general", "outcome": "helped",
                         "reviewer_has_condition": True}]}
            for i in range(3)]
    with tempfile.TemporaryDirectory() as td:
        labels = Path(td) / "labels.jsonl"
        out = Path(td) / "concern_stats.json"
        labels.write_text("".join(json.dumps(r) + "\n" for r in recs))
        main(["--labels", str(labels), "--out", str(out)])
        stats = json.loads(out.read_text())
        assert stats["cells"]["PA"]["acne_general"]["__all__"]["n"] == 3
        assert stats["smoothing_m"] == 20


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
```

- [ ] Run → FAIL (module missing).
- [ ] Create `src/recommendation/concern_stats.py`:

```python
"""Concern-stats aggregation (D-023): labels JSONL -> concern_stats.json.

Per-product x concern efficacy cells with Bayesian-m smoothing toward the
per-concern global help rate, plus skin-type sub-cells where n permits.
build_concern_stats is a pure function over a labels frame so plan 016's
bake-off can call it on a train-only slice; the CLI applies it to the full
cache. A product with no outcome rows for a concern gets NO cell — inference
falls down the ladder (concern cell -> acne_general -> pooled rating, spec).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import load_config


def labels_frame(records) -> pd.DataFrame:
    """Cache records -> one row per (review, label): status=='ok' only."""
    rows = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        for l in rec["labels"]:
            rows.append((rec["product_id"], rec["skin_type"],
                         l["concern"], l["outcome"]))
    return pd.DataFrame(rows, columns=["product_id", "skin_type",
                                       "concern", "outcome"])


def _cell(group: pd.DataFrame, m: float, prior: float) -> dict:
    helped = int((group["outcome"] == "helped").sum())
    worsened = int((group["outcome"] == "worsened").sum())
    unclear = int((group["outcome"] == "unclear").sum())
    n = helped + worsened
    return {
        "n": n, "helped": helped, "worsened": worsened, "n_unclear": unclear,
        "help_rate": (helped / n) if n else None,
        "smoothed": (helped + m * prior) / (n + m) if (n + m) else None,
    }


def build_concern_stats(df: pd.DataFrame, m: float,
                        sub_cell_min_n: int) -> dict:
    """df columns: product_id, skin_type, concern, outcome (one row/label)."""
    outcomes = df[df["outcome"].isin(["helped", "worsened"])]
    priors = {}
    for concern, g in outcomes.groupby("concern"):
        priors[concern] = float((g["outcome"] == "helped").mean())
    cells: dict = {}
    for (pid, concern), g in df.groupby(["product_id", "concern"]):
        prior = priors.get(concern)
        if prior is None:
            continue          # concern has no outcome rows anywhere
        cell = _cell(g, m, prior)
        if cell["n"] == 0 and cell["n_unclear"] == 0:
            continue
        entry = {"__all__": cell}
        for skin_type, sg in g.groupby("skin_type"):
            sub = _cell(sg, m, prior)
            if sub["n"] >= sub_cell_min_n:
                entry[skin_type] = sub
        cells.setdefault(pid, {})[concern] = entry
    return {"smoothing_m": m, "sub_cell_min_n": sub_cell_min_n,
            "priors": priors, "cells": cells}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    ccfg = load_config()["concern"]
    labels_path = Path(args.labels or ccfg["labels_path"])
    out_path = Path(args.out or ccfg["stats_path"])
    records = [json.loads(line) for line in labels_path.read_text().splitlines()
               if line.strip()]
    df = labels_frame(records)
    stats = build_concern_stats(df, ccfg["smoothing_m"], ccfg["sub_cell_min_n"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=1))
    per_concern = {c: sum(1 for p in stats["cells"].values() if c in p)
                   for c in stats["priors"]}
    print(json.dumps({"labeled_reviews": len(records),
                      "label_rows": len(df),
                      "products_with_cells": len(stats["cells"]),
                      "products_per_concern": per_concern,
                      "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] Run: `.venv/bin/python -m pytest tests/test_concern_stats.py -q` → all
  pass; standalone prints `ok`.
- [ ] Full suite: `.venv/bin/python -m pytest -q` → all pass, no regressions,
  no network.
- [ ] Commit: `feat: concern-stats aggregation — smoothed product x concern cells (#1)`

### Step 6: Docs — D-023, glossary, README

- [ ] Append to `docs/DECISIONS.md` (verbatim):

```markdown
## D-023 — Concern-efficacy labels: LLM-mined review text is the new ranking signal (2026-07-10)

**LOCKED.** Spec: `docs/superpowers/specs/2026-07-10-concern-efficacy-recommender-design.md`.
Review texts are the only place *product × acne-type outcome* exists in the
D-015 dataset. A one-time Anthropic Batch pass (Haiku, structured outputs)
labels prefiltered reviews with (concern, outcome ∈ helped/worsened/unclear);
labels are cached locally (`review_concern_labels.jsonl`) and the API is never
called at inference or in tests. Ordered go/no-go gates: **P1** mention
density (executed 2026-07-10 — **PASS**, 970 catalog products with an n≥15
acne-concern cell vs the 300 floor); **P2** calibration (≥30% outcome-bearing
yield on a ~2k sample AND ≥85% maintainer agreement on a 50-review
hand-check); **P3** the bake-off (a concern-conditioned candidate ships only
if it beats the pooled StatsRanker champion on BOTH pooled metrics under the
D-022 harness — else the engine keeps its v2 contract). Aggregates live in
`concern_stats.json` (Bayesian-m smoothing toward per-concern priors,
skin-type sub-cells, fallback ladder concern → acne_general → pooled rating).
```

- [ ] `CONTEXT.md` — add a glossary entry after **review-stats**:

```markdown
**concern-stats** — Per-product × concern efficacy aggregates mined from
review text via one-time LLM labeling (helped/worsened counts, smoothed help
rate; D-023). Feeds concern-conditioned ranking (gates P2/P3 pending) and the
report's per-concern evidence lines.
```

- [ ] `README.md` — under "## Run it", after the ranker lines:

```bash
# Concern-efficacy labeling (D-023). probe is free; calibrate/label call the
# Anthropic Batch API (needs ANTHROPIC_API_KEY; ~$1 / ~$60-90 one-time):
.venv/bin/python -m src.recommendation.concern_labels probe
.venv/bin/python -m src.recommendation.concern_labels calibrate
.venv/bin/python -m src.recommendation.concern_labels label --yes
.venv/bin/python -m src.recommendation.concern_stats
```

- [ ] **Verify**: `grep -c "D-023" docs/DECISIONS.md` ≥ 1;
  `grep -n "concern-stats" CONTEXT.md` → 1 match;
  `grep -n "concern_labels" README.md` → ≥ 1.
- [ ] Commit: `docs: D-023 concern-efficacy labeling decision + glossary + run-it (#1)`

### Step 7: OPERATOR-GATED — calibration run (gate P2, ~$1–2)

Requires API credentials (`ANTHROPIC_API_KEY`, currently NOT set). If absent:
**STOP and report** — Steps 1–6 are complete and fully verified by the fast
suite + free probe; this step is the operator's.

- [ ] `.venv/bin/python -m src.recommendation.concern_labels calibrate`
  (add the absolute `--reviews-dir`/`--catalog` flags if in a worktree).
  Expected: batch completes in minutes–an hour; prints the calibration report
  with `yield` and `gate_p2_yield`.
- [ ] If `gate_p2_yield: FAIL` (< 0.30) → STOP and report the numbers.
- [ ] Hand the maintainer `runs/concern/calibration_sample.csv` for the
  agreement check (≥ 85% of 50 spot-checked labels correct). **STOP here for
  maintainer sign-off regardless of yield** — the full $60–90 pass never runs
  without it.

### Step 8: OPERATOR-GATED — full labeling pass + aggregation (~$60–90)

Only after the maintainer approves gate P2.

- [ ] `.venv/bin/python -m src.recommendation.concern_labels label` (dry run —
  prints count + cost estimate). If the estimate exceeds **$100**, STOP.
- [ ] `.venv/bin/python -m src.recommendation.concern_labels label --yes`
  (calibration rows are already cached and are not re-billed; a crashed run is
  resumed by re-running the same command).
- [ ] `.venv/bin/python -m src.recommendation.concern_stats` → writes
  `data/processed/concern_stats.json`; record the printed summary
  (products_with_cells, per-concern counts) in the completion report.
- [ ] Artifacts are gitignored (`data/processed/`, `runs/`) — the committed
  deliverable is code + docs; the numbers go in the completion note.

## Test plan

Steps 2/3/5 are the test plan: 9 tests in `tests/test_concern_labels.py`
(prefilter, uid, row loading, cache write, rerun skip, parse-error caching,
API-error retry, invalid-label filtering, chunking, crash-resume) + 5 in
`tests/test_concern_stats.py` (smoothing math, unclear handling, sub-cell
min-n, non-ok filtering, CLI end-to-end). Both files runnable standalone
(`__main__` prints `ok`). The paid pipeline's empirical outcome (gates P2)
is verified by running, not asserted in the suite — same pattern as plan 013's
real-data gate.

## Done criteria

- [ ] `.venv/bin/python -m pytest tests/test_concern_labels.py tests/test_concern_stats.py -q` → all pass
- [ ] `.venv/bin/python -m pytest -q` → all pass, no regressions, no network
- [ ] Both test files print `ok` standalone
- [ ] Free probe run reproduces gate P1 PASS (≈970 ≥ 300)
- [ ] `import anthropic` appears ONLY inside `AnthropicBatchLabeler.__init__`
      (`grep -n "import anthropic" src/recommendation/concern_labels.py` → 1 hit, inside the class)
- [ ] `git diff --stat 509ab60 -- src/recommendation/engine.py src/recommendation/ranker.py src/recommendation/schema.py src/pipeline/tone.py` → empty
- [ ] Only in-scope files modified
- [ ] Steps 7–8 either executed with numbers recorded, or STOP-reported as
      operator-gated (expected when no API key is present)

## STOP conditions

- Drift check fails.
- Any fast test cannot pass without weakening an assertion — the seam design
  is wrong; report, don't patch around it.
- The free probe's gate P1 does not PASS (it did on 2026-07-10 with identical
  inputs — a FAIL means the prefilter or join drifted from this plan).
- Step 7: no API credentials → report and stop (expected).
- Step 7: `yield < 0.30` or maintainer agreement < 85% → gate P2 FAIL; the
  spec says the direction is reassessed — do NOT proceed to Step 8.
- Step 8: dry-run cost estimate > $100, or batch `failed` count > 5% of
  submissions after one resume attempt.
- Anything appears to require editing `engine.py`/`ranker.py`/`schema.py` —
  scope boundary is wrong (that's plans 016/017); stop.

## Maintenance notes

- **Plan 016 (bake-off)** consumes `review_concern_labels.jsonl` (it carries
  `author_id`, `rating`, `is_recommended` precisely so the reviewer-disjoint
  split and baselines can be computed) and calls `build_concern_stats` on the
  train slice. `concern_stats.json` as written by the CLI is the
  full-data artifact for the report; the bake-off must NOT eval on it.
- **Cost levers if ever re-run**: the system prompt is ~350 tokens/request —
  batching k reviews per request would cut cost ~40% at the price of
  attribution risk; deliberately not done (calibration quality first).
  `// ponytail: one review per request; group into k-review requests only if a re-run's cost matters.`
- **The lexicon is config** — extending a concern's term list changes only the
  prefilter recall (the LLM sees full text regardless); re-run `probe` after
  edits to re-check gate P1.
- **Refusals** are cached with `status: "refusal"` and excluded from stats;
  expect ~0 on this content (product reviews).
