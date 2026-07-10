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
