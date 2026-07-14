"""Concern-efficacy labeling (D-023): prefilter -> LLM batch labels -> JSONL cache.

Implements the offline labeling pass of the concern-efficacy recommender spec
(docs/superpowers/specs/2026-07-10-concern-efficacy-recommender-design.md).
Review text is the only place product x acne-type outcomes exist in the data;
this module extracts them ONCE via grouped OpenRouter calls into a local
append-only JSONL cache. Everything downstream reads the cache; inference and
the test suite never touch the API. Subcommands: probe (free, gate P1),
calibrate (gate P2 sample), label (the full pass).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _batch_schema(uids: list[str]) -> dict:
    """Structured-output schema retaining attribution inside grouped calls."""
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "uid": {"type": "string", "enum": uids},
                "labels": LABEL_SCHEMA["properties"]["labels"],
            },
            "required": ["uid", "labels"],
            "additionalProperties": False,
        }}},
        "required": ["results"],
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
    rows, seen = [], set()
    for file in files:
        chunks = pd.read_csv(file, usecols=USECOLS, chunksize=100_000,
                             dtype={"author_id": str, "product_id": str})
        for df in chunks:
            df = df[df["product_id"].isin(catalog_ids)].copy()
            text = (df["review_text"].fillna("") + " "
                    + df["review_title"].fillna("")).str.strip()
            lower = text.str.lower()
            mask = None
            for rx in patterns.values():
                match = lower.str.contains(rx)
                mask = match if mask is None else (mask | match)
            df = df.assign(text_joined=text)[mask]
            df["skin_type"] = df["skin_type"].fillna("unknown")
            df["skin_tone"] = df["skin_tone"].fillna("")
            for r in df.itertuples(index=False):
                if not r.text_joined:
                    continue
                uid = review_uid(r.author_id, r.product_id, r.text_joined)
                if uid in seen:
                    continue
                seen.add(uid)
                rows.append({
                    "uid": uid, "author_id": r.author_id,
                    "product_id": r.product_id, "skin_type": r.skin_type,
                    "skin_tone": r.skin_tone,
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


class OpenRouterLabeler:
    """OpenRouter structured-output calls with a durable local batch spool."""

    url = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str, spool_dir, reviews_per_request=10,
                 concurrency=20, session=None):
        import requests  # lazy: free CLI paths and tests need no HTTP client
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY or OPENROUTER_KEY is required")
        self.model = model
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = reviews_per_request
        self.concurrency = concurrency
        # requests' module-level API creates one session per call; unlike a
        # shared Session it is safe across this small thread pool.
        self.session = session or requests
        self.headers = {"Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "X-Title": "SkinScan concern labeling",
                        # Identical retry after a crash is served without billing.
                        "X-OpenRouter-Cache": "true",
                        "X-OpenRouter-Cache-TTL": "86400"}

    def _call(self, rows):
        uids = [r["uid"] for r in rows]
        reviews = "\n".join(json.dumps({"uid": r["uid"], "text": r["text"]})
                            for r in rows)
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 120 * len(rows),
            "reasoning": {"enabled": False},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": reviews}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "review_concern_labels", "strict": True,
                "schema": _batch_schema(uids)}},
            "provider": {"require_parameters": True},
        }
        try:
            response = self.session.post(self.url, headers=self.headers,
                                         json=body, timeout=120)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") == "content_filter":
                return [(uid, None, "refusal") for uid in uids]
            data = json.loads(choice["message"]["content"])
            by_uid = {item["uid"]: item["labels"] for item in data["results"]}
            return [(uid, json.dumps({"labels": by_uid[uid]}), None)
                    if uid in by_uid else (uid, None, "missing_result")
                    for uid in uids]
        except Exception as exc:  # requests and malformed provider responses retry next run
            return [(uid, None, type(exc).__name__) for uid in uids]

    def submit(self, rows) -> str:
        digest = hashlib.md5("|".join(r["uid"] for r in rows).encode()).hexdigest()
        batch_id = f"openrouter_{digest}"
        path = self.spool_dir / f"{batch_id}.jsonl"
        existing = {}
        if path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                existing[rec[0]] = rec
        todo = [r for r in rows if existing.get(r["uid"], [None, None, "retry"])[2]
                is not None]
        groups = [todo[i:i + self.group_size]
                  for i in range(0, len(todo), self.group_size)]
        with path.open("a") as spool, ThreadPoolExecutor(
                max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._call, group) for group in groups]
            for future in as_completed(futures):
                for result in future.result():
                    spool.write(json.dumps(result) + "\n")
                spool.flush()
        return batch_id

    def status(self, batch_id: str) -> str:
        return "ended"

    def fetch(self, batch_id: str):
        latest = {}
        for line in (self.spool_dir / f"{batch_id}.jsonl").read_text().splitlines():
            rec = json.loads(line)
            latest[rec[0]] = tuple(rec)
        return list(latest.values())


def estimate_cost(rows, ccfg) -> float:
    """Conservative token estimate for a grouped OpenRouter pass."""
    groups = (len(rows) + ccfg["reviews_per_request"] - 1) // ccfg["reviews_per_request"]
    input_tokens = sum(len(r["text"]) for r in rows) / 4 + 450 * groups
    output_tokens = 80 * len(rows)
    return (input_tokens / 1e6 * ccfg["input_price_per_million"]
            + output_tokens / 1e6 * ccfg["output_price_per_million"])


def _labeler(ccfg):
    return OpenRouterLabeler(
        ccfg["labeling_model"], ccfg["batch_spool_dir"],
        ccfg["reviews_per_request"], ccfg["request_concurrency"])


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
    labeler = _labeler(ccfg)
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


def cmd_label(rows, ccfg, yes: bool, p2_approved=False) -> dict | None:
    cache = load_cache(ccfg["labels_path"])
    todo = [r for r in rows if r["uid"] not in cache]
    est_usd = estimate_cost(todo, ccfg)
    print(f"to label: {len(todo)} of {len(rows)} "
          f"(est cost ${est_usd:.2f} on {ccfg['labeling_model']})")
    if not yes:
        print("dry run — pass --yes to submit")
        return None
    if not p2_approved:
        raise RuntimeError("full labeling requires maintainer P2 sign-off; "
                           "rerun with --p2-approved after the 50-row check")
    if est_usd > ccfg["max_budget_usd"]:
        raise RuntimeError(f"estimated ${est_usd:.2f} exceeds "
                           f"${ccfg['max_budget_usd']:.2f} budget ceiling")
    labeler = _labeler(ccfg)
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
            sp.add_argument("--p2-approved", action="store_true")
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
        cmd_label(rows, ccfg, args.yes, args.p2_approved)


if __name__ == "__main__":
    main()
