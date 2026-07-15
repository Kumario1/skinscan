"""Full Azure Phase 2 concern-labeling pass over the entire prefiltered corpus.

The default `concern_labels label` preflight estimates output tokens from the
50-row calibration, where reasoning dominates (~250 tok/row). Large batches
amortize reasoning to ~100 tok/row, so that estimate over-projects ~2x and
would falsely trip the budget preflight. This runner keeps BOTH real guards —
the P2 sign-off gate (>=85% agreement) and the per-request runtime budget
reservation (hard $max_budget_usd ceiling) — and skips only the pessimistic
pre-estimate. It is idempotent/resumable: cached uids are never re-billed.

Usage:
    python -m recsys.tools.run_full_concern_pass            # run
    python -m recsys.tools.run_full_concern_pass --limit N  # bounded warm-up
"""
from __future__ import annotations

import argparse
import json
import os

from src.config import load_config
from src.recommendation import concern_labels as cl


def select_corpus(rows, cache, limit):
    """Uid-sorted corpus to hand run_labeling, plus the rows it will newly label.

    run_labeling must always see the FULL cached population so it can drain
    leftover batches from a crashed run without discarding already-paid rows
    (an incomplete by_uid silently drops them). --limit therefore caps only how
    many NEW (uncached) rows are labeled: the corpus is every cached uid plus
    the first N uncached uids. Both modes are uid-sorted so full-mode and
    --limit-mode chunk batches identically.
    """
    ordered = sorted(rows, key=lambda r: r["uid"])
    uncached = [r for r in ordered if r["uid"] not in cache]
    if limit is None:
        return ordered, uncached
    labeled_new = {r["uid"] for r in uncached[:limit]}
    corpus = [r for r in ordered if r["uid"] in cache or r["uid"] in labeled_new]
    return corpus, uncached[:limit]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="label at most N uncached rows (uid-sorted); omit for the full corpus")
    parser.add_argument("--reviews-dir")
    parser.add_argument("--catalog")
    args = parser.parse_args(argv)

    cfg = load_config()
    ccfg = cfg["concern"]

    # Enforce the P2 maintainer gate exactly as `label --yes` does.
    report = cl._require_calibration_report(ccfg)
    print(json.dumps({"p2_gate": "PASS",
                      "measured_agreement": report.get("measured_agreement"),
                      "audited_rows": report.get("audited_rows"),
                      "prompt_version": report.get("prompt_version")}, indent=2))

    patterns = cl.compile_prefilter(ccfg["prefilter"])
    catalog = cl.load_catalog(args.catalog or cfg["paths"]["catalog_processed"])
    catalog_ids = {p.product_id for p in catalog}
    rows = cl.load_review_rows(args.reviews_dir or cfg["paths"]["reviews_raw"],
                               catalog_ids, patterns, ccfg["text_truncate_chars"])

    provider, model, prompt_version = cl._configured_labeler_identity(ccfg)
    cache = cl.load_cache(ccfg["labels_path"], prompt_version, provider, model)
    corpus, todo = select_corpus(rows, cache, args.limit)

    usage = cl.azure_usage_summary(ccfg.get("azure_usage_path"), model, None)
    in_price = float(os.environ["AZURE_INPUT_PRICE_PER_MILLION"])
    out_price = float(os.environ["AZURE_OUTPUT_PRICE_PER_MILLION"])
    spent = (usage["input_tokens"] / 1e6 * in_price
             + usage["output_tokens"] / 1e6 * out_price)
    print(f"provider={provider} model={model} prompt={prompt_version}")
    print(f"corpus={len(rows)} cached={len(rows) - len([r for r in rows if r['uid'] not in cache])} "
          f"todo={len(todo)} labeling_corpus={len(corpus)}")
    print(f"cumulative Azure spend so far: ${spent:.4f} of ${ccfg['max_budget_usd']:.2f} ceiling")
    print(f"batch={ccfg['reviews_per_request']} concurrency={ccfg['request_concurrency']} "
          f"reasoning={ccfg.get('azure_reasoning_effort')}")

    labeler = cl._labeler(ccfg)   # installs the hard budget + request reservation guard
    summary = cl.run_labeling(corpus, labeler,
                              ccfg["labels_path"], ccfg["batch_state_path"],
                              ccfg["batch_chunk_size"])

    usage = cl.azure_usage_summary(ccfg.get("azure_usage_path"), model, None)
    spent = (usage["input_tokens"] / 1e6 * in_price
             + usage["output_tokens"] / 1e6 * out_price)
    summary["azure_cumulative_cost_usd"] = round(spent, 4)
    summary["azure_cumulative_requests"] = usage["requests"]
    print(json.dumps(summary, indent=2))
    print(f"cumulative Azure spend now: ${spent:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
