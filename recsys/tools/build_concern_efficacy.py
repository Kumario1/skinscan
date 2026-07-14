"""LLM batch: review-text concern mining -> concern_efficacy signal store.

PHASE 2 STUB — a port of the D-023 pipeline (src/recommendation/
concern_labels.py -> concern_stats.py): prefilter review texts, label each with
(concern, outcome in helped/worsened/unclear) via a batch LLM call, cache to
JSONL, then aggregate to product x concern x skin-type cells with Bayesian
smoothing toward per-concern priors.

Store (signals/concern_efficacy.v1.json), keyed by product_id:
    {concern: {"all": {"n": int, "helped": int, "worsened": int,
                        "help_rate": float, "smoothed": float},
               "by_skin_type": {skin: {...same...}}}}
Cells carry n so consumers scale their confidence; the provider ladder falls
back concern cell -> acne_general -> pooled rating. Never called at inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROMPT_VERSION = "p1"
BUILDER = "recsys.tools.build_concern_efficacy@1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.parse_args(argv)
    raise SystemExit(
        "Phase 2 not enabled: port the D-023 mining pipeline from "
        "src/recommendation/concern_labels.py + concern_stats.py "
        "(prefilter -> LLM labels -> JSONL cache -> smoothed aggregate). "
        "See ARCHITECTURE.md 'Phases'."
    )


if __name__ == "__main__":
    raise SystemExit(main())
