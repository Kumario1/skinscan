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


class ConcernStatsRanker:
    """Duck-typed D-005 ranker over concern_stats.json: mean smoothed help-rate
    lift over the concern prior, across the report's concerns. Skin-type
    sub-cell when present, else __all__. Products without cells score 0, so the
    engine's rules-only order is untouched for them (D-019)."""

    def __init__(self, stats: dict, concerns: list[str]):
        self.stats = stats
        self.concerns = [c for c in concerns if c in stats.get("priors", {})]

    @classmethod
    def from_file(cls, path, concerns: list[str]) -> "ConcernStatsRanker":
        return cls(json.loads(Path(path).read_text()), concerns)

    def score(self, product, profile) -> float:
        cells = self.stats["cells"].get(product.product_id)
        if not cells or not self.concerns:
            return 0.0
        lifts = []
        for concern in self.concerns:
            cell = cells.get(concern)
            if not cell:
                continue
            sub = cell.get(profile.skin_type if profile else "") or cell["__all__"]
            lifts.append(sub["smoothed"] - self.stats["priors"][concern])
        return sum(lifts) / len(lifts) if lifts else 0.0


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
