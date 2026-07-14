"""Cached review labels -> concern-efficacy signal store.

The model-labeling pass is deliberately separate from aggregation: this command
consumes the append-only JSONL contract produced by the D-023 labeler. That
keeps deterministic store rebuilds free, resumable, and networkless.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from ..contracts import sha256_file
from .common import STORE_SCHEMA_VERSION, register_store, write_json

PROMPT_VERSION = "p7"
BUILDER = "recsys.tools.build_concern_efficacy@1"
ACNE_CONCERNS = frozenset((
    "acne_comedonal", "acne_inflammatory", "acne_cystic", "acne_general",
))
P3_TEST_FRACTION = 0.25


def p3_gate_passed(result: dict) -> bool:
    """Return whether a concern-conditioned candidate beats the pooled floor
    on both D-023 metrics: ROC-AUC and reviewer×concern pairwise ordering."""
    pooled = result.get("pooled") or {}
    champion = (
        pooled.get("champion")
        or pooled.get("stats_ranker")
        or pooled.get("bayesian")
    )
    if not champion:
        return False
    candidates = result.get("candidates") or {
        name: metrics for name, metrics in pooled.items()
        if name not in {"champion", "stats_ranker", "bayesian"}
    }
    champion_auc = champion.get("roc_auc")
    champion_pairwise = champion.get("pairwise")
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in (champion_auc, champion_pairwise)):
        return False
    return any(
        isinstance(metrics, dict)
        and all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in (metrics.get("roc_auc"), metrics.get("pairwise")))
        and metrics["roc_auc"] > champion_auc
        and metrics["pairwise"] > champion_pairwise
        for metrics in candidates.values()
    )


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(label == 1 for label in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        positive_rank_sum += average_rank * sum(label == 1 for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _p3_bakeoff(records: list[dict], smoothing_m: float,
                catalog_product_ids: frozenset[str] | None = None) -> dict | None:
    reviews = []
    outcomes = []
    for record in records:
        if record.get("status") != "ok":
            continue
        product_id = str(record.get("product_id"))
        if catalog_product_ids is not None and product_id not in catalog_product_ids:
            continue
        author_id = record.get("author_id")
        try:
            rating = float(record.get("rating"))
        except (TypeError, ValueError):
            continue
        if author_id is None or not math.isfinite(rating):
            continue
        author_id = str(author_id)
        is_test = int(hashlib.md5(author_id.encode()).hexdigest(), 16) % 1000 < int(P3_TEST_FRACTION * 1000)
        reviews.append((author_id, product_id, rating, is_test))
        for label in record.get("labels") or []:
            if (label.get("reviewer_has_condition") is True
                    and label.get("outcome") in {"helped", "worsened"}
                    and label.get("concern")):
                outcomes.append((
                    author_id, product_id, str(label["concern"]),
                    int(label["outcome"] == "helped"), is_test,
                ))
    if not reviews:
        return None

    train_reviews = [row for row in reviews if not row[3]]
    test_outcomes = [row for row in outcomes if row[4]]
    global_mean = sum(row[2] for row in train_reviews) / len(train_reviews) if train_reviews else None
    pooled_cells: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for _author, product_id, rating, _is_test in train_reviews:
        pooled_cells[product_id][0] += 1
        pooled_cells[product_id][1] += rating
    pooled_scores = {
        product_id: (total + smoothing_m * global_mean) / (count + smoothing_m)
        for product_id, (count, total) in pooled_cells.items()
    } if global_mean is not None else {}

    concern_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for _author, product_id, concern, label, is_test in outcomes:
        if is_test:
            continue
        concern_totals[concern][0] += 1
        concern_totals[concern][1] += label
        cells[(product_id, concern)][0] += 1
        cells[(product_id, concern)][1] += label
    priors = {
        concern: helped / count
        for concern, (count, helped) in concern_totals.items()
        if count
    }

    def concern_score(product_id: str, concern: str) -> float:
        source_concern = concern
        cell = cells.get((product_id, source_concern))
        if not cell and concern.startswith("acne_"):
            source_concern = "acne_general"
            cell = cells.get((product_id, source_concern))
        prior = priors.get(source_concern, priors.get(concern, 0.5))
        if not cell:
            return prior
        count, helped = cell
        return (helped + smoothing_m * prior) / (count + smoothing_m)

    candidate_labels = []
    candidate_scores = []
    pooled_scores_for_test = []
    pairwise_rows = []
    for author_id, product_id, concern, label, _is_test in test_outcomes:
        candidate_labels.append(label)
        candidate_scores.append(concern_score(product_id, concern))
        pooled_scores_for_test.append(pooled_scores.get(product_id, global_mean or 0.0))
        pairwise_rows.append((author_id, concern, label, candidate_scores[-1], pooled_scores_for_test[-1]))

    def pairwise(method_index: int) -> float | None:
        groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
        for author_id, concern, label, candidate, pooled in pairwise_rows:
            groups[(author_id, concern)].append((label, (candidate, pooled)[method_index]))
        total = hits = 0.0
        for rows in groups.values():
            positives = [score for label, score in rows if label == 1]
            negatives = [score for label, score in rows if label == 0]
            for positive in positives:
                for negative in negatives:
                    total += 1
                    hits += 1 if positive > negative else 0.5 if positive == negative else 0
        return hits / total if total else None

    candidate_metrics = {
        "roc_auc": _auc(candidate_labels, candidate_scores),
        "pairwise": pairwise(0),
    }
    champion_metrics = {
        "roc_auc": _auc(candidate_labels, pooled_scores_for_test),
        "pairwise": pairwise(1),
    }
    result = {
        "protocol": "D-023-P3",
        "test_fraction": P3_TEST_FRACTION,
        "n_train_reviews": len(train_reviews),
        "n_test_outcomes": len(test_outcomes),
        "pooled": {
            "champion": champion_metrics,
            "concern_conditioned": candidate_metrics,
        },
    }
    result["gate_passed"] = p3_gate_passed(result)
    return result


def _cell(counts: dict[str, int], prior: float, smoothing_m: float) -> dict:
    helped = counts.get("helped", 0)
    worsened = counts.get("worsened", 0)
    unclear = counts.get("unclear", 0)
    n = helped + worsened
    return {
        "n": n,
        "helped": helped,
        "worsened": worsened,
        "n_unclear": unclear,
        "help_rate": round(helped / n, 6) if n else None,
        "smoothed": round((helped + smoothing_m * prior) / (n + smoothing_m), 6),
    }


def build(
    labels_path: Path,
    out_path: Path,
    data_root: Path,
    *,
    catalog_products: int,
    catalog_product_ids: frozenset[str] | None = None,
    smoothing_m: float = 20,
    sub_cell_min_n: int = 5,
    p3_evaluation: dict | None = None,
) -> dict:
    records = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    p3 = (p3_evaluation if p3_evaluation is not None
          else _p3_bakeoff(records, smoothing_m, catalog_product_ids))
    if p3 is not None and not p3_gate_passed(p3):
        raise RuntimeError("P3 bake-off failed: no candidate beat the pooled champion on both metrics")
    global_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    product_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    skin_counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record.get("status") != "ok" or record.get("prompt_version") != PROMPT_VERSION:
            continue
        product_id = str(record["product_id"])
        if catalog_product_ids is not None and product_id not in catalog_product_ids:
            continue
        skin_type = str(record.get("skin_type") or "unknown")
        for label in record.get("labels") or []:
            if label.get("reviewer_has_condition") is not True:
                continue
            concern = label.get("concern")
            outcome = label.get("outcome")
            if not concern or outcome not in {"helped", "worsened", "unclear"}:
                continue
            global_counts[concern][outcome] += 1
            product_counts[(product_id, concern)][outcome] += 1
            skin_counts[(product_id, concern, skin_type)][outcome] += 1

    priors = {
        concern: counts["helped"] / (counts["helped"] + counts["worsened"])
        for concern, counts in global_counts.items()
        if counts["helped"] + counts["worsened"]
    }
    products: dict[str, dict] = {}
    for (product_id, concern), counts in sorted(product_counts.items()):
        if concern not in priors:
            continue
        all_cell = _cell(counts, priors[concern], smoothing_m)
        by_skin_type = {}
        for (pid, cell_concern, skin_type), skin in sorted(skin_counts.items()):
            if pid != product_id or cell_concern != concern:
                continue
            cell = _cell(skin, priors[concern], smoothing_m)
            if cell["n"] >= sub_cell_min_n:
                by_skin_type[skin_type] = cell
        products.setdefault(product_id, {})[concern] = {
            "all": all_cell,
            "by_skin_type": by_skin_type,
        }

    acne_n15 = sum(
        any(concern in ACNE_CONCERNS and entry["all"]["n"] >= 15
            for concern, entry in concerns.items())
        for concerns in products.values()
    )
    payload = {
        "schema_version": STORE_SCHEMA_VERSION,
        "kind": "concern_efficacy",
        "version": "v1",
        "prompt_version": PROMPT_VERSION,
        "smoothing_m": smoothing_m,
        "sub_cell_min_n": sub_cell_min_n,
        "confidence_n": 20,
        "priors": dict(sorted((key, round(value, 6)) for key, value in priors.items())),
        "products": products,
    }
    if p3 is not None:
        payload["p3"] = p3
    write_json(out_path, payload)
    coverage = {
        "products": len(products),
        "catalog_products": catalog_products,
        "products_with_acne_cell_n15": acne_n15,
    }
    if p3 is not None:
        coverage["p3_gate_passed"] = True
    register_store(
        data_root, name="concern_efficacy", kind="concern_efficacy", version="v1",
        store_path=out_path, builder=BUILDER,
        source={
            "labels_sha256": sha256_file(labels_path),
            "prompt_version": PROMPT_VERSION,
            **({"p3": p3} if p3 is not None else {}),
        },
        coverage=coverage,
    )
    return coverage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--smoothing-m", type=float, default=20)
    parser.add_argument("--sub-cell-min-n", type=int, default=5)
    parser.add_argument("--p3-eval", type=Path,
                        help="optional D-023 P3 evaluation JSON; otherwise derive it when metadata is present")
    args = parser.parse_args(argv)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_product_ids = frozenset(
        str(product["product_id"])
        for product in catalog.get("products") or []
        if product.get("product_id") is not None
    )
    coverage = build(
        args.labels, args.out, args.data_root,
        catalog_products=len(catalog_product_ids),
        catalog_product_ids=catalog_product_ids,
        smoothing_m=args.smoothing_m,
        sub_cell_min_n=args.sub_cell_min_n,
        p3_evaluation=(json.loads(args.p3_eval.read_text(encoding="utf-8"))
                       if args.p3_eval else None),
    )
    print(coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
