"""Learned re-ranker — the ML layer the engine's ranker hook consumes (D-005).

The rules engine (engine.py) NEVER imports ML; it exposes a duck-typed hook
(`recommend(..., ranker=...)`) that reorders rule-approved candidates by
`ranker.score(product, profile)` WITHIN the comedogenic partition. This module
builds that ranker:

  1. a training CLI that turns the Sephora reviews into (a) a satisfaction model
     (HistGradientBoostingClassifier on is_recommended) and (b) a per-product x
     skin-type review-stats table, and
  2. an inference `Ranker` the hook consumes, plus `load_ranker()` — a three-way
     loader (D-022 as amended 2026-07-10): model present -> Ranker; model absent
     but review-stats present -> StatsRanker (the statistical champion); both
     absent -> None, rules-only (D-019).

Per D-022 the model must EARN its place: it ships only if it beats a
global-popularity baseline AND a Bayesian-smoothed-rating baseline on both
pooled ROC-AUC and within-reviewer pairwise ordering, with metrics
disaggregated by skin-tone bucket ('unknown' is never dropped). A model that
fails the gate is never written to the model path.

Anti-skew (D-015): ONE feature path (product_features / reviewer_features) builds
columns for BOTH training and inference; the joblib bundle carries
feature_columns / brand_vocab / active_vocab so inference reconstructs the exact
training columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from ..config import load_config
from .import_catalog import CANONICAL_IDS, load_catalog

# closed active vocabulary, from the catalog importer's canonical id set.
ACTIVE_VOCAB = sorted(CANONICAL_IDS)

# categorical feature columns (native HGB categoricals via categorical_features="from_dtype").
CATEGORICAL_COLS = ["f_category", "f_brand", "f_skin_type", "f_tone_bucket"]

# fixed feature order — the bundle stores this; inference rebuilds columns in it.
FEATURE_COLUMNS = (
    [f"active__{a}" for a in ACTIVE_VOCAB]
    + ["f_category", "f_brand", "f_price"]
    + ["f_skin_type", "f_tone_bucket"]
)

METHODS = ["model", "popularity", "bayesian", "blended"]


def loves_nudges(loves, popularity_weight) -> dict:
    """{product_id: w * log1p(loves)/log1p(max_loves)} (D-028); empty when
    there's no loves data or the knob is off."""
    if not loves:
        return {}
    denom = math.log1p(max(loves.values()))
    if denom <= 0:
        return {}
    return {pid: popularity_weight * math.log1p(c) / denom
            for pid, c in loves.items()}
TONE_ROWS = ["light", "medium", "deep", "unknown"]  # 'unknown' never dropped (D-022)


# --- 3a. shared feature builders (anti-skew: one path, train + inference) ----
def product_features(product, brand_vocab) -> dict:
    """Product-side features. Missing price is OK per-row; an ENTIRELY-missing
    training price column crashes HistGradientBoostingClassifier 1.9 (handled in
    assemble_frame by coercing f_price to float64)."""
    feats = {f"active__{a}": int(a in product.actives) for a in ACTIVE_VOCAB}
    feats["f_category"] = product.category
    feats["f_brand"] = product.brand if product.brand in brand_vocab else "other"
    feats["f_price"] = product.price_usd
    return feats


def reviewer_features(skin_type, tone_bucket) -> dict:
    """Reviewer-side features. tone_bucket is ALREADY a bucket: training passes
    sephora_tone_bucket(raw), inference passes profile.tone_bucket."""
    return {
        "f_skin_type": skin_type or "unknown",
        "f_tone_bucket": tone_bucket or "unknown",
    }


def assemble_frame(records, feature_columns) -> pd.DataFrame:
    """Build the model frame in fixed column order. f_price -> float64 (all-None
    infers object dtype under pandas 3.0 and breaks HGB); CATEGORICAL_COLS ->
    category dtype so categorical_features='from_dtype' auto-detects them.

    `records` may be a DataFrame (bulk training/eval) or a list of dicts (single
    row inference); either way the output columns are exactly feature_columns.
    """
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    df = df.reindex(columns=feature_columns).copy()
    if "f_price" in df.columns:
        df["f_price"] = df["f_price"].astype("float64")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# --- 3b. loading + aggregation ---------------------------------------------
def load_reviews(reviews_dir) -> pd.DataFrame:
    """Concat reviews_*.csv (or a single file) reading only the six columns we
    use; drop unlabeled rows; add label + f_tone_bucket. Guards the three
    real-data landmines: string author/product ids (md5 needs .encode()), NaN
    skin_tone (sephora_tone_bucket(nan) raises), NaN skin_type (must not leak
    into f_skin_type / stats keys / the 'unknown never dropped' guarantee)."""
    from ..pipeline.tone import sephora_tone_bucket  # lazy: keeps matplotlib off the inference path

    path = Path(reviews_dir)
    files = sorted(path.glob("reviews_*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no reviews_*.csv under {path}")

    usecols = ["author_id", "rating", "is_recommended", "skin_tone", "skin_type", "product_id"]
    frames = [
        pd.read_csv(f, usecols=usecols, dtype={"author_id": str, "product_id": str})
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)

    df["is_recommended"] = pd.to_numeric(df["is_recommended"], errors="coerce")
    df = df.dropna(subset=["is_recommended"]).copy()
    df["label"] = df["is_recommended"].astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["skin_tone"] = df["skin_tone"].fillna("")
    df["f_tone_bucket"] = df["skin_tone"].map(sephora_tone_bucket)
    df["skin_type"] = df["skin_type"].fillna("unknown")
    return df


def brand_vocabulary(reviews_df, catalog_by_id, top_n) -> set:
    """Top-N brands by REVIEW count (brands resolved through the catalog)."""
    brands = reviews_df["product_id"].map(
        lambda pid: catalog_by_id[pid].brand if pid in catalog_by_id else None
    )
    counts = brands.dropna().value_counts()
    return set(counts.head(top_n).index)


def deterministic_test_mask(author_ids, test_fraction) -> "pd.Series":
    """Reviewer-disjoint split via a STABLE hash (md5, not builtin hash()): same
    author -> same side across processes, no state file."""
    cutoff = int(test_fraction * 1000)

    def in_test(aid: str) -> bool:
        digest = hashlib.md5(str(aid).encode()).hexdigest()
        return (int(digest, 16) % 1000) < cutoff

    return author_ids.map(in_test)


# --- 3c. model + baselines --------------------------------------------------
def train_model(X_train, y_train):
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", class_weight="balanced", random_state=0
    ).fit(X_train, y_train)


def popularity_baseline(train_df) -> dict:
    """product_id -> mean(label) on train (global recommend rate per product)."""
    return train_df.groupby("product_id")["label"].mean().to_dict()


def bayesian_baseline(train_df, m, global_mean_rating) -> dict:
    """product_id -> (sum_rating + m*global_mean_rating)/(n+m): mean rating shrunk
    toward the global prior by m pseudo-reviews."""
    grouped = train_df.groupby("product_id")["rating"].agg(["sum", "count"])
    return {
        pid: (row["sum"] + m * global_mean_rating) / (row["count"] + m)
        for pid, row in grouped.iterrows()
    }


# --- 3d. eval (the D-022 gate) ---------------------------------------------
def roc_auc(labels, scores) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pairwise_ordering_accuracy(df) -> float:
    """Over reviewers with BOTH labels: fraction of (pos, neg) score pairs with
    score(pos) > score(neg); ties count 0.5; nan if no mixed-label reviewer."""
    total = 0.0
    hits = 0.0
    for _aid, grp in df.groupby("author_id"):
        pos = grp.loc[grp["label"] == 1, "score"].to_numpy()
        neg = grp.loc[grp["label"] == 0, "score"].to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        for ps in pos:
            for ns in neg:
                total += 1.0
                if ps > ns:
                    hits += 1.0
                elif ps == ns:
                    hits += 0.5
    if total == 0.0:
        return float("nan")
    return hits / total


def _score_methods(train_df, test_df, model, feature_columns, bayes_m,
                   loves=None, popularity_weight=0.2) -> pd.DataFrame:
    """One scored frame: author_id, label, f_tone_bucket + a column per method."""
    global_pop = float(train_df["label"].mean())
    global_rating = float(train_df["rating"].mean())
    pop = popularity_baseline(train_df)
    bayes = bayesian_baseline(train_df, bayes_m, global_rating)
    nudges = loves_nudges(loves, popularity_weight)

    X_test = assemble_frame(test_df, feature_columns)
    model_scores = model.predict_proba(X_test)[:, 1]
    pids = test_df["product_id"]
    bayes_scores = pids.map(bayes).fillna(global_rating).to_numpy()
    return pd.DataFrame(
        {
            "author_id": test_df["author_id"].to_numpy(),
            "label": test_df["label"].to_numpy(),
            "f_tone_bucket": test_df["f_tone_bucket"].to_numpy(),
            "model": model_scores,
            "popularity": pids.map(pop).fillna(global_pop).to_numpy(),
            "bayesian": bayes_scores,
            # D-028: what StatsRanker actually ships — bayesian + loves nudge.
            "blended": bayes_scores + pids.map(nudges).fillna(0.0).to_numpy(),
        }
    )


def _metrics(frame) -> dict:
    out = {}
    for method in METHODS:
        sub = frame[["author_id", "label", method]].rename(columns={method: "score"})
        out[method] = {
            "roc_auc": roc_auc(sub["label"].to_numpy(), sub["score"].to_numpy()),
            "pairwise": pairwise_ordering_accuracy(sub),
        }
    return out


def evaluate(train_df, test_df, model, feature_columns, low_n_floor,
             *, bayes_m=20, reviews_dropped_no_catalog=0,
             loves=None, popularity_weight=0.2) -> dict:
    """Pooled + per-tone-bucket ROC-AUC and pairwise for model + both baselines.
    gate_passed = model beats BOTH baselines on BOTH pooled metrics (D-022);
    blended_gate_passed = the popularity blend costs <= 0.02 pooled pairwise
    vs bayesian (D-028 soft gate — the bias is a product choice, the harness
    only proves it isn't collapsing the ordering)."""
    scored = _score_methods(train_df, test_df, model, feature_columns, bayes_m,
                            loves=loves, popularity_weight=popularity_weight)
    pooled = _metrics(scored)

    by_tone = {}
    for bucket in TONE_ROWS:
        sub = scored[scored["f_tone_bucket"] == bucket]
        n = int(len(sub))
        row = {"n": n, "low_n": bool(n < low_n_floor)}
        row.update(_metrics(sub))
        by_tone[bucket] = row

    m_auc = pooled["model"]["roc_auc"]
    m_pw = pooled["model"]["pairwise"]
    gate = bool(
        m_auc > pooled["popularity"]["roc_auc"]
        and m_auc > pooled["bayesian"]["roc_auc"]
        and m_pw > pooled["popularity"]["pairwise"]
        and m_pw > pooled["bayesian"]["pairwise"]
    )
    return {
        "base_rate": float(train_df["label"].mean()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "reviews_dropped_no_catalog": int(reviews_dropped_no_catalog),
        "pooled": pooled,
        "by_tone": by_tone,
        "gate_passed": gate,
        "popularity_weight": float(popularity_weight),
        "blended_gate_passed": bool(
            pooled["blended"]["pairwise"]
            >= pooled["bayesian"]["pairwise"] - 0.02
        ),
    }


# --- 3e. review-stats artifact (train rows only) ---------------------------
def _cell_stats(grp) -> dict:
    return {
        "n": int(len(grp)),
        "mean_rating": float(grp["rating"].mean()),
        "pct_recommend": float(grp["label"].mean()),
    }


def build_loves_map(product_info_path, catalog_ids) -> "dict | None":
    """{product_id: loves_count} for catalog products (D-028); None when the
    product-info file is absent — the popularity nudge degrades to nothing."""
    path = Path(product_info_path) if product_info_path else None
    if path is None or not path.exists():
        return None
    info = pd.read_csv(path, usecols=["product_id", "loves_count"])
    info = info[info["product_id"].isin(catalog_ids)].dropna()
    return {str(pid): int(c) for pid, c in
            zip(info["product_id"], info["loves_count"])}


def build_review_stats(train_df, min_cell_size) -> dict:
    """{min_cell_size, base_rate, cells: {product_id: {'__all__': {...},
    '<skin_type>': {...}, ...}}} — the per-product x skin-type evidence table."""
    cells = {}
    for pid, grp in train_df.groupby("product_id"):
        cell = {"__all__": _cell_stats(grp)}
        for skin_type, sub in grp.groupby("skin_type"):
            cell[str(skin_type)] = _cell_stats(sub)
        cells[str(pid)] = cell
    return {
        "min_cell_size": int(min_cell_size),
        "base_rate": float(train_df["label"].mean()),
        "global_mean_rating": float(train_df["rating"].mean()),
        "cells": cells,
    }


# --- 3f. artifact I/O + inference class ------------------------------------
def save_bundle(path, model, brand_vocab, feature_columns, base_rate) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "brand_vocab": sorted(brand_vocab),
            "active_vocab": ACTIVE_VOCAB,
            "feature_columns": list(feature_columns),
            "base_rate": float(base_rate),
        },
        path,
    )


def evidence_cell(cells, min_cell_size, product_id, skin_type):
    """The report's per-product 'why' cell (shared by Ranker and StatsRanker):
    the product x skin_type cell when its n >= min_cell_size, else the '__all__'
    cell tagged fallback; None if the product is absent."""
    cell = cells.get(product_id)
    if cell is None:
        return None
    typed = cell.get(skin_type)
    if typed is not None and typed.get("n", 0) >= min_cell_size:
        return {**typed, "fallback": False, "cell": skin_type}
    all_cell = cell.get("__all__")
    if all_cell is None:
        return None
    return {**all_cell, "fallback": True, "cell": "all_reviewers"}


class Ranker:
    """Inference ranker consumed by the engine hook (D-005). score() returns the
    predicted P(is_recommended=1); HIGHER = better fit (the engine negates it)."""

    def __init__(self, bundle, stats, min_cell_size):
        self.bundle = bundle
        self.model = bundle["model"]
        self.brand_vocab = set(bundle.get("brand_vocab", []))
        self.feature_columns = bundle["feature_columns"]
        self.base_rate = bundle.get("base_rate", 0.5)
        self.stats = stats or {}
        self.cells = self.stats.get("cells", {})
        self.min_cell_size = min_cell_size
        # ponytail: per-(product,profile) memo dict; swap for a real batched
        # predict only if scoring shows up in a profile.
        self._memo: dict = {}

    def score(self, product, profile) -> float:
        skin_type = profile.skin_type if profile is not None else None
        tone_bucket = profile.tone_bucket if profile is not None else None
        key = (product.product_id, skin_type, tone_bucket)
        if key in self._memo:
            return self._memo[key]
        feats = product_features(product, self.brand_vocab)
        feats.update(reviewer_features(skin_type, tone_bucket))
        X = assemble_frame([feats], self.feature_columns)
        proba = float(self.model.predict_proba(X)[0, 1])
        self._memo[key] = proba
        return proba

    def evidence(self, product_id, skin_type) -> "dict | None":
        return evidence_cell(self.cells, self.min_cell_size, product_id, skin_type)


class StatsRanker:
    """The bake-off champion (D-022 amendment, 2026-07-10): orders candidates by
    the Bayesian-smoothed POOLED product rating from review_stats.json — the
    strongest orderer the review data supports (pairwise 0.609 on the D-022
    harness; see plans/ranker-v2-probe-evidence.md). Deliberately ignores the
    profile: per-skin-type cell ordering was measured and LOSES to pooled
    (0.606/0.596); skin-type cells stay evidence-only. No sklearn at inference.
    Same duck-typed hook as the learned Ranker; higher score = better."""

    def __init__(self, stats, m, min_cell_size, popularity_weight=0.2):
        self.cells = stats.get("cells", {})
        self.min_cell_size = min_cell_size
        prior = float(stats.get("global_mean_rating", 0.0))
        self._prior = prior
        self._scores = {
            pid: (cell["__all__"]["n"] * cell["__all__"]["mean_rating"] + m * prior)
                 / (cell["__all__"]["n"] + m)
            for pid, cell in self.cells.items()
            if cell.get("__all__")
        }
        # D-028: small deliberate popularity bias — w * log1p(loves) normalized
        # against the most-loved product. No loves map -> no nudge anywhere.
        self._nudge = loves_nudges(stats.get("loves"), popularity_weight)

    def score(self, product, profile) -> float:
        # profile intentionally unused (pooled beats per-profile; see class doc).
        base = self._scores.get(product.product_id, self._prior)
        return base + self._nudge.get(product.product_id, 0.0)

    def evidence(self, product_id, skin_type):
        return evidence_cell(self.cells, self.min_cell_size, product_id, skin_type)


def load_ranker(config=None) -> "Ranker | StatsRanker | None":
    """Three-way loader (D-022 as amended 2026-07-10):
    - learned model artifact present  -> Ranker (a model exists only if it
      passed the ratcheted gate at training time);
    - model absent, review-stats present -> StatsRanker (the statistical
      champion: Bayesian-smoothed pooled product rating);
    - both absent -> None (rules-only catalog order, D-019)."""
    if config is None:
        config = load_config()
    rcfg = config["ranker"]

    stats_path = Path(rcfg["review_stats_path"])
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = None

    model_path = Path(rcfg["model_path"])
    if model_path.exists():
        bundle = joblib.load(model_path)
        return Ranker(bundle, stats or {}, rcfg.get("min_cell_size", 5))
    if stats is not None:
        return StatsRanker(stats, rcfg.get("bayesian_prior_count", 20),
                           rcfg.get("min_cell_size", 5),
                           popularity_weight=rcfg.get("popularity_weight", 0.2))
    return None


# --- 3g. training orchestration + CLI --------------------------------------
def _nan_to_none(obj):
    """json.dump emits a bare NaN token (invalid JSON) for empty-bucket cells;
    convert every NaN float to None first."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _fmt(value) -> str:
    return "  nan" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def _print_eval_table(result) -> None:
    print(
        f"base_rate={result['base_rate']:.3f}  n_train={result['n_train']}  "
        f"n_test={result['n_test']}  dropped_no_catalog={result['reviews_dropped_no_catalog']}"
    )
    header = f"{'scope':<10} {'method':<11} {'roc_auc':>8} {'pairwise':>9} {'n':>6} {'low_n':>6}"
    print(header)
    for method in METHODS:
        cell = result["pooled"][method]
        print(f"{'POOLED':<10} {method:<11} {_fmt(cell['roc_auc']):>8} "
              f"{_fmt(cell['pairwise']):>9} {result['n_test']:>6} {'':>6}")
    for bucket in TONE_ROWS:
        row = result["by_tone"][bucket]
        for method in METHODS:
            cell = row[method]
            print(f"{bucket:<10} {method:<11} {_fmt(cell['roc_auc']):>8} "
                  f"{_fmt(cell['pairwise']):>9} {row['n']:>6} {str(row['low_n']):>6}")


def train_pipeline(reviews_dir, catalog_path, out_model, out_stats, out_eval,
                   config, verbose=True, product_info_path=None) -> dict:
    """Full run: load reviews + catalog -> brand vocab -> reviewer-disjoint split
    -> features -> train -> baselines -> evaluate. ALWAYS writes review_stats +
    eval; writes the MODEL bundle ONLY when the D-022 gate passes."""
    rcfg = config["ranker"]

    reviews_df = load_reviews(reviews_dir)
    catalog = load_catalog(catalog_path)
    catalog_by_id = {p.product_id: p for p in catalog}
    brand_vocab = brand_vocabulary(reviews_df, catalog_by_id, rcfg["brand_top_n"])

    # product features computed ONCE per product, then merged (not per review row).
    prod_feats = {p.product_id: product_features(p, brand_vocab) for p in catalog}
    prod_df = pd.DataFrame.from_dict(prod_feats, orient="index")

    n_reviews = len(reviews_df)
    merged = reviews_df.merge(prod_df, left_on="product_id", right_index=True, how="inner")
    reviews_dropped = n_reviews - len(merged)
    # bulk equivalent of reviewer_features(): load_reviews already filled NaN
    # skin_type with "unknown", so this column matches the inference path — if
    # reviewer_features() ever changes, change this line with it (anti-skew).
    merged["f_skin_type"] = merged["skin_type"]

    test_mask = deterministic_test_mask(merged["author_id"], rcfg["split_test_fraction"])
    train_df = merged[~test_mask].copy()
    test_df = merged[test_mask].copy()

    X_train = assemble_frame(train_df, FEATURE_COLUMNS)
    y_train = train_df["label"].to_numpy()
    model = train_model(X_train, y_train)

    loves = build_loves_map(product_info_path, set(catalog_by_id))
    result = evaluate(
        train_df, test_df, model, FEATURE_COLUMNS, rcfg["eval_low_n_floor"],
        bayes_m=rcfg["bayesian_prior_count"], reviews_dropped_no_catalog=reviews_dropped,
        loves=loves, popularity_weight=rcfg.get("popularity_weight", 0.2),
    )
    stats = build_review_stats(train_df, rcfg["min_cell_size"])
    if loves is not None:
        stats["loves"] = loves

    _write_json(out_stats, stats)
    _write_json(out_eval, _nan_to_none(result))

    if verbose:
        _print_eval_table(result)

    if result["gate_passed"]:
        save_bundle(out_model, model, brand_vocab, FEATURE_COLUMNS, result["base_rate"])
    elif verbose:
        print("gate FAILED — model NOT written; pipeline stays rules-only (D-022)")

    if verbose:
        print(f"gate_passed: {result['gate_passed']}")
    return result


def main(argv=None):
    config = load_config()
    rcfg = config["ranker"]
    paths = config["paths"]

    parser = argparse.ArgumentParser(
        description="Train the learned re-ranker (D-022): satisfaction model + "
                    "review-stats table, gated on beating popularity/rating baselines.",
    )
    parser.add_argument("--reviews-dir", default=paths["reviews_raw"],
                        help="dir of reviews_*.csv (default: paths.reviews_raw)")
    parser.add_argument("--catalog", default=paths["catalog_processed"],
                        help="catalog.json (default: paths.catalog_processed)")
    parser.add_argument("--out-model", default=rcfg["model_path"])
    parser.add_argument("--out-stats", default=rcfg["review_stats_path"])
    parser.add_argument("--out-eval", default=rcfg["eval_path"])
    parser.add_argument("--product-info", default=rcfg.get("product_info_path"),
                        help="product_info.csv with loves_count (D-028; "
                             "default: ranker.product_info_path)")
    args = parser.parse_args(argv)

    train_pipeline(args.reviews_dir, args.catalog, args.out_model,
                   args.out_stats, args.out_eval, config,
                   product_info_path=args.product_info)


if __name__ == "__main__":
    main()
