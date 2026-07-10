# Plan 013: Learned ranker — review aggregation, training CLI, baseline-gated eval, inference class

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat b197b6b..HEAD -- src/recommendation/ configs/default.yaml src/pipeline/tone.py`
> If any file below in "Current state" changed since this plan was written,
> compare the excerpts against the live code before proceeding; on a mismatch,
> treat it as a STOP condition.

## Status

- **Priority**: P1 (the ML layer of the milestone; GitHub issue #8)
- **Effort**: L
- **Risk**: MED — new sklearn training path; the empirical acceptance gate on real data is not guaranteed (see STOP conditions).
- **Depends on**: #3 (catalog with preserved Sephora ids — LANDED), #6 (review-tone→bucket mapping — LANDED), #7 (UserProfile + ranker hook — LANDED). All three are already on `main` at the planned-at SHA; no plan-file dependency.
- **Category**: direction (new capability) + tests
- **Planned at**: commit `b197b6b`, 2026-07-10
- **Issue**: Kumario1/skinscan#8 (parent: #1)

## Why this matters

The engine already exposes a ranker hook (`recommend(..., ranker=...)`) that reorders rule-approved candidates by `ranker.score(product, profile)`, but nothing implements that hook — today the app can only run rules-only. This plan builds the learned layer: a training CLI that turns the ~1.1M Sephora reviews into (a) a satisfaction model and (b) a per-product×skin-type review-stats table, plus an inference class the hook consumes. Per D-022 the model **must earn its place** — it ships only if it beats a global-popularity baseline and a Bayesian-smoothed-rating baseline on both ROC-AUC and within-reviewer pairwise ordering, with metrics disaggregated by skin-tone bucket (including `unknown`, never dropped). When the artifacts are missing the pipeline degrades cleanly to rules-only. After this lands, the recommender is genuinely hybrid and the report (a later issue) has real review evidence to cite.

## Current state

The seams this plugs into (read these before writing code):

- `src/recommendation/engine.py` — the rules engine. It **never imports ML**; the ranker is injected. The only contract you must satisfy is the sort key (lines 216–220):
  ```python
  def sort_key(p: Product):
      # comedogenic partition ALWAYS dominates; ranker only breaks ties within.
      if ranker is not None:
          return (len(p.comedogenic_flags), -ranker.score(p, profile))
      return (len(p.comedogenic_flags),)
  ```
  So `ranker.score(product, profile)` returns a float where **higher = better fit** (the engine negates it). `ranker=None` → today's exact rules-only order. `profile` may be `None` (the engine's signature is `recommend(report, catalog, profile=None, ranker=None, ...)`, line 77–79) — your `score` must tolerate `profile is None`. **Do not modify `engine.py`.**

- `tests/test_recommendation_engine.py` (lines 35–41) — the duck-typed contract the engine already tests against:
  ```python
  class StubRanker:
      """Duck-typed ranker (D-005): only reorders, never adds/removes/flags."""
      def __init__(self, scores): self.scores = scores
      def score(self, product, profile): return self.scores.get(product.product_id, 0.0)
  ```
  Your real `Ranker.score` must be drop-in compatible with this shape.

- `src/recommendation/schema.py` — the data types you consume (do not change them):
  ```python
  @dataclass
  class Product:
      product_id: str; name: str; brand: str; category: str
      actives: list[str] = ...; comedogenic_flags: list[str] = ...
      price_usd: Optional[float] = None; price_is_stale: bool = True
  SKIN_TYPES = {"combination", "dry", "normal", "oily"}
  TONE_BUCKETS = {"light", "medium", "deep"}   # plus "unknown" everywhere
  @dataclass
  class UserProfile:
      skin_type: str; tone_bucket: Optional[str] = None
      tone_source: str = "unknown"; pregnant_or_nursing: bool = False
  ```

- `src/recommendation/import_catalog.py` — provides `load_catalog(path) -> list[Product]`. The **Sephora product_id is preserved** through import (it is load-bearing for joining reviews); `product_from_row` honors a source-supplied `product_id`. So `catalog.json` products carry ids like `"P480274"` that join directly to `reviews.product_id`.

- `src/pipeline/tone.py` — issue #6 delivered `sephora_tone_bucket(value: str | None) -> str`, the single source that maps a raw review `skin_tone` string ("lightMedium", "mediumTan", …) to `light`/`medium`/`deep`/`unknown` via `configs/default.yaml` `tone.sephora_tone_buckets`. **Reuse it — never re-implement the mapping** (a copy would skew training against the report). Importing `tone` pulls in matplotlib; that is fine (matplotlib is a dependency and not in the forbidden ML-import set; `tests/test_tone.py` already imports this module).

- `configs/default.yaml` — the `ranker:` block already exists (keys reserved by issue #2, inert until now):
  ```yaml
  ranker:
    model_path: models/ranker/ranker.joblib
    review_stats_path: data/processed/review_stats.json
    min_cell_size: 5
  paths:
    catalog_processed: data/processed/catalog.json
    reviews_raw: data/raw/sephora       # reviews_*.csv live here alongside product_info.csv
  ```

- The **real reviews CSVs**: `data/raw/sephora/reviews_*.csv` (five files, ~1.1M rows total). Header (verified 2026-07-10) — note the **leading empty-name index column**:
  ```
  ,author_id,rating,is_recommended,helpfulness,...,skin_tone,eye_color,skin_type,hair_color,product_id,product_name,brand_name,price_usd
  ```
  The only columns you read: `author_id`, `rating`, `is_recommended`, `skin_tone`, `skin_type`, `product_id`. `is_recommended` is a float-string (`"1.0"`/`"0.0"`) and is **empty for unlabeled rows** (drop those). `skin_type` values already match `SKIN_TYPES`. `product_id` joins to the catalog. Read by column name (`usecols=`) so the empty-name index column is ignored.

Repo conventions to match:
- One module per concern, with an argparse `main()` at the bottom (see `import_catalog.py`, `tone.py`). Docstring names the decision it implements.
- Config is the single source of knobs; `from ..config import load_config`. CLI flags override config defaults.
- Tests are pure-Python, no TF/YOLO/mediapipe, and runnable standalone via a `__main__` block that calls each `test_*` then prints `ok` (see the bottom of `tests/test_import_sephora.py`). **sklearn/pandas/joblib ARE allowed in the default suite** — they are installed dependencies, not the forbidden heavy ML imports.
- Deterministic → idempotent (the importer is; the ranker must be too: a fixed hash split, no RNG state file).

Dependencies confirmed installed in `.venv`: scikit-learn 1.9.0, joblib 1.5.3, pandas 3.0.3, numpy. In sklearn 1.9, `HistGradientBoostingClassifier` supports `class_weight="balanced"` and `categorical_features="from_dtype"` (the default) — pandas `category`-dtype columns are auto-detected as categorical.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Ranker tests | `.venv/bin/python -m pytest tests/test_ranker.py` | all pass |
| Full fast suite | `.venv/bin/python -m pytest` | all pass (no new failures) |
| Standalone test run | `.venv/bin/python tests/test_ranker.py` | prints `ok` |
| Build catalog (prereq for real run) | `.venv/bin/python -m src.recommendation.import_catalog --csv data/raw/sephora/product_info.csv --format sephora` | writes `data/processed/catalog.json`, prints a log dict |
| Real training run | `.venv/bin/python -m src.recommendation.ranker` | writes model/stats/eval, prints eval table, `gate_passed` line |

## Suggested executor toolkit

- If available, use `superpowers:test-driven-development` — write `tests/test_ranker.py` against the fixture first, then implement until green.
- Structural patterns to copy: `tests/test_import_sephora.py` (tempfile end-to-end over a committed fixture CSV, `__main__` block) and `tests/test_recommendation_engine.py` (the `StubRanker` / `recommend` seam).

## Scope

**In scope** (the only files you create or modify):
- `src/recommendation/ranker.py` — **create**. Training CLI + baselines + eval + review-stats + `Ranker` inference class + `load_ranker`.
- `tests/test_ranker.py` — **create**.
- `tests/fixtures/reviews_sample.csv` — **create** (committed fixture engineered so the model beats popularity).
- `configs/default.yaml` — **edit**: add the new `ranker:` keys listed in Step 1 (nothing else).
- `README.md` — **edit**: add the ranker train command under "## Run it" (Step 7).
- `docs/CATALOG_SCHEMA.md` — **edit**: append a short "Review-stats & ranker artifacts" note (Step 7).
- `plans/README.md` — **edit**: status row + a new-section note (final step).

**Out of scope** (do NOT touch, even though they look related):
- `src/recommendation/engine.py` — the hook already exists; changing it breaks the D-005 trust contract and the existing engine tests. You only *consume* the hook.
- `src/recommendation/schema.py` — `Product`/`UserProfile`/`Concern` are locked contracts (D-008/D-009/D-021).
- `src/pipeline/tone.py` — reuse `sephora_tone_bucket`, do not edit it.
- The Streamlit app, the report/`analyze()`/`report.json` renderer, and any CLI that ties photo→report together — those are **later issues**. This plan delivers the ranker and its artifacts only; the engine-consumption test is the extent of the wiring here.
- `docs/DECISIONS.md` — D-022 already specifies the acceptance criteria; you are *implementing* it, not amending it. No dated note needed.

## Git workflow

- Branch off `main`: `git switch -c issue-8-learned-ranker` (observed conventions vary: `issue-N-*`, `codex/issue-N-*`, `worktree-issue-N-*` — any is fine; if a reviewer dispatched you into a worktree, work there).
- Commit per logical unit; message style is conventional-commits with the issue number, e.g. `feat: learned ranker — training CLI + baseline-gated eval + inference class (#8)` (match `git log`: `feat: engine v2 — ... (#7)`).
- Do NOT push or open a PR unless the operator instructed it. Do NOT merge to `main`.

## Steps

### Step 1: Add the consumed config keys

Edit the existing `ranker:` block in `configs/default.yaml` to read exactly:

```yaml
ranker:                             # learned re-ranker artifacts (D-022); engine runs rules-only if absent (D-019)
  model_path: models/ranker/ranker.joblib             # HistGradientBoostingClassifier bundle (joblib)
  review_stats_path: data/processed/review_stats.json  # per-product x skin-type review stats
  eval_path: runs/ranker/eval.json    # disaggregated eval artifact (mirrors the printed table)
  min_cell_size: 5                    # min reviews per product x skin-type cell before "all reviewers" fallback
  brand_top_n: 50                     # keep the top-N brands by review count as features; rest -> "other"
  bayesian_prior_count: 20            # m in the Bayesian-smoothed mean-rating baseline (global prior)
  split_test_fraction: 0.25           # reviewer-disjoint deterministic hash split -> test fraction
  eval_low_n_floor: 30                # per-tone-bucket test N below this -> low_n flag on that row
```

Every key is consumed in this plan (none speculative). Leave `paths:`, `tone:`, everything else untouched.

**Verify**: `.venv/bin/python -c "from src.config import load_config; r=load_config()['ranker']; print(sorted(r))"` →
`['bayesian_prior_count', 'brand_top_n', 'eval_low_n_floor', 'eval_path', 'min_cell_size', 'model_path', 'review_stats_path', 'split_test_fraction']`

### Step 2: Build the committed fixture reviews CSV

Create `tests/fixtures/reviews_sample.csv` engineered so a model using the **reviewer's skin_type × product interaction** beats a product-only popularity/rating baseline on a held-out reviewer split. The construction (deterministic, no RNG):

- Two products, both with an overall recommend rate of 0.5, so a product-only baseline carries no reliable signal about *which* product a given reviewer prefers: `PA` (an oily-skin favorite) and `PB` (a dry-skin favorite). (After the reviewer-disjoint split the per-product train rates drift off 0.5 and anti-correlate with the test side, so the baselines land *well below* the model — ~0.33 pooled AUC/pairwise in practice, not a 0.5 tie. Don't expect exactly 0.5; expect the model to win by a wide margin.)
- 60 reviewers `a00`…`a59`. Even index → `skin_type=oily`, odd index → `skin_type=dry`. Cycle `skin_tone` raw values through `["fair", "medium", "deep"]` so all buckets appear.
- Every reviewer reviews **both** products (guarantees mixed labels per reviewer → pairwise ordering is computable for each test reviewer).
- Labels encode the separable signal:
  - oily reviewer: `PA` → `is_recommended=1, rating=5`; `PB` → `is_recommended=0, rating=2`
  - dry reviewer: `PA` → `is_recommended=0, rating=2`; `PB` → `is_recommended=1, rating=5`

Generate it once with this snippet (run it, then **commit the produced CSV**; the snippet is throwaway):

```python
import csv
tones = ["fair", "medium", "deep"]
rows = []
for i in range(60):
    author = f"a{i:02d}"
    skin_type = "oily" if i % 2 == 0 else "dry"
    tone = tones[i % 3]
    if skin_type == "oily":
        pa, pb = (1, 5), (0, 2)
    else:
        pa, pb = (0, 2), (1, 5)
    rows.append([author, pa[1], float(pa[0]), tone, skin_type, "PA"])
    rows.append([author, pb[1], float(pb[0]), tone, skin_type, "PB"])
with open("tests/fixtures/reviews_sample.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["author_id", "rating", "is_recommended", "skin_tone", "skin_type", "product_id"])
    w.writerows(rows)
```

The test (Step 4) builds the matching catalog in-code. **Both products are category `treatment`** (so they compete in one routine step — a `treatment` and a `moisturizer` never compete, which would make test 4 vacuous) and **both carry an explicit `price_usd`** (see the crash in the note below):
```python
Product("PA", "SA Treatment", "BrandX", "treatment", actives=["salicylic_acid"], price_usd=24.0)
Product("PB", "Ceramide Treatment", "BrandY", "treatment", actives=["ceramides"], price_usd=18.0)
```

> **Landmine (verified on sklearn 1.9.0):** if a training feature column is *entirely* NaN, `HistGradientBoostingClassifier.fit` raises `ValueError: window shape cannot be larger than input array shape` (the binner chokes on an empty distinct-value set). Products default `price_usd=None`, so a two-product fixture with no prices makes `f_price` all-NaN and training crashes **before the gate** — the plan's STOP condition would then misfire. Giving the fixture products real prices avoids it. Partial NaN (some products priced, some not) is fine — that is the real-catalog case. Because of this, `assemble_frame` must coerce `f_price` to float (`.astype("float64")`) — all-`None` records infer `object` dtype under pandas 3.0 and would break HGB.

**Verify**: `.venv/bin/python -c "import csv; rows=list(csv.DictReader(open('tests/fixtures/reviews_sample.csv'))); print(len(rows), rows[0])"` → `120 {'author_id': 'a00', 'rating': '5', 'is_recommended': '1.0', 'skin_tone': 'fair', 'skin_type': 'oily', 'product_id': 'PA'}`

### Step 3: Implement `src/recommendation/ranker.py`

One module. Top-of-file docstring naming D-005/D-022/D-015. Structure it as these units (signatures are the contract; keep them):

**3a. Feature vocab + the shared feature builder (the anti-skew guarantee).** ONE code path builds product/reviewer features for both training and inference.

```python
# canonical active vocabulary = the catalog's active IDs, sorted & fixed
from .import_catalog import CANONICAL_IDS
ACTIVE_VOCAB = sorted(CANONICAL_IDS)          # ~30 stable columns

def product_features(product, brand_vocab) -> dict:
    """Product side: active multi-hot + category + brand(top-N or 'other') + price.
    `product` is anything with .actives/.category/.brand/.price_usd (a Product,
    or a namedtuple built from a catalog row)."""
    feats = {f"active__{a}": int(a in product.actives) for a in ACTIVE_VOCAB}
    feats["f_category"] = product.category
    feats["f_brand"] = product.brand if product.brand in brand_vocab else "other"
    feats["f_price"] = product.price_usd            # missing is fine PER-ROW; an ENTIRELY-missing
    return feats                                    # training column crashes HGB 1.9 (see Step 2 landmine)

def reviewer_features(skin_type, tone_bucket) -> dict:
    """Reviewer side. tone_bucket is ALREADY a bucket (light/medium/deep/unknown).
    Training passes sephora_tone_bucket(raw_skin_tone); inference passes
    profile.tone_bucket. skin_type None/unknown -> the string 'unknown'."""
    return {"f_skin_type": skin_type or "unknown",
            "f_tone_bucket": tone_bucket or "unknown"}

CATEGORICAL_COLS = ["f_category", "f_brand", "f_skin_type", "f_tone_bucket"]

def assemble_frame(records: list[dict], feature_columns: list[str]) -> "pd.DataFrame":
    """Records -> DataFrame with fixed column order, `f_price` coerced to float64
    (all-None records infer object dtype under pandas 3.0 and break HGB), and
    category dtypes set on CATEGORICAL_COLS via .astype('category') (so sklearn's
    categorical_features='from_dtype' picks them up)."""
```

**3b. Loading + aggregation.**
```python
def load_reviews(reviews_dir) -> "pd.DataFrame":
    """Concat reviews_*.csv reading only [author_id,rating,is_recommended,
    skin_tone,skin_type,product_id]; coerce is_recommended to numeric, DROP rows
    where it is NaN; add label=int(is_recommended), f_tone_bucket=
    skin_tone.map(sephora_tone_bucket). Return the frame.

    THREE real-data landmines (all verified crashing on data/raw/sephora):
    - read with dtype={"author_id": str, "product_id": str}. Numeric author ids
      otherwise come back as Python int, and the md5 split's id.encode() raises
      AttributeError: 'int' object has no attribute 'encode'.
    - skin_tone has ~100k NaN per file; sephora_tone_bucket(nan) raises
      (NaN is truthy, so `value or ""` doesn't catch it). Do
      df["skin_tone"] = df["skin_tone"].fillna("") BEFORE .map(sephora_tone_bucket).
    - skin_type has ~75k NaN; do df["skin_type"] = df["skin_type"].fillna("unknown")
      so NaN never leaks into f_skin_type, the review-stats keys, or the
      "unknown never dropped" guarantee."""

def brand_vocabulary(reviews_df, catalog_by_id, top_n) -> set:
    """Top-N brands by REVIEW count (join count of catalog brand over reviews)."""

def deterministic_test_mask(author_ids, test_fraction) -> "pd.Series[bool]":
    """Reviewer-disjoint split via a STABLE hash (hashlib.md5, NOT builtin hash()
    which is per-process salted): (int(md5(id).hexdigest(),16) % 1000)
    < test_fraction*1000. Same author -> same side, no state file."""
```
Join reviews to the catalog on `product_id` (inner join — reviews for dropped-category products are discarded; **count and report them** in the eval log). Build `X` via `product_features`+`reviewer_features` per row (product features computed once per product then merged; do not call it 1.1M times), set `feature_columns = [active__* ...] + ["f_category","f_brand","f_price"] + ["f_skin_type","f_tone_bucket"]`.

**3c. Model + baselines.**
```python
def train_model(X_train, y_train):
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", class_weight="balanced",
        random_state=0).fit(X_train, y_train)

def popularity_baseline(train_df) -> dict:        # product_id -> mean(label) on train
def bayesian_baseline(train_df, m, global_mean_rating) -> dict:
    # product_id -> (sum_rating + m*global_mean_rating) / (n + m)
```
For a test row, each baseline's score = the product's train scalar (fallback to the global mean for products unseen in train). Model score = `predict_proba(X_test)[:, 1]`. (Issue #8 phrases the review-stats artifact as *feeding* the Bayesian baseline; computing `bayesian_baseline` directly from `train_df` uses the same train rows and is functionally identical — either derive it from `build_review_stats`' `__all__` cells or keep this direct form; both are acceptable, don't build two divergent code paths.)

**3d. Eval (the D-022 gate).**
```python
def roc_auc(labels, scores) -> float           # sklearn.metrics.roc_auc_score; nan if one class
def pairwise_ordering_accuracy(df) -> float:
    """Over reviewers with BOTH labels present: fraction of (pos,neg) score pairs
    with score(pos) > score(neg); ties count 0.5. This mirrors the real job of
    reordering one person's candidate list. nan if no mixed-label reviewers."""
def evaluate(train_df, test_df, model, feature_columns, low_n_floor) -> dict:
    """Compute, for model + popularity + bayesian: pooled ROC-AUC and pairwise
    accuracy, AND per tone bucket (group test reviewers by f_tone_bucket, include
    'unknown', never drop; low_n flag when bucket test-N < low_n_floor). Return a
    dict with base_rate, n_train, n_test, reviews_dropped_no_catalog, pooled{},
    by_tone{}, and gate_passed."""
```
`gate_passed = (model pooled ROC-AUC > both baselines) AND (model pooled pairwise > both baselines)`. Print a readable table (tone buckets as rows/sections; model/pop/bayes columns; N and low_n shown) and a final `gate_passed: True/False` line.

When serializing `eval.json`, convert every NaN metric (empty-bucket cells) to `None` — `json.dump` otherwise emits a bare `NaN` token, which is invalid JSON and breaks any strict reader. (A small `nan -> None` sanitize pass over the dict before `json.dump`.)

**3e. Review-stats artifact** (train rows only):
```python
def build_review_stats(train_df, min_cell_size) -> dict:
    """{"min_cell_size": m, "base_rate": ..., "cells": {product_id: {
        "__all__": {"n":, "mean_rating":, "pct_recommend":},
        "<skin_type>": {...}, ...}}}. Feeds the Bayesian baseline's rating stats
    AND the report's evidence lines (later issue)."""
```

**3f. Artifact I/O + inference class.**
```python
def save_bundle(path, model, brand_vocab, feature_columns, base_rate):
    joblib.dump({"model": model, "brand_vocab": sorted(brand_vocab),
                 "active_vocab": ACTIVE_VOCAB, "feature_columns": feature_columns,
                 "base_rate": base_rate}, path)

class Ranker:
    def __init__(self, bundle, stats, min_cell_size): ...
    def score(self, product, profile) -> float:
        """predict_proba of is_recommended=1 for this product under this profile.
        Higher = better (engine negates it). profile may be None -> skin_type/tone
        'unknown'. Build a 1-row frame with self.bundle['feature_columns'] via the
        SAME product_features/reviewer_features path, in self.bundle order."""
    def evidence(self, product_id, skin_type) -> dict | None:
        """The report's per-product 'why' cell. Return the product x skin_type
        cell when its n >= min_cell_size, else the '__all__' cell with
        {'fallback': True, 'cell': 'all_reviewers'}; None if the product is absent."""

def load_ranker(config=None) -> "Ranker | None":
    """Load model bundle + review stats from config paths. Return None when the
    MODEL artifact is missing (the app passes this straight to recommend(); None
    -> exact rules-only order, the D-019 degradation). Load stats if present, else
    the Ranker carries empty stats (evidence() returns None)."""
```
Give `score` a tiny memo (`dict` keyed `(product_id, skin_type, tone_bucket)`) to satisfy "batched internally" cheaply.
`// ponytail: per-(product,profile) memo dict; swap for a real batched predict only if scoring shows up in a profile.`

**3g. `main(argv=None)`** — the training CLI. Flags with config defaults:
`--reviews-dir` (default `paths.reviews_raw`), `--catalog` (default `paths.catalog_processed`), `--out-model` (`ranker.model_path`), `--out-stats` (`ranker.review_stats_path`), `--out-eval` (`ranker.eval_path`). It runs the whole chain: load reviews → load catalog → brand vocab → split → build features → train → baselines → evaluate (print table).

**Then write artifacts with the D-022 gate as the guard:** always write `review_stats.json` and `eval.json` (creating parent dirs). Write the **model bundle only when `gate_passed`** — a model that failed the gate must never reach `ranker.model_path`, because `load_ranker` would load it and the app would silently ship a model D-022 rejected. On a failed gate, print a loud line: `gate FAILED — model NOT written; pipeline stays rules-only (D-022)`. End `main` by printing `gate_passed: <bool>`. (On the expected real-run pass, all three artifacts are written — acceptance criterion 2. The fixture gate passes, so the fixture test in Step 4 still gets a model to load.)

**Verify (import only)**: `.venv/bin/python -c "from src.recommendation.ranker import Ranker, load_ranker, product_features, ACTIVE_VOCAB; print(len(ACTIVE_VOCAB))"` → prints `22` (the count of `CANONICAL_IDS`; if the catalog vocab grows this number tracks it — any positive number with no import error is acceptable).

### Step 4: Write `tests/test_ranker.py`

Pure-Python, uses the committed fixture + an in-code catalog, tempfile for outputs. Model after `tests/test_import_sephora.py`. Cover:

1. **End-to-end fixture (aggregate → train → eval → load → score)** — run `main(["--reviews-dir","tests/fixtures/...", ...])` OR call the pipeline functions directly against `tests/fixtures/reviews_sample.csv` + the in-code catalog written to a tmp `catalog.json`. Assert the eval dict has `gate_passed is True` (**the fixture is engineered to pass** — the model beats both baselines on pooled ROC-AUC and pooled pairwise).
2. **Baselines are actually beaten** — assert `pooled["model"]["roc_auc"] > pooled["popularity"]["roc_auc"]` and `> pooled["bayesian"]["roc_auc"]`, same for `pairwise`.
3. **Disaggregation shape** — `by_tone` has `light`/`medium`/`deep` (from the fixture's tones) and a stable structure; each row carries `n` and `low_n`. Assert `unknown` is representable (present or explicitly empty, never silently dropped).
4. **Loaded ranker scores + reorders (spelled out — both fixture products are `treatment`, so they genuinely compete).** `load_ranker` on the tmp config paths returns a `Ranker`. Build a catalog `[PB, PC, PA]` where `PA`/`PB` are the two trained products (both `treatment`) and `PC` is a comedogenic salicylic clone: `Product("PC","SA Balm","BrandZ","treatment",actives=["salicylic_acid"],comedogenic_flags=["coconut_oil"],price_usd=9.0)`. Build a report with `Concern("acne_comedonal","nose",2,0.9)` **and** `Concern("dryness","left_cheek",1,0.9)` (so the target actives include both `salicylic_acid` and `ceramides`, pulling PA, PB, PC into the `treatment` step). Then:
   - oily `UserProfile(skin_type="oily")` → `recommend(...).routines["AM"]["treatment"]` product-ids == `["PA","PB","PC"]` (oily prefers PA; PC comedogenic sorts last).
   - dry `UserProfile(skin_type="dry")` → `["PB","PA","PC"]` (dry prefers PB; PC still last).
   The comedogenic-last invariant holding regardless of score mirrors `test_ranker_reorders_but_comedogenic_dominates` in the engine test. (Do **not** try to make a `treatment` and a `moisturizer` compete — they never share a step, and a tree that keys on `f_category` would tie the two anyway.)
5. **Evidence fallback below min cell size** — the fixture's train cells (PA/PB × oily/dry) all have `n ≈ 20–25`, so they never trip `min_cell_size=5`. To exercise the fallback, query a skin_type **absent** from the fixture: `ranker.evidence("PA","normal")` → the `__all__` cell with `fallback=True` / `cell="all_reviewers"`. Then `ranker.evidence("PA","oily")` → the oily cell with `fallback` falsey; and `ranker.evidence("NOPE","oily")` → `None`.
6. **Degradation** — `load_ranker` pointed at a non-existent model path returns `None`; `recommend(report, catalog, ranker=None)` still produces a `Recommendation` in stable rules order.
7. **Deterministic split** — `deterministic_test_mask` gives the same assignment across two calls / two processes (assert an author's side is stable; do not rely on builtin `hash()`).

Add a `__main__` block calling each test then `print("ok")`.

**Verify**: `.venv/bin/python -m pytest tests/test_ranker.py -q` → all pass; `.venv/bin/python tests/test_ranker.py` → prints `ok`.

### Step 5: Confirm the full fast suite stays green

**Verify**: `.venv/bin/python -m pytest -q` → all pass, no new failures, and the run does **not** import TF/YOLO/mediapipe (it won't — sklearn/pandas only).

### Step 6: Real training run (produces the shippable artifacts)

Prerequisite — build the real catalog if `data/processed/catalog.json` is absent:
`.venv/bin/python -m src.recommendation.import_catalog --csv data/raw/sephora/product_info.csv --format sephora`

Then run training:
`.venv/bin/python -m src.recommendation.ranker`

Expected: it prints the disaggregated eval table, reports the ~85% positive base rate and the count of reviews dropped for having no catalog product, writes `data/processed/review_stats.json` and `runs/ranker/eval.json`, and ends with a `gate_passed: <bool>` line. On this dataset the gate is *expected* to pass — in which case `models/ranker/ranker.joblib` is also written (per the 3g guard, the model is written only on a pass). If it does not pass, see the STOP note just below; a failed gate deliberately leaves no model behind.

These artifacts are **gitignored** (`models/`, `data/processed/`, `runs/` are all in `.gitignore`) — they are NOT committed; the committed deliverable is the code + fixture. Record the printed pooled table and `gate_passed` value in your completion report / the plans/README status note.

**If `gate_passed` is `False` on real data**: that is a legitimate scientific outcome, not a bug to paper over. Per D-022/D-005 the honest result is "ship rules-only." **STOP and report** the eval table — do NOT add features, leak the reviewer split, or tune metrics to force a pass. (You may double-check that the feature set matches this plan and the split is reviewer-disjoint; beyond that, report the negative result.)

**If the real reviews/catalog data is absent** (`data/raw/sephora/reviews_*.csv` missing): STOP and report — the fixture test (Steps 4–5) still fully validates the code path; the real run just needs the local Kaggle dump.

### Step 7: Docs

- `README.md` — under "## Run it", after the pipeline commands, add:
  ```bash
  # Train the learned ranker (needs data/processed/catalog.json + data/raw/sephora/reviews_*.csv):
  .venv/bin/python -m src.recommendation.ranker
  ```
- `docs/CATALOG_SCHEMA.md` — append a short subsection "## Review-stats & ranker artifacts" documenting the two JSON/bundle shapes (the `review_stats.json` `cells` structure and the joblib bundle keys), so the later report issue can cite the contract. Keep it to ~15 lines. Note the reviews are a **separate** artifact, not part of the catalog schema (D-009 unchanged).

**Verify**: `grep -n "src.recommendation.ranker" README.md` → 1 match; `grep -n "Review-stats" docs/CATALOG_SCHEMA.md` → 1 match.

## Test plan

- New file `tests/test_ranker.py` (see Step 4 for the seven cases), modeled structurally on `tests/test_import_sephora.py`.
- Fixture `tests/fixtures/reviews_sample.csv` (Step 2), the engineered "model beats popularity" corpus.
- The **fixture** end-to-end test is deterministic and MUST pass (we control the data). The **real** run's gate is empirical (Step 6) and is verified by running, not by an assertion in the suite.
- Verification: `.venv/bin/python -m pytest` → all pass including the new `tests/test_ranker.py` cases.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `.venv/bin/python -m pytest tests/test_ranker.py -q` → all pass; the end-to-end fixture test asserts `gate_passed is True`.
- [ ] `.venv/bin/python -m pytest -q` → all pass (full suite green, no regressions).
- [ ] `.venv/bin/python -c "from src.recommendation.ranker import load_ranker; print(load_ranker.__doc__ is not None)"` → `True` (module imports cleanly, sklearn not imported by the engine).
- [ ] `.venv/bin/python -c "from src.recommendation.engine import recommend"` still works unchanged; `git diff --stat b197b6b..HEAD -- src/recommendation/engine.py src/recommendation/schema.py src/pipeline/tone.py` → **empty** (those files untouched).
- [ ] `load_ranker` at a bogus model path returns `None`, and `recommend(..., ranker=None)` yields a `Recommendation` (degradation covered by a test).
- [ ] Real run (Step 6) executed: artifacts written and the printed `gate_passed` value + pooled table recorded in the completion note (whether `True` or `False`).
- [ ] Only in-scope files modified (`git status` shows nothing outside the Scope list).
- [ ] `plans/README.md` status row updated.

## STOP conditions

Stop and report back (do not improvise) if:

- The drift check shows `engine.py`, `schema.py`, `tone.py`, or the `ranker:`/`paths:` config blocks differ from the "Current state" excerpts.
- The fixture end-to-end test cannot reach `gate_passed is True` after the construction in Step 2 (the recipe is designed to pass; if it doesn't, something in the feature/split code is wrong — investigate that, don't weaken the assertion). If truly stuck, increasing the fixture to 100 reviewers is the one allowed adjustment; report it.
- `class_weight="balanced"` or `categorical_features="from_dtype"` is rejected by the installed sklearn (it shouldn't be — 1.9.0 supports both). If so, report; the documented fallback is explicit `sample_weight` + an integer-encoded categorical mask, but confirm before diverging.
- Step 6's real-data gate fails (`gate_passed: False`) — report the eval table; do NOT tune to force a pass (see Step 6).
- Implementing any of this appears to require editing `engine.py`, `schema.py`, or the report/app layer — that means the scope boundary is wrong; stop and report.
- The real reviews or catalog data is missing (Step 6) — report; the fixture suite still validates the code.

## Maintenance notes

For the owner after this lands:
- **The report issue consumes this.** `Ranker.evidence(product_id, skin_type)` returns the numbers behind the PRD's "41 reviewers with oily skin — 4.3 avg rating, 87% recommend" line; formatting is the report's job. When that issue is built, verify the fallback note (`fallback: True`, `cell: all_reviewers`) surfaces in the rendered "why" so a pooled cell is never silently presented as skin-type-specific.
- **The anti-skew guarantee lives in `product_features`/`reviewer_features`.** If you add a feature, add it in that one path so train and inference can never diverge. The persisted bundle carries `feature_columns`, `brand_vocab`, `active_vocab` precisely so inference reconstructs the exact training columns — don't recompute vocabularies at inference time.
- **The gate is a ratchet, not a one-time check.** If the feature set or dataset changes, re-run Step 6; if the model stops beating both baselines, the honest move is `load_ranker` returning None (rules-only) — the degradation path is already the safety net (D-005/D-019).
- **Determinism:** the split uses `hashlib.md5`, not builtin `hash()` (which is per-process salted). Keep it that way or reproducibility breaks silently.
- A reviewer should scrutinize: that `score` returns *higher = better* (the engine negates it — a sign flip would silently invert every routine), that `profile=None` is handled, and that the comedogenic partition still dominates the ranker in the engine's sort (the engine test already pins this; the ranker must not assume otherwise).
