# Plan 011: Build the catalog importer (CSV → catalog.json per CATALOG_SCHEMA)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `git diff --stat 1ebd544..HEAD -- src/recommendation/ tests/ configs/`
> Expected prior changes: 002 (configs), 003 (tests/test_recommendation_engine.py),
> 004 (configs concern_report key). `schema.py` and `engine.py` code must be
> unchanged from the excerpts referenced below; otherwise STOP.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/003-test-the-rules-brain.md (engine tests exist as safety net)
- **Category**: direction
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

The recommender (Stage 3) is fully specified — `docs/CATALOG_SCHEMA.md`
defines the normalized product shape, `configs/default.yaml` points at
`data/processed/catalog.json`, `schema.Product` is the runtime type, and
`engine.recommend()` consumes `list[Product]` — but NOTHING produces a
catalog. The recommender can only run on hand-built fixtures. This plan builds
the importer and normalizer against a committed fixture CSV. The real Kaggle
Sephora download (D-015) is NOT on disk; mapping its exact column names is a
documented follow-up, isolated to one function.

## Current state

- `docs/CATALOG_SCHEMA.md` — the spec. Key requirements: ingredient-normalized
  `actives` (canonical IDs, ~30 of them, listed in the doc), comedogenic flag
  list, closed category vocabulary `cleanser/treatment/serum/moisturizer/spf`
  (unmappable categories dropped), case-insensitive + punctuation-tolerant +
  parenthetical-alias normalizer ("Ascorbic Acid (Vitamin C)" → `vitamin_c`;
  "Sodium Hyaluronate" → `hyaluronic_acid`), idempotent + logged import
  (report products with ≥1 active vs zero).
- `src/recommendation/schema.py` — `Product` dataclass:

```python
@dataclass
class Product:
    product_id: str
    name: str
    brand: str
    category: str
    actives: list[str] = field(default_factory=list)
    comedogenic_flags: list[str] = field(default_factory=list)
    price_usd: Optional[float] = None
    price_is_stale: bool = True
```

- `configs/default.yaml` paths: `catalog_raw: data/raw/sephora`,
  `catalog_processed: data/processed/catalog.json`.
- `data/raw/sephora` does NOT exist locally (verified at planning time).
- The canonical actives + comedogenic lists in `docs/CATALOG_SCHEMA.md`
  sections "Canonical actives" and "Comedogenic flag list" — transcribe them
  into code from the doc, not from memory.
- Repo convention: stdlib over deps (use `csv`, not pandas), lazy heavy
  imports, `__main__`-runner tests (`tests/test_pipeline_collage.py`).

## Environment facts

- Fresh git worktree; `data/` absent. Interpreter:
  `/Users/princekumar/Documents/skinscan/.venv/bin/python` (no pytest).

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| New tests | `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_import_catalog.py` | prints `ok` |
| CLI smoke | `/Users/princekumar/Documents/skinscan/.venv/bin/python -m src.recommendation.import_catalog --csv tests/fixtures/catalog_sample.csv --out /tmp/catalog.json` | prints an import log, exit 0 |
| Regression | run every file in `tests/` | each prints `ok` |

## Scope

**In scope**:
- `src/recommendation/import_catalog.py` (create)
- `tests/fixtures/catalog_sample.csv` (create)
- `tests/test_import_catalog.py` (create)

**Out of scope**:
- `engine.py`, `schema.py` — consumers, unchanged.
- Downloading the Sephora dataset; adapting to its real column names
  (follow-up, see Maintenance notes).
- Fuzzy/edit-distance matching — v1 is exact-after-normalization plus the
  synonym table; note this ceiling in a `ponytail:` comment.

## Git workflow

- Stay on the worktree's branch. Commit style:
  `feat: catalog importer — csv to normalized catalog.json (D-009)`
- Do NOT push.

## Steps

### Step 1: The importer module

Create `src/recommendation/import_catalog.py` with this structure (stdlib
only: `csv`, `json`, `re`, `argparse`, `pathlib`):

- `CANONICAL_ACTIVES: dict[str, str]` — maps normalized ingredient strings →
  canonical IDs. Transcribe ALL actives from `docs/CATALOG_SCHEMA.md`
  "Canonical actives" (salicylic_acid … zinc), each with its obvious raw
  spellings, plus the doc's named synonyms: `"sodium hyaluronate":
  "hyaluronic_acid"`, `"ascorbic acid": "vitamin_c"`. Keys are
  post-normalization strings (lowercase, single-spaced).
- `COMEDOGENIC: dict[str, str]` — from the doc's "Comedogenic flag list"
  (`coconut oil` → `coconut_oil`, `isopropyl myristate`, `isopropyl
  palmitate`, `algae extract`; skip the vague "certain cocoa/wheat-germ
  derivatives" line — code needs exact strings; note that in a comment).
- `normalize_token(s)` — lowercase; extract parenthetical aliases: for
  `"Ascorbic Acid (Vitamin C)"` yield BOTH `ascorbic acid` and `vitamin c`;
  strip punctuation to spaces; collapse whitespace. Return list of candidate
  strings.
- `parse_ingredients(raw: str) -> tuple[list[str], list[str]]` — split on
  commas, run each token through `normalize_token`, look up candidates in
  `CANONICAL_ACTIVES` (also try the candidate with spaces → `_` since IDs are
  snake_case: `"vitamin c"` → `vitamin_c`) and `COMEDOGENIC`; return
  (sorted unique actives, sorted unique comedogenic flags). Unrecognized
  tokens are silently dropped (per the schema doc: "parse what you use").
- `product_from_row(row: dict, idx: int) -> Product | None` — expects fixture
  columns `name,brand,category,ingredients,price`; category lowercased must be
  in `schema.CATEGORIES` else return `None` (dropped); `product_id` =
  `f"p{idx:05d}"`; `price_usd` = float or None on empty/garbage;
  `price_is_stale=True` always.
- `import_csv(csv_path, out_path) -> dict` — reads with `csv.DictReader`,
  builds products, writes `out_path` as JSON (list of `dataclasses.asdict`
  dicts), creating parent dirs; returns and prints a log dict:
  `{"rows": N, "kept": N, "dropped_category": N, "with_actives": N, "zero_actives": N}`
  (zero-active products are KEPT — valid carriers per the schema doc).
- `load_catalog(path) -> list[Product]` — reads the JSON back into `Product`
  objects (this is what the engine will consume later).
- `main()` — argparse: `--csv` (required), `--out` (default from
  `src.config.load_config()["paths"]["catalog_processed"]`); prints the log.

**Verify**: `py_compile` exit 0.

### Step 2: The fixture

Create `tests/fixtures/catalog_sample.csv` with header
`name,brand,category,ingredients,price` and 8 rows exercising every rule:

1. Cleanser with `Salicylic Acid` → active matched.
2. Serum with `Ascorbic Acid (Vitamin C), Aqua` → parenthetical alias → `vitamin_c`.
3. Moisturizer with `Sodium Hyaluronate, Glycerin` → synonym → `hyaluronic_acid` + `glycerin`.
4. Moisturizer with `Coconut Oil, Ceramides` (mixed-case, e.g. `COCONUT OIL`) → active + comedogenic flag.
5. SPF row (category `spf`) with no recognized actives → kept, zero actives.
6. Row with category `Makeup` → dropped.
7. Treatment with `Benzoyl Peroxide 2.5%` → punctuation/number tolerance → `benzoyl_peroxide`.
8. Moisturizer with only unrecognized ingredients (`Aqua, Parfum`) → kept, zero actives.

Prices: mix of valid floats, empty, and garbage (`"?"`) to exercise price parsing.

### Step 3: Tests

Create `tests/test_import_catalog.py` (convention: `tests/test_pipeline_collage.py`):

1. `test_import_counts` — run `import_csv` on the fixture into a temp dir:
   log == `{"rows": 8, "kept": 7, "dropped_category": 1, "with_actives": 5, "zero_actives": 2}`.
2. `test_parenthetical_alias` — row 2's product has `actives == ["vitamin_c"]`.
3. `test_synonym_and_multi` — row 3 → `{"glycerin", "hyaluronic_acid"}`.
4. `test_comedogenic_flagged` — row 4 → `"coconut_oil" in comedogenic_flags`
   and `"ceramides" in actives`.
5. `test_dropped_category` — no product named like row 6 in the output.
6. `test_price_handling` — garbage price → `price_usd is None`;
   `price_is_stale` is True on all.
7. `test_load_catalog_round_trip` — `load_catalog` returns `Product`
   instances; feed them to `engine.recommend` with a clear-skin
   `ConcernReport` → returns a `Recommendation` without raising (proves the
   produced catalog is engine-compatible — the whole point).
8. `test_idempotent` — running `import_csv` twice produces byte-identical
   output files.

**Verify**: `/Users/princekumar/Documents/skinscan/.venv/bin/python tests/test_import_catalog.py` → `ok`

### Step 4: CLI smoke

**Verify**: the CLI command from the table → prints the log dict, creates the
out file, exit 0.

## Test plan

The 8 tests in Step 3 (test 7 is the integration proof). Regression: all
existing `tests/` files still `ok`.

## Done criteria

- [ ] `tests/test_import_catalog.py` prints `ok`
- [ ] CLI smoke passes as specified
- [ ] `grep -n "import pandas" src/recommendation/import_catalog.py` → no matches (stdlib csv only)
- [ ] All existing `tests/` files print `ok`
- [ ] `git status --porcelain` clean outside the in-scope list

## STOP conditions

- `schema.Product` doesn't match the excerpt.
- The canonical-actives section is missing from `docs/CATALOG_SCHEMA.md`
  (transcription source gone).
- You're tempted to add a dependency (pandas, rapidfuzz) — that's a design
  change, not an obstacle; stop and report instead.

## Maintenance notes

- Follow-up (separate, small): once `data/raw/sephora` is downloaded, write a
  `row_adapter` that renames the real Kaggle columns to
  `name,brand,category,ingredients,price` and maps Sephora's category taxonomy
  onto the closed five — isolated to `product_from_row`'s input.
- The normalizer is exact-match-after-normalization; if real-world INCI misses
  matter, the upgrade path is a synonym-table expansion first, fuzzy matching
  last (`ponytail:` comment marks this).
- Reviewer: check test 7 actually calls `recommend()` — it's the
  contract-level assertion.
