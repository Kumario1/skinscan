import json
from pathlib import Path

import pytest

from recsys.catalog import CatalogProduct
from recsys.contracts import Profile
from recsys.knowledge import load_knowledge
from recsys.signals import (
    ConcernEfficacySignal, ScoringContext, TargetConcern, load_providers,
)
from recsys.tools.build_concern_efficacy import (
    PROMPT_VERSION, _p3_bakeoff, build, resolve_prompt_version,
)


DATA = Path(__file__).parents[1] / "data"

# A P3 evaluation the D-023 gate accepts. build() refuses to run without one --
# derived from review metadata or supplied like this -- so tests that are about
# aggregation rather than the gate pass it explicitly.
PASSING_P3 = {
    "pooled": {
        "champion": {"roc_auc": 0.70, "pairwise": 0.60},
        "concern_conditioned": {"roc_auc": 0.71, "pairwise": 0.61},
    },
}


def _record(uid, outcome, skin_type="oily", product_id="p1", has_condition=True,
            prompt_version=PROMPT_VERSION):
    return {
        "uid": uid,
        "product_id": product_id,
        "skin_type": skin_type,
        "prompt_version": prompt_version,
        "status": "ok",
        "labels": [{
            "concern": "acne_comedonal",
            "outcome": outcome,
            "reviewer_has_condition": has_condition,
        }],
    }


def test_cached_labels_build_registered_concern_signal(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(_record(str(i), "helped" if i < 8 else "worsened"))
                                 for i in range(10)) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    build(labels, out, data_root, catalog_products=1, smoothing_m=20, sub_cell_min_n=5,
          p3_evaluation=PASSING_P3)

    store = json.loads(out.read_text())
    registry = json.loads((data_root / "signals" / "registry.json").read_text())
    assert registry["stores"][0]["kind"] == "concern_efficacy"
    assert registry["stores"][0]["coverage"] == {
        "catalog_products": 1,
        "products": 1,
        "products_with_acne_cell_n15": 0,
        "p3_gate_passed": True,
    }

    provider = ConcernEfficacySignal(store, {"version": "v1"})
    product = CatalogProduct(
        product_id="p1", name="Test", brand="Test", category="treatment",
        price_usd=10, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=("salicylic_acid",),
    )
    score = provider.score(product, "treatment", ScoringContext(
        targets=(TargetConcern("acne_comedonal", 3, 0.9),),
        profile=Profile(skin_type="oily"),
        knowledge=load_knowledge(DATA / "knowledge"),
        category_prices={},
    ))
    assert 0.5 < score.value < store["products"]["p1"]["acne_comedonal"]["by_skin_type"]["oily"]["smoothed"]
    assert "80% of 10 reviewers" in score.evidence
    assert score.details["matches"][0]["ladder"] == "exact"


def test_build_only_includes_products_in_selected_catalog(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join((
        json.dumps(_record("in", "helped")),
        json.dumps(_record("out", "helped", product_id="p2")),
    )) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    coverage = build(
        labels,
        out,
        data_root,
        catalog_products=1,
        catalog_product_ids=frozenset({"p1"}),
        p3_evaluation=PASSING_P3,
    )

    store = json.loads(out.read_text())
    assert set(store["products"]) == {"p1"}
    assert coverage["products"] == 1


def test_build_ignores_labels_from_reviewers_without_the_condition(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join((
        json.dumps(_record("condition", "helped", has_condition=True)),
        json.dumps(_record("generic", "worsened", has_condition=False)),
    )) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    build(labels, out, data_root, catalog_products=1, p3_evaluation=PASSING_P3)

    cell = json.loads(out.read_text())["products"]["p1"]["acne_comedonal"]["all"]
    assert cell["n"] == 1
    assert cell["helped"] == 1
    assert cell["worsened"] == 0


def test_concern_signal_falls_back_to_pooled_review_evidence():
    provider = ConcernEfficacySignal(
        {"products": {}}, {"version": "v1"},
        pooled_store={"products": {"p1": {
            "n": 100, "mean": 4.5, "smoothed": 4.4,
        }}},
    )
    product = CatalogProduct(
        product_id="p1", name="Test", brand="Test", category="treatment",
        price_usd=10, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=(),
    )
    score = provider.score(product, "treatment", ScoringContext(
        targets=(TargetConcern("acne_cystic", 3, 0.9),),
        profile=Profile(skin_type="oily"),
        knowledge=load_knowledge(DATA / "knowledge"),
        category_prices={},
    ))

    assert score is not None
    assert score.details["matches"][0]["ladder"] == "pooled"
    assert score.details["matches"][0]["n"] == 100
    assert "4.4★ across 100 pooled reviews" in score.evidence


def test_concern_signal_uses_general_acne_before_pooled_review_evidence():
    provider = ConcernEfficacySignal(
        {"products": {"p1": {
            "acne_comedonal": {"all": {"n": 0}},
            "acne_general": {"all": {
                "n": 10, "helped": 8, "worsened": 2, "help_rate": 0.8,
                "smoothed": 0.7,
            }},
        }}},
        {"version": "v1"},
        pooled_store={"products": {"p1": {
            "n": 100, "mean": 4.5, "smoothed": 4.4,
        }}},
    )
    product = CatalogProduct(
        product_id="p1", name="Test", brand="Test", category="treatment",
        price_usd=10, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=(),
    )
    score = provider.score(product, "treatment", ScoringContext(
        targets=(TargetConcern("acne_comedonal", 3, 0.9),),
        profile=Profile(skin_type="oily"),
        knowledge=load_knowledge(DATA / "knowledge"),
        category_prices={},
    ))

    assert score.details["matches"][0]["ladder"] == "acne_general"
    assert score.details["matches"][0]["cell_concern"] == "acne_general"
    assert "80% of 10 reviewers" in score.evidence


def test_loaded_concern_signal_receives_pooled_review_store(tmp_path):
    signals = tmp_path / "signals"
    signals.mkdir()
    concern_path = signals / "concern.json"
    review_path = signals / "review.json"
    concern_path.write_text(json.dumps({"products": {}}))
    review_path.write_text(json.dumps({"products": {"p1": {
        "n": 100, "mean": 4.5, "smoothed": 4.4,
    }}}))
    from recsys.contracts import sha256_file
    registry = {
        "schema_version": "recsys-registry-1",
        "stores": [
            {"name": "concern", "kind": "concern_efficacy", "version": "v1",
             "path": "signals/concern.json", "sha256": sha256_file(concern_path),
             "source": {"catalog_sha256": "catalog-1"}, "status": "active"},
            {"name": "review", "kind": "review_stats", "version": "v1",
             "path": "signals/review.json", "sha256": sha256_file(review_path),
             "source": {"catalog_sha256": "catalog-1"}, "status": "active"},
        ],
    }
    (signals / "registry.json").write_text(json.dumps(registry))

    providers, _, _ = load_providers(tmp_path, "catalog-1")
    concern = next(p for p in providers if isinstance(p, ConcernEfficacySignal))
    product = CatalogProduct(
        product_id="p1", name="Test", brand="Test", category="treatment",
        price_usd=10, size=None, format=None, spf=None, spf_source=None,
        inci=(), inci_sha256="", actives=(),
    )
    score = concern.score(product, "treatment", ScoringContext(
        targets=(TargetConcern("acne_cystic", 3, 0.9),),
        profile=Profile(skin_type="oily"),
        knowledge=load_knowledge(DATA / "knowledge"),
        category_prices={},
    ))

    assert score.details["matches"][0]["ladder"] == "pooled"


def test_failed_p3_bakeoff_does_not_register_concern_store(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(_record("one", "helped")) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"
    p3 = {
        "pooled": {
            "champion": {"roc_auc": 0.70, "pairwise": 0.60},
            "concern_conditioned": {"roc_auc": 0.71, "pairwise": 0.59},
        },
    }

    with pytest.raises(RuntimeError, match="P3 bake-off failed"):
        build(labels, out, data_root, catalog_products=1, p3_evaluation=p3)

    assert not out.exists()
    assert not (data_root / "signals" / "registry.json").exists()


def test_passing_p3_bakeoff_is_recorded_before_registration(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(_record("one", "helped")) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"
    p3 = {
        "pooled": {
            "champion": {"roc_auc": 0.70, "pairwise": 0.60},
            "concern_conditioned": {"roc_auc": 0.71, "pairwise": 0.61},
        },
    }

    coverage = build(labels, out, data_root, catalog_products=1, p3_evaluation=p3)

    store = json.loads(out.read_text())
    registry = json.loads((data_root / "signals" / "registry.json").read_text())
    assert coverage["p3_gate_passed"] is True
    assert store["p3"]["pooled"]["concern_conditioned"]["pairwise"] == 0.61
    assert registry["stores"][0]["source"]["p3"] == p3


def test_p3_bakeoff_excludes_other_prompt_versions():
    def evaluable(uid, prompt_version, outcome):
        record = _record(uid, outcome)
        record.update(uid=uid, author_id=uid, rating=5.0,
                      prompt_version=prompt_version)
        return record

    current = [evaluable(str(i), PROMPT_VERSION,
                         "helped" if i % 2 else "worsened") for i in range(8)]
    stale = [evaluable(f"stale{i}", "p1", "helped") for i in range(20)]

    only_current = _p3_bakeoff(current, smoothing_m=20, prompt_version=PROMPT_VERSION)
    with_stale = _p3_bakeoff(current + stale, smoothing_m=20,
                             prompt_version=PROMPT_VERSION)

    assert only_current is not None
    # Stale-prompt-version rows must never enter the bake-off population — the
    # gate has to evaluate exactly what the store aggregation later builds from.
    assert with_stale == only_current


def test_build_skips_unclear_only_cells_without_zero_division(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join((
        json.dumps(_record("u1", "unclear", product_id="p1")),
        json.dumps(_record("h1", "helped", product_id="p2")),
    )) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    # smoothing_m=0 turns an unclear-only (n=0) cell into a 0/0 division unless
    # _cell guards it; the cell must also be dropped rather than emitted at n=0.
    build(labels, out, data_root, catalog_products=2, smoothing_m=0,
          p3_evaluation=PASSING_P3)

    store = json.loads(out.read_text())
    assert "p1" not in store["products"]
    assert store["products"]["p2"]["acne_comedonal"]["all"]["n"] == 1


def test_evaluable_labels_auto_run_the_p3_gate(tmp_path):
    record = _record("one", "helped")
    record.update(author_id="one", rating=5.0)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(record) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    with pytest.raises(RuntimeError, match="P3 bake-off failed"):
        build(labels, out, data_root, catalog_products=1)

    assert not (data_root / "signals" / "registry.json").exists()


def test_a_single_version_file_builds_at_that_version_not_a_constant(tmp_path):
    # The original defect: a builder-local constant ("p10") filtering a ledger
    # written at another version ("p11") silently discarded 33,775 of 33,825
    # paid labels. The version now comes from the file: a ledger written
    # entirely at a version the builder has never heard of must build at that
    # version, with the store and its provenance saying so.
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(
        json.dumps(_record(str(i), "helped" if i < 8 else "worsened",
                           prompt_version="p12"))
        for i in range(10)) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    coverage = build(labels, out, data_root, catalog_products=1,
                     p3_evaluation=PASSING_P3)

    store = json.loads(out.read_text())
    registry = json.loads((data_root / "signals" / "registry.json").read_text())
    assert coverage["products"] == 1
    assert store["prompt_version"] == "p12"
    assert store["products"]["p1"]["acne_comedonal"]["all"]["n"] == 10
    assert registry["stores"][0]["source"]["prompt_version"] == "p12"


def test_a_mixed_version_file_errors_without_explicit_selection(tmp_path):
    # Two versions are two labeling policies; aggregating them together mixes
    # policies and guessing one silently discards the other. Ambiguity is the
    # operator's call to resolve, not the builder's to paper over.
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join((
        json.dumps(_record("old", "worsened", prompt_version="p11")),
        json.dumps(_record("new", "helped", prompt_version="p12")),
    )) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    with pytest.raises(SystemExit, match="mixes prompt versions"):
        build(labels, out, data_root, catalog_products=1, p3_evaluation=PASSING_P3)

    assert not out.exists()
    assert not (data_root / "signals" / "registry.json").exists()


def test_an_explicit_prompt_version_selects_one_policy_from_a_mixed_file(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(
        [json.dumps(_record(f"old{i}", "worsened", prompt_version="p11"))
         for i in range(3)]
        + [json.dumps(_record("new", "helped", prompt_version="p12"))]
    ) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    build(labels, out, data_root, catalog_products=1, p3_evaluation=PASSING_P3,
          prompt_version="p12")

    store = json.loads(out.read_text())
    cell = store["products"]["p1"]["acne_comedonal"]["all"]
    assert store["prompt_version"] == "p12"
    # only the selected policy's single row aggregates; the p11 rows do not
    assert (cell["n"], cell["helped"], cell["worsened"]) == (1, 1, 0)


def test_a_version_matching_zero_records_hard_fails_and_registers_nothing(tmp_path):
    # The recurrence path of the original bug: select a version the ledger
    # holds no rows at, and the filter yields nothing. That used to write
    # "products": {} and register it active with confident provenance.
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(_record("one", "helped", prompt_version="p11")) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    with pytest.raises(SystemExit, match="matches no ok records"):
        build(labels, out, data_root, catalog_products=1, p3_evaluation=PASSING_P3,
              prompt_version="p12")

    assert not out.exists()
    assert not (data_root / "signals" / "registry.json").exists()


def test_resolve_prompt_version_requires_ok_records():
    with pytest.raises(SystemExit, match="no ok records"):
        resolve_prompt_version([{"status": "error", "prompt_version": "p11"}])


def test_labels_without_review_metadata_fail_rather_than_skip_the_p3_gate(tmp_path):
    # _p3_bakeoff returning None used to slip past the gate entirely: a store
    # the D-023 protocol never examined registered exactly like one that had
    # passed it. No evaluable rows and no --p3-eval is a build failure.
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(_record("one", "helped")) + "\n")  # no author_id/rating
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    with pytest.raises(SystemExit, match="P3 bake-off has nothing to evaluate"):
        build(labels, out, data_root, catalog_products=1)

    assert not out.exists()
    assert not (data_root / "signals" / "registry.json").exists()


def test_zero_surviving_cells_never_write_an_empty_store(tmp_path):
    # Records exist at the right version but none survive aggregation (here:
    # no reviewer has the condition). An empty store scores every product at
    # neutral while registered as an active signal -- refuse to write it.
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(_record("one", "helped", has_condition=False)) + "\n")
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"

    with pytest.raises(SystemExit, match="empty store"):
        build(labels, out, data_root, catalog_products=1, p3_evaluation=PASSING_P3)

    assert not out.exists()
    assert not (data_root / "signals" / "registry.json").exists()


def test_cli_prompt_version_flag_reaches_the_build(tmp_path):
    from recsys.tools.build_concern_efficacy import main
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"products": [{"product_id": "p1"}]}))
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join((
        json.dumps(_record("old", "worsened", prompt_version="p11")),
        json.dumps(_record("new", "helped", prompt_version="p12")),
    )) + "\n")
    p3_eval = tmp_path / "p3.json"
    p3_eval.write_text(json.dumps(PASSING_P3))
    data_root = tmp_path / "data"
    out = data_root / "signals" / "concern_efficacy.v1.json"
    argv = ["--labels", str(labels), "--catalog", str(catalog), "--out", str(out),
            "--data-root", str(data_root), "--p3-eval", str(p3_eval)]

    with pytest.raises(SystemExit, match="mixes prompt versions"):
        main(argv)

    assert main(argv + ["--prompt-version", "p12"]) == 0
    assert json.loads(out.read_text())["prompt_version"] == "p12"
