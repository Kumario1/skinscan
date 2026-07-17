"""Legacy grouped concerns must not create exact-label product targets."""
import copy
import json

import pytest

from recsys.pipeline import run

from .test_pipeline_e2e import FIXTURES, _steps

ACNE_ACTIVES = {
    "salicylic_acid", "benzoyl_peroxide", "adapalene", "retinol",
    "azelaic_acid", "glycolic_acid", "sulfur",
}


def _analysis_with(tmp_path, severities):
    data = json.loads((FIXTURES / "analysis_v3_sample.json").read_text())
    data = copy.deepcopy(data)
    data["concerns"] = [
        {"concern": c, "severity": s, "confidence": 0.9, "sources": ["prediction"]}
        for c, s in sorted(severities.items())
    ]
    path = tmp_path / ("analysis-" + "-".join(sorted(severities)) + ".json")
    path.write_text(json.dumps(data))
    return path


def _run(tmp_path, severities):
    return run(_analysis_with(tmp_path, severities),
               FIXTURES / "profile_complete.json",
               generated_at="2026-07-16T00:00:00+00:00")


def _catalog_actives():
    catalog = json.loads(
        (FIXTURES.parents[1] / "data" / "catalog" / "seed_catalog.json").read_text()
    )
    return {p["product_id"]: set(p["actives"]) for p in catalog["products"]}


def _selected_serum(document):
    serums = {s["product_id"] for r in document["routines"]
              for s in _steps(r) if s["slot"] == "serum"}
    return serums


def test_clear_skin_is_maintenance_only(tmp_path):
    document = _run(tmp_path, {})
    assert document["status"] == "ok"
    for routine in document["routines"]:
        assert {s["slot"] for s in _steps(routine)} <= {"cleanser", "moisturizer", "spf"}


def test_dryness_only_routine_carries_no_acne_actives(tmp_path):
    document = _run(tmp_path, {"dryness": 3})
    assert document["status"] == "ok"
    actives = _catalog_actives()
    for routine in document["routines"]:
        for step in _steps(routine):
            assert not (actives[step["product_id"]] & ACNE_ACTIVES), (
                f"{step['product_id']} carries an acne active in a "
                "dryness-only routine"
            )


def test_grouped_concern_change_cannot_create_a_serum(tmp_path):
    dryness = _selected_serum(_run(tmp_path, {"dryness": 3}))
    acne = _selected_serum(_run(tmp_path, {"acne_comedonal": 3}))
    assert dryness == acne == set()


def test_cystic_group_name_cannot_create_a_product_target(tmp_path):
    document = _run(tmp_path, {"acne_cystic": 4})
    assert document["target_lesions"] == []


@pytest.mark.parametrize("concern", ["acne_comedonal", "acne_inflammatory"])
def test_single_concern_severity_does_not_change_support_products(tmp_path, concern):
    # By design: severity drives triage and the clinician-gated therapy plan,
    # not cosmetic support. One target normalizes its own severity away, so
    # light and severe pick identical support products. If this ever fails,
    # someone made severity rank products — bump this test only with a D-line.
    light = _run(tmp_path, {concern: 1})["selected_products"]
    severe = _run(tmp_path, {concern: 4})["selected_products"]
    assert light == severe
