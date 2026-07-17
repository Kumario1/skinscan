"""The golden pins decisions, not arithmetic: a product change must fail loudly,
a score change must not fail at all."""
import json
from pathlib import Path

import pytest

from recsys.evaluate import decisions, differences, evaluate
from recsys.pipeline import run

FIXTURES = Path(__file__).parent / "fixtures"
EVAL = Path(__file__).parents[1] / "eval"
MANIFEST = EVAL / "cases.json"
GOLDEN = EVAL / "golden" / "v3-complete-profile.json"


@pytest.fixture(scope="module")
def document():
    return run(FIXTURES / "analysis_v3_sample.json", FIXTURES / "profile_complete.json",
               generated_at="2026-07-14T00:00:00+00:00")


def _steps(routine):
    return routine["am"] + routine["pm"] + routine["per_label"]


def test_committed_golden_evaluation_cases_pass():
    assert evaluate(MANIFEST) == []


def test_the_golden_stays_short_enough_to_actually_read():
    # A golden nobody can read is re-pinned by reflex, which is the same as
    # having no golden at all. Anything that pushes it past a few screens is
    # detail without an oracle -- scores, signal values, per-product vetoes.
    assert len(GOLDEN.read_text().splitlines()) < 300


def test_a_score_that_moves_is_not_a_regression(document):
    nudged = json.loads(json.dumps(document))
    for routine in nudged["routines"]:
        for step in _steps(routine):
            step["why"]["score"] = 0.123456
            for signal in step["why"]["signals"]:
                signal["value"] = 0.5
                signal["evidence"] = "rewritten"
    assert decisions(nudged) == decisions(document)


def test_a_changed_product_is_a_regression_that_names_itself(document):
    changed = json.loads(json.dumps(document))
    changed["routines"][0]["am"][0]["product_id"] = "P000000"

    report = differences(decisions(document), decisions(changed))

    assert len(report) == 2
    assert all("best_overall" in line and "cleanser" in line for line in report)
    assert "P417238" in report[0] and "P000000" in report[1]


def test_one_dropped_veto_code_reports_one_line_not_a_cascade(document):
    # The sorted codes are a set. Diffing them by position renumbers every entry
    # after the one that moved, which is a wall of noise hiding a one-code change.
    thinner = json.loads(json.dumps(document))
    doomed = thinner["veto_log"]["profile"][0]["reason"]
    thinner["veto_log"]["profile"] = [v for v in thinner["veto_log"]["profile"]
                                      if v["reason"] != doomed]

    report = differences(decisions(document), decisions(thinner))

    assert report == [f'  veto_reasons.profile: removed "{doomed}"']


def test_a_reordered_routine_is_a_regression(document):
    # Contents-first diffing must not blind the golden to canonical step order.
    scrambled = json.loads(json.dumps(document))
    scrambled["routines"][0]["am"].reverse()

    report = differences(decisions(document), decisions(scrambled))

    assert len(report) == 1
    assert "reordered" in report[0]


def test_a_safety_check_that_disappears_is_a_regression(document):
    # Checks are pinned by rule name because "every check passed" is vacuously
    # true of a list a rule has quietly dropped out of.
    dropped = json.loads(json.dumps(document))
    for routine in dropped["routines"]:
        routine["safety_checks"] = [c for c in routine["safety_checks"]
                                    if c["rule"] != "retinoids_pm_only"]

    report = differences(decisions(document), decisions(dropped))

    assert any("retinoids_pm_only" in line and "(absent)" in line for line in report)


def test_a_veto_code_is_pinned_but_another_entry_carrying_it_is_not(document):
    known = document["veto_log"]["profile"][0]

    noisier = json.loads(json.dumps(document))
    noisier["veto_log"]["profile"].append(dict(known, product_id="P000000"))
    assert decisions(noisier) == decisions(document)

    novel = json.loads(json.dumps(document))
    novel["veto_log"]["profile"].append(dict(known, reason="brand_new_veto"))
    report = differences(decisions(document), decisions(novel))
    assert any("brand_new_veto" in line for line in report)


def test_update_regenerates_the_committed_golden(tmp_path):
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({
        "schema_version": "recsys-eval-1",
        "cases": [{
            "id": "regen",
            "analysis": str(FIXTURES / "analysis_v3_sample.json"),
            "profile": str(FIXTURES / "profile_complete.json"),
            "generated_at": "2026-07-14T00:00:00+00:00",
            "golden": "golden/regen.json",
        }],
    }))

    assert evaluate(manifest, update=True) == []
    assert evaluate(manifest) == []
    # the committed golden is what the current engine emits, not a stale pin
    assert json.loads((tmp_path / "golden" / "regen.json").read_text()) == json.loads(
        GOLDEN.read_text()
    )


def test_a_missing_golden_says_where_it_looked(tmp_path):
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({
        "schema_version": "recsys-eval-1",
        "cases": [{
            "id": "absent",
            "analysis": str(FIXTURES / "analysis_v3_sample.json"),
            "profile": str(FIXTURES / "profile_complete.json"),
            "generated_at": "2026-07-14T00:00:00+00:00",
            "golden": "golden/absent.json",
        }],
    }))

    failures = evaluate(manifest)

    assert len(failures) == 1
    assert "golden/absent.json" in failures[0]
