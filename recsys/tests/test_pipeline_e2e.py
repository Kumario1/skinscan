"""End-to-end: the real analysis artifact + seed catalog -> 5 valid routines."""
import json
from pathlib import Path

import pytest

from recsys.contracts import sha256_file
from recsys.knowledge import load_knowledge
from recsys.pipeline import run

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).parents[1] / "data"
ANALYSIS = FIXTURES / "analysis_v3_sample.json"
ARCHETYPE_IDS = ["best_overall", "budget", "gentle_sensitive", "minimal", "comprehensive"]

K = load_knowledge(DATA / "knowledge")


@pytest.fixture(scope="module")
def document():
    return run(ANALYSIS, FIXTURES / "profile_complete.json", generated_at="2026-07-14T00:00:00+00:00")


def _catalog_ids():
    catalog = json.loads((DATA / "catalog" / "seed_catalog.json").read_text())
    return {p["product_id"]: p for p in catalog["products"]}


def test_status_and_archetypes(document):
    assert document["status"] == "ok"
    assert [r["archetype"] for r in document["routines"]] == ARCHETYPE_IDS


def test_every_product_exists_in_catalog(document):
    catalog = _catalog_ids()
    for routine in document["routines"]:
        for step in routine["am"] + routine["pm"]:
            assert step["product_id"] in catalog


def test_session_safety_invariants(document):
    for routine in document["routines"]:
        for check in routine["safety_checks"]:
            assert check["passed"], f"{routine['archetype']}: {check['rule']}"
        # SPF steps appear only in AM; retinoid products only in PM
        catalog = _catalog_ids()
        for step in routine["pm"]:
            assert step["slot"] != "spf" or step["usage"] == "AM_PM"
        for step in routine["am"]:
            actives = set(catalog[step["product_id"]]["actives"])
            assert not (actives & K.retinoids), "retinoid in AM"


def test_triage_and_framing_passthrough(document):
    assert document["framing"]["cosmetic_only"] is True
    assert "not medical advice" in document["framing"]["text"]
    triage = document["triage"]
    assert triage["level"] == "routine_plus_review"
    assert len(triage["referral_reasons"]) == 3
    assert triage["professional_review_observations"] == [{"code": "nevus_observation"}]
    assert triage["see_doctor_note"]


def test_every_step_has_evidence_backed_why(document):
    for routine in document["routines"]:
        seen = set()
        for step in routine["am"] + routine["pm"]:
            if step["product_id"] in seen:
                continue
            seen.add(step["product_id"])
            why = step["why"]
            assert why["summary"]
            assert why["signals"], step["product_id"]
            assert any(s["evidence"] for s in why["signals"])


def test_data_versions_match_disk(document):
    catalog_entry = document["data_versions"]["catalog"]
    assert catalog_entry["sha256"] == sha256_file(catalog_entry["path"])
    for store in document["data_versions"]["signals"]:
        assert store["name"] in ("review_stats", "popularity")


def test_budget_archetype_respects_caps(document):
    budget = next(r for r in document["routines"] if r["archetype"] == "budget")
    catalog = _catalog_ids()
    assert budget["total_price_usd"] <= 75
    seen = set()
    for step in budget["am"] + budget["pm"]:
        seen.add(step["product_id"])
        assert catalog[step["product_id"]]["price_usd"] <= 20


def test_gentle_archetype_excludes_harsh_actives(document):
    gentle = next(r for r in document["routines"] if r["archetype"] == "gentle_sensitive")
    catalog = _catalog_ids()
    for step in gentle["am"] + gentle["pm"]:
        actives = set(catalog[step["product_id"]]["actives"])
        assert not (actives & K.gentle_excluded_actives), step["product_id"]


def test_routines_are_diverse(document):
    best = {s["product_id"] for r in document["routines"] if r["archetype"] == "best_overall"
            for s in r["am"] + r["pm"]}
    for routine in document["routines"]:
        if routine["archetype"] == "best_overall":
            continue
        ids = {s["product_id"] for s in routine["am"] + routine["pm"]}
        assert ids != best, routine["archetype"]


def test_determinism_byte_identical():
    kwargs = dict(profile_path=FIXTURES / "profile_complete.json",
                  generated_at="2026-07-14T00:00:00+00:00")
    a = json.dumps(run(ANALYSIS, **kwargs), sort_keys=True)
    b = json.dumps(run(ANALYSIS, **kwargs), sort_keys=True)
    assert a == b


def test_pregnancy_unknown_excludes_retinoids():
    document = run(ANALYSIS, FIXTURES / "profile_unknown.json",
                   generated_at="2026-07-14T00:00:00+00:00")
    catalog = _catalog_ids()
    for routine in document["routines"]:
        for step in routine["am"] + routine["pm"]:
            actives = set(catalog[step["product_id"]]["actives"])
            assert not (actives & K.retinoids), step["product_id"]
    reasons = {v["reason"] for v in document["veto_log"]["profile"]}
    assert "retinoid_pregnancy_status_excluded" in reasons


def test_allergy_removes_products(tmp_path):
    profile = json.loads((FIXTURES / "profile_complete.json").read_text())
    profile["allergies"] = ["niacinamide"]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    document = run(ANALYSIS, path, generated_at="2026-07-14T00:00:00+00:00")
    catalog = _catalog_ids()
    for routine in document["routines"]:
        for step in routine["am"] + routine["pm"]:
            assert "niacinamide" not in catalog[step["product_id"]]["actives"]
    assert any(v["reason"] == "profile_allergy:niacinamide"
               for v in document["veto_log"]["profile"])


def test_referral_only_path(tmp_path):
    data = json.loads(ANALYSIS.read_text())
    data["decision"]["triage_level"] = "derm_first"
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(data))
    document = run(path, FIXTURES / "profile_complete.json",
                   generated_at="2026-07-14T00:00:00+00:00")
    assert document["status"] == "referral_only"
    assert document["routines"] == []
    assert "dermatologist" in document["triage"]["see_doctor_note"]
    assert document["framing"]["cosmetic_only"] is True


def test_full_derived_data_root_uses_full_catalog_and_static_knowledge(tmp_path):
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "catalog_full.json").write_bytes(
        (DATA / "catalog" / "seed_catalog.json").read_bytes()
    )

    document = run(
        ANALYSIS,
        FIXTURES / "profile_complete.json",
        data_root=derived,
        generated_at="2026-07-14T00:00:00+00:00",
    )

    assert document["status"] == "ok"
    assert document["data_versions"]["catalog"]["path"] == str(derived / "catalog_full.json")
    assert document["data_versions"]["verification"]["products"] == 13
