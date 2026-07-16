"""End-to-end: verified products yield valid routines or explicit unavailability."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from recsys.catalog import load_catalog
from recsys.compose import Step
from recsys.contracts import load_analysis, resolve_profile, sha256_file
from recsys.explain import step_to_dict
from recsys.knowledge import load_knowledge
from recsys.pipeline import run
from recsys.scoring import ScoredCandidate

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).parents[1] / "data"
ANALYSIS = FIXTURES / "analysis_v3_sample.json"
ARCHETYPE_IDS = ["best_overall", "budget", "gentle_sensitive", "minimal", "comprehensive"]
AVAILABLE_ARCHETYPE_IDS = ["best_overall", "minimal", "comprehensive"]

K = load_knowledge(DATA / "knowledge")


@pytest.fixture(scope="module")
def document():
    return run(ANALYSIS, FIXTURES / "profile_complete.json", generated_at="2026-07-14T00:00:00+00:00")


def _catalog_ids():
    catalog = json.loads((DATA / "catalog" / "seed_catalog.json").read_text())
    return {p["product_id"]: p for p in catalog["products"]}


def _steps(routine):
    return routine["am"] + routine["pm"] + routine["per_label"]


def test_status_and_archetypes(document):
    assert document["status"] == "partial"
    assert [r["archetype"] for r in document["routines"]] == AVAILABLE_ARCHETYPE_IDS
    unavailable = {item["archetype"]: item["reasons"]
                   for item in document["unavailable_archetypes"]}
    assert set(unavailable) == set(ARCHETYPE_IDS) - set(AVAILABLE_ARCHETYPE_IDS)
    assert "required_role_missing:treatment" in unavailable["gentle_sensitive"]


def test_every_product_exists_in_catalog(document):
    catalog = _catalog_ids()
    for routine in document["routines"]:
        for step in _steps(routine):
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


def test_profile_used_contains_complete_profile_intake(document):
    profile = document["profile_used"]
    assert profile["tone_source"] == "self_report"
    assert profile["treatment_history"] == []
    assert profile["acne_duration_weeks"] == 16
    assert profile["painful_or_deep_lesions"] is False
    assert profile["prior_scarring"] is False


def test_every_step_has_evidence_backed_why(document):
    for routine in document["routines"]:
        seen = set()
        for step in _steps(routine):
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
    names = set()
    for store in document["data_versions"]["signals"]:
        names.add(store["name"])
        assert store["catalog_sha256"] == catalog_entry["sha256"]
    assert names == {"ingredient_analysis", "review_stats", "popularity"}
    assert not any("catalog_sha256" in warning for warning in document["warnings"])


def test_budget_archetype_fails_closed_when_verified_products_exceed_cap(document):
    budget = next(item for item in document["unavailable_archetypes"]
                  if item["archetype"] == "budget")
    assert "required_role_missing:treatment" in budget["reasons"]


def test_gentle_archetype_fails_closed_without_verified_gentle_treatment(document):
    gentle = next(item for item in document["unavailable_archetypes"]
                  if item["archetype"] == "gentle_sensitive")
    assert gentle["reasons"] == ["required_role_missing:treatment"]


def test_routines_are_diverse(document):
    best = {s["product_id"] for r in document["routines"] if r["archetype"] == "best_overall"
            for s in _steps(r)}
    for routine in document["routines"]:
        if routine["archetype"] == "best_overall":
            continue
        ids = {s["product_id"] for s in _steps(routine)}
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
        for step in _steps(routine):
            actives = set(catalog[step["product_id"]]["actives"])
            assert not (actives & K.retinoids), step["product_id"]
    reasons = {v["reason"] for v in document["veto_log"]["profile"]}
    assert "retinoid_pregnancy_status_excluded" in reasons


def _hybrid():
    return run(ANALYSIS, FIXTURES / "profile_complete.json",
               generated_at="2026-07-14T00:00:00+00:00", eligibility_mode="hybrid")


def test_hybrid_widens_catalog_beyond_verified_only():
    strict = run(ANALYSIS, FIXTURES / "profile_complete.json",
                 generated_at="2026-07-14T00:00:00+00:00")  # default strict
    hybrid = _hybrid()
    strict_products = {s["product_id"] for r in strict["routines"] for s in _steps(r)}
    hybrid_products = {s["product_id"] for r in hybrid["routines"] for s in _steps(r)}
    # hybrid draws on more of the catalog than the evidence-verified-only pool
    assert len(hybrid_products) > len(strict_products)
    assert hybrid["engine"]["eligibility_mode"] == "hybrid"


def test_hybrid_labels_verified_vs_category_derived():
    hybrid = _hybrid()
    seen = set()
    for routine in hybrid["routines"]:
        for step in _steps(routine):
            assert step["verification"] in ("verified", "category_derived")
            seen.add(step["verification"])
            if step["verification"] == "category_derived":
                assert any("category-derived" in n for n in step["notes"])
            assert step["prescription"] is False  # no Rx products in the seed catalog
    # the seed catalog has both verified and (in hybrid) category-derived products
    assert "category_derived" in seen


def test_hybrid_keeps_hard_ingredient_safety():
    # retinoid-during-pregnancy exclusion is a HARD gate — it must hold in hybrid.
    document = run(ANALYSIS, FIXTURES / "profile_unknown.json",
                   generated_at="2026-07-14T00:00:00+00:00", eligibility_mode="hybrid")
    catalog = _catalog_ids()
    for routine in document["routines"]:
        for step in _steps(routine):
            actives = set(catalog[step["product_id"]]["actives"])
            assert not (actives & K.retinoids), step["product_id"]


def test_prescription_derives_from_label_facts_not_a_phantom_attribute():
    products, _ = load_catalog(DATA / "catalog" / "seed_catalog.json")
    profile = resolve_profile(FIXTURES / "profile_complete.json", load_analysis(ANALYSIS))
    base = next(p for p in products if p.category == "treatment")

    def flag(**facts):
        step = Step(slot="treatment", usage="PM", scored=ScoredCandidate(
            product=replace(base, **facts), final=1.0, signals=()))
        return step_to_dict(step, K, profile)

    rx = flag(drug_actives=({"name": "tretinoin", "strength": "0.05%"},), otc_drug=False)
    assert rx["prescription"] is True
    assert any("consult a doctor" in note for note in rx["notes"])
    # An OTC drug is a drug, but it is not a prescription.
    assert flag(drug_actives=({"name": "benzoyl_peroxide", "strength": "2.5%"},),
                otc_drug=True)["prescription"] is False
    # A cosmetic carries no drug actives; otc_drug=False alone must not flag it.
    assert flag(drug_actives=(), otc_drug=False)["prescription"] is False


def test_allergy_removes_products(tmp_path):
    profile = json.loads((FIXTURES / "profile_complete.json").read_text())
    profile["allergies"] = ["niacinamide"]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    document = run(ANALYSIS, path, generated_at="2026-07-14T00:00:00+00:00")
    catalog = _catalog_ids()
    for routine in document["routines"]:
        for step in _steps(routine):
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


def _derived_root_with_drug(tmp_path, **over):
    import hashlib
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "catalog_full.json").write_bytes(
        (DATA / "catalog" / "seed_catalog.json").read_bytes()
    )
    spl = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/x.xml"
    row = {
        "product_id": "dailymed:x:1:azelaic_acid-20%", "name": "AZELEX",
        "brand": "DailyMed SPL", "category": "treatment", "price_usd": None,
        "format": "cream", "inci": [],
        "inci_sha256": hashlib.sha256(b"[]").hexdigest(),
        "actives": ["azelaic_acid"], "otc_drug": False, "label_source": spl,
        "drug_actives": [{"name": "azelaic_acid", "strength": "20%", "source": spl}],
    }
    row.update(over)
    (derived / "catalog_drug.json").write_text(json.dumps(
        {"schema_version": "recsys-catalog-1", "products": [row]}
    ))
    return derived


def test_prescription_options_are_listed_for_matching_concerns(tmp_path):
    document = run(ANALYSIS, FIXTURES / "profile_complete.json",
                   data_root=_derived_root_with_drug(tmp_path),
                   generated_at="2026-07-14T00:00:00+00:00", eligibility_mode="hybrid")
    options = document["prescription_options"]
    assert [o["name"] for o in options] == ["AZELEX"]
    assert options[0]["actives"] == [{"name": "azelaic_acid", "strength": "20%"}]
    assert "acne_comedonal" in options[0]["targets"]
    assert "doctor" in options[0]["note"]
    assert document["data_versions"]["drug_catalog"]["prescription"] == 1
    # D-033 surfaces prescription options; it does not rank them into a routine.
    placed = {s["product_id"] for r in document["routines"] for s in _steps(r)}
    assert "dailymed:x:1:azelaic_acid-20%" not in placed


def test_a_prescription_is_never_placed_even_when_it_would_win_the_slot(tmp_path):
    # The seed catalog alone cannot fill 'treatment' for the gentle archetype, so
    # if placement were merely a matter of ranking, an Rx would take the slot
    # here. It must stay listed, and the routine must stay honest about the gap.
    derived = _derived_root_with_drug(tmp_path)
    document = run(ANALYSIS, FIXTURES / "profile_complete.json", data_root=derived,
                   generated_at="2026-07-14T00:00:00+00:00", eligibility_mode="hybrid")
    placed = {s["product_id"] for r in document["routines"] for s in _steps(r)}
    assert not any(pid.startswith("dailymed:") for pid in placed)
    assert document["prescription_options"]
    # an unpriced row never lands in a total
    for routine in document["routines"]:
        priced = [s["price_usd"] for s in _steps(routine)]
        assert all(p is not None for p in priced), routine["archetype"]


def test_the_recommendation_tracks_the_concern_profile(tmp_path):
    # Two real photos returned identical products, which looks like the engine
    # ignoring its input; they simply shared all four concerns. Change the
    # concerns and the treatment must change with them -- and an Rx is offered
    # only for a concern its actives actually target.
    derived = _derived_root_with_drug(tmp_path)  # AZELEX: azelaic acid 20%
    analysis = json.loads(ANALYSIS.read_text())

    def for_concerns(*keep):
        data = dict(analysis, concerns=[c for c in analysis["concerns"]
                                        if c["concern"] in keep])
        path = tmp_path / ("an_" + "_".join(keep) + ".json")
        path.write_text(json.dumps(data))
        document = run(path, FIXTURES / "profile_complete.json", data_root=derived,
                       generated_at="2026-07-14T00:00:00+00:00",
                       eligibility_mode="hybrid")
        treatments = {s["product_id"] for r in document["routines"] for s in _steps(r)
                      if s["slot"] in ("treatment", "serum")}
        return treatments, {o["name"] for o in document["prescription_options"]}

    comedonal, comedonal_rx = for_concerns("acne_comedonal")
    scarring, scarring_rx = for_concerns("acne_scarring")

    assert comedonal and scarring
    assert comedonal != scarring, "treatment must follow the concern"
    # azelaic acid targets comedonal acne; nothing in knowledge maps it to scarring
    assert comedonal_rx == {"AZELEX"}
    assert scarring_rx == set(), "an Rx is never offered for a concern it does not target"


def test_an_otc_drug_row_is_not_offered_as_a_prescription(tmp_path):
    document = run(ANALYSIS, FIXTURES / "profile_complete.json",
                   data_root=_derived_root_with_drug(tmp_path, otc_drug=True),
                   generated_at="2026-07-14T00:00:00+00:00", eligibility_mode="hybrid")
    assert document["prescription_options"] == []


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

    assert document["status"] == "partial"
    assert document["data_versions"]["catalog"]["path"] == str(derived / "catalog_full.json")
    assert document["data_versions"]["verification"]["products"] == 14
