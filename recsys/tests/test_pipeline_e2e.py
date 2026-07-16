"""End-to-end: verified products yield valid routines or explicit unavailability."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from recsys.catalog import load_catalog
from recsys.compose import Step
from recsys.contracts import ContractViolation, load_analysis, resolve_profile, sha256_file
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

# Retinoids named literally, never read from K. K.retinoids is the table the gate
# filters with, so asserting a routine against it grades the engine with its own
# answer key: delete an entry and the gate stops vetoing and the assertion stops
# checking, in lockstep, and the test stays green while a pregnant user is handed
# a retinoid. test_gates.py's test_spf_gate is the pattern -- literal spf=15 and
# spf=30 against K.min_spf -- and this is the same move for retinoids.
#
# A deliberate lower bound, not a copy of the table: names may be added to
# K.retinoids (tazarotene, trifarotene, isotretinoin) without touching this, and
# every assertion below stays true. Pinning the table's exact contents is a
# knowledge test, not this file's job.
KNOWN_RETINOIDS = frozenset({"retinol", "retinal", "adapalene", "tretinoin"})

# Seed-catalog products that a human can confirm carry a retinoid by reading the
# label. Named so the pregnancy gate is checked against the world rather than
# against the table it consults.
RETINOL_PRODUCT_IDS = frozenset({"P269122", "P377533", "P439926", "P443842"})
# Retinyl acetate parses to no canonical active at all: this row is caught only
# by the raw-INCI scan, so it pins the other arm of the gate.
RETINYL_ESTER_ONLY_PRODUCT_ID = "P421275"


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
            assert not (actives & KNOWN_RETINOIDS), "retinoid in AM"


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
            assert not (actives & KNOWN_RETINOIDS), step["product_id"]
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
            assert not (actives & KNOWN_RETINOIDS), step["product_id"]


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


MINTED_DRUG_PRODUCT_ID = "P427406"  # a seed-catalog cosmetic, not in the overlay


def _derived_root_with_minted_drug_facts(tmp_path):
    """A derived root whose overlay mints a drug identity onto a seed cosmetic.

    The assertion carries DailyMed-cited drug_actives plus every usage fact the
    strict gates check -- and never mentions otc_drug, which is an OPTIONAL
    fact. The default overlay rides along unchanged so the other routines still
    compose exactly as in the plain seed run.
    """
    import hashlib
    import shutil
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "catalog_full.json").write_bytes(
        (DATA / "catalog" / "seed_catalog.json").read_bytes()
    )
    shutil.copytree(DATA / "verification", derived / "verification")
    spl = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/x.xml"
    body = b"minted prescription-strength label bytes"
    digest = hashlib.sha256(body).hexdigest()
    (derived / "verification" / "evidence" / digest).write_bytes(body)
    approved = json.loads((derived / "verification" / "approved.json").read_text())
    approved["products"].append({"product_id": MINTED_DRUG_PRODUCT_ID, "assertions": [{
        "status": "approved", "reviewer_id": "reviewer-1", "reviewer_type": "agent",
        "approved_at": "2026-07-13T00:00:00Z", "retrieved_at": "2026-07-13T00:00:00Z",
        "source_url": "https://example.test/minted", "source_sha256": digest,
        "facts": {
            "drug_actives": [
                {"name": "azelaic_acid", "strength": "20%", "source": spl}],
            "label_source": spl, "label_verified_at": "2026-07-13",
            "routine_roles": ["treatment"], "intended_areas": ["face"],
            "exposure": "leave_on", "format": "gel",
            "cadence": "pm", "cadence_source": "https://example.test/minted",
        },
    }]})
    (derived / "verification" / "approved.json").write_text(json.dumps(approved))
    return derived


@pytest.mark.parametrize("mode", ["strict", "hybrid"])
def test_an_overlay_minted_drug_row_with_unknown_otc_status_is_listed_never_placed(
        tmp_path, mode):
    """The overlay door accepts drug_actives without an otc_drug fact, so the
    row reaches the pipeline as a drug of UNKNOWN OTC status. Unknown is data,
    never a favorable default: the row must be treated as a prescription --
    surfaced in prescription_options for a doctor conversation -- and must
    never land in a routine step. Under `otc_drug is False` this exact row won
    the treatment slot in all three routines, published with "prescription":
    false and no doctor note."""
    document = run(ANALYSIS, FIXTURES / "profile_complete.json",
                   data_root=_derived_root_with_minted_drug_facts(tmp_path),
                   generated_at="2026-07-14T00:00:00+00:00", eligibility_mode=mode)

    placed = {s["product_id"] for r in document["routines"] for s in _steps(r)}
    assert MINTED_DRUG_PRODUCT_ID not in placed, "an Rx row must never be a routine step"
    vetoed = {v["product_id"] for v in document["veto_log"]["profile"]}
    listed = [o for o in document["prescription_options"]
              if o["actives"] == [{"name": "azelaic_acid", "strength": "20%"}]]
    assert listed or MINTED_DRUG_PRODUCT_ID in vetoed, (
        "a drug row that is not placed must be accounted for: listed as a "
        "prescription option or vetoed by name, never silently dropped")
    for option in listed:
        assert "doctor" in option["note"]


def test_the_pinned_retinoid_rows_draw_the_pregnancy_exclusion(tmp_path):
    """RETINOL_PRODUCT_IDS and RETINYL_ESTER_ONLY_PRODUCT_ID check the gate
    against the world: a human confirmed each of these labels carries a
    retinoid, so a pregnant profile must draw retinoid_pregnancy_status_excluded
    for every one of them, whatever K.retinoids happens to contain. Hybrid mode,
    because the treatment-category rows only become candidates there (strict
    requires verified drug_actives they do not have) and the pregnancy gate is
    HARD in both modes. P421275's retinyl acetate parses to no canonical
    active, so its veto can only come from the raw-INCI arm of the gate."""
    profile = json.loads((FIXTURES / "profile_complete.json").read_text())
    profile["pregnancy_status"] = "pregnant"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    document = run(ANALYSIS, path, generated_at="2026-07-14T00:00:00+00:00",
                   eligibility_mode="hybrid")

    pinned = RETINOL_PRODUCT_IDS | {RETINYL_ESTER_ONLY_PRODUCT_ID}
    excluded = {v["product_id"] for v in document["veto_log"]["profile"]
                if v["reason"] == "retinoid_pregnancy_status_excluded"}
    for product_id in sorted(pinned):
        assert product_id in excluded, product_id
    placed = {s["product_id"] for r in document["routines"] for s in _steps(r)}
    assert not (placed & pinned)


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


def test_a_data_root_without_a_catalog_names_what_is_missing(tmp_path):
    # The seed is a 60-product fixture. A derived root that carries no catalog of
    # its own has to say so by name: degrading to the fixture answers plausibly
    # from the wrong 60 products, which reads as a working run.
    derived = tmp_path / "derived"
    derived.mkdir()

    with pytest.raises(ContractViolation) as excinfo:
        run(ANALYSIS, FIXTURES / "profile_complete.json", data_root=derived,
            generated_at="2026-07-14T00:00:00+00:00")

    message = str(excinfo.value)
    assert "catalog_full.json" in message and "seed_catalog.json" in message
