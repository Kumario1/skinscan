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


def _reviewed_therapy_analysis(
        tmp_path, therapy="azelaic_acid", strength="20%", *, profile=None):
    data = json.loads(ANALYSIS.read_text())
    data["decision"].update(
        therapy_disposition="active_treatment",
        policy_reviewed=True,
        policy_version="test-reviewed:1",
    )
    data["policies"]["therapy"]["reviewed"] = True
    data["policies"]["therapy"].update(
        identity="test-reviewed:1", sha256="a" * 64,
    )
    data["therapy_plan"].update(
        primary={
            "therapy": therapy,
            "strength_band": strength,
            "exposure": "leave_on",
            "cadence": "per_label",
            "role": "treatment",
            "cadence_source": "test-reviewed:1",
        },
        deferred_reasons=[],
        policy_version="test-reviewed:1",
    )
    if profile is not None:
        data["input_profile"] = profile
    path = tmp_path / f"analysis-{therapy}-{strength}.json"
    path.write_text(json.dumps(data))
    return path


def test_status_and_archetypes(document):
    assert document["status"] == "ok"
    assert [r["archetype"] for r in document["routines"]] == ["best_overall"]
    assert [row["archetype"] for row in document["unselected_archetypes"]] == [
        "gentle_sensitive", "minimal", "comprehensive",
    ]
    # budget stays honestly unavailable: the seed catalog cannot fill it under its cap
    assert "budget" in {u["archetype"] for u in document["unavailable_archetypes"]}
    assert document["selected_regimen"] == document["routines"][0]
    # Schema-3 compatibility keeps only support products when therapy is
    # deferred. A grouped concern can no longer manufacture a serum target.
    assert set(document["selected_products"]) == {"cleanser", "moisturizer", "spf"}
    assert document["alternatives"] == {}
    assert document["care_decision"]["therapy_disposition"] == "defer"
    assert document["therapy_plan"]["primary"] is None
    assert "safe to start" not in document["triage"]["see_doctor_note"]


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


def test_public_output_selects_the_first_valid_routine(document):
    assert len(document["routines"]) == 1
    assert document["selected_regimen"] == document["routines"][0]
    assert len(document["selected_products"]) == len(set(document["selected_products"]))


def test_determinism_byte_identical():
    kwargs = dict(profile_path=FIXTURES / "profile_complete.json",
                  generated_at="2026-07-14T00:00:00+00:00")
    a = json.dumps(run(ANALYSIS, **kwargs), sort_keys=True)
    b = json.dumps(run(ANALYSIS, **kwargs), sort_keys=True)
    assert a == b


def test_unknown_intake_defers_primary_treatment(tmp_path):
    analysis = _reviewed_therapy_analysis(tmp_path)
    document = run(analysis, FIXTURES / "profile_unknown.json",
                   generated_at="2026-07-14T00:00:00+00:00")
    assert document["care_decision"]["therapy_disposition"] == "active_treatment"
    assert document["therapy_plan"]["primary"] is not None
    assert document["treatment_fulfillment"]["status"] == "deferred"
    # only the treatment slot is withheld on deferral; serums remain (D-029)
    assert all(step["slot"] != "treatment"
               for routine in document["routines"] for step in _steps(routine))


def _profile_file(tmp_path, **overrides):
    profile = json.loads((FIXTURES / "profile_complete.json").read_text())
    profile.update(overrides)
    path = tmp_path / "profile-combo.json"
    path.write_text(json.dumps(profile))
    return path


@pytest.mark.parametrize("overrides,status,reason", [
    ({}, "included", None),
    ({"age_years": 15}, "included", None),
    ({"pregnancy_status": "pregnant"}, "included", None),  # BP is not a retinoid
    ({"pregnancy_status": "unknown"}, "deferred",
     "required_profile_unknown:pregnancy_status"),
    ({"age_years": None}, "deferred", "required_profile_unknown:age_years"),
    ({"acne_duration_weeks": None}, "deferred",
     "required_profile_unknown:acne_duration_weeks"),
    ({"painful_or_deep_lesions": None}, "deferred",
     "required_profile_unknown:painful_or_deep_lesions"),
    ({"prior_scarring": None}, "deferred", "required_profile_unknown:prior_scarring"),
])
def test_intake_combinations_gate_the_treatment_slot(tmp_path, overrides, status, reason):
    analysis = _reviewed_therapy_analysis(tmp_path, "benzoyl_peroxide", "2.5%")
    document = run(analysis, _profile_file(tmp_path, **overrides),
                   generated_at="2026-07-14T00:00:00+00:00")
    assert document["treatment_fulfillment"]["status"] == status
    if status == "included":
        assert document["selected_products"]["treatment"] == "P188306"
    else:
        assert reason in document["treatment_fulfillment"]["reasons"]
        assert "treatment" not in document["selected_products"]


def test_pregnant_profile_gets_treatment_but_never_a_retinoid(tmp_path):
    analysis = _reviewed_therapy_analysis(tmp_path, "benzoyl_peroxide", "2.5%")
    document = run(analysis, _profile_file(tmp_path, pregnancy_status="pregnant"),
                   generated_at="2026-07-14T00:00:00+00:00")
    assert document["selected_products"]["treatment"] == "P188306"
    placed = {s["product_id"] for r in document["routines"] for s in _steps(r)}
    assert not (placed & (RETINOL_PRODUCT_IDS | {RETINYL_ESTER_ONLY_PRODUCT_ID}))


def test_bypass_flag_does_not_weaken_profile_or_pregnancy_gates(tmp_path):
    # An unreviewed-policy analysis with a primary parses only under the flag,
    # and the downstream safety gates still hold.
    analysis_path = _reviewed_therapy_analysis(tmp_path, "benzoyl_peroxide", "2.5%")
    data = json.loads(analysis_path.read_text())
    data["policies"]["therapy"].update(reviewed=False, sha256=None)
    analysis_path.write_text(json.dumps(data))

    with pytest.raises(ContractViolation, match="reviewed therapy policy"):
        run(analysis_path, FIXTURES / "profile_complete.json",
            generated_at="2026-07-14T00:00:00+00:00")

    unknown = run(analysis_path, _profile_file(tmp_path, pregnancy_status="unknown"),
                  generated_at="2026-07-14T00:00:00+00:00",
                  allow_unreviewed_policy=True)
    assert unknown["treatment_fulfillment"]["status"] == "deferred"
    placed = {s["product_id"] for r in unknown["routines"] for s in _steps(r)}
    assert not (placed & (RETINOL_PRODUCT_IDS | {RETINYL_ESTER_ONLY_PRODUCT_ID}))

    complete = run(analysis_path, FIXTURES / "profile_complete.json",
                   generated_at="2026-07-14T00:00:00+00:00",
                   allow_unreviewed_policy=True)
    assert complete["selected_products"]["treatment"] == "P188306"


def test_reviewed_plan_selects_only_the_exact_verified_therapy(tmp_path):
    analysis = _reviewed_therapy_analysis(
        tmp_path, "benzoyl_peroxide", "2.5%"
    )
    document = run(
        analysis, FIXTURES / "profile_complete.json",
        generated_at="2026-07-14T00:00:00+00:00",
    )

    assert document["care_decision"]["therapy_disposition"] == "active_treatment"
    assert document["selected_products"]["treatment"] == "P188306"
    assert [r["archetype"] for r in document["routines"]] == ["best_overall"]
    assert [row["archetype"] for row in document["unselected_archetypes"]] == (
        AVAILABLE_ARCHETYPE_IDS[1:]
    )


def _hybrid():
    return run(ANALYSIS, FIXTURES / "profile_complete.json",
               generated_at="2026-07-14T00:00:00+00:00", eligibility_mode="hybrid")


def test_strict_request_is_compatibility_only_and_uses_d035_hybrid():
    strict = run(ANALYSIS, FIXTURES / "profile_complete.json",
                 generated_at="2026-07-14T00:00:00+00:00",
                 eligibility_mode="strict")
    hybrid = _hybrid()
    strict_products = {s["product_id"] for r in strict["routines"] for s in _steps(r)}
    hybrid_products = {s["product_id"] for r in hybrid["routines"] for s in _steps(r)}
    assert hybrid_products == strict_products
    assert hybrid["engine"]["eligibility_mode"] == "hybrid"
    assert hybrid["engine"]["requested_eligibility_mode"] == "hybrid"
    assert not any("retired by D-035" in warning for warning in hybrid["warnings"])
    assert any("retired by D-035" in warning for warning in strict["warnings"])


def test_hybrid_emits_explicit_verification_status():
    hybrid = _hybrid()
    for routine in hybrid["routines"]:
        for step in _steps(routine):
            assert step["verification_status"] in {"verified", "partial", "unverified"}
            assert step["verification"] == step["verification_status"]
            assert step["prescription"] is False  # no Rx products in the seed catalog


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
    assert document["status"] == "ok"
    assert len(document["routines"]) == 1
    assert set(document["selected_products"]) == {"cleanser", "moisturizer", "spf"}
    assert "dermatologist" in document["triage"]["see_doctor_note"]
    assert "support only" in document["triage"]["see_doctor_note"]
    assert document["framing"]["cosmetic_only"] is True


def _derived_root_with_drug(tmp_path, **over):
    import hashlib
    import shutil
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
        "label_verified_at": "2026-07-14", "routine_roles": ["treatment"],
        "contraindications": [],
        "intended_areas": ["face"], "exposure": "leave_on",
        "cadence": "per_label", "cadence_source": spl,
    }
    row.update(over)
    (derived / "catalog_drug.json").write_text(json.dumps(
        {"schema_version": "recsys-catalog-1", "products": [row]}
    ))
    shutil.copytree(DATA / "verification", derived / "verification")
    evidence = b"synthetic label explicitly reviewed: contraindications none stated"
    digest = hashlib.sha256(evidence).hexdigest()
    (derived / "verification" / "evidence" / digest).write_bytes(evidence)
    approved_path = derived / "verification" / "approved.json"
    approved = json.loads(approved_path.read_text())
    approved["products"].append({
        "product_id": row["product_id"],
        "assertions": [{
            "status": "approved",
            "reviewer_id": "synthetic-test-reviewer",
            "reviewer_type": "agent",
            "approved_at": "2026-07-14T00:00:00+00:00",
            "source_url": spl,
            "retrieved_at": "2026-07-14T00:00:00+00:00",
            "source_sha256": digest,
            "facts": {"contraindications": []},
        }],
    })
    approved_path.write_text(json.dumps(approved))
    return derived


def test_prescription_options_are_listed_for_exact_reviewed_plan(tmp_path):
    analysis = _reviewed_therapy_analysis(tmp_path)
    document = run(analysis, FIXTURES / "profile_complete.json",
                   data_root=_derived_root_with_drug(tmp_path),
                   generated_at="2026-07-14T00:00:00+00:00", eligibility_mode="hybrid")
    options = document["prescription_options"]
    assert [o["name"] for o in options] == ["AZELEX"]
    assert options[0]["actives"] == [{"name": "azelaic_acid", "strength": "20%"}]
    assert options[0]["therapy_plan_match"] == {
        "therapy": "azelaic_acid",
        "strength_band": "20%",
        "exposure": "leave_on",
        "cadence": "per_label",
    }
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
    analysis = _reviewed_therapy_analysis(tmp_path)
    document = run(analysis, FIXTURES / "profile_complete.json", data_root=derived,
                   generated_at="2026-07-14T00:00:00+00:00", eligibility_mode="hybrid")
    placed = {s["product_id"] for r in document["routines"] for s in _steps(r)}
    assert not any(pid.startswith("dailymed:") for pid in placed)
    assert document["prescription_options"]
    # an unpriced row never lands in a total
    for routine in document["routines"]:
        priced = [s["price_usd"] for s in _steps(routine)]
        assert all(p is not None for p in priced), routine["archetype"]


def test_reviewed_therapy_plan_not_concern_union_controls_treatment(tmp_path):
    derived = _derived_root_with_drug(tmp_path)  # AZELEX: azelaic acid 20%
    azelaic = run(
        _reviewed_therapy_analysis(tmp_path, "azelaic_acid"),
        FIXTURES / "profile_complete.json", data_root=derived,
        generated_at="2026-07-14T00:00:00+00:00",
    )
    salicylic = run(
        _reviewed_therapy_analysis(tmp_path, "salicylic_acid"),
        FIXTURES / "profile_complete.json", data_root=derived,
        generated_at="2026-07-14T00:00:00+00:00",
    )

    assert {o["name"] for o in azelaic["prescription_options"]} == {"AZELEX"}
    assert salicylic["prescription_options"] == []


def test_plan_match_rejects_unplanned_combination_active(tmp_path):
    spl = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/x.xml"
    derived = _derived_root_with_drug(
        tmp_path,
        product_id="dailymed:x:1:azelaic_acid-20%-clindamycin-1%",
        name="Unplanned Combination",
        actives=["azelaic_acid", "clindamycin"],
        drug_actives=[
            {"name": "azelaic_acid", "strength": "20%", "source": spl},
            {"name": "clindamycin", "strength": "1%", "source": spl},
        ],
    )
    document = run(
        _reviewed_therapy_analysis(tmp_path, "azelaic_acid", "20%"),
        FIXTURES / "profile_complete.json", data_root=derived,
        generated_at="2026-07-14T00:00:00+00:00",
    )

    assert document["prescription_options"] == []
    assert document["care_decision"]["therapy_disposition"] == "active_treatment"
    assert document["therapy_plan"]["primary"] is not None
    assert document["treatment_fulfillment"] == {
        "status": "unfilled",
        "reasons": ["required_role_unfilled:treatment"],
    }
    assert document["status"] == "ok"
    assert set(document["selected_products"]) == {"cleanser", "moisturizer", "serum", "spf"}
    assert "support_only_treatment_deferred" in document["selected_regimen"]["notes"]


def test_generic_reviewed_strength_band_accepts_a_verified_label_strength(tmp_path):
    document = run(
        _reviewed_therapy_analysis(
            tmp_path, "azelaic_acid", "verified_otc_or_labeled"
        ),
        FIXTURES / "profile_complete.json",
        data_root=_derived_root_with_drug(tmp_path),
        generated_at="2026-07-14T00:00:00+00:00",
    )

    assert [option["name"] for option in document["prescription_options"]] == ["AZELEX"]
    assert document["prescription_options"][0]["therapy_plan_match"][
        "strength_band"
    ] == "verified_otc_or_labeled"


def test_reviewed_amount_must_match_the_verified_product_direction(tmp_path):
    analysis_path = _reviewed_therapy_analysis(tmp_path)
    data = json.loads(analysis_path.read_text())
    data["therapy_plan"]["primary"].update(
        amount="pea_sized", amount_source="test-reviewed:1",
    )
    analysis_path.write_text(json.dumps(data))

    document = run(
        analysis_path,
        FIXTURES / "profile_complete.json",
        data_root=_derived_root_with_drug(tmp_path, amount="thin_layer"),
        generated_at="2026-07-14T00:00:00+00:00",
    )

    assert document["prescription_options"] == []
    assert document["treatment_fulfillment"]["status"] == "unfilled"


@pytest.mark.parametrize(
    "band,listed",
    [("adapalene_0.1%_bp_2.5%", True), ("garbage_band", False)],
)
def test_combination_plan_binds_therapy_and_strength_band(tmp_path, band, listed):
    spl = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/x.xml"
    derived = _derived_root_with_drug(
        tmp_path,
        product_id="dailymed:x:1:adapalene-0.1%-benzoyl_peroxide-2.5%",
        name="Adapalene BP Combination",
        actives=["adapalene", "benzoyl_peroxide"],
        drug_actives=[
            {"name": "adapalene", "strength": "0.1%", "source": spl},
            {"name": "benzoyl_peroxide", "strength": "2.5%", "source": spl},
        ],
    )
    document = run(
        _reviewed_therapy_analysis(
            tmp_path, "adapalene_benzoyl_peroxide", band
        ),
        FIXTURES / "profile_complete.json", data_root=derived,
        generated_at="2026-07-14T00:00:00+00:00",
    )

    assert bool(document["prescription_options"]) is listed


def test_an_otc_drug_row_is_not_offered_as_a_prescription(tmp_path):
    analysis = _reviewed_therapy_analysis(tmp_path)
    document = run(analysis, FIXTURES / "profile_complete.json",
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
            "cadence": "per_label", "cadence_source": "https://example.test/minted",
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
    analysis = _reviewed_therapy_analysis(tmp_path)
    document = run(analysis, FIXTURES / "profile_complete.json",
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


def test_unplanned_retinoid_rows_never_enter_the_candidate_pool(tmp_path):
    profile = json.loads((FIXTURES / "profile_complete.json").read_text())
    profile["pregnancy_status"] = "pregnant"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    analysis = _reviewed_therapy_analysis(tmp_path, "azelaic_acid")
    document = run(analysis, path, generated_at="2026-07-14T00:00:00+00:00",
                   eligibility_mode="hybrid")

    pinned = RETINOL_PRODUCT_IDS | {RETINYL_ESTER_ONLY_PRODUCT_ID}
    placed = {s["product_id"] for r in document["routines"] for s in _steps(r)}
    assert not (placed & pinned)


def _derived_root_with_serum_facts(tmp_path, product_ids=("P427417", "P443842")):
    """Overlay mints the usage facts a serum needs to clear the gates
    (routine_roles/format/exposure/cadence/intended_areas) — the shape a
    verification batch produces — onto seed cosmetics."""
    import hashlib
    import shutil
    derived = tmp_path / "derived-serum"
    derived.mkdir()
    (derived / "catalog_full.json").write_bytes(
        (DATA / "catalog" / "seed_catalog.json").read_bytes()
    )
    shutil.copytree(DATA / "verification", derived / "verification")
    approved = json.loads((derived / "verification" / "approved.json").read_text())
    for pid in product_ids:
        body = f"synthetic serum label {pid}".encode()
        digest = hashlib.sha256(body).hexdigest()
        (derived / "verification" / "evidence" / digest).write_bytes(body)
        approved["products"].append({"product_id": pid, "assertions": [{
            "status": "approved", "reviewer_id": "reviewer-1", "reviewer_type": "agent",
            "approved_at": "2026-07-14T00:00:00Z", "retrieved_at": "2026-07-14T00:00:00Z",
            "source_url": "https://example.test/serum", "source_sha256": digest,
            "facts": {"routine_roles": ["serum"], "format": "serum",
                      "exposure": "leave_on", "cadence": "am_pm",
                      "cadence_source": "https://example.test/serum",
                      "intended_areas": ["face"]},
        }]})
    (derived / "verification" / "approved.json").write_text(json.dumps(approved))
    return derived


def test_grouped_concern_cannot_create_serum_or_treatment_candidates():
    from recsys.candidates import generate_candidates
    from recsys.signals import TargetConcern
    products, _ = load_catalog(DATA / "catalog" / "seed_catalog.json")
    targets = (TargetConcern("acne_inflammatory", 3, 0.9),)
    by_slot = generate_candidates(products, targets, K)
    assert by_slot["serum"] == []
    assert by_slot["treatment"] == []
    assert not generate_candidates(products, (), K)["serum"]


def test_verified_unrequested_serum_stays_out_of_the_routine(tmp_path):
    analysis = _reviewed_therapy_analysis(tmp_path, "benzoyl_peroxide", "2.5%")
    document = run(analysis, FIXTURES / "profile_complete.json",
                   data_root=_derived_root_with_serum_facts(tmp_path),
                   generated_at="2026-07-14T00:00:00+00:00")
    serum_steps = [s for r in document["routines"] for s in _steps(r)
                   if s["slot"] == "serum"]
    assert serum_steps == []
    assert document["selected_products"]["treatment"] == "P188306"


def test_retinol_serum_is_not_a_candidate_for_pregnant_profiles(tmp_path):
    analysis = _reviewed_therapy_analysis(tmp_path, "benzoyl_peroxide", "2.5%")
    profile = json.loads((FIXTURES / "profile_complete.json").read_text())
    profile["pregnancy_status"] = "pregnant"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    document = run(analysis, path,
                   data_root=_derived_root_with_serum_facts(tmp_path),
                   generated_at="2026-07-14T00:00:00+00:00")
    placed = {s["product_id"] for r in document["routines"] for s in _steps(r)}
    assert "P443842" not in placed
    assert not any(s["slot"] == "serum" for r in document["routines"] for s in _steps(r))


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
    assert document["data_versions"]["verification"]["products"] >= 18


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
