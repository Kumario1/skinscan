import json
import math
from pathlib import Path

import pytest

from recsys.contracts import (
    KNOWN_ACTIVE_IDS,
    ContractViolation,
    load_analysis,
    resolve_profile,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_loads_real_analysis_artifact():
    analysis = load_analysis(FIXTURES / "analysis_v3_sample.json")
    assert len(analysis.concerns) == 4
    names = {c.concern for c in analysis.concerns}
    assert names == {"acne_comedonal", "acne_inflammatory", "acne_scarring", "hyperpigmentation"}
    assert analysis.triage_level == "routine_plus_review"
    assert len(analysis.referral_reasons) == 3
    assert analysis.skin_tone_bucket == "medium"
    assert any(o["professional_review"] for o in analysis.safety_observations)
    assert analysis.analysis_sha256


def test_rejects_wrong_schema_version(tmp_path):
    data = json.loads((FIXTURES / "analysis_v3_sample.json").read_text())
    data["schema_version"] = "2.0"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="schema_version"):
        load_analysis(bad)


def test_rejects_unknown_concern(tmp_path):
    data = json.loads((FIXTURES / "analysis_v3_sample.json").read_text())
    data["concerns"][0]["concern"] = "wrinkles"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="concern"):
        load_analysis(bad)


def test_rejects_duplicate_concerns(tmp_path):
    data = json.loads((FIXTURES / "analysis_v3_sample.json").read_text())
    data["concerns"].append(dict(data["concerns"][0]))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="duplicate"):
        load_analysis(bad)


@pytest.mark.parametrize("confidence", [math.nan, math.inf, -0.1, 1.1])
def test_rejects_invalid_confidence(tmp_path, confidence):
    data = json.loads((FIXTURES / "analysis_v3_sample.json").read_text())
    data["concerns"][0]["confidence"] = confidence
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="confidence"):
        load_analysis(bad)


@pytest.mark.parametrize("confidence", ["high", [0.9]])
def test_rejects_non_numeric_confidence(tmp_path, confidence):
    data = json.loads((FIXTURES / "analysis_v3_sample.json").read_text())
    data["concerns"][0]["confidence"] = confidence
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="confidence"):
        load_analysis(bad)


def test_rejects_boolean_severity(tmp_path):
    data = json.loads((FIXTURES / "analysis_v3_sample.json").read_text())
    data["concerns"][0]["severity"] = True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="severity"):
        load_analysis(bad)


def test_accepts_valid_numeric_confidence_and_integer_severity(tmp_path):
    data = json.loads((FIXTURES / "analysis_v3_sample.json").read_text())
    data["concerns"][0]["severity"] = 3
    data["concerns"][0]["confidence"] = 0.75
    good = tmp_path / "good.json"
    good.write_text(json.dumps(data))
    analysis = load_analysis(good)
    assert analysis.concerns[0].severity == 3
    assert analysis.concerns[0].confidence == 0.75


def test_profile_precedence_file_wins():
    analysis = load_analysis(FIXTURES / "analysis_v3_sample.json")
    profile = resolve_profile(FIXTURES / "profile_complete.json", analysis)
    assert profile.skin_type == "oily"
    assert profile.pregnancy_status == "not_pregnant"
    assert profile.tone_source == "self_report"
    assert profile.treatment_history == ()
    assert profile.acne_duration_weeks == 16
    assert profile.painful_or_deep_lesions is False
    assert profile.prior_scarring is False
    assert profile.source == "file"
    assert profile.profile_sha256


def test_profile_falls_back_to_analysis_input_profile():
    analysis = load_analysis(FIXTURES / "analysis_v3_sample.json")
    profile = resolve_profile(None, analysis)
    assert profile.source == "analysis.input_profile"
    assert profile.skin_type == "unknown"
    assert profile.pregnancy_status == "unknown"
    assert profile.tone_source == "unknown"
    assert profile.treatment_history == ()
    assert profile.acne_duration_weeks is None
    assert profile.painful_or_deep_lesions is None
    assert profile.prior_scarring is None


@pytest.mark.parametrize("declared", [
    "Retinol", "RETINOL", "retinol ", "salicylic acid", "tretinoin", "vitamin c",
])
def test_rejects_current_actives_outside_the_canonical_vocabulary(tmp_path, declared):
    """gates.py matches current_actives against product.actives by exact set
    intersection, and product actives are canonical snake_case ids. An
    un-normalized "Retinol" intersects nothing, so the duplicate-active HARD
    gate silently passes a retinol serum to someone already using retinol --
    a contract violation turned into a no-op on a safety gate. Loud is the only
    safe answer here: this module cannot guess which active "BHA" means (the
    INCI table reads it as an antioxidant, a user means salicylic acid), and
    guessing on a HARD gate is worse than refusing. src.recommendation.schema
    .UserProfile already raises on this exact input; recsys ported the gate
    without the contract that made it sound.
    """
    data = json.loads((FIXTURES / "profile_complete.json").read_text())
    data["current_actives"] = [declared]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    analysis = load_analysis(FIXTURES / "analysis_v3_sample.json")

    with pytest.raises(ContractViolation, match="current_actives"):
        resolve_profile(path, analysis)


def test_accepts_canonical_current_actives(tmp_path):
    """The ids a closed intake form is expected to submit must pass through
    unchanged, in the exact form the gate intersects against."""
    data = json.loads((FIXTURES / "profile_complete.json").read_text())
    data["current_actives"] = ["retinol", "salicylic_acid"]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))

    profile = resolve_profile(path, load_analysis(FIXTURES / "analysis_v3_sample.json"))
    assert profile.current_actives == ("retinol", "salicylic_acid")
    assert {"retinol", "salicylic_acid"} <= KNOWN_ACTIVE_IDS


def test_current_actives_vocabulary_is_enforced_on_the_analysis_fallback_path(tmp_path):
    """resolve_profile has two doors -- an explicit --profile file and
    analysis.input_profile -- and both build the Profile the gates trust. The
    fallback is the one no human hand-writes, so it is the one that rots.
    """
    data = json.loads((FIXTURES / "analysis_v3_sample.json").read_text())
    data["input_profile"]["current_actives"] = ["Retinol"]
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ContractViolation, match="current_actives"):
        resolve_profile(None, load_analysis(path))


def test_free_text_profile_fields_stay_open(tmp_path):
    """Only current_actives has a closed vocabulary. Allergies are matched
    against raw INCI by inci.allergy_matches, which normalizes and resolves
    synonyms itself -- "fragrance" is a real allergen and never a canonical
    active, so validating allergies against the actives vocabulary would reject
    the exact input the allergen gate exists to serve. Conditions and
    medications are matched against free-text overlay contraindications, which
    have no vocabulary to validate against either.
    """
    data = json.loads((FIXTURES / "profile_complete.json").read_text())
    data["allergies"] = ["fragrance", "BHA"]
    data["sensitivity_conditions"] = ["rosacea"]
    data["current_medications"] = ["warfarin"]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))

    profile = resolve_profile(path, load_analysis(FIXTURES / "analysis_v3_sample.json"))
    assert profile.allergies == ("fragrance", "BHA")
    assert profile.sensitivity_conditions == ("rosacea",)
    assert profile.current_medications == ("warfarin",)


@pytest.mark.parametrize("field,value", [
    ("tone_source", "estimated"),
    ("treatment_history", "retinol"),
    ("acne_duration_weeks", -1),
    ("acne_duration_weeks", True),
    ("painful_or_deep_lesions", "no"),
    ("prior_scarring", 1),
])
def test_rejects_invalid_profile_intake_values(tmp_path, field, value):
    data = json.loads((FIXTURES / "profile_complete.json").read_text())
    data[field] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    analysis = load_analysis(FIXTURES / "analysis_v3_sample.json")

    with pytest.raises(ContractViolation, match=field):
        resolve_profile(path, analysis)
