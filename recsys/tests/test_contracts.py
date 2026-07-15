import json
import math
from pathlib import Path

import pytest

from recsys.contracts import ContractViolation, load_analysis, resolve_profile

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
