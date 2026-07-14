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


def test_profile_precedence_file_wins():
    analysis = load_analysis(FIXTURES / "analysis_v3_sample.json")
    profile = resolve_profile(FIXTURES / "profile_complete.json", analysis)
    assert profile.skin_type == "oily"
    assert profile.pregnancy_status == "not_pregnant"
    assert profile.source == "file"
    assert profile.profile_sha256


def test_profile_falls_back_to_analysis_input_profile():
    analysis = load_analysis(FIXTURES / "analysis_v3_sample.json")
    profile = resolve_profile(None, analysis)
    assert profile.source == "analysis.input_profile"
    assert profile.skin_type == "unknown"
    assert profile.pregnancy_status == "unknown"
