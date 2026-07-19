"""Offline checks for the agentrec engine, prompts, and persona fixtures. No network."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentrec import engine, prompts  # noqa: E402

PERSONA_DIR = ROOT / "agentrec" / "personas"
FIXTURES = sorted(PERSONA_DIR.glob("*.analysis.json"))
EXPECTED_TRIAGE = {
    "severe-nodular": "derm_first",
    "heavy-real": "routine_plus_review",
    "medium-oily": "routine_plus_review",
    "medium-dry": "routine_plus_review",
    "light-routine": "routine",
    "light-real": "routine_plus_review",
}


def _fixture(path):
    return json.loads(path.read_text())


def test_prompt_markers():
    for marker in ("derm_first", "review_sentiment", "STEP 4", "{image_section}",
                   "options_to_discuss_with_doctor"):
        assert marker in prompts.PROMPT_TEMPLATE, marker
    assert prompts.PROMPT_TEMPLATE.count("{image_section}") == 1
    for marker in ("hydroquinone", "isotretinoin", "one JSON object", "monitoring_only",
                   "pharmacy", "dosing", "ABCDE"):
        assert marker in prompts.SYSTEM_PROMPT, marker


def test_build_prompt_replace_keeps_schema_braces():
    with_images = engine.build_prompt([Path("/tmp/lesion_sheet.jpg"), Path("/tmp/detections.jpg")])
    assert "/tmp/lesion_sheet.jpg" in with_images and "Read" in with_images
    assert '"review_sentiment"' in with_images  # schema braces intact after .replace
    assert "{image_section}" not in with_images
    assert "No images" in engine.build_prompt([])


def test_extract_json():
    assert engine.extract_json('noise {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        engine.extract_json("no json here")


def test_fixtures_exist():
    assert len(FIXTURES) == 6, "run python -m agentrec.personas.make_personas first"
    index = json.loads((PERSONA_DIR / "personas.json").read_text())
    assert set(index) == set(EXPECTED_TRIAGE)
    for name, spec in index.items():
        assert (PERSONA_DIR / spec["analysis"]).exists()
        for rel in spec["images"]:
            assert rel.startswith("runs/e2e/"), rel


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_projection_on_fixture(path):
    ctx = engine.project_context(_fixture(path))
    assert set(ctx) <= set(engine.ALLOWED_KEYS)
    for banned in ("detections", "source_image_sha256", "semantic_inputs"):
        assert banned not in ctx
    assert len(json.dumps(ctx, separators=(",", ":"))) < 200_000


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_policy_consistency(path):
    """Mirror of decide_exact_label_care: fixtures must never drift from policy rules."""
    analysis = _fixture(path)
    counts = {f["lesion_type"]: f["count"] for f in analysis["lesion_findings"]}
    detected = {k for k, v in counts.items() if v > 0}
    if "nodule" in detected:
        expected = "derm_first"
    elif detected & {"atrophic_scar", "hypertrophic_scar", "melasma", "nevus", "other"}:
        expected = "routine_plus_review"
    else:
        expected = "routine"
    assert analysis["decision"]["triage_level"] == expected
    name = path.stem.replace(".analysis", "")
    assert analysis["decision"]["triage_level"] == EXPECTED_TRIAGE[name]
    statuses = {p["lesion_type"]: p["status"] for p in analysis["care_pathways"]}
    for lesion_type, count in counts.items():
        if count == 0:
            assert statuses[lesion_type] == "not_detected"
    if counts.get("nodule"):
        assert statuses["nodule"] == "clinician_only"
    if counts.get("nevus"):
        assert statuses["nevus"] == "monitoring_only"
    if name == "light-routine":
        assert analysis["decision"]["referral_reasons"] == []


def test_dry_persona_profile():
    dry = _fixture(PERSONA_DIR / "medium-dry.analysis.json")
    oily = _fixture(PERSONA_DIR / "medium-oily.analysis.json")
    assert dry["input_profile"]["skin_type"] == "dry"
    assert oily["input_profile"]["skin_type"] == "oily"
