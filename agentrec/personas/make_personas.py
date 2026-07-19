"""Derive the six persona analysis fixtures from saved e2e runs.

Usage:
    python -m agentrec.personas.make_personas

Rerunnable; overwrites the committed fixtures next to this file. Needs the gitignored
saved runs under a local checkout's runs/e2e (auto-located via the shared git dir).
Synthetic mutations are recomputed through the real policy code
(src.recommendation.lesion_care) so pathways/decision/therapy_plan can never drift
from policy semantics.
"""

import json
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from agentrec.engine import find_runs_root  # noqa: E402
from src.recommendation.lesion_care import (  # noqa: E402
    build_care_pathways,
    decide_exact_label_care,
    exact_label_therapy_plan,
    load_lesion_care_policy,
)

BASES = {
    "low-real": "runs/e2e/real-recommendations-v4-20260716/low-real",
    "high-real": "runs/e2e/real-recommendations-v4-20260716/high-real",
    "compact": "runs/e2e/postmerge-v4-authorized-20260716/compact-lesions",
}
IMAGE_NAMES = ("lesion_sheet.jpg", "detections.jpg")


def _load_policy():
    policy = load_lesion_care_policy(
        ROOT / "lesion_care_policy.proposed.json",
        report_path=ROOT / "LESION_CARE_EVIDENCE_REPORT.md",
        environment="test",
        input_types=("synthetic_profile", "fixture_image"),
    )
    if not policy.scope_authorized:
        raise SystemExit(f"policy scope not authorized: {policy.scope_reasons}")
    return policy


def _set_finding(analysis, lesion_type, count, regions, mean_conf, max_conf):
    for row in analysis["lesion_findings"]:
        if row["lesion_type"] == lesion_type:
            row.update(
                count=count,
                regions=sorted(regions),
                mean_detector_confidence=mean_conf,
                max_detector_confidence=max_conf,
                evidence_source="synthetic_fixture",
            )
            return
    raise KeyError(lesion_type)


def _zero_findings_except(analysis, keep):
    for row in analysis["lesion_findings"]:
        if row["lesion_type"] not in keep:
            row.update(
                count=0, regions=[],
                mean_detector_confidence=None, max_detector_confidence=None,
            )


def _concern(concern, labels, regions, severity, confidence):
    return {
        "concern": concern,
        "regions": sorted(regions),
        "severity": severity,
        "confidence": confidence,
        "lesion_count": sum(labels.values()),
        "evidence": {
            "labels": labels,
            "max_confidence": confidence,
            "affected_region_count": len(regions),
            "source": "synthetic_fixture",
        },
    }


def _recompute(analysis, policy):
    pathways = build_care_pathways(
        analysis["lesion_findings"], analysis["input_profile"], policy
    )
    analysis["care_pathways"] = pathways
    analysis["decision"] = {
        **analysis["decision"],
        **decide_exact_label_care(analysis["lesion_findings"], pathways),
    }
    analysis["therapy_plan"] = {
        **analysis["therapy_plan"],
        **exact_label_therapy_plan(pathways, policy),
    }


def _assert_recompute_faithful(analysis, original, name):
    for key in ("triage_level", "referral_reasons", "therapy_disposition"):
        assert analysis["decision"][key] == original["decision"][key], (
            name, key, analysis["decision"][key], original["decision"][key])
    orig_status = {p["lesion_type"]: p["status"] for p in original["care_pathways"]}
    new_status = {p["lesion_type"]: p["status"] for p in analysis["care_pathways"]}
    assert new_status == orig_status, (name, new_status, orig_status)


def build_personas(runs_root, policy):
    bases = {}
    for key, rel in BASES.items():
        path = runs_root / rel / "analysis.json"
        bases[key] = json.loads(path.read_text())

    def fresh(key):
        return json.loads(json.dumps(bases[key]))

    personas = {}

    # 1. severe-nodular: high-real + nodules -> derm_first. No images (photos lack nodules).
    severe = fresh("high-real")
    _set_finding(severe, "nodule", 3, ["chin_jaw", "left_cheek"], 0.82, 0.91)
    severe["input_profile"]["painful_or_deep_lesions"] = True
    severe["concerns"] = severe["concerns"] + [
        _concern("acne_cystic", {"nodule": 3}, ["chin_jaw", "left_cheek"], 4, 0.91)
    ]
    _recompute(severe, policy)
    assert severe["decision"]["triage_level"] == "derm_first", severe["decision"]
    personas["severe-nodular"] = (severe, [])

    # 2. heavy-real: high-real verbatim (recompute must be a no-op).
    heavy = fresh("high-real")
    _recompute(heavy, policy)
    _assert_recompute_faithful(heavy, bases["high-real"], "heavy-real")
    personas["heavy-real"] = (heavy, [f"{BASES['high-real']}/{n}" for n in IMAGE_NAMES])

    # 3. medium-oily: low-real verbatim.
    medium = fresh("low-real")
    _recompute(medium, policy)
    _assert_recompute_faithful(medium, bases["low-real"], "medium-oily")
    personas["medium-oily"] = (medium, [f"{BASES['low-real']}/{n}" for n in IMAGE_NAMES])

    # 4. medium-dry: low-real with a dry-skin profile (photos unaffected by profile).
    dry = fresh("low-real")
    dry["input_profile"]["skin_type"] = "dry"
    _recompute(dry, policy)
    _assert_recompute_faithful(dry, bases["low-real"], "medium-dry")
    personas["medium-dry"] = (dry, [f"{BASES['low-real']}/{n}" for n in IMAGE_NAMES])

    # 5. light-routine: comedones+papule only -> triage "routine". No images.
    light = fresh("low-real")
    _zero_findings_except(light, keep=())
    _set_finding(light, "closed_comedo", 3, ["forehead", "chin_jaw"], 0.71, 0.86)
    _set_finding(light, "papule", 1, ["left_cheek"], 0.74, 0.74)
    light["concerns"] = [
        _concern("acne_comedonal", {"closed_comedo": 3}, ["forehead", "chin_jaw"], 1, 0.86),
        _concern("acne_inflammatory", {"papule": 1}, ["left_cheek"], 1, 0.74),
    ]
    _recompute(light, policy)
    assert light["decision"]["triage_level"] == "routine", light["decision"]
    assert light["decision"]["referral_reasons"] == [], light["decision"]
    personas["light-routine"] = (light, [])

    # 6. light-real: compact-lesions verbatim.
    light_real = fresh("compact")
    _recompute(light_real, policy)
    _assert_recompute_faithful(light_real, bases["compact"], "light-real")
    personas["light-real"] = (light_real, [f"{BASES['compact']}/{n}" for n in IMAGE_NAMES])

    return personas


def main():
    runs_root = find_runs_root()
    if runs_root is None:
        raise SystemExit("runs/e2e not found in this or the main checkout")
    policy = _load_policy()
    personas = build_personas(runs_root, policy)

    index = {}
    for name, (analysis, images) in personas.items():
        fixture = PKG_DIR / f"{name}.analysis.json"
        fixture.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
        index[name] = {
            "analysis": fixture.name,
            "images": images,  # relative to a checkout containing runs/ (gitignored)
            "triage_level": analysis["decision"]["triage_level"],
        }
    (PKG_DIR / "personas.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(personas)} personas -> {PKG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
