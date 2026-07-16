"""Golden-file evaluation harness for deterministic recommendation cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run

_MISSING = object()


def _placements(steps: list[dict]) -> list[str]:
    return [f"{step['slot']}:{step['product_id']}" for step in steps]


def decisions(document: dict) -> dict:
    """Project a document onto the choices a reader can actually rule on.

    Scores, signal values and the per-product veto entries stay out. Nothing
    says a why.score of 0.579844 is the right number, so a diff there is a
    notification rather than a failure, and a golden carrying hundreds of them
    gets re-pinned by reflex instead of read. What survives is what a reader can
    answer: which product filled which slot in which session, which archetypes
    fell away and why, and which veto codes fired at all.
    """
    veto_log = document.get("veto_log") or {}
    compose_vetoes = (veto_log.get("compose") or {}).values()
    return {
        "schema_version": document["schema_version"],
        "status": document["status"],
        "framing": document["framing"],
        "triage": document["triage"],
        "profile_used": document["profile_used"],
        "target_concerns": document["target_concerns"],
        "warnings": document["warnings"],
        "routines": [
            {
                "archetype": routine["archetype"],
                "am": _placements(routine["am"]),
                "pm": _placements(routine["pm"]),
                "per_label": _placements(routine["per_label"]),
                "notes": routine["notes"],
                # Keyed by rule name, not reduced to "all passed": a check that
                # drops out of the list leaves every verdict in it still true.
                "safety_checks": {
                    check["rule"]: check["passed"] for check in routine["safety_checks"]
                },
            }
            for routine in document["routines"]
        ],
        "unavailable_archetypes": document.get("unavailable_archetypes") or [],
        "prescription_options": [
            option["name"] for option in document.get("prescription_options") or []
        ],
        "veto_reasons": {
            "profile": sorted({v["reason"] for v in veto_log.get("profile") or []}),
            "compose": sorted({v["reason"] for vetoes in compose_vetoes for v in vetoes}),
        },
    }


def _show(value) -> str:
    return "(absent)" if value is _MISSING else json.dumps(value)


def _label(items: list, index: int) -> str:
    """Name a list entry by its archetype where it has one: routines and
    unavailable_archetypes both read better keyed by name than by position."""
    if index < len(items) and isinstance(items[index], dict):
        archetype = items[index].get("archetype")
        if archetype is not None:
            return str(archetype)
    return str(index)


def _contents_differences(expected: list, actual: list, path: str) -> list[str]:
    """A list of plain values diffs by contents first, then by order. Diffing it
    position by position turns one dropped veto code into a cascade of
    shifted-by-one lines, which is the unreadable diff this golden exists to
    avoid: the reader has to reconstruct that one entry moved."""
    removed = list(expected)
    added = []
    for value in actual:
        if value in removed:
            removed.remove(value)
        else:
            added.append(value)
    lines = [f"  {path}: removed {_show(value)}" for value in removed]
    lines += [f"  {path}: added {_show(value)}" for value in added]
    if not lines and expected != actual:
        lines = [f"  {path}: reordered to {_show(actual)}"]
    return lines


def differences(expected, actual, path: str = "") -> list[str]:
    """Every leaf that moved, named by where it sits."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return [
            line
            for key in sorted(set(expected) | set(actual))
            for line in differences(
                expected.get(key, _MISSING),
                actual.get(key, _MISSING),
                f"{path}.{key}" if path else key,
            )
        ]
    if isinstance(expected, list) and isinstance(actual, list):
        if all(not isinstance(v, (dict, list)) for v in expected + actual):
            return _contents_differences(expected, actual, path)
        return [
            line
            for index in range(max(len(expected), len(actual)))
            for line in differences(
                expected[index] if index < len(expected) else _MISSING,
                actual[index] if index < len(actual) else _MISSING,
                f"{path}[{_label(expected if index < len(expected) else actual, index)}]",
            )
        ]
    if expected == actual:
        return []
    return [f"  {path}: expected {_show(expected)}, actual {_show(actual)}"]


def evaluate(manifest_path: str | Path, *, update: bool = False) -> list[str]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "recsys-eval-1":
        raise ValueError("expected recsys-eval-1 manifest")
    failures = []
    for case in manifest.get("cases") or []:
        base = manifest_path.parent
        actual = decisions(run(
            base / case["analysis"],
            base / case["profile"] if case.get("profile") else None,
            generated_at=case["generated_at"],
        ))
        golden_path = base / case["golden"]
        if update:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(
                json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        elif not golden_path.exists():
            failures.append(f"{case['id']}: no golden at {golden_path}")
        else:
            report = differences(json.loads(golden_path.read_text()), actual)
            if report:
                failures.append("\n".join([f"{case['id']}:"] + report))
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args(argv)
    failures = evaluate(args.manifest, update=args.update)
    if failures:
        print("golden mismatch — the recommendation changed:")
        for failure in failures:
            print(failure)
        print("\nRe-run with --update once every line above is a change you meant.")
        return 1
    print("golden evaluation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
