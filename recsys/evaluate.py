"""Golden-file evaluation harness for deterministic recommendation cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run


def canonicalize(document: dict) -> dict:
    value = json.loads(json.dumps(document))
    value["engine"]["git_commit"] = None
    catalog = value["data_versions"]["catalog"]
    catalog["path"] = Path(catalog["path"]).name
    verification = value["data_versions"].get("verification")
    if verification:
        verification["path"] = Path(verification["path"]).name
    return value


def evaluate(manifest_path: str | Path, *, update: bool = False) -> list[str]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "recsys-eval-1":
        raise ValueError("expected recsys-eval-1 manifest")
    failures = []
    for case in manifest.get("cases") or []:
        base = manifest_path.parent
        actual = canonicalize(run(
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
        elif not golden_path.exists() or json.loads(golden_path.read_text()) != actual:
            failures.append(case["id"])
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args(argv)
    failures = evaluate(args.manifest, update=args.update)
    if failures:
        print("golden mismatch: " + ", ".join(failures))
        return 1
    print("golden evaluation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
