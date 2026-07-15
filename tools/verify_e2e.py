#!/usr/bin/env python3
"""Check every stage of the recsys recommendation on a real analysis.json.

The unit tests prove each stage against fixtures. This proves the whole thing
against the real catalog, the real signal stores and a real photo's analysis --
the wiring the fixtures cannot see. Re-runnable; prints one PASS/FAIL table.

    python tools/verify_e2e.py                        # defaults below
    python tools/verify_e2e.py --analysis <a.json> --data-root recsys/data/derived
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.catalog import load_catalog  # noqa: E402
from recsys.contracts import load_analysis, resolve_profile  # noqa: E402
from recsys.knowledge import load_knowledge  # noqa: E402
from recsys.pipeline import run  # noqa: E402

PINNED = "2026-07-15T00:00:00+00:00"
results: list[tuple[str, bool, str]] = []


def check(stage: str, ok: bool, detail: str = "") -> bool:
    results.append((stage, bool(ok), detail))
    return bool(ok)


def _steps(routine: dict) -> list[dict]:
    return routine["am"] + routine["pm"] + routine["per_label"]


def verify(analysis_path: Path, data_root: Path, mode: str, runs: int) -> dict:
    knowledge = load_knowledge(ROOT / "recsys" / "data" / "knowledge")
    analysis = load_analysis(analysis_path)

    # 1. inputs
    check("inputs: analysis loads + hashes",
          bool(analysis.analysis_sha256) and bool(analysis.concerns),
          f"{len(analysis.concerns)} concerns, triage={analysis.triage_level}")
    profile = resolve_profile(None, analysis)
    check("inputs: profile resolves (unknown-safe)", profile.source is not None,
          f"source={profile.source}, pregnancy={profile.pregnancy_status}")

    # 2. catalogs -- resolved exactly as pipeline.py does, so this harness cannot
    # silently check a different catalog than the engine used.
    full = data_root / "catalog_full.json"
    catalog_path = full if full.exists() else data_root / "catalog" / "seed_catalog.json"
    cosmetics, _ = load_catalog(catalog_path)
    check("catalog: cosmetics load under the INCI contract", bool(cosmetics),
          f"{len(cosmetics)} products from {catalog_path.name}")
    # The drug catalog is optional, exactly as the pipeline treats it: present it
    # must load, absent the engine is cosmetics-only and that is a valid config.
    drug_path = data_root / "catalog_drug.json"
    drug = []
    if drug_path.exists():
        drug, _ = load_catalog(drug_path)
        check("catalog: drug rows load under the label contract", bool(drug),
              f"{len(drug)} rows, {sum(1 for p in drug if p.otc_drug is False)} prescription")
    else:
        check("catalog: drug rows (optional) not configured", True,
              f"no {drug_path.name} — cosmetics-only run")

    document = run(analysis_path, None, data_root=data_root,
                   generated_at=PINNED, eligibility_mode=mode)

    # 3. signal stores -- the silent failure mode: a catalog_sha256 mismatch
    # skips a store with only a warning and the ranker scores blind.
    signals = document["data_versions"]["signals"]
    names = {s["name"] for s in signals}
    catalog_sha = document["data_versions"]["catalog"]["sha256"]
    check("signals: every store loaded and bound to this catalog",
          names == {"ingredient_analysis", "popularity", "review_stats"}
          and all(s["catalog_sha256"] == catalog_sha for s in signals),
          f"{sorted(names)}")
    check("signals: no store was skipped", not document["warnings"],
          document["warnings"][0][:60] if document["warnings"] else "no warnings")

    # 4. knowledge + overlay
    check("knowledge: safety tables present",
          bool(knowledge.retinoids and knowledge.concern_actives and knowledge.archetypes),
          f"{len(knowledge.archetypes)} archetypes, {len(knowledge.retinoids)} retinoids")
    verification = document["data_versions"]["verification"]
    check("verification: overlay loads", bool(verification.get("products")),
          f"{verification.get('products')} verified products")

    # 5. targets
    targets = document["target_concerns"]
    check("targets: concerns selected by severity", bool(targets),
          ", ".join(f"{t['concern']}:{t['severity']}" for t in targets))

    # 6. gates -- fail-closed reason codes
    vetoes = document["veto_log"]["profile"]
    reasons = sorted({v["reason"].split(":")[0] for v in vetoes})
    check("gates: hard vetoes fire with reason codes", bool(vetoes),
          f"{len(vetoes)} vetoes: {reasons[:3]}")

    # Either routines, or an explicit reason per archetype. Never silently empty:
    # strict on the full catalog legitimately yields nothing, because only the
    # evidence-verified products are eligible and there are few of them.
    routines = document["routines"]
    unavailable = document.get("unavailable_archetypes") or []
    check("compose: routines built, or unavailability explained per archetype",
          bool(routines) or (bool(unavailable) and all(u["reasons"] for u in unavailable)),
          f"{len(routines)}/{len(knowledge.archetypes)} archetypes, status={document['status']}"
          + (f", unavailable: {unavailable[0]['reasons'][0]}" if not routines else ""))

    # 7. scoring is decomposable -- every number traceable to a named store
    decomposable = True
    for routine in routines:
        for step in _steps(routine):
            why = step["why"]
            if not why["signals"] or not any(s["evidence"] for s in why["signals"]):
                decomposable = False
    check("scoring: every step decomposes into named signals", decomposable)
    check("explain: every step carries an evidence-backed why",
          all(step["why"]["summary"] for r in routines for step in _steps(r)))

    # 8. safety invariants, checked on the output rather than trusted
    catalog = {p.product_id: p for p in cosmetics + drug}
    safety_ok = all(c["passed"] for r in routines for c in r["safety_checks"])
    am_retinoid = any(
        set(catalog[s["product_id"]].actives) & knowledge.retinoids
        for r in routines for s in r["am"]
    )
    check("safety: composer invariants hold (SPF AM, retinoid PM, no conflicts)",
          safety_ok and not am_retinoid)

    # 9. prescriptions: listed, never placed, never in a total
    options = document["prescription_options"]
    placed = [s for r in routines for s in _steps(r) if s["prescription"]]
    unpriced = [s for r in routines for s in _steps(r) if s["price_usd"] is None]
    check("prescription: every option is well-formed and targeted",
          all(o["actives"] and o["targets"] and o["label_source"] and "doctor" in o["note"]
              for o in options),
          ", ".join(o["name"] for o in options) or "none listed")
    check("prescription: never placed into a routine", not placed)
    check("prescription: no unpriced step distorts a total", not unpriced)

    # 10. determinism, across processes rather than in-process
    out = Path("/tmp/_verify_e2e")
    out.mkdir(exist_ok=True)
    digests = set()
    for i in range(runs):
        target = out / f"r{i}.json"
        subprocess.run(
            [sys.executable, "-m", "recsys", "recommend", "--analysis", str(analysis_path),
             "--data-root", str(data_root), "--eligibility-mode", mode,
             "--generated-at", PINNED, "--out", str(target)],
            cwd=ROOT, capture_output=True, check=True,
        )
        import hashlib
        digests.add(hashlib.sha256(target.read_bytes()).hexdigest())
    check(f"determinism: {runs} runs, {runs} processes, identical bytes",
          len(digests) == 1, f"{len(digests)} distinct sha256")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path,
                        default=ROOT / "runs/e2e/rx_test_1/analysis.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "recsys/data/derived")
    parser.add_argument("--mode", default="hybrid", choices=("strict", "hybrid"))
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print(f"analysis : {args.analysis}")
    print(f"data-root: {args.data_root}   mode: {args.mode}\n")
    verify(args.analysis, args.data_root, args.mode, args.runs)

    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} stages pass")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
