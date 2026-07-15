"""CLI: python -m recsys recommend --analysis <analysis.json> [...]"""
from __future__ import annotations

import argparse
import sys

from .pipeline import DEFAULT_DATA_ROOT, emit, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recommend", help="analysis.json + profile -> recommendations.json")
    rec.add_argument("--analysis", required=True, help="analysis.json (schema 3) from the pipeline")
    rec.add_argument("--profile", default=None, help="profile JSON (falls back to analysis input_profile)")
    rec.add_argument("--catalog", default=None, help="catalog JSON (default: bundled seed catalog)")
    rec.add_argument("--data-root", default=None, help=f"recsys data dir (default: {DEFAULT_DATA_ROOT})")
    rec.add_argument("--out", default="recommendations.json", help="output path")
    rec.add_argument("--generated-at", default=None, help="pin the timestamp (deterministic runs)")
    rec.add_argument("--eligibility-mode", default="strict", choices=("strict", "hybrid"),
                     help="strict: only evidence-verified products (default); "
                          "hybrid: whole catalog by category, verified products ranked/labeled higher")
    args = parser.parse_args(argv)

    document = run(
        analysis_path=args.analysis,
        profile_path=args.profile,
        catalog_path=args.catalog,
        data_root=args.data_root,
        generated_at=args.generated_at,
        eligibility_mode=args.eligibility_mode,
    )
    out = emit(document, args.out)
    routines = document.get("routines") or []
    print(f"{document['status']}: {len(routines)} routines -> {out}")
    for routine in routines:
        total = routine.get("total_price_usd")
        price = f"${total:.2f}" if total is not None else "n/a"
        print(f"  - {routine['archetype']}: {routine['slot_count']} steps, {price}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
