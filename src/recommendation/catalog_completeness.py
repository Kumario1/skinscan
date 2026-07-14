"""Report verified catalog coverage and fail when release inventory is incomplete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .import_catalog import build_completeness_report, load_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", nargs="+", type=Path)
    parser.add_argument("--support-minimum", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    products = [product for path in args.catalog for product in load_catalog(path)]
    report = build_completeness_report(products, support_minimum=args.support_minimum)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
