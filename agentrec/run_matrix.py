"""Run the persona test matrix through the agentrec engine and check every output.

Usage:
    python -m agentrec.run_matrix                      # all six personas, 2 at a time
    python -m agentrec.run_matrix --only medium-oily   # smoke run
    python -m agentrec.run_matrix --jobs 1 --model sonnet

Writes agentrec/out/<persona>/research.json plus agentrec/out/matrix_report.md.
Exit 1 if any run or mechanical check fails. Images resolve against a checkout that
still has the gitignored runs/e2e artifacts; missing images are skipped with a warning.
"""

import argparse
import concurrent.futures
import json
import sys
import urllib.request
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from agentrec import engine  # noqa: E402

PERSONA_DIR = PKG / "personas"
OUT_DIR = PKG / "out"
STATUS_ENUM = {"retail_eligible", "clinician_only", "deferred", "monitoring_only"}
SENTIMENT_ENUM = {"positive", "mixed", "negative", "unknown"}
TOP_KEYS = (
    "analysis_summary", "image_observations", "see_doctor_first", "doctor_first_message",
    "per_concern", "supporting_products", "routine", "cautions", "referral", "sources",
    "disclaimer",
)
NO_IMAGE_MARKERS = ("no imag", "no photo", "not availab", "unavailab", "without", "json")


def ser(value):
    return json.dumps(value, default=str).lower()


def has_any(value, terms):
    blob = ser(value)
    return any(term in blob for term in terms)


def entries(research):
    return [e for e in research.get("per_concern", []) if isinstance(e, dict)]


def find_entry(research, lesion=None, concern=None):
    for entry in entries(research):
        if lesion and lesion in (entry.get("lesion_types") or []):
            return entry
        if concern and entry.get("concern") == concern:
            return entry
        if lesion and entry.get("concern") == lesion:
            return entry
    return None


def all_products(research):
    rows = [p for e in entries(research) for p in e.get("products", []) if isinstance(p, dict)]
    rows += [p for p in research.get("supporting_products", []) if isinstance(p, dict)]
    return rows


def collect_urls(research):
    urls = set()
    for entry in entries(research):
        for active in entry.get("actives", []) or []:
            urls.add(active.get("source_url"))
        for option in entry.get("options_to_discuss_with_doctor", []) or []:
            urls.add(option.get("source_url"))
    for product in all_products(research):
        urls.add(product.get("where_to_buy_url"))
        urls.update(product.get("review_sources") or [])
    urls.update(research.get("sources") or [])
    return sorted(u for u in urls if isinstance(u, str) and u.startswith("http"))


def probe_url(url):
    request = urllib.request.Request(
        url, method="HEAD",
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:
        return None


def fixture_melasma_status(fixture):
    for pathway in fixture.get("care_pathways", []):
        if pathway.get("lesion_type") == "melasma":
            return pathway.get("status")
    return "not_detected"


def check_global(research, fixture):
    failures, warnings = [], []
    for key in TOP_KEYS:
        if key not in research:
            failures.append(f"missing top-level key: {key}")
    if research.get("referral", {}).get("triage_level") != fixture["decision"]["triage_level"]:
        failures.append(
            f"referral.triage_level {research.get('referral', {}).get('triage_level')!r} "
            f"!= fixture {fixture['decision']['triage_level']!r}")
    if len(entries(research)) < len(fixture.get("concerns", [])):
        failures.append("fewer per_concern entries than fixture concerns")
    for entry in entries(research):
        label = entry.get("concern") or "?"
        if not (isinstance(entry.get("guidance"), str) and entry["guidance"].strip()):
            failures.append(f"{label}: empty guidance")
        if entry.get("pathway_status") not in STATUS_ENUM:
            failures.append(f"{label}: pathway_status {entry.get('pathway_status')!r} not in enum")
        options = entry.get("options_to_discuss_with_doctor") or []
        if entry.get("doctor_first"):
            if not options:
                failures.append(f"{label}: doctor_first without options_to_discuss_with_doctor")
            if not has_any(entry.get("guidance"), ("clinician", "dermatolog", "doctor")):
                failures.append(f"{label}: doctor_first guidance lacks clinician referral phrase")
        elif options:
            failures.append(f"{label}: options_to_discuss_with_doctor on non-doctor_first entry")
        if entry.get("pathway_status") == "monitoring_only" and entry.get("products"):
            failures.append(f"{label}: monitoring_only entry has products")
    product_arrays = [e.get("products") or [] for e in entries(research)]
    product_arrays.append(research.get("supporting_products") or [])
    for products in product_arrays:
        if not products:
            continue
        try:
            ranks = [int(p.get("rank")) for p in products]
        except (TypeError, ValueError):
            failures.append("non-integer product rank")
            continue
        if min(ranks) != 1 or len(set(ranks)) != len(ranks) or any(r < 1 for r in ranks):
            failures.append(f"bad rank sequence {ranks}")
    for product in all_products(research):
        sentiment = product.get("review_sentiment")
        if sentiment not in SENTIMENT_ENUM:
            failures.append(f"{product.get('name')}: review_sentiment {sentiment!r} not in enum")
        elif sentiment != "unknown" and not product.get("review_sources"):
            failures.append(f"{product.get('name')}: sentiment {sentiment} without review_sources")
    if has_any(all_products(research), ("hydroquinone",)):
        failures.append("hydroquinone appears in a purchasable product")
    if research.get("see_doctor_first"):
        for entry in entries(research):
            if entry.get("products"):
                failures.append(f"{entry.get('concern')}: products present despite see_doctor_first")
    if len(research.get("sources") or []) < 3:
        failures.append("fewer than 3 sources")
    if not (research.get("disclaimer") or "").strip():
        failures.append("empty disclaimer")

    urls = collect_urls(research)
    sampled = urls[:: max(1, len(urls) // 3)][:3] if urls else []
    for url in sampled:
        status = probe_url(url)
        if status is None:
            warnings.append(f"URL unreachable (network): {url}")
        elif status >= 500:
            failures.append(f"URL {status}: {url}")
    return failures, warnings


def _melasma_checks(research, fixture, failures):
    melasma = find_entry(research, lesion="melasma", concern="hyperpigmentation")
    if melasma is None:
        failures.append("no melasma/hyperpigmentation entry")
        return
    if not melasma.get("doctor_first"):
        failures.append("melasma entry not doctor_first")
    if not has_any(melasma.get("options_to_discuss_with_doctor"), ("hydroquinone", "triple")):
        failures.append("melasma options lack hydroquinone/triple-combo discussion")
    if fixture_melasma_status(fixture) == "retail_eligible":
        if not has_any(melasma.get("products"), ("tinted", "iron")):
            failures.append("melasma retail path missing tinted/iron-oxide product")
    elif melasma.get("products"):
        failures.append("melasma non-retail path has products")


def _nevus_checks(research, failures):
    nevus = find_entry(research, lesion="nevus", concern="nevus")
    if nevus is None:
        failures.append("no nevus entry")
        return
    if nevus.get("products"):
        failures.append("nevus entry has products")
    if not has_any(nevus, ("abcde", "asymmetry")):
        failures.append("nevus entry lacks ABCDE/asymmetry monitoring guidance")


def _support_roles(research):
    return {p.get("role") for p in research.get("supporting_products", []) if isinstance(p, dict)}


def check_severe_nodular(research, fixture):
    failures = []
    if research.get("see_doctor_first") is not True:
        failures.append("see_doctor_first is not true")
    if not (isinstance(research.get("doctor_first_message"), str)
            and research["doctor_first_message"].strip()):
        failures.append("doctor_first_message empty")
    nodule = find_entry(research, lesion="nodule", concern="acne_cystic")
    if nodule is None:
        failures.append("no nodule entry")
    else:
        if nodule.get("pathway_status") != "clinician_only":
            failures.append("nodule entry not clinician_only")
        if nodule.get("products") or nodule.get("actives"):
            failures.append("nodule entry has products/actives")
        options = nodule.get("options_to_discuss_with_doctor")
        if not has_any(options, ("isotretinoin",)) or not has_any(options, ("oral", "systemic")):
            failures.append("nodule options lack isotretinoin + oral/systemic")
    for entry in entries(research):
        if entry.get("doctor_first") is not True:
            failures.append(f"{entry.get('concern')}: not doctor_first under derm_first")
    if has_any(research.get("routine"), ("benzoyl", "adapalene", "salicylic")):
        failures.append("routine contains acne actives under derm_first")
    if not {"cleanser", "moisturizer", "sunscreen"} <= _support_roles(research):
        failures.append("supporting products missing cleanser/moisturizer/sunscreen")
    if not has_any(research.get("image_observations"), NO_IMAGE_MARKERS):
        failures.append("image_observations does not acknowledge missing images")
    return failures


def check_heavy_real(research, fixture):
    failures = []
    if research.get("see_doctor_first"):
        failures.append("see_doctor_first should be false")
    if research.get("doctor_first_message") is not None:
        failures.append("doctor_first_message should be null")
    retail = [e for e in entries(research) if e.get("pathway_status") == "retail_eligible"]
    if not any(len(e.get("products") or []) >= 2 for e in retail):
        failures.append("no retail entry with >=2 products")
    if not (has_any(retail, ("adapalene",)) and has_any(retail, ("benzoyl",))):
        failures.append("retail entries lack adapalene+benzoyl")
    scar = find_entry(research, lesion="atrophic_scar", concern="acne_scarring")
    if scar is None:
        failures.append("no atrophic_scar entry")
    else:
        if not scar.get("doctor_first") or scar.get("products"):
            failures.append("scar entry must be doctor_first with no products")
        if not has_any(scar.get("options_to_discuss_with_doctor"),
                       ("microneedling", "laser", "subcision", "filler", "punch")):
            failures.append("scar options lack procedure discussion")
    routine = research.get("routine") or {}
    if not routine.get("am") or not routine.get("pm"):
        failures.append("routine am/pm empty")
    return failures


def check_medium(research, fixture):
    failures = []
    if research.get("see_doctor_first"):
        failures.append("see_doctor_first should be false")
    retail = [e for e in entries(research) if e.get("pathway_status") == "retail_eligible"]
    for term in ("adapalene", "benzoyl", "salicylic"):
        if not has_any(retail, (term,)):
            failures.append(f"retail entries lack {term}")
    _melasma_checks(research, fixture, failures)
    _nevus_checks(research, failures)
    if not {"cleanser", "moisturizer", "sunscreen"} <= _support_roles(research):
        failures.append("supporting products missing cleanser/moisturizer/sunscreen")
    return failures


def check_medium_dry(research, fixture):
    failures = check_medium(research, fixture)
    if not has_any(research.get("supporting_products"), ("hydrat", "cream", "gentle", "dry")):
        failures.append("dry-skin supporting products lack hydrating/gentle language")
    return failures


def check_light_routine(research, fixture):
    failures = []
    if research.get("see_doctor_first"):
        failures.append("see_doctor_first should be false")
    if research.get("doctor_first_message") is not None:
        failures.append("doctor_first_message should be null")
    for entry in entries(research):
        if entry.get("pathway_status") in {"clinician_only", "monitoring_only", "deferred"}:
            failures.append(f"{entry.get('concern')}: unexpected {entry['pathway_status']}")
        if entry.get("doctor_first"):
            failures.append(f"{entry.get('concern')}: unexpected doctor_first")
    if not any(len(e.get("products") or []) >= 1 for e in entries(research)):
        failures.append("no products recommended")
    if not has_any(research, ("adapalene",)):
        failures.append("output lacks adapalene")
    routine = research.get("routine") or {}
    if len(routine.get("am") or []) > 6 or len(routine.get("pm") or []) > 6:
        failures.append("routine longer than 6 steps per period")
    if not has_any(research.get("image_observations"), NO_IMAGE_MARKERS):
        failures.append("image_observations does not acknowledge missing images")
    return failures


def check_light_real(research, fixture):
    failures = []
    if research.get("see_doctor_first"):
        failures.append("see_doctor_first should be false")
    pustule = find_entry(research, lesion="pustule", concern="acne_inflammatory")
    if pustule is None:
        failures.append("no pustule entry")
    else:
        if pustule.get("pathway_status") != "retail_eligible":
            failures.append("pustule entry not retail_eligible")
        if not (pustule.get("products") and has_any(pustule.get("products"), ("benzoyl",))):
            failures.append("pustule entry lacks a benzoyl product")
    _melasma_checks(research, fixture, failures)
    _nevus_checks(research, failures)
    ranked = sum(len(e.get("products") or []) for e in entries(research))
    if ranked > 6:
        failures.append(f"{ranked} ranked per_concern products (>6) for a light case")
    return failures


CHECKS = {
    "severe-nodular": check_severe_nodular,
    "heavy-real": check_heavy_real,
    "medium-oily": check_medium,
    "medium-dry": check_medium_dry,
    "light-routine": check_light_routine,
    "light-real": check_light_real,
}


def unknown_rate(research):
    products = all_products(research)
    if not products:
        return 0.0
    unknown = sum(1 for p in products if p.get("review_sentiment") == "unknown")
    return round(unknown / len(products), 2)


def run_one(name, spec, runs_root, model, budget, timeout):
    analysis_path = PERSONA_DIR / spec["analysis"]
    images, image_warnings = [], []
    for rel in spec.get("images", []):
        if runs_root and (runs_root / rel).exists():
            images.append(runs_root / rel)
        else:
            image_warnings.append(f"image missing, skipped: {rel}")
    result = engine.run_research(
        analysis_path, images, OUT_DIR / name / "research.json",
        budget_usd=budget, timeout=timeout, model=model,
    )
    row = {"name": name, "result": result, "failures": [], "warnings": image_warnings}
    if not result["ok"]:
        row["failures"].append(f"run failed: {result['error']}")
        return row
    fixture = json.loads(analysis_path.read_text())
    research = result["research"]
    failures, warnings = check_global(research, fixture)
    failures += CHECKS[name](research, fixture)
    row["failures"] += failures
    row["warnings"] += warnings
    row["unknown_rate"] = unknown_rate(research)
    return row


def top_products(research, limit=3):
    named = []
    for entry in entries(research):
        for product in entry.get("products") or []:
            named.append(f"{product.get('brand')} {product.get('name')} "
                         f"(r{product.get('rank')}, {product.get('review_sentiment')})")
    return named[:limit]


def write_report(rows, results_by_name):
    lines = ["# agentrec persona matrix report", ""]
    lines.append("| persona | ok | cost | duration | unknown-rate | failures | warnings |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        result = row["result"]
        cost = result.get("cost")
        lines.append(
            f"| {row['name']} | {'PASS' if result['ok'] and not row['failures'] else 'FAIL'} "
            f"| {f'${cost:.2f}' if isinstance(cost, (int, float)) else '?'} "
            f"| {result.get('duration', '?')}s | {row.get('unknown_rate', '-')} "
            f"| {len(row['failures'])} | {len(row['warnings'])} |")
    for row in rows:
        lines += ["", f"## {row['name']}", ""]
        research = (row["result"].get("research") or {})
        if research:
            lines.append(f"- see_doctor_first: {research.get('see_doctor_first')}")
            lines.append(f"- top products: {'; '.join(top_products(research)) or 'none'}")
        for failure in row["failures"]:
            lines.append(f"- FAIL: {failure}")
        for warning in row["warnings"]:
            lines.append(f"- warn: {warning}")
    oily = (results_by_name.get("medium-oily") or {}).get("research")
    dry = (results_by_name.get("medium-dry") or {}).get("research")
    if oily and dry:
        lines += ["", "## oily vs dry supporting products", ""]
        for label, research in (("oily", oily), ("dry", dry)):
            supports = ", ".join(
                f"{p.get('role')}: {p.get('brand')} {p.get('name')}"
                for p in research.get("supporting_products", []))
            lines.append(f"- {label}: {supports}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = OUT_DIR / "matrix_report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None, help="persona names to run")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--model", default=None)
    parser.add_argument("--budget-usd", type=float, default=15.0)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args(argv)

    index = json.loads((PERSONA_DIR / "personas.json").read_text())
    names = args.only or sorted(index)
    unknown_names = set(names) - set(index)
    if unknown_names:
        raise SystemExit(f"unknown personas: {sorted(unknown_names)}")
    runs_root = engine.find_runs_root()

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_one, name, index[name], runs_root, args.model,
                        args.budget_usd, args.timeout): name
            for name in names
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            status = "PASS" if row["result"]["ok"] and not row["failures"] else "FAIL"
            print(f"{row['name']}: {status} ({len(row['failures'])} failures)")
    rows.sort(key=lambda r: names.index(r["name"]))
    report = write_report(rows, {r["name"]: r["result"] for r in rows})
    failed = [r["name"] for r in rows if not r["result"]["ok"] or r["failures"]]
    print(f"matrix: {len(rows) - len(failed)}/{len(rows)} pass -> {report}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
