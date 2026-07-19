"""Headless-agent research engine: feed a pipeline analysis.json to `claude -p`.

Usage:
    python -m agentrec.engine --check
    python -m agentrec.engine                                   # medium-oily fixture, no images
    python -m agentrec.engine --analysis <analysis.json> --images <sheet.jpg> <face.jpg>

Requires an authenticated `claude` CLI on PATH (bills the Max plan). Output goes to
agentrec/out/<name>/research.json unless --out is given; unusable results land in a
sibling research.raw.txt.

The `claude -p` CLI on the Max subscription is the permanent engine by owner
decision (2026-07-19) — no Agent SDK port; the SDK requires API-key per-token
billing, which this project deliberately avoids.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from agentrec.prompts import PROMPT_TEMPLATE, SYSTEM_PROMPT

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
OUT_DIR = PKG / "out"
DEFAULT_ANALYSIS = PKG / "personas" / "medium-oily.analysis.json"
DEFAULT_IMAGE_NAMES = ("lesion_sheet.jpg", "detections.jpg")

ALLOWED_KEYS = (
    "lesion_findings",
    "concerns",
    "care_pathways",
    "decision",
    "therapy_plan",
    "safety_observations",
    "input_profile",
    "skin_tone",
    "clear_skin",
)

IMAGE_DESCRIPTIONS = {
    "lesion_sheet.jpg": "a contact sheet of cropped detected lesions",
    "detections.jpg": "the full face with detection boxes",
}


def find_runs_root(probe="runs/e2e/real-recommendations-v4-20260716/low-real/analysis.json"):
    """Locate a checkout containing the gitignored runs/e2e artifacts.

    Worktrees carry only tracked files; fall back to the main checkout (via the shared
    git dir) and require `probe` to actually exist there.
    """
    candidates = [ROOT]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        candidates.append(Path(common).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        pass
    for candidate in candidates:
        if (candidate / probe).exists():
            return candidate
    return None


def project_context(analysis):
    return {k: analysis[k] for k in ALLOWED_KEYS if k in analysis}


def extract_json(text):
    try:
        start, end = text.index("{"), text.rindex("}")
    except ValueError as exc:
        raise ValueError("no JSON object found in claude result") from exc
    return json.loads(text[start : end + 1])


def build_prompt(image_paths):
    if image_paths:
        section = "\n".join(
            f"{i}. Use the Read tool on {path} - "
            f"{IMAGE_DESCRIPTIONS.get(Path(path).name, 'a pipeline image artifact')}."
            for i, path in enumerate(image_paths, 1)
        )
    else:
        section = (
            "No images are available for this run; rely on the JSON alone and say so in "
            "image_observations."
        )
    # ponytail: .replace, never .format — the template's JSON schema is full of braces
    return PROMPT_TEMPLATE.replace("{image_section}", section)


def run_claude(prompt, context_json, budget_usd, timeout, model, cwd):
    argv = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--allowedTools",
        "WebSearch,WebFetch,Read",
        "--max-budget-usd",
        str(budget_usd),
        "--append-system-prompt",
        SYSTEM_PROMPT,
    ]
    if model:
        argv += ["--model", model]
    start = time.monotonic()
    proc = subprocess.run(
        argv, input=context_json, capture_output=True, text=True, timeout=timeout, cwd=str(cwd)
    )
    return proc, time.monotonic() - start


def run_research(analysis_path, images, out_path, *, budget_usd=15.0, timeout=1200, model=None):
    """One research run. Returns a dict: {ok, out, cost, duration, error, research}."""
    analysis_path = Path(analysis_path)
    analysis = json.loads(analysis_path.read_text())
    context_json = json.dumps(project_context(analysis), separators=(",", ":"), sort_keys=True)
    images = [Path(p) for p in images]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix(".raw.txt")
    cwd = images[0].parent if images else analysis_path.parent

    try:
        proc, duration = run_claude(
            build_prompt(images), context_json, budget_usd, timeout, model, cwd
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        raw_path.write_text(stdout or "")
        return {"ok": False, "error": f"timeout after {timeout}s", "out": str(raw_path)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": f"failed to run claude: {exc}", "out": None}

    envelope = {}
    result_text = proc.stdout or ""
    try:
        envelope = json.loads(proc.stdout)
        result_text = envelope.get("result") or ""
    except (TypeError, ValueError):
        pass
    if proc.returncode != 0 or envelope.get("is_error"):
        raw_path.write_text(result_text or proc.stdout or "")
        tail = (proc.stderr or "").strip()[-2000:]
        return {
            "ok": False,
            "error": f"claude exited {proc.returncode} "
                     f"({envelope.get('subtype') or 'unknown'}) {tail}".strip(),
            "out": str(raw_path),
            "cost": envelope.get("total_cost_usd"),
            "duration": round(duration, 1),
        }
    try:
        research = extract_json(result_text)
    except ValueError:
        raw_path.write_text(result_text)
        return {
            "ok": False,
            "error": "could not parse JSON from result",
            "out": str(raw_path),
            "cost": envelope.get("total_cost_usd"),
            "duration": round(duration, 1),
        }

    payload = {
        "claude": {
            "session_id": envelope.get("session_id"),
            "total_cost_usd": envelope.get("total_cost_usd"),
            "duration_s": round(duration, 1),
            "model": model,
        },
        "research": research,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {
        "ok": True,
        "out": str(out_path),
        "cost": envelope.get("total_cost_usd"),
        "duration": round(duration, 1),
        "research": research,
    }


def check():
    assert "{image_section}" in PROMPT_TEMPLATE
    assert PROMPT_TEMPLATE.count("{image_section}") == 1
    for marker in ("derm_first", "review_sentiment", "STEP 4"):
        assert marker in PROMPT_TEMPLATE, marker
    for marker in ("hydroquinone", "isotretinoin", "one JSON object"):
        assert marker in SYSTEM_PROMPT, marker
    assert extract_json('noise {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    try:
        extract_json("no json here")
    except ValueError:
        pass
    else:
        raise AssertionError("extract_json accepted brace-free text")
    with_images = build_prompt([Path("/tmp/lesion_sheet.jpg")])
    assert "Read" in with_images and "/tmp/lesion_sheet.jpg" in with_images
    assert '"review_sentiment"' in with_images  # schema braces survived .replace
    assert "No images" in build_prompt([])
    print("check OK")
    return 0


def _default_images(analysis_path):
    return [
        p for name in DEFAULT_IMAGE_NAMES if (p := analysis_path.parent / name).exists()
    ]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument(
        "--images",
        type=Path,
        nargs="*",
        default=None,
        help="images for the agent to view; default: sibling lesion_sheet/detections jpgs; bare --images disables",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="default: agentrec/out/<analysis stem>/research.json")
    parser.add_argument("--budget-usd", type=float, default=15.0)
    parser.add_argument("--timeout", type=int, default=1200, help="seconds; the real runaway control")
    parser.add_argument("--model", default=None, help="passthrough to claude --model")
    parser.add_argument("--check", action="store_true", help="offline self-test; never invokes claude")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.check:
        return check()
    analysis_path = args.analysis.resolve()
    images = _default_images(analysis_path) if args.images is None else [p.resolve() for p in args.images]
    name = analysis_path.stem.replace(".analysis", "")
    out_path = args.out or OUT_DIR / name / "research.json"
    result = run_research(
        analysis_path, images, out_path,
        budget_usd=args.budget_usd, timeout=args.timeout, model=args.model,
    )
    if not result["ok"]:
        print(f"agentrec: {result['error']} -> {result.get('out')}", file=sys.stderr)
        return 1
    research = result["research"]
    per_concern = research.get("per_concern", [])
    actives = sorted(
        {
            active.get("name")
            for entry in per_concern
            if isinstance(entry, dict)
            for active in entry.get("actives", [])
            if isinstance(active, dict) and active.get("name")
        }
    )
    cost = result.get("cost")
    cost_str = f"${cost:.2f}" if isinstance(cost, (int, float)) else "$?"
    print(
        f"agentrec: {cost_str} {result['duration']:.0f}s {len(per_concern)} concern entries, "
        f"see_doctor_first={research.get('see_doctor_first')}, "
        f"actives: {', '.join(actives) or 'none'} -> {result['out']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
