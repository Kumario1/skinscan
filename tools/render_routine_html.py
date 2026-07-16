#!/usr/bin/env python3
"""Render a skinscan routine.json into a readable HTML page."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def esc(value: Any) -> str:
    if value is None:
        return "—"
    return html.escape(str(value))


def chips(items: list[Any] | None, *, empty: str = "—") -> str:
    if not items:
        return f'<span class="muted">{empty}</span>'
    return "".join(f'<span class="chip">{esc(x)}</span>' for x in items)


def money(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"${float(value):.0f}"
    except (TypeError, ValueError):
        return esc(value)


def product_card(role: str, product: dict[str, Any] | None, *, ranking: str | None = None) -> str:
    if not product:
        return f"""
        <article class="card empty">
          <div class="eyebrow">{esc(role)}</div>
          <h3>None selected</h3>
        </article>
        """
    spf = product.get("spf")
    spf_bit = f" · SPF {esc(spf)}" if spf else ""
    ranking_bit = (
        f'<div class="meta">Ranking: <code>{esc(ranking)}</code></div>' if ranking else ""
    )
    return f"""
    <article class="card">
      <div class="eyebrow">{esc(role)}</div>
      <h3>{esc(product.get("brand"))}</h3>
      <p class="title">{esc(product.get("name"))}</p>
      <div class="meta">
        <code>{esc(product.get("product_id"))}</code>
        · {money(product.get("price_usd"))}{spf_bit}
        · tier {esc(product.get("tier"))}
      </div>
      {ranking_bit}
      <div class="row"><span class="label">Actives</span>{chips(product.get("actives"))}</div>
      <div class="row"><span class="label">Cadence</span><span>{esc(product.get("cadence"))}</span></div>
      <div class="row"><span class="label">Evidence</span><span>{esc(product.get("evidence_grade"))}</span></div>
      <div class="row"><span class="label">Format</span><span>{esc(product.get("format"))} · {esc(product.get("exposure"))}</span></div>
    </article>
    """


def regimen_list(steps: list[dict[str, Any]], products: dict[str, Any]) -> str:
    if not steps:
        return '<p class="muted">No steps</p>'
    rows = []
    for i, step in enumerate(steps, 1):
        role = step.get("role") or ""
        product = products.get(role) or {}
        name = product.get("name") or role
        brand = product.get("brand") or ""
        rows.append(
            f"""
            <li>
              <span class="step-n">{i}</span>
              <div>
                <strong>{esc(brand)}</strong> {esc(name)}
                <div class="meta">{esc(role)} · {esc(step.get("cadence"))}</div>
              </div>
            </li>
            """
        )
    return "<ol class='regimen'>" + "".join(rows) + "</ol>"


def concern_rows(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "<tr><td colspan='4' class='muted'>No decision evidence</td></tr>"
    rows = []
    for item in evidence:
        reasons = chips(item.get("reasons"), empty="none")
        rows.append(
            f"""
            <tr>
              <td><strong>{esc(item.get("concern"))}</strong></td>
              <td>{esc(item.get("quality"))}</td>
              <td>{esc(item.get("source"))}</td>
              <td>{reasons}</td>
            </tr>
            """
        )
    return "".join(rows)


def alt_rows(alternatives: dict[str, list[dict[str, Any]]]) -> str:
    blocks = []
    for role, products in (alternatives or {}).items():
        if not products:
            continue
        cards = "".join(product_card(role, p) for p in products)
        blocks.append(f"<h3 class='subhead'>{esc(role)} alternatives</h3><div class='grid'>{cards}</div>")
    return "".join(blocks) or '<p class="muted">No alternatives</p>'


def kv_table(data: dict[str, Any], keys: list[str] | None = None) -> str:
    items = keys or list(data.keys())
    rows = []
    for key in items:
        value = data.get(key)
        if isinstance(value, list):
            shown = chips(value) if value else '<span class="muted">[]</span>'
        elif isinstance(value, dict):
            shown = f"<code>{esc(json.dumps(value, sort_keys=True))}</code>"
        else:
            shown = esc(value)
        rows.append(f"<tr><th>{esc(key)}</th><td>{shown}</td></tr>")
    return "<table class='kv'>" + "".join(rows) + "</table>"


def render(data: dict[str, Any], source_path: str) -> str:
    decision = data.get("decision") or {}
    therapy = data.get("therapy_plan") or {}
    release = data.get("release_eligibility") or {}
    products = data.get("selected_products") or {}
    regimen = data.get("selected_regimen") or {}
    profile = data.get("input_profile") or {}
    ranking = {
        e.get("role"): e.get("ranking_basis")
        for e in (data.get("explanation") or [])
        if isinstance(e, dict)
    }
    eligible = bool(release.get("eligible"))
    status = data.get("validation_status", "unknown")
    triage = decision.get("triage_level") or "—"
    disposition = decision.get("therapy_disposition") or "—"

    product_cards = "".join(
        product_card(role, products.get(role), ranking=ranking.get(role))
        for role in sorted(products.keys()) or ["cleanser", "moisturizer", "sunscreen"]
    )
    if not products:
        product_cards = '<p class="muted">No selected products</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Routine · {esc(Path(source_path).name)}</title>
  <style>
    :root {{
      --bg: #f7f5f1;
      --ink: #1c1a17;
      --muted: #6b645c;
      --line: #ddd6cb;
      --card: #fffdf9;
      --accent: #0f5c4c;
      --warn: #8a4b12;
      --bad: #8b1e1e;
      --ok: #1f5b2f;
      --chip: #ece6dc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 15px/1.45 "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 24px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #fffefb, var(--bg));
    }}
    header h1 {{
      margin: 0 0 6px;
      font: 600 28px/1.15 "Iowan Old Style", "Palatino Linotype", Palatino, serif;
    }}
    header .path {{ color: var(--muted); font-size: 13px; word-break: break-all; }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 20px 24px 64px;
      display: grid;
      gap: 22px;
    }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px 18px 14px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 13px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .subhead {{
      margin: 18px 0 10px;
      font-size: 15px;
      text-transform: none;
      letter-spacing: 0;
      color: var(--ink);
    }}
    .banner {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }}
    .stat {{
      background: #f3efe7;
      border-radius: 10px;
      padding: 12px;
    }}
    .stat .label {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .stat strong {{ font-size: 16px; }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .pill.ok {{ background: #d9eedf; color: var(--ok); }}
    .pill.bad {{ background: #f3d7d7; color: var(--bad); }}
    .pill.warn {{ background: #f5e2cc; color: var(--warn); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
      background: #fff;
    }}
    .card.empty {{ opacity: 0.7; }}
    .eyebrow {{
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 4px;
    }}
    .card h3 {{ margin: 0 0 4px; font-size: 18px; }}
    .card .title {{ margin: 0 0 8px; color: var(--ink); }}
    .meta {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .row {{
      display: flex;
      gap: 10px;
      align-items: flex-start;
      margin-top: 6px;
      font-size: 13px;
    }}
    .label {{
      min-width: 72px;
      color: var(--muted);
    }}
    .chip {{
      display: inline-block;
      background: var(--chip);
      border-radius: 999px;
      padding: 2px 8px;
      margin: 0 4px 4px 0;
      font-size: 12px;
    }}
    .muted {{ color: var(--muted); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    table.kv th {{
      width: 38%;
      color: var(--muted);
      font-weight: 500;
    }}
    .two {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 720px) {{
      .two {{ grid-template-columns: 1fr; }}
    }}
    ol.regimen {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 10px;
    }}
    ol.regimen li {{
      display: flex;
      gap: 12px;
      align-items: flex-start;
    }}
    .step-n {{
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: var(--accent);
      color: white;
      display: grid;
      place-items: center;
      font-size: 12px;
      font-weight: 700;
      flex: 0 0 auto;
    }}
    details {{
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 8px;
    }}
    summary {{
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 12px;
    }}
    ul.flags {{
      margin: 0;
      padding-left: 18px;
    }}
    ul.flags li {{ margin-bottom: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Routine reader</h1>
    <div class="path">{esc(source_path)}</div>
  </header>
  <main>
    <section>
      <h2>At a glance</h2>
      <div class="banner">
        <div class="stat">
          <span class="label">Validation</span>
          <strong><span class="pill {"ok" if status == "valid" else "warn"}">{esc(status)}</span></strong>
        </div>
        <div class="stat">
          <span class="label">Triage</span>
          <strong>{esc(triage)}</strong>
        </div>
        <div class="stat">
          <span class="label">Therapy</span>
          <strong>{esc(disposition)}</strong>
        </div>
        <div class="stat">
          <span class="label">Release</span>
          <strong><span class="pill {"ok" if eligible else "bad"}">{"eligible" if eligible else "blocked"}</span></strong>
        </div>
        <div class="stat">
          <span class="label">Generated</span>
          <strong>{esc(data.get("generated_at"))}</strong>
        </div>
        <div class="stat">
          <span class="label">Schema</span>
          <strong>{esc(data.get("schema_version"))}</strong>
        </div>
      </div>
    </section>

    <section>
      <h2>Decision</h2>
      <div class="row"><span class="label">Reasons</span>{chips(decision.get("referral_reasons"))}</div>
      <div class="row"><span class="label">Policy</span><span><code>{esc(decision.get("policy_version"))}</code> · reviewed={esc(decision.get("policy_reviewed"))}</span></div>
      <table style="margin-top:12px">
        <thead>
          <tr><th>Concern</th><th>Quality</th><th>Source</th><th>Evidence</th></tr>
        </thead>
        <tbody>
          {concern_rows(decision.get("decision_evidence") or [])}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Therapy plan</h2>
      <div class="row"><span class="label">Support</span>{chips(therapy.get("support_roles"))}</div>
      <div class="row"><span class="label">Primary</span><span>{esc(therapy.get("primary"))}</span></div>
      <div class="row"><span class="label">Deferred</span>{chips(therapy.get("deferred_reasons"))}</div>
      <div class="row"><span class="label">Course</span><span>{esc(therapy.get("course_weeks"))} weeks · review @ {esc(therapy.get("review_at_weeks"))}</span></div>
      <div class="row"><span class="label">Policy</span><span><code>{esc(therapy.get("policy_version"))}</code></span></div>
    </section>

    <section>
      <h2>AM / PM regimen</h2>
      <div class="two">
        <div>
          <h3 class="subhead">Morning</h3>
          {regimen_list(regimen.get("am") or [], products)}
        </div>
        <div>
          <h3 class="subhead">Evening</h3>
          {regimen_list(regimen.get("pm") or [], products)}
        </div>
      </div>
    </section>

    <section>
      <h2>Selected products</h2>
      <div class="grid">{product_cards}</div>
    </section>

    <section>
      <h2>Alternatives</h2>
      {alt_rows(data.get("alternatives") or {})}
    </section>

    <section>
      <h2>Flags & release</h2>
      <div class="row"><span class="label">Flags</span></div>
      <ul class="flags">{"".join(f"<li><code>{esc(f)}</code></li>" for f in (data.get("flags") or [])) or "<li class='muted'>none</li>"}</ul>
      <div class="row" style="margin-top:12px"><span class="label">Blocked by</span>{chips(release.get("reasons"))}</div>
    </section>

    <section>
      <h2>Input profile</h2>
      {kv_table(profile)}
    </section>

    <section>
      <h2>Provenance</h2>
      {kv_table({
          "sample_id": (data.get("dataset") or {}).get("sample_id"),
          "dataset": (data.get("dataset") or {}).get("name"),
          "split": (data.get("dataset") or {}).get("split"),
          "git_commit": (data.get("code") or {}).get("git_commit"),
          "dirty": (data.get("code") or {}).get("dirty"),
          "catalog_state": (data.get("catalog") or {}).get("state"),
          "catalog_sha256": (data.get("catalog") or {}).get("sha256"),
          "config_sha256": data.get("config_sha256"),
          "replay_key": data.get("replay_key"),
          "source_image_sha256": data.get("source_image_sha256"),
          "ranker": (data.get("ranker") or {}).get("state"),
          "triage_policy": ((data.get("policies") or {}).get("triage") or {}).get("identity"),
          "therapy_policy": ((data.get("policies") or {}).get("therapy") or {}).get("identity"),
      })}
      <details>
        <summary>Raw JSON (collapsed)</summary>
        <pre style="overflow:auto;max-height:420px;font-size:12px;background:#f3efe7;padding:12px;border-radius:8px">{esc(json.dumps(data, indent=2, ensure_ascii=False))}</pre>
      </details>
    </section>
  </main>
</body>
</html>
"""


RECSYS_STYLE = """
  :root { --bg:#f7f5f1; --ink:#1c1a17; --muted:#6b645c; --line:#ddd6cb; --card:#fffdf9;
          --accent:#0f5c4c; --warn:#8a4b12; --chip:#ece6dc; --rx:#7a2e8a; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#171614; --ink:#ece7df; --muted:#9d958a; --line:#332f2a; --card:#201e1b;
            --accent:#5fd3b4; --warn:#e0a25f; --chip:#2b2823; --rx:#d99bec; }
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:2rem 1.25rem; background:var(--bg); color:var(--ink);
         font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }
  .wrap { max-width:1100px; margin:0 auto; }
  h1 { font-size:1.5rem; margin:0 0 .2rem; }
  .sub { color:var(--muted); font-size:.85rem; margin-bottom:1.5rem; word-break:break-all; }
  .bar { display:flex; flex-wrap:wrap; gap:.5rem; margin-bottom:1.5rem; }
  .stat { background:var(--card); border:1px solid var(--line); border-radius:8px;
          padding:.5rem .8rem; }
  .stat b { display:block; font-size:.7rem; text-transform:uppercase; letter-spacing:.05em;
            color:var(--muted); font-weight:600; }
  .chip { display:inline-block; background:var(--chip); border-radius:99px;
          padding:.1rem .55rem; font-size:.75rem; margin:0 .2rem .2rem 0; }
  .routine { background:var(--card); border:1px solid var(--line); border-radius:10px;
             padding:1rem 1.1rem; margin-bottom:1.1rem; }
  .routine h2 { font-size:1.05rem; margin:0 0 .15rem; }
  .rationale { color:var(--muted); font-size:.82rem; margin:0 0 .8rem; }
  .sess { font-size:.7rem; text-transform:uppercase; letter-spacing:.06em;
          color:var(--muted); font-weight:700; margin:.7rem 0 .3rem; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  td { padding:.4rem .5rem; border-top:1px solid var(--line); vertical-align:top; }
  td.slot { color:var(--muted); white-space:nowrap; width:5.5rem; }
  td.price { text-align:right; white-space:nowrap; color:var(--muted); }
  .why { color:var(--muted); font-size:.78rem; }
  .tag { font-size:.68rem; border-radius:4px; padding:.05rem .35rem; margin-left:.35rem;
         white-space:nowrap; }
  .ok { background:var(--accent); color:var(--bg); }
  .der { border:1px solid var(--line); color:var(--muted); }
  .rx { background:var(--rx); color:var(--bg); }
  .scroll { overflow-x:auto; }
  .un { color:var(--warn); font-size:.82rem; }
"""


def _recsys_steps(title: str, steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    rows = []
    for step in steps:
        badge = ('<span class="tag ok">verified</span>'
                 if step.get("verification") == "verified"
                 else '<span class="tag der">category-derived</span>')
        if step.get("prescription"):
            badge += '<span class="tag rx">Rx · see a doctor</span>'
        why = ((step.get("why") or {}).get("summary")) or ""
        rows.append(
            f'<tr><td class="slot">{esc(step.get("slot"))}</td>'
            f'<td><strong>{esc(step.get("name"))}</strong>{badge}'
            f'<div class="why">{esc(step.get("brand"))} · {esc(why)}</div></td>'
            f'<td class="price">{money(step.get("price_usd"))}</td></tr>'
        )
    return (f'<div class="sess">{esc(title)}</div>'
            f'<div class="scroll"><table>{"".join(rows)}</table></div>')


def render_recsys(data: dict[str, Any], source_path: str) -> str:
    """Render a recsys recommendations.json (archetype routines, not one regimen)."""
    triage = data.get("triage") or {}
    engine = data.get("engine") or {}
    concerns = "".join(
        f'<span class="chip">{esc(c["concern"])} · sev {esc(c["severity"])}</span>'
        for c in data.get("target_concerns") or []
    )
    blocks = []
    for routine in data.get("routines") or []:
        blocks.append(
            f'<section class="routine"><h2>{esc(routine.get("title"))} · '
            f'{money(routine.get("total_price_usd"))}</h2>'
            f'<p class="rationale">{esc(routine.get("rationale"))}</p>'
            + _recsys_steps("Morning", routine.get("am") or [])
            + _recsys_steps("Evening", routine.get("pm") or [])
            + _recsys_steps("Per label", routine.get("per_label") or [])
            + "</section>"
        )
    for item in data.get("unavailable_archetypes") or []:
        blocks.append(
            f'<section class="routine"><h2>{esc(item.get("archetype"))}</h2>'
            f'<p class="un">unavailable — {chips(item.get("reasons"))}</p></section>'
        )
    options = data.get("prescription_options") or []
    if options:
        rows = "".join(
            f'<tr><td><strong>{esc(o.get("name"))}</strong>'
            f'<span class="tag rx">Rx</span>'
            f'<div class="why">{esc(o.get("format"))} · '
            + esc(", ".join(f'{a["name"].replace("_", " ")} {a["strength"]}'
                            for a in o.get("actives") or []))
            + f'</div></td><td>{chips(o.get("targets"))}</td></tr>'
            for o in options
        )
        blocks.append(
            '<section class="routine"><h2>Prescription options to ask a doctor about</h2>'
            '<p class="rationale">These are prescription-strength and cannot be bought '
            'over the counter. They are listed, not ranked into the routines above: a '
            'doctor or dermatologist decides whether one is right for you and can '
            'prescribe it.</p>'
            f'<div class="scroll"><table>{rows}</table></div></section>'
        )
    stats = {
        "status": data.get("status"), "eligibility": engine.get("eligibility_mode"),
        "triage": triage.get("level"), "routines": len(data.get("routines") or []),
        "catalog": Path((data.get("data_versions") or {}).get(
            "catalog", {}).get("path", "—")).name,
    }
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Recommendations · {esc(Path(source_path).name)}</title>
<style>{RECSYS_STYLE}</style></head><body><div class="wrap">
<h1>Routine recommendations</h1>
<p class="sub">{esc(source_path)}</p>
<div class="bar">{''.join(
    f'<div class="stat"><b>{esc(k)}</b>{esc(v)}</div>' for k, v in stats.items())}</div>
<div class="bar"><div class="stat" style="flex:1"><b>target concerns</b>{concerns}</div></div>
{''.join(blocks)}
<section class="routine"><h2>Notes</h2>
<p class="rationale">{esc((data.get("framing") or {}).get("text"))}</p>
<p class="rationale">{esc(triage.get("see_doctor_note"))}</p></section>
</div></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "routine_json",
        type=Path,
        nargs="?",
        default=Path("runs/e2e/real_test_0/routine.json"),
        help="Path to routine.json (default: runs/e2e/real_test_0/routine.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: same dir as input, routine.html)",
    )
    args = parser.parse_args()
    src = args.routine_json
    if not src.exists():
        raise SystemExit(f"File not found: {src}")
    out = args.output or src.with_name("routine.html")
    data = json.loads(src.read_text())
    # recsys emits archetype routines; the legacy pipeline emits one regimen.
    renderer = render_recsys if "routines" in data else render
    out.write_text(renderer(data, str(src)), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
