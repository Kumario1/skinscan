"""recommendations.json -> single-file HTML report for human review.

Usage: python -m recsys.tools.render_html <recommendations.json> [out.html]
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

# ponytail: string-built HTML, no templating dep — this is a dev review tool

CSS = """
body{font-family:-apple-system,Segoe UI,sans-serif;max-width:960px;margin:2rem auto;
     padding:0 1rem;color:#1a1a2e;background:#fafafa;line-height:1.45}
h1{font-size:1.5rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:2px solid #ddd;padding-bottom:.3rem}
h3{font-size:1.05rem;margin-bottom:.3rem}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:9px;font-size:.75rem;font-weight:600;margin-right:.3rem}
.b-warn{background:#fff3cd;color:#7a5c00}.b-ok{background:#d4edda;color:#155724}
.b-info{background:#d6e4ff;color:#1c3d8f}.b-rx{background:#f8d7da;color:#721c24}
.b-cat{background:#e2e3e5;color:#41464b}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:1rem;margin:.7rem 0;
      box-shadow:0 1px 2px rgba(0,0,0,.04)}
.step{border-left:4px solid #6c8ebf;padding:.5rem .8rem;margin:.5rem 0;background:#fff;
      border-radius:0 6px 6px 0;border-top:1px solid #eee;border-right:1px solid #eee;border-bottom:1px solid #eee}
.slot{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#666;font-weight:700}
.price{float:right;color:#333;font-weight:600}
.why{font-size:.85rem;color:#444;margin:.3rem 0 0}
.signals{font-size:.78rem;color:#666;margin:.3rem 0 0;padding-left:1rem}
.signals li{margin:.1rem 0}
.notes{font-size:.78rem;color:#996c00;margin:.2rem 0 0}
.framing{background:#fff3cd;border:1px solid #ffe69c;border-radius:8px;padding:.8rem 1rem;font-size:.9rem}
.triage{background:#f8d7da;border:1px solid #f1aeb5;border-radius:8px;padding:.8rem 1rem;font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.85rem}
td,th{border:1px solid #ddd;padding:.35rem .6rem;text-align:left}
th{background:#f0f0f0}
.meta{font-size:.75rem;color:#888;margin-top:2rem}
details summary{cursor:pointer;font-weight:600;font-size:.9rem;margin:.5rem 0}
"""


def esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def render_step(s: dict) -> str:
    ver = s.get("verification", "")
    ver_badge = {"verified": "b-ok", "category_derived": "b-cat"}.get(ver, "b-info")
    rx = '<span class="badge b-rx">Rx</span>' if s.get("prescription") else ""
    price = f'<span class="price">${s["price_usd"]:.2f}</span>' if s.get("price_usd") is not None else ""
    why = s.get("why") or {}
    sig_items = "".join(
        f'<li><b>{esc(g["name"])}</b> ({g["value"]:.2f}): {esc(g.get("evidence",""))}</li>'
        for g in why.get("signals", [])
    )
    notes = "".join(f'<div class="notes">⚠ {esc(n)}</div>' for n in s.get("notes", []))
    d = s.get("directions") or {}
    dir_bits = " · ".join(esc(d[k]) for k in ("amount", "cadence") if d.get(k))
    directions = f'<div class="why"><b>Directions:</b> {dir_bits}</div>' if dir_bits else ""
    actives = ", ".join(str(item).replace("_", " ") for item in s.get("actives", []))
    active_html = (
        f'<div class="why"><b>Active ingredients:</b> {esc(actives)}</div>'
        if actives else '<div class="why"><b>Active ingredients:</b> none identified</div>'
    )
    ingredients = ", ".join(str(item) for item in s.get("ingredients", []))
    ingredient_html = (
        f'<details><summary>Full ingredient list</summary><div class="why">'
        f'{esc(ingredients)}</div></details>'
        if ingredients else ""
    )
    return f"""<div class="step">
      <span class="slot">{esc(s.get('slot'))}</span> {price}
      <h3>{esc(s.get('brand'))} — {esc(s.get('name'))}
        {rx}<span class="badge {ver_badge}">{esc(ver)}</span>
        <span class="badge b-info">{esc(s.get('usage'))}</span></h3>
      <div class="why">{esc(why.get('summary',''))}</div>
      {directions}
      {active_html}
      {ingredient_html}
      <details><summary>signals (score {why.get('score',0):.3f})</summary>
        <ul class="signals">{sig_items}</ul></details>
      {notes}
    </div>"""


def render_routine(r: dict) -> str:
    am = "".join(render_step(s) for s in r.get("am", []))
    pm = "".join(render_step(s) for s in r.get("pm", []))
    checks = " ".join(
        f'<span class="badge {"b-ok" if c["passed"] else "b-rx"}">{esc(c["rule"])} {"✓" if c["passed"] else "✗"}</span>'
        for c in r.get("safety_checks", [])
    )
    notes = "".join(f'<div class="notes">⚠ {esc(n)}</div>' for n in r.get("notes", []))
    total = f'${r["total_price_usd"]:.2f}' if r.get("total_price_usd") is not None else "—"
    return f"""<div class="card">
      <h2 style="margin-top:0;border:none">{esc(r.get('title'))}
        <span class="price">{total} · {r.get('slot_count')} products</span></h2>
      <div class="why">{esc(r.get('rationale',''))}</div>
      <h3>☀️ AM</h3>{am or '<p>—</p>'}
      <h3>🌙 PM</h3>{pm or '<p>—</p>'}
      <div style="margin-top:.6rem">{checks}</div>
      {notes}
    </div>"""


def render(doc: dict) -> str:
    concerns = "".join(
        f"<tr><td>{esc(c['lesion_type'])}</td><td>{c['count']}</td>"
        f"<td>{'yes' if c['selected_for_treatment'] else 'no'}</td>"
        f"<td>{'yes' if c.get('referral_emphasis') else 'no'}</td></tr>"
        for c in doc.get("target_lesions", [])
    )
    triage = doc.get("triage") or {}
    triage_html = ""
    if triage:
        reasons = ", ".join(esc(x) for x in triage.get("referral_reasons", []))
        triage_html = f"""<div class="triage"><b>Triage: {esc(triage.get('level'))}</b><br>
          {esc(triage.get('see_doctor_note',''))}<br>
          <small>Reasons: {reasons or '—'}</small></div>"""
    routines = "".join(render_routine(r) for r in doc.get("routines", []))
    if not routines:
        routines = (
            '<div class="triage"><b>No recommendation available.</b><br>'
            f'{esc(doc.get("reason", "The selector did not return a valid regimen."))}</div>'
        )
    rx_opts = "".join(
        f"""<div class="step"><h3>{esc(p.get('name'))} <span class="badge b-rx">Rx</span>
        <span class="badge b-cat">{esc(p.get('format'))}</span></h3>
        <div class="why">{esc(p.get('why',''))}</div>
        <div class="notes">{esc(p.get('note',''))}</div></div>"""
        for p in doc.get("prescription_options", [])
    )
    prof = doc.get("profile_used") or {}
    prof_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>"
        for k, v in prof.items() if v not in (None, [], "unknown")
    ) or "<tr><td colspan=2>all unknown / defaults</td></tr>"
    warnings = "".join(f'<div class="notes">⚠ {esc(w)}</div>' for w in doc.get("warnings", []))
    eng = doc.get("engine") or {}
    selection = doc.get("selection") or {}
    selection_html = ""
    if selection:
        cache = (
            "cache hit" if selection.get("cache_hit") is True
            else "cache miss" if selection.get("cache_hit") is False
            else "cache unavailable"
        )
        usage = selection.get("usage") or {}
        usage_bits = []
        if "input_tokens" in usage:
            usage_bits.append(f'{esc(usage["input_tokens"])} input tokens')
        if "output_tokens" in usage:
            usage_bits.append(f'{esc(usage["output_tokens"])} output tokens')
        if usage.get("estimated_cost_usd") is not None:
            usage_bits.append(f'${usage["estimated_cost_usd"]:.6f}')
        usage_text = " · " + " · ".join(usage_bits) if usage_bits else ""
        latency_text = (
            f' · {esc(selection["latency_ms"])} ms'
            if selection.get("latency_ms") is not None else ""
        )
        deployment = selection.get("deployment") or selection.get("model", "")
        model_identity = selection.get("model_identity") or selection.get("model", "")
        response_model = selection.get("model") or model_identity
        selection_html = (
            '<div class="card"><b>Selection:</b> '
            f'{esc(selection.get("source", "unknown"))} · '
            f'{esc(selection.get("provider", ""))}/{esc(deployment)}'
            f' ({esc(model_identity)}; response {esc(response_model)}) · '
            f'{esc(selection.get("prompt_version", ""))} · {cache} · '
            f'{esc(selection.get("candidate_count", "—"))} candidates'
            f'{latency_text}{usage_text}</div>'
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkinScan recommendations</title><style>{CSS}</style></head><body>
<h1>SkinScan recommendation report</h1>
<div class="framing">{esc((doc.get('framing') or {}).get('text',''))}</div>
{triage_html}
{warnings}
{selection_html}
<h2>Detected concerns</h2>
<table><tr><th>concern</th><th>severity</th><th>treated</th><th>referral emphasis</th></tr>{concerns}</table>
<h2>Profile used</h2>
<table>{prof_rows}</table>
<h2>Routines</h2>
{routines}
<h2>Prescription options (doctor-guided)</h2>
{rx_opts or '<p>none</p>'}
<div class="meta">status={esc(doc.get('status'))} · engine {esc(eng.get('version'))}
 @ {esc(eng.get('git_commit'))} · mode {esc(eng.get('eligibility_mode'))}
 · generated {esc(doc.get('generated_at'))}<br>
 image sha256 {esc((doc.get('inputs') or {}).get('source_image_sha256'))}</div>
</body></html>"""


def main() -> int:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")
    out.write_text(render(json.loads(src.read_text())), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
