"""Loads the hand-authored knowledge files (data/knowledge/*.json) into one
validated object. These files are the single source of truth for concern→active
mapping, safety vocabularies, and archetype definitions — code never hardcodes
an active name."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import CONCERNS, SLOTS, ContractViolation, sha256_file


@dataclass(frozen=True)
class Knowledge:
    concern_actives: dict[str, frozenset[str]]
    phrasing: dict[str, str]
    referral_emphasis: frozenset[str]
    retinoids: frozenset[str]
    treatment_actives: frozenset[str]
    active_conflicts: frozenset[frozenset[str]]
    pregnancy_excluded_statuses: frozenset[str]
    pm_pinned_actives: frozenset[str]
    am_preferred_actives: frozenset[str]
    pm_preferred_actives: frozenset[str]
    gentle_excluded_actives: frozenset[str]
    gentle_allowlist: frozenset[str]
    min_spf: int
    archetypes: tuple[dict, ...]
    default_weights: dict[str, float]
    file_sha256s: dict[str, str]


def load_knowledge(knowledge_dir: str | Path) -> Knowledge:
    knowledge_dir = Path(knowledge_dir)
    paths = {name: knowledge_dir / f"{name}.json"
             for name in ("concern_actives", "safety_rules", "archetypes")}
    raw = {}
    for name, path in paths.items():
        if not path.exists():
            raise ContractViolation(f"knowledge.{name}", f"missing {path}")
        raw[name] = json.loads(path.read_text(encoding="utf-8"))

    ca = raw["concern_actives"]
    for concern in ca.get("actives", {}):
        if concern not in CONCERNS:
            raise ContractViolation("knowledge.concern_actives", f"unknown concern {concern!r}")

    sr = raw["safety_rules"]
    prefs = sr.get("session_preferences", {})

    arch = raw["archetypes"]
    archetypes = tuple(arch.get("archetypes") or [])
    if len(archetypes) != 5:
        raise ContractViolation("knowledge.archetypes", f"expected 5 archetypes, got {len(archetypes)}")
    for a in archetypes:
        for slot in a.get("slots", []):
            if slot not in SLOTS:
                raise ContractViolation("knowledge.archetypes", f"{a.get('id')}: unknown slot {slot!r}")

    return Knowledge(
        concern_actives={k: frozenset(v) for k, v in ca.get("actives", {}).items()},
        phrasing=dict(ca.get("phrasing") or {}),
        referral_emphasis=frozenset(ca.get("referral_emphasis") or []),
        retinoids=frozenset(sr.get("retinoids") or []),
        treatment_actives=frozenset(sr.get("treatment_actives") or []),
        active_conflicts=frozenset(frozenset(pair) for pair in sr.get("active_conflicts") or []),
        pregnancy_excluded_statuses=frozenset(sr.get("pregnancy_excluded_statuses") or []),
        pm_pinned_actives=frozenset(prefs.get("pm_pinned_actives") or []),
        am_preferred_actives=frozenset(prefs.get("am_preferred_actives") or []),
        pm_preferred_actives=frozenset(prefs.get("pm_preferred_actives") or []),
        gentle_excluded_actives=frozenset((sr.get("gentle") or {}).get("excluded_actives") or []),
        gentle_allowlist=frozenset((sr.get("gentle") or {}).get("allowlist_treatment_actives") or []),
        min_spf=int(sr.get("min_spf", 30)),
        archetypes=archetypes,
        default_weights=dict(arch.get("default_weights") or {}),
        file_sha256s={name: sha256_file(path) for name, path in paths.items()},
    )
