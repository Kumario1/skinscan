"""Shared helpers for build tools: deterministic JSON artifacts and the signal
registry. No timestamps in artifacts — byte-identical rebuilds are a test."""
from __future__ import annotations

import json
from pathlib import Path

from ..contracts import sha256_file
from ..signals import REGISTRY_SCHEMA_VERSION

DEFAULT_RAW_DIR = Path("data/raw/sephora")
STORE_SCHEMA_VERSION = "recsys-store-1"


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def update_registry(data_root: str | Path, entry: dict) -> Path:
    """Insert or replace this store's registry entry (keyed by name)."""
    data_root = Path(data_root)
    registry_path = data_root / "signals" / "registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"schema_version": REGISTRY_SCHEMA_VERSION, "stores": []}
    stores = [e for e in registry.get("stores", []) if e.get("name") != entry["name"]]
    stores.append(entry)
    registry["stores"] = sorted(stores, key=lambda e: e["name"])
    return write_json(registry_path, registry)


def register_store(
    data_root: str | Path, *, name: str, kind: str, version: str,
    store_path: Path, builder: str, source: dict, coverage: dict,
) -> None:
    data_root = Path(data_root)
    update_registry(data_root, {
        "name": name,
        "kind": kind,
        "version": version,
        "path": str(store_path.relative_to(data_root)),
        "sha256": sha256_file(store_path),
        "builder": builder,
        "source": source,
        "coverage": coverage,
        "status": "active",
    })
