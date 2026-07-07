"""Shared 5-line helpers used across pipeline scripts."""


def require(path, label):
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")
