"""Minimal `.env` loader. Secrets live in `.env` at the repo root and are never committed."""

import os
from coach.config import ROOT


def _dotenv() -> dict[str, str]:
    # read each time, not cached: a Riot dev key changes daily and a running app should
    # pick up the new one
    path = ROOT / ".env"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("\"'")
    return out


def get(key: str) -> str | None:
    """The environment wins over `.env`."""
    return os.environ.get(key) or _dotenv().get(key)
