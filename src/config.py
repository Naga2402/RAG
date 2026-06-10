"""Central config loader. Reads config/settings.yaml and expands ${ENV} refs."""
from __future__ import annotations
import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_ENV_RE = re.compile(r"\$\{([^}]+)\}")

# Load .env (gitignored) so secrets like PG_PASSWORD are available to ${VAR}
# expansion below — the same .env that docker-compose reads.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _expand(obj):
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return _ENV_RE.sub(lambda m: os.getenv(m.group(1), ""), obj)
    return obj


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _expand(raw)


CFG = load_config()
