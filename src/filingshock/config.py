"""Configuration loading and hashing.

All research-relevant settings live in config/default.yaml. The config hash is recorded in
every artifact manifest so stale artifacts can be detected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
PLACEHOLDER_MARKER = "UNCONFIGURED"


@dataclass(frozen=True)
class Company:
    cik: int
    ticker: str
    name: str
    pilot: bool = False
    delisted: bool = False
    symbol_history: str | None = None

    @property
    def cik10(self) -> str:
        return f"{self.cik:010d}"


class Config:
    def __init__(self, raw: dict[str, Any], path: Path | None = None):
        self.raw = raw
        self.path = path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else Path(os.environ.get("FILINGSHOCK_CONFIG", DEFAULT_CONFIG))
        with open(p) as fh:
            raw = yaml.safe_load(fh)
        return cls(raw, p)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def data_dir(self) -> Path:
        d = Path(os.environ.get("FILINGSHOCK_DATA_DIR", self.raw["project"]["data_dir"]))
        return d if d.is_absolute() else PROJECT_ROOT / d

    @property
    def sec_user_agent(self) -> str:
        return os.environ.get("FILINGSHOCK_SEC_USER_AGENT", self.raw["sec"]["user_agent"])

    @property
    def user_agent_is_placeholder(self) -> bool:
        return PLACEHOLDER_MARKER in self.sec_user_agent

    def companies(self, pilot_only: bool = False) -> list[Company]:
        out = []
        for c in self.raw["universe"]["companies"]:
            comp = Company(
                cik=int(c["cik"]),
                ticker=str(c["ticker"]),
                name=str(c["name"]),
                pilot=bool(c.get("pilot", False)),
                delisted=bool(c.get("delisted", False)),
                symbol_history=c.get("symbol_history"),
            )
            if pilot_only and not comp.pilot:
                continue
            out.append(comp)
        return out

    def company_by_cik(self, cik: int) -> Company | None:
        for c in self.companies():
            if c.cik == int(cik):
                return c
        return None

    def section_hash(self, *sections: str) -> str:
        """Stable hash of selected config sections (used for cache keys and manifests)."""
        payload = {s: self.raw.get(s) for s in sections} if sections else self.raw
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        raw = copy.deepcopy(self.raw)
        for dotted, value in overrides.items():
            node = raw
            keys = dotted.split(".")
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value
        return Config(raw, self.path)
