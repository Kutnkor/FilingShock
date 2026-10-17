"""Configuration loading and hashing.

All research-relevant settings live in config/default.yaml. The config hash is recorded in
every artifact manifest so stale artifacts can be detected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
PLACEHOLDER_MARKER = "UNCONFIGURED"


class ConfigurationError(ValueError):
    """Invalid configuration, diagnosed before network access or dataset writes."""


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
        try:
            with open(p, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"cannot read YAML configuration: {p.name}") from exc
        cfg = cls(raw, p)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not isinstance(self.raw, dict):
            raise ConfigurationError("configuration must be a YAML mapping")
        sections = ("project", "sec", "universe", "market_data", "timing", "target", "reference", "features",
                    "model", "splits")
        bad = [name for name in sections if not isinstance(self.raw.get(name), dict)]
        if bad:
            raise ConfigurationError("missing or invalid configuration sections: " + ", ".join(bad))
        errors = []
        for section, key in (("project", "name"), ("project", "data_dir"), ("sec", "user_agent"),
                             ("timing", "exchange_calendar"), ("features", "version"),
                             ("features", "topic_lexicon"), ("market_data", "csv_dir"),
                             ("reference", "primary_policy")):
            if not isinstance(self.raw[section].get(key), str) or not self.raw[section][key].strip():
                errors.append(f"{section}.{key} must be a nonempty string")

        def number(section, key, minimum=0, integer=False, inclusive=False):
            value = self.raw[section].get(key)
            valid = (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                     and (not integer or isinstance(value, int))
                     and (value >= minimum if inclusive else value > minimum))
            if not valid:
                errors.append(f"{section}.{key} must be a finite {'integer' if integer else 'number'} "
                              f"{'at least' if inclusive else 'greater than'} {minimum}")
            return valid

        for section, key in (("sec", "max_requests_per_second"), ("sec", "timeout_seconds"),
                             ("target", "multiplier"), ("target", "min_denominator"),
                             ("target", "max_abs_return_sanity")):
            number(section, key)
        for section, key in (("sec", "backoff_base_seconds"), ("timing", "availability_lag_minutes")):
            number(section, key, inclusive=True)
        number("sec", "max_retries", integer=True, inclusive=True)
        for section, key in (("target", "lookback_sessions"), ("target", "max_window_sessions"),
                             ("timing", "cutoff_minutes_before_open"), ("splits", "min_train_events"),
                             ("splits", "min_train_positives"), ("splits", "min_calibration_positives"),
                             ("model", "bootstrap_reps"), ("splits", "first_test_year"), ("splits", "last_test_year")):
            number(section, key, integer=True)
        if not errors:
            if self.raw["target"]["max_window_sessions"] < self.raw["target"]["lookback_sessions"]:
                errors.append("target.max_window_sessions must cover target.lookback_sessions")
            if self.raw["splits"]["first_test_year"] > self.raw["splits"]["last_test_year"]:
                errors.append("splits.first_test_year must not exceed splits.last_test_year")
        grid = self.raw["model"].get("c_grid")
        if not isinstance(grid, list) or not grid or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in grid):
            errors.append("model.c_grid must contain finite positive numbers")
        if self.raw["market_data"].get("provider") not in ("csv", "yfinance"):
            errors.append("market_data.provider must be csv or yfinance")
        if not isinstance(self.raw["project"].get("data_dir"), str) or not self.raw["project"]["data_dir"].strip():
            errors.append("project.data_dir must be a nonempty path")
        companies = self.raw["universe"].get("companies")
        if not isinstance(companies, list) or not companies:
            errors.append("universe.companies must be a nonempty list")
        else:
            ciks, tickers = set(), set()
            for i, company in enumerate(companies):
                if (not isinstance(company, dict) or type(company.get("cik")) is not int
                        or not 0 < company["cik"] < 10**10 or not isinstance(company.get("ticker"), str)
                        or not company["ticker"].strip() or not isinstance(company.get("name"), str)
                        or not company["name"].strip()):
                    errors.append(f"universe.companies[{i}] needs a positive CIK, ticker and name")
                    continue
                if company["cik"] in ciks or company["ticker"] in tickers:
                    errors.append(f"universe.companies[{i}] duplicates a CIK or ticker")
                ciks.add(company["cik"])
                tickers.add(company["ticker"])
        window = self.raw["universe"].get("filing_window", {})
        try:
            if date.fromisoformat(str(window["accepted_from"])) > date.fromisoformat(str(window["accepted_to"])):
                raise ValueError("reversed window")
        except (KeyError, TypeError, ValueError):
            errors.append("universe.filing_window needs ordered ISO accepted_from/accepted_to dates")
        if errors:
            raise ConfigurationError("; ".join(errors))

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
        return not self.sec_user_agent.strip() or PLACEHOLDER_MARKER in self.sec_user_agent

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
        cfg = Config(raw, self.path)
        cfg.validate()
        return cfg
