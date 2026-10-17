"""Small synthetic fixtures. Ordinary tests never touch the network or the research data directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filingshock.config import Config
from filingshock.storage import DataLayout
from filingshock.timing import SessionCalendar

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def cal() -> SessionCalendar:
    return SessionCalendar("XNYS")


def make_bars(sessions: pd.DatetimeIndex, seed: int = 0, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(sessions)
    oc = rng.normal(0, 0.01, n)
    gap = rng.normal(0, 0.005, n)
    opens, closes = np.empty(n), np.empty(n)
    prev = base
    for i in range(n):
        opens[i] = prev * (1 + gap[i])
        closes[i] = opens[i] * (1 + oc[i])
        prev = closes[i]
    hi = np.maximum(opens, closes) * 1.005
    lo = np.minimum(opens, closes) * 0.995
    return pd.DataFrame({"session": sessions, "open": opens, "high": hi, "low": lo, "close": closes,
                         "volume": rng.integers(1_000_000, 2_000_000, n).astype(float), "dividends": 0.0, "splits": 0.0})


@pytest.fixture()
def mini_layout(tmp_path, cal) -> tuple[Config, DataLayout]:
    """A tiny but complete pre-extraction data directory: filings table + price store."""
    cfg = Config.load().with_overrides({"universe.companies": [{"cik": 1, "ticker": "TEST", "name": "Test Co"}],
                                        "universe.filing_window.accepted_from": "2024-01-01",
                                        "universe.filing_window.accepted_to": "2024-12-31"})
    layout = DataLayout(tmp_path).ensure()
    sessions = cal.sessions[(cal.sessions >= "2023-01-03") & (cal.sessions <= "2024-12-31")]
    from filingshock.storage import write_json_atomic, write_parquet_atomic
    store_dir = layout.raw_market / "yfinance"
    man = {}
    for sym, seed in [("TEST", 1), ("SPY", 2), ("IGV", 3)]:
        bars = make_bars(sessions, seed)
        write_parquet_atomic(store_dir / f"{sym}.parquet", bars)
        man[sym] = {"status": "ok", "rows": len(bars), "last": str(sessions[-1].date()), "provider": {}}
    write_json_atomic(store_dir / "manifest.json", man)
    acc = lambda s: pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")  # noqa: E731
    rows = [
        # after-close 10-Q -> eligible, target 2024-05-02
        ("0000000001-24-000001", "10-Q", "2024-05-01", "2024-03-31", acc("2024-05-01 16:10")),
        # second 10-Q for the same target session -> duplicate
        ("0000000001-24-000002", "10-Q", "2024-05-01", "2024-03-31", acc("2024-05-01 16:40")),
        # amendment -> unsupported form
        ("0000000001-24-000003", "10-Q/A", "2024-05-03", "2024-03-31", acc("2024-05-03 16:20")),
        # accepted during the session -> excluded
        ("0000000001-24-000004", "10-Q", "2024-08-01", "2024-06-30", acc("2024-08-01 11:00")),
        # pre-market, available before cutoff -> eligible, same-day target
        ("0000000001-24-000005", "10-Q", "2024-10-30", "2024-09-30", acc("2024-10-30 07:45")),
        # earnings 8-K in the same window as the first 10-Q
        ("0000000001-24-000006", "8-K", "2024-05-01", "2024-05-01", acc("2024-05-01 16:05")),
    ]
    f = pd.DataFrame(rows, columns=["accession_number", "form_type", "filing_date", "report_date", "acceptance_utc"])
    f["cik"] = 1
    f["ticker"] = "TEST"
    f["company_name"] = "Test Co"
    f["items"] = np.where(f["form_type"] == "8-K", "2.02,9.01", "")
    f["is_earnings_8k"] = f["form_type"] == "8-K"
    f["in_window"] = True
    f["primary_document"] = "doc.htm"
    write_parquet_atomic(layout.interim / "filings.parquet", f)
    return cfg, layout


@pytest.fixture(scope="session")
def mdna_html() -> bytes:
    return (FIXTURES / "sample_10q.html").read_bytes()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
