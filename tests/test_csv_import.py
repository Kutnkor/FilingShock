"""CSV ingestion failures stay local to the affected symbol, without network access."""

import pandas as pd
import pytest

from filingshock.config import Config
from filingshock.sources.market import CsvImportProvider, MarketDataUnavailable, ingest_prices
from filingshock.storage import DataLayout


@pytest.mark.parametrize("content", [b"", b'date,open,close\n"unfinished', b"date,open,close\n\xff,1,1"])
def test_unreadable_csv_is_a_provider_error(tmp_path, content):
    (tmp_path / "BAD.csv").write_bytes(content)
    with pytest.raises(MarketDataUnavailable, match="BAD: cannot read import file"):
        CsvImportProvider(tmp_path).fetch_daily("BAD", "2024-01-01", "2025-01-01")


def test_broken_csv_does_not_abort_other_symbols(tmp_path):
    (tmp_path / "BAD.csv").write_bytes(b"")
    (tmp_path / "GOOD.csv").write_text("date,open,close\n2024-01-02,100,101\n")
    cfg = Config.load().with_overrides({"market_data.provider": "csv", "market_data.csv_dir": str(tmp_path)})
    results = ingest_prices(cfg, DataLayout(tmp_path / "cache"), ["BAD", "GOOD"],
                            now=pd.Timestamp("2024-01-04", tz="UTC"))
    assert results["BAD"].startswith("unavailable:")
    assert results["GOOD"].startswith("ok (")


@pytest.mark.parametrize("date", ["not-a-date", "", "2024-02-30", "2024-01-02T00:00:00-05:00"])
def test_invalid_or_ambiguous_session_dates_are_reported(tmp_path, date):
    (tmp_path / "BAD.csv").write_text(f"date,open,close\n{date},100,101\n")
    with pytest.raises(MarketDataUnavailable, match="BAD: invalid session dates"):
        CsvImportProvider(tmp_path).fetch_daily("BAD", "2024-01-01", "2025-01-01")


def test_naive_dates_are_normalized_sorted_and_end_exclusive(tmp_path):
    (tmp_path / "GOOD.csv").write_text(
        "date,open,close\n2024-01-03,100,101\n2024-01-02 16:00:00,90,91\n2024-01-04,110,111\n")
    bars = CsvImportProvider(tmp_path).fetch_daily("GOOD", "2024-01-02", "2024-01-04")
    assert bars["session"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert bars["open"].tolist() == [90, 100]


@pytest.mark.parametrize("second_date", ["2024-01-02", "2024-01-02 16:00:00"])
def test_duplicate_sessions_are_rejected_instead_of_counted_twice(tmp_path, second_date):
    (tmp_path / "BAD.csv").write_text(f"date,open,close\n2024-01-02,100,101\n{second_date},100,102\n")
    with pytest.raises(MarketDataUnavailable, match="duplicate session dates.*2024-01-02"):
        CsvImportProvider(tmp_path).fetch_daily("BAD", "2024-01-01", "2025-01-01")


def test_duplicate_sessions_outside_requested_window_do_not_block_import(tmp_path):
    (tmp_path / "GOOD.csv").write_text(
        "date,open,close\n2023-12-29,100,101\n2023-12-29,100,102\n2024-01-02,100,103\n")
    bars = CsvImportProvider(tmp_path).fetch_daily("GOOD", "2024-01-01", "2025-01-01")
    assert len(bars) == 1
