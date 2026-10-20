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
