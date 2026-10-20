"""Market cache paths and concurrent publications must preserve provenance."""

import pandas as pd
import pytest

from filingshock.sources.market import CsvImportProvider, MarketDataUnavailable, PriceStore
from filingshock.storage import DataLayout


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "..\\escape", "C:escape", "", ".", "..", "bad\x00name"])
def test_market_names_must_be_single_path_components(tmp_path, name):
    layout = DataLayout(tmp_path)
    with pytest.raises(MarketDataUnavailable, match="path component"):
        PriceStore(layout, name)
    store = PriceStore(layout, "csv")
    with pytest.raises(MarketDataUnavailable, match="path component"):
        store.path(name)
    with pytest.raises(MarketDataUnavailable, match="path component"):
        CsvImportProvider(tmp_path).fetch_daily(name, "2024-01-01", "2025-01-01")
    with pytest.raises(MarketDataUnavailable, match="path component"):
        store.save(name, pd.DataFrame(), CsvImportProvider(tmp_path), "unavailable")
    assert not store.manifest.exists()


@pytest.mark.parametrize("symbol", ["BRK.B", "BRK-B", "^GSPC", "EURUSD=X"])
def test_market_symbol_punctuation_is_preserved(tmp_path, symbol):
    store = PriceStore(DataLayout(tmp_path), "yfinance")
    assert store.path(symbol) == tmp_path / "raw" / "market" / "yfinance" / f"{symbol}.parquet"
