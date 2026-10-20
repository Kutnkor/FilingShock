"""Market cache paths and concurrent publications must preserve provenance."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pandas as pd
import pytest

from filingshock.sources.market import CsvImportProvider, MarketDataUnavailable, PriceStore
from filingshock.storage import DataLayout, sha256_file


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


def test_concurrent_price_saves_keep_both_manifest_entries(tmp_path, monkeypatch):
    layout = DataLayout(tmp_path)
    first, second = PriceStore(layout, "csv"), PriceStore(layout, "csv")
    entered, release, second_started = Event(), Event(), Event()
    read_manifest = first._read_manifest

    def paused_read():
        snapshot = read_manifest()
        entered.set()
        assert release.wait(5)
        return snapshot

    monkeypatch.setattr(first, "_read_manifest", paused_read)
    bars = pd.DataFrame({"session": [pd.Timestamp("2024-01-02")], "open": [100.0], "high": [102.0],
                         "low": [99.0], "close": [101.0], "volume": [1000.0], "dividends": [0.0], "splits": [0.0]})
    provider = CsvImportProvider(tmp_path)

    def save_second():
        second_started.set()
        second.save("SECOND", bars, provider, "ok")

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first.save, "FIRST", bars, provider, "ok")
        try:
            assert entered.wait(5)
            b = pool.submit(save_second)
            assert second_started.wait(5)
            with pytest.raises(TimeoutError):
                b.result(timeout=0.2)  # The second writer must wait for publication of the first.
        finally:
            release.set()
        a.result(timeout=5)
        b.result(timeout=5)
    entries = second.manifest_entries()
    assert set(entries) == {"FIRST", "SECOND"}
    for symbol in entries:
        assert entries[symbol]["sha256"] == sha256_file(second.path(symbol))
        assert entries[symbol]["rows"] == len(second.load(symbol))
