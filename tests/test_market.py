"""Offline price-ingestion regressions; providers are replaced by deterministic failures."""


from filingshock.sources import market
from filingshock.storage import sha256_file


class UnavailableProvider(market.MarketDataProvider):
    name = "yfinance"

    def describe(self):
        return {"provider": self.name}

    def fetch_daily(self, symbol, start, end):
        raise market.MarketDataUnavailable("temporary provider outage")


def test_failed_refresh_preserves_prices_and_actual_retrieval_time(mini_layout, monkeypatch):
    cfg, layout = mini_layout
    store = market.PriceStore(layout, "yfinance")
    before = (sha256_file(store.path("TEST")), store.manifest.read_bytes())
    monkeypatch.setattr(market, "make_provider", lambda cfg: UnavailableProvider())
    result = market.ingest_prices(cfg, layout, ["TEST"], refresh=True)
    assert result["TEST"].startswith("refresh_failed")
    assert before == (sha256_file(store.path("TEST")), store.manifest.read_bytes())
    assert store.load("TEST").shape[0] > 0


def test_initial_unavailable_symbol_still_has_an_explicit_status(mini_layout, monkeypatch):
    cfg, layout = mini_layout
    monkeypatch.setattr(market, "make_provider", lambda cfg: UnavailableProvider())
    result = market.ingest_prices(cfg, layout, ["DELISTED"], refresh=True)
    store = market.PriceStore(layout, "yfinance")
    assert result["DELISTED"].startswith("unavailable")
    assert store.manifest_entries()["DELISTED"]["status"] == "unavailable"
    assert not store.path("DELISTED").exists()


def test_prices_command_reports_refresh_failure(monkeypatch, capsys, tmp_path):
    from filingshock.cli import EXIT_FAIL, main

    monkeypatch.setenv("FILINGSHOCK_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("filingshock.pipeline.step_prices", lambda *a, **k: {"TEST": "refresh_failed: provider outage"})
    assert main(["prices", "--refresh"]) == EXIT_FAIL
    assert "refresh_failed" in capsys.readouterr().out
