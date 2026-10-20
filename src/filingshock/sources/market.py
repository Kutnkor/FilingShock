"""Market-data provider interface, implementations and the local daily-bar store.

Business logic depends only on the normalised daily-bar schema:
    session (datetime64, tz-naive date), open, high, low, close, volume, dividends, splits
Providers differ in coverage, adjustment conventions and redistribution rights; each one
describes itself so the conventions are recorded with the data.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from filingshock.config import Config
from filingshock.storage import DataLayout, file_lock, sha256_file, write_parquet_atomic

log = logging.getLogger(__name__)

BAR_COLUMNS = ["session", "open", "high", "low", "close", "volume", "dividends", "splits"]
BAR_SETTLE_MINUTES = 60      # a session's bar is treated as final only this long after its close


class MarketDataUnavailable(RuntimeError):
    pass


def _market_component(value: str) -> str:
    if (not isinstance(value, str) or not value or value in (".", "..")
            or any(c in value for c in ("/", "\\", ":")) or any(ord(c) < 32 for c in value)):
        raise MarketDataUnavailable("invalid market-data path component")
    return value


class MarketDataProvider(ABC):
    name: str

    @abstractmethod
    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return normalised daily bars for [start, end)."""

    @abstractmethod
    def describe(self) -> dict:
        """Coverage, adjustment, timestamp and redistribution conventions."""


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance via the yfinance package (unofficial, free).

    Conventions (verified empirically and documented in docs/data_card.md):
    - open/high/low/close are split-adjusted, not dividend-adjusted (auto_adjust=False), so
      open and close within one session share the same adjustment basis.
    - Split adjustments are applied retroactively by the vendor; the stored vintage is
      identified by its retrieval timestamp and hash.
    - Delisted securities are generally unavailable (survivorship limitation).
    - Yahoo's terms do not permit redistribution; raw bars are never committed.
    """

    name = "yfinance"

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "price_adjustment": "split-adjusted OHLC; not dividend-adjusted",
            "session_timestamps": "exchange-local trading date",
            "delisted_coverage": "generally unavailable",
            "redistribution": "not permitted (Yahoo terms); local research cache only",
            "quota": "unofficial endpoint; rate limited without published quota",
        }

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        try:
            hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False, actions=True,
                                             raise_errors=True)
        except Exception as exc:  # yfinance raises a variety of exception types
            raise MarketDataUnavailable(f"{symbol}: {type(exc).__name__}: {exc}") from exc
        if hist is None or hist.empty:
            raise MarketDataUnavailable(f"{symbol}: provider returned no rows")
        df = hist.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        out = pd.DataFrame({
            "session": pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize(),
            "open": df["Open"].astype(float),
            "high": df["High"].astype(float),
            "low": df["Low"].astype(float),
            "close": df["Close"].astype(float),
            "volume": df["Volume"].astype(float),
            "dividends": df.get("Dividends", pd.Series(0.0, index=df.index)).astype(float),
            "splits": df.get("Stock Splits", pd.Series(0.0, index=df.index)).astype(float),
        })
        return out.sort_values("session").reset_index(drop=True)


class CsvImportProvider(MarketDataProvider):
    """Import user-supplied daily bars: <dir>/<SYMBOL>.csv with columns
    date,open,high,low,close,volume[,dividends,splits]. Use this for licensed data
    (e.g. CRSP) or for delisted securities the free provider cannot serve."""

    name = "csv"

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def describe(self) -> dict:
        return {"provider": self.name, "directory": str(self.directory),
                "price_adjustment": "as supplied by the user; must be consistent within a session",
                "redistribution": "governed by the user's data licence"}

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self.directory / f"{_market_component(symbol)}.csv"
        if not path.exists():
            raise MarketDataUnavailable(f"{symbol}: no import file at {path}")
        try:
            raw = pd.read_csv(path)
        except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise MarketDataUnavailable(f"{symbol}: cannot read import file: {exc}") from exc
        raw.columns = [c.strip().lower() for c in raw.columns]
        missing = {"date", "open", "close"} - set(raw.columns)
        if missing:
            raise MarketDataUnavailable(f"{symbol}: import file lacks columns {sorted(missing)}")
        try:
            sessions = pd.to_datetime(raw["date"].astype("string"), format="mixed", errors="raise")
            if sessions.isna().any() or sessions.dt.tz is not None:
                raise ValueError("use nonempty exchange-local dates without a timezone")
            sessions = sessions.dt.normalize()
        except (ValueError, TypeError, AttributeError) as exc:
            raise MarketDataUnavailable(f"{symbol}: invalid session dates: {exc}") from exc
        out = pd.DataFrame({"session": sessions})
        for c in ["open", "high", "low", "close", "volume", "dividends", "splits"]:
            out[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else (
                0.0 if c in ("dividends", "splits") else np.nan)
        out = out[(out["session"] >= start) & (out["session"] < end)]
        duplicate = out["session"].duplicated(keep=False)
        if duplicate.any():
            dates = out.loc[duplicate, "session"].dt.strftime("%Y-%m-%d").unique()
            raise MarketDataUnavailable(f"{symbol}: duplicate session dates: {', '.join(dates[:5])}")
        return out.sort_values("session").reset_index(drop=True)


def make_provider(cfg: Config) -> MarketDataProvider:
    md = cfg["market_data"]
    if md["provider"] == "yfinance":
        return YFinanceProvider()
    if md["provider"] == "csv":
        p = Path(md["csv_dir"])
        return CsvImportProvider(p if p.is_absolute() else cfg.data_dir.parent / p)
    raise ValueError(f"unknown market data provider {md['provider']!r}")


class PriceStore:
    """Local cache of normalised daily bars, one Parquet file per provider/symbol."""

    def __init__(self, layout: DataLayout, provider_name: str):
        self.dir = layout.raw_market / _market_component(provider_name)
        self.manifest = self.dir / "manifest.json"

    def path(self, symbol: str) -> Path:
        return self.dir / f"{_market_component(symbol)}.parquet"

    def has(self, symbol: str) -> bool:
        return self.path(symbol).exists()

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self.path(symbol)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df["session"] = pd.to_datetime(df["session"])
        return df

    def _read_manifest(self) -> dict:
        if self.manifest.exists():
            return json.loads(self.manifest.read_text())
        return {}

    def manifest_entries(self) -> dict:
        return self._read_manifest()

    def save(self, symbol: str, df: pd.DataFrame, provider: MarketDataProvider, status: str,
             error: str | None = None, extra: dict | None = None) -> None:
        self.path(symbol)  # Validate even unavailable symbols, which have no Parquet file.
        self.dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self.manifest.with_suffix(".lock")):
            entry = {"symbol": symbol, "status": status, "error": error,
                     "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "provider": provider.describe(), **(extra or {})}
            if status == "ok":
                write_parquet_atomic(self.path(symbol), df[BAR_COLUMNS])
                entry.update({"rows": int(len(df)), "first": str(df["session"].min().date()),
                              "last": str(df["session"].max().date()), "sha256": sha256_file(self.path(symbol))})
            man = self._read_manifest()
            man[symbol] = entry
            from filingshock.storage import write_json_atomic
            write_json_atomic(self.manifest, man)


def resolve_history_end(value, now: datetime | None = None) -> str:
    """Exclusive end date for a price request. A fixed date (the research default in config) is used as
    given; "today" means tomorrow's UTC date, i.e. everything up to now, for live use. Either way,
    ingest_prices drops bars of sessions that had not settled when the request started."""
    if str(value).strip().lower() == "today":
        now = pd.Timestamp(now or datetime.now(timezone.utc))
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        return str((now.normalize() + pd.Timedelta(days=1)).date())
    return str(pd.Timestamp(value).date())


def drop_unsettled_sessions(df: pd.DataFrame, cal, as_of) -> tuple[pd.DataFrame, int]:
    """Remove bars whose session had not closed BAR_SETTLE_MINUTES before as_of: during a session a
    provider returns a partial bar whose close is the latest trade, not the closing price.

    cal: timing.SessionCalendar. Sessions outside the calendar count as unsettled from their own date on.
    """
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    settle = pd.Timedelta(minutes=BAR_SETTLE_MINUTES)
    recent = df["session"] >= as_of.tz_localize(None).normalize() - pd.Timedelta(days=3)

    def unsettled(session) -> bool:
        i = cal.index_of(session)
        if i is None:
            return pd.Timestamp(session).normalize() >= as_of.tz_localize(None).normalize()
        return cal.close_utc(i) + settle > as_of

    drop = pd.Series(False, index=df.index)
    drop[recent] = df.loc[recent, "session"].map(unsettled).astype(bool)
    return df[~drop].reset_index(drop=True), int(drop.sum())


def ingest_prices(cfg: Config, layout: DataLayout, symbols: list[str], refresh: bool = False,
                  history_end: str | None = None, now: datetime | None = None) -> dict:
    """history_end overrides market_data.history_end for this run only (a date or "today");
    now: request time (tests inject a fixed clock)."""
    from filingshock.timing import SessionCalendar

    provider = make_provider(cfg)
    store = PriceStore(layout, provider.name)
    md = cfg["market_data"]
    end = resolve_history_end(history_end or md["history_end"], now)
    cal = SessionCalendar(cfg["timing"]["exchange_calendar"])
    results = {}
    for sym in symbols:
        if store.has(sym) and not refresh:
            results[sym] = "cached"
            continue
        try:
            as_of = now or datetime.now(timezone.utc)          # taken before the request: conservative
            df, dropped = drop_unsettled_sessions(provider.fetch_daily(sym, md["history_start"], end), cal, as_of)
            if df.empty:
                raise MarketDataUnavailable(f"{sym}: no settled sessions before {end}")
            store.save(sym, df, provider, "ok", extra={"requested_end": end, "unsettled_sessions_dropped": dropped})
            results[sym] = f"ok ({len(df)} rows, last {df['session'].max().date()})"
        except MarketDataUnavailable as exc:
            entry = store.manifest_entries().get(sym, {})
            if store.has(sym) and entry.get("status") == "ok":
                results[sym] = f"refresh_failed (cached history preserved): {exc}"
                log.warning("price refresh failed for %s; retaining original file and retrieval metadata: %s", sym, exc)
                continue
            store.save(sym, pd.DataFrame(columns=BAR_COLUMNS), provider, "unavailable", str(exc))
            results[sym] = f"unavailable: {exc}"
            log.warning("price data unavailable for %s: %s", sym, exc)
    return results


def bar_quality(df: pd.DataFrame, max_abs_oc: float) -> pd.DataFrame:
    """Per-session validity flags. A bar is valid for return computation only if open and close
    are positive and finite, and consistent with the reported high/low range."""
    out = df.copy()
    o, c, h, lo = out["open"], out["close"], out["high"], out["low"]
    out["oc_ret"] = c / o - 1.0
    tol = 1e-6
    ok = np.isfinite(o) & np.isfinite(c) & (o > 0) & (c > 0)
    range_ok = ~(np.isfinite(h) & np.isfinite(lo)) | (
        (o <= h * (1 + tol)) & (o >= lo * (1 - tol)) & (c <= h * (1 + tol)) & (c >= lo * (1 - tol)))
    out["flag_range_inconsistent"] = ok & ~range_ok
    out["flag_extreme_return"] = ok & (out["oc_ret"].abs() > max_abs_oc)
    out["flag_zero_volume"] = out["volume"].fillna(0) <= 0
    out["valid"] = ok & range_ok & ~out["flag_extreme_return"]
    out.loc[~out["valid"], "oc_ret"] = np.nan
    return out
