"""Market and event-context features from sessions completed before the decision cutoff.

Every function here receives price history that has ALREADY been truncated to sessions
strictly before the target session, so target-session prices cannot enter by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_FEATURES = ["mkt_absoc_mean_20", "mkt_absoc_mean_60", "mkt_ret_vol_60", "mkt_large_move_freq_250",
                   "mkt_log_rel_volume_5_60", "mkt_prev_session_abs_ret", "mkt_spy_absoc_20",
                   "mkt_igv_absoc_20", "mkt_prev_event_abs_ratio"]
CONTEXT_FEATURES = ["ctx_earnings_8k_in_window", "ctx_log_days_since_earnings_8k", "ctx_log_filing_lag_days"]


def _absoc(hist: pd.DataFrame) -> pd.Series:
    return hist.loc[hist["valid"], "oc_ret"].abs()


def stock_features(hist: pd.DataFrame, lookback: int = 60, multiplier: float = 2.0) -> dict:
    """hist: bar_quality() rows strictly before the target session."""
    out: dict = {}
    a = _absoc(hist)
    out["mkt_absoc_mean_20"] = float(a.tail(20).mean()) if len(a) >= 20 else None
    out["mkt_absoc_mean_60"] = float(a.tail(60).mean()) if len(a) >= 60 else None
    closes = hist["close"].where(hist["close"] > 0)
    lr = np.log(closes).diff().dropna()
    out["mkt_ret_vol_60"] = float(lr.tail(60).std()) if len(lr) >= 60 else None
    # frequency of earlier large moves: each past session judged against ITS OWN prior-60 baseline
    if len(a) >= lookback + 120:
        base = a.rolling(lookback, min_periods=lookback).mean().shift(1)
        flag = (a > multiplier * base).where(base.notna())
        last = flag.tail(250).dropna()
        out["mkt_large_move_freq_250"] = float(last.mean()) if len(last) >= 120 else None
    else:
        out["mkt_large_move_freq_250"] = None
    vol = hist["volume"].where(hist["volume"] > 0)
    if vol.tail(60).notna().sum() >= 50 and vol.tail(5).notna().sum() >= 4:
        out["mkt_log_rel_volume_5_60"] = float(np.log(vol.tail(5).mean() / vol.tail(60).mean()))
    else:
        out["mkt_log_rel_volume_5_60"] = None
    out["mkt_prev_session_abs_ret"] = float(abs(lr.iloc[-1])) if len(lr) else None
    return out


def benchmark_absoc(hist: pd.DataFrame | None, n: int = 20) -> float | None:
    if hist is None or hist.empty:
        return None
    a = _absoc(hist)
    return float(a.tail(n).mean()) if len(a) >= n else None


def context_features(cutoff_utc: pd.Timestamp, window_start_utc: pd.Timestamp,
                     acceptance_utc: pd.Timestamp, report_date, earnings_8k: pd.DataFrame,
                     lag: pd.Timedelta) -> tuple[dict, dict]:
    """earnings_8k: this company's 8-Ks with Item 2.02, columns acceptance_utc, accession_number."""
    out: dict = {}
    prov: dict = {}
    if earnings_8k is not None and len(earnings_8k):
        avail = earnings_8k["acceptance_utc"] + lag
        known = earnings_8k[avail <= cutoff_utc]
        in_win = known[known["acceptance_utc"] >= window_start_utc]
        out["ctx_earnings_8k_in_window"] = int(len(in_win) > 0)
        prov["earnings_8k_in_window"] = list(in_win["accession_number"])
        if len(known):
            last = known["acceptance_utc"].max()
            days = (cutoff_utc - last).total_seconds() / 86400
            out["ctx_log_days_since_earnings_8k"] = float(np.log1p(days)) if days <= 400 else None
            prov["last_earnings_8k_acceptance_utc"] = str(last)
        else:
            out["ctx_log_days_since_earnings_8k"] = None
    else:
        out["ctx_earnings_8k_in_window"] = 0
        out["ctx_log_days_since_earnings_8k"] = None
    if report_date:
        lag_days = (acceptance_utc.tz_convert("America/New_York").normalize().tz_localize(None)
                    - pd.Timestamp(report_date)).days
        out["ctx_log_filing_lag_days"] = float(np.log1p(max(lag_days, 0)))
    else:
        out["ctx_log_filing_lag_days"] = None
    return out, prov
