"""Open-to-close target, past-only threshold, eligibility rules."""

import numpy as np
import pandas as pd

from filingshock.sources.market import bar_quality
from filingshock.target import TargetSpec, historical_threshold, realized_outcome

from .conftest import make_bars

SPEC = TargetSpec(lookback=60, multiplier=2.0, max_window=90, min_denominator=0.001)


def bars(n=120, seed=0):
    s = pd.bdate_range("2024-01-02", periods=n)
    return bar_quality(make_bars(s, seed), 0.5)


def test_open_to_close_return_and_label():
    b = bars()
    th = historical_threshold(b.iloc[:100], SPEC)
    tgt = b.iloc[100].copy()
    tgt["open"], tgt["close"] = 100.0, 100.0 * (1 + 3 * th.m)
    tgt["high"], tgt["low"] = tgt["close"] * 1.001, 99.9
    tgt["valid"] = True
    o = realized_outcome(tgt, th)
    assert np.isclose(o.r, 3 * th.m) and np.isclose(o.a, 3 * th.m) and o.y == 1
    tgt["close"] = 100.0 * (1 - 1.5 * th.m)
    o = realized_outcome(tgt, th)
    assert o.r < 0 and o.y == 0          # magnitude below 2m, direction irrelevant


def test_threshold_uses_only_prior_sessions():
    b = bars()
    th = historical_threshold(b.iloc[:100], SPEC)
    expected = b.iloc[40:100]["oc_ret"].abs().mean()
    assert np.isclose(th.m, expected) and th.n_valid == 60
    assert th.window_last_session == b.iloc[99]["session"]


def test_changing_target_close_changes_outcome_not_threshold():
    b = bars()
    hist = b.iloc[:100]
    th1 = historical_threshold(hist, SPEC)
    t = b.iloc[100].copy()
    o1 = realized_outcome(t, th1)
    t2 = t.copy()
    t2["close"] = t["open"] * 1.2
    t2["high"] = t2["close"]
    th2 = historical_threshold(hist, SPEC)
    o2 = realized_outcome(t2, th2)
    assert th1.m == th2.m and o1.a != o2.a and o2.y == 1


def test_overnight_gap_is_not_the_target():
    b = bars()
    th = historical_threshold(b.iloc[:100], SPEC)
    t = b.iloc[100].copy()
    t["open"] = b.iloc[99]["close"] * 1.25          # 25% overnight gap
    t["close"] = t["open"] * 1.0001
    t["high"], t["low"] = t["open"] * 1.001, t["open"] * 0.999
    assert realized_outcome(t, th).y == 0


def test_insufficient_history():
    b = bars(n=50)
    assert historical_threshold(b, SPEC).status == "insufficient_history"


def test_invalid_bars_are_skipped_not_zeroed():
    b = bars()
    b.loc[95:99, "open"] = np.nan
    q = bar_quality(b.drop(columns=["oc_ret", "valid"]), 0.5)
    th = historical_threshold(q.iloc[:100], SPEC)
    assert th.status == "ok" and th.n_valid == 60
    assert pd.Timestamp(th.window_last_session) == q.iloc[94]["session"]


def test_near_zero_denominator():
    s = pd.bdate_range("2024-01-02", periods=100)
    b = make_bars(s)
    b["close"] = b["open"] * 1.00001
    b["high"], b["low"] = b[["open", "close"]].max(axis=1), b[["open", "close"]].min(axis=1)
    assert historical_threshold(bar_quality(b, 0.5), SPEC).status == "near_zero_denominator"


def test_extreme_return_marks_bar_invalid():
    s = pd.bdate_range("2024-01-02", periods=5)
    b = make_bars(s)
    b.loc[2, "close"] = b.loc[2, "open"] * 1.8
    b.loc[2, "high"] = b.loc[2, "close"]
    q = bar_quality(b, 0.5)
    assert not q.loc[2, "valid"] and np.isnan(q.loc[2, "oc_ret"])
