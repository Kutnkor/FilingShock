"""Primary target: unusually large open-to-close move in the target session.

    r = close / open - 1                    (target session only)
    a = |r|
    m = mean(|open-to-close|) over the previous `lookback` VALID sessions, all completed before
        the decision cutoff, drawn from at most `max_window` exchange sessions
    y = 1 if a > multiplier * m else 0

The threshold uses history strictly before the target session. Nothing here is a model input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TargetSpec:
    lookback: int = 60
    multiplier: float = 2.0
    max_window: int = 90
    min_denominator: float = 0.001

    @classmethod
    def from_config(cls, cfg) -> "TargetSpec":
        t = cfg["target"]
        return cls(int(t["lookback_sessions"]), float(t["multiplier"]), int(t["max_window_sessions"]),
                   float(t["min_denominator"]))


@dataclass
class Threshold:
    status: str                      # ok | insufficient_history | near_zero_denominator
    m: float | None = None
    threshold: float | None = None
    n_valid: int = 0
    window_first_session: pd.Timestamp | None = None
    window_last_session: pd.Timestamp | None = None
    sessions_used: list | None = None


def historical_threshold(history: pd.DataFrame, spec: TargetSpec) -> Threshold:
    """history: quality-checked bars strictly BEFORE the target session (bar_quality output)."""
    window = history.tail(spec.max_window)
    valid = window[window["valid"]]
    if len(valid) < spec.lookback:
        return Threshold("insufficient_history", n_valid=int(len(valid)))
    used = valid.tail(spec.lookback)
    m = float(used["oc_ret"].abs().mean())
    th = Threshold("ok", m, spec.multiplier * m, int(len(used)), used["session"].iloc[0],
                   used["session"].iloc[-1], [str(s.date()) for s in used["session"]])
    if not np.isfinite(m) or m < spec.min_denominator:
        th.status = "near_zero_denominator"
    return th


@dataclass
class Outcome:
    status: str                      # ok | target_bar_missing | target_bar_invalid
    open: float | None = None
    close: float | None = None
    r: float | None = None
    a: float | None = None
    y: int | None = None


def realized_outcome(target_bar: pd.Series | None, threshold: Threshold) -> Outcome:
    if target_bar is None:
        return Outcome("target_bar_missing")
    if not bool(target_bar["valid"]):
        return Outcome("target_bar_invalid", float(target_bar["open"]), float(target_bar["close"]))
    o, c = float(target_bar["open"]), float(target_bar["close"])
    r = c / o - 1.0
    a = abs(r)
    y = int(a > threshold.threshold) if threshold.status == "ok" else None
    return Outcome("ok", o, c, r, a, y)
