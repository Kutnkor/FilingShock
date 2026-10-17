"""Exchange sessions and the information-timing policy (docs/timing_policy.md).

All timestamps are handled as timezone-aware UTC. Exchange schedules come from the
exchange_calendars XNYS calendar, which encodes holidays, early closes and DST.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime
from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

ET = "America/New_York"


@lru_cache(maxsize=4)
def _calendar(name: str):
    return xcals.get_calendar(name, start="2005-01-03", end="2027-12-31")


class SessionCalendar:
    """Thin wrapper with vectorised lookups over regular-session opens and closes."""

    def __init__(self, name: str = "XNYS"):
        self.name = name
        cal = _calendar(name)
        sched = cal.schedule
        self.sessions = pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
        self.opens = pd.DatetimeIndex(sched["open"]).tz_convert("UTC")
        self.closes = pd.DatetimeIndex(sched["close"]).tz_convert("UTC")
        self._open_ns = self.opens.asi8
        self._close_ns = self.closes.asi8
        self._pos = {d: i for i, d in enumerate(self.sessions)}

    def index_of(self, session: pd.Timestamp) -> int | None:
        return self._pos.get(pd.Timestamp(session).normalize().tz_localize(None)
                             if pd.Timestamp(session).tzinfo else pd.Timestamp(session).normalize())

    def is_session(self, day) -> bool:
        return self.index_of(pd.Timestamp(day)) is not None

    def next_open_index(self, ts_utc: pd.Timestamp) -> int:
        """Index of the first session whose open is strictly after ts_utc."""
        return int(np.searchsorted(self._open_ns, pd.Timestamp(ts_utc).value, side="right"))

    def session(self, i: int) -> pd.Timestamp:
        return self.sessions[i]

    def open_utc(self, i: int) -> pd.Timestamp:
        return self.opens[i]

    def close_utc(self, i: int) -> pd.Timestamp:
        return self.closes[i]

    def is_early_close(self, i: int) -> bool:
        local = self.closes[i].tz_convert(ET)
        return (local.hour, local.minute) < (16, 0)


@dataclass
class TimingPolicy:
    cutoff_minutes_before_open: int = 30
    availability_lag_minutes: int = 15
    after_hours_cutoff_et: str = "17:30"
    next_day_dissemination_et: str = "06:00"

    @classmethod
    def from_config(cls, cfg) -> "TimingPolicy":
        t = cfg["timing"]
        return cls(int(t["cutoff_minutes_before_open"]), int(t["availability_lag_minutes"]),
                   str(t["edgar_after_hours_cutoff_et"]), str(t["edgar_next_day_dissemination_et"]))

    @staticmethod
    def _hm(s: str) -> dtime:
        h, m = s.split(":")
        return dtime(int(h), int(m))


@dataclass
class TimingResult:
    acceptance_utc: pd.Timestamp | None
    assumed_available_utc: pd.Timestamp | None
    availability_method: str
    target_session: pd.Timestamp | None = None
    target_open_utc: pd.Timestamp | None = None
    target_close_utc: pd.Timestamp | None = None
    decision_cutoff_utc: pd.Timestamp | None = None
    prev_session: pd.Timestamp | None = None
    prev_close_utc: pd.Timestamp | None = None
    status: str = "unknown"          # eligible | excluded
    reason: str | None = None
    flags: list[str] = field(default_factory=list)


def assumed_availability(acceptance_utc: pd.Timestamp, filing_date: str | None,
                         policy: TimingPolicy) -> tuple[pd.Timestamp, str, list[str]]:
    """Historical availability proxy: acceptance time plus a conservative lag.

    EDGAR does not publish a first-public-availability timestamp. Submissions that start
    transmission after 17:30 ET receive the next business day's filing date and are
    disseminated the next business day, so for those we use filing date 06:00 ET + lag.
    """
    flags: list[str] = []
    lag = pd.Timedelta(minutes=policy.availability_lag_minutes)
    acc_et = acceptance_utc.tz_convert(ET)
    method = "acceptance_plus_lag"
    avail = acceptance_utc + lag
    fdate = pd.Timestamp(filing_date).date() if filing_date else None
    after_hours = acc_et.time() > policy._hm(policy.after_hours_cutoff_et)
    if fdate is not None and fdate < acc_et.date():
        flags.append("filing_date_before_acceptance_date")
    if after_hours or (fdate is not None and fdate > acc_et.date()):
        method = "next_day_dissemination_plus_lag"
        if fdate is not None and fdate > acc_et.date():
            day = pd.Timestamp(fdate)
        else:
            day = pd.Timestamp(acc_et.date()) + pd.offsets.BDay(1)
            flags.append("after_hours_acceptance_without_next_day_filing_date")
        hm = policy._hm(policy.next_day_dissemination_et)
        dissem = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hm.hour,
                              minute=hm.minute, tz=ET).tz_convert("UTC")
        avail = max(avail, dissem + lag)
        if not after_hours:
            flags.append("next_day_filing_date_before_after_hours_cutoff")
    return avail, method, flags


def classify_timing(acceptance_utc, filing_date, cal: SessionCalendar, policy: TimingPolicy) -> TimingResult:
    """Apply the primary cohort rule: filings accepted after the preceding regular-session close
    whose assumed availability is no later than the decision cutoff of the next session."""
    if acceptance_utc is None or pd.isna(acceptance_utc):
        return TimingResult(None, None, "none", status="excluded", reason="missing_acceptance_timestamp")
    acceptance_utc = pd.Timestamp(acceptance_utc).tz_convert("UTC")
    avail, method, flags = assumed_availability(acceptance_utc, filing_date, policy)
    i = cal.next_open_index(acceptance_utc)
    if i >= len(cal.sessions) or i == 0:
        return TimingResult(acceptance_utc, avail, method, status="excluded",
                            reason="outside_calendar_range", flags=flags)
    prev_close = cal.close_utc(i - 1)
    res = TimingResult(acceptance_utc, avail, method, prev_session=cal.session(i - 1),
                       prev_close_utc=prev_close, flags=flags)
    if cal.is_early_close(i - 1):
        res.flags.append("previous_session_early_close")
    if acceptance_utc < prev_close:
        res.status, res.reason = "excluded", "accepted_during_regular_session"
        return res
    cutoff = cal.open_utc(i) - pd.Timedelta(minutes=policy.cutoff_minutes_before_open)
    res.target_session = cal.session(i)
    res.target_open_utc = cal.open_utc(i)
    res.target_close_utc = cal.close_utc(i)
    res.decision_cutoff_utc = cutoff
    if cal.is_early_close(i):
        res.flags.append("target_session_early_close")
    if avail > cutoff:
        res.status, res.reason = "excluded", "available_after_decision_cutoff"
        return res
    res.status = "eligible"
    return res
