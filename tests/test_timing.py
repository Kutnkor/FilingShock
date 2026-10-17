"""Target-session selection, holidays, early closes, DST and the after-hours dissemination rule."""

import pandas as pd
import pytest

from filingshock.timing import TimingPolicy, assumed_availability, classify_timing

POL = TimingPolicy(cutoff_minutes_before_open=30, availability_lag_minutes=15)


def et(s):
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


def run(cal, acc_et, filing_date=None):
    a = et(acc_et)
    return classify_timing(a, filing_date or str(a.tz_convert("America/New_York").date()), cal, POL)


def test_after_close_targets_next_session(cal):
    r = run(cal, "2024-06-26 16:03")
    assert r.status == "eligible"
    assert r.target_session == pd.Timestamp("2024-06-27")
    assert r.decision_cutoff_utc == et("2024-06-27 09:00")
    assert r.assumed_available_utc == et("2024-06-26 16:18")


def test_premarket_filing_targets_same_day(cal):
    r = run(cal, "2024-03-27 08:05")
    assert r.status == "eligible" and r.target_session == pd.Timestamp("2024-03-27")


def test_premarket_after_cutoff_is_excluded(cal):
    r = run(cal, "2024-03-27 08:50")   # available 09:05 > cutoff 09:00
    assert r.status == "excluded" and r.reason == "available_after_decision_cutoff"


def test_accepted_during_session_is_excluded(cal):
    r = run(cal, "2024-03-27 11:00")
    assert r.status == "excluded" and r.reason == "accepted_during_regular_session"


def test_exactly_at_close_counts_as_after_close(cal):
    assert run(cal, "2024-03-27 16:00").status == "eligible"


def test_early_close_day(cal):
    # 2023-11-24 closed at 13:00 ET; a 13:30 filing is after that close; next session is Monday
    r = run(cal, "2023-11-24 13:30")
    assert r.status == "eligible" and r.target_session == pd.Timestamp("2023-11-27")
    assert "previous_session_early_close" in r.flags
    # 12:30 on the same day is during the (shortened) session
    assert run(cal, "2023-11-24 12:30").reason == "accepted_during_regular_session"


def test_holiday_is_skipped(cal):
    # 2024-07-03 early close 13:00; 2024-07-04 holiday; next session 2024-07-05
    r = run(cal, "2024-07-03 16:30")
    assert r.target_session == pd.Timestamp("2024-07-05")


@pytest.mark.parametrize("acc,target,cutoff_utc_hour", [
    ("2024-03-08 16:30", "2024-03-11", 13),   # Friday EST -> Monday EDT: 09:00 EDT = 13:00 UTC
    ("2024-11-01 16:30", "2024-11-04", 14),   # Friday EDT -> Monday EST: 09:00 EST = 14:00 UTC
])
def test_dst_transitions(cal, acc, target, cutoff_utc_hour):
    r = run(cal, acc)
    assert r.target_session == pd.Timestamp(target)
    assert r.decision_cutoff_utc.hour == cutoff_utc_hour and r.decision_cutoff_utc.minute == 0


def test_after_1730_uses_next_day_dissemination():
    a = et("2024-06-26 18:10")
    avail, method, _ = assumed_availability(a, "2024-06-27", POL)
    assert method == "next_day_dissemination_plus_lag"
    assert avail == et("2024-06-27 06:15")


def test_friday_after_hours_filing_dated_monday(cal):
    r = classify_timing(et("2024-03-08 19:00"), "2024-03-11", cal, POL)
    assert r.status == "eligible"
    assert r.assumed_available_utc == et("2024-03-11 06:15")
    assert r.target_session == pd.Timestamp("2024-03-11")


def test_missing_acceptance_is_excluded(cal):
    r = classify_timing(None, "2024-01-02", cal, POL)
    assert r.status == "excluded" and r.reason == "missing_acceptance_timestamp"


def test_index_page_acceptance_parsing_and_timezone():
    from filingshock.sources.sec import parse_index_page_accepted
    html = (b'<div class="infoHead">Filing Date</div>\n<div class="info">2019-04-24</div>\n'
            b'         <div class="infoHead">Accepted</div>\n         <div class="info">2019-04-24 16:10:10</div>')
    val = parse_index_page_accepted(html)
    assert val == "2019-04-24 16:10:10"
    utc = pd.Timestamp(val).tz_localize("America/New_York").tz_convert("UTC")
    assert utc == pd.Timestamp("2019-04-24 20:10:10", tz="UTC")       # EDT: +4 h, not the API's 00:10:10Z
    assert parse_index_page_accepted(b"<html>no field</html>") is None
