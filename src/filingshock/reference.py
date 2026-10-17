"""Reference-filing selection.

prior_year_same_quarter (primary): the company's original 10-Q whose period end lies one year
  before the current period end (+/- tolerance for 52/53-week calendars), accepted before
  the current filing.
nearest_prior_10q (secondary): the most recent earlier original 10-Q.
The two policies are never mixed under one feature name.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ReferenceChoice:
    policy: str
    status: str                    # found | missing
    reference_accession: str | None = None
    reference_report_date: str | None = None
    period_gap_days: int | None = None
    reason: str | None = None


def select_reference(filings_10q: pd.DataFrame, current: pd.Series, policy: str,
                     tolerance_days: int = 21) -> ReferenceChoice:
    """filings_10q: original 10-Q rows (all years, all companies) with report_date and acceptance_utc."""
    same = filings_10q[(filings_10q["cik"] == current["cik"])
                       & (filings_10q["accession_number"] != current["accession_number"])
                       & (filings_10q["acceptance_utc"] < current["acceptance_utc"])]
    if pd.isna(current.get("report_date")) or not current.get("report_date"):
        return ReferenceChoice(policy, "missing", reason="current_report_date_missing")
    cur_end = pd.Timestamp(current["report_date"])
    same = same[same["report_date"].notna() & (same["report_date"] != "")]
    ends = pd.to_datetime(same["report_date"])
    same = same.assign(_end=ends)
    same = same[same["_end"] < cur_end]
    if same.empty:
        return ReferenceChoice(policy, "missing", reason="no_earlier_10q_in_history")
    if policy == "prior_year_same_quarter":
        gap = (cur_end - same["_end"]).dt.days
        cand = same.assign(_gap=gap, _dev=(gap - 364).abs())
        cand = cand[cand["_dev"] <= tolerance_days + 1]
        if cand.empty:
            return ReferenceChoice(policy, "missing", reason="prior_year_same_quarter_10q_not_found")
        best = cand.sort_values(["_dev", "acceptance_utc"]).iloc[0]
    elif policy == "nearest_prior_10q":
        best = same.sort_values(["_end", "acceptance_utc"]).iloc[-1]
        best = best.copy()
        best["_gap"] = (cur_end - best["_end"]).days
        if best["_gap"] > 200:
            return ReferenceChoice(policy, "missing", reason="nearest_prior_10q_too_old")
    else:
        raise ValueError(policy)
    return ReferenceChoice(policy, "found", best["accession_number"], str(best["report_date"]),
                           int(best["_gap"]))
