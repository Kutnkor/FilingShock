"""Point-in-time financial features from SEC Company Facts.

Each fact in Company Facts carries the accession number of the filing that reported it.
For an event with decision cutoff T we admit only facts whose reporting filing was accepted
(per EDGAR acceptance time, plus the availability lag) no later than T. When several admissible
filings report the same (concept, period), the latest admissible report wins, so a later
restatement can never reach an earlier event.

Concept mapping is explicit (CONCEPTS) and every derived value records the concept and the
accession it came from. Missing values stay missing; they are never treated as zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ordered fallbacks. A ratio or growth rate always uses ONE concept for numerator periods.
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueServicesNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
QUARTER_DAYS = (75, 105)
PERIODIC_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-QT"}

FIN_FEATURES = ["fin_rev_yoy", "fin_op_margin", "fin_op_margin_yoy_pp", "fin_ocf_margin",
                "fin_cash_to_assets", "fin_liab_to_assets", "fin_liab_to_assets_yoy_change"]


def load_company_facts(path: Path) -> pd.DataFrame:
    """Flatten the us-gaap USD facts we use into a long table."""
    doc = json.loads(Path(path).read_bytes())
    gaap = doc.get("facts", {}).get("us-gaap", {})
    wanted = {c for cs in CONCEPTS.values() for c in cs}
    rows = []
    for concept in wanted:
        node = gaap.get(concept)
        if not node:
            continue
        for unit, vals in node.get("units", {}).items():
            if unit != "USD":
                continue
            for v in vals:
                rows.append({"concept": concept, "unit": unit, "start": v.get("start"), "end": v.get("end"),
                             "val": float(v["val"]), "accn": v.get("accn"), "form": v.get("form"),
                             "filed": v.get("filed"), "fy": v.get("fy"), "fp": v.get("fp")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["start"] = pd.to_datetime(df["start"])
    df["duration_days"] = (df["end"] - df["start"]).dt.days
    return df


def admissible_facts(facts: pd.DataFrame, accn_available: dict[str, pd.Timestamp],
                     cutoff_utc: pd.Timestamp) -> pd.DataFrame:
    """Facts reported by periodic filings that were available by the cutoff; latest report wins."""
    if facts.empty:
        return facts
    f = facts[facts["form"].isin(PERIODIC_FORMS)].copy()
    f["available_utc"] = f["accn"].map(accn_available)
    f = f[f["available_utc"].notna() & (f["available_utc"] <= cutoff_utc)]
    f = f.sort_values("available_utc")
    key = ["concept", "start", "end"]
    f["start"] = f["start"].fillna(pd.Timestamp("1900-01-01"))
    return f.drop_duplicates(key, keep="last")


def _near(series: pd.Series, target: pd.Timestamp, tol_days: int) -> pd.Series:
    return (series - target).abs() <= pd.Timedelta(days=tol_days)


@dataclass
class FlowValue:
    value: float | None
    concept: str | None = None
    accn: str | None = None
    method: str | None = None


def quarterly_flow(f: pd.DataFrame, concepts: list[str], period_end: pd.Timestamp,
                   tol_days: int = 7, force_concept: str | None = None) -> FlowValue:
    """Three-month flow ending at period_end. Uses a reported quarterly value, else derives it as
    YTD(period_end) - YTD(previous quarter end) when both share the same fiscal-year start."""
    for concept in ([force_concept] if force_concept else concepts):
        c = f[(f["concept"] == concept) & _near(f["end"], period_end, tol_days)]
        q = c[c["duration_days"].between(*QUARTER_DAYS)]
        if not q.empty:
            r = q.iloc[-1]
            return FlowValue(r["val"], concept, r["accn"], "reported_quarter")
        ytd = c[c["duration_days"] > QUARTER_DAYS[1]].sort_values("duration_days")
        for _, y in ytd.iterrows():
            prev = f[(f["concept"] == concept) & (f["start"] == y["start"])
                     & (f["end"] < y["end"]) & ((y["end"] - f["end"]).dt.days.between(*QUARTER_DAYS))]
            if not prev.empty:
                p = prev.iloc[-1]
                return FlowValue(y["val"] - p["val"], concept, f"{y['accn']}|{p['accn']}", "ytd_difference")
    return FlowValue(None)


def instant(f: pd.DataFrame, concepts: list[str], at: pd.Timestamp, tol_days: int = 7) -> FlowValue:
    for concept in concepts:
        c = f[(f["concept"] == concept) & _near(f["end"], at, tol_days) & f["duration_days"].isna()]
        if c.empty:
            c = f[(f["concept"] == concept) & _near(f["end"], at, tol_days)
                  & (f["start"] == pd.Timestamp("1900-01-01"))]
        if not c.empty:
            r = c.iloc[-1]
            return FlowValue(r["val"], concept, r["accn"], "instant")
    return FlowValue(None)


def _ratio(num, den):
    if num is None or den is None or not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return None
    return num / den


def financial_features(facts: pd.DataFrame, accn_available: dict[str, pd.Timestamp],
                       cutoff_utc: pd.Timestamp, period_end: pd.Timestamp) -> tuple[dict, dict]:
    """Return (features, provenance). Features use only facts admissible at cutoff_utc."""
    f = admissible_facts(facts, accn_available, cutoff_utc)
    feats = {k: None for k in FIN_FEATURES}
    prov: dict = {"n_admissible_facts": int(len(f))}
    if f.empty:
        prov["status"] = "no_admissible_facts"
        return feats, prov
    prior_end = period_end - pd.Timedelta(days=364)
    # revenue: choose the first concept that has both the current and the prior-year quarter
    rev_c = rev_p = FlowValue(None)
    for concept in CONCEPTS["revenue"]:
        a = quarterly_flow(f, CONCEPTS["revenue"], period_end, force_concept=concept)
        if a.value is None:
            continue
        b = quarterly_flow(f, CONCEPTS["revenue"], prior_end, tol_days=14, force_concept=concept)
        rev_c, rev_p = a, b
        if b.value is not None:
            break
    op_c = quarterly_flow(f, CONCEPTS["operating_income"], period_end)
    op_p = quarterly_flow(f, CONCEPTS["operating_income"], prior_end, tol_days=14)
    ocf_c = quarterly_flow(f, CONCEPTS["operating_cash_flow"], period_end)
    cash = instant(f, CONCEPTS["cash"], period_end)
    assets = instant(f, CONCEPTS["assets"], period_end)
    equity = instant(f, CONCEPTS["equity"], period_end)
    assets_p = instant(f, CONCEPTS["assets"], prior_end, tol_days=14)
    equity_p = instant(f, CONCEPTS["equity"], prior_end, tol_days=14)

    g = _ratio(rev_c.value, rev_p.value)
    feats["fin_rev_yoy"] = None if g is None else g - 1.0
    feats["fin_op_margin"] = _ratio(op_c.value, rev_c.value)
    m_prior = _ratio(op_p.value, rev_p.value)
    if feats["fin_op_margin"] is not None and m_prior is not None:
        feats["fin_op_margin_yoy_pp"] = feats["fin_op_margin"] - m_prior
    feats["fin_ocf_margin"] = _ratio(ocf_c.value, rev_c.value)
    feats["fin_cash_to_assets"] = _ratio(cash.value, assets.value)
    eq_ratio = _ratio(equity.value, assets.value)
    feats["fin_liab_to_assets"] = None if eq_ratio is None else 1.0 - eq_ratio
    eq_ratio_p = _ratio(equity_p.value, assets_p.value)
    if feats["fin_liab_to_assets"] is not None and eq_ratio_p is not None:
        feats["fin_liab_to_assets_yoy_change"] = feats["fin_liab_to_assets"] - (1.0 - eq_ratio_p)
    for name, v in [("revenue_q", rev_c), ("revenue_prior_year_q", rev_p), ("operating_income_q", op_c),
                    ("operating_income_prior_year_q", op_p), ("ocf_q", ocf_c), ("cash", cash),
                    ("assets", assets), ("equity", equity), ("assets_prior_year", assets_p),
                    ("equity_prior_year", equity_p)]:
        prov[name] = {"value": v.value, "concept": v.concept, "accn": v.accn, "method": v.method}
    prov["status"] = "ok"
    return feats, prov
