"""Information-cutoff enforcement through the real event-construction path."""

import numpy as np
import pandas as pd
import pytest

from filingshock.events import build_candidates
from filingshock.features.financial import admissible_facts, financial_features, quarterly_flow
from filingshock.modeling.models import assert_admissible_columns
from filingshock.sources.market import PriceStore
from filingshock.storage import write_parquet_atomic


def _mutate_after(layout, symbol, after: str, factor: float):
    store = PriceStore(layout, "yfinance")
    df = store.load(symbol)
    m = df["session"] >= pd.Timestamp(after)
    for c in ("open", "high", "low", "close"):
        df.loc[m, c] = df.loc[m, c] * factor
    df.loc[m, "close"] = df.loc[m, "close"] * 1.07       # also change open-to-close moves
    df.loc[m, "high"] = df.loc[m, ["open", "close", "high"]].max(axis=1)
    write_parquet_atomic(store.path(symbol), df)


def test_funnel_stages_on_fixture(mini_layout):
    cfg, layout = mini_layout
    disc, parts = build_candidates(cfg, layout)
    by = disc.set_index("accession_number")
    assert by.loc["0000000001-24-000002", "exclusion_reason"] == "duplicate_company_target_session"
    assert by.loc["0000000001-24-000003", "exclusion_reason"] == "amendment_not_primary_form"
    assert by.loc["0000000001-24-000004", "exclusion_reason"] == "accepted_during_regular_session"
    assert pd.isna(by.loc["0000000001-24-000001", "exclusion_stage"])
    assert by.loc["0000000001-24-000005", "target_session"] == pd.Timestamp("2024-10-30")
    assert parts["outcomes"]["event_id"].is_unique


def test_future_data_mutation_leaves_features_unchanged(mini_layout):
    cfg, layout = mini_layout
    _, before = build_candidates(cfg, layout)
    # mutate everything from the first event's target session onwards, for the stock and benchmarks
    for sym in ("TEST", "SPY", "IGV"):
        _mutate_after(layout, sym, "2024-05-02", 1.3)
    _, after = build_candidates(cfg, layout)
    f0 = before["market_features"].set_index("event_id").sort_index()
    f1 = after["market_features"].set_index("event_id").sort_index()
    eid = "ev-0000000001-24-000001"
    cols = [c for c in f0.columns if c != "mkt_prev_event_abs_ratio"]
    pd.testing.assert_series_equal(f0.loc[eid, cols], f1.loc[eid, cols])
    o0 = before["outcomes"].set_index("event_id").loc[eid]
    o1 = after["outcomes"].set_index("event_id").loc[eid]
    assert o0["threshold_abs_move"] == o1["threshold_abs_move"]      # threshold is past-only
    assert o0["abs_r"] != o1["abs_r"]                                # the outcome itself did change


def test_design_matrix_guard_rejects_outcome_fields():
    catalog = {"mkt_absoc_mean_20", "txt_mdna_prop_added", "y", "abs_r", "event_id", "target_session"}
    assert_admissible_columns(["mkt_absoc_mean_20", "txt_mdna_prop_added__missing"], catalog)
    for bad in ["y", "abs_r", "event_id", "target_session"]:
        with pytest.raises(ValueError):
            assert_admissible_columns([bad], catalog)


def _facts(rows):
    df = pd.DataFrame(rows, columns=["concept", "start", "end", "val", "accn", "form"])
    df["unit"] = "USD"
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["duration_days"] = (df["end"] - df["start"]).dt.days
    return df


def test_restatement_after_cutoff_is_not_used():
    facts = _facts([
        ("Revenues", "2024-01-01", "2024-03-31", 100.0, "A1", "10-Q"),
        ("Revenues", "2024-01-01", "2024-03-31", 90.0, "A2", "10-Q/A"),     # restated later
    ])
    avail = {"A1": pd.Timestamp("2024-05-01", tz="UTC"), "A2": pd.Timestamp("2024-09-01", tz="UTC")}
    f = admissible_facts(facts, avail, pd.Timestamp("2024-06-01", tz="UTC"))
    assert list(f["val"]) == [100.0]
    f2 = admissible_facts(facts, avail, pd.Timestamp("2024-10-01", tz="UTC"))
    assert list(f2["val"]) == [90.0]                                       # latest admissible report wins


def test_quarterly_versus_year_to_date_flows():
    facts = _facts([
        ("NetCashProvidedByUsedInOperatingActivities", "2024-01-01", "2024-06-30", 300.0, "Q2", "10-Q"),
        ("NetCashProvidedByUsedInOperatingActivities", "2024-01-01", "2024-03-31", 120.0, "Q1", "10-Q"),
        # different fiscal-year start: must not be differenced against the 6-month value
        ("NetCashProvidedByUsedInOperatingActivities", "2023-10-01", "2024-03-31", 999.0, "X", "10-Q"),
    ])
    facts["available_utc"] = pd.Timestamp("2024-01-01", tz="UTC")
    v = quarterly_flow(facts, ["NetCashProvidedByUsedInOperatingActivities"], pd.Timestamp("2024-06-30"))
    assert v.value == 180.0 and v.method == "ytd_difference"


def test_missing_financials_stay_missing_not_zero():
    facts = _facts([("Revenues", "2024-01-01", "2024-03-31", 100.0, "A1", "10-Q")])
    feats, prov = financial_features(facts, {"A1": pd.Timestamp("2024-05-01", tz="UTC")},
                                     pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2024-03-31"))
    assert feats["fin_rev_yoy"] is None and feats["fin_op_margin"] is None
    assert not any(isinstance(v, (int, float)) and v == 0 and not np.isnan(v) for v in feats.values() if v is not None)
