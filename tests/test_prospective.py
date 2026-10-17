"""Prospective scoring: single-event features, production model, ledger recording and outcome scoring after
the target session (synthetic, offline)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from filingshock import PARSER_VERSION
from filingshock.events import build_candidates, single_event_features
from filingshock.features.market import MARKET_FEATURES
from filingshock.features.text import LM_FILE
from filingshock.forecasts import CATEGORY, ForecastLedger, LedgerError, record_pending_forecasts
from filingshock.modeling.experiment import model_specs
from filingshock.sources.market import PriceStore
from filingshock.sources.sec import SecClient
from filingshock.storage import write_json_atomic, write_parquet_atomic
from filingshock.target import TargetSpec, historical_threshold

PENDING = "0000000001-24-000005"          # pre-market 10-Q accepted 2024-10-30 07:45 ET, target 2024-10-30
REFERENCE = "0000000001-23-000010"        # prior-year same-quarter 10-Q (outside the mini filing window)
CUTOFF = pd.Timestamp("2024-10-30 13:00", tz="UTC")      # 09:00 ET = open - 30 min
BEFORE = datetime(2024, 10, 30, 12, 10, tzinfo=timezone.utc)
FIRST_SEEN = pd.Timestamp("2024-10-30 07:47", tz="America/New_York").tz_convert("UTC")


def _paras(texts):
    out, pos = [], 0
    for i, (t, head) in enumerate(texts):
        out.append({"ordinal": i, "block_idx": i, "text": t, "is_heading": head, "char_start": pos,
                    "char_end": pos + len(t)})
        pos += len(t) + 1
    return out


def _section(layout, acc, mdna, risk):
    summary = {"accession_number": acc, "cik": 1, "primary_document": "doc.htm", "parser_version": PARSER_VERSION,
               "download_status": "ok", "source_sha256": "0" * 64, "mdna_status": "extracted", "risk_status": "extracted"}
    write_json_atomic(layout.sections_dir / f"{acc}.json",
                      {"parser_version": PARSER_VERSION, "summary": summary, "mdna": _paras(mdna), "risk": _paras(risk)})


def _synthetic_dataset(layout, years=range(2016, 2024), per_year=40, seed=7):
    """Matured research events to fit the production models on (no relation to the mini filings)."""
    rng = np.random.default_rng(seed)
    feats = model_specs()["C"]
    ev, fe, oc = [], [], []
    for yr in years:
        sessions = pd.bdate_range(f"{yr}-01-05", f"{yr}-12-15")[::6][:per_year]
        for k, s in enumerate(sessions):
            eid = f"ev-syn-{yr}-{k:03d}"
            cutoff = (s.tz_localize("America/New_York") + pd.Timedelta(hours=9)).tz_convert("UTC")
            x = rng.normal(size=len(feats))
            p = 1 / (1 + np.exp(-(-1.1 + 0.9 * x[0] - 0.6 * x[-1])))
            ev.append({"event_id": eid, "ticker": "SYN", "cik": 99, "target_session": s, "decision_cutoff_utc": cutoff,
                       "acceptance_utc": cutoff - pd.Timedelta(hours=14), "eligibility_status": "eligible"})
            fe.append({"event_id": eid, **dict(zip(feats, x)), "decision_cutoff_utc": cutoff})
            oc.append({"event_id": eid, "y": int(rng.random() < p), "abs_r": 0.01, "threshold_abs_move": 0.02,
                       "outcome_available_utc": cutoff + pd.Timedelta(hours=7)})
    write_parquet_atomic(layout.processed / "events.parquet", pd.DataFrame(ev))
    write_parquet_atomic(layout.processed / "features.parquet", pd.DataFrame(fe))
    write_parquet_atomic(layout.processed / "outcomes.parquet", pd.DataFrame(oc))
    write_json_atomic(layout.processed / "dataset_manifest.json", {"dataset_id": "ds-synthetic"})


@pytest.fixture()
def live_layout(mini_layout):
    """mini_layout plus what a live run needs: first-seen times, a reference 10-Q, extracted sections,
    a tiny LM dictionary and a matured research dataset."""
    cfg, layout = mini_layout
    f = pd.read_parquet(layout.interim / "filings.parquet")
    ref = f[f["accession_number"] == PENDING].copy()
    ref["accession_number"], ref["filing_date"], ref["report_date"] = REFERENCE, "2023-10-31", "2023-09-30"
    ref["acceptance_utc"] = pd.Timestamp("2023-10-31 16:10", tz="America/New_York").tz_convert("UTC")
    ref["in_window"] = False
    f = pd.concat([f, ref], ignore_index=True)
    f["first_seen_utc"] = f["acceptance_utc"] + pd.Timedelta(minutes=2)
    write_parquet_atomic(layout.interim / "filings.parquet", f)
    _section(layout, REFERENCE,
             [("Overview", True), ("Revenue grew on subscription demand.", False),
              ("We may face uncertain demand in some regions.", False)],
             [("Risk factors", True), ("Competition is intense.", False)])
    _section(layout, PENDING,
             [("Overview", True), ("Revenue grew on subscription demand.", False),
              ("Litigation and a decline in bookings created a loss in one segment.", False),
              ("We may face uncertain demand in some regions and uncertain currency effects.", False)],
             [("Risk factors", True), ("Competition is intense.", False), ("New regulation may be required.", False)])
    (layout.raw_resources / LM_FILE).write_text(
        "Word,Negative,Uncertainty,Litigious,Constraining\nLOSS,2009,0,0,0\nDECLINE,2009,0,0,0\n"
        "UNCERTAIN,0,2009,0,0\nMAY,0,2009,0,0\nLITIGATION,0,0,2009,0\nREQUIRED,0,0,0,2009\n")
    _synthetic_dataset(layout)
    return cfg, layout


def _client(cfg, layout):
    return SecClient(cfg, layout, offline=True)      # cached sections only; never touches the network


def test_pending_event_is_recorded_before_cutoff(live_layout):
    cfg, layout = live_layout
    res = record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: BEFORE)
    assert [(r["accession_number"], r["status"]) for r in res] == [(PENDING, "recorded")], res
    led = ForecastLedger(layout)
    (rec,) = led.records()
    assert led.verify_chain() == (True, "ok")
    assert rec["event_id"] == f"ev-{PENDING}" and rec["prediction_category"] == CATEGORY
    assert pd.Timestamp(rec["generated_at_utc"]) == pd.Timestamp(BEFORE) < pd.Timestamp(rec["decision_cutoff_utc"]) == CUTOFF
    assert pd.Timestamp(rec["observed_at_utc"]) == FIRST_SEEN
    assert rec["model_version"] == "logistic_l2:prod2024:ds-synthetic" and rec["dataset_id"] == "ds-synthetic"
    assert (rec["training_data_through_year"], rec["calibration_year"]) == (2022, 2023)
    assert 0 < rec["p_B"] < 1 and 0 < rec["p_C"] < 1 and rec["probability"] == rec["p_C"]
    assert 0 < rec["p_A"] < 1 and 0 < rec["p_constant"] < 1
    assert rec["reference_accession"] == REFERENCE and rec["target_session"] == "2024-10-30"
    # threshold = 2 x mean |open-to-close| of the 60 valid sessions before the target session
    bars = PriceStore(layout, "yfinance").load("TEST")
    hist = bars[bars["session"] < "2024-10-30"].tail(60)
    assert rec["threshold_abs_move"] == pytest.approx(2 * (hist["close"] / hist["open"] - 1).abs().mean())
    assert rec["features"]["mkt_prev_event_abs_ratio"] is not None
    assert rec["features"]["txt_mdna_prop_added"] > 0
    assert list((layout.forecasts / "models").glob("logistic_l2__prod2024__ds-synthetic__*.json"))
    # a second run never overwrites or duplicates the forecast
    again = record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: BEFORE)
    assert [r["status"] for r in again] == ["already_recorded"]
    assert len(led.records()) == 1


def test_nothing_is_recorded_at_or_after_cutoff(live_layout):
    cfg, layout = live_layout
    for now in (CUTOFF, CUTOFF + pd.Timedelta(hours=3)):
        assert record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda t=now: t.to_pydatetime()) == []
    # pending when scoring starts, but the cutoff passes before the record is appended
    ticks = [BEFORE, (CUTOFF + pd.Timedelta(seconds=1)).to_pydatetime()]
    res = record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: ticks.pop(0) if len(ticks) > 1 else ticks[0])
    assert res[0]["status"] == "skipped" and "decision cutoff" in res[0]["reason"]
    assert ForecastLedger(layout).records() == []


def test_changed_model_parameters_create_a_new_forecast(live_layout):
    cfg, layout = live_layout
    first = record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: BEFORE)[0]["record"]
    changed = cfg.with_overrides({"model.c_grid": [0.03]})
    second = record_pending_forecasts(changed, layout, _client(changed, layout), clock=lambda: BEFORE)[0]
    assert second["status"] == "recorded"
    assert second["record"]["model_version"] == first["model_version"]
    assert second["record"]["model_parameters_sha256"] != first["model_parameters_sha256"]
    assert second["record"]["forecast_id"] != first["forecast_id"]
    again = record_pending_forecasts(changed, layout, _client(changed, layout), clock=lambda: BEFORE)[0]
    assert again["status"] == "already_recorded"
    assert len(ForecastLedger(layout).records()) == 2


def test_tampered_model_manifest_blocks_recording(live_layout):
    import json

    cfg, layout = live_layout
    record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: BEFORE)
    path = next((layout.forecasts / "models").glob("*.json"))
    content = json.loads(path.read_text())
    content["year"] = 1900
    path.write_text(json.dumps(content))
    res = record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: BEFORE)
    assert res[0]["status"] == "skipped" and "integrity verification" in res[0]["reason"]
    assert len(ForecastLedger(layout).records()) == 1
    assert json.loads(path.read_text())["year"] == 1900   # damaged evidence is not silently overwritten


def test_filings_not_yet_accepted_or_seen_are_not_pending(live_layout):
    cfg, layout = live_layout
    early = datetime(2024, 10, 30, 11, 46, tzinfo=timezone.utc)      # 07:46 ET: accepted, not yet seen
    assert record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: early) == []
    assert record_pending_forecasts(cfg, layout, _client(cfg, layout),
                                    clock=lambda: datetime(2024, 10, 29, 20, 0, tzinfo=timezone.utc)) == []
    assert ForecastLedger(layout).records() == []


def test_filing_observed_after_cutoff_is_not_recorded(live_layout):
    cfg, layout = live_layout
    f = pd.read_parquet(layout.interim / "filings.parquet")
    f.loc[f["accession_number"] == PENDING, "first_seen_utc"] = CUTOFF + pd.Timedelta(minutes=1)
    write_parquet_atomic(layout.interim / "filings.parquet", f)
    assert record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: BEFORE) == []
    led = ForecastLedger(layout)
    rec = {"event_id": "ev-x", "decision_cutoff_utc": str(CUTOFF), "observed_at_utc": str(CUTOFF + pd.Timedelta(minutes=1))}
    with pytest.raises(LedgerError, match="observed after the decision cutoff"):
        led.append(rec, now=BEFORE)
    with pytest.raises(LedgerError, match="later than the generation time"):
        led.append({**rec, "observed_at_utc": str(pd.Timestamp(BEFORE) + pd.Timedelta(minutes=1))}, now=BEFORE)
    assert led.records() == []


def _rewrite_from(layout, symbol, start: str, drop: bool):
    store = PriceStore(layout, "yfinance")
    df = store.load(symbol)
    m = df["session"] >= pd.Timestamp(start)
    if drop:
        df = df[~m]                                       # live situation: the target bar does not exist yet
    else:
        for c in ("open", "high", "low", "close", "volume"):
            df.loc[m, c] = df.loc[m, c] * 1.4
        df.loc[m, "close"] = df.loc[m, "close"] * 1.09
        df.loc[m, "high"] = df.loc[m, ["open", "close", "high"]].max(axis=1)
    write_parquet_atomic(store.path(symbol), df)


@pytest.mark.parametrize("drop", [False, True])
def test_single_event_features_ignore_bars_at_or_after_target_session(live_layout, drop):
    cfg, layout = live_layout
    before = single_event_features(cfg, layout, PENDING, _client(cfg, layout))
    assert before.status == "ok", (before.stage, before.reason)
    for sym in ("TEST", "SPY", "IGV"):
        _rewrite_from(layout, sym, "2024-10-30", drop)
    after = single_event_features(cfg, layout, PENDING, _client(cfg, layout))
    assert after.status == "ok", (after.stage, after.reason)
    pd.testing.assert_series_equal(pd.Series(before.features, dtype=float), pd.Series(after.features, dtype=float))
    assert before.threshold.threshold == after.threshold.threshold


def test_single_event_market_features_match_historical_build(live_layout):
    cfg, layout = live_layout
    _, parts = build_candidates(cfg, layout)
    hist = parts["market_features"].set_index("event_id").loc[f"ev-{PENDING}"]
    live = single_event_features(cfg, layout, PENDING, _client(cfg, layout))
    for k in MARKET_FEATURES:
        lv, hv = live.features[k], hist[k]
        assert (lv is None and pd.isna(hv)) or lv == pytest.approx(hv), k
    bars = PriceStore(layout, "yfinance").load("TEST")
    from filingshock.sources.market import bar_quality
    q = bar_quality(bars, 0.5)
    th = historical_threshold(q[q["session"] < "2024-10-30"], TargetSpec.from_config(cfg))
    assert live.threshold.threshold == pytest.approx(th.threshold)


def test_stale_price_history_blocks_scoring(live_layout):
    cfg, layout = live_layout
    for sym in ("TEST", "SPY", "IGV"):
        _rewrite_from(layout, sym, "2024-10-25", drop=True)       # last bar two sessions before the target
    sf = single_event_features(cfg, layout, PENDING, _client(cfg, layout))
    assert (sf.status, sf.stage, sf.reason) == ("excluded", "valid_price_history", "price_history_stale")


def test_first_seen_is_kept_across_discovery_runs(tmp_path):
    from filingshock.discovery import carry_first_seen
    from filingshock.storage import DataLayout
    layout = DataLayout(tmp_path).ensure()
    t0, t1 = pd.Timestamp("2026-10-01", tz="UTC"), pd.Timestamp("2026-10-20", tz="UTC")
    first = carry_first_seen(pd.DataFrame({"accession_number": ["a"]}), layout, now=t0)
    write_parquet_atomic(layout.interim / "filings.parquet", first)
    second = carry_first_seen(pd.DataFrame({"accession_number": ["a", "b"]}), layout, now=t1)
    assert list(second["first_seen_utc"]) == [t0, t1]


def test_prospective_rows_stay_out_of_research_tables(live_layout):
    cfg, layout = live_layout
    record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: BEFORE)
    from filingshock.forecasts import ledger_frame
    from filingshock.ui.data import AppData, event_table
    events = pd.read_parquet(layout.processed / "events.parquet").head(3)
    for c, v in [("company_name", "Syn"), ("form_type", "10-Q"), ("report_date", None), ("exclusion_reason", None)]:
        events[c] = v
    data = AppData("research", layout.root, events, None, None, None, {"dataset_id": "ds-synthetic"}, None, None,
                   None, None, None, prospective=ledger_frame(layout))
    research = event_table(data)
    explorer = event_table(data, include_prospective=True)
    assert len(research) == 3 and CATEGORY not in set(research["prediction_status"])
    extra = explorer[explorer["prediction_status"] == CATEGORY]
    assert list(extra["event_id"]) == [f"ev-{PENDING}"] and list(extra["eligibility_status"]) == ["prospective"]
    assert (explorer["eligibility_status"] == "eligible").sum() == (research["eligibility_status"] == "eligible").sum()
    assert extra["p_C"].iloc[0] == pytest.approx(ForecastLedger(layout).records()[0]["p_C"])


# ---------------------------------------------------------------- outcome scoring after the target session
TARGET = pd.Timestamp("2024-10-30")
SETTLED = pd.Timestamp("2024-10-30 21:00", tz="UTC")        # 16:00 ET close + BAR_SETTLE_MINUTES
SCORE_AT = datetime(2024, 10, 31, 12, 0, tzinfo=timezone.utc)


def _recorded(live_layout) -> dict:
    cfg, layout = live_layout
    (res,) = record_pending_forecasts(cfg, layout, _client(cfg, layout), clock=lambda: BEFORE)
    assert res["status"] == "recorded", res
    return res["record"]


def _set_retrieved(layout, when):
    store = PriceStore(layout, "yfinance")
    man = store.manifest_entries()
    for sym in man:
        man[sym]["retrieved_at"] = pd.Timestamp(when).isoformat()
    write_json_atomic(store.manifest, man)


def _set_target_move(layout, r: float):
    store = PriceStore(layout, "yfinance")
    df = store.load("TEST")
    i = df.index[df["session"] == TARGET][0]
    o = df.at[i, "open"]
    df.at[i, "close"] = o * (1 + r)
    df.at[i, "high"], df.at[i, "low"] = max(o, o * (1 + r)) * 1.001, min(o, o * (1 + r)) * 0.999
    write_parquet_atomic(store.path("TEST"), df)


def _files(root):
    import hashlib
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_outcome_uses_the_threshold_stored_in_the_forecast(live_layout):
    from filingshock.prospective_outcomes import OutcomeLog, score_prospective_forecasts
    from filingshock.sources.market import bar_quality
    cfg, layout = live_layout
    rec = _recorded(live_layout)
    th = rec["threshold_abs_move"]
    _set_target_move(layout, 1.5 * th)                       # a large move against the stored threshold
    store = PriceStore(layout, "yfinance")
    df = store.load("TEST")
    hist = df["session"] < TARGET                             # history revised after the forecast: |oc| = 2 x stored
    sign = np.where(np.arange(hist.sum()) % 2 == 0, 1.0, -1.0)
    df.loc[hist, "close"] = df.loc[hist, "open"] * (1 + sign * 2 * th)
    df.loc[hist, "high"] = df.loc[hist, ["open", "close"]].max(axis=1) * 1.001
    df.loc[hist, "low"] = df.loc[hist, ["open", "close"]].min(axis=1) * 0.999
    write_parquet_atomic(store.path("TEST"), df)
    _set_retrieved(layout, SETTLED + pd.Timedelta(minutes=5))
    res = score_prospective_forecasts(cfg, layout, clock=lambda: SCORE_AT)
    assert [r["status"] for r in res] == ["scored"], res
    out = res[0]["record"]
    assert out["forecast_id"] == rec["forecast_id"] and out["forecast_record_sha256"] == rec["record_sha256"]
    assert out["threshold_abs_move"] == th and out["threshold_source"] == "forecast_record"
    assert out["abs_r"] == pytest.approx(1.5 * th) and out["y"] == 1 and out["outcome_status"] == "ok"
    assert out["brier_B"] == pytest.approx((rec["p_B"] - 1) ** 2) and out["brier_C"] == pytest.approx((rec["p_C"] - 1) ** 2)
    assert out["brier_A"] == pytest.approx((rec["p_A"] - 1) ** 2)
    assert out["brier_constant"] == pytest.approx((rec["p_constant"] - 1) ** 2)
    # a threshold recomputed from today's store would have said y = 0
    q = bar_quality(store.load("TEST"), 0.5)
    recomputed = historical_threshold(q[q["session"] < TARGET], TargetSpec.from_config(cfg)).threshold
    assert recomputed > out["abs_r"]
    assert OutcomeLog(layout).verify_chain() == (True, "ok")


def test_nothing_is_scored_before_the_close_or_a_fresh_target_bar(live_layout):
    from filingshock.prospective_outcomes import OutcomeLog, score_prospective_forecasts
    cfg, layout = live_layout
    rec = _recorded(live_layout)

    def reason(now):
        (r,) = score_prospective_forecasts(cfg, layout, clock=lambda: now)
        assert r["status"] == "pending", r
        return r["reason"].split(":")[0]

    assert reason(SCORE_AT) == "price_retrieval_time_unknown"
    _set_retrieved(layout, SETTLED - pd.Timedelta(minutes=30))      # refreshed after the close, before it settled
    assert reason(SCORE_AT) == "prices_not_refreshed_after_close"
    _set_retrieved(layout, SETTLED)
    for now in (BEFORE, CUTOFF + pd.Timedelta(hours=3), SETTLED - pd.Timedelta(seconds=1)):   # session open / closing
        assert reason(pd.Timestamp(now).to_pydatetime()) == "target_session_not_closed"
    _rewrite_from(layout, "TEST", "2024-10-30", drop=True)
    assert reason(SETTLED.to_pydatetime()) == "target_bar_not_in_price_store"
    assert OutcomeLog(layout).records() == []
    with pytest.raises(LedgerError, match="before its target session has closed"):
        OutcomeLog(layout).append({"forecast_id": rec["forecast_id"], "target_close_utc": rec["target_close_utc"]},
                                  now=SETTLED - pd.Timedelta(minutes=1))


def test_rescoring_adds_nothing_and_research_artifacts_are_untouched(live_layout):
    from filingshock.forecasts import ledger_path
    from filingshock.prospective_outcomes import OUTCOMES_NAME, OutcomeLog, score_prospective_forecasts
    cfg, layout = live_layout
    _recorded(live_layout)
    write_json_atomic(layout.experiments / "run-x" / "metrics.json", {"primary": {"estimate": 0.0}})
    _set_retrieved(layout, SETTLED)
    before = _files(layout.root)
    assert [r["status"] for r in score_prospective_forecasts(cfg, layout, clock=lambda: SCORE_AT)] == ["scored"]
    after = _files(layout.root)
    added = set(after) - set(before)
    assert added == {f"forecasts/{OUTCOMES_NAME}", f"forecasts/{OUTCOMES_NAME}.lock"}
    assert {k: v for k, v in after.items() if k not in added} == before       # ledger, research tables, metrics
    blob = (layout.forecasts / OUTCOMES_NAME).read_bytes()
    again = score_prospective_forecasts(cfg, layout, clock=lambda: SCORE_AT + pd.Timedelta(days=30))
    assert [r["status"] for r in again] == ["already_scored"]
    assert (layout.forecasts / OUTCOMES_NAME).read_bytes() == blob
    log_ = OutcomeLog(layout)
    with pytest.raises(LedgerError, match="already scored"):
        log_.append(log_.records()[0], now=SCORE_AT)
    assert ForecastLedger(layout).verify_chain() == (True, "ok") and ledger_path(layout).exists()


def test_invalid_target_bar_is_recorded_without_a_score(live_layout):
    from filingshock.prospective_outcomes import outcomes_frame, prospective_summary, score_prospective_forecasts
    cfg, layout = live_layout
    _recorded(live_layout)
    _set_target_move(layout, 0.8)                            # above max_abs_return_sanity: probable data error
    _set_retrieved(layout, SETTLED)
    (r,) = score_prospective_forecasts(cfg, layout, clock=lambda: SCORE_AT)
    assert r["status"] == "scored" and r["record"]["outcome_status"] == "target_bar_invalid"
    assert r["record"]["y"] is None and r["record"]["brier_C"] is None
    s = prospective_summary(outcomes_frame(layout))
    assert (s["n_invalid_target_bar"], s["n_scored_events"], s["pooled"]) == (1, 0, None)


def test_pooled_prospective_brier_needs_enough_events():
    from filingshock.prospective_outcomes import prospective_summary
    rows = [{"forecast_id": f"f{i}", "forecast_sequence": i, "event_id": f"ev-{i % 3}", "outcome_status": "ok",
             "y": y, "brier_B": (0.2 - y) ** 2, "brier_C": (pb - y) ** 2}
            for i, (y, pb) in enumerate([(1, 0.6), (0, 0.1), (0, 0.3), (1, 0.0)])]   # f3 repeats ev-0 later
    df = pd.DataFrame(rows)
    assert prospective_summary(df)["pooled"] is None and prospective_summary(df)["n_scored_events"] == 3
    p = prospective_summary(df, min_events=3)["pooled"]
    assert p["n_events"] == 3 and p["n_positive"] == 1
    assert p["brier_C"] == pytest.approx((0.16 + 0.01 + 0.09) / 3)              # earliest record per event
    assert p["delta_brier_C_minus_B"] == pytest.approx(p["brier_C"] - (0.64 + 0.04 + 0.04) / 3)


def test_optional_baselines_use_paired_earliest_forecasts_only():
    from filingshock.prospective_outcomes import prospective_summary

    rows = pd.DataFrame([
        {"event_id": "old", "forecast_sequence": 0, "outcome_status": "ok", "y": 1,
         "brier_B": 0.1, "brier_C": 0.2},
        {"event_id": "new", "forecast_sequence": 1, "outcome_status": "ok", "y": 0,
         "brier_B": 0.2, "brier_C": 0.3, "brier_A": 0.4, "brier_constant": 0.5},
        {"event_id": "old", "forecast_sequence": 2, "outcome_status": "ok", "y": 1,
         "brier_B": 0.1, "brier_C": 0.0, "brier_A": 0.9, "brier_constant": 0.9},
    ])
    summary = prospective_summary(rows, min_events=1)
    assert summary["n_scored_events"] == 2
    paired = summary["baseline_comparisons"]["A"]
    assert paired["n_paired_events"] == 1
    assert paired["pooled"]["brier_C"] == pytest.approx(0.3)
    assert paired["pooled"]["delta_brier_C_minus_baseline"] == pytest.approx(-0.1)


def test_production_baselines_ignore_future_year_labels(live_layout):
    from filingshock.modeling.production import fit_production_model

    cfg, layout = live_layout
    _synthetic_dataset(layout, years=range(2016, 2025))
    first = fit_production_model(cfg, layout, 2024)
    outcomes = pd.read_parquet(layout.processed / "outcomes.parquet")
    future = outcomes["event_id"].str.startswith("ev-syn-2024-")
    outcomes.loc[future, "y"] = 1 - outcomes.loc[future, "y"]
    write_parquet_atomic(layout.processed / "outcomes.parquet", outcomes)
    second = fit_production_model(cfg, layout, 2024)
    features = pd.read_parquet(layout.processed / "features.parquet").head(1)
    assert first.predict(features) == second.predict(features)
    assert first.baseline_probability == pytest.approx(outcomes.loc[~future, "y"].mean())


def test_score_prospective_cli(live_layout, monkeypatch, capsys):
    from filingshock.cli import main
    from filingshock.prospective_outcomes import OutcomeLog
    _, layout = live_layout
    _recorded(live_layout)
    _set_retrieved(layout, SETTLED)
    monkeypatch.setenv("FILINGSHOCK_DATA_DIR", str(layout.root))
    assert main(["score-prospective"]) == 0
    assert len(OutcomeLog(layout).records()) == 1
    assert main(["score-prospective"]) == 0 and len(OutcomeLog(layout).records()) == 1
    out = capsys.readouterr().out
    assert "already_scored" in out and "pooled prospective Brier withheld: 1 scored event(s)" in out


def test_history_end_today_never_stores_an_unsettled_bar(tmp_path, cal):
    from filingshock.config import Config
    from filingshock.sources.market import ingest_prices, resolve_history_end
    from filingshock.storage import DataLayout
    from tests.conftest import make_bars
    default = Config.load()["market_data"]["history_end"]
    assert str(default).lower() != "today" and resolve_history_end(default) == str(pd.Timestamp(default).date())
    assert resolve_history_end("today", now=datetime(2024, 10, 30, 23, 0, tzinfo=timezone.utc)) == "2024-10-31"
    sessions = cal.sessions[(cal.sessions >= "2024-09-03") & (cal.sessions <= "2024-10-31")]
    csv_dir = tmp_path / "import"
    csv_dir.mkdir()
    make_bars(sessions).rename(columns={"session": "date"}).to_csv(csv_dir / "TEST.csv", index=False)
    cfg = Config.load().with_overrides({"market_data.provider": "csv", "market_data.csv_dir": str(csv_dir)})
    layout = DataLayout(tmp_path / "data").ensure()
    store = PriceStore(layout, "csv")
    for now, last in [(datetime(2024, 10, 30, 17, 0, tzinfo=timezone.utc), "2024-10-29"),    # session in progress
                      (datetime(2024, 10, 30, 20, 30, tzinfo=timezone.utc), "2024-10-29"),   # closed, not settled
                      (datetime(2024, 10, 30, 21, 0, tzinfo=timezone.utc), "2024-10-30")]:   # settled
        ingest_prices(cfg, layout, ["TEST"], refresh=True, history_end="today", now=now)
        assert str(store.load("TEST")["session"].max().date()) == last, now
    entry = store.manifest_entries()["TEST"]
    assert entry["requested_end"] == "2024-10-31" and entry["unsettled_sessions_dropped"] == 0
    ingest_prices(cfg, layout, ["TEST"], refresh=True, now=datetime(2024, 10, 30, 17, 0, tzinfo=timezone.utc))
    assert store.manifest_entries()["TEST"]["requested_end"] == str(pd.Timestamp(default).date())
    assert str(store.load("TEST")["session"].max().date()) == "2024-10-29"
