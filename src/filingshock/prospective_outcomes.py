"""Outcome scoring for prospective forecasts, kept apart from the research evidence.

Once a forecast's target session has closed and the price store holds a bar for that session retrieved
at least BAR_SETTLE_MINUTES after the close, the realized open-to-close move is compared with the
threshold STORED in the forecast record. The threshold is never recomputed: the forecast fixed its own
target before the session. Results go to a separate append-only, hash-chained file
(forecasts/prospective_outcomes.jsonl, one record per forecast_id, never overwritten); the forecast
ledger is only read. Nothing here reads or writes the research tables, experiment predictions or metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from filingshock import __version__
from filingshock.config import Config
from filingshock.events import _bars
from filingshock.forecasts import ForecastLedger, HashChainedLog, LedgerError, _jsonable
from filingshock.sources.market import BAR_SETTLE_MINUTES, PriceStore
from filingshock.storage import DataLayout
from filingshock.target import Outcome, Threshold, realized_outcome

log = logging.getLogger(__name__)
OUTCOMES_NAME = "prospective_outcomes.jsonl"
OUTCOME_CATEGORY = "prospective_outcome_after_close"
POOLED_MIN_EVENTS = 20          # pooled Brier scores are reported only from this many scored events
SETTLE = pd.Timedelta(minutes=BAR_SETTLE_MINUTES)
OUTCOME_COLUMNS = ["forecast_id", "forecast_record_sha256", "forecast_sequence", "event_id", "ticker",
                   "target_session", "target_close_utc", "model_version", "threshold_abs_move", "outcome_status",
                   "open", "close", "r_open_to_close", "abs_r", "y", "p_B", "p_C", "brier_B", "brier_C",
                   "price_retrieved_at_utc", "scored_at_utc", "prediction_category", "sequence", "record_sha256"]


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class OutcomeLog(HashChainedLog):
    def __init__(self, layout: DataLayout):
        super().__init__(layout.forecasts / OUTCOMES_NAME)

    def append(self, rec: dict, now: datetime | None = None) -> dict:
        now = _utc(now or datetime.now(timezone.utc))
        if now < _utc(rec["target_close_utc"]) + SETTLE:
            raise LedgerError("cannot score a prospective forecast before its target session has closed and settled")
        ok, msg = self.verify_chain()
        if not ok:
            raise LedgerError(f"outcome log integrity check failed: {msg}")
        if any(r["forecast_id"] == rec["forecast_id"] for r in self.records()):
            raise LedgerError(f"forecast {rec['forecast_id']} is already scored (outcomes are never overwritten)")
        rec = dict(rec)
        rec["prediction_category"] = OUTCOME_CATEGORY
        rec["scored_at_utc"] = now.isoformat()
        return self._append_chained(rec)


def stored_threshold(forecast: dict) -> Threshold:
    """The forecast's own threshold, exactly as recorded before the session."""
    return Threshold("ok", m=forecast.get("m_mean_abs_oc_60"), threshold=float(forecast["threshold_abs_move"]),
                     n_valid=int(forecast.get("threshold_n_sessions") or 0))


def outcome_record(forecast: dict, oc: Outcome, price_entry: dict, provider: str) -> dict:
    """Outcome record for one forecast (before append adds category, scoring time and the chain)."""
    y = oc.y if oc.status == "ok" else None
    rec = {
        "forecast_id": forecast["forecast_id"], "forecast_record_sha256": forecast["record_sha256"],
        "forecast_sequence": forecast["sequence"], "event_id": forecast["event_id"], "ticker": forecast["ticker"],
        "accession_number": forecast.get("accession_number"), "model_version": forecast["model_version"],
        "dataset_id": forecast.get("dataset_id"), "generated_at_utc": forecast.get("generated_at_utc"),
        "decision_cutoff_utc": forecast.get("decision_cutoff_utc"), "target_session": forecast["target_session"],
        "target_close_utc": forecast["target_close_utc"],
        "threshold_abs_move": forecast["threshold_abs_move"], "threshold_source": "forecast_record",
        "outcome_status": oc.status, "open": oc.open, "close": oc.close, "r_open_to_close": oc.r, "abs_r": oc.a, "y": y,
        "p_B": forecast.get("p_B"), "p_C": forecast.get("p_C"),
        "brier_B": (forecast["p_B"] - y) ** 2 if y is not None and forecast.get("p_B") is not None else None,
        "brier_C": (forecast["p_C"] - y) ** 2 if y is not None and forecast.get("p_C") is not None else None,
        "price_provider": provider, "price_symbol": forecast["ticker"],
        "price_retrieved_at_utc": price_entry.get("retrieved_at"), "price_file_sha256": price_entry.get("sha256"),
        "code_version": __version__,
    }
    return _jsonable(rec)


def score_prospective_forecasts(cfg: Config, layout: DataLayout,
                                clock: Callable[[], datetime] | None = None) -> list[dict]:
    """Score every ledger record whose target session has closed and whose target bar is in the price store.

    Returns one row per forecast record: status scored | already_scored | pending, with the reason a
    pending record cannot be scored yet. A record is scored only when
      - the clock is past target_close_utc + BAR_SETTLE_MINUTES,
      - the symbol's price file was retrieved after that time (and not after the clock), and
      - the store has a bar for the target session.
    An invalid target bar (bar_quality) is recorded as outcome_status target_bar_invalid with y = null,
    like the research funnel's exclusion, so it cannot be re-fetched until a favourable bar appears.
    """
    clock = clock or (lambda: datetime.now(timezone.utc))
    ledger, outcomes = ForecastLedger(layout), OutcomeLog(layout)
    for name, chain in (("forecast ledger", ledger), ("outcome log", outcomes)):
        ok, msg = chain.verify_chain()
        if not ok:
            raise LedgerError(f"{name} integrity check failed: {msg}")
    scored = {r["forecast_id"] for r in outcomes.records()}
    provider = cfg["market_data"]["provider"]
    store = PriceStore(layout, provider)
    manifest = store.manifest_entries()
    max_abs = float(cfg["target"]["max_abs_return_sanity"])
    bars_by_symbol: dict[str, pd.DataFrame | None] = {}
    results = []
    for f in ledger.records():
        res = {"forecast_id": f["forecast_id"], "ticker": f["ticker"], "event_id": f["event_id"],
               "target_session": f["target_session"], "target_close_utc": f["target_close_utc"],
               "status": "pending", "reason": None, "record": None}
        results.append(res)
        if f["forecast_id"] in scored:
            res["status"], res["reason"] = "already_scored", "outcomes are never overwritten"
            continue
        settled_at = _utc(f["target_close_utc"]) + SETTLE
        now = _utc(clock())
        if now < settled_at:
            res["reason"] = "target_session_not_closed"
            continue
        if f.get("threshold_abs_move") is None:
            res["reason"] = "forecast_record_has_no_threshold"
            continue
        sym = f["ticker"]
        if sym not in bars_by_symbol:
            bars_by_symbol[sym] = _bars(store, sym, max_abs)
        bars, entry = bars_by_symbol[sym], manifest.get(sym) or {}
        if bars is None:
            res["reason"] = "price_history_unavailable"
            continue
        if not entry.get("retrieved_at"):
            res["reason"] = "price_retrieval_time_unknown: refresh prices"
            continue
        retrieved = _utc(entry["retrieved_at"])
        if retrieved < settled_at:
            res["reason"] = ("prices_not_refreshed_after_close: run `filingshock prices --refresh --history-end today` "
                             f"after {settled_at.isoformat()}")
            continue
        if retrieved > now:
            res["reason"] = "price_file_newer_than_clock"          # only with a simulated clock
            continue
        target = bars[bars["session"] == pd.Timestamp(f["target_session"])]
        if target.empty:
            res["reason"] = "target_bar_not_in_price_store"
            continue
        oc = realized_outcome(target.iloc[0], stored_threshold(f))
        try:
            res["record"] = outcomes.append(outcome_record(f, oc, entry, provider), now=clock())
        except LedgerError as exc:
            res["reason"] = str(exc)
            continue
        res["status"] = "scored"
        scored.add(f["forecast_id"])
        log.info("scored prospective forecast %s (%s %s): |r|=%s threshold=%.4f y=%s", f["forecast_id"], sym,
                 f["target_session"], oc.a, f["threshold_abs_move"], oc.y)
    return results


def outcomes_frame(layout: DataLayout) -> pd.DataFrame:
    recs = OutcomeLog(layout).records()
    return pd.DataFrame(recs) if recs else pd.DataFrame(columns=OUTCOME_COLUMNS)


def prospective_summary(outcomes: pd.DataFrame, min_events: int = POOLED_MIN_EVENTS) -> dict:
    """Counts and, from min_events scored events on, pooled Brier scores of the prospective outcomes.

    Pooled over the earliest-recorded forecast per event (a later model version for the same event never
    replaces it), and only over valid outcomes. Never mixed with the research sample.
    """
    valid = outcomes[(outcomes["outcome_status"] == "ok") & outcomes["y"].notna()] if len(outcomes) else outcomes
    first = valid.sort_values("forecast_sequence").drop_duplicates("event_id", keep="first") if len(valid) else valid
    out = {"n_outcome_records": int(len(outcomes)),
           "n_invalid_target_bar": int((outcomes["outcome_status"] == "target_bar_invalid").sum()) if len(outcomes) else 0,
           "n_scored_events": int(len(first)), "min_events_for_pooled": min_events, "pooled": None}
    if len(first) >= min_events:
        y = first["y"].astype(float)
        bB, bC = first["brier_B"].astype(float).mean(), first["brier_C"].astype(float).mean()
        out["pooled"] = {"n_events": int(len(first)), "n_positive": int(y.sum()), "prevalence": float(y.mean()),
                         "brier_B": float(bB), "brier_C": float(bC), "delta_brier_C_minus_B": float(bC - bB)}
    return out
