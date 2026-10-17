"""Append-only, hash-chained ledger of prospective forecasts, and the scoring path that fills it.

A prospective forecast is valid only if it was generated before the event's decision cutoff, using
a filing the system actually observed (observed_at_utc) before that cutoff. Records are never edited
or overwritten; a newer model produces a new record. Each record stores the SHA-256 of the previous
record so silent edits are detectable (verify_chain).

Prospective records are kept apart from the research evidence: they never enter an experiment's
predictions or metrics (the research sample ends with the protocol's filing window). Their outcomes are
scored separately, after the target session closes (prospective_outcomes.py).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from filingshock import __version__
from filingshock.config import Config
from filingshock.storage import DataLayout

log = logging.getLogger(__name__)
LEDGER_NAME = "prospective_forecasts.jsonl"
CATEGORY = "prospective_recorded_before_session"


class LedgerError(RuntimeError):
    pass


class HashChainedLog:
    """Append-only JSONL file in which every record stores the SHA-256 of the previous one."""

    def __init__(self, path: Path):
        self.path = path

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise LedgerError(f"{self.path.name}: invalid UTF-8") from exc
        if text and not text.endswith("\n"):
            raise LedgerError(f"{self.path.name}: incomplete final record (missing newline)")
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"{self.path.name}: invalid JSON on line {line_number}") from exc
            if not isinstance(rec, dict):
                raise LedgerError(f"{self.path.name}: line {line_number} is not a record object")
            records.append(rec)
        return records

    @staticmethod
    def _hash(rec: dict) -> str:
        body = {k: v for k, v in rec.items() if k != "record_sha256"}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def verify_chain(self) -> tuple[bool, str]:
        try:
            records = self.records()
        except LedgerError as exc:
            return False, str(exc)
        return self._verify_records(records)

    @classmethod
    def _verify_records(cls, records: list[dict]) -> tuple[bool, str]:
        prev = None
        for i, rec in enumerate(records):
            if type(rec.get("sequence")) is not int or rec["sequence"] != i:
                return False, f"record {i} has an invalid sequence"
            if rec.get("prev_record_sha256") != prev:
                return False, f"record {i} breaks the chain"
            if cls._hash(rec) != rec.get("record_sha256"):
                return False, f"record {i} content does not match its hash"
            prev = rec["record_sha256"]
        return True, "ok"

    def verified_records(self) -> list[dict]:
        """Read only records whose hashes, ordering and JSON framing are valid."""
        records = self.records()
        ok, msg = self._verify_records(records)
        if not ok:
            raise LedgerError(f"ledger integrity check failed: {msg}")
        return records

    def _append_chained(self, rec: dict) -> dict:
        existing = self.records()
        rec = dict(rec)
        rec["sequence"] = len(existing)
        rec["prev_record_sha256"] = existing[-1]["record_sha256"] if existing else None
        rec["record_sha256"] = self._hash(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:      # append-only
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec


class ForecastLedger(HashChainedLog):
    def __init__(self, layout: DataLayout):
        super().__init__(layout.forecasts / LEDGER_NAME)

    def append(self, rec: dict, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        cutoff = pd.Timestamp(rec["decision_cutoff_utc"])
        observed = pd.Timestamp(rec["observed_at_utc"])
        if pd.Timestamp(now) >= cutoff:
            raise LedgerError("cannot record a prospective forecast at or after the decision cutoff")
        if observed > cutoff:
            raise LedgerError("filing was observed after the decision cutoff")
        if observed > pd.Timestamp(now):
            raise LedgerError("filing observation time is later than the generation time")
        ok, msg = self.verify_chain()
        if not ok:
            raise LedgerError(f"ledger integrity check failed: {msg}")
        rec = dict(rec)
        rec["prediction_category"] = CATEGORY
        rec["generated_at_utc"] = pd.Timestamp(now).isoformat()
        return self._append_chained(rec)


def find_pending_filings(cfg: Config, layout: DataLayout, now: datetime | None = None,
                         filings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Original 10-Qs whose decision cutoff is still in the future (requires a fresh discovery run)."""
    from filingshock.discovery import load_filings
    from filingshock.timing import SessionCalendar, TimingPolicy, classify_timing

    now = pd.Timestamp(now or datetime.now(timezone.utc))
    f = load_filings(layout) if filings is None else filings
    q = f[(f["form_type"] == "10-Q") & (f["acceptance_utc"] >= now - pd.Timedelta(days=5)) & (f["acceptance_utc"] <= now)]
    if "first_seen_utc" in q:                     # a filing cannot be pending before discovery has seen it
        q = q[pd.to_datetime(q["first_seen_utc"], utc=True) <= now]
    cal, pol = SessionCalendar(cfg["timing"]["exchange_calendar"]), TimingPolicy.from_config(cfg)
    rows = []
    for _, r in q.iterrows():
        t = classify_timing(r["acceptance_utc"], r["filing_date"], cal, pol)
        if t.status == "eligible" and t.decision_cutoff_utc > now:
            rows.append({**r.to_dict(), "decision_cutoff_utc": t.decision_cutoff_utc,
                         "target_session": t.target_session})
    return pd.DataFrame(rows)


def _jsonable(v):
    """JSON-safe copy for ledger records: timestamps as ISO strings, NaN as null, numpy as Python."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return pd.Timestamp(v).isoformat()
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    return v


def _day(ts) -> str | None:
    return str(pd.Timestamp(ts).date()) if ts is not None and pd.notna(ts) else None


def forecast_record(pending: pd.Series, sf, pm, probs: dict, cfg: Config) -> dict:
    """Ledger record for one scored event (before append adds category, generation time and the chain)."""
    from filingshock.modeling.production import PRIMARY_FAMILY

    ev, th, tgt = sf.event, sf.threshold, cfg["target"]
    rec = {
        "forecast_id": "pf-" + hashlib.sha256(f"{ev['event_id']}|{pm.version}|{pm.parameters_sha256}".encode()).hexdigest()[:16],
        "event_id": ev["event_id"], "ticker": ev["ticker"], "cik": ev["cik"], "company_name": ev.get("company_name"),
        "accession_number": sf.accession_number, "form_type": ev["form_type"], "report_date": ev.get("report_date"),
        "filing_date": ev.get("filing_date"), "acceptance_utc": ev["acceptance_utc"],
        "acceptance_source": pending.get("acceptance_source"),
        "observed_at_utc": pending["first_seen_utc"], "observed_at_source": "discovery_first_seen",
        "assumed_available_utc": ev["assumed_available_utc"], "availability_method": ev["availability_method"],
        "timing_flags": ev.get("timing_flags"), "prev_session": _day(ev["prev_session"]),
        "target_session": _day(ev["target_session"]), "target_open_utc": ev["target_open_utc"],
        "target_close_utc": ev["target_close_utc"], "decision_cutoff_utc": ev["decision_cutoff_utc"],
        "model_version": pm.version, "model_family": PRIMARY_FAMILY, "model_parameters_sha256": pm.parameters_sha256,
        "training_data_through_year": pm.year - 2, "calibration_year": pm.year - 1,
        "dataset_id": pm.dataset_id, "feature_version": pm.feature_version, "code_version": __version__,
        "target_definition": f"|close/open - 1| in the target session > {float(tgt['multiplier']):g} x mean "
                             f"|open-to-close| of the previous {int(tgt['lookback_sessions'])} valid sessions",
        "threshold_abs_move": th.threshold, "m_mean_abs_oc_60": th.m, "threshold_n_sessions": th.n_valid,
        "threshold_window_first": _day(th.window_first_session), "threshold_window_last": _day(th.window_last_session),
        "probability": probs["C"]["p_cal"],
        "p_B": probs["B"]["p_cal"], "p_C": probs["C"]["p_cal"],
        "p_B_raw": probs["B"]["p_raw"], "p_C_raw": probs["C"]["p_raw"],
        "calibration_method_B": probs["B"]["calibration_method"], "calibration_method_C": probs["C"]["calibration_method"],
        "reference_accession": sf.reference_accession,
        "features": sf.features, "feature_provenance": sf.provenance,
    }
    return _jsonable(rec)


def record_pending_forecasts(cfg: Config, layout: DataLayout, client,
                             clock: Callable[[], datetime] | None = None) -> list[dict]:
    """Score every pending 10-Q with the production models B and C and append one ledger record per
    (event, model version), only while the decision cutoff is still in the future.

    Returns one row per pending filing: status recorded | already_recorded | skipped, with the reason.
    clock: source of the current time (tests inject a fixed clock); it is read again right before
    each append, so a forecast that finishes after its cutoff is not recorded.
    """
    from filingshock.discovery import load_filings
    from filingshock.events import single_event_features
    from filingshock.modeling.experiment import model_specs
    from filingshock.modeling.production import (
        ProductionModelNotReady,
        fit_production_model,
        save_production_manifest,
    )

    clock = clock or (lambda: datetime.now(timezone.utc))
    ledger = ForecastLedger(layout)
    ok, msg = ledger.verify_chain()
    if not ok:
        raise LedgerError(f"ledger integrity check failed: {msg}")
    filings = load_filings(layout)
    pending = find_pending_filings(cfg, layout, now=clock(), filings=filings)
    done = {(r["event_id"], r["model_version"]) for r in ledger.records()}
    needed = set(model_specs()["C"])
    models: dict[int, object] = {}
    results = []
    for _, p in pending.iterrows():
        res = {"ticker": p["ticker"], "accession_number": p["accession_number"],
               "event_id": "ev-" + p["accession_number"], "decision_cutoff_utc": p["decision_cutoff_utc"],
               "status": "skipped", "reason": None, "record": None}
        results.append(res)
        cutoff = pd.Timestamp(p["decision_cutoff_utc"])
        observed = p.get("first_seen_utc")
        if observed is None or pd.isna(observed):
            res["reason"] = "observed_time_unknown: the filings table has no first_seen_utc; run `filingshock discover`"
            continue
        if pd.Timestamp(observed) > cutoff:
            res["reason"] = "observed_after_cutoff"
            continue
        year = pd.Timestamp(p["target_session"]).year
        if year not in models:
            try:
                models[year] = fit_production_model(cfg, layout, year)
                save_production_manifest(layout, models[year])
            except ProductionModelNotReady as exc:
                models[year] = exc
        pm = models[year]
        if isinstance(pm, Exception):
            res["reason"] = str(pm)
            continue
        if (res["event_id"], pm.version) in done:
            res["status"], res["reason"] = "already_recorded", f"a record for {pm.version} exists (never overwritten)"
            continue
        if pm.max_outcome_available_utc > cutoff:        # cannot happen with annual blocks; guarded anyway
            res["reason"] = "training_outcome_not_matured_at_cutoff"
            continue
        sf = single_event_features(cfg, layout, p["accession_number"], client, filings=filings)
        if sf.status != "ok":
            res["reason"] = f"{sf.stage}: {sf.reason}"
            continue
        if not needed <= set(sf.features):
            raise RuntimeError(f"single-event features missing {sorted(needed - set(sf.features))}")
        probs = pm.predict(pd.DataFrame([sf.features]))
        try:
            res["record"] = ledger.append(forecast_record(p, sf, pm, probs, cfg), now=clock())
        except LedgerError as exc:
            res["reason"] = str(exc)
            continue
        res["status"], res["reason"] = "recorded", None
        done.add((res["event_id"], pm.version))
        log.info("recorded prospective forecast %s for %s (P_C=%.3f, cutoff %s)", res["record"]["forecast_id"],
                 p["accession_number"], probs["C"]["p_cal"], cutoff)
    return results


LEDGER_COLUMNS = ["forecast_id", "event_id", "ticker", "company_name", "accession_number", "form_type", "report_date",
                  "acceptance_utc", "observed_at_utc", "decision_cutoff_utc", "target_session", "target_close_utc",
                  "generated_at_utc", "model_version", "dataset_id", "threshold_abs_move", "p_B", "p_C", "probability",
                  "reference_accession", "prediction_category", "sequence", "record_sha256"]


def ledger_frame(layout: DataLayout) -> pd.DataFrame:
    recs = ForecastLedger(layout).verified_records()
    return pd.DataFrame(recs) if recs else pd.DataFrame(columns=LEDGER_COLUMNS)


def ledger_path(layout: DataLayout) -> Path:
    return layout.forecasts / LEDGER_NAME
