"""Read-only access to saved artifacts for the interface. Nothing here trains models or downloads data.

Sources of truth:
    processed/events.parquet, features.parquet, outcomes.parquet, funnel.json, dataset_manifest.json
    experiments/<run>/predictions.parquet, metrics.json, manifest.json   (primary run via primary_run.json)
    interim/sections/<accession>.json, interim/alignments/<event_id>.parquet
    forecasts/prospective_forecasts.jsonl   (prospective ledger; shown apart from research metrics)
    forecasts/prospective_outcomes.jsonl    (outcomes of matured prospective forecasts; same separation)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from filingshock.config import PROJECT_ROOT, Config
from filingshock.storage import DataLayout

SAMPLE_DIR = PROJECT_ROOT / "sample"


def mode() -> str:
    return os.environ.get("FILINGSHOCK_MODE", "research")


def layout() -> DataLayout:
    if mode() == "sample":
        return DataLayout(SAMPLE_DIR)
    return DataLayout(Config.load().data_dir)


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0


@st.cache_data(show_spinner=False)
def _parquet(path: str, mtime: float) -> pd.DataFrame | None:
    return pd.read_parquet(path) if mtime >= 0 else None


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float):
    return json.loads(Path(path).read_text()) if mtime >= 0 else None


@st.cache_data(show_spinner=False)
def _ledger(root: str, mtime: float) -> tuple[pd.DataFrame, bool, str]:
    from filingshock.forecasts import ForecastLedger, ledger_frame
    lay = DataLayout(Path(root))
    ok, msg = ForecastLedger(lay).verify_chain()
    return ledger_frame(lay), ok, msg


@st.cache_data(show_spinner=False)
def _prospective_outcomes(root: str, mtime: float) -> tuple[pd.DataFrame, bool, str]:
    from filingshock.prospective_outcomes import OutcomeLog, outcomes_frame
    lay = DataLayout(Path(root))
    ok, msg = OutcomeLog(lay).verify_chain()
    return outcomes_frame(lay), ok, msg


def parquet(p: Path) -> pd.DataFrame | None:
    return _parquet(str(p), _mtime(p))


def jsonfile(p: Path):
    return _json(str(p), _mtime(p))


@dataclass
class AppData:
    mode: str
    root: Path
    events: pd.DataFrame | None
    features: pd.DataFrame | None
    outcomes: pd.DataFrame | None
    funnel: dict | None
    dataset: dict | None
    run_id: str | None
    run_manifest: dict | None
    metrics: dict | None
    predictions: pd.DataFrame | None
    package: dict | None
    prospective: pd.DataFrame | None = None          # prospective ledger records (never research evidence)
    ledger_integrity: tuple[bool, str] = (True, "ok")
    prospective_outcomes: pd.DataFrame | None = None  # outcomes of matured prospective forecasts (same separation)
    outcomes_integrity: tuple[bool, str] = (True, "ok")

    @property
    def has_dataset(self) -> bool:
        return self.events is not None and not self.events.empty

    @property
    def has_outcomes(self) -> bool:
        return self.outcomes is not None and not self.outcomes.empty

    @property
    def run_is_stale(self) -> bool:
        if not self.run_manifest or not self.dataset:
            return False
        return self.run_manifest["dataset"]["dataset_id"] != self.dataset["dataset_id"]


def load() -> AppData:
    lay = layout()
    proc = lay.processed
    run_ptr = jsonfile(lay.experiments / "primary_run.json")
    run_id = run_ptr["run_id"] if run_ptr else None
    run_dir = lay.experiments / run_id if run_id else None
    complete = bool(run_dir and (run_dir / "COMPLETE").exists())
    from filingshock.forecasts import ledger_path
    from filingshock.prospective_outcomes import OUTCOMES_NAME
    led, led_ok, led_msg = _ledger(str(lay.root), _mtime(ledger_path(lay)))
    pout, pout_ok, pout_msg = _prospective_outcomes(str(lay.root), _mtime(lay.forecasts / OUTCOMES_NAME))
    return AppData(
        mode=mode(), root=lay.root,
        events=parquet(proc / "events.parquet"), features=parquet(proc / "features.parquet"),
        outcomes=parquet(proc / "outcomes.parquet"), funnel=jsonfile(proc / "funnel.json"),
        dataset=jsonfile(proc / "dataset_manifest.json"),
        run_id=run_id if complete else None,
        run_manifest=jsonfile(run_dir / "manifest.json") if complete else None,
        metrics=jsonfile(run_dir / "metrics.json") if complete else None,
        predictions=parquet(run_dir / "predictions.parquet") if complete else None,
        package=jsonfile(lay.root / "package_manifest.json"),
        prospective=led, ledger_integrity=(led_ok, led_msg),
        prospective_outcomes=pout, outcomes_integrity=(pout_ok, pout_msg),
    )


def section_doc(accession: str) -> dict | None:
    return jsonfile(layout().sections_dir / f"{accession}.json")


def alignment(event_id: str) -> pd.DataFrame | None:
    return parquet(layout().alignments_dir / f"{event_id}.parquet")


def extractions() -> pd.DataFrame | None:
    return parquet(layout().interim / "extractions.parquet")


def other_runs() -> list[dict]:
    lay = layout()
    out = []
    if not lay.experiments.exists():
        return out
    for d in sorted(lay.experiments.iterdir()):
        if d.is_dir() and (d / "COMPLETE").exists():
            m = jsonfile(d / "metrics.json")
            man = jsonfile(d / "manifest.json")
            out.append({"run_id": d.name, "run_name": man.get("run_name"), "note": man.get("note"),
                        "subset": man.get("subset"), "outcome_definition": man.get("outcome_definition"),
                        "multiplier_override": d.name.split("multiplier-")[-1] if "multiplier-" in d.name else None,
                        "dataset_id": man["dataset"]["dataset_id"], "metrics": m})
    return out


def event_table(data: AppData, include_prospective: bool = False) -> pd.DataFrame:
    """Explorer table joining events, outcomes and primary-run predictions (no business logic beyond joins).

    include_prospective appends one row per prospectively forecast filing (latest ledger record) with
    eligibility_status "prospective": these rows are outside the research sample and its metrics.
    """
    ev = data.events.copy()
    if data.outcomes is not None and not data.outcomes.empty:
        ev = ev.merge(data.outcomes[["event_id", "open", "close", "r_open_to_close", "abs_r", "threshold_abs_move", "y"]],
                      on="event_id", how="left")
    else:
        for c in ["open", "close", "r_open_to_close", "abs_r", "threshold_abs_move", "y"]:
            ev[c] = float("nan")          # not included (e.g. sample package without provider prices)
    if data.predictions is not None:
        p = data.predictions[data.predictions["family"] == "logistic_l2"]
        wide = p.pivot_table(index="event_id", columns="model", values="p_cal")
        for m in ("B", "C"):
            ev[f"p_{m}"] = ev["event_id"].map(wide[m]) if m in wide else pd.NA
        cat = p.drop_duplicates("event_id").set_index("event_id")["prediction_category"]
        ev["prediction_category"] = ev["event_id"].map(cat)
        yr = p.drop_duplicates("event_id").set_index("event_id")["test_year"]
        ev["test_year"] = ev["event_id"].map(yr)
    else:
        ev["p_B"] = ev["p_C"] = float("nan")
        ev["prediction_category"] = pd.NA
        ev["test_year"] = pd.NA
    ev["prediction_status"] = ev.apply(prediction_status, axis=1)
    if include_prospective and data.prospective is not None and not data.prospective.empty:
        ev = pd.concat([ev, prospective_rows(data.prospective, set(ev["event_id"]))], ignore_index=True)
    return ev


def prospective_rows(ledger: pd.DataFrame, known_ids: set) -> pd.DataFrame:
    """Explorer rows for prospective forecasts (latest record per event). Events already in the
    historical table are skipped; under the protocol's filing window this cannot happen."""
    lat = ledger.sort_values("sequence").drop_duplicates("event_id", keep="last")
    lat = lat[~lat["event_id"].isin(known_ids)].reset_index(drop=True)

    def quality(prov):
        return ((prov or {}).get("alignment") or {}).get("mdna_align_alignment_quality") if isinstance(prov, dict) else None

    return pd.DataFrame({
        "event_id": lat["event_id"], "ticker": lat["ticker"], "company_name": lat["company_name"],
        "cik": lat["cik"], "accession_number": lat["accession_number"], "form_type": lat["form_type"],
        "report_date": lat["report_date"], "filing_date": lat.get("filing_date"),
        "acceptance_utc": pd.to_datetime(lat["acceptance_utc"], utc=True),
        "observed_at_utc": pd.to_datetime(lat["observed_at_utc"], utc=True),
        "decision_cutoff_utc": pd.to_datetime(lat["decision_cutoff_utc"], utc=True),
        "target_session": pd.to_datetime(lat["target_session"]),
        "target_close_utc": pd.to_datetime(lat["target_close_utc"], utc=True),
        "generated_at_utc": pd.to_datetime(lat["generated_at_utc"], utc=True),
        "availability_method": "observed_by_discovery", "eligibility_status": "prospective",
        "exclusion_stage": None, "exclusion_reason": None, "fiscal_period_label": None,
        "reference_filing_id": lat["reference_accession"], "model_version": lat["model_version"],
        "forecast_dataset_id": lat["dataset_id"], "forecast_id": lat["forecast_id"],
        "threshold_abs_move": lat["threshold_abs_move"], "p_B": lat["p_B"], "p_C": lat["p_C"],
        "y": float("nan"), "abs_r": float("nan"), "r_open_to_close": float("nan"),
        "prediction_category": CATEGORY_PROSPECTIVE, "prediction_status": CATEGORY_PROSPECTIVE,
        "mdna_align_alignment_quality": lat["feature_provenance"].map(quality) if "feature_provenance" in lat else None,
        "data_quality_flags": "[]",
    })


def prediction_status(r) -> str:
    """Machine-readable prediction state for one event (see docs/state_matrix.md)."""
    if r.get("eligibility_status") != "eligible":
        return "not_supported_excluded"
    if pd.notna(r.get("p_C")):
        return "retrospective_out_of_time"
    ts = pd.Timestamp(r["target_session"]) if pd.notna(r.get("target_session")) else None
    if ts is not None and ts.year < 2014:
        return "no_prediction_training_period"
    return "no_prediction_fold_not_ready"


CATEGORY_PROSPECTIVE = "prospective_recorded_before_session"

PREDICTION_STATUS_LABEL = {
    "retrospective_out_of_time": "Retrospective out-of-time",
    "no_prediction_training_period": "No prediction (training period)",
    "no_prediction_fold_not_ready": "No prediction (fold not ready)",
    "not_supported_excluded": "Not supported (excluded)",
    "prospective_recorded_before_session": "Prospective (recorded before session)",
    "synthetic_ui_example": "Synthetic UI example",
}
