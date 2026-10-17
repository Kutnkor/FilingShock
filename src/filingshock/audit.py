"""Auditing: per-event admissibility explanations, timestamp reconciliation, artifact reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from filingshock.events import FUNNEL_STAGES, feature_groups
from filingshock.modeling.evaluate import metric_block
from filingshock.sources.sec import SecClient, parse_index_header_acceptance
from filingshock.storage import DataLayout, read_json, write_json_atomic

FEATURE_AVAILABILITY = {
    "market": "exchange sessions strictly before the target session (provider daily bars)",
    "context": "8-K Item 2.02 filings with acceptance + lag <= decision cutoff; filing lag from period end",
    "financial": "Company Facts values reported by periodic filings with acceptance + lag <= decision cutoff",
    "text": "current filing and its prior-year same-quarter reference, both available by the cutoff",
}


def explain_event(layout: DataLayout, event_id: str) -> dict:
    """Structured explanation of why an event is (in)admissible, stage by stage."""
    ev = pd.read_parquet(layout.processed / "events.parquet").set_index("event_id")
    if event_id not in ev.index:
        return {"event_id": event_id, "found": False}
    r = ev.loc[event_id]
    stages = []
    failed_at = r["exclusion_stage"] if pd.notna(r["exclusion_stage"]) else None
    reached = True
    for stage, desc in FUNNEL_STAGES[1:]:
        if not reached:
            stages.append({"stage": stage, "result": "not_evaluated", "description": desc})
            continue
        if stage == failed_at:
            stages.append({"stage": stage, "result": "failed", "reason": r["exclusion_reason"], "description": desc})
            reached = False
        else:
            stages.append({"stage": stage, "result": "passed", "description": desc})

    def ts(x):
        return None if pd.isna(x) else str(x)

    out = {
        "event_id": event_id, "found": True, "ticker": r["ticker"], "accession_number": r["accession_number"],
        "form_type": r["form_type"], "eligibility_status": r["eligibility_status"], "stages": stages,
        "timing": {"acceptance_utc": ts(r["acceptance_utc"]), "filing_date": r["filing_date"],
                   "assumed_available_utc": ts(r["assumed_available_utc"]),
                   "availability_method": r["availability_method"],
                   "availability_is_assumption": True, "observed_at_utc": ts(r.get("observed_at_utc")),
                   "decision_cutoff_utc": ts(r["decision_cutoff_utc"]), "target_session": ts(r["target_session"]),
                   "timing_flags": json.loads(r["timing_flags"]) if isinstance(r["timing_flags"], str) else []},
        "data_quality_flags": json.loads(r["data_quality_flags"]) if isinstance(r["data_quality_flags"], str) else [],
        "feature_availability": FEATURE_AVAILABILITY,
        "assumption_dependent_inputs": ["availability proxy (acceptance + lag) instead of observed time",
                                        "Company Facts as a current aggregate of as-filed values"],
    }
    if r["eligibility_status"] == "eligible":
        f = pd.read_parquet(layout.processed / "features.parquet").set_index("event_id").loc[event_id]
        groups = feature_groups()
        out["missing_features"] = {g: [c for c in groups[g] if pd.isna(f.get(c))]
                                   for g in ("market", "context", "financial", "text")}
        out["reference_filing_id"] = r["reference_filing_id"]
    return out


def reconcile_timestamps(client: SecClient, layout: DataLayout, sample: pd.DataFrame) -> pd.DataFrame:
    """Compare submissions-API acceptance (UTC) with index-header ACCEPTANCE-DATETIME (Eastern)."""
    rows = []
    for _, r in sample.iterrows():
        try:
            rr = client.index_headers(int(r["cik"]), r["accession_number"])
            raw = parse_index_header_acceptance(rr.path.read_bytes())
        except Exception as exc:  # network or parsing failure is a finding, not a crash
            rows.append({"accession_number": r["accession_number"], "status": f"unavailable: {exc}"})
            continue
        if raw is None:
            rows.append({"accession_number": r["accession_number"], "status": "header_field_missing"})
            continue
        hdr = pd.Timestamp(raw).tz_localize("America/New_York").tz_convert("UTC")
        api = pd.Timestamp(r["acceptance_utc"])
        diff = (api - hdr).total_seconds()
        rows.append({"accession_number": r["accession_number"], "ticker": r["ticker"], "api_utc": str(api),
                     "header_eastern": raw, "header_as_utc": str(hdr), "difference_seconds": diff,
                     "status": "match" if abs(diff) < 1 else "mismatch"})
    df = pd.DataFrame(rows)
    write_json_atomic(layout.processed / "timestamp_reconciliation.json", df.to_dict(orient="records"))
    return df


def reconcile_run(run_dir: Path) -> dict:
    """Recompute headline metrics from the canonical predictions and compare with metrics.json."""
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    metrics = read_json(run_dir / "metrics.json")
    checks = []
    for key, stored in metrics["models"].items():
        fam, model = key.split(":", 1)
        p = preds[(preds["family"] == fam) & (preds["model"] == model)]
        mb = metric_block(p["y"], p["p_cal"])
        for k in ("n_events", "n_positive", "brier", "log_loss", "average_precision"):
            a, b = mb[k], stored["calibrated"][k]
            ok = (a is None and b is None) or (a is not None and b is not None and np.isclose(a, b, atol=1e-12))
            checks.append({"model": key, "metric": k, "recomputed": a, "stored": b, "ok": bool(ok)})
    events = preds["event_id"].nunique()
    sample_ok = events == metrics["sample"]["n_events"]
    return {"run_id": run_dir.name, "all_ok": all(c["ok"] for c in checks) and sample_ok,
            "n_checks": len(checks), "failed": [c for c in checks if not c["ok"]],
            "sample_events_match": sample_ok}


def extraction_audit_table(layout: DataLayout, sample_csv: Path) -> pd.DataFrame:
    s = pd.read_csv(sample_csv)
    x = pd.read_parquet(layout.interim / "extractions.parquet").set_index("accession_number")
    cols = ["mdna_status", "mdna_method", "mdna_n_paragraphs", "mdna_n_words", "mdna_n_tables",
            "mdna_heuristic_quality", "mdna_n_start_candidates", "risk_status", "risk_n_words", "parser_version"]
    return s.join(x[cols], on="accession_number")


def acceptance_reconciliation_summary(layout: DataLayout, previous_events: Path | None = None) -> dict:
    """Summarise API-vs-index-page acceptance discrepancies and, if a previous event table is given,
    which events changed timing classification because of the correction."""
    from filingshock.discovery import load_filings
    f = load_filings(layout)
    chk = f[f["acceptance_source"] == "filing_index_page"]
    d = chk["acceptance_discrepancy_seconds"].round()
    disc = chk[d.abs() >= 1]
    by_form = disc.groupby("form_type").size().to_dict()
    out = {
        "n_checked": int(len(chk)), "n_discrepant": int(len(disc)),
        "share_discrepant": float(len(disc) / len(chk)) if len(chk) else None,
        "difference_counts": {str(int(k)): int(v) for k, v in d[d.abs() >= 1].value_counts().items()},
        "discrepant_by_form": {k: int(v) for k, v in by_form.items()},
        "discrepant_by_year": {str(k): int(v) for k, v in
                               disc["acceptance_utc"].dt.year.value_counts().sort_index().items()},
        "n_unverified": int((f["acceptance_source"] != "filing_index_page").sum()),
        "direction": "API later than index page" if (disc["acceptance_discrepancy_seconds"] > 0).all() else "mixed",
    }
    if previous_events is not None and previous_events.exists():
        old = pd.read_parquet(previous_events).set_index("event_id")
        new = pd.read_parquet(layout.processed / "events.parquet").set_index("event_id")
        common = old.index.intersection(new.index)
        o_stat = old.loc[common, "exclusion_reason"].fillna("eligible")
        n_stat = new.loc[common, "exclusion_reason"].fillna("eligible")
        changed = common[(o_stat != n_stat).to_numpy()]
        tgt_changed = common[(old.loc[common, "target_session"].astype(str) != new.loc[common, "target_session"].astype(str)).to_numpy()]
        out["n_classification_changed"] = int(len(changed))
        out["classification_changes"] = (pd.DataFrame({"before": o_stat[changed], "after": n_stat[changed]})
                                         .value_counts().rename("events").reset_index().to_dict(orient="records"))
        out["n_target_session_changed"] = int(len(tgt_changed))
        out["previously_eligible_now_excluded"] = sorted(changed[(o_stat[changed] == "eligible").to_numpy()].tolist())
    write_json_atomic(layout.processed / "acceptance_reconciliation.json", out)
    return out
