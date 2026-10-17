"""Offline sample package: a redistributable subset that lets the interface run without downloads.

Included: SEC-derived content (event metadata, extracted sections, alignments, financial and text features),
aggregate experiment results and the saved model probabilities for the selected events.
Excluded: provider price data and every per-event value derived from it (open/close, returns, thresholds,
labels, market features), because the provider's terms do not permit redistribution.
Selection rule (documented, outcome-blind): for each company, the most recent final-sample event that has a
retrospective out-of-time prediction, plus the earliest final-sample event (training period, no prediction).
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from filingshock.config import Config
from filingshock.storage import DataLayout, read_json, write_json_atomic

PRICE_DERIVED_EVENT_COLS = []          # events table carries no price values
PRICE_DERIVED_PRED_COLS = ["y", "abs_r", "threshold_abs_move"]


def build_sample_package(cfg: Config, layout: DataLayout, out_dir: Path, n_events: int = 12) -> dict:
    out = DataLayout(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out.ensure()
    ev = pd.read_parquet(layout.processed / "events.parquet")
    feats = pd.read_parquet(layout.processed / "features.parquet")
    run_id = read_json(layout.experiments / "primary_run.json")["run_id"]
    run_dir = layout.experiments / run_id
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    final = ev[ev["eligibility_status"] == "eligible"].sort_values("acceptance_utc")
    with_pred = final[final["event_id"].isin(preds["event_id"])]
    recent = with_pred.groupby("ticker").tail(1).sort_values("acceptance_utc", ascending=False).head(n_events)
    early = final.groupby("ticker").head(1).head(2)
    chosen = pd.concat([recent, early]).drop_duplicates("event_id")
    ids = set(chosen["event_id"])

    # events: chosen eligible events + all excluded rows (SEC metadata only) so exclusions remain inspectable
    sample_events = ev[ev["event_id"].isin(ids) | (ev["eligibility_status"] != "eligible")].copy()
    sample_events.to_parquet(out.processed / "events.parquet", index=False)
    keep = [c for c in feats.columns if not c.startswith("mkt_")]
    feats[feats["event_id"].isin(ids)][keep].to_parquet(out.processed / "features.parquet", index=False)
    funnel = read_json(layout.processed / "funnel.json")
    funnel["note"] = "Funnel of the full research dataset; the sample package contains only a subset of events."
    write_json_atomic(out.processed / "funnel.json", funnel)
    dm = read_json(layout.processed / "dataset_manifest.json")
    dm["price_files"] = {}
    dm["sample_package_note"] = "Price-derived values removed; counts describe the full research dataset."
    write_json_atomic(out.processed / "dataset_manifest.json", dm)
    for name in ("acceptance_reconciliation.json",):
        if (layout.processed / name).exists():
            shutil.copy2(layout.processed / name, out.processed / name)

    accs = set(chosen["accession_number"]) | set(chosen["reference_filing_id"].dropna())
    for a in accs:
        src = layout.sections_dir / f"{a}.json"
        if src.exists():
            shutil.copy2(src, out.sections_dir / src.name)
    for e in ids:
        src = layout.alignments_dir / f"{e}.parquet"
        if src.exists():
            shutil.copy2(src, out.alignments_dir / src.name)
    x = pd.read_parquet(layout.interim / "extractions.parquet")
    x[x["accession_number"].isin(accs)].to_parquet(out.interim / "extractions.parquet", index=False)

    rdir = out.experiments / run_id
    rdir.mkdir(parents=True)
    for f in ("manifest.json", "metrics.json", "selection_log.json", "COMPLETE"):
        shutil.copy2(run_dir / f, rdir / f)
    p = preds[preds["event_id"].isin(ids)].drop(columns=PRICE_DERIVED_PRED_COLS)
    p.to_parquet(rdir / "predictions.parquet", index=False)
    write_json_atomic(out.experiments / "primary_run.json", {"run_id": run_id, "note": "copied from research data"})
    # sensitivity runs: aggregate metrics only
    for d in layout.experiments.iterdir():
        if d.is_dir() and d.name != run_id and (d / "COMPLETE").exists():
            man = read_json(d / "manifest.json")
            if man["dataset"]["dataset_id"] == dm["dataset_id"]:
                (out.experiments / d.name).mkdir()
                for f in ("manifest.json", "metrics.json", "COMPLETE"):
                    shutil.copy2(d / f, out.experiments / d.name / f)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset_id": dm["dataset_id"], "source_run_id": run_id,
        "selection_rule": __doc__.split("Selection rule")[1].strip(),
        "events_included": sorted(ids), "companies": sorted(chosen["ticker"].unique()),
        "excluded_content": ["provider price bars", "target-session open/close/returns", "thresholds and labels",
                             "market (price-derived) features", "raw filing HTML (link to SEC.gov instead)"],
        "licensing": {"sec_content": "SEC EDGAR filings are public records of the U.S. government.",
                      "prices": "Yahoo Finance data (via yfinance) is not redistributable; not included.",
                      "loughran_mcdonald": "Dictionary file not included; only per-event derived rates are."},
    }
    write_json_atomic(out.root / "package_manifest.json", manifest)
    (out.root / "README.md").write_text(
        "# FilingShock offline sample package\n\nStart with `filingshock app --sample`. See package_manifest.json for the "
        "selection rule and what was excluded and why. Nothing here triggers downloads or retraining.\n")
    size = sum(f.stat().st_size for f in out.root.rglob("*") if f.is_file())
    return {"events": len(ids), "companies": manifest["companies"], "size_mb": round(size / 1e6, 2), "dir": str(out_dir)}
