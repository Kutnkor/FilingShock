"""Orchestration of the data-preparation steps shared by the CLI and the interface."""

from __future__ import annotations

import logging

import pandas as pd

from filingshock.config import Config
from filingshock.discovery import PERIODIC_FORMS, discover_filings, load_filings, reconcile_acceptance
from filingshock.events import (
    build_candidates,
    build_event_tables,
    dataset_manifest,
    needed_accessions,
    write_event_tables,
)
from filingshock.extraction import run_extraction
from filingshock.features.text import fetch_lm_dictionary
from filingshock.sources.market import ingest_prices
from filingshock.sources.sec import SecClient
from filingshock.storage import DataLayout, write_json_atomic

log = logging.getLogger(__name__)


def step_discover(cfg: Config, layout: DataLayout, pilot: bool = False, offline: bool = False,
                  refresh: bool = False) -> pd.DataFrame:
    client = SecClient(cfg, layout, offline=offline)
    df = discover_filings(cfg, layout, client, cfg.companies(pilot_only=pilot),
                          submissions_max_age_days=0 if refresh else 30)
    log.info("discovery complete: %d filings (%d SEC requests)", len(df), client.request_count)
    # Acceptance times are reconciled against each filing's index page (docs/timing_policy.md).
    start = pd.Timestamp(cfg["universe"]["filing_window"]["accepted_from"], tz="UTC") - pd.Timedelta(days=550)

    def needs(f):
        return ((f["form_type"].isin(PERIODIC_FORMS) | f["is_earnings_8k"]) & (f["acceptance_utc"] >= start))

    df = reconcile_acceptance(layout, client, needs)
    win = cfg["universe"]["filing_window"]
    lo = pd.Timestamp(win["accepted_from"], tz="UTC")
    hi = pd.Timestamp(win["accepted_to"], tz="UTC") + pd.Timedelta(days=1)
    df["in_window"] = (df["acceptance_utc"] >= lo) & (df["acceptance_utc"] < hi)
    from filingshock.storage import write_parquet_atomic
    write_parquet_atomic(layout.interim / "filings.parquet", df)
    n_bad = int((df["acceptance_discrepancy_seconds"].abs() >= 1).sum())
    log.info("acceptance reconciliation: %d filings checked, %d API/index discrepancies (%d SEC requests)",
             int((df["acceptance_source"] == "filing_index_page").sum()), n_bad, client.request_count)
    return df


def step_prices(cfg: Config, layout: DataLayout, refresh: bool = False, history_end: str | None = None) -> dict:
    syms = [c.ticker for c in cfg.companies()] + list(cfg["universe"]["benchmark_symbols"].values())
    return ingest_prices(cfg, layout, syms, refresh=refresh, history_end=history_end)


def step_resources(cfg: Config, layout: DataLayout, offline: bool = False) -> None:
    client = SecClient(cfg, layout, offline=offline)
    filings = load_filings(layout)
    for cik in sorted(filings["cik"].unique()):
        client.company_facts(int(cik))
    if not offline:
        fetch_lm_dictionary(layout, cfg.sec_user_agent)


def step_build(cfg: Config, layout: DataLayout, offline: bool = False, force_extract: bool = False) -> dict:
    client = SecClient(cfg, layout, offline=offline)
    disc, parts = build_candidates(cfg, layout)
    refs, need = needed_accessions(disc, load_filings(layout), cfg)
    log.info("extracting sections for %d filings (events + references)", len(need))
    extractions = run_extraction(client, layout, need, force=force_extract)
    tables = build_event_tables(cfg, layout, disc, parts, refs, extractions)
    manifest = dataset_manifest(cfg, layout, tables)
    write_event_tables(layout, cfg, tables, manifest)
    write_json_atomic(layout.state / "build_state.json", {"status": "complete", "dataset_id": manifest["dataset_id"],
                                                         "sec_requests": client.request_count})
    log.info("dataset %s: %d final events across %d companies", manifest["dataset_id"],
             manifest["n_final_events"], manifest["n_companies_final"])
    return {"manifest": manifest, "funnel": tables["funnel"]}
