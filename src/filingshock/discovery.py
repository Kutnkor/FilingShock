"""Filing discovery from SEC submissions history."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from filingshock.config import Company, Config
from filingshock.sources.sec import SecClient, submissions_to_frame
from filingshock.storage import DataLayout, write_json_atomic, write_parquet_atomic

log = logging.getLogger(__name__)

PERIODIC_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A", "10-KT", "10-QT", "10-KT/A", "10-QT/A"}
PRIMARY_FORM = "10-Q"
CONTEXT_FORMS = {"8-K", "8-K/A"}


def carry_first_seen(filings: pd.DataFrame, layout: DataLayout, now: datetime | None = None) -> pd.DataFrame:
    """first_seen_utc: when a discovery run first listed the accession (the observed time used by
    prospective forecasts). Values from earlier runs are kept, so it never moves; accessions that are
    new, or that predate this column, get the current run's time (later than the truth, never earlier)."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    path = layout.interim / "filings.parquet"
    prev: dict = {}
    if path.exists():
        old = pd.read_parquet(path)
        if "first_seen_utc" in old:
            prev = dict(zip(old["accession_number"], pd.to_datetime(old["first_seen_utc"], utc=True)))
    out = filings.copy()
    out["first_seen_utc"] = pd.to_datetime(out["accession_number"].map(prev), utc=True).fillna(now)
    return out


def discover_filings(cfg: Config, layout: DataLayout, client: SecClient,
                     companies: list[Company], submissions_max_age_days: float = 30) -> pd.DataFrame:
    """Build the filings table (periodic reports and 8-Ks) for the configured universe.

    submissions_max_age_days=0 forces fresh submissions (needed to see filings for prospective forecasts).
    """
    frames, meta = [], {}
    for comp in companies:
        doc = client.submissions(comp.cik, max_age_days=submissions_max_age_days)
        df = submissions_to_frame(doc)
        df = df[df["form_type"].isin(PERIODIC_FORMS | CONTEXT_FORMS)].copy()
        df["ticker"] = comp.ticker
        df["company_name"] = comp.name
        frames.append(df)
        meta[comp.ticker] = {
            "cik": comp.cik, "sec_name": doc.get("name"), "sic": doc.get("sic"),
            "sic_description": doc.get("sicDescription"), "current_tickers": doc.get("tickers"),
            "exchanges": doc.get("exchanges"), "fiscal_year_end": doc.get("fiscalYearEnd"),
            "former_names": doc.get("formerNames"), "submissions_sha256": doc.get("_sha256"),
            "n_filings_in_submissions": int(len(submissions_to_frame(doc))),
        }
        log.info("%s: %d periodic/8-K filings in submissions history", comp.ticker, len(df))
    filings = pd.concat(frames, ignore_index=True)
    win = cfg["universe"]["filing_window"]
    lo = pd.Timestamp(win["accepted_from"], tz="UTC")
    hi = pd.Timestamp(win["accepted_to"], tz="UTC") + pd.Timedelta(days=1)
    filings["in_window"] = (filings["acceptance_utc"] >= lo) & (filings["acceptance_utc"] < hi)
    filings["items"] = filings["items"].fillna("").astype(str)
    filings["is_earnings_8k"] = filings["form_type"].isin(CONTEXT_FORMS) & filings["items"].str.contains("2.02")
    filings = carry_first_seen(filings, layout)
    write_parquet_atomic(layout.interim / "filings.parquet", filings)
    write_json_atomic(layout.interim / "companies.json", meta)
    return filings


def load_filings(layout: DataLayout) -> pd.DataFrame:
    df = pd.read_parquet(layout.interim / "filings.parquet")
    df["acceptance_utc"] = pd.to_datetime(df["acceptance_utc"], utc=True)
    return df


def reconcile_acceptance(layout: DataLayout, client: SecClient, mask_fn=None, progress_every: int = 200) -> pd.DataFrame:
    """Replace submissions-API acceptance times with the filing index page's 'Accepted' field.

    The audit found that the submissions API's acceptanceDateTime is, for some filings, later than the
    acceptance time on the filing's own index page by exactly the Eastern UTC offset (4 h in EDT, 5 h in
    EST). The index page (Eastern local time) is treated as authoritative; the API value is kept in
    acceptance_api_utc and every discrepancy is preserved as a quality finding.
    """
    from filingshock.sources.sec import SecAccessError, index_page, parse_index_page_accepted

    f = pd.read_parquet(layout.interim / "filings.parquet")
    f["acceptance_utc"] = pd.to_datetime(f["acceptance_utc"], utc=True)
    if "acceptance_api_utc" not in f:
        f["acceptance_api_utc"] = f["acceptance_utc"]
        f["acceptance_index_et"] = None
        f["acceptance_source"] = "submissions_api_unverified"
    todo = f.index[(mask_fn(f) if mask_fn else pd.Series(True, index=f.index)) & f["acceptance_index_et"].isna()]
    for k, i in enumerate(todo, 1):
        row = f.loc[i]
        try:
            r = index_page(client, int(row["cik"]), row["accession_number"])
            val = parse_index_page_accepted(r.path.read_bytes())
        except SecAccessError as exc:
            log.warning("index page unavailable for %s: %s", row["accession_number"], exc)
            val = None
        if val:
            f.at[i, "acceptance_index_et"] = val
        if k % progress_every == 0:
            log.info("acceptance reconciliation: %d/%d (%d SEC requests)", k, len(todo), client.request_count)
    has = f["acceptance_index_et"].notna()
    idx_utc = pd.to_datetime(f.loc[has, "acceptance_index_et"]).dt.tz_localize("America/New_York",
                                                                               ambiguous="NaT").dt.tz_convert("UTC")
    f.loc[has, "acceptance_utc"] = idx_utc
    f.loc[has, "acceptance_source"] = "filing_index_page"
    f["acceptance_discrepancy_seconds"] = (f["acceptance_api_utc"] - f["acceptance_utc"]).dt.total_seconds()
    write_parquet_atomic(layout.interim / "filings.parquet", f)
    return f
