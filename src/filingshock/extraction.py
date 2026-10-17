"""Document download + section extraction step (resumable, idempotent)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import pandas as pd

from filingshock import PARSER_VERSION
from filingshock.parsing.html import html_to_blocks
from filingshock.parsing.sections import extract_mdna, extract_risk, find_headings
from filingshock.sources.sec import SecAccessError, SecClient
from filingshock.storage import DataLayout, sha256_file, write_json_atomic, write_parquet_atomic

log = logging.getLogger(__name__)


def extract_filing(client: SecClient, layout: DataLayout, row: pd.Series, force: bool = False) -> dict:
    """Download (if needed) and extract MD&A + risk sections for one filing. Returns a summary row."""
    out_path = layout.sections_dir / f"{row['accession_number']}.json"
    if out_path.exists() and not force:
        doc = json.loads(out_path.read_text())
        if doc.get("parser_version") == PARSER_VERSION:
            return doc["summary"]
    base = {"accession_number": row["accession_number"], "cik": int(row["cik"]),
            "primary_document": row["primary_document"], "parser_version": PARSER_VERSION,
            "source_url": client.document_url(row["cik"], row["accession_number"], row["primary_document"])}
    try:
        r = client.filing_document(row["cik"], row["accession_number"], row["primary_document"])
    except SecAccessError as exc:
        summary = {**base, "download_status": "failed", "download_error": str(exc),
                   "mdna_status": "not_attempted", "risk_status": "not_attempted"}
        return summary  # not persisted: a later run retries the download
    blocks = html_to_blocks(r.path.read_bytes())
    heads = find_headings(blocks)
    mdna = extract_mdna(blocks, heads)
    risk = extract_risk(blocks, heads, mdna.end_block)
    summary = {**base, "download_status": "ok", "source_sha256": sha256_file(r.path),
               "n_blocks": len(blocks)}
    for name, ext in (("mdna", mdna), ("risk", risk)):
        for k, v in ext.summary().items():
            if k not in ("section", "parser_version"):
                summary[f"{name}_{k}"] = v
    doc = {"parser_version": PARSER_VERSION, "summary": summary,
           "mdna": [asdict(p) for p in mdna.paragraphs], "risk": [asdict(p) for p in risk.paragraphs]}
    write_json_atomic(out_path, doc)
    return summary


def run_extraction(client: SecClient, layout: DataLayout, filings: pd.DataFrame, force: bool = False,
                   progress_every: int = 50) -> pd.DataFrame:
    rows = []
    total = len(filings)
    for k, (_, row) in enumerate(filings.iterrows(), 1):
        rows.append(extract_filing(client, layout, row, force=force))
        if k % progress_every == 0 or k == total:
            log.info("extraction progress: %d/%d filings (%d SEC requests so far)", k, total, client.request_count)
    df = pd.DataFrame(rows)
    path = layout.interim / "extractions.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["accession_number"].isin(df["accession_number"])]
        df = pd.concat([old, df], ignore_index=True)
    write_parquet_atomic(path, df)
    return df
