# Data provenance and lineage

```
SEC submissions JSON ──► interim/filings.parquet ◄── filing index pages ("Accepted", Eastern)
        (raw/sec/submissions)        │  acceptance_utc, acceptance_api_utc, acceptance_source
                                     ▼
                 timing.classify_timing  (XNYS calendar, availability proxy, cutoff)
                                     ▼
raw/market/yfinance/*.parquet ──► target.historical_threshold / realized_outcome ──► processed/outcomes.parquet
        (manifest.json: sha256, retrieved_at)      features.market (history < target) ─┐
                                                                                         │
raw/sec/filings/<cik>/<acc>/*.htm ──► parsing.html → parsing.sections ──► interim/sections/<acc>.json
        (retrieval_manifest.jsonl: sha256)          (parser fs-parser-1.1)               │
                                     reference.select_reference ──► parsing.align ──► interim/alignments/<event>.parquet
                                                                                         │
raw/sec/companyfacts/*.json ──► features.financial (facts admissible at cutoff) ────────┤
raw/resources/LM dictionary ──► features.text (point-in-time word lists) ──────────────┤
                                                                                         ▼
                                       processed/events.parquet  +  processed/features.parquet
                                       processed/funnel.json, dataset_manifest.json (dataset_id)
                                                                                         ▼
                                 modeling.experiment ──► experiments/<run_id>/{manifest, predictions, metrics,
                                                                             selection_log}.json/.parquet + COMPLETE
                                                                                         ▼
                                         ui/data.py (read-only) ──► Streamlit pages; reporting ──► docs/
```

Tracing a displayed excerpt back to its source: the Comparison page shows the paragraph identifiers
(`R-###`/`C-###` = ordinal in the extracted section), the accession numbers and links to the SEC filing.
`interim/sections/<accession>.json` stores each paragraph's source block index and character offsets, the
parser version, and the SHA-256 of the raw document it came from (`summary.source_sha256`).

Tracing a displayed number: every probability in the interface is a row of
`experiments/<run_id>/predictions.parquet` (event_id, model, family, test year, calibration method,
generation time, dataset id). `filingshock verify-artifacts` recomputes the summary metrics from that file.

Identifiers: `event_id = "ev-" + accession number`; `dataset_id` hashes the final event ids, document
hashes, feature values and outcomes; `run_id` is the UTC start time plus the run name.
