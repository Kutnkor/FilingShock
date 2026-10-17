# Data card

## Sources

| Source | What | Access | Terms | In repository? |
|---|---|---|---|---|
| SEC EDGAR submissions API (`data.sec.gov/submissions`) | Filing lists, forms, 8-K items, filing dates, API acceptance times | Free; declared User-Agent; ≤ 4 requests/s (SEC limit 10/s) | Public records | No (re-downloadable) |
| EDGAR filing index pages (`…/<accession>-index.htm`) | Authoritative "Accepted" time (Eastern) | Same | Public records | No |
| EDGAR primary 10-Q documents | MD&A and risk-factor text | Same | Public records | Extracted sections in `sample/` only |
| SEC Company Facts API | us-gaap facts with reporting accession numbers | Same | Public records | No |
| Yahoo Finance via `yfinance` 1.7.0 | Daily OHLCV, dividends, splits for 14 stocks, SPY, IGV | Unofficial endpoint, no key | **Not redistributable** | No — import path provided |
| Loughran–McDonald Master Dictionary 1993–2025 (SRAF, Notre Dame) | Uncertainty, negative, litigious, constraining word lists with year-added flags | Free download | Free for academic research; commercial licence required | No |
| `resources/dictionaries/topics_v1.yaml` | Project-authored topic patterns | — | Project file | Yes |

Retrieval timestamps and SHA-256 hashes: `data/raw/sec/retrieval_manifest.jsonl`,
`data/raw/market/yfinance/manifest.json`, `data/raw/resources/lm_manifest.json`. All were retrieved on
2026-10-17; no earlier data vintages exist in this project.

## Coverage

* Universe: 21 pre-specified SIC 7372 companies (selection rule in `docs/research_protocol.md`); 14 have
  free price history; 7 acquired/delisted companies (ANSS, CTXS, VMW, COUP, SPLK, NUAN, CA) do not.
* Filing window: acceptance 2010-01-01 to 2026-09-30. Prices: 2009-01-02 to 2026-10-16.
* Unit of observation: one company-filing event (original 10-Q) with one target session.
* Funnel counts and the final sample size: `data/processed/funnel.json` (shown on the Results page and
  in `docs/results.md`).

## Known data issues (measured)

* **SEC submissions API acceptance times**: for a share of filings the API value is later than the filing
  index page by exactly the Eastern UTC offset (+4 h / +5 h). Index-page times are used; statistics in
  `data/processed/acceptance_reconciliation.json` and the Methodology page.
* Price bars matched the XNYS calendar session-for-session for every symbol; no invalid bars were found.
  OHLC are split-adjusted, not dividend-adjusted; open and close in one session share a basis.
* Large overnight gaps (> 35 %) inside a threshold window are flagged per event (e.g., Gen Digital's 2020
  special dividend), not removed; they do not affect open-to-close returns.
* Company Facts is a current aggregate. Point-in-time use relies on each fact's reporting accession and
  that filing's acceptance time; whether every originally filed value is still present was not verified.
* Debt tags change across companies and years, so leverage is measured as 1 − equity/assets.

## Personal data

None. Filings are corporate disclosures. The SEC User-Agent in `config/default.yaml` is an explicit
placeholder (`contact-UNCONFIGURED@example.invalid`); set `FILINGSHOCK_SEC_USER_AGENT` to your own
contact before running large downloads.
