# Information-timing policy

Version: timing-1.0 (frozen 2026-10-17, before any model was trained). Configured in
`config/default.yaml` → `timing`. Implemented in `src/filingshock/timing.py`; enforced again in
`events.py` (feature builders only receive history before the target session) and tested in
`tests/test_timing.py` and `tests/test_leakage.py`.

## Timestamp fields kept separately

| Field | Meaning | Source |
|---|---|---|
| `report_date` | Fiscal-period end (CONFORMED PERIOD OF REPORT) | SEC submissions API |
| `filing_date` | Official EDGAR filing date | SEC submissions API |
| `acceptance_utc` | EDGAR acceptance timestamp, UTC | "Accepted" field of the filing's EDGAR index page (Eastern → UTC); `acceptance_api_utc` keeps the submissions-API value |
| `assumed_available_utc` | Historical availability **proxy** (see below) | derived, `availability_method` says how |
| `observed_at_utc` | When FilingShock itself first saw the document live | prospective forecasts only (`first_seen_utc` of the first `discover` run that listed the accession, kept across runs); empty for historical events |
| retrieval timestamp | When the file was downloaded for this project | `data/raw/sec/retrieval_manifest.jsonl` (never used as availability) |
| `decision_cutoff_utc` | Information cutoff for the prediction | target open − 30 min |
| `generated_at_utc` | When a prediction was actually computed | prediction records |
| `target_open_utc`, `target_close_utc` | Regular-session open and close of the target session | XNYS calendar |
| `outcome_available_utc` | When the realized label became knowable | target close |
| `scored_at_utc` | When a prospective forecast's outcome was scored | no earlier than target close + 60 min, from a price file retrieved after that (`forecasts/prospective_outcomes.jsonl`); bars of sessions not yet closed + 60 min are never stored |

## What was verified

* For most filings the submissions API's `acceptanceDateTime` is genuine UTC: Adobe 2024-06-13 index header
  16:06:10 ET ↔ API 20:06:10Z (EDT) and 2022-12-15 API 21:06:10Z ↔ 16:06:10 EST.
* **But not for all.** The planned timestamp reconciliation (`filingshock audit --timestamps 14`) found 4 of
  11 checkable filings where the API value was later than the filing's own header by exactly the Eastern
  UTC offset (+4 h in EDT, +5 h in EST), e.g. Microsoft's 10-Q of 2019-04-24: header 16:10:10 ET, API
  00:10:10Z (= 20:10 ET). The header time matches Microsoft's practice of filing minutes after its 16:05
  earnings release. An error of this kind can make a filing accepted during trading look like an
  after-close filing.
* **Resolution (applied to every filing used):** acceptance times are taken from the "Accepted" field of
  each filing's EDGAR index page (`<accession>-index.htm`, Eastern local time), which exists for every
  filing (some older `-index-headers.html` files return 404). The API value is preserved as
  `acceptance_api_utc`, the difference as `acceptance_discrepancy_seconds`, and the source as
  `acceptance_source`. Discrepancy statistics and every event whose timing classification changed are
  written to `data/processed/acceptance_reconciliation.json` and shown on the Methodology page.
  This correction was made **after** the first primary results had been seen; see the protocol change log.
* SEC webmaster FAQ: filings are "often available on sec.gov within 1-3 minutes of the EDGAR system
  timestamp", the lag "can increase significantly", and "there is no timestamp to indicate when filing
  content is first available".
* EDGAR operating rules: live submissions that begin transmission after 17:30 ET receive the next
  business day's filing date and are not disseminated until the next business day.

## Availability proxy (historical reconstruction)

An acceptance timestamp is **not proof** of public availability. FilingShock therefore uses:

* `acceptance_plus_lag`: `assumed_available = acceptance + 15 min` (configurable, conservative relative
  to the 1–3 minute typical lag stated by the SEC).
* `next_day_dissemination_plus_lag`: if acceptance is after 17:30 ET **or** the filing date is later than
  the acceptance date, `assumed_available = max(acceptance + lag, filing_date 06:00 ET + lag)`.

Every event records `availability_method` and timing flags. Historical events never receive an
`observed_at_utc`; a document downloaded today is never backdated as if it had been observed then.

## Primary event cohort and target session

Let *S* be the first exchange session whose regular open is after the acceptance time and *P* the
session before it.

1. If acceptance < close(*P*) the filing arrived during regular trading → excluded
   (`accepted_during_regular_session`). This includes early-close days, where close(*P*) is 13:00 ET.
2. Otherwise the decision cutoff is open(*S*) − 30 min. If `assumed_available` > cutoff → excluded
   (`available_after_decision_cutoff`).
3. Otherwise the event is timing-eligible and *S* is the target session.

Exchange schedules come from the `exchange_calendars` XNYS calendar (holidays, early closes, DST),
interpreted in America/New_York; all stored timestamps are UTC. Price bars from the provider were
checked to match the calendar session-for-session for every symbol used.

## Information admissible as a feature

* Prices: only sessions strictly before *S* (enforced by slicing before feature computation).
* Financial facts: only facts reported by periodic filings with `acceptance + lag ≤ cutoff`; for the
  same period the latest admissible report wins (no later restatement can enter).
* Earnings-release context: 8-K Item 2.02 filings with `acceptance + lag ≤ cutoff`.
* Previous-event information: only events whose outcomes matured (target close) before the cutoff.
* Text: the current filing and its reference filing, both available by the cutoff.

Training uses an event only after its `outcome_available_utc`; annual splits guarantee this
(see `docs/research_protocol.md`).
