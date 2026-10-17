# Known limitations

Concrete to this dataset and experiment; numbers come from saved artifacts (`docs/results.md`,
`data/processed/funnel.json`, `data/processed/acceptance_reconciliation.json`).

1. **Fixed, survivor-heavy cohort.** 21 hand-picked SIC 7372 companies; the 7 acquired or delisted ones have
   no free price history, so every final event comes from 14 companies that still trade. Results describe
   this cohort, not the software sector.
2. **Availability is a proxy.** EDGAR publishes no first-public-availability time. Acceptance + 15 minutes
   (or next-day 06:00 ET for after-17:30 submissions) is assumed. A live system would record observed times;
   none of the historical events have one.
3. **Timestamp source error.** The SEC submissions API mis-states acceptance times for a share of filings
   (exactly the Eastern UTC offset too late). The corrected index-page times are used; filings whose index
   page could not be read keep the unverified API time and are counted in the reconciliation summary.
4. **Earnings releases.** 176 of 573 final events (31%) have an earnings 8-K (Item 2.02) in the same pre-cutoff
   window. For those, the target session mostly reflects the earnings release; the subgroup analysis is
   reported separately and is small.
5. **Concurrent news** beyond 8-K Item 2.02 (press releases, analyst actions, macro news) is not observed.
6. **Small independent sample.** 467 out-of-time events and 125 large moves; the primary ΔBrier interval is
   about ±0.005 wide on each side, so smaller effects cannot be distinguished from zero, and per-year figures are noisy.
7. **Data vintages.** Prices and Company Facts were retrieved in October 2026. Vendor split re-adjustments
   and the API's current aggregation are what is stored; earlier vintages are not available.
8. **Parsing.** MD&A extraction succeeded for every parsed filing and was audited manually on 50 pilot filings
   and a 14-filing stratified sample, by one reviewer. Alignment quality labels are heuristics.
9. **Dictionaries.** The Loughran–McDonald lists are applied point-in-time by year flag, but the base lists
   (flag 2009) were finalised in 2011; the topic lexicon was written in 2026.
10. **Prospective forecasting.** `filingshock forecast-pending` scores 10-Qs awaiting their decision cutoff
    through the same per-event feature functions as the historical build and appends models A/B/C plus a constant to the
    hash-chained ledger, only before the cutoff and only for filings first seen by discovery before it. No
    real prospective forecast has been recorded yet (`data/forecasts/` is empty), so there is no prospective
    evidence. `filingshock score-prospective` scores a record only after its target session has closed and
    settled (close + 60 min) and only from a price file retrieved after that, against the threshold stored
    in the forecast (never recomputed); results go to a separate hash-chained
    `data/forecasts/prospective_outcomes.jsonl`, one record per forecast, never overwritten, and never enter
    the research metrics. An invalid target bar is recorded as such (y = null) rather than retried; a missing
    target bar keeps the forecast pending. The Results page shows a pooled prospective Brier only from 20
    scored events (earliest forecast per event), without an interval or a conclusion: a handful of quarterly
    events per company cannot confirm or overturn the research result. Live conditions differ from the
    historical reconstruction: Company Facts may not yet contain the new filing's XBRL facts at scoring time
    (recorded per forecast as `current_filing_facts_in_company_facts`); prices must be refreshed through the
    previous session with `prices --refresh --history-end today` (stale history blocks scoring; the config's
    fixed `history_end` stays the research default). That refresh replaces the local price files with a new
    vendor vintage, which a later research `build` would then read; bars of sessions that had not closed
    and settled when the request started are never stored. Observed times exist only from the first `discover`
    run after this feature (earlier filings get that run's time, which is later than the truth, never earlier); and
    the production models are refitted from the frozen research dataset, so a 2027 forecast is calibrated
    on the 2026 events in the filing window (to 2026-09-30) only.
11. **Prediction ≠ causation ≠ profit.** Neither a positive nor a null predictive result says anything about
    the causal effect of disclosures or about trading profitability.

12. **Local evidence integrity.** Version 0.2 serializes cooperating writers and validates record hashes,
    timing, probabilities and outcome links. A local hash chain is not an externally timestamped or
    signed publication record. Someone able to rewrite the whole local store can recompute a new chain.
    The lock implementation targets local filesystems; network/distributed filesystem semantics are
    not claimed. The Windows locking branch has not been executed in the local macOS verification.
