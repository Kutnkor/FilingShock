# Presentation pack

All numbers come from run `20261017T090148Z-primary` on dataset `ds-1698abec6cff` (see `docs/results.md`).
Screenshots in `docs/screenshots/` were captured from the running app with headless Chrome on 2026-10-17.

## One-page overview

**Question.** Do changes in a 10-Q's MD&A (vs the same quarter a year earlier) add information, beyond
market and financial data, about an unusually large open-to-close move in the next trading session?

**Data.** 21 pre-specified SEC SIC 7372 software companies; 1,282 periodic reports (2010–2026) → 573
eligible 10-Q events from 14 companies with free price history. Acceptance times reconciled against EDGAR
index pages after finding the submissions API wrong by +4/+5 h for 38% of checked filings.

**Design.** Information cutoff 30 minutes before the open; target = |open→close| > 2 × the stock's 60-session
mean; Models A (market + context), B (+ financial), C (+ 11 interpretable text-change features); identical
folds, tuning and Platt calibration; annual expanding window; 467 out-of-time events, 125 large moves.

**Result.** ΔBrier(C − B) = +0.0006 (95% CI −0.0039 to +0.0055): **inconclusive — no measurable incremental
value from text.** The best model was A (Brier 0.1856, average precision 0.411 vs 0.259 for the constant).

**What is solid.** Point-in-time pipeline with tested leakage guards, audited extraction, reproducible and
reconcilable artifacts, honest unavailable states. **What is not.** Small sample, fixed survivor-heavy
cohort, proxy availability times.

## Figures

| Figure | File |
|---|---|
| Model comparison (paired ΔBrier with intervals) | `figures/paired_differences.png` |
| Calibration (B and C, 8 equal-count bins) | `figures/calibration.png` |
| Brier by test year | `figures/yearly_brier.png` |
| Eligibility funnel | `figures/eligibility_funnel.png` |
| Filing comparison (Adobe FY2024 Q3 vs FY2023 Q3) | `screenshots/comparison_1440.png`, narrow layout `screenshots/comparison_390.png` |
| Results screen | `screenshots/results_1440.png` |

## Cases (selection rules fixed in advance, applied to all pooled Model C out-of-time predictions)

* **Success — highest-scored event that exceeded its threshold.** Intuit FY2017 Q3 10-Q, accepted 2017-05-23
  16:07 ET, target 2017-05-24. P(C) = 94.4% (B: 92.2%). Threshold 1.09%; realized open→close −1.35% → large move.
  An earnings 8-K was released in the same window, and B already scored it almost as high: the
  probability comes mainly from the earnings timing, not from the text.
* **Failure — highest-scored event that did not exceed its threshold.** Tyler Technologies FY2025 Q3, accepted
  2025-10-29 16:56 ET, target 2025-10-30. P(C) = 75.9% (B: 76.1%). Threshold 2.31%; realized +1.93% → not large.
* **Failure — lowest-scored event that did exceed.** Gen Digital FY2016 Q3, accepted 2016-02-04 16:06 ET,
  target 2016-02-05. P(C) = 5.9%. Threshold 2.15%; realized −4.73% → large move missed, although 57% of its
  MD&A paragraphs were new relative to the prior-year quarter.

All three coincide with an earnings release. These are illustrations; the aggregate evidence is the paired
comparison above. They can be opened in the app through Prediction & Historical Replay → Demonstration cases.

## One-minute demonstration script

1. *(Overview, 10 s)* "FilingShock asks whether changes in 10-Q text help predict an unusually large move in the
   next session. The banner shows the data version, and that it's historical, not live."
2. *(Select Adobe FY2024 Q3 → Filing Comparison, 15 s)* "Left is the same quarter last year, right is this
   quarter. By default only changed paragraphs show: added, removed, edited, each with its match score and a
   link to the filing on SEC.gov."
3. *(Prediction & Replay, 15 s)* "At 9:00 ET the next morning, the text model gave 25.5% that the move would
   exceed 2.14%, against 24.2% without text. Reveal: the stock moved 0.99%, not a large move."
4. *(Research Results, 15 s)* "Across 467 out-of-time events, adding text changed the Brier score by +0.0006,
   with an interval from −0.004 to +0.006: no measurable gain. Market timing context did most of the work."
5. *(Methodology, 5 s)* "Every assumption, including a timestamp error we found in the SEC API, is documented here."

## Technical experiment summary

Primary protocol v1.0 frozen before modelling; two data fixes logged afterwards (parser fix before any model;
acceptance-time correction after the first results — the superseded run is reported). L2 logistic regression,
C ∈ {0.01…1.0} chosen by inner-validation log loss, Platt calibration on the preceding year, constant baseline
from matured training events. Primary: pooled paired ΔBrier with a quarter-cluster bootstrap (2,000 draws).
Company-cluster and uncalibrated variants, three ablations, a secondary boosting family, two alternative
thresholds and an earnings/no-earnings split were all pre-specified; none supports a text benefit, and the
no-earnings subgroup leans slightly against text (+0.0077, CI +0.0002 to +0.0165, unadjusted for
multiplicity). A label-shuffle check on development data and mutation tests show no leakage path. Re-runs are
bit-identical.

## Résumé-style description (verified work only)

Built FilingShock, a reproducible Python research application that tests whether changes in SEC 10-Q
disclosures improve forecasts of large next-session stock moves: point-in-time ingestion of SEC EDGAR filings
and financial facts (including detection and correction of an acceptance-timestamp error affecting 38% of
checked filings), deterministic MD&A extraction and paragraph alignment, leakage-tested feature construction,
calibrated temporal model comparison with clustered bootstrap inference, and a five-screen Streamlit interface.
Found no measurable incremental value of text features on 467 out-of-time events (ΔBrier +0.0006, 95% CI
−0.0039 to +0.0055).
