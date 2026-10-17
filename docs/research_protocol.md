# FilingShock research protocol

**Version 1.0 — frozen 2026-10-17, before any model was fitted or any out-of-time prediction was
inspected.** This protocol was written inside the project repository; it was **not** registered with an
external registry, so it is a versioned internal protocol, not a formal preregistration.

What had been looked at when this version was frozen: filing discovery counts, timing-classification
counts, price-history availability, extraction success rates, reference availability and per-year event
counts (needed for readiness checks). No feature–outcome relationship and no model score had been
computed.

Changes made after this freeze are logged in the [change log](#change-log) with their reason and whether
results had been seen.

## 1. Question

Given a company's historical market behaviour and financial fundamentals, do changes in its 10-Q
MD&A (and risk-factor) language provide incremental predictive information about an unusually large
open-to-close move in the next eligible regular session?

The study is filing-conditioned and predictive. It does not estimate the causal effect of a filing,
forecast the sign of the move, or evaluate trading profitability.

## 2. Universe and selection rule

A pre-specified, fixed cohort of established U.S. companies that file 10-Qs and are classified by the
SEC under SIC 7372 (Services-Prepackaged Software): ADBE, INTU, ADSK, ORCL, CRM, MSFT, CDNS, SNPS, PTC,
TYL, NOW, CVLT, GEN, MANH, plus companies later acquired or delisted (ANSS, CTXS, VMW, COUP, SPLK, NUAN,
CA). The list was chosen by the author from general knowledge of the sector, without looking at any
outcome. It is **not** a historical index-membership universe: survivors are over-represented, and the
acquired companies drop out at the price stage because the free provider has no history for them
(documented survivorship limitation; a CSV import path accepts licensed data for them).

Pilot (pipeline validation only, not evidence): ADBE, INTU, ADSK, ORCL, CRM; 50 10-Qs spread evenly
over 2010–2026 (`data/interim/pilot_sample.csv`).

## 3. Events and eligibility

* Filing window: EDGAR acceptance 2010-01-01 to 2026-09-30.
* Primary form: original 10-Q. 10-K, 10-Q/A and 10-K/A are excluded from the primary study but kept in
  the funnel.
* Timing (docs/timing_policy.md): accepted at or after the previous regular-session close and assumed
  available (acceptance + 15 min, or next-day 06:00 ET dissemination + 15 min for after-17:30 filings) no
  later than the decision cutoff = target open − 30 minutes.
* Uniqueness: one event per (company, target session); the earliest-accepted original 10-Q is kept and
  any other is excluded as `duplicate_company_target_session`.
* Price validity: target-session bar present with positive finite open/close consistent with its
  high/low; at least 60 valid sessions within the previous 90 sessions; mean |open-to-close| ≥ 0.1 %;
  |r| > 50 % is treated as a probable data error (bar invalid), never silently used.
* Text: MD&A extracted from the current filing; primary reference (prior-year same-quarter 10-Q, period
  end 364 ± 22 days earlier, accepted before the current filing) found and its MD&A extracted.
* Missing financial facts never exclude an event (imputed with missingness indicators).

The main comparison uses the **common event set** that passes every stage. Events lost only at the
text stages are counted in the funnel; their count is reported.

## 4. Target (fixed before evaluation)

r = close/open − 1 for the target session; a = |r|; m = mean |open-to-close| of the previous 60 valid
sessions (all before the cutoff); **y = 1 if a > 2 × m**. The lookback (60) and multiplier (2.0) are
fixed. Alternatives (multiplier 1.5 and 2.5) are sensitivity analyses only.

## 5. Features (fs-features-1.0)

* **Market (9)**: mean |oc| over 20 and 60 sessions, 60-session close-to-close volatility, frequency of
  earlier large moves (250 sessions, each judged against its own prior-60 baseline), log 5/60-session
  volume ratio, previous-session |return|, SPY and IGV 20-session mean |oc|, |r|/m of the company's
  previous event whose outcome had matured.
* **Event context (3)**: Item 2.02 8-K accepted in the same pre-cutoff window, log days since the last
  admissible Item 2.02 8-K, log filing lag (period end → acceptance).
* **Financial (7)**, point-in-time from Company Facts: quarterly revenue YoY growth, operating margin,
  YoY operating-margin change (percentage points), operating-cash-flow margin (quarterly, YTD
  differences allowed only with matching fiscal-year start), cash/assets, liabilities/assets
  (1 − equity/assets), YoY change in liabilities/assets. Debt-specific tags were not used because
  their concepts change across companies and years.
* **Text (11)**, no learned vocabulary:
  * length: log MD&A word-count ratio (current/reference);
  * dictionary: Loughran–McDonald uncertainty and negative rates per 1,000 words (point-in-time
    membership by year flag) and their changes vs the reference; share of added or substantially edited
    paragraphs mentioning a topic from `topics-1.0`;
  * paragraph change: added share, removed share, substantially-edited share, lexical cosine distance,
    risk-factor added share (defined only when both risk sections are fully present; otherwise missing).

Model A = market + context; B = A + financial; **C = B + text**.

## 6. Models, selection and calibration

* Primary family: L2 logistic regression, natural class frequencies (no weighting or resampling).
* Secondary family (A, B, C only): shallow histogram gradient boosting, depth ≤ 3.
* Preprocessing fitted on the fit block only: median imputation + missingness indicators,
  1st/99th-percentile winsorisation, standardisation.
* Hyperparameter budget: 5 settings per family (logistic C ∈ {0.01, 0.03, 0.1, 0.3, 1.0}); chosen by
  log loss on the inner-validation year; identical for every model.
* Calibration: Platt scaling fitted on the calibration year using predictions of the model fitted on the
  fit block. If the fitted slope is ≤ 0, an intercept-only recalibration (slope fixed at 1) is used and
  the method is recorded.
* Constant baseline: event rate of all matured events before the test year.

## 7. Temporal design

Annual expanding window, test years 2014–2026 subject to readiness. For test year Y: fit block = years
≤ Y−2 (inner selection: train ≤ Y−3, validate Y−2), calibration = Y−1, test = Y. Readiness per fold:
≥ 60 fit events, ≥ 10 of each class in the fit block, ≥ 4 of each class in the calibration block,
both classes present in inner train and inner validation. A fold failing readiness produces no
predictions for any model; its events are reported as "no out-of-time prediction".

## 8. Primary comparison and decision rule

**ΔBrier = Brier(C) − Brier(B)** on paired, calibrated, pooled out-of-time predictions for the same
events; negative favours text. 95 % interval: cluster bootstrap over calendar quarters (2,000 draws).

* Upper bound < 0 → "supported: text improved probability quality".
* Lower bound > 0 → "text worsened probability quality".
* Otherwise → "inconclusive".

Secondary: ΔAverage Precision, Δlog loss, calibration curves, precision and lift in the top 10 % / 20 %
(retrospective ranking diagnostics, not trading rules), ROC-AUC, per-year Brier, company-cluster
bootstrap, uncalibrated ΔBrier.

## 9. Ablations, sensitivity and falsification

* Ablations: B + length, B + dictionary, B + paragraph change, C (full).
* Sensitivity: multiplier 1.5 and 2.5; subgroup of events without an earnings 8-K in the window (the
  10-Q arrives separately from the earnings release) and its complement; secondary model family.
* Falsification (pipeline diagnostics): label shuffle on development data only; design-matrix column
  whitelist; future-data mutation invariance and target-close mutation tests (test suite).

## 10. What is not claimed

No causal effect, no direction forecast, no profitability, no claim that the fixed cohort represents
the software sector, and no claim that historical availability was observed (it is a proxy).

## Change log

| Date | Version | Change | Results seen? |
|---|---|---|---|
| 2026-10-17 | 1.0 | Initial freeze before model fitting. | No |
| 2026-10-17 | 1.0 (data fix 1) | Parser fs-parser-1.1: accept a 10-Q MD&A heading mislabelled "Item 7" (one Salesforce filing) and drop running page headers ("PART I Item 2"). Found in the manual extraction audit. Final sample 636 → 637 events. | No (no model had been fitted) |
| 2026-10-17 | 1.0 (data fix 2) | **Acceptance times corrected.** The pre-planned timestamp reconciliation found that the SEC submissions API's `acceptanceDateTime` is later than the filing's own EDGAR index page by exactly the Eastern UTC offset (+4 h EDT / +5 h EST) for 1,068 of 2,834 checked filings (38 %), across all years and forms. This can make a filing accepted during trading look like an after-close filing. All timing now uses index-page acceptance times; the event set, features and all experiments were rebuilt with the protocol otherwise unchanged. The superseded primary run (`20261017T083659Z-primary`, and its first identical run) is kept and reported. | **Yes** — the first primary and sensitivity results had been seen. The correction fixes a data error; no analysis choice was changed. |
| 2026-10-17 | Prospective operational extension | Record Model A and the matured training-prevalence constant beside B/C for future forecasts. Optional baselines use paired earliest-forecast events with explicit coverage; older records are not backfilled. Historical runs and the primary comparison are unchanged. | Yes — motivated by the published historical comparison; no prospective outcomes had been collected. |
