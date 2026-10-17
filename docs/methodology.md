# Methodology

This page summarises how FilingShock answers its question. The binding specification is the
[research protocol](research_protocol.md); timing details are in the [timing policy](timing_policy.md).

1. **Discovery.** SEC submissions history for 21 pre-specified SIC 7372 companies; acceptance times are
   reconciled against each filing's EDGAR index page.
2. **Events.** Original 10-Qs accepted after the previous regular close and assumed available (acceptance
   + 15 min, or next-day 06:00 ET + 15 min after 17:30 ET) by 30 minutes before the next open. One event
   per company and target session.
3. **Target.** y = 1 if |close/open − 1| in the target session exceeds 2 × the mean |open-to-close| of the
   previous 60 valid sessions.
4. **Text.** MD&A extracted deterministically (longest span after a standalone Item 2 heading); the
   prior-year same-quarter 10-Q is the reference; paragraphs aligned one-to-one with explainable scores.
   Eleven interpretable features, no learned vocabulary; Loughran–McDonald lists applied point-in-time.
5. **Numbers.** Market and context features from sessions and filings before the cutoff; financial
   features from Company Facts restricted to filings available by the cutoff.
6. **Models.** Constant, A, B, C (+ ablations) with identical folds, preprocessing, tuning budget and
   calibration; L2 logistic regression primary, shallow boosting secondary.
7. **Evaluation.** Annual expanding window (fit ≤ Y−2, calibrate Y−1, test Y), pooled out-of-time
   predictions, paired ΔBrier(C − B) with a quarter-clustered bootstrap; decision rule fixed in advance.
8. **Checks.** Leakage tests (future-data and target-close mutation), design-matrix guard, label shuffle
   on development data, artifact reconciliation, determinism re-run.

See also: [feature catalog](feature_catalog.md) · [results](results.md) · [data card](data_card.md) ·
[model card](model_card.md) · [extraction audit](extraction_audit.md) · [limitations](limitations.md).
