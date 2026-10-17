# Experiment results

Generated from `data/experiments/20261017T090148Z-primary/metrics.json` (dataset `ds-1698abec6cff`, code 0.1.0, git `44c99bd2`).

## Scope

Pooled out-of-time predictions, test years 2014–2026: 467 events, 14 companies, 125 large moves (prevalence 0.268), target sessions 2014-01-24 to 2026-09-23. Each event is scored once, by the model of its own test year (fit ≤ Y−2, calibrated on Y−1).

## Primary comparison (pre-specified)

**ΔBrier = Brier(C) − Brier(B) = +0.0006, 95% interval [-0.0039, +0.0055]** (cluster bootstrap over calendar quarters, 2000 draws).

Conclusion under the protocol's decision rule: **inconclusive: interval includes zero**.

## All models (calibrated probabilities)

| Model | Brier ↓ | Log loss ↓ | Avg. precision ↑ | ROC-AUC | Precision top 10% | Lift top 10% |
|---|---|---|---|---|---|---|
| Constant baseline | 0.1965 | 0.5822 | 0.259 | 0.488 | 0.213 (k=47) | 0.79× |
| A: market + context | 0.1856 | 0.5522 | 0.411 | 0.663 | 0.468 (k=47) | 1.75× |
| B: A + financial | 0.1898 | 0.5650 | 0.390 | 0.626 | 0.404 (k=47) | 1.51× |
| C: B + text (primary) | 0.1904 | 0.5652 | 0.382 | 0.626 | 0.468 (k=47) | 1.75× |
| B + text length | 0.1903 | 0.5653 | 0.386 | 0.626 | 0.383 (k=47) | 1.43× |
| B + dictionary features | 0.1907 | 0.5673 | 0.378 | 0.624 | 0.468 (k=47) | 1.75× |
| B + paragraph-change features | 0.1911 | 0.5692 | 0.382 | 0.616 | 0.404 (k=47) | 1.51× |
| A (shallow boosting) | 0.1930 | 0.5752 | 0.383 | 0.606 | 0.489 (k=47) | 1.83× |
| B (shallow boosting) | 0.1865 | 0.5617 | 0.445 | 0.645 | 0.511 (k=47) | 1.91× |
| C (shallow boosting) | 0.1882 | 0.5645 | 0.405 | 0.633 | 0.489 (k=47) | 1.83× |

Top-10% precision ranks a completed holdout retrospectively; it is not a deployable selection rule.

## Paired differences (first − second)

| Comparison | ΔBrier | 95% interval | ΔAvg. precision | Resampling unit |
|---|---|---|---|---|
| hgb_shallow:C vs hgb_shallow:B | +0.0017 | [-0.0078, +0.0108] | -0.040 | quarter |
| logistic_l2:A vs constant:constant | -0.0109 | [-0.0241, +0.0030] | +0.152 | quarter |
| logistic_l2:B vs logistic_l2:A | +0.0042 | [-0.0026, +0.0114] | -0.021 | quarter |
| logistic_l2:B+dictionary vs logistic_l2:B | +0.0010 | [-0.0028, +0.0049] | -0.012 | quarter |
| logistic_l2:B+length vs logistic_l2:B | +0.0005 | [-0.0011, +0.0023] | -0.004 | quarter |
| logistic_l2:B+paragraph_change vs logistic_l2:B | +0.0013 | [-0.0023, +0.0054] | -0.008 | quarter |
| logistic_l2:C vs constant:constant | -0.0061 | [-0.0184, +0.0069] | +0.123 | quarter |
| logistic_l2:C vs logistic_l2:B | +0.0006 | [-0.0039, +0.0055] | -0.008 | quarter |
| logistic_l2:C vs logistic_l2:B (company clusters) | +0.0006 | [-0.0034, +0.0050] | -0.008 | ticker |
| logistic_l2:C vs logistic_l2:B (uncalibrated) | -0.0017 | [-0.0078, +0.0044] | +0.004 | quarter |

## Per-year Brier (logistic models)

| Year | Events | Large moves | Constant | B | C | C − B |
|---|---|---|---|---|---|---|
| 2014 | 30 | 9 | 0.2113 | 0.2238 | 0.2243 | +0.0006 |
| 2015 | 33 | 5 | 0.1431 | 0.1424 | 0.1405 | -0.0018 |
| 2016 | 34 | 8 | 0.1801 | 0.1369 | 0.1607 | +0.0237 |
| 2017 | 38 | 10 | 0.1942 | 0.1924 | 0.1876 | -0.0048 |
| 2018 | 35 | 8 | 0.1767 | 0.1658 | 0.1594 | -0.0064 |
| 2019 | 41 | 11 | 0.1968 | 0.1749 | 0.1688 | -0.0061 |
| 2020 | 39 | 7 | 0.1521 | 0.1810 | 0.1979 | +0.0168 |
| 2021 | 36 | 10 | 0.2019 | 0.2100 | 0.2064 | -0.0036 |
| 2022 | 38 | 9 | 0.1808 | 0.1823 | 0.1790 | -0.0034 |
| 2023 | 41 | 14 | 0.2343 | 0.2156 | 0.2155 | -0.0001 |
| 2024 | 38 | 15 | 0.2591 | 0.2068 | 0.2044 | -0.0024 |
| 2025 | 36 | 10 | 0.2008 | 0.2193 | 0.2010 | -0.0182 |
| 2026 | 28 | 9 | 0.2214 | 0.2192 | 0.2384 | +0.0191 |

## Sensitivity analyses (pre-specified)

| Run | Events | Prevalence | ΔBrier (C − B) | 95% interval | Reading |
|---|---|---|---|---|---|
| sens-multiplier-1.5 | 467 | 0.379 | +0.0021 | [-0.0040, +0.0088] | inconclusive: interval includes zero |
| sens-multiplier-2.5 | 426 | 0.188 | +0.0002 | [-0.0044, +0.0049] | inconclusive: interval includes zero |
| sub-no-earnings-8k | 256 | 0.188 | +0.0077 | [+0.0002, +0.0165] | text worsened probability quality |
| sub-earnings-8k | 79 | 0.456 | +0.0010 | [-0.0339, +0.0362] | inconclusive: interval includes zero |

Readings apply the primary decision rule mechanically. Sensitivity and subgroup results are secondary, several were run, and none is adjusted for multiple comparisons.

## Falsification and pipeline checks

* Label shuffle (development data only): mean Brier 0.2087 vs 0.1972 for a constant — no spurious improvement.
* Design-matrix guard: only catalogued `mkt_/ctx_/fin_/txt_` columns may enter a model (enforced at fit time).
* Future-data mutation and target-close mutation: `tests/test_leakage.py`, `tests/test_target.py`.
* Events without a prediction: 106 of 573 (2010–2013 serve only as training and calibration data).

## Superseded run (before the acceptance-time correction)

`20261017T083659Z-primary` used submissions-API acceptance times, later found to be wrong by the Eastern UTC offset for some filings. It had 499 events and ΔBrier -0.0005 [-0.0066, +0.0054] (inconclusive: interval includes zero). It is kept for transparency and is not the primary result.

