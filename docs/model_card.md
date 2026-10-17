# Model card

## Models

| Name | Inputs | Family |
|---|---|---|
| Constant | Event rate of matured events before the test year | — |
| A | 9 market + 3 event-context features | L2 logistic regression (primary); shallow gradient boosting (secondary) |
| B | A + 7 point-in-time financial features | same |
| **C** | B + 11 text-change features | same |
| Ablations | B + one text group (length, dictionary, paragraph change) | L2 logistic regression |

Feature definitions: `docs/feature_catalog.md`. Hyperparameter grid: 5 settings per family, chosen by
inner-validation log loss. Probabilities are Platt-calibrated on the year before each test year
(intercept-only fallback if the slope is not positive). No class weighting or resampling.

## Output

P(|open→close| of the next eligible session > 2 × the stock's 60-session mean |open→close|), evaluated at a
cutoff 30 minutes before the open. **Magnitude only**; no direction, no price target, no trading rule.

## Intended use

Research on whether disclosure-text changes add information beyond market and financial data, for a fixed
cohort of U.S. software companies, with explicit timing assumptions.

## Not intended for

Trading or investment decisions; companies outside the cohort without re-validation; causal claims about
disclosures; real-time use without the prospective information policy (observed timestamps).

## Evaluation

Expanding-window annual folds (fit ≤ Y−2, calibrate Y−1, test Y), pooled out-of-time predictions,
primary metric ΔBrier(C − B) with a quarter-clustered bootstrap. Numbers: `docs/results.md`, regenerated
from saved artifacts by `filingshock report`.

## Caveats

* Small sample: 467 out-of-time events (125 large moves); the primary 95% interval spans about ±0.005 Brier, so smaller effects cannot be resolved.
* Availability times are assumptions (acceptance + lag), not observed publication times.
* 176 of 573 final events (31%) coincide with an earnings release in the same pre-cutoff window; there the
  next session mainly reflects the release.
* Attribution shown in the interface is coefficient × standardized value for the logistic model (log-odds,
  before calibration), not SHAP and not causal.
