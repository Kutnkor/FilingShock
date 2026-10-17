"""Expanding-window annual splits with separate fit, selection, calibration and test blocks.

For test year Y:
    fit block          events with target session in years <= Y-2
    inner train / val  years <= Y-3 / year Y-2 (hyperparameter selection only)
    calibration block  year Y-1 (Platt scaling only)
    test block         year Y
All outcomes used for fitting or calibration matured before the first test event's cutoff,
because blocks are separated by calendar-year boundaries and each block ends with a session close.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class Fold:
    test_year: int
    fit_ids: list[str]
    inner_train_ids: list[str]
    inner_val_ids: list[str]
    cal_ids: list[str]
    test_ids: list[str]
    ready: bool = True
    readiness_failures: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def describe(self) -> dict:
        d = asdict(self)
        for k in ("fit_ids", "inner_train_ids", "inner_val_ids", "cal_ids", "test_ids"):
            d[k] = len(d[k])
        d["fit_years"] = f"<= {self.test_year - 2}"
        d["calibration_year"] = self.test_year - 1
        d["inner_validation_year"] = self.test_year - 2
        return d


def make_fold(df: pd.DataFrame, y: int, min_train_events: int, min_train_positives: int,
              min_cal_positives: int) -> Fold:
    """The fold for test year y, with its readiness checks. Readiness never looks at the test block,
    so the same fold also defines the production model for prospective forecasts in year y."""
    d = df.assign(year=pd.to_datetime(df["target_session"]).dt.year)
    fit = d[d["year"] <= y - 2]
    cal = d[d["year"] == y - 1]
    test = d[d["year"] == y]
    f = Fold(y, list(fit["event_id"]), list(fit[fit["year"] <= y - 3]["event_id"]),
             list(fit[fit["year"] == y - 2]["event_id"]), list(cal["event_id"]), list(test["event_id"]))
    pos_fit, pos_cal = int(fit["y"].sum()), int(cal["y"].sum())
    f.counts = {"fit_events": len(fit), "fit_positives": pos_fit, "cal_events": len(cal),
                "cal_positives": pos_cal, "test_events": len(test), "test_positives": int(test["y"].sum()),
                "inner_train_events": len(f.inner_train_ids), "inner_val_events": len(f.inner_val_ids)}
    if len(fit) < min_train_events:
        f.readiness_failures.append(f"fit_events<{min_train_events}")
    if pos_fit < min_train_positives or (len(fit) - pos_fit) < min_train_positives:
        f.readiness_failures.append(f"fit_class_count<{min_train_positives}")
    if pos_cal < min_cal_positives or (len(cal) - pos_cal) < min_cal_positives:
        f.readiness_failures.append(f"calibration_class_count<{min_cal_positives}")
    inner_train = fit[fit["year"] <= y - 3]
    if inner_train["y"].nunique() < 2 or fit[fit["year"] == y - 2]["y"].nunique() < 2:
        f.readiness_failures.append("inner_selection_single_class")
    f.ready = not f.readiness_failures
    return f


def make_folds(df: pd.DataFrame, first_test_year: int, last_test_year: int, min_train_events: int,
               min_train_positives: int, min_cal_positives: int) -> list[Fold]:
    """df: event_id, target_session, y (only the final common-event sample)."""
    folds = [make_fold(df, y, min_train_events, min_train_positives, min_cal_positives)
             for y in range(first_test_year, last_test_year + 1)]
    return [f for f in folds if f.test_ids]
