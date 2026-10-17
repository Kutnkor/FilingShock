"""Temporal splits, train-only preprocessing, calibration isolation, metrics and common cohorts."""

import numpy as np
import pandas as pd
import pytest

from filingshock.modeling.evaluate import avg_precision, brier, cluster_bootstrap_diff, conclusion_from_ci, metric_block
from filingshock.modeling.models import PlattCalibrator, Preprocessor
from filingshock.modeling.splits import make_folds


def events(n_per_year=40, years=range(2010, 2020), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        days = pd.bdate_range(f"{y}-01-05", f"{y}-12-20")
        for k in range(n_per_year):
            rows.append({"event_id": f"e{y}{k:03d}", "target_session": days[rng.integers(len(days))],
                         "y": int(rng.random() < 0.3)})
    return pd.DataFrame(rows)


def test_fold_blocks_are_disjoint_and_ordered():
    df = events()
    folds = make_folds(df, 2014, 2019, 60, 10, 4)
    d = df.set_index("event_id")
    for f in folds:
        fit, cal, test = set(f.fit_ids), set(f.cal_ids), set(f.test_ids)
        assert not (fit & cal) and not (fit & test) and not (cal & test)
        assert d.loc[list(fit), "target_session"].max() < d.loc[list(cal), "target_session"].min()
        assert d.loc[list(cal), "target_session"].max() < d.loc[list(test), "target_session"].min()
        assert set(f.inner_train_ids) | set(f.inner_val_ids) == fit
        assert d.loc[f.inner_val_ids, "target_session"].dt.year.unique().tolist() == [f.test_year - 2]


def test_same_date_events_stay_in_one_block():
    df = events()
    df.loc[df.index[:5], "target_session"] = pd.Timestamp("2015-06-01")
    for f in make_folds(df, 2014, 2019, 60, 10, 4):
        ids = set(df.loc[df.index[:5], "event_id"])
        blocks = [set(f.fit_ids), set(f.cal_ids), set(f.test_ids)]
        assert sum(1 for b in blocks if ids & b) <= 1


def test_readiness_fails_when_calibration_block_has_one_class():
    df = events()
    df.loc[df["target_session"].dt.year == 2015, "y"] = 0
    f2016 = [f for f in make_folds(df, 2016, 2016, 60, 10, 4)][0]
    assert not f2016.ready and any("calibration" in r for r in f2016.readiness_failures)


def test_preprocessor_is_fit_on_training_rows_only():
    tr = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, 10.0, 10.0, 10.0]})
    te = pd.DataFrame({"a": [100.0, np.nan], "b": [0.0, 20.0]})
    pre = Preprocessor(["a", "b"]).fit(tr)
    med = pre.medians["a"]
    out = pre.transform(te)
    assert pre.medians["a"] == med == 2.0                     # unchanged by test data
    assert out["a__missing"].tolist() == [0.0, 1.0]
    assert out["a"].iloc[0] == (pre.hi["a"] - pre.mean["a"]) / pre.std["a"]   # winsorised at train bounds


def test_platt_uses_only_calibration_inputs_and_falls_back():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.9, 400)
    y = (rng.uniform(size=400) < p).astype(int)
    cal = PlattCalibrator().fit(p, y)
    assert cal.method == "platt" and cal.b > 0
    y_anti = (rng.uniform(size=400) < 1 - p).astype(int)
    cal2 = PlattCalibrator().fit(p, y_anti)
    assert cal2.method == "intercept_only_fallback" and cal2.b == 1.0
    q = np.array([0.1, 0.5, 0.9])
    assert np.all(np.diff(cal2.transform(q)) > 0)             # ranking preserved


def test_metrics_and_undefined_values():
    assert brier([0, 1], [0.0, 1.0]) == 0.0
    ap, reason = avg_precision([0, 0, 0], [0.1, 0.2, 0.3])
    assert ap is None and "no positive" in reason
    mb = metric_block([0, 0, 0], [0.1, 0.2, 0.3])
    assert mb["roc_auc"] is None and mb["roc_auc_unavailable_reason"]


def test_paired_bootstrap_on_identical_models_is_zero():
    df = pd.DataFrame({"y": [0, 1] * 50, "a": np.linspace(0.1, 0.9, 100), "q": np.repeat(range(10), 10)})
    df["b"] = df["a"]
    r = cluster_bootstrap_diff(df, "a", "b", "q", reps=50, seed=0)
    assert r["brier"]["estimate"] == 0.0 and r["brier"]["ci95"] == [0.0, 0.0]


@pytest.mark.parametrize("ci,word", [([-0.02, -0.001], "supported"), ([-0.01, 0.01], "inconclusive"),
                                     ([0.001, 0.02], "worsened")])
def test_decision_rule(ci, word):
    assert word in conclusion_from_ci(ci)
