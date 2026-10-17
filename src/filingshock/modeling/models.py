"""Model families, train-only preprocessing and Platt calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

EPS = 1e-6
ALLOWED_FEATURE_PREFIXES = ("mkt_", "ctx_", "fin_", "txt_")
FORBIDDEN_FEATURE_NAMES = {"y", "event_id", "abs_r", "r_open_to_close", "threshold_abs_move", "open", "close",
                           "m_mean_abs_oc_60", "abs_ratio", "outcome_available_utc", "target_session"}
FORBIDDEN_FEATURE_FRAGMENTS = ("target", "outcome", "realized", "threshold")


def assert_admissible_columns(columns: list[str], catalog: set[str]) -> None:
    """Pipeline guard: the design matrix may contain only catalogued features (plus their missing
    indicators), never identifiers or outcome-derived fields."""
    for c in columns:
        base = c[:-len("__missing")] if c.endswith("__missing") else c
        if base not in catalog:
            raise ValueError(f"column {c!r} is not in the feature catalog")
        if (base in FORBIDDEN_FEATURE_NAMES or not base.startswith(ALLOWED_FEATURE_PREFIXES)
                or any(tok in base for tok in FORBIDDEN_FEATURE_FRAGMENTS)):
            raise ValueError(f"column {c!r} looks like an identifier or outcome field")


@dataclass
class Preprocessor:
    """Median imputation + missingness indicators + 1/99 winsorisation + standardisation,
    all estimated on the training rows only."""
    features: list[str]
    medians: dict = field(default_factory=dict)
    lo: dict = field(default_factory=dict)
    hi: dict = field(default_factory=dict)
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)
    indicator_for: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        for c in self.features:
            col = pd.to_numeric(X[c], errors="coerce").astype(float)
            if col.notna().sum() == 0:
                self.dropped.append(c)
                continue
            self.medians[c] = float(col.median())
            if col.isna().any():
                self.indicator_for.append(c)
            filled = col.fillna(self.medians[c])
            self.lo[c], self.hi[c] = float(filled.quantile(0.01)), float(filled.quantile(0.99))
            w = filled.clip(self.lo[c], self.hi[c])
            self.mean[c], self.std[c] = float(w.mean()), float(w.std(ddof=0)) or 1.0
        return self

    @property
    def output_columns(self) -> list[str]:
        return [c for c in self.features if c not in self.dropped] + [f"{c}__missing" for c in self.indicator_for]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.features:
            if c in self.dropped:
                continue
            col = pd.to_numeric(X[c], errors="coerce").astype(float)
            w = col.fillna(self.medians[c]).clip(self.lo[c], self.hi[c])
            out[c] = (w - self.mean[c]) / self.std[c]
        for c in self.indicator_for:
            out[f"{c}__missing"] = pd.to_numeric(X[c], errors="coerce").isna().astype(float)
        return pd.DataFrame(out, index=X.index)[self.output_columns]


def make_estimator(family: str, param, seed: int):
    if family == "logistic_l2":
        return LogisticRegression(C=float(param), solver="lbfgs", max_iter=5000)  # L2 is the default penalty
    if family == "hgb_shallow":
        depth, lr = param
        return HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr, max_iter=150,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    raise ValueError(family)


def param_grid(family: str, cfg_model: dict) -> list:
    if family == "logistic_l2":
        return list(cfg_model["c_grid"])
    if family == "hgb_shallow":   # same budget (5 settings) as the logistic grid
        return [(1, 0.03), (2, 0.03), (2, 0.1), (3, 0.03), (3, 0.1)]
    raise ValueError(family)


@dataclass
class FittedModel:
    family: str
    param: object
    pre: Preprocessor
    est: object

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.est.predict_proba(self.pre.transform(X))[:, 1]

    def coefficients(self) -> dict | None:
        if self.family != "logistic_l2":
            return None
        return {"intercept": float(self.est.intercept_[0]),
                **{c: float(v) for c, v in zip(self.pre.output_columns, self.est.coef_[0])}}


def fit_model(family: str, param, X: pd.DataFrame, y: np.ndarray, features: list[str], seed: int) -> FittedModel:
    pre = Preprocessor(features).fit(X)
    est = make_estimator(family, param, seed)
    est.fit(pre.transform(X), y)
    return FittedModel(family, param, pre, est)


def select_param(family: str, grid: list, X_tr, y_tr, X_va, y_va, features, seed) -> tuple[object, list[dict]]:
    """Choose the hyperparameter with the lowest validation log loss. Every attempt is logged."""
    attempts = []
    for p in grid:
        m = fit_model(family, p, X_tr, y_tr, features, seed)
        pv = np.clip(m.predict(X_va), EPS, 1 - EPS)
        attempts.append({"param": p if not isinstance(p, tuple) else list(p),
                         "val_log_loss": float(log_loss(y_va, pv, labels=[0, 1])),
                         "val_brier": float(np.mean((pv - y_va) ** 2))})
    best = min(range(len(grid)), key=lambda i: (attempts[i]["val_log_loss"], i))
    return grid[best], attempts


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    a: float = 0.0
    b: float = 1.0
    method: str = "identity"

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        z = _logit(p_raw).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(z, y)
        b = float(lr.coef_[0, 0])
        if b > 0:
            self.a, self.b, self.method = float(lr.intercept_[0]), b, "platt"
        else:
            # pre-specified fallback: calibration-in-the-large (slope fixed at 1) keeps ranking intact;
            # the intercept a in logit(p') = a + logit(p) is found by a 1-D grid search on log loss
            grid = np.linspace(-5, 5, 2001)
            ll = [log_loss(y, 1 / (1 + np.exp(-(a + z.ravel()))), labels=[0, 1]) for a in grid]
            self.a, self.b, self.method = float(grid[int(np.argmin(ll))]), 1.0, "intercept_only_fallback"
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a + self.b * _logit(p_raw))))
