"""Evaluation metrics, paired differences and dependence-aware bootstrap intervals.

Undefined metrics are returned as None together with a reason; they are never replaced by zero.
Average Precision uses sklearn.metrics.average_precision_score (step-wise; NOT trapezoidal PR-AUC).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-15


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def logloss(y, p) -> float:
    y, p = np.asarray(y, float), _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def avg_precision(y, p):
    y = np.asarray(y)
    if y.sum() == 0:
        return None, "no positive events"
    return float(average_precision_score(y, p)), None


def roc_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None, "only one class present"
    return float(roc_auc_score(y, p)), None


def precision_at(y, p, frac: float, order_key=None):
    """Precision among the top `frac` share of events by score (ties broken by event order)."""
    y, p = np.asarray(y), np.asarray(p, float)
    n = len(y)
    k = max(int(math.ceil(frac * n)), 1)
    idx = np.lexsort((np.arange(n) if order_key is None else order_key, -p))[:k]
    return float(y[idx].mean()), k


def metric_block(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    prev = float(y.mean()) if len(y) else None
    ap, ap_reason = avg_precision(y, p)
    auc, auc_reason = roc_auc(y, p)
    p10, k10 = precision_at(y, p, 0.10)
    p20, k20 = precision_at(y, p, 0.20)
    return {
        "n_events": int(len(y)), "n_positive": int(y.sum()), "prevalence": prev,
        "brier": brier(y, p), "log_loss": logloss(y, p),
        "average_precision": ap, "average_precision_unavailable_reason": ap_reason,
        "roc_auc": auc, "roc_auc_unavailable_reason": auc_reason,
        "precision_top10pct": p10, "k_top10pct": k10, "lift_top10pct": (p10 / prev) if prev else None,
        "precision_top20pct": p20, "k_top20pct": k20, "lift_top20pct": (p20 / prev) if prev else None,
        "mean_predicted": float(p.mean()) if len(p) else None,
    }


def calibration_bins(y, p, n_bins: int = 8) -> list[dict]:
    """Quantile bins with support counts (sparse bins are reported, not hidden)."""
    df = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)})
    if df.empty:
        return []
    q = min(n_bins, max(1, df["p"].nunique()))
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q, labels=False)
    out = []
    for b, g in df.groupby("bin"):
        out.append({"bin": int(b), "n": int(len(g)), "mean_predicted": float(g["p"].mean()),
                    "observed_rate": float(g["y"].mean()), "p_min": float(g["p"].min()), "p_max": float(g["p"].max())})
    return out


def cluster_bootstrap_diff(df: pd.DataFrame, model_a: str, model_b: str, cluster_col: str,
                           reps: int, seed: int) -> dict:
    """Paired difference metric(a) - metric(b) on the same events; clusters resampled with replacement.
    df has columns event_id, y, cluster_col and one probability column per model."""
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    groups = {c: g.index.to_numpy() for c, g in df.groupby(cluster_col)}
    y = df["y"].to_numpy().astype(int)
    pa, pb = df[model_a].to_numpy(), df[model_b].to_numpy()

    def diffs(idx):
        yy = y[idx]
        out = {"brier": brier(yy, pa[idx]) - brier(yy, pb[idx]),
               "log_loss": logloss(yy, pa[idx]) - logloss(yy, pb[idx])}
        if yy.sum() > 0:
            out["average_precision"] = (average_precision_score(yy, pa[idx]) - average_precision_score(yy, pb[idx]))
        return out

    point = diffs(np.arange(len(df)))
    draws: dict[str, list[float]] = {k: [] for k in ("brier", "log_loss", "average_precision")}
    for _ in range(reps):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        d = diffs(idx)
        for k, v in d.items():
            draws[k].append(v)
    res = {}
    for k, v in point.items():
        arr = np.asarray(draws[k])
        res[k] = {"estimate": float(v),
                  "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if len(arr) else None,
                  "share_draws_below_zero": float((arr < 0).mean()) if len(arr) else None,
                  "valid_draws": int(len(arr))}
    res["cluster"] = cluster_col
    res["n_clusters"] = int(len(clusters))
    res["reps"] = reps
    res["convention"] = f"metric({model_a}) - metric({model_b}); negative Brier/log-loss favours {model_a}"
    return res


def conclusion_from_ci(ci: list[float] | None) -> str:
    """Pre-specified decision rule on the paired Delta-Brier (C - B) 95% interval."""
    if ci is None:
        return "unavailable"
    lo, hi = ci
    if hi < 0:
        return "supported: text improved probability quality"
    if lo > 0:
        return "text worsened probability quality"
    return "inconclusive: interval includes zero"
