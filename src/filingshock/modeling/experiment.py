"""Experiment runner: fair A/B/C comparison on one common event set with temporal folds.

Artifacts (written to a temporary directory and atomically renamed, then marked COMPLETE):
    manifest.json      run id, code/data/config versions, folds, seeds, parameters, calibration
    predictions.parquet canonical per-event out-of-time predictions (metrics are recomputable)
    metrics.json       pooled, per-year, calibration, paired differences with bootstrap intervals
    selection_log.json every hyperparameter-selection attempt
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from filingshock import __version__
from filingshock.config import PROJECT_ROOT, Config
from filingshock.events import feature_groups
from filingshock.modeling.evaluate import calibration_bins, cluster_bootstrap_diff, conclusion_from_ci, metric_block
from filingshock.modeling.models import (
    FittedModel,
    PlattCalibrator,
    assert_admissible_columns,
    fit_model,
    param_grid,
    select_param,
)
from filingshock.modeling.splits import make_folds
from filingshock.storage import DataLayout, read_json, write_json_atomic

log = logging.getLogger(__name__)

PREDICTION_CATEGORY = "retrospective_out_of_time"


def model_specs() -> dict[str, list[str]]:
    g = feature_groups()
    a = g["market"] + g["context"]
    b = a + g["financial"]
    return {
        "A": a, "B": b, "C": b + g["text"],
        "B+length": b + g["text_length"], "B+dictionary": b + g["text_dictionary"],
        "B+paragraph_change": b + g["text_paragraph_change"],
    }


PRIMARY_FAMILY = "logistic_l2"
SECONDARY_FAMILY = "hgb_shallow"
SECONDARY_MODELS = ["A", "B", "C"]


def _jsonable_param(p):
    return list(p) if isinstance(p, tuple) else p


@dataclass
class CalibratedModel:
    """One model of one fold: hyperparameter chosen on the inner split, fitted on the fit block,
    Platt-calibrated on the calibration block."""
    model: str
    family: str
    features: list[str]
    param: object
    attempts: list[dict]
    fitted: FittedModel
    platt: PlattCalibrator

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p_raw = self.fitted.predict(X)
        return p_raw, self.platt.transform(p_raw)

    def parameters(self) -> dict:
        pre = self.fitted.pre
        return {"param": _jsonable_param(self.param),
                "calibration": {"method": self.platt.method, "a": self.platt.a, "b": self.platt.b},
                "coefficients": self.fitted.coefficients(),
                "imputation_medians": pre.medians, "missing_indicators": pre.indicator_for,
                "winsor_bounds": {c: [pre.lo[c], pre.hi[c]] for c in pre.lo},
                "standardization": {c: [pre.mean[c], pre.std[c]] for c in pre.mean},
                "dropped_all_missing": pre.dropped}


def fit_calibrated_model(model: str, family: str, features: list[str], inner_train: pd.DataFrame,
                         inner_val: pd.DataFrame, fit: pd.DataFrame, cal: pd.DataFrame, mcfg: dict, seed: int,
                         catalog: set[str]) -> CalibratedModel:
    """Shared by the temporal experiment and the production model for prospective forecasts."""
    best, attempts = select_param(family, param_grid(family, mcfg), inner_train, inner_train["y"].to_numpy(),
                                  inner_val, inner_val["y"].to_numpy(), features, seed)
    fitted = fit_model(family, best, fit, fit["y"].to_numpy(), features, seed)
    assert_admissible_columns(fitted.pre.output_columns, catalog)
    platt = PlattCalibrator().fit(fitted.predict(cal), cal["y"].to_numpy())
    return CalibratedModel(model, family, features, best, attempts, fitted, platt)


def load_dataset(layout: DataLayout) -> tuple[pd.DataFrame, dict]:
    ev = pd.read_parquet(layout.processed / "events.parquet")
    feats = pd.read_parquet(layout.processed / "features.parquet")
    outc = pd.read_parquet(layout.processed / "outcomes.parquet")
    manifest = read_json(layout.processed / "dataset_manifest.json")
    final = ev[ev["eligibility_status"] == "eligible"][["event_id", "ticker", "cik", "target_session",
                                                         "decision_cutoff_utc", "acceptance_utc"]]
    df = final.merge(feats.drop(columns=["decision_cutoff_utc"]), on="event_id", how="left")
    df = df.merge(outc[["event_id", "y", "abs_r", "threshold_abs_move", "outcome_available_utc"]], on="event_id", how="left")
    if df["event_id"].duplicated().any():
        raise ValueError("duplicate event ids in the experiment dataset")
    if df["y"].isna().any():
        raise ValueError("final-sample events without a realized outcome")
    df["y"] = df["y"].astype(int)
    df["target_session"] = pd.to_datetime(df["target_session"])
    df = df.sort_values(["decision_cutoff_utc", "event_id"]).reset_index(drop=True)
    return df, manifest


def _git_state() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        st = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return {"commit": None, "note": "not a git repository"}
        return {"commit": rev.stdout.strip(), "dirty_files": [ln for ln in st.stdout.splitlines() if ln][:200]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "note": str(exc)}


def run_experiment(cfg: Config, layout: DataLayout, run_name: str = "primary", subset: str | None = None,
                   outcome_override: pd.DataFrame | None = None, include_secondary: bool = True,
                   reps: int | None = None, note: str | None = None) -> Path:
    df, data_manifest = load_dataset(layout)
    if outcome_override is not None:
        df = df.drop(columns=["y"]).merge(outcome_override[["event_id", "y"]], on="event_id", how="inner")
        df["y"] = df["y"].astype(int)
    if subset == "no_earnings_8k_in_window":
        df = df[df["ctx_earnings_8k_in_window"] == 0].reset_index(drop=True)
    elif subset == "earnings_8k_in_window":
        df = df[df["ctx_earnings_8k_in_window"] == 1].reset_index(drop=True)
    elif subset is not None:
        raise ValueError(subset)
    mcfg, scfg = cfg["model"], cfg["splits"]
    seed = int(mcfg["random_seed"])
    reps = int(reps or mcfg["bootstrap_reps"])
    specs = model_specs()
    catalog = set(c for cols in specs.values() for c in cols)
    folds = make_folds(df[["event_id", "target_session", "y"]], int(scfg["first_test_year"]),
                       int(scfg["last_test_year"]), int(scfg["min_train_events"]), int(scfg["min_train_positives"]),
                       int(scfg["min_calibration_positives"]))
    generated_at = datetime.now(timezone.utc)
    run_id = f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}-{run_name}"
    idx = df.set_index("event_id")
    rows, selection_log, fold_params, label_shuffle = [], [], [], []
    jobs = [(m, PRIMARY_FAMILY) for m in specs] + ([(m, SECONDARY_FAMILY) for m in SECONDARY_MODELS] if include_secondary else [])
    for fold in folds:
        if not fold.ready:
            continue
        fit, itr, iva, cal, test = (idx.loc[fold.fit_ids], idx.loc[fold.inner_train_ids], idx.loc[fold.inner_val_ids],
                                    idx.loc[fold.cal_ids], idx.loc[fold.test_ids])
        # sanity: every training/calibration outcome matured before the first test cutoff
        first_test_cutoff = test["decision_cutoff_utc"].min()
        assert fit["outcome_available_utc"].max() <= first_test_cutoff
        assert cal["outcome_available_utc"].max() <= first_test_cutoff
        prior = pd.concat([fit, cal])
        base_rate = float(prior["y"].mean())
        for eid in test.index:
            rows.append({"event_id": eid, "test_year": fold.test_year, "model": "constant", "family": "constant",
                         "p_raw": base_rate, "p_cal": base_rate, "calibration_method": "none (training prevalence)",
                         "param": None})
        for model, family in jobs:
            cm = fit_calibrated_model(model, family, specs[model], itr, iva, fit, cal, mcfg, seed, catalog)
            selection_log.append({"test_year": fold.test_year, "model": model, "family": family,
                                  "chosen": _jsonable_param(cm.param), "attempts": cm.attempts})
            p_raw, p_cal = cm.predict(test)
            fold_params.append({"test_year": fold.test_year, "model": model, "family": family, **cm.parameters()})
            for eid, pr, pc in zip(test.index, p_raw, p_cal):
                rows.append({"event_id": eid, "test_year": fold.test_year, "model": model, "family": family,
                             "p_raw": float(pr), "p_cal": float(pc), "calibration_method": cm.platt.method,
                             "param": json.dumps(_jsonable_param(cm.param))})
            if model == "C" and family == PRIMARY_FAMILY:
                # development-only falsification check: shuffled inner-train labels, inner-validation block
                rng = np.random.default_rng(seed + fold.test_year)
                ys = rng.permutation(itr["y"].to_numpy())
                m_sh = fit_model(family, cm.param, itr, ys, cm.features, seed)
                pv = m_sh.predict(iva)
                yv = iva["y"].to_numpy()
                label_shuffle.append({"test_year": fold.test_year, "inner_val_year": fold.test_year - 2,
                                      "brier_shuffled": float(np.mean((pv - yv) ** 2)),
                                      "brier_constant": float(np.mean((itr["y"].mean() - yv) ** 2)),
                                      "n_val": int(len(yv))})
    preds = pd.DataFrame(rows)
    meta_cols = ["event_id", "ticker", "target_session", "decision_cutoff_utc", "y", "abs_r", "threshold_abs_move",
                 "outcome_available_utc", "ctx_earnings_8k_in_window"]
    preds = preds.merge(df[meta_cols], on="event_id", how="left")
    preds["quarter"] = preds["target_session"].dt.to_period("Q").astype(str)
    preds["prediction_category"] = PREDICTION_CATEGORY
    preds["generated_at_utc"] = generated_at.isoformat(timespec="seconds")
    preds["simulated_cutoff_utc"] = preds["decision_cutoff_utc"]
    preds["training_data_through_year"] = preds["test_year"] - 2
    preds["calibration_year"] = preds["test_year"] - 1
    preds["run_id"] = run_id
    preds["dataset_id"] = data_manifest["dataset_id"]
    preds["feature_version"] = cfg["features"]["version"]
    preds["model_version"] = preds.apply(lambda r: f"{r['family']}:{r['model']}:fold{r['test_year']}", axis=1)

    metrics = compute_metrics(preds, folds, seed, reps)
    metrics["label_shuffle_dev_check"] = {
        "description": "Model C fitted on shuffled inner-training labels, scored on the inner-validation year "
                       "(development data only; no test data involved). Expect no improvement over a constant.",
        "per_fold": label_shuffle,
        "mean_brier_shuffled": float(np.mean([r["brier_shuffled"] for r in label_shuffle])) if label_shuffle else None,
        "mean_brier_constant": float(np.mean([r["brier_constant"] for r in label_shuffle])) if label_shuffle else None,
    }
    manifest = {
        "run_id": run_id, "run_name": run_name, "note": note, "subset": subset,
        "outcome_definition": "override" if outcome_override is not None else "primary (config target)",
        "generated_at_utc": generated_at.isoformat(timespec="seconds"), "code_version": __version__, "git": _git_state(),
        "config": cfg.raw, "config_hash": cfg.section_hash(), "dataset": data_manifest,
        "model_specs": specs, "primary_family": PRIMARY_FAMILY,
        "secondary_family": SECONDARY_FAMILY if include_secondary else None,
        "folds": [f.describe() for f in folds], "fold_parameters": fold_params,
        "random_seed": seed, "bootstrap_reps": reps,
        "n_events_in_dataset": int(len(df)),
        "n_events_with_predictions": int(preds["event_id"].nunique()),
        "events_without_prediction": sorted(set(df["event_id"]) - set(preds["event_id"])),
        "reproducibility": "Deterministic given identical data files, config and library versions "
                           "(lbfgs/HGB with fixed seeds). Expected tolerance: metrics equal to ~1e-9; "
                           "different BLAS/library versions may shift probabilities at ~1e-6.",
    }
    return _write_run(layout, run_id, manifest, preds, metrics, selection_log)


def compute_metrics(preds: pd.DataFrame, folds, seed: int, reps: int) -> dict:
    wide = preds.pivot_table(index="event_id", columns=["family", "model"], values="p_cal")
    wide_raw = preds.pivot_table(index="event_id", columns=["family", "model"], values="p_raw")
    meta = preds.drop_duplicates("event_id").set_index("event_id")[["y", "quarter", "ticker", "test_year",
                                                                    "target_session", "ctx_earnings_8k_in_window"]]
    out: dict = {"scope": "pooled out-of-time predictions across all ready annual folds (each event scored once, "
                          "by the model of its own test year)"}
    ev = meta.loc[wide.index]
    out["sample"] = {"n_events": int(len(ev)), "n_companies": int(ev["ticker"].nunique()),
                     "n_positive": int(ev["y"].sum()), "prevalence": float(ev["y"].mean()) if len(ev) else None,
                     "first_target_session": str(ev["target_session"].min().date()) if len(ev) else None,
                     "last_target_session": str(ev["target_session"].max().date()) if len(ev) else None,
                     "test_years": sorted(int(y) for y in ev["test_year"].unique())}
    out["folds"] = [f.describe() for f in folds]
    models = {}
    for col in wide.columns:
        fam, model = col
        key = f"{fam}:{model}"
        p = wide[col]
        ok = p.notna()
        models[key] = {"calibrated": metric_block(meta.loc[p.index[ok], "y"], p[ok]),
                       "raw": metric_block(meta.loc[p.index[ok], "y"], wide_raw[col][ok]),
                       "calibration_bins": calibration_bins(meta.loc[p.index[ok], "y"], p[ok])}
    out["models"] = models
    # per year
    per_year = []
    for yr, g in meta.loc[wide.index].groupby("test_year"):
        for col in wide.columns:
            p = wide.loc[g.index, col]
            mb = metric_block(g["y"], p)
            per_year.append({"test_year": int(yr), "model": f"{col[0]}:{col[1]}", **{k: mb[k] for k in
                             ("n_events", "n_positive", "prevalence", "brier", "log_loss", "average_precision")}})
    out["per_year"] = per_year
    # paired differences
    frame = meta.loc[wide.index].copy()
    for col in wide.columns:
        frame[f"{col[0]}:{col[1]}"] = wide[col]
    frame = frame.reset_index()
    pairs = [("logistic_l2:C", "logistic_l2:B"), ("logistic_l2:B", "logistic_l2:A"),
             ("logistic_l2:A", "constant:constant"), ("logistic_l2:C", "constant:constant"),
             ("logistic_l2:B+length", "logistic_l2:B"), ("logistic_l2:B+dictionary", "logistic_l2:B"),
             ("logistic_l2:B+paragraph_change", "logistic_l2:B"), ("hgb_shallow:C", "hgb_shallow:B")]
    diffs = {}
    for a, b in pairs:
        if a in frame and b in frame:
            diffs[f"{a} vs {b}"] = cluster_bootstrap_diff(frame, a, b, "quarter", reps, seed)
    if "logistic_l2:C" in frame and "logistic_l2:B" in frame:
        diffs["logistic_l2:C vs logistic_l2:B (company clusters)"] = cluster_bootstrap_diff(
            frame, "logistic_l2:C", "logistic_l2:B", "ticker", reps, seed)
        raw = meta.loc[wide_raw.index].copy().reset_index()
        raw["C_raw"] = wide_raw[("logistic_l2", "C")].to_numpy()
        raw["B_raw"] = wide_raw[("logistic_l2", "B")].to_numpy()
        diffs["logistic_l2:C vs logistic_l2:B (uncalibrated)"] = cluster_bootstrap_diff(raw, "C_raw", "B_raw", "quarter", reps, seed)
    out["paired_differences"] = diffs
    primary = diffs.get("logistic_l2:C vs logistic_l2:B")
    out["primary"] = {
        "comparison": "Delta Brier = Brier(C) - Brier(B), calibrated logistic models, pooled out-of-time; negative favours text",
        "estimate": primary["brier"]["estimate"] if primary else None,
        "ci95": primary["brier"]["ci95"] if primary else None,
        "conclusion": conclusion_from_ci(primary["brier"]["ci95"]) if primary else "unavailable",
    }
    # subgroups (descriptive, not used for the conclusion)
    sub = {}
    for name, mask in [("earnings_8k_in_window", frame["ctx_earnings_8k_in_window"] == 1),
                       ("no_earnings_8k_in_window", frame["ctx_earnings_8k_in_window"] == 0)]:
        g = frame[mask]
        if len(g) and "logistic_l2:C" in g:
            sub[name] = {m: metric_block(g["y"], g[m]) for m in ("constant:constant", "logistic_l2:A",
                                                                  "logistic_l2:B", "logistic_l2:C") if m in g}
    out["subgroups"] = sub
    return out


def _write_run(layout: DataLayout, run_id: str, manifest: dict, preds: pd.DataFrame, metrics: dict,
               selection_log: list) -> Path:
    final_dir = layout.experiments / run_id
    tmp_dir = layout.experiments / f".tmp-{run_id}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    write_json_atomic(tmp_dir / "manifest.json", manifest)
    preds.to_parquet(tmp_dir / "predictions.parquet", index=False)
    write_json_atomic(tmp_dir / "metrics.json", metrics)
    write_json_atomic(tmp_dir / "selection_log.json", selection_log)
    (tmp_dir / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat(timespec="seconds"))
    os.replace(tmp_dir, final_dir)
    log.info("experiment %s written to %s", run_id, final_dir)
    return final_dir


def list_runs(layout: DataLayout) -> list[dict]:
    out = []
    if not layout.experiments.exists():
        return out
    for d in sorted(layout.experiments.iterdir()):
        if d.name.startswith(".") or not d.is_dir():
            continue
        complete = (d / "COMPLETE").exists()
        entry = {"run_id": d.name, "complete": complete, "path": str(d)}
        if complete:
            m = read_json(d / "manifest.json")
            entry.update({"run_name": m.get("run_name"), "dataset_id": m["dataset"]["dataset_id"],
                          "generated_at_utc": m["generated_at_utc"], "subset": m.get("subset"),
                          "outcome_definition": m.get("outcome_definition"), "note": m.get("note")})
        out.append(entry)
    return out


def set_primary_run(layout: DataLayout, run_id: str) -> None:
    write_json_atomic(layout.experiments / "primary_run.json", {"run_id": run_id,
                      "set_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
