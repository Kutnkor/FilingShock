"""Production models for prospective forecasts (research protocol sections 6-7 applied to the live year).

For a pending event whose target session falls in year Y, models B and C are fitted exactly like the
research fold for test year Y: fit block = events with target session in years <= Y-2 (hyperparameter
chosen on inner train <= Y-3 / validation Y-2), Platt calibration on year Y-1. Nothing from year Y is
used, every training and calibration outcome closed before year Y began, and the same readiness rules
apply. The fitted parameters are written once to forecasts/models/<version>.json and their hash is
stored with every ledger record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from filingshock import __version__
from filingshock.config import Config
from filingshock.modeling.experiment import (
    PRIMARY_FAMILY,
    CalibratedModel,
    fit_calibrated_model,
    load_dataset,
    model_specs,
)
from filingshock.modeling.splits import Fold, make_fold
from filingshock.storage import DataLayout, file_lock, read_json, write_json_atomic

PRODUCTION_MODELS = ("B", "C")


class ProductionModelNotReady(RuntimeError):
    pass


@dataclass
class ProductionModel:
    year: int
    dataset_id: str
    feature_version: str
    fold: Fold
    models: dict[str, CalibratedModel]
    max_outcome_available_utc: pd.Timestamp
    fitted_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def version(self) -> str:
        return f"{PRIMARY_FAMILY}:prod{self.year}:{self.dataset_id}"

    def predict(self, X: pd.DataFrame) -> dict[str, dict]:
        out = {}
        for name, cm in self.models.items():
            p_raw, p_cal = cm.predict(X)
            out[name] = {"p_raw": float(p_raw[0]), "p_cal": float(p_cal[0]), "calibration_method": cm.platt.method}
        return out

    def parameters(self) -> dict:
        """Deterministic content (no timestamps): what the parameter hash covers."""
        return {"version": self.version, "family": PRIMARY_FAMILY, "year": self.year, "dataset_id": self.dataset_id,
                "feature_version": self.feature_version, "code_version": __version__,
                "training_data_through_year": self.year - 2, "inner_validation_year": self.year - 2,
                "calibration_year": self.year - 1, "fold": self.fold.describe(),
                "max_outcome_available_utc": str(self.max_outcome_available_utc),
                "models": {name: {"features": cm.features, **cm.parameters(), "selection_attempts": cm.attempts}
                           for name, cm in self.models.items()}}

    @property
    def parameters_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.parameters(), sort_keys=True, default=str).encode()).hexdigest()


def fit_production_model(cfg: Config, layout: DataLayout, year: int) -> ProductionModel:
    df, data_manifest = load_dataset(layout)
    scfg, mcfg = cfg["splits"], cfg["model"]
    fold = make_fold(df[["event_id", "target_session", "y"]], year, int(scfg["min_train_events"]),
                     int(scfg["min_train_positives"]), int(scfg["min_calibration_positives"]))
    if not fold.ready:
        raise ProductionModelNotReady(f"production fold {year} not ready: {', '.join(fold.readiness_failures)}")
    idx = df.set_index("event_id")
    fit, itr, iva, cal = (idx.loc[fold.fit_ids], idx.loc[fold.inner_train_ids], idx.loc[fold.inner_val_ids],
                          idx.loc[fold.cal_ids])
    specs = model_specs()
    catalog = set(c for cols in specs.values() for c in cols)
    seed = int(mcfg["random_seed"])
    models = {m: fit_calibrated_model(m, PRIMARY_FAMILY, specs[m], itr, iva, fit, cal, mcfg, seed, catalog)
              for m in PRODUCTION_MODELS}
    matured = pd.concat([fit, cal])["outcome_available_utc"].max()
    return ProductionModel(year, data_manifest["dataset_id"], cfg["features"]["version"], fold, models, matured)


def save_production_manifest(layout: DataLayout, pm: ProductionModel) -> Path:
    """Written once per (version, parameter hash); an existing manifest is never overwritten."""
    path = layout.forecasts / "models" / f"{pm.version.replace(':', '__')}__{pm.parameters_sha256[:12]}.json"
    with file_lock(path.with_suffix(".json.lock")):
        if path.exists():
            try:
                stored = read_json(path)
                parameters = {k: v for k, v in stored.items() if k not in ("parameters_sha256", "fitted_at_utc")}
                digest = hashlib.sha256(json.dumps(parameters, sort_keys=True, default=str).encode()).hexdigest()
                valid = stored.get("parameters_sha256") == pm.parameters_sha256 == digest
            except (ValueError, AttributeError):
                valid = False
            if not valid:
                raise ProductionModelNotReady(f"cached model manifest failed integrity verification: {path.name}")
        else:
            write_json_atomic(path, {**pm.parameters(), "parameters_sha256": pm.parameters_sha256,
                                     "fitted_at_utc": pm.fitted_at_utc})
    return path
