"""Generated documentation: every number in these files is read from saved artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from filingshock.features.catalog import CATALOG
from filingshock.storage import DataLayout, read_json

LABEL = {"constant:constant": "Constant baseline", "logistic_l2:A": "A: market + context",
         "logistic_l2:B": "B: A + financial", "logistic_l2:C": "C: B + text (primary)",
         "logistic_l2:B+length": "B + text length", "logistic_l2:B+dictionary": "B + dictionary features",
         "logistic_l2:B+paragraph_change": "B + paragraph-change features", "hgb_shallow:A": "A (shallow boosting)",
         "hgb_shallow:B": "B (shallow boosting)", "hgb_shallow:C": "C (shallow boosting)"}


def _fmt(x, d=4, signed=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "unavailable"
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def feature_catalog_md(path: Path) -> None:
    lines = ["# Feature catalog (fs-features-1.0)", "",
             "Generated from `src/filingshock/features/catalog.py`. Model A = market + context; B adds financial; "
             "C adds the three text groups.", "",
             "| Feature | Group | Definition | Source | Units | Transformation | Availability rule |",
             "|---|---|---|---|---|---|---|"]
    for c in CATALOG:
        lines.append(f"| `{c['name']}` | {c['group']} | {c['definition']} | {c['source']} | {c['units']} | "
                     f"{c['transformation']} | {c['availability']} |")
    lines += ["", "Preprocessing per fold, fitted on the fit block only: median imputation with a missingness "
              "indicator for any feature missing in training, 1st/99th-percentile winsorisation, standardisation."]
    path.write_text("\n".join(lines) + "\n")


def results_md(layout: DataLayout, path: Path, superseded_run: str | None = None) -> dict:
    run_id = read_json(layout.experiments / "primary_run.json")["run_id"]
    rdir = layout.experiments / run_id
    m, man = read_json(rdir / "metrics.json"), read_json(rdir / "manifest.json")
    s, p = m["sample"], m["primary"]
    L = ["# Experiment results", "",
         f"Generated from `data/experiments/{run_id}/metrics.json` (dataset `{man['dataset']['dataset_id']}`, "
         f"code {man['code_version']}, git `{str((man.get('git') or {}).get('commit') or 'n/a')[:8]}`).", "",
         "## Scope", "",
         f"Pooled out-of-time predictions, test years {s['test_years'][0]}–{s['test_years'][-1]}: "
         f"{s['n_events']} events, {s['n_companies']} companies, {s['n_positive']} large moves "
         f"(prevalence {s['prevalence']:.3f}), target sessions {s['first_target_session']} to "
         f"{s['last_target_session']}. Each event is scored once, by the model of its own test year "
         "(fit ≤ Y−2, calibrated on Y−1).", "",
         "## Primary comparison (pre-specified)", "",
         f"**ΔBrier = Brier(C) − Brier(B) = {p['estimate']:+.4f}, 95% interval [{p['ci95'][0]:+.4f}, "
         f"{p['ci95'][1]:+.4f}]** (cluster bootstrap over calendar quarters, {man['bootstrap_reps']} draws).", "",
         f"Conclusion under the protocol's decision rule: **{p['conclusion']}**.", "",
         "## All models (calibrated probabilities)", "",
         "| Model | Brier ↓ | Log loss ↓ | Avg. precision ↑ | ROC-AUC | Precision top 10% | Lift top 10% |",
         "|---|---|---|---|---|---|---|"]
    for k in [k for k in LABEL if k in m["models"]]:
        c = m["models"][k]["calibrated"]
        L.append(f"| {LABEL.get(k, k)} | {_fmt(c['brier'])} | {_fmt(c['log_loss'])} | {_fmt(c['average_precision'], 3)} | "
                 f"{_fmt(c['roc_auc'], 3)} | {_fmt(c['precision_top10pct'], 3)} (k={c['k_top10pct']}) | "
                 f"{_fmt(c['lift_top10pct'], 2)}× |")
    L += ["", "Top-10% precision ranks a completed holdout retrospectively; it is not a deployable selection rule.", "",
          "## Paired differences (first − second)", "",
          "| Comparison | ΔBrier | 95% interval | ΔAvg. precision | Resampling unit |", "|---|---|---|---|---|"]
    for k, v in m["paired_differences"].items():
        b = v["brier"]
        ap = v.get("average_precision", {}).get("estimate")
        L.append(f"| {k} | {b['estimate']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {_fmt(ap, 3, True)} | {v['cluster']} |")
    L += ["", "## Per-year Brier (logistic models)", "", "| Year | Events | Large moves | Constant | B | C | C − B |",
          "|---|---|---|---|---|---|---|"]
    py = pd.DataFrame(m["per_year"])
    for yr, g in py.groupby("test_year"):
        gg = g.set_index("model")
        b, c, k = gg.loc["logistic_l2:B", "brier"], gg.loc["logistic_l2:C", "brier"], gg.loc["constant:constant", "brier"]
        L.append(f"| {yr} | {int(gg['n_events'].iloc[0])} | {int(gg['n_positive'].iloc[0])} | {k:.4f} | {b:.4f} | {c:.4f} | {c - b:+.4f} |")
    L += ["", "## Sensitivity analyses (pre-specified)", "", "| Run | Events | Prevalence | ΔBrier (C − B) | 95% interval | Reading |",
          "|---|---|---|---|---|---|"]
    for d in sorted(layout.experiments.iterdir()):
        if not d.is_dir() or d.name == run_id or not (d / "COMPLETE").exists():
            continue
        mm, mn = read_json(d / "metrics.json"), read_json(d / "manifest.json")
        if mn["dataset"]["dataset_id"] != man["dataset"]["dataset_id"] or mn.get("run_name") == "primary":
            continue
        pp = mm["primary"]
        L.append(f"| {mn['run_name']} | {mm['sample']['n_events']} | {mm['sample']['prevalence']:.3f} | "
                 f"{pp['estimate']:+.4f} | [{pp['ci95'][0]:+.4f}, {pp['ci95'][1]:+.4f}] | {pp['conclusion']} |")
    L += ["", "Readings apply the primary decision rule mechanically. Sensitivity and subgroup results are secondary, "
          "several were run, and none is adjusted for multiple comparisons."]
    ls = m.get("label_shuffle_dev_check", {})
    L += ["", "## Falsification and pipeline checks", "",
          f"* Label shuffle (development data only): mean Brier {ls.get('mean_brier_shuffled', float('nan')):.4f} vs "
          f"{ls.get('mean_brier_constant', float('nan')):.4f} for a constant — no spurious improvement.",
          "* Design-matrix guard: only catalogued `mkt_/ctx_/fin_/txt_` columns may enter a model (enforced at fit time).",
          "* Future-data mutation and target-close mutation: `tests/test_leakage.py`, `tests/test_target.py`.",
          f"* Events without a prediction: {len(man['events_without_prediction'])} of {man['n_events_in_dataset']} "
          "(2010–2013 serve only as training and calibration data).", ""]
    if superseded_run and (layout.experiments / superseded_run / "metrics.json").exists():
        sm = read_json(layout.experiments / superseded_run / "metrics.json")
        sp = sm["primary"]
        L += ["## Superseded run (before the acceptance-time correction)", "",
              f"`{superseded_run}` used submissions-API acceptance times, later found to be wrong by the Eastern UTC offset "
              f"for some filings. It had {sm['sample']['n_events']} events and ΔBrier {sp['estimate']:+.4f} "
              f"[{sp['ci95'][0]:+.4f}, {sp['ci95'][1]:+.4f}] ({sp['conclusion']}). It is kept for transparency and is "
              "not the primary result.", ""]
    path.write_text("\n".join(L) + "\n")
    return {"run_id": run_id, "primary": p}


def export_figures(layout: DataLayout, out_dir: Path) -> list[str]:
    """Static PNG copies of the results charts for documents (same builders as the interface)."""
    import vl_convert as vlc

    from filingshock import charts
    run_id = read_json(layout.experiments / "primary_run.json")["run_id"]
    m = read_json(layout.experiments / run_id / "metrics.json")
    funnel = read_json(layout.processed / "funnel.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for name, ch, title in [
        ("paired_differences.png", charts.forest_chart(m), "Paired ΔBrier with 95% intervals (pooled out-of-time)"),
        ("calibration.png", charts.calibration_chart(m), "Calibration of Models B and C (pooled out-of-time, 8 equal-count bins)"),
        ("yearly_brier.png", charts.yearly_chart(m), "Brier score by test year (lower is better)"),
        ("eligibility_funnel.png", charts.funnel_chart(funnel), "Eligibility funnel (periodic reports → final sample)"),
    ]:
        spec = charts.style(ch.properties(width=640, title=title), background=charts.SURFACE).to_dict()
        (out_dir / name).write_bytes(vlc.vegalite_to_png(spec, scale=2))
        made.append(name)
    return made
