"""Research Results: read-only view of saved experiment artifacts (nothing is retrained here)."""

import json

import pandas as pd
import streamlit as st

from filingshock import charts
from filingshock.ui import components as ui
from filingshock.ui import data as D
from filingshock.ui.shared import data_banner, require_dataset

MODEL_LABEL = {"constant:constant": "Constant baseline", "logistic_l2:A": "A · market + context",
               "logistic_l2:B": "B · A + financial", "logistic_l2:C": "C · B + text (primary)",
               "logistic_l2:B+length": "B + text length", "logistic_l2:B+dictionary": "B + dictionary",
               "logistic_l2:B+paragraph_change": "B + paragraph change",
               "hgb_shallow:A": "A (shallow boosting)", "hgb_shallow:B": "B (shallow boosting)",
               "hgb_shallow:C": "C (shallow boosting)"}


data = D.load()
ui.page_header("Research Results",
               "Does adding text-change features (Model C) improve on market + financial features (Model B) for the same "
               "events? Every number below is read from saved experiment artifacts.")
data_banner(data)
if not require_dataset(data):
    st.stop()
if data.metrics is None:
    ui.status("unavail", "No completed experiment is available.",
              "Evaluation results have not been produced for this dataset. Run "
              "`filingshock experiment --name primary --set-primary`. No illustrative numbers are shown in their place.")
    st.stop()

m, man = data.metrics, data.run_manifest
s = m["sample"]
ui.facts([
    ("Run", f"{data.run_id} · {man.get('note') or ''}"),
    ("Scope", "Pooled out-of-time predictions: each event scored once by the model of its own test year "
              "(fit ≤ Y−2, calibrated on Y−1, tested on Y)"),
    ("Sample", f"{s['n_events']} events · {s['n_companies']} companies · {s['n_positive']} large moves · "
               f"prevalence {ui.pct(s['prevalence'], 1)} · target sessions {s['first_target_session']} to {s['last_target_session']}"),
    ("Data / code", f"dataset {man['dataset']['dataset_id']} · feature set {man['config']['features']['version']} · "
                    f"code {man['code_version']} (git {str((man.get('git') or {}).get('commit') or 'n/a')[:8]})"),
])

# ---------------------------------------------------------------- primary result
prim = m["primary"]
ui.section_title("Primary comparison (pre-specified)")
est, ci = prim["estimate"], prim["ci95"]
kind = "info" if prim["conclusion"].startswith("inconclusive") else ("warn" if "worsened" in prim["conclusion"] else "info")
ui.status(kind, f"ΔBrier = Brier(C) − Brier(B) = {est:+.4f}  (95% interval {ci[0]:+.4f} to {ci[1]:+.4f})",
          f"Conclusion under the protocol's decision rule: {prim['conclusion']}. Negative values would favour text. "
          "The interval comes from a cluster bootstrap over calendar quarters (events in the same quarter are resampled "
          f"together), {man['bootstrap_reps']:,} draws. A Brier score is a forecast-quality measure (mean squared error of "
          "the probability), not an accuracy percentage.")

# ---------------------------------------------------------------- model table
ui.section_title("Model comparison on the common event set")
rows = []
order = ["constant:constant", "logistic_l2:A", "logistic_l2:B", "logistic_l2:C", "logistic_l2:B+length",
         "logistic_l2:B+dictionary", "logistic_l2:B+paragraph_change", "hgb_shallow:A", "hgb_shallow:B", "hgb_shallow:C"]
for k in order:
    if k not in m["models"]:
        continue
    c = m["models"][k]["calibrated"]
    rows.append({"Model": MODEL_LABEL.get(k, k), "Events": c["n_events"], "Large moves": c["n_positive"],
                 "Brier ↓": f"{c['brier']:.4f}", "Log loss ↓": f"{c['log_loss']:.4f}",
                 "Avg. precision ↑": "unavailable" if c["average_precision"] is None else f"{c['average_precision']:.3f}",
                 "ROC-AUC": "unavailable" if c["roc_auc"] is None else f"{c['roc_auc']:.3f}",
                 "Precision, top 10%": f"{c['precision_top10pct']:.3f} (k={c['k_top10pct']})",
                 "Lift, top 10%": f"{c['lift_top10pct']:.2f}×",
                 "Precision, top 20%": f"{c['precision_top20pct']:.3f} (k={c['k_top20pct']})"})
ui.html_table(pd.DataFrame(rows), right={"Events", "Large moves", "Brier ↓", "Log loss ↓", "Avg. precision ↑", "ROC-AUC",
                                         "Lift, top 10%"},
              highlight_first_col_value="C · B + text (primary)",
              caption_text=f"Calibrated probabilities, pooled {s['test_years'][0]}–{s['test_years'][-1]}. Lift is relative to "
                           f"the observed prevalence of {ui.pct(s['prevalence'], 1)}. Top-k precision ranks a completed holdout "
                           "after the fact: it is a ranking diagnostic, not a deployable selection rule. Average precision is "
                           "sklearn's step-wise estimator, not trapezoidal PR-AUC.")

# ---------------------------------------------------------------- paired differences
ui.section_title("Paired differences with dependence-aware intervals")
ddf = charts.paired_differences_frame(m)
st.altair_chart(charts.style(charts.forest_chart(m)), width="stretch")
tbl = ddf.assign(**{"ΔBrier": ddf["estimate"].map(lambda x: f"{x:+.4f}"),
                    "95% interval": ddf.apply(lambda r: f"{r['lo']:+.4f} to {r['hi']:+.4f}", axis=1),
                    "ΔAvg. precision": ddf["ap"].map(lambda x: "unavailable" if pd.isna(x) else f"{x:+.3f}"),
                    "Resampling unit": ddf["cluster"]})[["comparison", "ΔBrier", "95% interval", "ΔAvg. precision", "Resampling unit"]]
with st.expander("Table view of the paired differences"):
    ui.html_table(tbl, right={"ΔBrier", "ΔAvg. precision"})
ui.caption("Orange marks the pre-specified primary comparison. 'quarter' rows resample calendar quarters; 'ticker' rows "
           "resample companies; '(uncalibrated)' uses raw model probabilities. Ablation rows add one text group at a time to B.")

# ---------------------------------------------------------------- calibration
ui.section_title("Calibration: do predicted probabilities match observed frequencies?")
cdf = charts.calibration_frame(m)
cc1, cc2 = st.columns([1.3, 1])
with cc1:
    st.altair_chart(charts.style(charts.calibration_chart(m)), width="stretch")
with cc2:
    ui.caption("Eight equal-count bins per model, pooled out-of-time. Point size shows how many events support each "
               "bin; with roughly 60 events per bin, each observed share has a binomial standard error of about "
               "±5–6 percentage points, so small departures from the dashed diagonal are not evidence of miscalibration.")
    ct = cdf[["model", "bin", "n", "mean_predicted", "observed_rate"]].copy()
    ct["mean_predicted"] = ct["mean_predicted"].map(lambda x: f"{x:.3f}")
    ct["observed_rate"] = ct["observed_rate"].map(lambda x: f"{x:.3f}")
    with st.expander("Table view of calibration bins"):
        ui.html_table(ct, right={"n", "mean_predicted", "observed_rate"})

# ---------------------------------------------------------------- per-year
ui.section_title("Stability over time")
py = charts.yearly_frame(m)
st.altair_chart(charts.style(charts.yearly_chart(m)), width="stretch")
wide = py.pivot_table(index="test_year", columns="series", values="brier")
counts = py.drop_duplicates("test_year").set_index("test_year")[["n_events", "n_positive"]]
yt = counts.join(wide).reset_index()
yt["C − B"] = yt["Model C (with text)"] - yt["Model B (no text)"]
for c in ["Constant baseline", "Model B (no text)", "Model C (with text)"]:
    yt[c] = yt[c].map(lambda x: f"{x:.4f}")
yt["C − B"] = yt["C − B"].map(lambda x: f"{x:+.4f}")
with st.expander("Table view: Brier by test year"):
    ui.html_table(yt.rename(columns={"test_year": "Year", "n_events": "Events", "n_positive": "Large moves"}),
                  right={"Events", "Large moves", "Constant baseline", "Model B (no text)", "Model C (with text)", "C − B"})
ui.caption("Each year has roughly 30–42 events, so single-year differences are noisy; no per-year interval is shown "
           "because none was estimated. The pooled comparison above is the pre-specified evidence.")

# ---------------------------------------------------------------- sensitivity
ui.section_title("Pre-specified sensitivity analyses")
srows = []
for run in D.other_runs():
    if run["run_name"] in (None, "primary") or run["dataset_id"] != data.dataset["dataset_id"]:
        continue
    pm = run["metrics"]["primary"]
    sm = run["metrics"]["sample"]
    srows.append({"Analysis": {"sens-multiplier-1.5": "Threshold 1.5 × mean", "sens-multiplier-2.5": "Threshold 2.5 × mean",
                               "sub-no-earnings-8k": "10-Q not accompanied by an earnings 8-K",
                               "sub-earnings-8k": "10-Q filed with an earnings 8-K"}.get(run["run_name"], run["run_name"]),
                  "Events": sm["n_events"], "Prevalence": ui.pct(sm["prevalence"], 1),
                  "ΔBrier (C − B)": f"{pm['estimate']:+.4f}", "95% interval": f"{pm['ci95'][0]:+.4f} to {pm['ci95'][1]:+.4f}",
                  "Reading": pm["conclusion"]})
if srows:
    ui.html_table(pd.DataFrame(srows), right={"Events", "Prevalence", "ΔBrier (C − B)"},
                  caption_text="Same protocol and folds; subgroup runs refit all models within the subgroup. Fewer events "
                               "appear when a fold fails readiness (too few events of one class). Readings apply the "
                               "primary decision rule mechanically; these secondary results are not adjusted for "
                               "multiple comparisons.")
else:
    ui.status("unavail", "No sensitivity runs saved for this dataset.", "Run `filingshock sensitivity`.")

# ---------------------------------------------------------------- funnel
ui.section_title("Eligibility funnel")
fun = data.funnel
st.altair_chart(charts.style(charts.funnel_chart(fun)), width="stretch")
reasons = [{"Stage": s_["stage"], "Exclusion reason": k, "Filings": v} for s_ in fun["stages"] for k, v in s_["reasons"].items()]
rdf = pd.DataFrame(reasons)
ui.html_table(rdf, right={"Filings"})
if True:  # record inspector (Journey C: pick an exclusion reason and inspect affected filings)
    choice = st.selectbox("Inspect records for an exclusion reason", ["—"] + list(rdf["Exclusion reason"]), key="res_reason")
    if choice != "—":
        ex = data.events[data.events["exclusion_reason"] == choice].sort_values("acceptance_utc")
        show = pd.DataFrame({"Company": ex["ticker"], "Form": ex["form_type"], "Accepted (ET)": ex["acceptance_utc"].map(ui.ts_et),
                             "Filing date": ex["filing_date"], "Accession": ex["accession_number"]})
        st.dataframe(show, hide_index=True, width="stretch", height=260)
        ui.caption(f"{len(ex)} filings. Delisted companies fail at the price stage because the free provider has no history "
                   "for them; this is a survivorship limitation of the cohort, not a parsing failure. Licensed data can be "
                   "added with `filingshock prices --provider csv`.")

# ---------------------------------------------------------------- prospective ledger (not research evidence)
ui.section_title("Prospective forecasts (ledger, excluded from the results above)")
led_ok, led_msg = data.ledger_integrity
if not led_ok:
    ui.status("error", "The prospective ledger failed its integrity check.",
              f"{led_msg}. A record was edited or removed after it was written; `filingshock forecast-pending` refuses to "
              "append until this is resolved.")
led = data.prospective
if led is None or led.empty:
    ui.status("unavail", "No prospective forecast has been recorded yet.",
              "`filingshock forecast-pending` records a forecast only for a 10-Q first seen by discovery before its "
              "decision cutoff, and only while the cutoff is still in the future. Records are append-only and never "
              "enter the metrics above.")
else:
    from filingshock.prospective_outcomes import OUTCOME_COLUMNS, prospective_summary
    po_ok, po_msg = data.outcomes_integrity
    if not po_ok:
        ui.status("error", "The prospective outcome log failed its integrity check.",
                  f"{po_msg}. An outcome record was edited or removed after it was written; `filingshock score-prospective` "
                  "refuses to append until this is resolved.")
    pout = data.prospective_outcomes if data.prospective_outcomes is not None else pd.DataFrame(columns=OUTCOME_COLUMNS)
    now = pd.Timestamp.now(tz="UTC")
    lt = led.sort_values("sequence").merge(
        pout.reindex(columns=["forecast_id", "outcome_status", "abs_r", "y", "brier_A", "brier_B", "brier_C",
                              "brier_constant"]).add_prefix("o_"),
        left_on="forecast_id", right_on="o_forecast_id", how="left")

    def session_state(r) -> str:
        if r["o_outcome_status"] == "ok":
            return "closed; outcome scored"
        if r["o_outcome_status"] == "target_bar_invalid":
            return "closed; target bar invalid (recorded, not scored)"
        if pd.Timestamp(r["target_close_utc"]) > now:
            return "not closed yet"
        return "closed; awaiting score-prospective"

    ltab = pd.DataFrame({
        "Company": lt["ticker"], "Accession": lt["accession_number"],
        "Observed (ET)": lt["observed_at_utc"].map(ui.ts_et), "Generated (ET)": lt["generated_at_utc"].map(ui.ts_et),
        "Cutoff (ET)": lt["decision_cutoff_utc"].map(ui.ts_et), "Target session": lt["target_session"],
        "Threshold |move|": lt["threshold_abs_move"].map(lambda x: ui.pct(x, 2)),
        "P(B)": lt["p_B"].map(lambda x: ui.pct(x, 1)), "P(C)": lt["p_C"].map(lambda x: ui.pct(x, 1)),
        "Realized |move|": lt["o_abs_r"].map(lambda x: ui.pct(x, 2)),
        "y": lt["o_y"].map(lambda x: ui.DASH if pd.isna(x) else str(int(x))),
        "Brier B": lt["o_brier_B"].map(lambda x: ui.num(x, 4)), "Brier C": lt["o_brier_C"].map(lambda x: ui.num(x, 4)),
        "Model": lt["model_version"],
        "Session": lt.apply(session_state, axis=1),
    })
    for name, label in (("A", "A"), ("constant", "constant")):
        if f"p_{name}" in lt and lt[f"p_{name}"].notna().any():
            ltab[f"P({label})"] = lt[f"p_{name}"].map(lambda x: ui.pct(x, 1))
            ltab[f"Brier {label}"] = lt[f"o_brier_{name}"].map(lambda x: ui.num(x, 4))
    ui.html_table(ltab, right={"Threshold |move|", "P(A)", "P(B)", "P(C)", "P(constant)", "Realized |move|", "y",
                               "Brier A", "Brier B", "Brier C", "Brier constant"},
                  caption_text=f"Prospective only. {len(lt)} record(s) in forecasts/prospective_forecasts.jsonl, integrity "
                               f"{led_msg}; {len(pout)} outcome record(s) in forecasts/prospective_outcomes.jsonl, integrity "
                               f"{po_msg}. Each forecast was generated before its decision cutoff by production models fitted "
                               "on matured events (fit ≤ Y−2, Platt calibration on Y−1). Outcomes are scored after the target "
                               "session closes against the threshold stored in the forecast record (never recomputed); "
                               "y = 1 if realized |move| > threshold, Brier = (p − y)². None of this enters the metrics above.")
    ps = prospective_summary(pout)
    for name, comparison in ps["baseline_comparisons"].items():
        summary = comparison["pooled"]
        if summary:
            ui.facts([(f"Prospective C vs {name}",
                       f"{comparison['n_paired_events']} paired events · Brier C {ui.num(summary['brier_C'], 4)} · "
                       f"baseline {ui.num(summary['brier_baseline'], 4)} · "
                       f"C − baseline {ui.num(summary['delta_brier_C_minus_baseline'], 4, signed=True)}")])
        else:
            ui.caption(f"C vs {name}: {comparison['n_paired_events']} paired scored events; "
                       f"needs {ps['min_events_for_pooled']} for a pooled comparison. Older forecasts without this "
                       "baseline are excluded from that comparison, never backfilled.")
    pooled = ps["pooled"]
    if pooled:
        ui.facts([
            ("Prospective pooled sample", f"{pooled['n_events']} scored events · {pooled['n_positive']} large moves · "
                                          f"prevalence {ui.pct(pooled['prevalence'], 1)}"),
            ("Prospective Brier", f"B {ui.num(pooled['brier_B'], 4)} · C {ui.num(pooled['brier_C'], 4)} · "
                                  f"C − B {ui.num(pooled['delta_brier_C_minus_B'], 4, signed=True)}"),
        ])
        ui.caption("Earliest-recorded forecast per event, valid outcomes only. A descriptive prospective check, separate "
                   "from the research sample; no confidence interval or conclusion is drawn from it here.")
    else:
        ui.caption(f"{ps['n_scored_events']} prospective event(s) scored so far. Pooled prospective Brier scores are shown "
                   f"from {ps['min_events_for_pooled']} scored events on; per-record values above are not a performance estimate.")

# ---------------------------------------------------------------- checks and downloads
ui.section_title("Pipeline diagnostics and artifacts")
ls = m.get("label_shuffle_dev_check", {})
ui.facts([
    ("Label-shuffle check (development data only)",
     f"mean Brier {ls.get('mean_brier_shuffled', float('nan')):.4f} with shuffled labels vs {ls.get('mean_brier_constant', float('nan')):.4f} "
     "for a constant — shuffled labels give no improvement, as expected"),
    ("Events without a prediction", f"{len(man['events_without_prediction'])} of {man['n_events_in_dataset']} final-sample events "
                                    "(2010–2013 are used only for fitting and calibrating later folds)"),
    ("Reproducibility", man.get("reproducibility", "")),
])
pcsv = data.predictions.drop(columns=[c for c in ("decision_cutoff_utc", "outcome_available_utc", "simulated_cutoff_utc")
                                      if c in data.predictions]).to_csv(index=False).encode()
d1, d2 = st.columns(2)
d1.download_button("Download canonical predictions (CSV)", pcsv, file_name=f"predictions_{data.run_id}.csv", mime="text/csv",
                   width="stretch")
d2.download_button("Download metrics (JSON)", json.dumps(m, indent=2).encode(), file_name=f"metrics_{data.run_id}.json",
                   mime="application/json", width="stretch")
