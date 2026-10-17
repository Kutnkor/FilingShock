"""Prediction & Historical Replay: what was knowable at the cutoff, then (on request) what happened."""

import pandas as pd
import streamlit as st

from filingshock.features.catalog import CATALOG_BY_NAME
from filingshock.ui import components as ui
from filingshock.ui import data as D
from filingshock.ui.shared import data_banner, definition_strip, require_dataset

data = D.load()
ui.page_header("Prediction & Historical Replay",
               "Inspect what was knowable at the information cutoff and the saved out-of-time prediction, then reveal "
               "what happened in the target session. The reveal is a presentation step; the prediction was computed "
               "without the outcome.")
data_banner(data)
if not require_dataset(data):
    st.stop()

table = D.event_table(data)
preds = data.predictions

# ---------------------------------------------------------------- demonstration cases (documented rules)
if preds is not None and data.has_outcomes:
    pc = preds[(preds["family"] == "logistic_l2") & (preds["model"] == "C")].dropna(subset=["y"])
    if len(pc):
        cases = {
            "Highest-scored event that exceeded the threshold": pc[pc["y"] == 1].nlargest(1, "p_cal"),
            "Highest-scored event that did not (false alarm)": pc[pc["y"] == 0].nlargest(1, "p_cal"),
            "Lowest-scored event that exceeded the threshold (miss)": pc[pc["y"] == 1].nsmallest(1, "p_cal"),
            "Event closest to the median score": pc.iloc[[int((pc["p_cal"] - pc["p_cal"].median()).abs().argmin())]],
        }
        with st.expander("Demonstration cases (selected by fixed rules, not representative of aggregate performance)"):
            ui.caption("Each case is chosen mechanically from all pooled out-of-time Model C predictions in the primary run. "
                       "Selected cases include mistakes on purpose. Aggregate evidence is on the Research Results page.")
            cols = st.columns(4)
            for col, (lab, df) in zip(cols, cases.items()):
                if len(df):
                    eid = df.iloc[0]["event_id"]
                    if col.button(lab, key=f"case_{lab}", width="stretch"):
                        ui.set_selected_event(eid)
                        st.rerun()

r = ui.event_picker(table, key="replay_event", only_eligible=True)
if r is None:
    ui.status("unavail", "No final-sample events are available.", "")
    st.stop()
ui.identity_strip(r)
definition_strip(data)

eid = r["event_id"]
ev_preds = preds[preds["event_id"] == eid] if preds is not None else pd.DataFrame()
feat_row = data.features.set_index("event_id").loc[eid] if data.features is not None and eid in set(data.features["event_id"]) else None
thr = r.get("threshold_abs_move")

left, right = st.columns([1.15, 1])
with left:
    ui.section_title("Before the session (information at the cutoff)")
    ui.facts([
        ("EDGAR acceptance", ui.ts_et(r["acceptance_utc"], True)),
        ("Assumed available", f"{ui.ts_et(r['assumed_available_utc'], True)} — "
                              f"{'acceptance + 15 min' if r['availability_method'] == 'acceptance_plus_lag' else 'next-day 06:00 ET dissemination + 15 min'}; "
                              "an assumption, not an observed time"),
        ("Information cutoff", ui.ts_et(r["decision_cutoff_utc"], True)),
        ("Target session", f"{ui.day(r['target_session'])} (open {ui.ts_et(r['target_open_utc'])}, close {ui.ts_et(r['target_close_utc'])})"),
        ("Earnings 8-K (Item 2.02) in same window",
         "unknown" if feat_row is None else ("yes — the session also reflects the earnings release"
                                             if feat_row.get("ctx_earnings_8k_in_window") == 1 else "no")),
        ("Threshold for a 'large move'", "not available in this package" if pd.isna(thr) else
         f"|open→close| > {ui.pct(thr, 2)} (2 × 60-session mean of {ui.pct(thr / 2, 2)})"),
    ])

with right:
    ui.section_title("Saved prediction")
    if ev_preds.empty:
        status = r["prediction_status"]
        if status == "no_prediction_training_period":
            ui.status("unavail", "No out-of-time prediction exists for this event.",
                      "Its target session falls before the first test year (2014). Events from 2010–2013 are used only to "
                      "fit and calibrate later models; FilingShock does not show an in-sample score as if it were a forecast.")
        else:
            ui.status("unavail", "No out-of-time prediction exists for this event.",
                      "Its test-year fold did not pass the protocol's readiness checks, so no model produced a score.")
    else:
        pv = ev_preds.set_index(["family", "model"])
        pC = pv.loc[("logistic_l2", "C"), "p_cal"]
        pB = pv.loc[("logistic_l2", "B"), "p_cal"]
        pK = pv.loc[("constant", "constant"), "p_cal"]
        test_year = int(ev_preds["test_year"].iloc[0])
        st.markdown(
            f'<div class="fs-prob"><div><div class="big">{ui.pct(pC, 1)}</div><div class="lab">Model C (with text): '
            f'probability that |open→close| exceeds {ui.pct(thr, 2) if pd.notna(thr) else "the threshold"}</div></div>'
            f'<div><div class="alt">{ui.pct(pB, 1)}</div><div class="lab">Model B (market + financial, no text)</div></div>'
            f'<div><div class="alt">{ui.pct(pK, 1)}</div><div class="lab">Constant baseline (past event rate)</div></div></div>',
            unsafe_allow_html=True)
        cal = pv.loc[("logistic_l2", "C"), "calibration_method"]
        ui.facts([
            ("Prediction category", "Retrospective out-of-time (computed in a later research run under the historical cutoff)"),
            ("Model", f"L2 logistic regression, fold {test_year}: fitted on events through {test_year - 2}, "
                      f"calibrated ({cal}) on {test_year - 1}"),
            ("Run", f"{data.run_id} · dataset {data.dataset['dataset_id']}"),
            ("Generated", str(ev_preds["generated_at_utc"].iloc[0])[:19].replace("T", " ") + " UTC (research run, not live)"),
        ])
        st.markdown(ui.chip("Magnitude probability, not a direction or trading signal", "unavail"), unsafe_allow_html=True)

# ---------------------------------------------------------------- features and attribution
ui.section_title("What the models saw")
if feat_row is None:
    ui.status("unavail", "Feature snapshot not available.", "")
else:
    rows = []
    for name in ["txt_mdna_prop_added", "txt_mdna_prop_removed", "txt_mdna_prop_substantial", "txt_mdna_lexical_distance",
                 "txt_mdna_log_len_ratio", "txt_mdna_uncertainty_change", "txt_mdna_negative_change", "txt_changed_topic_share",
                 "txt_risk_prop_added", "fin_rev_yoy", "fin_op_margin_yoy_pp", "mkt_absoc_mean_20", "mkt_absoc_mean_60",
                 "ctx_earnings_8k_in_window"]:
        c = CATALOG_BY_NAME.get(name, {})
        if name not in feat_row.index:
            rows.append({"Feature": name, "Value": "not included in this package (price-derived)",
                         "Units": c.get("units", ""), "Definition": c.get("definition", "")})
            continue
        v = feat_row[name]
        rows.append({"Feature": name, "Value": "missing (imputed in models)" if pd.isna(v) else f"{float(v):.4g}",
                     "Units": c.get("units", ""), "Definition": c.get("definition", "")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ui.caption("Observed values of selected features for this event (text features compare the current MD&A with the "
               "prior-year same-quarter MD&A). The full catalog is on the Methodology page.")

    if not ev_preds.empty and data.run_manifest:
        fp = [x for x in data.run_manifest["fold_parameters"]
              if x["test_year"] == int(ev_preds["test_year"].iloc[0]) and x["model"] == "C" and x["family"] == "logistic_l2"]
        needed = [f for f in (fp[0].get("coefficients") or {}) if f != "intercept" and not f.endswith("__missing")] if fp else []
        if fp and any(f not in feat_row.index for f in needed):
            ui.status("unavail", "Attribution not available in this package.",
                      "It needs the price-derived market features, which are not redistributed.")
        elif fp and fp[0].get("coefficients") and fp[0].get("standardization"):
            p = fp[0]
            contrib = []
            for f, coef in p["coefficients"].items():
                if f == "intercept":
                    continue
                if f.endswith("__missing"):
                    base = f[:-9]
                    z = 1.0 if pd.isna(feat_row.get(base)) else 0.0
                else:
                    x = feat_row.get(f)
                    x = p["imputation_medians"][f] if pd.isna(x) else float(x)
                    lo, hi = p["winsor_bounds"][f]
                    mu, sd = p["standardization"][f]
                    z = (min(max(x, lo), hi) - mu) / sd
                contrib.append({"Feature": f, "Coefficient": coef, "Standardized value": z, "Contribution (log-odds)": coef * z})
            cdf = pd.DataFrame(contrib)
            cdf = cdf.reindex(cdf["Contribution (log-odds)"].abs().sort_values(ascending=False).index).head(8)
            with st.expander("Model C attribution: coefficient × standardized value (largest 8)"):
                st.dataframe(cdf.round(3), hide_index=True, width="stretch")
                ui.caption("For L2 logistic regression these terms add up (with the intercept) to the raw log-odds before "
                           "Platt calibration. They describe how this fitted model weighs inputs; they are not causal effects, "
                           "not SHAP values, and a large contribution does not mean the disclosure moved the price.")

# ---------------------------------------------------------------- reveal
ui.section_title("After the session")
key = f"revealed_{eid}"
if not data.has_outcomes:
    ui.status("unavail", "Realized outcome not available here.",
              "This package does not include provider price data, so the target session's prices and label cannot be shown. "
              "Run the research pipeline locally to add them.")
elif not st.session_state.get(key):
    if st.button("Reveal realized outcome", type="primary"):
        st.session_state[key] = True
        st.rerun()
    ui.caption("Daily open and close only; no intraday path is available or shown.")
else:
    o = data.outcomes.set_index("event_id").loc[eid]
    yv = int(o["y"])
    lines = [("Open", ui.money(o["open"])), ("Close", ui.money(o["close"])),
             ("Signed open→close return (context only)", ui.pct(o["r_open_to_close"], 2, signed=True)),
             ("Absolute move", ui.pct(o["abs_r"], 2)),
             ("Threshold (2 × 60-session mean)", ui.pct(o["threshold_abs_move"], 2)),
             ("Move relative to its 60-session mean", f"{o['abs_r'] / o['m_mean_abs_oc_60']:.2f}×"),
             ("Realized label", "Large move (y = 1)" if yv == 1 else "Not a large move (y = 0)"),
             ("Threshold history", f"{int(o['threshold_n_sessions'])} sessions, {ui.day(o['threshold_window_first'])} to "
                                   f"{ui.day(o['threshold_window_last'])}")]
    if not ev_preds.empty:
        pC = ev_preds[(ev_preds["model"] == "C") & (ev_preds["family"] == "logistic_l2")]["p_cal"].iloc[0]
        pB = ev_preds[(ev_preds["model"] == "B") & (ev_preds["family"] == "logistic_l2")]["p_cal"].iloc[0]
        lines += [("Squared error, Model C", f"{(pC - yv) ** 2:.3f}  (Brier contribution; lower is better)"),
                  ("Squared error, Model B", f"{(pB - yv) ** 2:.3f}"),
                  ("Did text help on this event?", "yes" if (pC - yv) ** 2 < (pB - yv) ** 2 else
                   ("no" if (pC - yv) ** 2 > (pB - yv) ** 2 else "no difference"))]
    with st.container(border=True):
        ui.facts(lines)
    ui.caption("One event cannot show whether text adds value; see Research Results for the pooled, paired comparison. "
               "The signed return is context: the models forecast magnitude only.")
    if st.button("Hide outcome again"):
        st.session_state[key] = False
        st.rerun()
