"""Overview / Event Explorer."""

import pandas as pd
import streamlit as st

from filingshock.ui import components as ui
from filingshock.ui import data as D
from filingshock.ui.shared import data_banner, definition_strip, require_dataset

data = D.load()
ui.page_header("FilingShock",
               "Do changes in a company's 10-Q disclosures improve forecasts of an unusually large price move "
               "in the next trading session, beyond market and financial information?")
data_banner(data)
if not require_dataset(data):
    st.stop()
definition_strip(data)

table = D.event_table(data, include_prospective=True)
table["year"] = pd.to_datetime(table["acceptance_utc"]).dt.tz_convert("America/New_York").dt.year

# ---------------------------------------------------------------- filters (persist across pages)
companies = sorted(table["ticker"].unique())
years = (int(table["year"].min()), int(table["year"].max()))
STATUS_OPTS = ["All", "Eligible (final sample)", "Excluded", "Prospective (outside research sample)"]
PRED_OPTS = list(D.PREDICTION_STATUS_LABEL)[:5]
FORM_OPTS = sorted(table["form_type"].unique())

with st.container():
    c1, c2, c3, c4 = st.columns([2.2, 1.6, 1.3, 1.3])
    sel_comp = c1.multiselect("Companies", companies, default=ui.persisted("f_comp", []),
                              placeholder="All companies", key="w_comp")
    sel_years = c2.slider("Acceptance year", years[0], years[1], ui.persisted("f_years", years), key="w_years")
    sel_status = c3.selectbox("Eligibility", STATUS_OPTS, index=STATUS_OPTS.index(ui.persisted("f_status", "All")),
                              key="w_status")
    sel_forms = c4.multiselect("Form type", FORM_OPTS, default=ui.persisted("f_forms", ["10-Q"]), key="w_forms",
                               placeholder="All forms")
    c5, c6, c7 = st.columns([2.2, 2.2, 1.2])
    sel_pred = c5.multiselect("Prediction state", PRED_OPTS, default=ui.persisted("f_pred", []),
                              format_func=lambda k: D.PREDICTION_STATUS_LABEL[k], placeholder="Any state", key="w_pred")
    qual_opts = ["good", "fair", "weak"]
    sel_qual = c6.multiselect("MD&A alignment quality (heuristic)", qual_opts, default=ui.persisted("f_qual", []),
                              placeholder="Any quality", key="w_qual")
    show_outcomes = c7.toggle("Show realized outcomes", value=ui.persisted("f_out", True), key="w_out",
                              help="Hide outcomes if you want to use Historical Replay without seeing results first.")
for k, v in [("f_comp", sel_comp), ("f_years", sel_years), ("f_status", sel_status), ("f_forms", sel_forms),
             ("f_pred", sel_pred), ("f_qual", sel_qual), ("f_out", show_outcomes)]:
    ui.persist(k, v)

f = table
if sel_comp:
    f = f[f["ticker"].isin(sel_comp)]
f = f[f["year"].between(*sel_years)]
if sel_status == "Eligible (final sample)":
    f = f[f["eligibility_status"] == "eligible"]
elif sel_status == "Excluded":
    f = f[f["eligibility_status"] == "excluded"]
elif sel_status.startswith("Prospective"):
    f = f[f["eligibility_status"] == "prospective"]
if sel_forms:
    f = f[f["form_type"].isin(sel_forms)]
if sel_pred:
    f = f[f["prediction_status"].isin(sel_pred)]
if sel_qual:
    f = f[f["mdna_align_alignment_quality"].isin(sel_qual)]

active = []
if sel_comp:
    active.append("companies: " + ", ".join(sel_comp))
if sel_years != years:
    active.append(f"years {sel_years[0]}–{sel_years[1]}")
if sel_status != "All":
    active.append(sel_status.lower())
if sel_forms and sel_forms != ["10-Q"]:
    active.append("forms: " + ", ".join(sel_forms))
elif not sel_forms:
    active.append("all forms")
if sel_pred:
    active.append("prediction: " + ", ".join(D.PREDICTION_STATUS_LABEL[p] for p in sel_pred))
if sel_qual:
    active.append("alignment: " + ", ".join(sel_qual))

a1, a2 = st.columns([5, 1])
n_prosp = int((table["eligibility_status"] == "prospective").sum())
a1.markdown(f"**{len(f):,}** of {len(table):,} discovered periodic reports shown"
            + (f" (including {n_prosp} prospective forecast{'s' if n_prosp != 1 else ''})" if n_prosp else "")
            + (f" · filters: {'; '.join(active)}" if active else " · default view (original 10-Qs)"))
if a2.button("Reset filters", width="stretch", disabled=not active):
    for k in ["f_comp", "f_years", "f_status", "f_forms", "f_pred", "f_qual"]:
        st.session_state["_persist"].pop(k, None)
    for k in ["w_comp", "w_years", "w_status", "w_forms", "w_pred", "w_qual"]:
        st.session_state.pop(k, None)
    st.rerun()

if f.empty:
    ui.status("info", "No events match these filters.",
              "Widen the year range, clear the company or prediction-state filters, or use Reset filters.")
    st.stop()

# ---------------------------------------------------------------- table
f = f.sort_values("acceptance_utc", ascending=False).reset_index(drop=True)
view = pd.DataFrame({
    "Company": f["ticker"],
    "Form": f["form_type"],
    "Fiscal period": f["fiscal_period_label"].fillna(f["report_date"]),
    "Accepted (ET)": f["acceptance_utc"].map(ui.ts_et),
    "Target session": f["target_session"].map(ui.day),
    "Availability": f["availability_method"].map({"acceptance_plus_lag": "acceptance + 15 min (assumed)",
                                                  "next_day_dissemination_plus_lag": "next-day 06:00 ET + 15 min (assumed)",
                                                  "observed_by_discovery": "observed by discovery (prospective)"}).fillna("—"),
    "Eligibility": f.apply(lambda r: {"eligible": "eligible", "prospective": "prospective (outside research sample)"}.get(
        r["eligibility_status"], f"excluded: {r['exclusion_reason']}"), axis=1),
    "Prediction": f["prediction_status"].map(D.PREDICTION_STATUS_LABEL),
    "P(large move), C": pd.to_numeric(f["p_C"], errors="coerce") * 100,
    "Threshold |move|": pd.to_numeric(f["threshold_abs_move"], errors="coerce") * 100,
})
if show_outcomes:
    view["Realized |move|"] = pd.to_numeric(f["abs_r"], errors="coerce") * 100
    view["Large move?"] = f["y"].map({1: "yes", 0: "no"}).fillna("—")
view["Data quality"] = f.apply(lambda r: ", ".join(
    [x for x in (r.get("mdna_align_alignment_quality"),) if isinstance(x, str) and x != "good"]
    + (["flags: " + r["data_quality_flags"].strip("[]").replace('"', "")] if isinstance(r["data_quality_flags"], str)
       and r["data_quality_flags"] not in ("[]", "") else [])) or "ok", axis=1)
view["Event id"] = f["event_id"]

ui.caption("Probabilities are calibrated Model C (text-augmented) out-of-time estimates from the primary run; prospective "
           "rows show the production model's forecast as recorded in the ledger before the session (not part of the "
           "research metrics). 'Threshold' is 2 × the stock's mean absolute open-to-close move over the previous 60 sessions. "
           "Select a row to open its comparison and replay.")
evt = st.dataframe(
    view, width="stretch", hide_index=True, height=440, on_select="rerun", selection_mode="single-row",
    key="explorer_table",
    column_config={
        "P(large move), C": st.column_config.NumberColumn(format="%.1f%%", help="Calibrated probability that |open→close| exceeds the threshold. Magnitude only."),
        "Threshold |move|": st.column_config.NumberColumn(format="%.2f%%"),
        "Realized |move|": st.column_config.NumberColumn(format="%.2f%%"),
        "Event id": st.column_config.TextColumn(width="small"),
    })
rows = evt.selection.rows if evt and evt.selection else []
if rows:
    ui.set_selected_event(f.loc[rows[0], "event_id"])

csv = view.assign(dataset_id=data.dataset["dataset_id"], run_id=data.run_id or "none",
                  units="percent columns are percentages of the opening price").to_csv(index=False).encode()
st.download_button("Export filtered table (CSV)", csv, file_name=f"filingshock_events_{data.dataset['dataset_id']}.csv",
                   mime="text/csv")

# ---------------------------------------------------------------- selected event
sel = ui.selected_event_id(table)
ui.section_title("Selected event")
if sel is None:
    ui.status("info", "No event selected.", "Select a row in the table to see its identity and open its views.")
elif table.loc[table["event_id"] == sel, "eligibility_status"].iloc[0] == "prospective":
    r = table[table["event_id"] == sel].iloc[0]
    ui.identity_strip(r)
    ui.status("info", "Prospective forecast, recorded before the session",
              "This filing is after the research filing window. Its forecast was generated by the production models "
              "(fit ≤ Y−2, calibrated on Y−1) and appended to the hash-chained ledger before the decision cutoff. It is "
              "not part of any research metric. Comparison and replay views cover the research sample only.")
    ui.facts([("Observed by discovery", ui.ts_et(r["observed_at_utc"], True)),
              ("Forecast generated", ui.ts_et(r["generated_at_utc"], True)),
              ("Decision cutoff", ui.ts_et(r["decision_cutoff_utc"], True)),
              ("P(large move): Model C / Model B", f"{ui.pct(r['p_C'], 1)} / {ui.pct(r['p_B'], 1)}"),
              ("Threshold |move| (known before the cutoff)", ui.pct(r["threshold_abs_move"], 2)),
              ("Model version", f"{r['model_version']} · dataset {r['forecast_dataset_id']}"),
              ("Reference filing", str(r["reference_filing_id"])), ("Forecast id", str(r["forecast_id"]))])
else:
    r = table[table["event_id"] == sel].iloc[0]
    ui.identity_strip(r)
    b1, b2, b3 = st.columns(3)
    if b1.button("Open filing comparison", width="stretch", disabled=r["eligibility_status"] != "eligible"):
        st.switch_page("pages/comparison.py")
    if b2.button("Open prediction & replay", width="stretch", disabled=r["eligibility_status"] != "eligible"):
        st.switch_page("pages/replay.py")
    if b3.button("Explain eligibility", width="stretch"):
        st.session_state["_explain"] = sel
    if r["eligibility_status"] != "eligible":
        ui.status("unavail", "This filing is outside the final sample.",
                  f"Excluded at stage '{r['exclusion_stage']}' because: {r['exclusion_reason']}. "
                  "Comparison and prediction views are available only for final-sample events.")
    if st.session_state.get("_explain") == sel:
        from filingshock.audit import explain_event
        exp = explain_event(D.layout(), sel)
        st.dataframe(pd.DataFrame(exp["stages"]), hide_index=True, width="stretch")
        ui.facts([("Acceptance", ui.ts_et(exp["timing"]["acceptance_utc"], True)),
                  ("Assumed available", ui.ts_et(exp["timing"]["assumed_available_utc"], True) + " — assumption, not observed"),
                  ("Decision cutoff", ui.ts_et(exp["timing"]["decision_cutoff_utc"], True)),
                  ("Timing flags", ", ".join(exp["timing"]["timing_flags"]) or "none"),
                  ("Missing features", "; ".join(f"{g}: {', '.join(v)}" for g, v in exp.get("missing_features", {}).items() if v) or "none")])
