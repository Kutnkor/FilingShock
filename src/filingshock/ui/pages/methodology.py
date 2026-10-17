"""Methodology & Data Quality: concrete answers drawn from configuration, manifests and saved artifacts."""

import json

import pandas as pd
import streamlit as st

from filingshock.config import PROJECT_ROOT
from filingshock.features.catalog import CATALOG
from filingshock.ui import components as ui
from filingshock.ui import data as D
from filingshock.ui.shared import data_banner, require_dataset

data = D.load()
ui.page_header("Methodology & Data Quality",
               "How events, features, models and evaluations are constructed, and where the data is weak.")
data_banner(data)
if not require_dataset(data):
    st.stop()

man = data.run_manifest or {}
cfg = man.get("config") or {}
ev = data.events
final = ev[ev["eligibility_status"] == "eligible"]
tabs = st.tabs(["Method", "Data quality", "Feature catalog", "Limitations"])

with tabs[0]:
    tgt = cfg.get("target", {})
    tim = cfg.get("timing", {})
    q = [
        ("What exactly is predicted?",
         f"For each eligible 10-Q event, the probability that the target session's open-to-close move |close/open − 1| "
         f"exceeds {tgt.get('multiplier', 2.0):g} × the mean absolute open-to-close move of the previous "
         f"{tgt.get('lookback_sessions', 60)} valid sessions. A large move in either direction counts. The overnight gap "
         "and the intraday high–low range are excluded, so a big intraday reversal that ends near the open is not a large move."),
        ("When is the prediction made?",
         f"At the decision cutoff, {tim.get('cutoff_minutes_before_open', 30)} minutes before the target session's regular "
         "open (09:00 ET on normal days). All research predictions are retrospective: computed in a 2026 research run with "
         "models that only used information available before each event's cutoff."),
        ("What information is permitted?",
         "Price bars from sessions before the target session; Company Facts values reported by filings accepted (plus a "
         "15-minute lag) before the cutoff, with the latest admissible report winning; 8-K Item 2.02 filings available "
         "by the cutoff; the current and reference 10-Q text. Target-session prices live only in the outcome table, "
         "which no feature builder reads."),
        ("How are events selected?",
         f"{len(ev):,} periodic reports discovered for {ev['ticker'].nunique()} companies; {len(final):,} original 10-Qs "
         f"from {final['ticker'].nunique()} companies pass every stage (form, timing, uniqueness, prices, MD&A extraction, "
         "prior-year reference). Filings accepted during regular trading are excluded because the market could react "
         "before the cutoff. The full funnel is on the Research Results page."),
        ("How is the acceptance time established?",
         "From the 'Accepted' field of each filing's EDGAR index page (Eastern time). The submissions API value is kept "
         "for comparison: the audit found it later than the index page by exactly the Eastern UTC offset for some filings, "
         "so it is not used for timing where the index page is available. Public availability is still an assumption: "
         "acceptance + 15 minutes, or 06:00 ET + 15 minutes on the next business day for submissions after 17:30 ET."),
        ("How are sections extracted?",
         "The filing HTML is converted to ordered text and table blocks (hidden inline-XBRL content, scripts and styles "
         "removed). Short standalone headings matching 'Item 2 … Management's Discussion and Analysis' are candidates; "
         "the candidate that opens the longest span before the next Item 3/Item 4/Part II heading is chosen, which rejects "
         "table-of-contents entries and cross-references. Financial tables are counted but excluded from paragraphs."),
        ("How are reports aligned?",
         "Deterministically: exact matches, then number-masked exact matches, then greedy one-to-one matching on word and "
         "word-pair Jaccard similarity (≥ 0.40). Unmatched paragraphs are added or removed; matches outside the longest "
         "in-order chain are flagged as moved. Every pair keeps its scores; Jaccard < 0.55 is labelled low-confidence."),
        ("How are financial periods normalised?",
         "Quarterly flows use reported three-month values (75–105 days) or, for cash flow, year-to-date differences only when "
         "both values share the same fiscal-year start. Balance-sheet items are instants at period end. Growth rates use one "
         "XBRL concept for both periods. Missing values remain missing, with indicators; zero or negative denominators are undefined."),
        ("How are models compared?",
         "Constant baseline, A (market + context), B (A + financial) and C (B + text) use the same events, folds, "
         "preprocessing, five-setting hyperparameter budget and calibration. The primary comparison is the paired "
         "ΔBrier = Brier(C) − Brier(B), with a quarter-clustered bootstrap interval."),
        ("How is calibration performed?",
         "Platt scaling on the year before each test year, using predictions from a model that never saw that year. If the "
         "fitted slope is not positive, a pre-specified intercept-only adjustment is used instead."),
        ("Which results are reproducible?",
         "Running `filingshock experiment --name primary` twice on the same data produced identical per-event probabilities "
         "(maximum difference 0.0) and identical intervals. `filingshock verify-artifacts` recomputes the headline metrics "
         "from the saved predictions and compares them with metrics.json."),
        ("Which assumptions remain unresolved?",
         "Exact public availability times (proxy only); whether Company Facts contains every value exactly as originally "
         "filed; whether the fixed cohort's results generalise; concurrent news not captured by 8-K Item 2.02."),
    ]
    for title, body in q:
        with st.expander(title, expanded=title.startswith("What exactly")):
            st.markdown(ui.esc(body))
    if man.get("fold_parameters"):
        fp = pd.DataFrame(man["fold_parameters"])
        fp = fp[(fp["family"] == "logistic_l2") & fp["model"].isin(["B", "C"])]
        fp["calibration method"] = fp["calibration"].map(lambda c: c["method"])
        fp["Platt a"] = fp["calibration"].map(lambda c: round(c["a"], 3))
        fp["Platt b"] = fp["calibration"].map(lambda c: round(c["b"], 3))
        fp["chosen C (L2)"] = fp["param"]
        with st.expander("Per-fold hyperparameters and calibration (primary run)"):
            st.dataframe(fp[["test_year", "model", "chosen C (L2)", "calibration method", "Platt a", "Platt b"]],
                         hide_index=True, width="stretch")

with tabs[1]:
    ext = D.extractions()
    ui.section_title("Section extraction")
    if ext is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Filings parsed", f"{len(ext):,}")
        c2.metric("MD&A extracted", f"{(ext['mdna_status'] == 'extracted').mean():.1%}")
        c3.metric("Median MD&A words", f"{int(ext['mdna_n_words'].median()):,}")
        rs = ext["risk_status"].value_counts().rename_axis("risk section status").reset_index(name="filings")
        rs["meaning"] = rs["risk section status"].map({
            "extracted": "full risk-factor text present", "no_material_changes": "states no material changes since the 10-K",
            "short_reference": "only refers to another report's risk factors", "absent": "no Part II Item 1A heading",
            "failed": "heading found, extraction failed"})
        ui.html_table(rs, right={"filings"}, caption_text="A missing or short risk section is not evidence of lower risk.")
        meth = ext["mdna_method"].value_counts().rename_axis("MD&A extraction method").reset_index(name="filings")
        ui.html_table(meth, right={"filings"}, caption_text="'nonstandard_item_number_heading': a 10-Q that labels its MD&A "
                      "'Item 7' (found in the manual audit). Heuristic quality scores are rule-based, not probabilities.")
    audit_p = PROJECT_ROOT / "docs" / "extraction_audit.csv"
    if audit_p.exists():
        with st.expander("Manual extraction audit (stratified sample, one filing per company)"):
            st.dataframe(pd.read_csv(audit_p), hide_index=True, width="stretch")
            ui.caption("Reviewed by reading the first, second and last paragraphs of each extracted section and the risk "
                       "section classification. Findings are summarised in docs/extraction_audit.md.")

    ui.section_title("Timing")
    am = final["availability_method"].value_counts().rename_axis("availability method (final sample)").reset_index(name="events")
    ui.html_table(am, right={"events"})
    if "acceptance_source" in ev:
        src = ev["acceptance_source"].value_counts().rename_axis("acceptance source (all discovered)").reset_index(name="filings")
        ui.html_table(src, right={"filings"})
    rec = D.jsonfile(D.layout().processed / "acceptance_reconciliation.json")
    if rec:
        ui.facts([("Filings with an index-page acceptance time", f"{rec['n_checked']:,}"),
                  ("Submissions-API value differs from the index page", f"{rec['n_discrepant']:,} ({rec['share_discrepant']:.1%})"),
                  ("Observed differences (seconds → filings)", ", ".join(f"{k}: {v}" for k, v in rec["difference_counts"].items())),
                  ("Final-sample events whose timing classification changed",
                   f"{rec.get('n_classification_changed', 'not computed')}")])
    flags = final["timing_flags"].map(lambda s: json.loads(s) if isinstance(s, str) else []).explode().dropna()
    dq = final["data_quality_flags"].map(lambda s: json.loads(s) if isinstance(s, str) else []).explode().dropna()
    allf = pd.concat([flags, dq]).value_counts().rename_axis("flag").reset_index(name="final-sample events")
    if len(allf):
        ui.html_table(allf, right={"final-sample events"},
                      caption_text="Flags are recorded, not hidden. 'large_overnight_gap_in_threshold_window' marks a >35% "
                                   "overnight gap (e.g. a special dividend or corporate action) inside the 90-session window; "
                                   "open-to-close returns are unaffected because open and close share one adjustment basis.")

    ui.section_title("Price coverage")
    pf = (data.dataset or {}).get("price_files", {})
    if pf:
        pcov = pd.DataFrame([{"symbol": k, "status": v.get("status"), "rows": v.get("rows"), "last session": v.get("last"),
                              "retrieved (UTC)": v.get("retrieved_at")} for k, v in pf.items()])
        ui.html_table(pcov.fillna("—"), right={"rows"},
                      caption_text="Provider: yfinance (unofficial Yahoo endpoint). OHLC are split-adjusted, not dividend-adjusted. "
                                   "Bars were checked session-by-session against the XNYS calendar.")
    else:
        ui.status("unavail", "Price data not included in this package.", "")

with tabs[2]:
    cat = pd.DataFrame(CATALOG)
    groups = {"market": "Model A", "context": "Model A", "financial": "Model B adds", "text_length": "Model C adds",
              "text_dictionary": "Model C adds", "text_paragraph_change": "Model C adds"}
    cat.insert(1, "model", cat["group"].map(groups))
    st.dataframe(cat, hide_index=True, width="stretch", height=560)
    ui.caption("Every model input with its definition, source, units, transformation and availability rule. Preprocessing "
               "(median imputation with missingness indicators, 1st/99th-percentile winsorisation, standardisation) is fitted "
               "on each fold's training block only.")

with tabs[3]:
    n_del = int((ev["exclusion_reason"] == "price_history_unavailable_from_provider").sum())
    ms = (data.metrics or {}).get("sample", {})
    feats = data.features
    n_earn = int(feats["ctx_earnings_8k_in_window"].sum()) if feats is not None and "ctx_earnings_8k_in_window" in feats else "Some"
    lim = [
        ("Fixed, survivor-heavy cohort", f"14 software companies with free price history; {n_del} filings from 7 acquired or "
                                          "delisted companies drop out at the price stage. Results describe this cohort only."),
        ("Availability is a proxy", "No historical first-public-availability timestamp exists; acceptance + 15 minutes "
                                    "(or next-day 06:00 ET) is assumed. Live forecasts would record observed times instead."),
        ("Earnings announcements", f"{n_earn} "
                                   "final events coincide with an earnings 8-K in the same window; for those, the next session "
                                   "mainly reflects the earnings release, not the 10-Q text."),
        ("Small independent sample", f"{ms.get('n_events', '—')} out-of-time events with {ms.get('n_positive', '—')} large moves: "
                                     "the primary interval spans roughly ±0.005 in Brier score, so smaller differences "
                                     "cannot be distinguished from zero."),
        ("Concurrent news", "Only 8-K Item 2.02 is used as context; analyst actions, press releases and macro news are not observed."),
        ("Archived data vintages", "Company Facts and Yahoo prices were retrieved in October 2026. Earlier vintages (before "
                                   "restatements or split re-adjustments) are not available; admissibility is enforced by "
                                   "the reporting filing's acceptance time, not by a historical snapshot."),
        ("Parsing", "Extraction succeeded for every parsed filing, but quality was audited manually on a small sample only; "
                    "paragraph alignment quality is a heuristic label."),
        ("Prediction is not causation or profit", "A predictive association, or its absence, says nothing about the causal "
                                                  "effect of disclosures, and a magnitude probability is not a trading strategy."),
    ]
    for t, b in lim:
        st.markdown(f"**{ui.esc(t)}.** {ui.esc(b)}")
