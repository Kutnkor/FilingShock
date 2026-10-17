"""Event construction, eligibility funnel, targets and feature snapshots.

One supervised observation = one company-filing event. Forecast inputs (features.parquet) and
outcomes (outcomes.parquet) are written to separate tables; feature builders only ever receive
price history truncated before the target session.

The per-event functions (pre_cutoff_market, prev_event_abs_ratio, document_stage_exclusion,
event_document_features) are shared by the historical build and by single_event_features, the
feature path for prospective forecasts, so both compute features in exactly the same way.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from filingshock import ALIGNER_VERSION, PARSER_VERSION, __version__
from filingshock.config import PROJECT_ROOT, Config
from filingshock.discovery import PERIODIC_FORMS, load_filings
from filingshock.features.financial import FIN_FEATURES, financial_features, load_company_facts
from filingshock.features.market import (
    CONTEXT_FEATURES,
    MARKET_FEATURES,
    benchmark_absoc,
    context_features,
    stock_features,
)
from filingshock.features.text import (
    LM_FILE,
    TEXT_FEATURES,
    PointInTimeLexicon,
    TopicLexicon,
    load_section,
    risk_prop_added,
    section_text_features,
)
from filingshock.parsing.align import align_paragraphs
from filingshock.reference import select_reference
from filingshock.sources.market import PriceStore, bar_quality
from filingshock.storage import DataLayout, write_json_atomic, write_parquet_atomic
from filingshock.target import TargetSpec, Threshold, historical_threshold, realized_outcome
from filingshock.timing import SessionCalendar, TimingPolicy, classify_timing

log = logging.getLogger(__name__)

FUNNEL_STAGES = [
    ("discovered", "Periodic reports discovered in the filing window"),
    ("supported_form", "Original 10-Q (amendments and 10-Ks are outside the primary study)"),
    ("eligible_timing", "Accepted after the prior close and available by the decision cutoff"),
    ("unique_event", "One filing per company and target session"),
    ("valid_price_history", "Target-session bar and 60-session threshold history are valid"),
    ("mdna_extracted", "MD&A section extracted from the current filing"),
    ("reference_usable", "Prior-year same-quarter 10-Q found and its MD&A extracted"),
]
FINAL_STAGE = "final_sample"
OVERNIGHT_GAP_FLAG = 0.35


def _bars(store: PriceStore, symbol: str, max_abs: float) -> pd.DataFrame | None:
    df = store.load(symbol)
    if df is None or df.empty:
        return None
    q = bar_quality(df, max_abs)
    q["prev_close"] = q["close"].shift(1)
    return q.reset_index(drop=True)


def _fiscal_labels(facts: pd.DataFrame) -> dict[str, str]:
    if facts is None or facts.empty:
        return {}
    f = facts[facts["form"] == "10-Q"].dropna(subset=["fy", "fp"])
    lab = f.groupby("accn").apply(lambda g: f"FY{int(g['fy'].mode().iloc[0])} {g['fp'].mode().iloc[0]}",
                                  include_groups=False)
    return lab.to_dict()


def _benchmarks(cfg: Config, store: PriceStore, max_abs: float) -> dict[str, pd.DataFrame | None]:
    return {k: _bars(store, s, max_abs) for k, s in cfg["universe"]["benchmark_symbols"].items()}


# ---------------------------------------------------------------------------- per-event functions
def pre_cutoff_market(bars: pd.DataFrame, bench: dict, target_session, spec: TargetSpec
                      ) -> tuple[pd.DataFrame, Threshold, dict]:
    """History, past-only threshold and market features for one event.

    Only bars strictly before the target session are read (the stock's and the benchmarks'), so the
    target-session bar, if it exists, cannot influence anything returned here.
    """
    sess = pd.Timestamp(target_session)
    history = bars[bars["session"] < sess]
    sf = stock_features(history, spec.lookback, spec.multiplier)
    for key, name in (("market", "mkt_spy_absoc_20"), ("sector", "mkt_igv_absoc_20")):
        b = bench.get(key)
        sf[name] = benchmark_absoc(b[b["session"] < sess]) if b is not None else None
    return history, historical_threshold(history, spec), sf


def prev_event_abs_ratio(earlier: pd.DataFrame, cutoff_utc) -> float | None:
    """|r| / m of the company's most recent earlier event whose outcome had matured by the cutoff.

    earlier: one company's events with decision_cutoff_utc, abs_ratio and outcome_available_utc.
    """
    known = earlier[(earlier["decision_cutoff_utc"] < cutoff_utc) & (earlier["outcome_available_utc"] <= cutoff_utc)]
    return float(known.sort_values("decision_cutoff_utc")["abs_ratio"].iloc[-1]) if len(known) else None


def document_stage_exclusion(mdna_status: str, download_failed: bool, reference: dict | None,
                             reference_mdna_status: str) -> tuple[str, str] | None:
    """Extraction and reference stages for one event: (stage, reason) if excluded, else None.

    reference: the primary-policy ReferenceChoice as a mapping (status, reason, reference_accession).
    """
    if mdna_status != "extracted":
        reason = "mdna_" + (mdna_status if mdna_status != "failed" else "extraction_failed")
        if mdna_status == "not_attempted" and download_failed:
            reason = "document_download_failed"
        return "mdna_extracted", reason
    if reference is None or reference["status"] != "found":
        return "reference_usable", "reference_missing:" + (reference["reason"] if reference is not None else "not_evaluated")
    if reference_mdna_status != "extracted":
        return "reference_usable", "reference_mdna_not_extracted"
    return None


@dataclass
class DocumentFeatureContext:
    """Resources shared by every event's document-based features (loaded once per build)."""
    layout: DataLayout
    lag: pd.Timedelta
    lexicon: PointInTimeLexicon
    topics: TopicLexicon
    accn_available: dict
    earnings_8k: pd.DataFrame
    facts_cache: dict = field(default_factory=dict)
    fiscal_labels: dict = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: Config, layout: DataLayout, filings: pd.DataFrame) -> "DocumentFeatureContext":
        lag = pd.Timedelta(minutes=int(cfg["timing"]["availability_lag_minutes"]))
        return cls(layout, lag, PointInTimeLexicon.from_csv(layout.raw_resources / LM_FILE),
                   TopicLexicon(PROJECT_ROOT / cfg["features"]["topic_lexicon"]),
                   dict(zip(filings["accession_number"], filings["acceptance_utc"] + lag)),
                   filings[filings["is_earnings_8k"]])

    def company_facts(self, cik: int) -> pd.DataFrame:
        if cik not in self.facts_cache:
            p = self.layout.raw_sec / "companyfacts" / f"CIK{cik:010d}.json"
            self.facts_cache[cik] = load_company_facts(p) if p.exists() else pd.DataFrame()
            self.fiscal_labels.update(_fiscal_labels(self.facts_cache[cik]))
        return self.facts_cache[cik]


@dataclass
class EventDocumentFeatures:
    context: dict
    context_provenance: dict
    financial: dict
    financial_provenance: dict
    text: dict
    observations: dict
    alignment: dict


def event_document_features(fc: DocumentFeatureContext, r, reference_accession: str) -> EventDocumentFeatures:
    """Context, financial and text features for one event, from information available at its cutoff.

    r: event row with event_id, cik, accession_number, acceptance_utc, report_date, prev_close_utc,
    decision_cutoff_utc and target_session. Both sections must already be extracted.
    """
    cik = int(r["cik"])
    cutoff = r["decision_cutoff_utc"]
    cf, cprov = context_features(cutoff, r["prev_close_utc"], r["acceptance_utc"], r["report_date"],
                                 fc.earnings_8k[fc.earnings_8k["cik"] == cik], fc.lag)
    ff, fprov = financial_features(fc.company_facts(cik), fc.accn_available, cutoff, pd.Timestamp(r["report_date"]))
    cur_doc, ref_doc = load_section(fc.layout, r["accession_number"]), load_section(fc.layout, reference_accession)
    year = pd.Timestamp(r["target_session"]).year
    tf, obs, al_sum = _text_for_event(fc.layout, r["event_id"], cur_doc, ref_doc, fc.lexicon, fc.topics, year)
    return EventDocumentFeatures(cf, cprov, ff, fprov, tf, obs, al_sum)


# ---------------------------------------------------------------------------- historical build
def build_candidates(cfg: Config, layout: DataLayout, filings: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Stages that need no documents: form, timing, uniqueness, price history and target.

    filings: defaults to the discovered filings table; single_event_features passes one company's subset.
    """
    filings = load_filings(layout) if filings is None else filings
    cal = SessionCalendar(cfg["timing"]["exchange_calendar"])
    pol = TimingPolicy.from_config(cfg)
    spec = TargetSpec.from_config(cfg)
    max_abs = float(cfg["target"]["max_abs_return_sanity"])
    store = PriceStore(layout, cfg["market_data"]["provider"])

    disc = filings[filings["form_type"].isin(PERIODIC_FORMS) & filings["in_window"]].copy()
    disc = disc.sort_values("acceptance_utc").reset_index(drop=True)
    disc["event_id"] = "ev-" + disc["accession_number"]
    disc["exclusion_stage"] = None
    disc["exclusion_reason"] = None
    disc["data_quality_flags"] = [[] for _ in range(len(disc))]

    def exclude(mask, stage, reason):
        m = mask & disc["exclusion_stage"].isna()
        disc.loc[m, "exclusion_stage"] = stage
        disc.loc[m, "exclusion_reason"] = reason if isinstance(reason, str) else reason[m]

    exclude(disc["form_type"].str.endswith("/A"), "supported_form", "amendment_not_primary_form")
    exclude(disc["form_type"] != "10-Q", "supported_form", "form_" + disc["form_type"].str.replace("/", "_") + "_outside_primary_study")

    # --- timing
    timing_cols = {k: [] for k in ["assumed_available_utc", "availability_method", "target_session",
                                   "target_open_utc", "target_close_utc", "decision_cutoff_utc",
                                   "prev_session", "prev_close_utc", "timing_status", "timing_reason",
                                   "timing_flags"]}
    for r in disc.itertuples():
        t = classify_timing(r.acceptance_utc, r.filing_date, cal, pol)
        timing_cols["assumed_available_utc"].append(t.assumed_available_utc)
        timing_cols["availability_method"].append(t.availability_method)
        timing_cols["target_session"].append(t.target_session)
        timing_cols["target_open_utc"].append(t.target_open_utc)
        timing_cols["target_close_utc"].append(t.target_close_utc)
        timing_cols["decision_cutoff_utc"].append(t.decision_cutoff_utc)
        timing_cols["prev_session"].append(t.prev_session)
        timing_cols["prev_close_utc"].append(t.prev_close_utc)
        timing_cols["timing_status"].append(t.status)
        timing_cols["timing_reason"].append(t.reason)
        timing_cols["timing_flags"].append(t.flags)
    for k, v in timing_cols.items():
        disc[k] = v
    for c in ["assumed_available_utc", "target_open_utc", "target_close_utc", "decision_cutoff_utc", "prev_close_utc"]:
        disc[c] = pd.to_datetime(disc[c], utc=True)
    disc["target_session"] = pd.to_datetime(disc["target_session"])
    exclude(disc["timing_status"] != "eligible", "eligible_timing", disc["timing_reason"].fillna("timing_unknown"))

    # --- uniqueness: earliest-accepted original 10-Q wins per (company, target session)
    alive = disc["exclusion_stage"].isna()
    dup = disc[alive].sort_values("acceptance_utc").duplicated(["cik", "target_session"], keep="first")
    dup_ids = set(disc[alive].sort_values("acceptance_utc")[dup]["event_id"])
    exclude(disc["event_id"].isin(dup_ids), "unique_event", "duplicate_company_target_session")

    # --- prices, threshold, outcome, market features
    outcome_rows, feat_rows = [], []
    bars_cache: dict[str, pd.DataFrame | None] = {}
    bench = _benchmarks(cfg, store, max_abs)
    for i, r in disc.iterrows():
        if pd.notna(r["exclusion_stage"]):
            continue
        sym = r["ticker"]
        if sym not in bars_cache:
            bars_cache[sym] = _bars(store, sym, max_abs)
        bars = bars_cache[sym]
        if bars is None:
            disc.at[i, "exclusion_stage"], disc.at[i, "exclusion_reason"] = "valid_price_history", "price_history_unavailable_from_provider"
            continue
        pos = bars.index[bars["session"] == r["target_session"]]
        if len(pos) == 0:
            disc.at[i, "exclusion_stage"], disc.at[i, "exclusion_reason"] = "valid_price_history", "target_session_bar_missing"
            continue
        p = int(pos[0])
        history, th, sf = pre_cutoff_market(bars, bench, r["target_session"], spec)   # strictly before the target session
        oc = realized_outcome(bars.iloc[p], th)
        flags = list(r["timing_flags"])
        win = history.tail(spec.max_window)
        if win["prev_close"].notna().any():
            gaps = (win["open"] / win["prev_close"] - 1).abs()
            if (gaps > OVERNIGHT_GAP_FLAG).any():
                flags.append("large_overnight_gap_in_threshold_window")
        if bars.iloc[p]["flag_zero_volume"]:
            flags.append("target_session_zero_volume")
        if win["flag_range_inconsistent"].any():
            flags.append("range_inconsistent_bar_in_window")
        disc.at[i, "data_quality_flags"] = flags
        if th.status != "ok":
            disc.at[i, "exclusion_stage"], disc.at[i, "exclusion_reason"] = "valid_price_history", th.status
            continue
        if oc.status != "ok":
            disc.at[i, "exclusion_stage"], disc.at[i, "exclusion_reason"] = "valid_price_history", oc.status
            continue
        outcome_rows.append({
            "event_id": r["event_id"], "target_session": r["target_session"], "open": oc.open, "close": oc.close,
            "r_open_to_close": oc.r, "abs_r": oc.a, "m_mean_abs_oc_60": th.m, "threshold_abs_move": th.threshold,
            "y": oc.y, "threshold_window_first": th.window_first_session, "threshold_window_last": th.window_last_session,
            "threshold_n_sessions": th.n_valid, "threshold_sessions": json.dumps(th.sessions_used),
            "outcome_available_utc": r["target_close_utc"], "abs_ratio": oc.a / th.m,
        })
        feat_rows.append({"event_id": r["event_id"], **sf})
    outcomes = pd.DataFrame(outcome_rows)
    mfeats = pd.DataFrame(feat_rows)
    # previous event information whose outcome was already known at this cutoff
    if not outcomes.empty:
        tmp = disc[["event_id", "cik", "decision_cutoff_utc"]].merge(outcomes[["event_id", "abs_ratio", "outcome_available_utc"]], on="event_id")
        prev_vals = {}
        for _, g in tmp.groupby("cik"):
            for eid, cutoff in zip(g["event_id"], g["decision_cutoff_utc"]):
                prev_vals[eid] = prev_event_abs_ratio(g, cutoff)
        mfeats["mkt_prev_event_abs_ratio"] = mfeats["event_id"].map(prev_vals)
    meta = {"calendar": cal.name, "timing_policy": pol.__dict__, "target_spec": spec.__dict__}
    return disc, {"outcomes": outcomes, "market_features": mfeats, "meta": meta}


def needed_accessions(disc: pd.DataFrame, filings: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference choices for survivors so far, and the set of filings whose sections are needed."""
    tol = int(cfg["reference"]["period_tolerance_days"])
    q10 = filings[filings["form_type"] == "10-Q"].copy()
    alive = disc[disc["exclusion_stage"].isna()]
    refs = []
    for _, r in alive.iterrows():
        for pol in ("prior_year_same_quarter", "nearest_prior_10q"):
            ch = select_reference(q10, r, pol, tol)
            refs.append({"event_id": r["event_id"], **ch.__dict__})
    refs = pd.DataFrame(refs)
    acc = set(alive["accession_number"]) | set(refs["reference_accession"].dropna())
    return refs, q10[q10["accession_number"].isin(acc)]


def build_event_tables(cfg: Config, layout: DataLayout, disc: pd.DataFrame, parts: dict,
                       refs: pd.DataFrame, extractions: pd.DataFrame) -> dict:
    """Extraction/reference stages, text and financial features, final tables and funnel."""
    filings = load_filings(layout)
    ext = extractions.set_index("accession_number")
    primary = refs[refs["policy"] == cfg["reference"]["primary_policy"]].set_index("event_id")
    secondary = refs[refs["policy"] == cfg["reference"]["secondary_policy"]].set_index("event_id")

    def mstatus(acc):
        return ext.loc[acc, "mdna_status"] if acc in ext.index else "not_attempted"

    for i, r in disc.iterrows():
        if pd.notna(r["exclusion_stage"]):
            continue
        ch = primary.loc[r["event_id"]] if r["event_id"] in primary.index else None
        excl = document_stage_exclusion(mstatus(r["accession_number"]), r["accession_number"] in ext.index, ch,
                                        mstatus(ch["reference_accession"]) if ch is not None else "not_attempted")
        if excl is not None:
            disc.at[i, "exclusion_stage"], disc.at[i, "exclusion_reason"] = excl
    disc["eligibility_status"] = np.where(disc["exclusion_stage"].isna(), "eligible", "excluded")

    # ---- features for final events
    fc = DocumentFeatureContext.load(cfg, layout, filings)
    text_rows, fin_rows, ctx_rows, obs_rows, align_rows = [], [], [], [], []
    final = disc[disc["eligibility_status"] == "eligible"]
    for _, r in final.iterrows():
        d = event_document_features(fc, r, primary.loc[r["event_id"], "reference_accession"])
        ctx_rows.append({"event_id": r["event_id"], **d.context, "ctx_provenance": json.dumps(d.context_provenance, default=str)})
        fin_rows.append({"event_id": r["event_id"], **d.financial, "fin_provenance": json.dumps(d.financial_provenance, default=str)})
        text_rows.append({"event_id": r["event_id"], **d.text})
        obs_rows.append({"event_id": r["event_id"], **d.observations})
        align_rows.append({"event_id": r["event_id"], **d.alignment})
    labels = fc.fiscal_labels

    feats = parts["market_features"].merge(pd.DataFrame(ctx_rows), on="event_id", how="right")
    feats = feats.merge(pd.DataFrame(fin_rows), on="event_id", how="left").merge(pd.DataFrame(text_rows), on="event_id", how="left")
    feats["feature_version"] = cfg["features"]["version"]
    feats["computed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    feats = feats.merge(disc[["event_id", "decision_cutoff_utc"]], on="event_id", how="left")

    events = disc.copy()
    events["fiscal_period_label"] = events["accession_number"].map(labels)
    events = events.merge(primary[["reference_accession", "reference_report_date", "period_gap_days", "reason"]]
                          .rename(columns={"reference_accession": "reference_filing_id",
                                           "reference_report_date": "reference_period_end",
                                           "period_gap_days": "reference_gap_days",
                                           "reason": "reference_missing_reason"}),
                          left_on="event_id", right_index=True, how="left")
    events = events.merge(secondary[["reference_accession"]].rename(columns={"reference_accession": "secondary_reference_filing_id"}),
                          left_on="event_id", right_index=True, how="left")
    sha = ext["source_sha256"].to_dict() if "source_sha256" in ext else {}
    url = ext["source_url"].to_dict() if "source_url" in ext else {}
    events["source_document_hash"] = events["accession_number"].map(sha)
    events["source_url"] = events["accession_number"].map(url)
    events["observed_at_utc"] = pd.NaT   # historical reconstruction: never observed live
    events["prediction_category_default"] = "retrospective_out_of_time"
    events["data_quality_flags"] = events["data_quality_flags"].apply(json.dumps)
    events["timing_flags"] = events["timing_flags"].apply(json.dumps)
    events = events.merge(pd.DataFrame(align_rows), on="event_id", how="left")
    events = events.merge(pd.DataFrame(obs_rows), on="event_id", how="left")

    outcomes = parts["outcomes"]
    funnel = compute_funnel(events)
    return {"events": events, "features": feats, "outcomes": outcomes, "funnel": funnel}


# ---------------------------------------------------------------------------- single event (prospective)
# Price-stage exclusions that depend on the target-session bar: not knowable before the cutoff.
TARGET_BAR_REASONS = {"target_session_bar_missing", "target_bar_missing", "target_bar_invalid"}


@dataclass
class SingleEventFeatures:
    accession_number: str
    status: str                          # ok | excluded
    stage: str | None = None             # funnel stage that excluded the event
    reason: str | None = None
    event: dict | None = None            # timing fields of the candidate row
    features: dict | None = None         # all catalogued model inputs
    threshold: Threshold | None = None
    reference_accession: str | None = None
    provenance: dict = field(default_factory=dict)


def single_event_features(cfg: Config, layout: DataLayout, accession: str, client,
                          filings: pd.DataFrame | None = None,
                          facts_max_age_days: float | None = 0.0) -> SingleEventFeatures:
    """Feature snapshot for one filing at its decision cutoff, for a prospective forecast.

    Uses the historical build's own functions: build_candidates (form, timing, uniqueness, and the
    outcomes of the company's earlier events), pre_cutoff_market (bars strictly before the target
    session), extract_filing + select_reference + document_stage_exclusion, and event_document_features.
    The target-session bar is never read and no outcome is computed for this event.

    client: a SecClient (used for the current/reference documents and a Company Facts refresh;
    offline clients use cached files only).
    """
    from filingshock.extraction import extract_filing
    from filingshock.sources.sec import SecAccessError

    filings = load_filings(layout) if filings is None else filings
    cur = filings[filings["accession_number"] == accession]
    if cur.empty:
        return SingleEventFeatures(accession, "excluded", "discovered", "accession_not_in_filings_table")
    cur = cur.iloc[0]
    lo = pd.Timestamp(cfg["universe"]["filing_window"]["accepted_from"], tz="UTC")
    company = filings[(filings["cik"] == cur["cik"]) & (filings["acceptance_utc"] >= lo)
                      & (filings["acceptance_utc"] <= cur["acceptance_utc"])].assign(in_window=True)
    disc, parts = build_candidates(cfg, layout, filings=company)
    r = disc[disc["accession_number"] == accession].iloc[0]
    out = SingleEventFeatures(accession, "excluded", event={k: r.get(k) for k in (
        "event_id", "ticker", "cik", "company_name", "form_type", "report_date", "filing_date", "acceptance_utc",
        "assumed_available_utc", "availability_method", "target_session", "target_open_utc", "target_close_utc",
        "decision_cutoff_utc", "prev_session", "prev_close_utc", "timing_flags")})
    stage, reason = r["exclusion_stage"], r["exclusion_reason"]
    if pd.notna(stage) and not (stage == "valid_price_history" and reason in TARGET_BAR_REASONS):
        out.stage, out.reason = stage, reason
        return out

    # ---- market: bars strictly before the target session only
    spec = TargetSpec.from_config(cfg)
    max_abs = float(cfg["target"]["max_abs_return_sanity"])
    store = PriceStore(layout, cfg["market_data"]["provider"])
    bars = _bars(store, r["ticker"], max_abs)
    if bars is None:
        out.stage, out.reason = "valid_price_history", "price_history_unavailable_from_provider"
        return out
    history, th, mf = pre_cutoff_market(bars, bench=_benchmarks(cfg, store, max_abs),
                                        target_session=r["target_session"], spec=spec)
    last = history["session"].max() if len(history) else None
    out.provenance["price_history_last_session"] = str(last.date()) if last is not None else None
    if last is None or last < pd.Timestamp(r["prev_session"]):
        out.stage, out.reason = "valid_price_history", "price_history_stale"   # refresh prices, then retry
        return out
    if th.status != "ok":
        out.stage, out.reason = "valid_price_history", th.status
        return out
    out.threshold = th
    earlier = parts["outcomes"]
    if not earlier.empty:
        earlier = earlier.merge(disc[["event_id", "decision_cutoff_utc"]], on="event_id")
        mf["mkt_prev_event_abs_ratio"] = prev_event_abs_ratio(earlier, r["decision_cutoff_utc"])
    else:
        mf["mkt_prev_event_abs_ratio"] = None

    # ---- documents: current filing and its primary reference
    q10 = filings[filings["form_type"] == "10-Q"]
    choice = select_reference(q10, r, cfg["reference"]["primary_policy"], int(cfg["reference"]["period_tolerance_days"]))
    cur_sum = extract_filing(client, layout, cur)
    ref_sum = {}
    if choice.status == "found":
        ref_sum = extract_filing(client, layout, q10[q10["accession_number"] == choice.reference_accession].iloc[0])
    excl = document_stage_exclusion(cur_sum.get("mdna_status", "not_attempted"), cur_sum.get("download_status") == "failed",
                                    asdict(choice), ref_sum.get("mdna_status", "not_attempted"))
    if excl is not None:
        out.stage, out.reason = excl
        return out
    out.reference_accession = choice.reference_accession
    try:   # Company Facts may already include the current filing's XBRL facts; cached copy otherwise
        client.company_facts(int(r["cik"]), max_age_days=facts_max_age_days)
    except SecAccessError as exc:
        log.warning("Company Facts refresh failed for CIK %s; using the cached copy: %s", r["cik"], exc)
    fc = DocumentFeatureContext.load(cfg, layout, filings)
    d = event_document_features(fc, r, choice.reference_accession)
    facts = fc.company_facts(int(r["cik"]))
    out.features = {**mf, **d.context, **d.financial, **d.text}
    out.provenance.update({
        "context": d.context_provenance, "financial": d.financial_provenance, "alignment": d.alignment,
        "current_filing_facts_in_company_facts": bool(len(facts) and (facts["accn"] == accession).any()),
        "reference_period_gap_days": choice.period_gap_days, "current_source_sha256": cur_sum.get("source_sha256"),
    })
    out.status = "ok"
    return out


def _text_for_event(layout, event_id, cur_doc, ref_doc, lexicon, topics, year):
    from filingshock.features.text import TEXT_FEATURES as TF
    cur_m, ref_m = cur_doc["mdna"], ref_doc["mdna"]
    res = align_paragraphs([p["text"] for p in ref_m], [p["text"] for p in cur_m])
    pairs = res.pairs.copy()
    pairs["is_heading"] = [
        (cur_m[int(c)]["is_heading"] if pd.notna(c) else ref_m[int(rr)]["is_heading"])
        for c, rr in zip(pairs["cur_ordinal"], pairs["ref_ordinal"])]
    pairs["section"] = "mdna"
    tf, obs = section_text_features(cur_m, ref_m, pairs, res.summary, lexicon, topics, year)
    cur_rs, ref_rs = cur_doc["summary"].get("risk_status"), ref_doc["summary"].get("risk_status")
    risk_pairs = None
    if cur_rs == "extracted" and ref_rs == "extracted":
        rres = align_paragraphs([p["text"] for p in ref_doc["risk"]], [p["text"] for p in cur_doc["risk"]])
        risk_pairs = rres.pairs.copy()
        risk_pairs["is_heading"] = [
            (cur_doc["risk"][int(c)]["is_heading"] if pd.notna(c) else ref_doc["risk"][int(rr)]["is_heading"])
            for c, rr in zip(risk_pairs["cur_ordinal"], risk_pairs["ref_ordinal"])]
        risk_pairs["section"] = "risk"
    n_cur_risk = sum(1 for p in cur_doc["risk"] if not p["is_heading"])
    tf["txt_risk_prop_added"] = risk_prop_added(cur_rs, ref_rs, risk_pairs, n_cur_risk)
    all_pairs = pd.concat([pairs] + ([risk_pairs] if risk_pairs is not None else []), ignore_index=True)
    write_parquet_atomic(layout.alignments_dir / f"{event_id}.parquet", all_pairs)
    summ = {f"mdna_align_{k}": v for k, v in res.summary.items()}
    summ["risk_status_current"], summ["risk_status_reference"] = cur_rs, ref_rs
    assert set(tf) == set(TF), set(tf) ^ set(TF)
    return tf, obs, summ


def compute_funnel(events: pd.DataFrame) -> dict:
    stages = []
    remaining = len(events)
    for stage, desc in FUNNEL_STAGES:
        lost = events[events["exclusion_stage"] == stage]
        reasons = lost["exclusion_reason"].value_counts().to_dict()
        if stage == "discovered":
            stages.append({"stage": stage, "description": desc, "remaining": remaining, "excluded": 0, "reasons": {}})
            continue
        remaining -= len(lost)
        stages.append({"stage": stage, "description": desc, "remaining": remaining, "excluded": int(len(lost)),
                       "reasons": {k: int(v) for k, v in reasons.items()}})
    final = events[events["eligibility_status"] == "eligible"]
    stages.append({"stage": FINAL_STAGE, "description": "Final experiment sample (common to all models)",
                   "remaining": int(len(final)), "excluded": 0, "reasons": {}})
    by_company = events.groupby("ticker").apply(
        lambda g: {"discovered": int(len(g)), "final": int((g["eligibility_status"] == "eligible").sum())},
        include_groups=False).to_dict()
    return {"stages": stages, "by_company": by_company, "n_companies_final": int(final["cik"].nunique())}


def write_event_tables(layout: DataLayout, cfg: Config, tables: dict, data_manifest: dict) -> None:
    proc = layout.processed
    write_parquet_atomic(proc / "events.parquet", tables["events"])
    write_parquet_atomic(proc / "features.parquet", tables["features"])
    write_parquet_atomic(proc / "outcomes.parquet", tables["outcomes"])
    write_json_atomic(proc / "funnel.json", tables["funnel"])
    write_json_atomic(proc / "dataset_manifest.json", data_manifest)


def feature_groups() -> dict[str, list[str]]:
    from filingshock.features.text import TEXT_GROUPS
    return {"market": MARKET_FEATURES, "context": CONTEXT_FEATURES, "financial": FIN_FEATURES,
            "text": TEXT_FEATURES, **{f"text_{k}": v for k, v in TEXT_GROUPS.items()}}


def dataset_manifest(cfg: Config, layout: DataLayout, tables: dict) -> dict:
    import hashlib
    ev = tables["events"]
    final = ev[ev["eligibility_status"] == "eligible"]
    h = hashlib.sha256()
    for col in ["event_id", "source_document_hash"]:
        h.update("|".join(final[col].astype(str)).encode())
    feat = tables["features"].drop(columns=["computed_at_utc"], errors="ignore")
    h.update(pd.util.hash_pandas_object(feat.drop(columns=[c for c in feat.columns if c.endswith("provenance")]), index=False).values.tobytes())
    h.update(pd.util.hash_pandas_object(tables["outcomes"], index=False).values.tobytes())
    price_manifest = PriceStore(layout, cfg["market_data"]["provider"]).manifest_entries()
    return {
        "dataset_id": "ds-" + h.hexdigest()[:12],
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": __version__, "parser_version": PARSER_VERSION, "aligner_version": ALIGNER_VERSION,
        "feature_version": cfg["features"]["version"],
        "config_hash": cfg.section_hash("universe", "timing", "target", "reference", "features", "market_data"),
        "n_discovered": int(len(ev)), "n_final_events": int(len(final)),
        "n_companies_final": int(final["cik"].nunique()),
        "first_target_session": str(final["target_session"].min().date()) if len(final) else None,
        "last_target_session": str(final["target_session"].max().date()) if len(final) else None,
        "price_provider": cfg["market_data"]["provider"],
        "price_files": {k: {kk: v.get(kk) for kk in ("sha256", "retrieved_at", "rows", "last", "status")}
                        for k, v in price_manifest.items()},
        "lm_dictionary_file": LM_FILE,
    }


def recompute_outcomes(cfg: Config, layout: DataLayout, lookback: int, multiplier: float) -> pd.DataFrame:
    """Alternative target definitions for sensitivity analyses (same events, same timing)."""
    spec = TargetSpec(lookback, multiplier, int(cfg["target"]["max_window_sessions"]),
                      float(cfg["target"]["min_denominator"]))
    max_abs = float(cfg["target"]["max_abs_return_sanity"])
    store = PriceStore(layout, cfg["market_data"]["provider"])
    ev = pd.read_parquet(layout.processed / "events.parquet")
    ev = ev[ev["eligibility_status"] == "eligible"]
    rows, cache = [], {}
    for _, r in ev.iterrows():
        if r["ticker"] not in cache:
            cache[r["ticker"]] = _bars(store, r["ticker"], max_abs)
        bars = cache[r["ticker"]]
        p = int(bars.index[bars["session"] == r["target_session"]][0])
        th = historical_threshold(bars.iloc[:p], spec)
        oc = realized_outcome(bars.iloc[p], th)
        if th.status == "ok" and oc.status == "ok":
            rows.append({"event_id": r["event_id"], "y": oc.y, "threshold_abs_move": th.threshold})
    return pd.DataFrame(rows)
