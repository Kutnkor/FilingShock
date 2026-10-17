"""Page chrome shared by all screens: data banner, dataset guard, event definition."""

from __future__ import annotations

import streamlit as st

from filingshock.ui import components as ui
from filingshock.ui.data import AppData


def data_banner(data: AppData) -> None:
    if not data.has_dataset:
        return
    ds = data.dataset or {}
    prices = ds.get("price_files", {})
    last_price = max((v.get("last") or "" for v in prices.values()), default="") or "not included"
    bits = [
        ("Data", "offline sample package" if data.mode == "sample" else "local research cache (historical, not live)"),
        ("Dataset", f"{ds.get('dataset_id', '—')} · built {str(ds.get('created_at_utc', '—'))[:16].replace('T', ' ')} UTC"),
        ("Coverage", f"{ds.get('n_final_events', '—')} final events · {ds.get('n_companies_final', '—')} companies · "
                     f"target sessions {ds.get('first_target_session', '—')} to {ds.get('last_target_session', '—')}"),
        ("Latest price session in cache", last_price),
        ("Primary experiment", data.run_id or "none saved"),
    ]
    st.markdown('<div class="fs-strip">' + "".join(
        f'<span><span class="k">{ui.esc(k)}</span><span class="v">{ui.esc(v)}</span></span>' for k, v in bits) + "</div>",
        unsafe_allow_html=True)
    if data.mode == "sample":
        ui.status("info", "Offline sample package",
                  "Contains real SEC filings, extracted sections, alignments, financial and text features, saved "
                  "aggregate results and model probabilities for a few events. Provider price data and price-derived "
                  "values (opening/closing prices, returns, thresholds, market features) are not included because the "
                  "provider's terms do not allow redistribution. Run the research pipeline to add them locally.")
    if data.run_is_stale:
        ui.status("warn", "Saved results belong to a different dataset version.",
                  f"The primary run was computed on {data.run_manifest['dataset']['dataset_id']}, but the current dataset is "
                  f"{data.dataset['dataset_id']}. Results and predictions may not match the events shown. "
                  "Re-run `filingshock experiment --name primary --set-primary`.")


def require_dataset(data: AppData) -> bool:
    if data.has_dataset:
        return True
    ui.status("unavail", "No local dataset has been prepared.",
              "FilingShock reads saved artifacts only; it never downloads or trains when a page loads. "
              "Prepare data with `filingshock reproduce` (needs network access and a configured SEC User-Agent), "
              "or open the bundled offline example with `filingshock app --sample` if `sample/` exists.")
    return False


def definition_strip(data: AppData) -> None:
    cfg = (data.run_manifest or {}).get("config") or {}
    tgt = cfg.get("target", {"lookback_sessions": 60, "multiplier": 2.0})
    tim = cfg.get("timing", {"cutoff_minutes_before_open": 30})
    ui.status("info", "What is predicted",
              f"The probability that the next eligible session's open-to-close move, |close/open − 1|, exceeds "
              f"{tgt['multiplier']:g} × the stock's mean absolute open-to-close move over its previous "
              f"{tgt['lookback_sessions']} sessions. Magnitude only, not direction. Information cutoff: "
              f"{tim['cutoff_minutes_before_open']} minutes before the regular open. Overnight gaps and the intraday "
              "high–low range are not part of the target.")
