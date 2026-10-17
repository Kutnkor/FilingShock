"""Reusable interface patterns. Every value inserted into HTML is escaped first."""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from filingshock.ui.data import PREDICTION_STATUS_LABEL

CSS = (Path(__file__).parent / "theme.css").read_text()
DASH = "—"


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _missing(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def pct(x, digits: int = 1, signed: bool = False) -> str:
    if _missing(x):
        return DASH
    s = f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"
    return s


def pp(x, digits: int = 1) -> str:
    """Percentage-point difference of two fractions."""
    return DASH if _missing(x) else f"{x * 100:+.{digits}f} pp"


def num(x, digits: int = 3, signed: bool = False) -> str:
    if _missing(x):
        return DASH
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def money(x) -> str:
    return DASH if _missing(x) else f"${x:,.2f}"


def ts_et(x, with_utc: bool = False) -> str:
    if _missing(x):
        return DASH
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    s = t.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET")
    if with_utc:
        s += f" ({t.tz_convert('UTC').strftime('%H:%M')} UTC)"
    return s


def day(x) -> str:
    return DASH if _missing(x) else pd.Timestamp(x).strftime("%Y-%m-%d")


def page_header(title: str, context: str | None = None) -> None:
    st.markdown(f'<div class="fs-head"><h1>{esc(title)}</h1>' + (f"<p>{esc(context)}</p>" if context else "") + "</div>",
                unsafe_allow_html=True)


def chip(label: str, kind: str = "info", title: str | None = None) -> str:
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="fs-chip {esc(kind)}"{t}>{esc(label)}</span>'


def status(kind: str, title: str, body: str = "", html_body: bool = False) -> None:
    """kind: info | warn | error | unavail. Rendered with role=status for assistive technology."""
    b = body if html_body else esc(body)
    st.markdown(f'<div class="fs-msg {esc(kind)}" role="status"><strong>{esc(title)}</strong>{b}</div>',
                unsafe_allow_html=True)


def facts(items: list[tuple[str, str]], raw_values: bool = False) -> None:
    rows = "".join(f"<dt>{esc(k)}</dt><dd>{v if raw_values else esc(v)}</dd>" for k, v in items)
    st.markdown(f'<dl class="fs-facts">{rows}</dl>', unsafe_allow_html=True)


def source_link(url: str | None, label: str) -> str:
    if not url:
        return esc(label) + " (source link unavailable)"
    return f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(label)} ↗</a>'


def caption(text: str) -> None:
    st.markdown(f'<div class="fs-caption">{esc(text)}</div>', unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="fs-section-title" role="heading" aria-level="2">{esc(text)}</div>', unsafe_allow_html=True)


def status_chip_for_event(r) -> str:
    if r.get("eligibility_status") == "eligible":
        return chip("Eligible", "accent")
    if r.get("eligibility_status") == "prospective":
        return chip("Prospective forecast", "accent")
    return chip("Excluded: " + str(r.get("exclusion_reason")), "unavail")


def prediction_chip(status_key: str) -> str:
    kind = {"retrospective_out_of_time": "info", "prospective_recorded_before_session": "accent",
            "synthetic_ui_example": "warn"}.get(status_key, "unavail")
    return chip(PREDICTION_STATUS_LABEL.get(status_key, status_key), kind)


def identity_strip(r) -> None:
    fp = r.get("fiscal_period_label")
    parts = [f'<span class="fs-strip-title">{esc(r["company_name"])} ({esc(r["ticker"])})</span>',
             f'<span><span class="k">Form</span><span class="v">{esc(r["form_type"])}</span></span>',
             f'<span><span class="k">Fiscal period</span><span class="v">{esc(fp if isinstance(fp, str) else DASH)} '
             f'(ends {esc(r.get("report_date") or DASH)})</span></span>',
             f'<span><span class="k">Accepted</span><span class="v">{esc(ts_et(r["acceptance_utc"]))}</span></span>',
             f'<span><span class="k">Target session</span><span class="v">{esc(day(r.get("target_session")))}</span></span>',
             status_chip_for_event(r)]
    st.markdown(f'<div class="fs-strip">{"".join(parts)}</div>', unsafe_allow_html=True)


def event_label(r) -> str:
    fp = r.get("fiscal_period_label")
    fp = fp if isinstance(fp, str) else (r.get("report_date") or "")
    return f"{r['ticker']} · {r['form_type']} {fp} · accepted {ts_et(r['acceptance_utc'])} · {r['event_id']}"


def html_table(df: pd.DataFrame, right: set[str] | None = None, highlight_first_col_value: str | None = None,
               caption_text: str | None = None) -> None:
    right = right or set()
    head = "".join(f'<th class="{"r" if c in right else ""}" scope="col">{esc(c)}</th>' for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cls = ' class="primary"' if highlight_first_col_value is not None and str(row.iloc[0]) == highlight_first_col_value else ""
        cells = "".join(f'<td class="{"r" if c in right else ""}">{esc(row[c])}</td>' for c in df.columns)
        body.append(f"<tr{cls}>{cells}</tr>")
    cap = f"<caption class='fs-caption' style='caption-side:bottom;text-align:left'>{esc(caption_text)}</caption>" if caption_text else ""
    st.markdown(f'<div class="fs-scroll"><table class="fs-table">{cap}<thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)


def persisted(key: str, default):
    """Value of a filter that survives page switches (widget state is cleared when a page is not shown)."""
    store = st.session_state.setdefault("_persist", {})
    return store.get(key, default)


def persist(key: str, value) -> None:
    st.session_state.setdefault("_persist", {})[key] = value


def selected_event_id(events: pd.DataFrame) -> str | None:
    qp = st.query_params.get("event")
    if qp and qp in set(events["event_id"]):
        return qp
    sid = persisted("event", None)
    return sid if sid in set(events["event_id"]) else None


def set_selected_event(event_id: str | None) -> None:
    persist("event", event_id)
    if event_id:
        st.query_params["event"] = event_id
    elif "event" in st.query_params:
        del st.query_params["event"]


def event_picker(events: pd.DataFrame, key: str, label: str = "Event", only_eligible: bool = False) -> pd.Series | None:
    pool = events[events["eligibility_status"] == "eligible"] if only_eligible else events
    pool = pool.sort_values(["ticker", "acceptance_utc"])
    if pool.empty:
        return None
    ids = list(pool["event_id"])
    current = selected_event_id(events)
    idx = ids.index(current) if current in ids else 0
    labels = {r["event_id"]: event_label(r) for _, r in pool.iterrows()}
    choice = st.selectbox(label, ids, index=idx, format_func=lambda e: labels[e], key=key,
                          help="Type to search by ticker, fiscal period or event id.")
    if choice != current:
        set_selected_event(choice)
    return pool[pool["event_id"] == choice].iloc[0]


def finite(x) -> bool:
    return not _missing(x) and math.isfinite(float(x))
