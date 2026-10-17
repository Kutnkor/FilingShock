"""Filing Comparison: reference filing (left) vs current filing (right), paragraph-aligned."""

import re

import pandas as pd
import streamlit as st
import yaml

from filingshock.config import PROJECT_ROOT
from filingshock.parsing.align import word_diff
from filingshock.ui import components as ui
from filingshock.ui import data as D
from filingshock.ui.shared import data_banner, require_dataset

STATUS_META = {
    "unchanged": ("Unchanged", "same", "Identical text after whitespace normalisation."),
    "numeric_only": ("Numbers changed", "edited", "Same wording; only numbers differ."),
    "moved": ("Moved", "moved", "Same text (numbers may differ) at a different relative position."),
    "minor_edit": ("Minor edit", "edited", "Matched paragraph with word-sequence similarity ≥ 0.90."),
    "substantial_edit": ("Substantial edit", "edited", "Matched paragraph with word-sequence similarity < 0.90 (Jaccard ≥ 0.40)."),
    "added": ("Added", "added", "Current paragraph with no counterpart in the reference filing."),
    "removed": ("Removed", "removed", "Reference paragraph with no counterpart in the current filing."),
}
FILTERS = {"All paragraphs": None, "Changed only": ["numeric_only", "moved", "minor_edit", "substantial_edit", "added", "removed"],
           "Added": ["added"], "Removed": ["removed"], "Edited": ["minor_edit", "substantial_edit", "numeric_only"],
           "Moved": ["moved"]}
PAGE_SIZE = 25

data = D.load()
ui.page_header("Filing Comparison", "What changed in management's discussion between the prior-year same-quarter "
               "10-Q (left) and the current 10-Q (right). Labels describe editing status, not good or bad news.")
data_banner(data)
if not require_dataset(data):
    st.stop()

ev = data.events
r = ui.event_picker(ev, key="cmp_event", only_eligible=True)
if r is None:
    ui.status("unavail", "No final-sample events are available.", "Build the dataset first.")
    st.stop()
ui.identity_strip(r)

cur_doc = D.section_doc(r["accession_number"])
ref_acc = r.get("reference_filing_id")
ref_doc = D.section_doc(ref_acc) if isinstance(ref_acc, str) else None
pairs_all = D.alignment(r["event_id"])
ext = D.extractions()
ext = ext.set_index("accession_number") if ext is not None else None


def src_url(acc):
    if ext is not None and acc in ext.index:
        return ext.loc[acc, "source_url"]
    return None


if cur_doc is None or ref_doc is None or pairs_all is None:
    ui.status("unavail", "Comparison artifacts are missing for this event.",
              "The extracted sections or the paragraph alignment were not found in the local data. "
              "Re-run `filingshock build` to regenerate them.")
    st.stop()

ref_row = None
if isinstance(ref_acc, str):
    m = ev[ev["accession_number"] == ref_acc]
    ref_row = m.iloc[0] if len(m) else None
c1, c2 = st.columns(2)
with c1:
    ui.facts([
        ("Reference (left)", ui.esc(f"{ref_acc} · period end {r.get('reference_period_end') or '—'}")),
        ("Reference policy", ui.esc("prior-year same fiscal quarter, period end "
                                    f"{int(r['reference_gap_days']) if pd.notna(r.get('reference_gap_days')) else '—'} days earlier")),
        ("Source", ui.source_link(src_url(ref_acc), "Reference filing on SEC.gov")),
    ], raw_values=True)
with c2:
    ui.facts([
        ("Current (right)", ui.esc(f"{r['accession_number']} · period end {r.get('report_date') or '—'}")),
        ("Alignment", ui.esc(f"{cur_doc['summary'].get('parser_version')} extraction · fs-align-1.0 deterministic matching · "
                             f"quality: {r.get('mdna_align_alignment_quality') or '—'} (heuristic)")),
        ("Source", ui.source_link(src_url(r["accession_number"]), "Current filing on SEC.gov")),
    ], raw_values=True)

# ---------------------------------------------------------------- controls
k1, k2, k3 = st.columns([1.3, 2.2, 1.6])
section = k1.radio("Section", ["MD&A", "Risk factors"], horizontal=True, key="cmp_section")
sec_key = "mdna" if section == "MD&A" else "risk"
show = k2.radio("Show", list(FILTERS), horizontal=True, key="cmp_show", index=1)
layout_mode = k3.radio("Layout", ["Side by side", "Stacked pairs"], horizontal=True, key="cmp_layout",
                       help="Stacked pairs keeps paragraphs readable on narrow screens.")
k4, k5, k6 = st.columns([2.2, 2, 1.4])
query = k4.text_input("Search text", key="cmp_query", placeholder="e.g. backlog, macroeconomic, receivables")
topics_doc = yaml.safe_load((PROJECT_ROOT / "resources/dictionaries/topics_v1.yaml").read_text())
topic_rx = {t: re.compile(r"\b(?:" + "|".join(p) + r")\b", re.I) for t, p in topics_doc["topics"].items()}
sel_topics = k5.multiselect("Topic (topics-1.0 lexicon)", list(topic_rx), key="cmp_topics", placeholder="Any topic",
                            format_func=lambda t: t.replace("_", " "))
word_level = k6.toggle("Word-level changes", key="cmp_words", help="Show insertions and deletions inside edited paragraphs.")

cur_paras, ref_paras = cur_doc[sec_key], ref_doc[sec_key]
pairs = pairs_all[pairs_all["section"] == sec_key] if "section" in pairs_all else pairs_all
if sec_key == "risk":
    cs, rs = cur_doc["summary"].get("risk_status"), ref_doc["summary"].get("risk_status")
    explain = {"extracted": "full risk-factor text present", "no_material_changes": "states no material changes since the 10-K",
               "short_reference": "only refers to the risk factors in another report", "absent": "no Part II Item 1A found",
               "failed": "heading found but extraction failed"}
    if pairs.empty:
        ui.status("unavail", "No paragraph comparison for risk factors in this pair.",
                  f"Current filing: {explain.get(cs, cs)}. Reference filing: {explain.get(rs, rs)}. A comparison is made "
                  "only when both filings contain full risk-factor text. A missing or short risk section is not evidence "
                  "of lower risk; the 10-K risk factors still apply.")
        if cur_paras:
            with st.expander("Current filing risk section text"):
                for p in cur_paras:
                    st.markdown(f'<div class="fs-cmp-cell">{ui.esc(p["text"])}</div>', unsafe_allow_html=True)
        st.stop()

# ---------------------------------------------------------------- summary counts (clickable via the Show control)
body = pairs[~pairs["is_heading"]] if "is_heading" in pairs else pairs
cnt = body["status"].value_counts()
st.markdown('<div class="fs-counts">' + "".join(
    f'<span>{ui.chip(STATUS_META[s][0], STATUS_META[s][1], STATUS_META[s][2])} <b>{int(cnt.get(s, 0))}</b></span>'
    for s in STATUS_META) + f'<span class="fs-muted">body paragraphs · current {sum(1 for p in cur_paras if not p["is_heading"])}'
    f' · reference {sum(1 for p in ref_paras if not p["is_heading"])} · numbers changed in {int(body["numeric_change"].sum())} matched pairs'
    f' · low-confidence matches {int(body["low_confidence"].sum())}</span></div>', unsafe_allow_html=True)
with st.expander("What do these labels mean?"):
    for s, (lab, _, d) in STATUS_META.items():
        st.markdown(f"**{lab}** — {d}")
    st.markdown("Match scores: *Jaccard* is the overlap of word and word-pair sets after masking numbers; *edit ratio* "
                "is the word-sequence similarity of a matched pair. Low-confidence matches have Jaccard < 0.55. "
                "'Topic' filters use the project-authored topics-1.0 lexicon and mean the paragraph mentions the topic, "
                "not that the topic is new or risky.")


# ---------------------------------------------------------------- filter rows
def text_of(o, paras):
    return paras[int(o)]["text"] if pd.notna(o) else ""


rows = pairs.copy()
rows["cur_text"] = rows["cur_ordinal"].map(lambda o: text_of(o, cur_paras))
rows["ref_text"] = rows["ref_ordinal"].map(lambda o: text_of(o, ref_paras))
if FILTERS[show] is not None:   # filtered views show body paragraphs only; "All" keeps headings for orientation
    rows = rows[rows["status"].isin(FILTERS[show]) & ~rows["is_heading"]]
if query.strip():
    q = query.strip().lower()
    rows = rows[rows["cur_text"].str.lower().str.contains(q, regex=False) | rows["ref_text"].str.lower().str.contains(q, regex=False)]
if sel_topics:
    rows = rows[rows.apply(lambda x: any(topic_rx[t].search(x["cur_text"] + " " + x["ref_text"]) for t in sel_topics), axis=1)]

n = len(rows)
if n == 0:
    ui.status("info", "No paragraphs match.", "Change the 'Show' filter, clear the search box or remove topic filters.")
    st.stop()
pages = (n - 1) // PAGE_SIZE + 1
pc1, pc2 = st.columns([1, 4])
page = pc1.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=f"cmp_page_{r['event_id']}_{sec_key}_{show}")
pc2.markdown(f"<div class='fs-caption' style='margin-top:2rem'>{n} paragraph rows · page {page} of {pages} · "
             f"{PAGE_SIZE} per page</div>", unsafe_allow_html=True)
chunk = rows.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]


def highlight(text: str) -> str:
    out = ui.esc(text)
    if query.strip():
        out = re.sub("(" + re.escape(ui.esc(query.strip())) + ")", r'<mark class="fs-hit">\1</mark>', out, flags=re.I)
    return out


def render_diff(ref_text: str, cur_text: str) -> tuple[str, str]:
    left, right = [], []
    for op, t in word_diff(ref_text, cur_text):
        e = ui.esc(t)
        if op == "equal":
            left.append(e)
            right.append(e)
        elif op == "delete":
            left.append(f'<del class="fs-del">{e}</del>')
        else:
            right.append(f'<ins class="fs-ins">{e}</ins>')
    return " ".join(left), " ".join(right)


def cell(text_html: str, kind: str, side_label: str, empty_msg: str, heading: bool) -> str:
    if not text_html:
        return f'<div class="fs-cmp-cell empty">{ui.esc(empty_msg)}</div>'
    cls = f"fs-cmp-cell {kind}" + (" heading" if heading else "")
    return f'<div class="{cls}" aria-label="{ui.esc(side_label)}">{text_html}</div>'


html_rows = []
if layout_mode == "Side by side":
    html_rows.append(f'<div class="fs-cmp-head"><div><b>Reference</b> · {ui.esc(ref_acc)} · period end '
                     f'{ui.esc(r.get("reference_period_end") or "—")}</div><div><b>Current</b> · {ui.esc(r["accession_number"])} · '
                     f'period end {ui.esc(r.get("report_date") or "—")}</div></div>')
for _, x in chunk.iterrows():
    lab, kind, desc = STATUS_META[x["status"]]
    edited = x["status"] in ("minor_edit", "substantial_edit", "numeric_only")
    if word_level and edited:
        lt, rt = render_diff(x["ref_text"], x["cur_text"])
    else:
        lt, rt = highlight(x["ref_text"]), highlight(x["cur_text"])
    rid = f"R-{int(x['ref_ordinal']):03d}" if pd.notna(x["ref_ordinal"]) else "—"
    cid = f"C-{int(x['cur_ordinal']):03d}" if pd.notna(x["cur_ordinal"]) else "—"
    meta = [ui.chip(lab, kind, desc), f"<code>{rid} → {cid}</code>"]
    if pd.notna(x.get("jaccard")) and x["status"] not in ("added", "removed"):
        meta.append(f"Jaccard {x['jaccard']:.2f} · edit ratio {x['edit_ratio']:.2f}")
    if x.get("numeric_change"):
        meta.append(ui.chip("Numbers differ", "info"))
    if x.get("low_confidence"):
        meta.append(ui.chip("Low-confidence match", "warn", "Jaccard below 0.55; the pairing may be wrong."))
    if x.get("moved") and x["status"] != "moved":
        meta.append(ui.chip("Also moved", "moved"))
    meta_html = f'<div class="fs-cmp-meta">{" ".join(meta)}</div>'
    left = cell(lt, "removed" if x["status"] == "removed" else ("edited" if edited else ""), "reference paragraph",
                "No counterpart in the reference filing", bool(x.get("is_heading")))
    right = cell(rt, "added" if x["status"] == "added" else ("edited" if edited else ""), "current paragraph",
                 "No counterpart in the current filing", bool(x.get("is_heading")))
    if layout_mode == "Side by side":
        html_rows.append(f'<div class="fs-cmp-row">{meta_html}{left}{right}</div>')
    else:
        html_rows.append(f'<div class="fs-stack">{meta_html}<div class="fs-side-label">Reference</div>{left}'
                         f'<div class="fs-side-label">Current</div>{right}</div>')
st.markdown("".join(html_rows), unsafe_allow_html=True)
ui.caption("Paragraph identifiers R-### / C-### are ordinal positions within the extracted section of each filing "
           "(stable for parser fs-parser-1.1). Text is rendered as escaped plain text; source HTML is never executed. "
           "Financial tables are excluded from the paragraph comparison.")
