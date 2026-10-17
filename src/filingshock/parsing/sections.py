"""MD&A (Part I, Item 2) and risk-factor (Part II, Item 1A) extraction from 10-Q filings.

Extraction is deterministic. Table-of-contents entries and cross-references are rejected by
choosing, among all heading candidates, the one that opens the longest span before the next
section heading, and by requiring headings to be short standalone blocks.

Status semantics (never collapsed):
  extracted            section found and extracted
  no_material_changes  risk section present but only states there are no material changes
  short_reference      risk section present but only refers to another report
  absent               no heading for the section was found
  failed               a heading was found but no usable span could be extracted
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from filingshock import PARSER_VERSION
from filingshock.parsing.html import Block, is_page_artifact, narrative_like_table

ITEM_RE = re.compile(r"^(?:part\s+i{1,2}\s*[,.\-:]?\s*)?item\s*(\d{1,2}[a-z]?)\s*[.:\-]?\s*(.*)$", re.I)
PART2_RE = re.compile(r"^part\s+ii\b", re.I)
MDNA_TITLE = re.compile(r"^management'?s?\s+discussion\s+and\s+analysis", re.I)
MKT_RISK_TITLE = re.compile(r"^quantitative\s+and\s+qualitative\s+disclosures?", re.I)
CONTROLS_TITLE = re.compile(r"^controls\s+and\s+procedures", re.I)
RISK_TITLE = re.compile(r"^risk\s+factors\b", re.I)
SIGNATURES = re.compile(r"^signatures?$", re.I)
NO_MATERIAL = re.compile(
    r"(no|not\s+been\s+any|have\s+not\s+been|were\s+no)\s+material\s+changes?|"
    r"no\s+material\s+(?:change|modification)|has\s+not\s+materially\s+changed|"
    r"not\s+materially\s+changed", re.I)
RUNNING_HEADER = re.compile(
    r"^(?:part\s+i{1,2}|item\s*\d{1,2}[a-z]?)\.?(?:\s*[,.\-]?\s*(?:part\s+i{1,2}|item\s*\d{1,2}[a-z]?)\.?)?$", re.I)
HEADING_MAX_CHARS = 250
MIN_MDNA_WORDS = 300
MIN_RISK_FULL_WORDS = 400


@dataclass
class Paragraph:
    ordinal: int
    block_idx: int
    text: str
    is_heading: bool
    char_start: int
    char_end: int


@dataclass
class SectionExtraction:
    section: str                      # mdna | risk
    status: str
    method: str | None = None
    failure_reason: str | None = None
    start_block: int | None = None
    end_block: int | None = None
    n_paragraphs: int = 0
    n_words: int = 0
    n_chars: int = 0
    n_tables: int = 0
    heuristic_quality: float = 0.0    # heuristic score in [0,1]; NOT a calibrated probability
    parser_version: str = PARSER_VERSION
    n_start_candidates: int = 0
    paragraphs: list[Paragraph] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("paragraphs")
        return d


@dataclass
class Heading:
    block_idx: int
    item: str | None      # e.g. "2", "1a"; None for title-only headings
    title: str
    consumed: int = 1     # how many blocks form the heading (an "ITEM 2" line plus a title line)


def _heading_text(b: Block) -> str | None:
    if b.kind == "table" and not (narrative_like_table(b) or len(b.text) < HEADING_MAX_CHARS):
        return None
    if len(b.text) > HEADING_MAX_CHARS:
        return None
    return b.text


def find_headings(blocks: list[Block]) -> list[Heading]:
    out: list[Heading] = []
    i = 0
    while i < len(blocks):
        t = _heading_text(blocks[i])
        if t is None:
            i += 1
            continue
        m = ITEM_RE.match(t)
        if m:
            item, rest = m.group(1).lower(), m.group(2).strip()
            consumed = 1
            if not rest and i + 1 < len(blocks):
                nxt = _heading_text(blocks[i + 1])
                if nxt and not ITEM_RE.match(nxt):
                    rest, consumed = nxt, 2
            out.append(Heading(i, item, rest, consumed))
            i += consumed
            continue
        if PART2_RE.match(t):
            out.append(Heading(i, "part2", t))
        elif MDNA_TITLE.match(t) or MKT_RISK_TITLE.match(t) or CONTROLS_TITLE.match(t) \
                or (RISK_TITLE.match(t) and len(t) < 40) or SIGNATURES.match(t):
            out.append(Heading(i, None, t))
        i += 1
    return out


def _is_mdna_start(h: Heading) -> bool:
    # "7" accepts 10-Qs that mislabel the MD&A with the 10-K item number (observed in the audit)
    return bool(MDNA_TITLE.match(h.title)) and h.item in ("2", "7", None)


def _is_mdna_end(h: Heading) -> bool:
    if h.item in ("3", "4", "part2"):
        return True
    if h.item == "1" and "legal" in h.title.lower():
        return True
    return h.item is None and bool(MKT_RISK_TITLE.match(h.title) or CONTROLS_TITLE.match(h.title))


def _is_risk_start(h: Heading) -> bool:
    return (h.item == "1a" and bool(RISK_TITLE.match(h.title))) or (
        h.item is None and bool(RISK_TITLE.match(h.title)) and len(h.title) < 40)


def _is_risk_end(h: Heading) -> bool:
    if h.item in ("2", "3", "4", "5", "6"):
        return True
    return h.item is None and bool(SIGNATURES.match(h.title))


def _span_paragraphs(blocks: list[Block], start: int, end: int) -> tuple[list[Paragraph], int]:
    """Narrative paragraphs in blocks[start:end], with page artifacts removed and paragraphs
    split by page breaks merged back together."""
    texts: list[tuple[int, str]] = []
    n_tables = 0
    for b in blocks[start:end]:
        if b.kind == "table" and not narrative_like_table(b):
            n_tables += 1
            continue
        if is_page_artifact(b.text):
            continue
        t = b.text.strip()
        if not t or sum(ch.isalpha() for ch in t) < 2:
            continue
        if len(t) <= 40 and RUNNING_HEADER.match(t):   # running page headers such as "PART I Item 2"
            continue
        if texts and not re.search(r"[.:;?!)\"']$", texts[-1][1]) and t[0].islower() \
                and len(texts[-1][1]) > 60:
            texts[-1] = (texts[-1][0], texts[-1][1] + " " + t)
            continue
        texts.append((b.idx, t))
    paras, pos = [], 0
    for k, (bidx, t) in enumerate(texts):
        is_heading = len(t) < 120 and not re.search(r"[.:;?!]$", t) and len(t.split()) <= 14
        paras.append(Paragraph(k, bidx, t, is_heading, pos, pos + len(t)))
        pos += len(t) + 2
    return paras, n_tables


def _best_span(blocks, headings, is_start, is_end, after_block: int = -1):
    """Among start candidates, choose the one with the most words before its end heading."""
    best = None
    starts = [h for h in headings if is_start(h) and h.block_idx > after_block]
    for h in starts:
        first = h.block_idx + h.consumed
        end_h = next((e for e in headings if e.block_idx >= first and is_end(e)), None)
        end = end_h.block_idx if end_h else len(blocks)
        words = sum(len(b.text.split()) for b in blocks[first:end]
                    if b.kind == "text" or narrative_like_table(b))
        cand = (words, h, first, end, end_h is not None)
        if best is None or words > best[0]:
            best = cand
    return best, len(starts)


def extract_mdna(blocks: list[Block], headings: list[Heading] | None = None) -> SectionExtraction:
    headings = headings if headings is not None else find_headings(blocks)
    best, n = _best_span(blocks, headings, _is_mdna_start, _is_mdna_end)
    ext = SectionExtraction("mdna", "absent", n_start_candidates=n)
    if best is None:
        ext.failure_reason = "no_mdna_heading_found"
        return ext
    words, h, first, end, has_end = best
    paras, n_tables = _span_paragraphs(blocks, first, end)
    ext.start_block, ext.end_block, ext.n_tables = h.block_idx, end, n_tables
    ext.method = ("item_heading_longest_span" if h.item == "2" else
                  "nonstandard_item_number_heading" if h.item else "title_heading_longest_span")
    body = [p for p in paras if not p.is_heading]
    ext.n_words = sum(len(p.text.split()) for p in body)
    if ext.n_words < MIN_MDNA_WORDS:
        ext.status, ext.failure_reason = "failed", f"span_too_short({ext.n_words}_words)"
        return ext
    ext.status = "extracted"
    ext.paragraphs = paras
    ext.n_paragraphs = len(body)
    ext.n_chars = sum(len(p.text) for p in body)
    q = 1.0
    if h.item != "2":
        q -= 0.2
    if not has_end:
        q -= 0.3
    if n > 3:
        q -= 0.1
    ext.heuristic_quality = round(max(q, 0.0), 2)
    return ext


def extract_risk(blocks: list[Block], headings: list[Heading] | None = None,
                 mdna_end_block: int | None = None) -> SectionExtraction:
    headings = headings if headings is not None else find_headings(blocks)
    after = mdna_end_block if mdna_end_block is not None else -1
    best, n = _best_span(blocks, headings, _is_risk_start, _is_risk_end, after_block=after - 1)
    ext = SectionExtraction("risk", "absent", n_start_candidates=n)
    if best is None:
        ext.failure_reason = "no_part2_item1a_heading_found"
        return ext
    words, h, first, end, has_end = best
    paras, n_tables = _span_paragraphs(blocks, first, end)
    ext.start_block, ext.end_block, ext.n_tables = h.block_idx, end, n_tables
    ext.method = "item_heading_longest_span" if h.item else "title_heading_longest_span"
    body = [p for p in paras if not p.is_heading]
    ext.paragraphs = paras
    ext.n_paragraphs = len(body)
    ext.n_words = sum(len(p.text.split()) for p in body)
    ext.n_chars = sum(len(p.text) for p in body)
    full_text = " ".join(p.text for p in body)
    if ext.n_words == 0:
        ext.status, ext.failure_reason = "failed", "empty_span"
    elif ext.n_words < MIN_RISK_FULL_WORDS and NO_MATERIAL.search(full_text):
        ext.status = "no_material_changes"
    elif ext.n_words < MIN_RISK_FULL_WORDS:
        ext.status = "short_reference"
    else:
        ext.status = "extracted"
    q = 1.0 - (0.0 if h.item else 0.2) - (0.0 if has_end else 0.3)
    ext.heuristic_quality = round(max(q, 0.0), 2)
    return ext
