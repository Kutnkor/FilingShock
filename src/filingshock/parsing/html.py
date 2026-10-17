"""Convert filing HTML into an ordered list of text blocks.

The source HTML is treated strictly as data: scripts, styles, hidden elements and the
inline-XBRL header are dropped; nothing from the document is ever rendered as HTML.
Each block keeps its ordinal position and character offset in the cleaned text so displayed
excerpts can be traced back to the filing.
"""

from __future__ import annotations

import html as htmlmod
import re
import unicodedata
from dataclasses import dataclass

from lxml import etree
from lxml import html as lxml_html

BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "center", "blockquote",
    "section", "article", "body", "dd", "dt", "dl", "pre", "hr", "page", "form",
}
SKIP_TAGS = {"script", "style", "head", "title", "noscript", "header"}
HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)
PAGE_NUMBER = re.compile(
    r"^(page\s*)?[-–—(\[]?\s*([0-9]{1,3}|[ivxlc]{1,6})\s*[-–—)\]]?$|^[a-z]?-?\d{1,3}-?$", re.I)
TOC_LINK = re.compile(r"^(back\s+to\s+)?(table\s+of\s+contents|index)$", re.I)
WS = re.compile(r"\s+")


@dataclass
class Block:
    idx: int
    kind: str            # text | table
    text: str
    n_rows: int = 0
    digit_share: float = 0.0


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace(" ", " ").replace("​", "")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("•", "•")
    return WS.sub(" ", s).strip()


def _local(tag) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if ":" in tag:
        prefix, name = tag.split(":", 1)
        return f"{prefix.lower()}:{name.lower()}"
    return tag.lower()


def _is_hidden(el) -> bool:
    style = el.get("style") if hasattr(el, "get") else None
    return bool(style and HIDDEN_STYLE.search(style))


def _table_text(el) -> tuple[str, int]:
    rows = []
    for tr in el.iter():
        if _local(tr.tag) != "tr":
            continue
        cells = []
        for td in tr:
            if _local(td.tag) in ("td", "th") and not _is_hidden(td):
                t = normalize_text("".join(td.itertext()))
                if t:
                    cells.append(t)
        if cells:
            rows.append(" ".join(cells))
    return " ".join(rows), len(rows)


def html_to_blocks(raw: bytes) -> list[Block]:
    """Ordered text and table blocks from filing HTML."""
    try:
        doc = lxml_html.document_fromstring(raw)
    except (etree.ParserError, ValueError):
        text = raw.decode("utf-8", errors="replace")
        doc = lxml_html.document_fromstring(f"<html><body><pre>{htmlmod.escape(text)}</pre></body></html>")
    blocks: list[Block] = []
    buf: list[str] = []

    def flush():
        if buf:
            t = normalize_text("".join(buf))
            buf.clear()
            if t:
                blocks.append(Block(len(blocks), "text", t))

    def walk(el):
        tag = _local(el.tag)
        if not tag:  # comments, processing instructions
            if el.tail:
                buf.append(el.tail)
            return
        if tag in SKIP_TAGS or tag == "ix:header" or _is_hidden(el):
            if el.tail:
                buf.append(el.tail)
            return
        if tag == "table":
            flush()
            t, n_rows = _table_text(el)
            if t:
                digits = sum(ch.isdigit() for ch in t)
                blocks.append(Block(len(blocks), "table", t, n_rows, digits / max(len(t), 1)))
            if el.tail:
                buf.append(el.tail)
            return
        if tag == "br":
            flush()
            if el.tail:
                buf.append(el.tail)
            return
        is_block = tag in BLOCK_TAGS
        if is_block:
            flush()
        if el.text:
            buf.append(el.text)
        for child in el:
            walk(child)
        if is_block:
            flush()
        if el.tail:
            buf.append(el.tail)

    body = doc.find("body")
    walk(body if body is not None else doc)
    flush()
    return blocks


def is_page_artifact(text: str) -> bool:
    t = text.strip()
    if len(t) <= 12 and PAGE_NUMBER.match(t):
        return True
    if len(t) <= 30 and TOC_LINK.match(t):
        return True
    return False


def narrative_like_table(b: Block) -> bool:
    """Layout tables (headings, bullets) behave like text; financial tables do not."""
    return b.kind == "table" and b.n_rows <= 2 and len(b.text) < 1500 and b.digit_share < 0.08
