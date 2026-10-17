"""Section extraction (ToC rejection, hidden content, page artifacts) and paragraph alignment."""

from filingshock.parsing.align import align_paragraphs, word_diff
from filingshock.parsing.html import html_to_blocks
from filingshock.parsing.sections import extract_mdna, extract_risk, find_headings


def _extract(html):
    bl = html_to_blocks(html)
    hs = find_headings(bl)
    m = extract_mdna(bl, hs)
    return bl, m, extract_risk(bl, hs, m.end_block)


def test_toc_and_cross_references_are_not_the_mdna(mdna_html):
    _, m, _ = _extract(mdna_html)
    assert m.status == "extracted"
    assert m.n_start_candidates == 2                  # ToC line + real heading; table ToC is ignored
    texts = [p.text for p in m.paragraphs]
    assert texts[1].startswith("The following discussion")
    assert texts[-1].startswith("For a discussion of market risk")
    assert not any("QUANTITATIVE AND QUALITATIVE" in t for t in texts)


def test_hidden_and_script_content_removed(mdna_html):
    bl, _, _ = _extract(mdna_html)
    joined = " ".join(b.text for b in bl)
    assert "hidden text" not in joined and "alert(" not in joined


def test_page_artifacts_removed_and_split_paragraph_merged(mdna_html):
    _, m, _ = _extract(mdna_html)
    texts = [p.text for p in m.paragraphs]
    assert "23" not in texts and "Table of Contents" not in texts and "PART I Item 2" not in texts
    merged = [t for t in texts if t.startswith("Demand from enterprise")]
    assert merged and merged[0].endswith("executive approvals.")


def test_risk_section_semantics(mdna_html):
    _, _, r = _extract(mdna_html)
    assert r.status == "no_material_changes"
    html = mdna_html.replace(b"<p>ITEM 1A. RISK FACTORS</p>", b"")
    _, _, r2 = _extract(html)
    assert r2.status == "absent"                     # absent is not "no risk"


def test_missing_mdna_is_reported_as_absent():
    _, m, _ = _extract(b"<html><body><p>ITEM 1. FINANCIAL STATEMENTS</p><p>text</p></body></html>")
    assert m.status == "absent" and m.failure_reason == "no_mdna_heading_found"


def test_alignment_statuses_and_one_to_one():
    ref = ["Revenue grew 10% to $100 million in the quarter.",
           "We face competition from many vendors in all of our markets worldwide.",
           "Our credit facility contains customary covenants and restrictions on indebtedness.",
           "This paragraph about legacy hardware products was removed entirely."]
    cur = ["Revenue grew 12% to $112 million in the quarter.",
           "Our credit facility contains customary covenants and restrictions on indebtedness.",
           "We face strong competition from many vendors in all of our markets worldwide, including new AI entrants.",
           "A brand new paragraph about generative AI product launches and pricing changes."]
    res = align_paragraphs(ref, cur)
    st = {(r.ref_ordinal if r.ref_ordinal is not None else None, r.cur_ordinal): r.status
          for r in res.pairs.itertuples()}
    assert res.pairs.loc[res.pairs["cur_ordinal"] == 0, "status"].item() == "numeric_only"
    assert res.pairs.loc[res.pairs["cur_ordinal"] == 3, "status"].item() == "added"
    assert res.pairs.loc[res.pairs["ref_ordinal"] == 3, "status"].item() == "removed"
    assert res.pairs.loc[res.pairs["cur_ordinal"] == 2, "status"].item() in ("minor_edit", "substantial_edit")
    assert res.pairs.loc[res.pairs["cur_ordinal"] == 1, "moved"].item() or res.pairs.loc[res.pairs["cur_ordinal"] == 2, "moved"].item()
    matched = res.pairs.dropna(subset=["ref_ordinal", "cur_ordinal"])
    assert matched["ref_ordinal"].is_unique and matched["cur_ordinal"].is_unique
    assert res.summary["n_added"] == 1 and res.summary["n_removed"] == 1
    assert st  # pairs exist


def test_identical_repeated_paragraphs_match_once_each():
    ref = ["Same text.", "Same text.", "Other text here."]
    cur = ["Same text.", "Other text here."]
    res = align_paragraphs(ref, cur)
    assert res.summary["n_removed"] == 1 and res.summary["n_added"] == 0


def test_word_diff_marks_changes():
    ops = word_diff("revenue grew 10 percent", "revenue grew 12 percent")
    assert ("delete", "10") in ops and ("insert", "12") in ops
