# Extraction audit

Two manual reviews were performed. "Manual" here means the developer read the extracted text against
the structure of the source filing; no second reviewer and no inter-rater measure exist.

## 1. Pipeline-validation pilot (50 filings)

Five companies (ADBE, INTU, ADSK, ORCL, CRM), ten 10-Qs each, spaced evenly over 2010–2026
(`data/interim/pilot_sample.csv`), covering Donnelley, Workiva and self-filed HTML styles, older
table-based layouts and inline-XBRL documents.

| Finding | Effect | Resolution |
|---|---|---|
| Oracle 2019-03-18: heading split into letter-spaced spans ("I tem 2. … o f Financial") | MD&A not found (1 of 50) | Table-cell text is now joined without inserted spaces (inline elements do not create word breaks). 50/50 extracted afterwards. |
| Oracle 10-Qs since 2011: Part II Item 1A only refers to the 10-K | Correctly classified `short_reference` | None needed; status kept distinct from "absent" and "no material changes". |
| Underscore separator lines and table footnotes inside MD&A | Separator lines appeared as paragraphs | Paragraphs with fewer than two letters are dropped; short footnotes are kept. |
| Tables of contents (table form and short-line form) | Would yield a near-empty "MD&A" | Longest-span rule selects the real section; verified on every pilot filing (`n_start_candidates`). |

## 2. Stratified audit of the final sample (14 filings, one per company)

Random draw (`random_state=7`) of one final-sample filing per company (`data/interim/audit_sample.csv`).
For each filing the first, second and last MD&A paragraphs and the risk-section classification were
checked against the source. Full table: [`extraction_audit.csv`](extraction_audit.csv).

| Finding | Resolution |
|---|---|
| Microsoft: running page headers "PART I" / "Item 2" kept as heading paragraphs | fs-parser-1.1 drops short running headers. Headings are excluded from all text features, so features were not affected; the display was. |
| Salesforce 2011-05-26 (found through the funnel, not the random draw): MD&A labelled "ITEM 7" in a 10-Q | fs-parser-1.1 accepts item 7 with the MD&A title and records method `nonstandard_item_number_heading` (quality score reduced). |
| Salesforce 2015: missing space where the source has no whitespace between elements | Cosmetic; not fixed. |
| Oracle: "* Not meaningful" table footnotes retained as short paragraphs | Minor; not fixed. |

All 14 audited sections began and ended at the correct boundaries, and every risk-section
classification matched the source text.

## Extraction statistics (fs-parser-1.1)

Every filing that reached the extraction stage was parsed: MD&A extracted for all of them, one through the
non-standard item-number rule. Risk-section statuses across parsed filings: full text, "no material
changes" and "short reference" (counts on the Methodology & Data Quality page). The heuristic quality
score is rule-based (deductions for a missing end heading, a title-only heading, or many candidates);
it is **not** a validated probability of correct extraction.
