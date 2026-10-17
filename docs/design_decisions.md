# Design decisions

Running note for the interface. Update it when a decision changes.

## Users and the main task

Quantitative researchers, data practitioners and technical reviewers who want to (1) understand what is
predicted, (2) read what changed in a filing, (3) see a genuine out-of-time prediction and its outcome, and
(4) judge whether text improved the numeric baseline. First-time visitors need the question and the
target definition immediately; repeat users need filters that persist and fast navigation between an
event's comparison, prediction and provenance.

## Visual direction: restrained research workbench

Approaches considered:

1. **Dark trading terminal** — dense and familiar to market users, but it frames the tool as a trading
   product and makes long prose tiring to read. Rejected.
2. **Editorial report** — generous serif typography, but too little density for the event table and results.
3. **Research workbench (chosen)** — warm off-white page (`#f7f6f2`), white document surfaces, ink text,
   one interactive accent (deep blue `#1f4e79`), compact tables, rules instead of cards.

Reasons: filings are long prose that must be compared side by side; numbers must be scanned in tables;
the product should not resemble a trading signal.

## Tokens (`src/filingshock/ui/theme.css`)

* Text: ink `#1b1f24`, supporting `#464e5a`, muted `#5d6571`.
* Surfaces: page `#f7f6f2`, document `#ffffff`, secondary `#f1efe9`, rules `#d9d5cb`.
* Accent / selection: `#1f4e79` on `#e3ebf3`.
* States: info, warning, error, unavailable each have an ink and a tint; every state message has a bold
  title and text, and is rendered with `role="status"`.
* Diff status: added (teal), removed (brown), edited (indigo), moved/unchanged (grey). These colours
  encode **editing status only**; none is green/red, so added or removed text never reads as good or bad
  news. Every chip carries a text label, so colour is never the only cue.
* Typography: system sans (Streamlit's bundled Source Sans) for the interface; a serif stack (Iowan Old
  Style / Charter / Georgia) for filing text, capped at 68 characters per line; tabular numerals in data.
  No web-font downloads.
* Spacing scale 4/8/12/16/24/32 px; one 3 px radius; no shadows; motion disabled under
  `prefers-reduced-motion`.

## Patterns

Page heading with context · data banner (mode, dataset id and build time, coverage, latest price
session, primary run) · event identity strip · compact fact list · data table (Streamlit dataframe or
escaped HTML table) · comparison row/stack · source link (opens SEC.gov in a new tab) · status message ·
chart caption stating cohort, units, sample size and metric direction.

## Probability and outcome semantics

* Probability is always labelled "probability that |open→close| exceeds X%" with the threshold value
  beside it, plus Model B and the constant baseline for context. The word "confidence" is not used.
* A grey chip states "Magnitude probability, not a direction or trading signal".
* Realized outcomes are neutral text ("Large move (y = 1)"), never green/red.

## Tables

The Event Explorer uses Streamlit's dataframe (horizontal scroll, sortable columns, single-row
selection) instead of converting rows to cards on narrow screens, so columns remain comparable. Results
tables are escaped HTML in a horizontal-scroll wrapper; the primary model row is tinted.

## Comparison behaviour

* Default view shows **changed paragraphs only**, in document order, 25 rows per page.
* Summary chips (counts per status) with definitions in an expander; the "Show" control filters to a
  category. Search highlights matches with `<mark>`; topic filters use the topics-1.0 lexicon.
* Paragraph view by default; "Word-level changes" toggles `<ins>/<del>` inside edited pairs.
* Layout: side by side with a sticky source header, or **stacked pairs** for narrow screens. At widths
  below 760 px the side-by-side grid collapses to one column automatically; synchronized scrolling is
  not implemented because additions and removals make positional sync unreliable — rows are already
  aligned pairs.
* All filing text is HTML-escaped before highlight markup is added.

## Responsive adaptations

* 16 px side gutters below 640 px; filter rows wrap (Streamlit columns stack on narrow screens).
* Comparison: stacked pairs + single-column grid; Replay: before/after panels stack.
* Charts use container width; every chart has a table view in an expander.

## Persistence

* Session-only: filters (kept across page switches in `st.session_state['_persist']`), revealed outcomes.
* URL: `?event=<event_id>` selects an event on the comparison and replay pages (shareable locally).
* Durable: saved experiment artifacts, the prospective forecast ledger. No user preferences are stored.

## Revisions

* 2026-10-17: forest plot ordered primary → robustness → ablations → nested; calibration legend restored;
  year labels horizontal; funnel reason table moved to full width (columns were clipped in a half-width column).
* 2026-10-17: sidebar starts in `auto` state — at 390 px the expanded sidebar covered the content and intercepted
  clicks (found with a headless-Chrome check); fact-list label column capped at 12rem so values are not squeezed.
