"""Feature catalog: definition, source, units, transformation and availability rule per feature."""

from __future__ import annotations

PRICES = "Provider daily bars (yfinance; split-adjusted, not dividend-adjusted)"
FACTS = "SEC Company Facts (us-gaap, USD), filtered to filings available by the cutoff"
TEXT = "Extracted MD&A / risk sections of the current 10-Q and its prior-year same-quarter reference"
BEFORE = "Sessions strictly before the target session"
AT_CUTOFF = "Reported by filings with acceptance + lag <= decision cutoff"

CATALOG: list[dict] = [
    # market
    {"name": "mkt_absoc_mean_20", "group": "market", "definition": "Mean |close/open - 1| over the last 20 valid sessions",
     "source": PRICES, "units": "fraction (0.01 = 1%)", "transformation": "none", "availability": BEFORE},
    {"name": "mkt_absoc_mean_60", "group": "market", "definition": "Mean |close/open - 1| over the last 60 valid sessions (= m, the threshold base)",
     "source": PRICES, "units": "fraction", "transformation": "none", "availability": BEFORE},
    {"name": "mkt_ret_vol_60", "group": "market", "definition": "Std. dev. of daily log close-to-close returns, last 60 sessions",
     "source": PRICES, "units": "log return", "transformation": "none", "availability": BEFORE},
    {"name": "mkt_large_move_freq_250", "group": "market", "definition": "Share of the last 250 sessions whose |open-to-close| exceeded 2x the mean of its own previous 60 sessions",
     "source": PRICES, "units": "share", "transformation": "none", "availability": BEFORE},
    {"name": "mkt_log_rel_volume_5_60", "group": "market", "definition": "log(mean volume last 5 sessions / mean volume last 60 sessions)",
     "source": PRICES, "units": "log ratio", "transformation": "log", "availability": BEFORE},
    {"name": "mkt_prev_session_abs_ret", "group": "market", "definition": "|log close-to-close return| of the last completed session",
     "source": PRICES, "units": "log return", "transformation": "abs", "availability": BEFORE},
    {"name": "mkt_spy_absoc_20", "group": "market", "definition": "SPY mean |open-to-close| over the last 20 sessions (market activity)",
     "source": PRICES, "units": "fraction", "transformation": "none", "availability": BEFORE},
    {"name": "mkt_igv_absoc_20", "group": "market", "definition": "IGV (software ETF) mean |open-to-close| over the last 20 sessions (sector activity)",
     "source": PRICES, "units": "fraction", "transformation": "none", "availability": BEFORE},
    {"name": "mkt_prev_event_abs_ratio", "group": "market", "definition": "|r| / m of the company's previous filing event whose outcome had matured",
     "source": PRICES, "units": "ratio", "transformation": "none", "availability": "Previous event's target close <= decision cutoff"},
    # context
    {"name": "ctx_earnings_8k_in_window", "group": "context", "definition": "1 if an 8-K with Item 2.02 (results of operations) was accepted between the previous close and the cutoff",
     "source": "SEC submissions (8-K items)", "units": "0/1", "transformation": "none", "availability": AT_CUTOFF},
    {"name": "ctx_log_days_since_earnings_8k", "group": "context", "definition": "log(1 + days from the last admissible Item 2.02 8-K to the cutoff); missing if > 400 days or none",
     "source": "SEC submissions (8-K items)", "units": "log days", "transformation": "log1p", "availability": AT_CUTOFF},
    {"name": "ctx_log_filing_lag_days", "group": "context", "definition": "log(1 + days from fiscal-period end to acceptance)",
     "source": "SEC submissions", "units": "log days", "transformation": "log1p", "availability": "Known at acceptance"},
    # financial
    {"name": "fin_rev_yoy", "group": "financial", "definition": "Quarterly revenue / same-quarter revenue one year earlier - 1 (same XBRL concept for both)",
     "source": FACTS, "units": "fraction", "transformation": "missing if prior revenue <= 0", "availability": AT_CUTOFF},
    {"name": "fin_op_margin", "group": "financial", "definition": "Quarterly operating income / quarterly revenue",
     "source": FACTS, "units": "fraction", "transformation": "missing if revenue <= 0", "availability": AT_CUTOFF},
    {"name": "fin_op_margin_yoy_pp", "group": "financial", "definition": "Operating margin minus the same quarter's margin one year earlier",
     "source": FACTS, "units": "fraction difference (0.01 = 1 percentage point)", "transformation": "none", "availability": AT_CUTOFF},
    {"name": "fin_ocf_margin", "group": "financial", "definition": "Quarterly operating cash flow / quarterly revenue; quarterly OCF may be YTD(t) - YTD(t-1q) only when both share the fiscal-year start",
     "source": FACTS, "units": "fraction", "transformation": "YTD differencing where needed", "availability": AT_CUTOFF},
    {"name": "fin_cash_to_assets", "group": "financial", "definition": "Cash and cash equivalents / total assets at period end",
     "source": FACTS, "units": "fraction", "transformation": "none", "availability": AT_CUTOFF},
    {"name": "fin_liab_to_assets", "group": "financial", "definition": "1 - stockholders' equity / total assets at period end (leverage)",
     "source": FACTS, "units": "fraction", "transformation": "none", "availability": AT_CUTOFF},
    {"name": "fin_liab_to_assets_yoy_change", "group": "financial", "definition": "Leverage minus leverage at the same period end one year earlier",
     "source": FACTS, "units": "fraction difference", "transformation": "none", "availability": AT_CUTOFF},
    # text
    {"name": "txt_mdna_log_len_ratio", "group": "text_length", "definition": "log(MD&A words in current / words in reference)",
     "source": TEXT, "units": "log ratio", "transformation": "log", "availability": "Both filings available by the cutoff"},
    {"name": "txt_mdna_uncertainty_rate", "group": "text_dictionary", "definition": "Loughran-McDonald uncertainty words per 1,000 MD&A words (point-in-time list)",
     "source": TEXT + "; LM Master Dictionary 1993-2025", "units": "per 1,000 words", "transformation": "none", "availability": "Word lists restricted to words added by the event year"},
    {"name": "txt_mdna_uncertainty_change", "group": "text_dictionary", "definition": "Uncertainty rate (current) - uncertainty rate (reference)",
     "source": TEXT + "; LM", "units": "per 1,000 words", "transformation": "difference", "availability": "as above"},
    {"name": "txt_mdna_negative_rate", "group": "text_dictionary", "definition": "Loughran-McDonald negative words per 1,000 MD&A words (point-in-time list)",
     "source": TEXT + "; LM", "units": "per 1,000 words", "transformation": "none", "availability": "as above"},
    {"name": "txt_mdna_negative_change", "group": "text_dictionary", "definition": "Negative rate (current) - negative rate (reference)",
     "source": TEXT + "; LM", "units": "per 1,000 words", "transformation": "difference", "availability": "as above"},
    {"name": "txt_changed_topic_share", "group": "text_dictionary", "definition": "Share of added or substantially edited MD&A paragraphs that mention a topics-1.0 topic (liquidity, debt, receivables, demand, customer concentration, litigation, constraint)",
     "source": TEXT + "; resources/dictionaries/topics_v1.yaml", "units": "share", "transformation": "none", "availability": "Both filings available by the cutoff"},
    {"name": "txt_mdna_prop_added", "group": "text_paragraph_change", "definition": "Added MD&A paragraphs / current body paragraphs",
     "source": TEXT + "; aligner fs-align-1.0", "units": "share", "transformation": "none", "availability": "Both filings available by the cutoff"},
    {"name": "txt_mdna_prop_removed", "group": "text_paragraph_change", "definition": "Removed paragraphs / reference body paragraphs",
     "source": TEXT, "units": "share", "transformation": "none", "availability": "as above"},
    {"name": "txt_mdna_prop_substantial", "group": "text_paragraph_change", "definition": "Substantially edited paragraphs (word-sequence similarity < 0.90) / current body paragraphs",
     "source": TEXT, "units": "share", "transformation": "none", "availability": "as above"},
    {"name": "txt_mdna_lexical_distance", "group": "text_paragraph_change", "definition": "1 - cosine similarity of the two sections' raw word-count vectors (no learned vocabulary)",
     "source": TEXT, "units": "0-1", "transformation": "none", "availability": "as above"},
    {"name": "txt_risk_prop_added", "group": "text_paragraph_change", "definition": "Added risk-factor paragraphs / current risk paragraphs; defined only when both risk sections are fully present (missing otherwise, with an indicator)",
     "source": TEXT, "units": "share", "transformation": "none", "availability": "as above"},
]
CATALOG_BY_NAME = {c["name"]: c for c in CATALOG}
