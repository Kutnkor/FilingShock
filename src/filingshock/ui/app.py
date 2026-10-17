"""FilingShock Streamlit entry point: navigation and shared page chrome only (no business logic)."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `streamlit run` without installation

from filingshock.ui import components as ui  # noqa: E402

st.set_page_config(page_title="FilingShock", page_icon="📄", layout="wide", initial_sidebar_state="auto")
ui.inject_css()

PAGES = Path(__file__).parent / "pages"
nav = st.navigation({
    "Research": [
        st.Page(PAGES / "overview.py", title="Overview / Event Explorer", url_path="overview", default=True),
        st.Page(PAGES / "comparison.py", title="Filing Comparison", url_path="comparison"),
        st.Page(PAGES / "replay.py", title="Prediction & Historical Replay", url_path="replay"),
        st.Page(PAGES / "results.py", title="Research Results", url_path="results"),
        st.Page(PAGES / "methodology.py", title="Methodology & Data Quality", url_path="methodology"),
    ]
})
nav.run()
