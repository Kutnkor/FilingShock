"""Interface smoke tests and reconciliation between interface totals and saved artifacts.

These run against the local research data or the bundled sample package when present and are skipped
otherwise (a clean checkout has neither until `filingshock reproduce` or `demo-prepare` has run).
"""

import os
from pathlib import Path

import pytest

from filingshock.config import PROJECT_ROOT, Config

PAGES = PROJECT_ROOT / "src" / "filingshock" / "ui" / "pages"
RESEARCH = Config.load().data_dir / "processed" / "events.parquet"
SAMPLE = PROJECT_ROOT / "sample" / "processed" / "events.parquet"


def _run_page(name: str, mode: str, query: dict | None = None):
    from streamlit.testing.v1 import AppTest
    os.environ["FILINGSHOCK_MODE"] = mode
    at = AppTest.from_file(str(PAGES / f"{name}.py"), default_timeout=120)
    for k, v in (query or {}).items():
        at.query_params[k] = v
    at.run()
    return at


@pytest.mark.parametrize("mode,path", [("research", RESEARCH), ("sample", SAMPLE)])
@pytest.mark.parametrize("page", ["overview", "comparison", "replay", "results", "methodology"])
def test_pages_render_without_exceptions(mode, path, page):
    if not Path(path).exists():
        pytest.skip(f"no {mode} data present")
    at = _run_page(page, mode)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.markdown) > 3          # the page rendered real content, not just a stop


def test_ui_totals_reconcile_with_artifacts():
    if not RESEARCH.exists():
        pytest.skip("no research data present")
    os.environ["FILINGSHOCK_MODE"] = "research"
    from filingshock.ui import data as D
    d = D.load.__wrapped__() if hasattr(D.load, "__wrapped__") else D.load()
    if d.metrics is None:
        pytest.skip("no primary run")
    table = D.event_table(d)
    with_pred = table[table["prediction_status"] == "retrospective_out_of_time"]
    assert len(with_pred) == d.metrics["sample"]["n_events"]
    assert int(with_pred["y"].sum()) == d.metrics["sample"]["n_positive"]
    assert (table["eligibility_status"] == "eligible").sum() == d.dataset["n_final_events"]
    assert d.funnel["stages"][-1]["remaining"] == d.dataset["n_final_events"]
    assert d.run_manifest["dataset"]["dataset_id"] == d.dataset["dataset_id"]
    from filingshock.audit import reconcile_run
    assert reconcile_run(D.layout().experiments / d.run_id)["all_ok"]


def test_results_panel_shows_prospective_outcomes_apart_from_research(tmp_path, monkeypatch):
    if not SAMPLE.exists():
        pytest.skip("no sample package present")
    from datetime import datetime, timezone

    from filingshock.forecasts import ForecastLedger
    from filingshock.prospective_outcomes import OutcomeLog
    from filingshock.storage import DataLayout
    for d in ("processed", "experiments", "interim"):          # sample research artifacts, own forecasts dir
        (tmp_path / d).symlink_to(SAMPLE.parents[1] / d)
    layout = DataLayout(tmp_path)
    led = ForecastLedger(layout)

    def forecast(i, session, cutoff, close):
        return {"forecast_id": f"pf-{i}", "event_id": f"ev-p{i}", "ticker": "ADSK", "accession_number": f"acc-{i}",
                "observed_at_utc": "2026-10-14T11:00:00+00:00", "decision_cutoff_utc": cutoff.isoformat(),
                "target_session": session, "target_close_utc": close.isoformat(), "threshold_abs_move": 0.031,
                "p_B": 0.2, "p_C": 0.3, "probability": 0.3, "model_version": "logistic_l2:prod2026:ds-x"}

    c1 = datetime(2026, 10, 14, 13, 0, tzinfo=timezone.utc)
    f1 = led.append(forecast(1, "2026-10-14", c1, c1.replace(hour=20)), now=c1.replace(hour=12))
    c2 = datetime(2099, 1, 5, 14, 0, tzinfo=timezone.utc)
    led.append(forecast(2, "2099-01-05", c2, c2.replace(hour=21)), now=c1.replace(hour=12))
    OutcomeLog(layout).append({"forecast_id": "pf-1", "forecast_sequence": f1["sequence"], "event_id": "ev-p1",
                               "target_close_utc": f1["target_close_utc"], "outcome_status": "ok", "abs_r": 0.047,
                               "y": 1, "brier_B": 0.64, "brier_C": 0.49}, now=datetime(2026, 10, 15, tzinfo=timezone.utc))
    monkeypatch.setenv("FILINGSHOCK_DATA_DIR", str(tmp_path))
    at = _run_page("results", "research")
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    for expected in ("closed; outcome scored", "not closed yet", "4.70%", "0.4900", "Prospective only.",
                     "1 prospective event(s) scored so far"):
        assert expected in text, expected
