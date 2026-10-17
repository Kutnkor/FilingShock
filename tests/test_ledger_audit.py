"""A valid local hash chain must also reference the actual immutable forecast evidence."""

import json
from datetime import datetime, timezone

from filingshock.forecasts import ForecastLedger
from filingshock.ledger_audit import verify_ledgers
from filingshock.prospective_outcomes import OutcomeLog
from filingshock.storage import DataLayout


def test_empty_ledgers_are_reported_explicitly_without_creating_files(tmp_path):
    report = verify_ledgers(DataLayout(tmp_path))
    assert report["all_ok"]
    assert report["logs"]["forecasts"] == {"status": "empty", "n_records": 0}
    assert list(tmp_path.iterdir()) == []


def test_rehashed_outcome_cannot_reference_different_forecast_evidence(tmp_path):
    layout = DataLayout(tmp_path).ensure()
    forecast = ForecastLedger(layout).append(
        {"forecast_id": "f1", "event_id": "e1", "model_version": "m1", "probability": 0.2,
         "observed_at_utc": "2030-01-02T20:00:00Z", "decision_cutoff_utc": "2030-01-03T14:00:00Z"},
        now=datetime(2030, 1, 2, 21, tzinfo=timezone.utc))
    log = OutcomeLog(layout)
    outcome = log.append({"forecast_id": "f1", "event_id": "e1", "forecast_sequence": forecast["sequence"],
                          "forecast_record_sha256": forecast["record_sha256"], "target_close_utc": "2030-01-03T21:00:00Z"},
                         now=datetime(2030, 1, 4, tzinfo=timezone.utc))
    assert verify_ledgers(layout)["all_ok"]
    outcome["forecast_record_sha256"] = "0" * 64
    outcome["record_sha256"] = log._hash(outcome)
    log.path.write_text(json.dumps(outcome) + "\n")
    assert log.verify_chain() == (True, "ok")
    report = verify_ledgers(layout)
    assert not report["all_ok"]
    assert "forecast_record_sha256" in report["errors"][0]["reason"]


def test_verify_ledgers_cli_fails_with_structured_corruption_report(tmp_path, monkeypatch, capsys):
    from filingshock.cli import EXIT_FAIL, main

    monkeypatch.setenv("FILINGSHOCK_DATA_DIR", str(tmp_path))
    layout = DataLayout(tmp_path).ensure()
    ForecastLedger(layout).path.write_text("{broken}\n")
    assert main(["verify-ledgers", "--json"]) == EXIT_FAIL
    report = json.loads(capsys.readouterr().out)
    assert report["logs"]["forecasts"]["status"] == "invalid"
    assert report["logs"]["forecasts"]["n_records"] is None
