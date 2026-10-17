"""Append-only, hash-chained prospective forecast ledger."""

import json
from datetime import datetime, timezone

import pytest

from filingshock.forecasts import CATEGORY, ForecastLedger, LedgerError
from filingshock.storage import DataLayout


def rec(i):
    return {"forecast_id": f"f{i}", "event_id": f"ev-{i}", "observed_at_utc": "2030-01-02T21:00:00+00:00",
            "decision_cutoff_utc": "2030-01-03T14:00:00+00:00", "probability": 0.2 + i / 100,
            "model_version": "logistic_l2:C:fold2030"}


def test_append_before_cutoff_and_chain(tmp_path):
    led = ForecastLedger(DataLayout(tmp_path).ensure())
    now = datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc)
    a = led.append(rec(1), now=now)
    b = led.append(rec(2), now=now)
    assert a["prediction_category"] == CATEGORY and b["prev_record_sha256"] == a["record_sha256"]
    assert led.verify_chain() == (True, "ok")


def test_cannot_record_after_cutoff(tmp_path):
    led = ForecastLedger(DataLayout(tmp_path).ensure())
    with pytest.raises(LedgerError):
        led.append(rec(1), now=datetime(2030, 1, 3, 15, 0, tzinfo=timezone.utc))


def test_tampering_is_detected_and_blocks_appends(tmp_path):
    led = ForecastLedger(DataLayout(tmp_path).ensure())
    now = datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc)
    led.append(rec(1), now=now)
    led.append(rec(2), now=now)
    lines = led.path.read_text().splitlines()
    first = json.loads(lines[0])
    first["probability"] = 0.99                      # silent overwrite of an earlier forecast
    led.path.write_text(json.dumps(first) + "\n" + lines[1] + "\n")
    ok, _ = led.verify_chain()
    assert not ok
    with pytest.raises(LedgerError):
        led.append(rec(3), now=now)


@pytest.mark.parametrize("blob", ["{broken}\n", "[]\n", "null\n", "{\"sequence\":0}", "\xff\n"])
def test_corrupt_records_report_integrity_failures(tmp_path, blob):
    led = ForecastLedger(DataLayout(tmp_path).ensure())
    led.path.write_bytes(blob.encode("latin-1"))
    ok, reason = led.verify_chain()
    assert not ok and reason
    with pytest.raises(LedgerError):
        led.append(rec(1), now=datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc))


def test_rehashed_sequence_corruption_is_rejected(tmp_path):
    led = ForecastLedger(DataLayout(tmp_path).ensure())
    row = led.append(rec(1), now=datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc))
    row["sequence"] = 7
    row["record_sha256"] = led._hash(row)
    led.path.write_text(json.dumps(row) + "\n")
    assert led.verify_chain() == (False, "record 0 has an invalid sequence")
    with pytest.raises(LedgerError, match="invalid sequence"):
        led.verified_records()


def test_corrupt_ledgers_are_withheld_from_ui(tmp_path):
    from filingshock.prospective_outcomes import OutcomeLog
    from filingshock.ui.data import _ledger, _prospective_outcomes

    layout = DataLayout(tmp_path).ensure()
    for log, reader in ((ForecastLedger(layout), _ledger), (OutcomeLog(layout), _prospective_outcomes)):
        log.path.write_text("{broken}\n")
        frame, ok, reason = reader(str(tmp_path), log.path.stat().st_mtime_ns)
        assert frame.empty and not ok and "invalid JSON" in reason
