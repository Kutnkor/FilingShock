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
