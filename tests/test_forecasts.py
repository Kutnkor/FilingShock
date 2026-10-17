"""Append-only, hash-chained prospective forecast ledger."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from filingshock.forecasts import CATEGORY, ForecastLedger, LedgerError
from filingshock.storage import DataLayout


def rec(i):
    return {"forecast_id": f"f{i}", "event_id": f"ev-{i}", "observed_at_utc": "2030-01-02T21:00:00+00:00",
            "decision_cutoff_utc": "2030-01-03T14:00:00+00:00", "probability": 0.2 + i / 100,
            "model_version": "logistic_l2:C:fold2030"}


def _append_in_process(args):
    root, i = args
    return ForecastLedger(DataLayout(Path(root))).append(rec(i), now=datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc))


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


def test_concurrent_writers_preserve_the_entire_chain(tmp_path):
    layout = DataLayout(tmp_path).ensure()
    now = datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc)

    def append(i):
        return ForecastLedger(layout).append(rec(i), now=now)

    first = append(0)
    prefix = ForecastLedger(layout).path.read_bytes()
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(append, range(1, 41)))
    led = ForecastLedger(layout)
    assert led.verify_chain() == (True, "ok")
    assert len(led.records()) == 41
    assert {r["forecast_id"] for r in [first, *rows]} == {f"f{i}" for i in range(41)}
    assert led.path.read_bytes().startswith(prefix)


def test_independent_processes_preserve_the_chain(tmp_path):
    layout = DataLayout(tmp_path).ensure()
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(_append_in_process, [(str(tmp_path), i) for i in range(12)]))
    led = ForecastLedger(layout)
    assert led.verify_chain() == (True, "ok")
    assert len(led.records()) == 12


def test_failed_publication_preserves_previous_evidence(tmp_path, monkeypatch):
    led = ForecastLedger(DataLayout(tmp_path).ensure())
    now = datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc)
    led.append(rec(1), now=now)
    before = led.path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated filesystem publication failure")

    with monkeypatch.context() as scoped:
        scoped.setattr("filingshock.storage.os.replace", fail_replace)
        with pytest.raises(OSError, match="publication failure"):
            led.append(rec(2), now=now)
    assert led.path.read_bytes() == before
    assert led.verify_chain() == (True, "ok")
    assert list(led.path.parent.glob(".*.tmp")) == []


def test_contending_writer_wait_is_bounded(tmp_path):
    from filingshock.storage import file_lock

    path = tmp_path / "transaction.lock"
    with file_lock(path):
        with pytest.raises(TimeoutError, match="writer lock"):
            with file_lock(path, timeout=0.01):
                pytest.fail("a second writer acquired an already-held lock")


@pytest.mark.parametrize("kind", ["forecast", "outcome"])
def test_concurrent_duplicate_records_are_written_once(tmp_path, kind):
    from filingshock.prospective_outcomes import OutcomeLog

    layout = DataLayout(tmp_path).ensure()
    log = ForecastLedger(layout) if kind == "forecast" else OutcomeLog(layout)
    row = rec(1) if kind == "forecast" else {"forecast_id": "f1", "target_close_utc": "2029-12-01T21:00:00Z"}

    def append(_):
        try:
            log.append(row, now=datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc))
            return True
        except LedgerError as exc:
            assert "already" in str(exc)
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(append, range(16))) == 1
    assert log.verify_chain() == (True, "ok") and len(log.records()) == 1


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5", None])
def test_invalid_forecast_probabilities_are_never_persisted(tmp_path, probability):
    led = ForecastLedger(DataLayout(tmp_path).ensure())
    row = {**rec(1), "probability": probability}
    with pytest.raises(LedgerError, match="finite probability"):
        led.append(row, now=datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc))
    assert not led.path.exists()


@pytest.mark.parametrize("timestamp", [None, "NaT", "not-a-date", "2030-01-02T21:00:00"])
def test_invalid_observation_times_do_not_bypass_cutoff_checks(tmp_path, timestamp):
    led = ForecastLedger(DataLayout(tmp_path).ensure())
    with pytest.raises(LedgerError, match="timezone-aware"):
        led.append({**rec(1), "observed_at_utc": timestamp}, now=datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc))
    assert not led.path.exists()


def test_rehashed_invalid_probability_is_withheld_from_readers(tmp_path):
    led = ForecastLedger(DataLayout(tmp_path).ensure())
    row = led.append(rec(1), now=datetime(2030, 1, 2, 22, 0, tzinfo=timezone.utc))
    row["probability"] = 2.0
    row["record_sha256"] = led._hash(row)
    led.path.write_text(json.dumps(row) + "\n")
    ok, reason = led.verify_chain()
    assert not ok and "finite probability" in reason


@pytest.mark.parametrize("close", [None, "NaT", "2030-01-02T21:00:00", "invalid"])
def test_invalid_close_times_cannot_bypass_outcome_settlement(tmp_path, close):
    from filingshock.prospective_outcomes import OutcomeLog

    log = OutcomeLog(DataLayout(tmp_path).ensure())
    with pytest.raises(LedgerError, match="timezone-aware"):
        log.append({"forecast_id": "f1", "target_close_utc": close}, now=datetime(2030, 1, 4, tzinfo=timezone.utc))
    assert not log.path.exists()


def test_inconsistent_outcome_scores_are_not_accepted(tmp_path):
    from filingshock.prospective_outcomes import OutcomeLog

    log = OutcomeLog(DataLayout(tmp_path).ensure())
    row = {"forecast_id": "f1", "target_close_utc": "2030-01-02T21:00:00Z", "outcome_status": "ok",
           "y": 1, "p_C": 0.2, "brier_C": 0.0}
    with pytest.raises(LedgerError, match="brier_C"):
        log.append(row, now=datetime(2030, 1, 4, tzinfo=timezone.utc))
    assert not log.path.exists()
