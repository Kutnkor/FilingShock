"""Read-only verification of forecast/outcome chains and their evidence links."""

from filingshock.forecasts import ForecastLedger, LedgerError
from filingshock.prospective_outcomes import OutcomeLog
from filingshock.storage import DataLayout


def verify_ledgers(layout: DataLayout) -> dict:
    report = {"all_ok": True, "logs": {}, "errors": []}
    verified = {}
    for name, log in (("forecasts", ForecastLedger(layout)), ("outcomes", OutcomeLog(layout))):
        try:
            records = log.verified_records()
        except LedgerError as exc:
            report["logs"][name] = {"status": "invalid", "n_records": None}
            report["errors"].append({"log": name, "reason": str(exc)})
            continue
        verified[name] = records
        report["logs"][name] = {"status": "valid" if records else "empty", "n_records": len(records)}
        ids = [r.get("forecast_id") for r in records]
        if None in ids or len(ids) != len(set(ids)):
            report["errors"].append({"log": name, "reason": "missing or duplicate forecast_id"})
    if len(verified) == 2:
        forecasts = {r.get("forecast_id"): r for r in verified["forecasts"]}
        for outcome in verified["outcomes"]:
            fid = outcome.get("forecast_id")
            forecast = forecasts.get(fid)
            if forecast is None:
                report["errors"].append({"log": "outcomes", "forecast_id": fid, "reason": "forecast not found"})
                continue
            for field, expected in (("forecast_record_sha256", forecast["record_sha256"]),
                                    ("forecast_sequence", forecast["sequence"]), ("event_id", forecast.get("event_id"))):
                if outcome.get(field) != expected:
                    report["errors"].append({"log": "outcomes", "forecast_id": fid,
                                             "reason": f"{field} does not match the recorded forecast"})
    report["all_ok"] = not report["errors"]
    return report
