"""Sparse submissions remain usable for later index-page acceptance reconciliation."""

import pandas as pd

from filingshock.sources.sec import submissions_to_frame


def test_missing_acceptance_timestamp_is_explicitly_unknown():
    doc = {"cik": "1", "filings": {"recent": {
        "accessionNumber": ["0000000001-24-000001"], "form": ["10-Q"], "filingDate": ["2024-05-01"]}}}
    frame = submissions_to_frame(doc)
    assert len(frame) == 1
    assert frame.loc[0, "form_type"] == "10-Q"
    assert pd.isna(frame.loc[0, "acceptance_utc"])
    assert pd.isna(frame.loc[0, "acceptance_raw"])
    assert str(frame["acceptance_utc"].dt.tz) == "UTC"


def test_empty_submission_history_preserves_output_schema():
    empty = submissions_to_frame({"cik": "1", "filings": {"recent": {}}})
    populated = submissions_to_frame({"cik": "1", "filings": {"recent": {
        "accessionNumber": ["acc"], "acceptanceDateTime": ["2024-05-01T20:10:00Z"]}}})
    assert empty.empty
    assert list(empty.columns) == list(populated.columns)
    assert str(empty["acceptance_utc"].dt.tz) == "UTC"


def test_recent_record_wins_over_older_duplicate():
    frame = submissions_to_frame({"cik": 1, "filings": {"recent": {
        "accessionNumber": ["acc"], "acceptanceDateTime": ["2024-05-01T20:10:00Z"]}},
        "_older": [{"accessionNumber": ["acc", "older"],
                    "acceptanceDateTime": ["2024-05-01T21:10:00Z", "2023-05-01T20:10:00Z"]}]})
    assert frame["accession_number"].tolist() == ["older", "acc"]
    assert frame.loc[1, "acceptance_utc"] == pd.Timestamp("2024-05-01T20:10:00Z")
