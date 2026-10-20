"""Untrusted submission names must not escape their local cache or alter a request URL."""

import pytest

from filingshock.config import Config
from filingshock.sources.sec import SecAccessError, SecClient
from filingshock.storage import DataLayout


@pytest.mark.parametrize("name", ["../outside.htm", "/tmp/outside.htm", "..\\outside.htm", "doc.htm?x=1",
                                  "doc.htm#fragment", "C:outside.htm", "bad\x00.htm", ".", "..", ""])
def test_external_document_names_cannot_escape_the_cache(tmp_path, name):
    client = SecClient(Config.load(), DataLayout(tmp_path).ensure(), offline=True)
    with pytest.raises(SecAccessError, match="path component"):
        client.document_path(1, "0000000001-24-000001", name)
    with pytest.raises(SecAccessError, match="path component"):
        client.document_url(1, "0000000001-24-000001", name)


def test_ordinary_sec_document_names_are_preserved(tmp_path):
    client = SecClient(Config.load(), DataLayout(tmp_path), offline=True)
    accession = "0000000001-24-000001"
    path = client.document_path(1, accession, "form10-q_2024.htm")
    assert path == tmp_path / "raw" / "sec" / "filings" / "1" / accession / "form10-q_2024.htm"
    assert client.document_url(1, accession, "form10-q_2024.htm").endswith("/000000000124000001/form10-q_2024.htm")


@pytest.mark.parametrize("name", ["%2e%2e", "%2e%2e%2foutside.htm", "doc%2fother.htm", "doc%3fdownload=1"])
def test_percent_encoded_names_cannot_change_sec_request_paths(tmp_path, name):
    client = SecClient(Config.load(), DataLayout(tmp_path), offline=True)
    with pytest.raises(SecAccessError, match="path component"):
        client.document_url(1, "0000000001-24-000001", name)
    with pytest.raises(SecAccessError, match="path component"):
        client.document_url(1, name, "doc.htm")
