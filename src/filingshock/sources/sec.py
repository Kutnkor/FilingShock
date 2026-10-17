"""SEC EDGAR client with throttling, caching, bounded retries and a retrieval manifest.

Downloaded content is untrusted research input: it is stored and parsed as data only.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from filingshock.config import Config
from filingshock.storage import DataLayout, sha256_bytes, write_bytes_atomic

log = logging.getLogger(__name__)

DATA_BASE = "https://data.sec.gov"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


class SecAccessError(RuntimeError):
    """Raised when SEC content cannot be retrieved after bounded retries."""


@dataclass
class Retrieval:
    url: str
    path: Path
    status: str          # downloaded | cached
    sha256: str
    retrieved_at: str    # UTC ISO timestamp of the actual download (never backdated)
    n_bytes: int


class SecClient:
    def __init__(self, cfg: Config, layout: DataLayout, offline: bool = False, session=None):
        self.cfg = cfg
        self.layout = layout
        self.offline = offline
        sec = cfg["sec"]
        self.min_interval = 1.0 / float(sec["max_requests_per_second"])
        self.max_retries = int(sec["max_retries"])
        self.backoff_base = float(sec["backoff_base_seconds"])
        self.timeout = float(sec["timeout_seconds"])
        self._last_request = 0.0
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": cfg.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.manifest_path = layout.raw_sec / "retrieval_manifest.jsonl"
        self.request_count = 0

    # ------------------------------------------------------------------ transport
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, url: str) -> bytes:
        if self.offline:
            raise SecAccessError(f"offline mode: {url} is not cached")
        last_err: str | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.timeout)
                self.request_count += 1
            except requests.RequestException as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code == 404:
                    raise SecAccessError(f"404 not found: {url}")
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code not in (429, 500, 502, 503, 504, 403):
                    break
            if attempt < self.max_retries:
                delay = self.backoff_base * (2 ** attempt)
                log.warning("SEC request failed (%s); retry %d/%d in %.1fs: %s",
                            last_err, attempt + 1, self.max_retries, delay, url)
                time.sleep(delay)
        raise SecAccessError(f"failed after {self.max_retries} retries ({last_err}): {url}")

    def _record(self, r: Retrieval) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "a") as fh:
            fh.write(json.dumps({
                "url": r.url, "path": str(r.path.relative_to(self.layout.root)), "status": r.status,
                "sha256": r.sha256, "retrieved_at": r.retrieved_at, "n_bytes": r.n_bytes,
            }) + "\n")

    def fetch(self, url: str, path: Path, max_age_days: float | None = None) -> Retrieval:
        """Return cached content if present (and fresh enough), else download it.

        max_age_days=None means the content is immutable (e.g. an accession's documents).
        """
        if path.exists():
            age_days = (time.time() - path.stat().st_mtime) / 86400
            if max_age_days is None or age_days <= max_age_days or self.offline:
                data = path.read_bytes()
                return Retrieval(url, path, "cached", sha256_bytes(data), "", len(data))
        data = self._get(url)
        write_bytes_atomic(path, data)
        r = Retrieval(url, path, "downloaded", sha256_bytes(data),
                      datetime.now(timezone.utc).isoformat(timespec="seconds"), len(data))
        self._record(r)
        return r

    # ------------------------------------------------------------------ endpoints
    def submissions(self, cik: int, max_age_days: float = 30) -> dict:
        """Full submissions history: recent block plus any older paginated files."""
        cik10 = f"{cik:010d}"
        base = self.layout.raw_sec / "submissions"
        main = self.fetch(f"{DATA_BASE}/submissions/CIK{cik10}.json", base / f"CIK{cik10}.json", max_age_days)
        doc = json.loads(main.path.read_bytes())
        older = []
        for f in doc.get("filings", {}).get("files", []):
            r = self.fetch(f"{DATA_BASE}/submissions/{f['name']}", base / f["name"], max_age_days)
            older.append(json.loads(r.path.read_bytes()))
        doc["_older"] = older
        doc["_sha256"] = main.sha256
        return doc

    def company_facts(self, cik: int, max_age_days: float = 30) -> Path:
        cik10 = f"{cik:010d}"
        r = self.fetch(f"{DATA_BASE}/api/xbrl/companyfacts/CIK{cik10}.json",
                       self.layout.raw_sec / "companyfacts" / f"CIK{cik10}.json", max_age_days)
        return r.path

    @staticmethod
    def document_url(cik: int, accession: str, filename: str) -> str:
        return f"{ARCHIVE_BASE}/{int(cik)}/{accession.replace('-', '')}/{filename}"

    def document_path(self, cik: int, accession: str, filename: str) -> Path:
        return self.layout.raw_sec / "filings" / str(int(cik)) / accession / filename

    def filing_document(self, cik: int, accession: str, filename: str) -> Retrieval:
        return self.fetch(self.document_url(cik, accession, filename),
                          self.document_path(cik, accession, filename), max_age_days=None)

    def index_headers(self, cik: int, accession: str) -> Retrieval:
        name = f"{accession}-index-headers.html"
        return self.fetch(self.document_url(cik, accession, name),
                          self.document_path(cik, accession, name), max_age_days=None)


def submissions_to_frame(doc: dict) -> pd.DataFrame:
    """Flatten recent + older submission blocks into one filings table (one row per accession)."""
    blocks = [doc["filings"]["recent"]] + list(doc.get("_older", []))
    frames = [pd.DataFrame(b) for b in blocks if b and len(b.get("accessionNumber", []))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "accessionNumber": "accession_number", "filingDate": "filing_date", "reportDate": "report_date",
        "acceptanceDateTime": "acceptance_raw", "form": "form_type", "primaryDocument": "primary_document",
        "primaryDocDescription": "primary_doc_description", "isXBRL": "is_xbrl",
        "isInlineXBRL": "is_inline_xbrl",
    })
    df = df.drop_duplicates("accession_number")
    # The submissions API reports acceptanceDateTime in UTC ("...Z"); verified against index
    # headers (which state Eastern time) in docs/timing_policy.md.
    df["acceptance_utc"] = pd.to_datetime(df["acceptance_raw"], utc=True, errors="coerce")
    df["cik"] = int(doc["cik"])
    keep = ["cik", "accession_number", "form_type", "filing_date", "report_date", "acceptance_raw",
            "acceptance_utc", "items", "primary_document", "primary_doc_description", "size",
            "is_xbrl", "is_inline_xbrl"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    return df[keep].sort_values("acceptance_utc").reset_index(drop=True)


def parse_index_header_acceptance(html: bytes) -> str | None:
    """Extract ACCEPTANCE-DATETIME (Eastern local time, YYYYMMDDHHMMSS) from an index-headers page."""
    text = html.decode("latin-1", errors="replace")
    marker = "ACCEPTANCE-DATETIME&gt;"
    i = text.find(marker)
    if i < 0:
        marker = "<ACCEPTANCE-DATETIME>"
        i = text.find(marker)
    if i < 0:
        return None
    val = text[i + len(marker): i + len(marker) + 14]
    return val if val.isdigit() else None


ACCEPTED_RE = re.compile(rb'infoHead">Accepted</div>\s*<div class="info">(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})</div>')


def parse_index_page_accepted(html: bytes) -> str | None:
    """'Accepted' field of an EDGAR filing index page (Eastern local time, 'YYYY-MM-DD HH:MM:SS')."""
    m = ACCEPTED_RE.search(html)
    return m.group(1).decode() if m else None


def index_page(client: "SecClient", cik: int, accession: str) -> Retrieval:
    name = f"{accession}-index.htm"
    return client.fetch(client.document_url(cik, accession, name),
                        client.document_path(cik, accession, name), max_age_days=None)
