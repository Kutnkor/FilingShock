"""Local storage helpers: data layout, atomic writes, hashing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataLayout:
    root: Path

    @property
    def raw_sec(self) -> Path:
        return self.root / "raw" / "sec"

    @property
    def raw_market(self) -> Path:
        return self.root / "raw" / "market"

    @property
    def raw_resources(self) -> Path:
        return self.root / "raw" / "resources"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def sections_dir(self) -> Path:
        return self.interim / "sections"

    @property
    def alignments_dir(self) -> Path:
        return self.interim / "alignments"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def forecasts(self) -> Path:
        return self.root / "forecasts"

    @property
    def state(self) -> Path:
        return self.root / "state"

    def ensure(self) -> "DataLayout":
        for p in [self.raw_sec, self.raw_market, self.raw_resources, self.interim, self.sections_dir,
                  self.alignments_dir, self.processed, self.experiments, self.forecasts, self.state]:
            p.mkdir(parents=True, exist_ok=True)
        return self


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_replace(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        writer(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_bytes_atomic(path: Path, data: bytes) -> None:
    _atomic_replace(path, lambda t: t.write_bytes(data))


def write_json_atomic(path: Path, obj: Any) -> None:
    blob = json.dumps(obj, indent=2, sort_keys=True, default=str).encode()
    write_bytes_atomic(path, blob)


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    _atomic_replace(path, lambda t: df.to_parquet(t, index=False))


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
