# Engineering review — 2026-10-17

These changes concern operational reliability. They do not rerun, replace, or strengthen the published
historical result. Each item below links its rationale to primary documentation and regression tests.

## Ledger parsing and ordering

JSONL logs must contain complete UTF-8 record objects. A valid checksum does not establish that sequence
numbers are contiguous; the reader checks both. A partial final write must block further writes rather
than be joined to the next record. Corrupt data must produce an integrity diagnostic, not an uncaught
JSON parser traceback. Tests: `tests/test_forecasts.py`.

The UI withholds records from a corrupt forecast or outcome log and displays its integrity failure.
It never renders unverified probabilities or realized scores. Both adapters read one verified snapshot.

Reference: [Python JSON decoder and JSONDecodeError](https://docs.python.org/3.11/library/json.html).

## Concurrency and persistence design

The existing read/check/append operation is not one transaction. Concurrent writers can validate the
same tail and produce conflicting sequence numbers. The intended fix serializes that whole operation
with a stable lock file and atomically replaces the ledger with its unchanged prefix plus one record.
Readers then see a complete old or new snapshot. A local hash chain can detect accidental edits; it
cannot prove publication time or prevent someone with filesystem access from rewriting the whole chain.

Implemented: a bounded, process-level sidecar lock covers integrity checks, duplicate checks, sequence
allocation and persistence. Outcome uniqueness is checked inside the same transaction. Readers need no
write permission or lock files because replacement exposes complete snapshots. Existing record bytes
are preserved. The ledger APIs are intended for a local filesystem with cooperating writers, not as a
distributed database or externally anchored audit service. Never remove `.lock` files while writers run.
Tests exercise independent ledger instances, simultaneous appends and duplicate forecasts/outcomes.

References: [Python file locks](https://docs.python.org/3.11/library/fcntl.html#fcntl.flock),
[Windows byte-range locks](https://docs.python.org/3.11/library/msvcrt.html#msvcrt.locking), and
[atomic replacement](https://docs.python.org/3.11/library/os.html#os.replace).

## Forecast identity

The production model name contains the year and dataset, but tuning or model-code changes can produce
different parameters with that same name. Forecast deduplication now includes the full parameter hash;
a repeated identical model is idempotent, while a changed model gets its own immutable forecast record.
The original record remains available to the earliest-forecast-per-event evaluation rule.

Reference: [scikit-learn model provenance and reproducibility](https://scikit-learn.org/stable/model_persistence.html).
