# Engineering review — 2026-10-17

These changes concern operational reliability. They do not rerun, replace, or strengthen the published
historical result. Each item below links its rationale to primary documentation and regression tests.

## Ledger parsing and ordering

JSONL logs must contain complete UTF-8 record objects. A valid checksum does not establish that sequence
numbers are contiguous; the reader checks both. A partial final write must block further writes rather
than be joined to the next record. Corrupt data must produce an integrity diagnostic, not an uncaught
JSON parser traceback. Tests: `tests/test_forecasts.py`.

Reference: [Python JSON decoder and JSONDecodeError](https://docs.python.org/3.11/library/json.html).

## Concurrency and persistence design

The existing read/check/append operation is not one transaction. Concurrent writers can validate the
same tail and produce conflicting sequence numbers. The intended fix serializes that whole operation
with a stable lock file and atomically replaces the ledger with its unchanged prefix plus one record.
Readers then see a complete old or new snapshot. A local hash chain can detect accidental edits; it
cannot prove publication time or prevent someone with filesystem access from rewriting the whole chain.

References: [Python file locks](https://docs.python.org/3.11/library/fcntl.html#fcntl.flock),
[Windows byte-range locks](https://docs.python.org/3.11/library/msvcrt.html#msvcrt.locking), and
[atomic replacement](https://docs.python.org/3.11/library/os.html#os.replace).
