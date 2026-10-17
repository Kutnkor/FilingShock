# Engineering review — 2026-10-17

These changes concern operational reliability. They do not rerun, replace, or strengthen the published
historical result. Each item below links its rationale to primary documentation and regression tests.

## Ledger parsing and ordering

JSONL logs must contain complete UTF-8 record objects. A valid checksum does not establish that sequence
numbers are contiguous; the reader checks both. A partial final write must block further writes rather
than be joined to the next record. Corrupt data must produce an integrity diagnostic, not an uncaught
JSON parser traceback. Tests: `tests/test_forecasts.py`.

Forecast semantics are checked on append and reuse: identifiers are nonempty, probabilities are finite
numbers in [0, 1], and observed/generated/cutoff timestamps identify real timezone-aware instants in
that order. `NaT` cannot silently bypass a comparison. Rehashing an invalid probability does not make
it eligible for display.

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

Production manifests are also checked against their actual canonical parameter hash when reused.
A damaged JSON file, mismatched hash or changed parameters blocks recording instead of silently
reusing or overwriting evidence. First creation is serialized with the same local lock primitive.

## Prospective baseline comparison

The saved primary result in `docs/results.md` shows Model A outperforming B/C in Brier score in this
sample, while its interval against the constant still spans zero. This motivates recording A and the
past-event-rate constant in future forecasts. It does not justify claiming a validated live advantage.
Both use the same matured data blocks as the historical protocol. Outcome summaries use the earliest
forecast per event and explicitly report the paired coverage of each optional baseline. Older records
remain missing; no retrospective baseline is attached to an earlier forecast.

Reference: [scikit-learn probability scoring and Brier loss](https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss).

## Failed market refreshes

A refresh failure previously replaced a good price manifest entry with an unavailable entry carrying
a new retrieval timestamp, even though the cached bars were old. Prospective scoring relies on that
timestamp, so freshness must describe an actual successful fetch. A failed refresh now leaves both
the good bars and their metadata unchanged and makes the command fail visibly. Initial coverage gaps
still get an explicit unavailable status. Regression tests are offline in `tests/test_market.py`.
