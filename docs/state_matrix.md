# Interface state matrix

Machine-readable prediction states (`ui/data.py::prediction_status`):
`retrospective_out_of_time`, `no_prediction_training_period`, `no_prediction_fold_not_ready`,
`not_supported_excluded`, `prospective_recorded_before_session`, `synthetic_ui_example` (reserved; no
synthetic examples exist). "Not calculated", "not supported", "not available" and a genuine zero are
never collapsed: unavailable values render as "—" or an explicit sentence, never 0 % or 50 %.

| State | Where | What the user sees | Actions | Source of truth |
|---|---|---|---|---|
| No local dataset prepared | all pages | "No local dataset has been prepared" with the commands to build one or open the sample | none in-app (CLI) | `processed/events.parquet` absent |
| Offline sample package | all pages | Info banner: SEC-derived content only; prices and price-derived values excluded and why | explore events, comparisons, saved probabilities | `FILINGSHOCK_MODE=sample`, `sample/package_manifest.json` |
| Documents present, prices missing | Replay, Explorer | Threshold "not available in this package"; outcome panel explains prices are not included; market features "not included (price-derived)" | inspect comparison and probability | `outcomes.parquet` absent |
| Company has filings but no final events | Explorer | Rows shown with "excluded: <reason>"; comparison/replay buttons disabled | Explain eligibility | `events.exclusion_reason` |
| Filter returns nothing | Explorer, Comparison | "No events/paragraphs match" with how to widen | Reset filters | filtered frame empty |
| MD&A extraction failed | Explorer | Row "excluded: mdna_…"; stage-by-stage explanation | Explain eligibility | `extractions.mdna_status` |
| Reference filing missing | Explorer | "excluded: reference_missing:<reason>" | Explain eligibility | `events.reference_missing_reason` |
| Low alignment quality | Explorer, Comparison | Quality column shows "fair/weak (heuristic)"; per-pair "Low-confidence match" chip | filter by quality | `mdna_align_alignment_quality`, `pairs.low_confidence` |
| Availability assumed, not observed | Explorer, Replay | Availability column "(assumed)"; Replay states "an assumption, not an observed time" | — | `availability_method`, `observed_at_utc` empty |
| Features valid, no eligible model | Replay | "No out-of-time prediction exists" + reason (training period or fold not ready) | inspect features, comparison | prediction absent in primary run |
| Prediction exists, outcome not matured | Replay, Results | Replay: not reachable for historical events. Results ledger panel: Session "not closed yet" or "closed; awaiting score-prospective"; realized \|move\|, y and Brier show "—" | `filingshock score-prospective` (after `prices --refresh --history-end today`) | `target_close_utc` > now, or no record in `forecasts/prospective_outcomes.jsonl` |
| Outcome exists, no valid OOT prediction | Replay | Outcome can still be revealed; prediction panel explains why no score exists | Reveal | outcomes present, prediction absent |
| Prospective forecast recorded | Explorer, Results | Explorer row "prospective (outside research sample)" with chip and observed/generated/cutoff times, P(C)/P(B), threshold, model version; Results panel lists every ledger record and its integrity, apart from the research metrics | filter by eligibility or prediction state | `forecasts/prospective_forecasts.jsonl` (no real record yet) |
| Prospective outcome scored | Results | Ledger row shows realized \|move\|, y (vs the stored threshold) and per-record Brier B/C, Session "closed; outcome scored"; an invalid target bar shows "target bar invalid (recorded, not scored)". Below the table: scored-event count, and pooled prospective Brier B/C and C − B only from 20 scored events (earliest forecast per event); labelled prospective, never mixed with the research metrics | — | `forecasts/prospective_outcomes.jsonl`, `prospective_outcomes.prospective_summary` (no real record yet) |
| Prospective ledger tampered | Results | Error: integrity check failed; `forecast-pending` refuses to append | — | `ForecastLedger.verify_chain` |
| Prospective outcome log tampered | Results | Error: outcome log failed its integrity check; `score-prospective` refuses to append | — | `OutcomeLog.verify_chain` |
| Saved results from a different dataset | all pages | Warning: run computed on dataset X, current dataset Y; re-run command | — | `run_manifest.dataset.dataset_id` ≠ `dataset_manifest.dataset_id` |
| No completed experiment | Results | "No completed experiment is available"; no illustrative numbers | CLI command shown | `primary_run.json` / `COMPLETE` marker absent |
| Network failure / throttling | CLI | Bounded retries with exponential backoff, then a clear error and exit code 1; cached files are reused | re-run command | `SecClient._get` |
| Incomplete preparation job | CLI / app | Interrupted extraction resumes from cached documents and section files; an interrupted experiment leaves only a `.tmp-*` directory that the app ignores | re-run | `COMPLETE` marker, atomic renames |
| Risk section not comparable | Comparison | Explains each filing's risk status; "a missing or short risk section is not evidence of lower risk" | read current risk text | `risk_status` |
