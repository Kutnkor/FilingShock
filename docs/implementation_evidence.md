# Implementation evidence

Links each major requirement to code, commands, tests or inspected output. Status: **met**, **partial**, **not met**.

| Requirement | Evidence | Status |
|---|---|---|
| Real filing ingestion with fair access | `sources/sec.py` (User-Agent, 4 req/s, cache, bounded retries, manifest); `data/raw/sec/retrieval_manifest.jsonl` | met (placeholder contact — set `FILINGSHOCK_SEC_USER_AGENT`) |
| Separate timestamp fields, UTC storage | `timing.py`, `docs/timing_policy.md`, events columns | met |
| Timestamp reconciliation with source documents | `discovery.reconcile_acceptance`, `processed/acceptance_reconciliation.json`; 1,068/2,834 API errors corrected | met |
| Holidays, early closes, DST | `tests/test_timing.py` (13 tests) | met |
| Availability proxy labelled as assumption | `availability_method`, UI "(assumed)", Replay text | met |
| Earnings release vs filing separation | `ctx_earnings_8k_in_window`, subgroup runs | met (8-K Item 2.02 only) |
| Protocol before final evaluation, logged changes | `docs/research_protocol.md` change log | met (not externally registered) |
| Provider interface + CSV import | `sources/market.py` (`YFinanceProvider`, `CsvImportProvider`) | met |
| Resumable, idempotent ingestion | cached fetches, section JSON keyed by parser version, `COMPLETE` markers | met |
| Event uniqueness, amendments | `events.build_candidates`; `tests/test_leakage.py::test_funnel_stages_on_fixture` | met |
| Point-in-time financials, restatements, YTD | `features/financial.py`; `tests/test_leakage.py` (restatement, YTD, missing) | met |
| Robust MD&A extraction with statuses | `parsing/sections.py`; `tests/test_parsing.py`; `docs/extraction_audit.md` | met |
| Manual extraction audit | 50-filing pilot + 14-filing stratified audit (`docs/extraction_audit.csv`) | met (single reviewer) |
| Reference selection explicit | `reference.py`; `reference_filing_id`, `reference_gap_days` | met |
| Explainable paragraph alignment | `parsing/align.py`; `tests/test_parsing.py` | met |
| Interpretable text features, versioned dictionaries | `features/text.py`, `resources/dictionaries/topics_v1.yaml`, LM point-in-time | met |
| Feature catalog | `features/catalog.py`, `docs/feature_catalog.md`, Methodology tab | met |
| Eligibility funnel with reasons | `processed/funnel.json`, Results page, `filingshock audit` | met |
| Fair A/B/C comparison, same events | `modeling/experiment.py`; 467 events per model in `metrics.json` | met |
| Temporal folds, separate selection/calibration | `modeling/splits.py`; `tests/test_modeling.py` | met |
| Train-only preprocessing | `modeling/models.Preprocessor`; test | met |
| Calibration isolated from testing | Platt on Y−1 from a model fitted ≤ Y−2; test | met |
| Metrics incl. AP (not PR-AUC), top-k, lift, undefined handling | `modeling/evaluate.py`; tests | met |
| Dependence-aware uncertainty | quarter- and company-cluster bootstrap | met |
| Ablations, sensitivity, falsification | `filingshock sensitivity`; label shuffle; mutation tests | met |
| Prediction categories, append-only prospective ledger | `ui/data.prediction_status`, `forecasts.py`, `tests/test_forecasts.py` | met |
| Prospective scoring path (single event, production model, ledger) | `events.single_event_features` (shares `pre_cutoff_market`, `prev_event_abs_ratio`, `document_stage_exclusion`, `event_document_features` with the build), `modeling/production.py` (fold-Y rule via `splits.make_fold` + `experiment.fit_calibrated_model`), `forecasts.record_pending_forecasts`, `discovery.carry_first_seen`, Overview rows + Results ledger panel; `tests/test_prospective.py` (recorded before cutoff, nothing at/after cutoff or before first seen, features unchanged when bars at/after the target session are mutated or removed, live = historical market features, stale prices block scoring, research tables unaffected) | met for the code path — no real prospective forecast recorded yet. Refactor verified on real data: rebuilt dataset identical (`ds-1698abec6cff`), experiment predictions identical to the primary run (max difference 0.0); a simulated prospective run on a real 2026 event reproduced its 30 features, threshold and fold-2026 P(B)/P(C) exactly |
| Prospective outcome scoring, apart from research metrics | `prospective_outcomes.score_prospective_forecasts` (`realized_outcome` on the target bar against the threshold stored in the forecast record; separate hash-chained `forecasts/prospective_outcomes.jsonl` via `forecasts.HashChainedLog`, one record per `forecast_id`), `filingshock score-prospective`, Results ledger panel (realized \|move\|, y, per-record Brier B/C; pooled Brier only from 20 scored events); live price refresh via `prices --history-end today` with `sources/market.drop_unsettled_sessions`; `tests/test_prospective.py` (stored threshold used even when a recomputed one would flip y, nothing scored before close + settle or from a price file retrieved earlier, missing target bar stays pending, re-running adds nothing, only the outcome file changes — ledger, research tables and metrics byte-identical, invalid bar recorded without y, pooled summary threshold, CLI, partial bars never stored), `tests/test_ui.py` (Results panel renders the outcome states) | met for the code path — no real prospective outcome yet (the ledger is empty). Results panel checked in a browser on the real research artifacts with synthetic ledger and outcome records (all five session states and the pooled branch) |
| Five-screen interface on real artifacts | `ui/pages/*`; `tests/test_ui.py` (research and sample modes) | met |
| UI totals reconcile with artifacts | `tests/test_ui.py::test_ui_totals_reconcile_with_artifacts`, `filingshock verify-artifacts` (50/50) | met |
| Offline demo without downloads/retraining | `sample/` via `filingshock demo-prepare`; `filingshock app --sample` | met |
| Reproducibility | determinism re-runs (max probability difference 0.0 on both datasets); run manifests | met |
| Visual verification | screenshots at 1440/768/390 px in `docs/screenshots/`; issues found and fixed (sidebar overlay at 390 px, label column width, chart legend/order) | met for inspected screens |
| Accessibility | text labels on all status colours, visible focus outline (spot-checked keyboard order on filters), `role="status"` messages, table views for charts | partial — no full WCAG audit, no screen-reader test |
| CI | `.github/workflows/ci.yml` (ruff + offline pytest) | written, not executed on a CI service |

## Commands run and outcomes (2026-10-17)

* `filingshock discover` + acceptance reconciliation: 7,052 filings, 2,834 index pages checked.
* `filingshock build`: dataset `ds-1698abec6cff`, 573 final events, 14 companies.
* `filingshock experiment --name primary --set-primary`: run `20261017T090148Z-primary`.
* `filingshock sensitivity`: four runs on the same dataset.
* `filingshock verify-artifacts`: all 50 checks pass.
* `pytest -q`: 59 passed. `ruff check src tests`: clean.
