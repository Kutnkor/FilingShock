"""FilingShock command-line interface. The interface and the CLI call the same services.

Exit codes: 0 success, 1 failure, 2 setup/configuration problem, 3 required data unavailable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys

from filingshock.config import PROJECT_ROOT, Config, ConfigurationError
from filingshock.storage import DataLayout, read_json

log = logging.getLogger("filingshock")
EXIT_OK, EXIT_FAIL, EXIT_SETUP, EXIT_DATA = 0, 1, 2, 3


def _ctx(args) -> tuple[Config, DataLayout]:
    cfg = Config.load(args.config)
    return cfg, DataLayout(cfg.data_dir).ensure()


def cmd_validate_setup(args) -> int:
    problems, warnings = [], []
    if sys.version_info < (3, 11):
        problems.append(f"Python >= 3.11 required, found {sys.version.split()[0]}")
    for mod in ["pandas", "numpy", "pyarrow", "sklearn", "streamlit", "altair", "lxml", "requests",
                "exchange_calendars", "yaml"]:
        try:
            importlib.import_module(mod)
        except ImportError:
            problems.append(f"missing package: {mod}")
    try:
        cfg, layout = _ctx(args)
    except Exception as exc:
        problems.append(f"configuration could not be loaded: {exc}")
        cfg = layout = None
    if cfg is not None:
        if cfg.user_agent_is_placeholder:
            warnings.append("SEC User-Agent is the UNCONFIGURED placeholder. Set FILINGSHOCK_SEC_USER_AGENT "
                            "to 'Your Name your.email@domain' before downloading from SEC.")
        if len(cfg.companies()) == 0:
            problems.append("universe has no companies")
        from filingshock.timing import SessionCalendar
        try:
            SessionCalendar(cfg["timing"]["exchange_calendar"])
        except Exception as exc:
            problems.append(f"exchange calendar unavailable: {exc}")
        state = {
            "filings": (layout.interim / "filings.parquet").exists(),
            "events": (layout.processed / "events.parquet").exists(),
            "primary_run": (layout.experiments / "primary_run.json").exists(),
            "sample_package": (PROJECT_ROOT / "sample" / "processed" / "events.parquet").exists(),
        }
        print("data state:", json.dumps(state))
    if args.network and cfg is not None:
        import requests
        try:
            r = requests.get("https://data.sec.gov/submissions/CIK0000796343.json",
                             headers={"User-Agent": cfg.sec_user_agent}, timeout=15)
            print("SEC reachability:", r.status_code)
        except requests.RequestException as exc:
            warnings.append(f"SEC not reachable: {exc}")
    for w in warnings:
        print("WARNING:", w)
    for p in problems:
        print("PROBLEM:", p)
    print("setup ok" if not problems else "setup has problems")
    return EXIT_SETUP if problems else EXIT_OK


def _require_downloads_allowed(cfg: Config, args) -> int | None:
    if getattr(args, "offline", False):
        return None
    if cfg.user_agent_is_placeholder and not args.allow_placeholder_agent:
        print("Refusing to contact SEC with the placeholder User-Agent. Set FILINGSHOCK_SEC_USER_AGENT "
              "or pass --allow-placeholder-agent for a small, throttled pilot.")
        return EXIT_SETUP
    return None


def cmd_discover(args) -> int:
    cfg, layout = _ctx(args)
    if (rc := _require_downloads_allowed(cfg, args)) is not None:
        return rc
    from filingshock.pipeline import step_discover
    df = step_discover(cfg, layout, pilot=args.pilot, offline=args.offline, refresh=args.refresh)
    print(f"{len(df)} filings discovered; periodic 10-Qs in window: "
          f"{int(((df.form_type == '10-Q') & df.in_window).sum())}")
    return EXIT_OK


def cmd_prices(args) -> int:
    cfg, layout = _ctx(args)
    if args.provider:
        cfg = cfg.with_overrides({"market_data.provider": args.provider})
    if args.history_end and not args.refresh:
        print("Note: --history-end applies only to symbols fetched in this run; add --refresh to extend cached files.")
    from filingshock.pipeline import step_prices
    res = step_prices(cfg, layout, refresh=args.refresh, history_end=args.history_end)
    for k, v in res.items():
        print(f"{k:6s} {v}")
    return EXIT_FAIL if any(v.startswith("refresh_failed") for v in res.values()) else EXIT_OK


def cmd_resources(args) -> int:
    cfg, layout = _ctx(args)
    if (rc := _require_downloads_allowed(cfg, args)) is not None:
        return rc
    from filingshock.pipeline import step_resources
    step_resources(cfg, layout, offline=args.offline)
    print("company facts and dictionary resources ready")
    return EXIT_OK


def cmd_build(args) -> int:
    cfg, layout = _ctx(args)
    if not (layout.interim / "filings.parquet").exists():
        print("No filings table. Run `filingshock discover` first.")
        return EXIT_DATA
    if (rc := _require_downloads_allowed(cfg, args)) is not None:
        return rc
    from filingshock.pipeline import step_build
    res = step_build(cfg, layout, offline=args.offline, force_extract=args.force_extract)
    for s in res["funnel"]["stages"]:
        print(f"{s['stage']:22s} remaining={s['remaining']:5d} excluded={s['excluded']:4d} {s['reasons']}")
    print("dataset:", res["manifest"]["dataset_id"])
    return EXIT_OK


def cmd_experiment(args) -> int:
    cfg, layout = _ctx(args)
    if not (layout.processed / "events.parquet").exists():
        print("No event dataset. Run `filingshock build` first.")
        return EXIT_DATA
    from filingshock.events import recompute_outcomes
    from filingshock.modeling.experiment import run_experiment, set_primary_run
    override = None
    if args.multiplier is not None or args.lookback is not None:
        override = recompute_outcomes(cfg, layout, args.lookback or int(cfg["target"]["lookback_sessions"]),
                                      args.multiplier or float(cfg["target"]["multiplier"]))
    d = run_experiment(cfg, layout, run_name=args.name, subset=args.subset, outcome_override=override,
                       include_secondary=not args.no_secondary, reps=args.reps, note=args.note)
    if args.set_primary:
        set_primary_run(layout, d.name)
    m = read_json(d / "metrics.json")
    print("run:", d.name)
    print("primary:", json.dumps(m["primary"]))
    return EXIT_OK


def cmd_sensitivity(args) -> int:
    cfg, layout = _ctx(args)
    from filingshock.events import recompute_outcomes
    from filingshock.modeling.experiment import run_experiment
    lb = int(cfg["target"]["lookback_sessions"])
    runs = [
        ("sens-multiplier-1.5", dict(outcome_override=recompute_outcomes(cfg, layout, lb, 1.5))),
        ("sens-multiplier-2.5", dict(outcome_override=recompute_outcomes(cfg, layout, lb, 2.5))),
        ("sub-no-earnings-8k", dict(subset="no_earnings_8k_in_window")),
        ("sub-earnings-8k", dict(subset="earnings_8k_in_window")),
    ]
    for name, kw in runs:
        d = run_experiment(cfg, layout, run_name=name, include_secondary=False, reps=args.reps,
                           note="pre-specified sensitivity analysis (protocol section 9)", **kw)
        m = read_json(d / "metrics.json")
        print(f"{name:24s} n={m['sample']['n_events']} prev={m['sample']['prevalence']:.3f} "
              f"dBrier={m['primary']['estimate']:+.4f} ci={m['primary']['ci95']} -> {m['primary']['conclusion']}")
    return EXIT_OK


def cmd_verify(args) -> int:
    cfg, layout = _ctx(args)
    from filingshock.audit import reconcile_run
    run_id = args.run or read_json(layout.experiments / "primary_run.json")["run_id"]
    res = reconcile_run(layout.experiments / run_id)
    print(json.dumps(res, indent=2, default=str))
    return EXIT_OK if res["all_ok"] else EXIT_FAIL


def cmd_verify_ledgers(args) -> int:
    _, layout = _ctx(args)
    from filingshock.ledger_audit import verify_ledgers

    report = verify_ledgers(layout)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, state in report["logs"].items():
            print(f"{name}: {state['status']} ({state['n_records']} records)")
        for error in report["errors"]:
            print(f"ERROR {error['log']}: {error['reason']}")
        print("ledger verification passed" if report["all_ok"] else "ledger verification failed")
    return EXIT_OK if report["all_ok"] else EXIT_FAIL


def cmd_audit(args) -> int:
    cfg, layout = _ctx(args)
    from filingshock import audit
    if args.event:
        print(json.dumps(audit.explain_event(layout, args.event), indent=2, default=str))
        return EXIT_OK
    if args.timestamps:
        from filingshock.discovery import load_filings
        from filingshock.sources.sec import SecClient
        if (rc := _require_downloads_allowed(cfg, args)) is not None:
            return rc
        f = load_filings(layout)
        sample = f[(f.form_type == "10-Q") & f.in_window].groupby("ticker").sample(
            n=1, random_state=11).head(args.timestamps)
        df = audit.reconcile_timestamps(SecClient(cfg, layout), layout, sample)
        print(df.to_string())
        return EXIT_OK if (df["status"] == "match").all() else EXIT_FAIL
    funnel = read_json(layout.processed / "funnel.json")
    for s in funnel["stages"]:
        print(f"{s['stage']:22s} remaining={s['remaining']:5d} excluded={s['excluded']:4d} {s['reasons']}")
    return EXIT_OK


def cmd_forecast_pending(args) -> int:
    cfg, layout = _ctx(args)
    from filingshock.forecasts import ForecastLedger, record_pending_forecasts
    from filingshock.sources.sec import SecClient
    ok, msg = ForecastLedger(layout).verify_chain()
    print("ledger integrity:", msg)
    if not ok:
        return EXIT_FAIL
    if not (layout.processed / "events.parquet").exists():
        print("No event dataset to fit the production models on. Run `filingshock build` first.")
        return EXIT_DATA
    if (rc := _require_downloads_allowed(cfg, args)) is not None:
        return rc
    results = record_pending_forecasts(cfg, layout, SecClient(cfg, layout, offline=args.offline))
    if not results:
        print("No 10-Q in the universe currently awaits its decision cutoff. Nothing recorded. "
              "(Run `filingshock discover --refresh` first to refresh the filings table.)")
        return EXIT_OK
    for r in results:
        rec = r["record"] or {}
        detail = (f"P(C)={rec['p_C']:.3f} P(B)={rec['p_B']:.3f} threshold={rec['threshold_abs_move']:.4f} "
                  f"{rec['model_version']} generated {rec['generated_at_utc']}") if rec else r["reason"]
        print(f"{r['ticker']:5s} {r['accession_number']} cutoff {r['decision_cutoff_utc']} {r['status']:16s} {detail}")
    print(f"{sum(r['status'] == 'recorded' for r in results)} forecast(s) recorded in "
          f"{ForecastLedger(layout).path.relative_to(layout.root)}")
    return EXIT_OK


def cmd_score_prospective(args) -> int:
    cfg, layout = _ctx(args)
    from filingshock.forecasts import LedgerError
    from filingshock.prospective_outcomes import (
        OutcomeLog,
        outcomes_frame,
        prospective_summary,
        score_prospective_forecasts,
    )
    try:
        results = score_prospective_forecasts(cfg, layout)
    except LedgerError as exc:
        print(exc)
        return EXIT_FAIL
    if not results:
        print("The prospective ledger has no records. Nothing to score.")
        return EXIT_OK
    for r in results:
        rec = r["record"] or {}
        if rec.get("outcome_status") == "ok":
            detail = (f"|r|={rec['abs_r']:.4f} threshold={rec['threshold_abs_move']:.4f} y={rec['y']} "
                      f"Brier B={rec['brier_B']:.4f} C={rec['brier_C']:.4f}")
        elif rec:
            detail = f"{rec['outcome_status']} (recorded, y not scored)"
        else:
            detail = r["reason"]
        print(f"{r['ticker']:5s} {r['forecast_id']} target {r['target_session']} {r['status']:14s} {detail}")
    n_new = sum(r["status"] == "scored" for r in results)
    print(f"{n_new} outcome(s) appended to {OutcomeLog(layout).path.relative_to(layout.root)} "
          "(prospective; never part of the research metrics)")
    s = prospective_summary(outcomes_frame(layout))
    if s["pooled"]:
        p = s["pooled"]
        print(f"pooled prospective: {p['n_events']} events, {p['n_positive']} large moves, Brier B={p['brier_B']:.4f} "
              f"C={p['brier_C']:.4f} (C-B {p['delta_brier_C_minus_B']:+.4f})")
    else:
        print(f"pooled prospective Brier withheld: {s['n_scored_events']} scored event(s), "
              f"needs {s['min_events_for_pooled']}")
    return EXIT_OK


def cmd_demo_prepare(args) -> int:
    cfg, layout = _ctx(args)
    from filingshock.demo import build_sample_package
    out = build_sample_package(cfg, layout, PROJECT_ROOT / "sample", n_events=args.n_events)
    print(json.dumps(out, indent=2))
    return EXIT_OK


def cmd_report(args) -> int:
    cfg, layout = _ctx(args)
    from filingshock import reporting
    docs = PROJECT_ROOT / "docs"
    reporting.feature_catalog_md(docs / "feature_catalog.md")
    res = reporting.results_md(layout, docs / "results.md", superseded_run=args.superseded)
    figs = reporting.export_figures(layout, docs / "figures") if not args.no_figures else []
    print("wrote docs/feature_catalog.md, docs/results.md", ("and docs/figures/" + ", ".join(figs)) if figs else "")
    print("primary:", res["primary"]["conclusion"])
    return EXIT_OK


def cmd_app(args) -> int:
    cmd = [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "src" / "filingshock" / "ui" / "app.py"),
           "--server.port", str(args.port), "--server.headless", "true"]
    env_note = "sample package" if args.sample else "local research data"
    print(f"Starting FilingShock ({env_note}) on http://localhost:{args.port}")
    import os
    env = dict(os.environ)
    if args.sample:
        env["FILINGSHOCK_MODE"] = "sample"
    return subprocess.call(cmd, env=env, cwd=PROJECT_ROOT)


def cmd_reproduce(args) -> int:
    steps = [["discover"], ["prices"], ["resources"], ["build"], ["experiment", "--name", "primary", "--set-primary"],
             ["verify-artifacts"]]
    for s in steps:
        extra = ["--allow-placeholder-agent"] if args.allow_placeholder_agent and s[0] in ("discover", "resources", "build") else []
        print(">>> filingshock", " ".join(s + extra))
        rc = main(s + extra + (["--config", args.config] if args.config else []))
        if rc != EXIT_OK:
            return rc
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="filingshock", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=None, help="config YAML (default: config/default.yaml)")
        sp.add_argument("-v", "--verbose", action="store_true")
        sp.set_defaults(fn=fn)
        return sp

    sp = add("validate-setup", cmd_validate_setup, "check runtime, configuration and data state")
    sp.add_argument("--network", action="store_true", help="also check SEC reachability")
    for name, fn, help_ in [("discover", cmd_discover, "discover filings from SEC submissions"),
                            ("resources", cmd_resources, "download Company Facts and the LM dictionary"),
                            ("build", cmd_build, "download documents, extract sections, build events and features")]:
        sp = add(name, fn, help_)
        sp.add_argument("--offline", action="store_true", help="use cached files only")
        sp.add_argument("--allow-placeholder-agent", action="store_true")
        if name == "discover":
            sp.add_argument("--pilot", action="store_true", help="pilot companies only")
            sp.add_argument("--refresh", action="store_true",
                            help="re-download submissions even if cached (needed before forecast-pending)")
        if name == "build":
            sp.add_argument("--force-extract", action="store_true")
    sp = add("prices", cmd_prices, "ingest daily bars (yfinance) or import CSV files")
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument("--provider", choices=["yfinance", "csv"])
    sp.add_argument("--history-end", metavar="DATE|today",
                    help="exclusive end date for this run, overriding market_data.history_end (the research default "
                         "is unchanged); 'today' fetches every settled session up to now, for prospective use")
    sp = add("experiment", cmd_experiment, "train and evaluate models A/B/C on temporal folds")
    sp.add_argument("--name", default="adhoc")
    sp.add_argument("--subset", choices=["no_earnings_8k_in_window", "earnings_8k_in_window"])
    sp.add_argument("--multiplier", type=float)
    sp.add_argument("--lookback", type=int)
    sp.add_argument("--no-secondary", action="store_true")
    sp.add_argument("--reps", type=int)
    sp.add_argument("--note")
    sp.add_argument("--set-primary", action="store_true")
    sp = add("sensitivity", cmd_sensitivity, "run the pre-specified sensitivity analyses")
    sp.add_argument("--reps", type=int)
    sp = add("verify-artifacts", cmd_verify, "recompute metrics from saved predictions and compare")
    sp.add_argument("--run")
    sp = add("verify-ledgers", cmd_verify_ledgers, "verify prospective log integrity and outcome-to-forecast links")
    sp.add_argument("--json", action="store_true", help="emit a machine-readable integrity report")
    sp = add("audit", cmd_audit, "funnel summary, per-event admissibility, timestamp reconciliation")
    sp.add_argument("--event")
    sp.add_argument("--timestamps", type=int, help="reconcile N acceptance timestamps against index headers")
    sp.add_argument("--allow-placeholder-agent", action="store_true")
    sp = add("forecast-pending", cmd_forecast_pending,
             "score 10-Qs awaiting their decision cutoff and record prospective forecasts in the ledger")
    sp.add_argument("--offline", action="store_true", help="use cached documents and Company Facts only")
    sp.add_argument("--allow-placeholder-agent", action="store_true")
    add("score-prospective", cmd_score_prospective,
        "score recorded prospective forecasts whose target session has closed (separate outcome log)")
    sp = add("demo-prepare", cmd_demo_prepare, "build the redistributable offline sample package")
    sp.add_argument("--n-events", type=int, default=12)
    sp = add("report", cmd_report, "regenerate docs/results.md, docs/feature_catalog.md and docs/figures from artifacts")
    sp.add_argument("--superseded", help="run id of a superseded primary run to mention")
    sp.add_argument("--no-figures", action="store_true")
    sp = add("app", cmd_app, "start the Streamlit interface")
    sp.add_argument("--port", type=int, default=8501)
    sp.add_argument("--sample", action="store_true", help="use the bundled sample package")
    sp = add("reproduce", cmd_reproduce, "run the full pipeline and the primary experiment")
    sp.add_argument("--allow-placeholder-agent", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        return args.fn(args)
    except ConfigurationError as exc:
        print(f"Invalid configuration: {exc}")
        return EXIT_SETUP
    except FileNotFoundError as exc:
        print(f"Required data not found: {exc}")
        return EXIT_DATA
    except KeyboardInterrupt:
        print("Interrupted; completed steps are cached and the command can be re-run.")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
