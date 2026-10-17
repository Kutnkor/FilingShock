# Contributing

* Keep business logic out of `src/filingshock/ui/pages/`; pages read saved artifacts through `ui/data.py`.
* Anything that changes timing, targets, features, splits or models must be reflected in
  `config/default.yaml` and logged in `docs/research_protocol.md` (change log, with whether results had been seen).
* Run `ruff check src tests` and `pytest -q` before proposing changes. Tests must not touch the network;
  mark any that do with `@pytest.mark.network` (deselected by default).
* Never commit `data/`: it contains SEC downloads and provider price data that may not be redistributed.
* Treat downloaded filings as untrusted data: never render them as HTML without escaping, never execute them.
* Use evidence-based wording in documentation; report negative and inconclusive results as such.
