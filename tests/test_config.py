"""Bad research settings fail before network calls or filesystem side effects."""

import pytest

from filingshock.config import Config, ConfigurationError


@pytest.mark.parametrize("key,value", [
    ("sec.max_requests_per_second", 0), ("sec.max_retries", -1), ("sec.timeout_seconds", float("nan")),
    ("target.lookback_sessions", 0), ("target.multiplier", float("inf")), ("target.lookback_sessions", True),
    ("target.max_window_sessions", 10), ("model.c_grid", []), ("model.c_grid", [0]),
    ("market_data.provider", "unknown"), ("universe.companies", []),
    ("splits.first_test_year", 3000), ("universe.filing_window.accepted_from", "not-a-date"),
    ("sec.user_agent", 123), ("timing.exchange_calendar", None),
])
def test_invalid_settings_are_rejected(key, value):
    with pytest.raises(ConfigurationError):
        Config.load().with_overrides({key: value})


@pytest.mark.parametrize("yaml", ["", "[]", "project: [broken", "project: false"])
def test_malformed_config_has_a_setup_exit_code(tmp_path, capsys, monkeypatch, yaml):
    from filingshock.cli import EXIT_SETUP, main

    path = tmp_path / "bad.yaml"
    path.write_text(yaml)
    data_dir = tmp_path / "must-not-exist"
    monkeypatch.setenv("FILINGSHOCK_DATA_DIR", str(data_dir))
    assert main(["prices", "--config", str(path)]) == EXIT_SETUP
    assert "Invalid configuration" in capsys.readouterr().out
    assert not data_dir.exists()


def test_missing_configuration_is_not_reported_as_missing_research_data(tmp_path, capsys):
    from filingshock.cli import EXIT_SETUP, main

    assert main(["prices", "--config", str(tmp_path / "missing.yaml")]) == EXIT_SETUP
    assert "cannot read YAML configuration" in capsys.readouterr().out


def test_empty_user_agent_is_treated_as_unconfigured(monkeypatch):
    monkeypatch.setenv("FILINGSHOCK_SEC_USER_AGENT", " ")
    assert Config.load().user_agent_is_placeholder


def test_app_passes_the_selected_configuration_to_streamlit(tmp_path, monkeypatch):
    import subprocess

    import yaml

    from filingshock.cli import main

    raw = Config.load().raw
    raw["project"]["data_dir"] = str(tmp_path / "custom-data")
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(raw))
    calls = []

    def launch(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return 0

    monkeypatch.setattr(subprocess, "call", launch)
    assert main(["app", "--config", str(path), "--sample"]) == 0
    assert calls[0][1]["env"]["FILINGSHOCK_CONFIG"] == str(path.resolve())
    assert calls[0][1]["env"]["FILINGSHOCK_MODE"] == "sample"
