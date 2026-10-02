from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from frp import __version__
from frp.cli import app

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_env_prints_valid_json() -> None:
    result = runner.invoke(app, ["env"])
    assert result.exit_code == 0
    assert "python" in json.loads(result.stdout)


def test_check_config_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("ALLOW_UNENCRYPTED", "true")
    result = runner.invoke(app, ["check-config"])
    assert result.exit_code == 0


def test_check_config_fails_without_an_encryption_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ENCRYPTION_KEYS", "ENCRYPTION_KEY_FILE", "ALLOW_UNENCRYPTED"):
        monkeypatch.delenv(name, raising=False)
    result = runner.invoke(app, ["check-config"])
    assert result.exit_code == 2 and "No encryption key" in result.output


def test_check_config_invalid_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    result = runner.invoke(app, ["check-config"])
    assert result.exit_code == 2
