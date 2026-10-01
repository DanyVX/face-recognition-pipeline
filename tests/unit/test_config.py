from __future__ import annotations

from pathlib import Path

import pytest

from frp.config import ConfigError, Settings, load_settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run in an empty directory so a developer's real .env never leaks into tests."""
    monkeypatch.chdir(tmp_path)
    for name in ("LOG_LEVEL", "STORE_RAW_IMAGES", "DATA_DIR", "MODEL_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_privacy_preserving() -> None:
    settings = load_settings()
    assert settings.store_raw_images is False
    assert settings.log_level == "INFO"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORE_RAW_IMAGES", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = load_settings()
    assert settings.store_raw_images is True
    assert settings.log_level == "DEBUG"


def test_invalid_value_fails_fast_with_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        load_settings()


def test_settings_are_immutable() -> None:
    settings = Settings()
    with pytest.raises(ValueError, match="frozen"):
        settings.store_raw_images = True  # type: ignore[misc]


def test_dotenv_file_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LOG_LEVEL=ERROR\n", encoding="utf-8")
    assert load_settings().log_level == "ERROR"
