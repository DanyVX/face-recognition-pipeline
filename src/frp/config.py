"""Application settings, read from environment variables (and an optional ``.env`` file).

Variable names match the project spec (for example ``STORE_RAW_IMAGES``), so there is no
prefix. Settings are validated at startup; invalid values fail fast with a clear message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when the environment holds invalid configuration."""


class Settings(BaseSettings):
    """All runtime configuration. Add a field here and to ``.env.example`` together."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # Raw uploaded images are biometric data: never persisted unless explicitly enabled.
    store_raw_images: bool = False
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")


def load_settings() -> Settings:
    """Load settings, converting validation errors into an actionable ``ConfigError``."""
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']).upper()}: {err['msg']}" for err in exc.errors()
        )
        msg = f"Invalid configuration ({problems}). See .env.example for valid values."
        raise ConfigError(msg) from exc
