"""Application settings, read from environment variables (and an optional ``.env`` file).

Variable names match the project spec (for example ``STORE_RAW_IMAGES``), so there is no
prefix. Settings are validated at startup; invalid values fail fast with a clear message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when the environment holds invalid configuration."""


class Settings(BaseSettings):
    """All runtime configuration. Add a field here and to ``.env.example`` together."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", frozen=True, env_ignore_empty=True
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["text", "json"] = "text"
    # Raw uploaded images are biometric data: never persisted unless explicitly enabled.
    store_raw_images: bool = False
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")

    # Detection
    det_min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    min_face_px: int = Field(20, ge=1)
    max_faces: int = Field(10, ge=1)

    # Embedding: faces per inference call (bounds memory for large batches).
    embed_max_batch: int = Field(16, ge=1)

    # ONNX Runtime. Providers are tried in order; unavailable ones fall back to CPU with a log line.
    onnx_providers: str = "CUDAExecutionProvider,CPUExecutionProvider"
    onnx_threads: int | None = Field(None, ge=1)  # None = ORT default (all cores)

    # Gallery and matching. The thresholds are PROVISIONAL placeholders until the LFW results in
    # docs/RESULTS.md are used to choose operating points; set them deliberately for your use.
    db_filename: str = "frp.db"
    index_backend: Literal["flat", "hnsw", "ivf"] = "flat"
    aggregation: Literal["max", "mean"] = "max"
    verify_threshold: float = Field(0.4, ge=-1.0, le=1.0)
    identify_threshold: float = Field(0.4, ge=-1.0, le=1.0)
    duplicate_threshold: float = Field(0.6, ge=-1.0, le=1.0)
    duplicate_policy: Literal["ignore", "warn", "reject"] = "warn"
    max_embeddings_per_identity: int = Field(50, ge=1)
    enroll_face_policy: Literal["largest", "reject"] = "reject"
    query_face_policy: Literal["largest", "reject"] = "largest"
    quality_thresholds_file: Path | None = None  # None = built-in lenient defaults

    # HTTP API
    api_host: str = "127.0.0.1"
    api_port: int = Field(8000, ge=1, le=65535)
    max_upload_bytes: int = Field(10 * 1024 * 1024, ge=1024)
    max_images_per_enroll: int = Field(10, ge=1)
    max_image_side: int = Field(4096, ge=64)
    max_image_pixels: int = Field(128_000_000, ge=1024)
    request_timeout_seconds: float = Field(30.0, gt=0)
    upload_timeout_seconds: float = Field(15.0, gt=0)
    rate_limit_per_minute: int = Field(60, ge=1)
    rate_limit_burst: int = Field(20, ge=1)
    max_concurrent_inference: int = Field(2, ge=1)
    shutdown_grace_seconds: float = Field(20.0, ge=0)
    return_embeddings: bool = False  # when false, embeddings are never returned by the API
    audit_enabled: bool = True
    metrics_require_auth: bool = True
    idempotency_ttl_seconds: int = Field(86_400, ge=1)

    # Encryption at rest (AES-256-GCM). Provide ENCRYPTION_KEYS ("id:BASE64,id2:BASE64") or
    # ENCRYPTION_KEY_FILE. Embeddings and index snapshots are always encrypted when a key exists.
    encryption_keys: str | None = None
    encryption_active_key: str | None = None  # default: the first key listed
    encryption_key_file: Path | None = None
    # Demo/dev only: create the key file if missing. A key stored beside the data does not protect
    # against theft of the data volume.
    allow_key_generation: bool = False
    # Explicit opt-out for development and tests; the service refuses to start unencrypted
    # otherwise.
    allow_unencrypted: bool = False

    @property
    def provider_list(self) -> list[str]:
        return [p.strip() for p in self.onnx_providers.split(",") if p.strip()]

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename


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
