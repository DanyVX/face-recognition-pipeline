"""ONNX Runtime helpers shared by the detector and embedder backends."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import onnxruntime as ort

from frp.errors import ModelFileError

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_model_file(path: Path, expected_sha256: str | None) -> str:
    """Return the file's SHA-256, raising an actionable error if it is missing or wrong."""
    if not path.is_file():
        msg = (
            f"Model file not found: {path}. "
            "Run `uv run python scripts/download_models.py` to fetch it."
        )
        raise ModelFileError(msg)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        msg = (
            f"Model file {path} failed its integrity check "
            f"(expected sha256 {expected_sha256}, got {actual}). "
            "Delete it and re-run `uv run python scripts/download_models.py`."
        )
        raise ModelFileError(msg)
    return actual


def resolve_providers(requested: list[str] | None = None) -> list[str]:
    """Pick execution providers, falling back to CPU with a log line instead of crashing."""
    wanted = requested or [CUDA_PROVIDER, CPU_PROVIDER]
    available = set(ort.get_available_providers())
    chosen = [p for p in wanted if p in available]
    missing = [p for p in wanted if p not in available]
    if missing:
        logger.warning(
            "ONNX providers %s unavailable; using %s instead",
            missing,
            chosen or [CPU_PROVIDER],
        )
    return chosen or [CPU_PROVIDER]


def create_session(
    model_path: Path,
    *,
    providers: list[str] | None = None,
    intra_op_threads: int | None = None,
    expected_sha256: str | None = None,
) -> ort.InferenceSession:
    """Create an inference session after verifying the model file's integrity."""
    verify_model_file(model_path, expected_sha256)
    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only; ORT is chatty about harmless graph rewrites
    if intra_op_threads is not None:
        options.intra_op_num_threads = intra_op_threads
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=resolve_providers(providers)
    )


def session_providers(session: Any) -> list[str]:
    """Providers actually in use (for logging into benchmark results)."""
    getter = getattr(session, "get_providers", None)
    return list(getter()) if callable(getter) else []
