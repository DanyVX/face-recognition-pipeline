"""Shared helpers for benchmark scripts: result files with environment and commit hash."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from frp.envinfo import collect_env

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and paths into plain JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    return value


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> Path:
    """Write JSON via a temp file and rename, so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as tmp:
            json.dump(to_jsonable(dict(document)), tmp, indent=2)
            tmp.write("\n")
        Path(tmp_name).replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def write_result(
    name: str,
    result: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    directory: Path = RESULTS_DIR,
) -> Path:
    """Save ``results/<name>.json`` with environment info, git commit and parameters."""
    document = {
        "name": name,
        "env": collect_env(ROOT),
        "params": dict(params),
        "result": dict(result),
    }
    return write_json_atomic(directory / f"{name}.json", document)


def load_result(name: str, *, directory: Path = RESULTS_DIR) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    return loaded
