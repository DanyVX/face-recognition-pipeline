"""Collect hardware, library and git information for embedding in result files.

Every benchmark result file records this (CLAUDE.md section 3). Values that cannot be
determined are recorded explicitly as ``None`` with a reason, never silently omitted.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

TRACKED_PACKAGES = (
    "numpy",
    "onnxruntime",
    "opencv-python-headless",
    "pillow",
    "faiss-cpu",
    "fastapi",
    "pydantic",
    "typer",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git(args: list[str], cwd: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603  # fixed argv, no shell, resolved git path
            [git, *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip()


def git_info(repo: Path | None = None) -> dict[str, Any]:
    """Return the current commit and dirty flag, or ``None`` values with a reason."""
    cwd = repo or Path.cwd()
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"commit": None, "dirty": None, "reason": "git unavailable or not a repository"}
    status = _git(["status", "--porcelain"], cwd)
    return {"commit": commit, "dirty": bool(status), "reason": None}


def collect_env(repo: Path | None = None) -> dict[str, Any]:
    """Snapshot of the environment, suitable for dumping into ``results/*.json``."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "packages": _package_versions(),
        "git": git_info(repo),
    }
