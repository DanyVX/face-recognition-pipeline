"""Cross-platform task runner. The Makefile delegates here so Windows works without make.

Usage: ``uv run python scripts/dev.py <task>`` (or ``make <task>``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(*cmd: str) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode  # noqa: S603


def _uv(*args: str) -> int:
    return _run("uv", *args)


def _first_failure(*codes: Callable[[], int]) -> int:
    for step in codes:
        code = step()
        if code != 0:
            return code
    return 0


def setup() -> int:
    return _first_failure(
        lambda: _uv("sync"),
        lambda: _uv("run", "pre-commit", "install"),
    )


def lint() -> int:
    return _first_failure(
        lambda: _uv("run", "ruff", "check", "."),
        lambda: _uv("run", "ruff", "format", "--check", "."),
    )


def format_() -> int:
    return _first_failure(
        lambda: _uv("run", "ruff", "check", "--fix", "."),
        lambda: _uv("run", "ruff", "format", "."),
    )


def typecheck() -> int:
    return _uv("run", "mypy")


def test() -> int:
    return _uv("run", "pytest", "--cov", "--cov-report=term-missing")


def test_fast() -> int:
    return _uv("run", "pytest", "-x", "-q")


def _bench_module(module: str, *args: str) -> Callable[[], int]:
    return lambda: _uv("run", "python", "-m", module, *args)


def _first_failure_steps(steps: list[Callable[[], int]]) -> int:
    return _first_failure(*steps)


def bench() -> int:
    """Full benchmark suite (needs `data/lfw` and the models; takes hours on a CPU laptop).

    Download first: ``uv run python scripts/download_models.py`` and
    ``uv run python scripts/download_lfw.py``. Every step is resumable.
    """
    modules = (
        "bench.prepare_lfw",
        "bench.embed_lfw",
        "bench.lfw_verification",
        "bench.open_set",
        "bench.index_scaling",
        "bench.latency",
        "bench.precision",
        "bench.robustness",
        "bench.calibrate_quality",
        "bench.report",
    )
    return _first_failure_steps([_bench_module(m) for m in modules])


def bench_smoke() -> int:
    """Tiny benchmark path that finishes in minutes and never overwrites real results.

    Always runs the synthetic index benchmark; when LFW and the models are present it also runs a
    reduced LFW verification (the first 15 pairs per class per fold).
    """
    steps: list[Callable[[], int]] = [
        _bench_module("bench.index_scaling", "--tiny", "--name", "smoke_index_scaling")
    ]
    if (ROOT / "data" / "lfw" / "lfw").is_dir() and (ROOT / "models" / "det_10g.onnx").is_file():
        steps += [
            _bench_module("bench.prepare_lfw", "--limit", "15"),
            _bench_module("bench.embed_lfw", "--limit", "15"),
            _bench_module(
                "bench.lfw_verification", "--limit", "15", "--no-extended", "--name", "smoke_lfw"
            ),
        ]
    return _first_failure_steps(steps)


def run() -> int:
    return _uv("run", "frp", "--help")


def docker_build() -> int:
    if not (ROOT / "Dockerfile").exists():
        print("No Dockerfile yet: it arrives in M7/M8.", file=sys.stderr)
        return 2
    return _run("docker", "build", "-t", "frp", ".")


def clean() -> int:
    caches = (
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        "htmlcov",
        "build",
        "dist",
    )
    for name in caches:
        shutil.rmtree(ROOT / name, ignore_errors=True)
    for path in ROOT.rglob("__pycache__"):
        if ".venv" not in path.parts:
            shutil.rmtree(path, ignore_errors=True)
    (ROOT / ".coverage").unlink(missing_ok=True)
    return 0


TASKS: dict[str, Callable[[], int]] = {
    "setup": setup,
    "lint": lint,
    "format": format_,
    "typecheck": typecheck,
    "test": test,
    "test-fast": test_fast,
    "bench": bench,
    "bench-smoke": bench_smoke,
    "run": run,
    "docker-build": docker_build,
    "clean": clean,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(TASKS))
    return TASKS[parser.parse_args().task]()


if __name__ == "__main__":
    sys.exit(main())
