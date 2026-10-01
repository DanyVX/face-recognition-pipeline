from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from frp.imageio import load_image
from frp.models import MODELS

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
MODELS_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "models"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "needs_models: needs downloaded model weights (skipped when absent; "
        "fetch with scripts/download_models.py)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = [s.filename for s in MODELS.values() if not (MODELS_DIR / s.filename).is_file()]
    if not missing:
        return
    skip = pytest.mark.skip(
        reason=f"model files missing ({', '.join(missing)}); run scripts/download_models.py"
    )
    for item in items:
        if "needs_models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS_DIR


@pytest.fixture(scope="session")
def astronaut_bytes() -> bytes:
    """Public-domain NASA portrait (see DATA.md for provenance)."""
    return (FIXTURES / "astronaut.jpg").read_bytes()


@pytest.fixture(scope="session")
def astronaut_rgb(astronaut_bytes: bytes) -> NDArray[np.uint8]:
    return load_image(astronaut_bytes).array
