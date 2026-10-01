from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from frp import cli
from frp.debug import OVERLAY_HEIGHT
from tests.conftest import FIXTURES

pytestmark = pytest.mark.needs_models


def test_debug_align_end_to_end_on_the_portrait(
    tmp_path: Path, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_DIR", str(models_dir))
    out = tmp_path / "overlay.png"
    result = CliRunner().invoke(
        cli.app, ["debug-align", str(FIXTURES / "astronaut.jpg"), "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    with Image.open(out) as overlay:
        assert overlay.size[1] == OVERLAY_HEIGHT
        assert overlay.size[0] > OVERLAY_HEIGHT  # detection panel plus the aligned crop panel
