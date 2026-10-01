from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from frp import cli
from frp.align import align_face
from frp.debug import OVERLAY_HEIGHT, alignment_overlay
from frp.detect import Face
from frp.errors import NoFaceError
from tests.synthetic import synthetic_scene

runner = CliRunner()


class FakeDetector:
    def __init__(self, faces: list[Face]) -> None:
        self.faces = faces

    def detect(self, image: np.ndarray) -> list[Face]:
        return self.faces


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def synthetic_face() -> tuple[np.ndarray, Face]:
    scene, landmarks = synthetic_scene()
    box = np.array(
        [
            landmarks[:, 0].min() - 10,
            landmarks[:, 1].min() - 10,
            landmarks[:, 0].max() + 10,
            landmarks[:, 1].max() + 10,
        ],
        dtype=np.float32,
    )
    return scene, Face(box=box, landmarks=landmarks, score=0.99)


def test_overlay_has_fixed_height_and_is_deterministic() -> None:
    scene, face = synthetic_face()
    aligned = align_face(scene, face.landmarks)
    first = alignment_overlay(scene, [face], aligned)
    second = alignment_overlay(scene, [face], aligned)
    assert first.dtype == np.uint8 and first.shape[0] == OVERLAY_HEIGHT and first.shape[2] == 3
    np.testing.assert_array_equal(first, second)


def test_overlay_for_failed_alignment_shows_the_reason() -> None:
    scene, face = synthetic_face()
    overlay = alignment_overlay(scene, [face], None, "BAD_FIT: mirrored")
    assert overlay.shape[0] == OVERLAY_HEIGHT


def test_cli_writes_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene, face = synthetic_face()
    src = tmp_path / "face.png"
    src.write_bytes(png_bytes(scene))
    monkeypatch.setattr(cli, "build_detector", lambda: FakeDetector([face]))
    out = tmp_path / "overlay.png"
    result = runner.invoke(cli.app, ["debug-align", str(src), "--out", str(out)])
    assert result.exit_code == 0, result.output
    with Image.open(out) as img:
        assert img.size[1] == OVERLAY_HEIGHT


def test_cli_default_output_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene, face = synthetic_face()
    src = tmp_path / "me.png"
    src.write_bytes(png_bytes(scene))
    monkeypatch.setattr(cli, "build_detector", lambda: FakeDetector([face]))
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(cli.app, ["debug-align", str(src)]).exit_code == 0
    assert (tmp_path / "me_align.png").is_file()


def test_cli_reports_no_face(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene, _ = synthetic_face()
    src = tmp_path / "x.png"
    src.write_bytes(png_bytes(scene))
    monkeypatch.setattr(cli, "build_detector", lambda: FakeDetector([]))
    result = runner.invoke(cli.app, ["debug-align", str(src)])
    assert result.exit_code == 1
    assert NoFaceError.code in result.output


def test_cli_reports_unreadable_and_invalid_images(tmp_path: Path) -> None:
    missing = runner.invoke(cli.app, ["debug-align", str(tmp_path / "nope.png")])
    assert missing.exit_code == 1
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    result = runner.invoke(cli.app, ["debug-align", str(bad)])
    assert result.exit_code == 1
    assert "UNSUPPORTED_IMAGE_FORMAT" in result.output


def test_cli_still_writes_overlay_when_alignment_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene, face = synthetic_face()
    mirrored = Face(box=face.box, landmarks=face.landmarks[[1, 0, 2, 4, 3]], score=0.9)
    src = tmp_path / "m.png"
    src.write_bytes(png_bytes(scene))
    monkeypatch.setattr(cli, "build_detector", lambda: FakeDetector([mirrored]))
    out = tmp_path / "o.png"
    result = runner.invoke(cli.app, ["debug-align", str(src), "-o", str(out)])
    assert result.exit_code == 1
    assert "BAD_FIT" in result.output
    assert out.is_file()  # the diagnostic picture is still produced
