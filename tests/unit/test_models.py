"""Model download/verification and ONNX helpers, tested offline with file:// URLs."""

from __future__ import annotations

import hashlib
import logging
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from frp.errors import ModelFileError
from frp.models import MAX_DOWNLOAD_BYTES, MODELS, ModelSpec, ensure_model, sha256_of
from frp.onnx import resolve_providers, verify_model_file

PAYLOAD = b"pretend-onnx-bytes" * 100
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def make_zip(tmp_path: Path, member: str = "model.onnx", payload: bytes = PAYLOAD) -> Path:
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, payload)
    return archive


def spec_for(archive: Path, **overrides: object) -> ModelSpec:
    base = ModelSpec(
        key="fake",
        filename="model.onnx",
        sha256=PAYLOAD_SHA,
        url=archive.as_uri(),
        size_bytes=archive.stat().st_size,
        license="test",
        archive_member="model.onnx",
        download_sha256=sha256_of(archive),
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_downloads_extracts_and_cleans_up(tmp_path: Path) -> None:
    archive = make_zip(tmp_path)
    models_dir = tmp_path / "models"
    path = ensure_model(spec_for(archive), models_dir)
    assert path.read_bytes() == PAYLOAD
    assert [p.name for p in models_dir.iterdir()] == ["model.onnx"]  # staging dir removed


def test_already_present_model_is_not_downloaded_again(tmp_path: Path) -> None:
    archive = make_zip(tmp_path)
    models_dir = tmp_path / "models"
    spec = spec_for(archive)
    ensure_model(spec, models_dir)
    archive.unlink()  # a second download would now fail
    assert ensure_model(spec, models_dir).read_bytes() == PAYLOAD


def test_corrupt_existing_file_is_replaced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    archive = make_zip(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model.onnx").write_bytes(b"garbage")
    with caplog.at_level(logging.WARNING):
        path = ensure_model(spec_for(archive), models_dir)
    assert path.read_bytes() == PAYLOAD
    assert "fails its checksum" in caplog.text


def test_archive_checksum_mismatch_is_refused(tmp_path: Path) -> None:
    archive = make_zip(tmp_path)
    spec = spec_for(archive, download_sha256="0" * 64)
    with pytest.raises(ModelFileError, match="archive"):
        ensure_model(spec, tmp_path / "models")
    assert not (tmp_path / "models" / "model.onnx").exists()


def test_extracted_file_checksum_mismatch_removes_the_file(tmp_path: Path) -> None:
    archive = make_zip(tmp_path)
    spec = spec_for(archive, sha256="f" * 64)
    with pytest.raises(ModelFileError, match="checksum"):
        ensure_model(spec, tmp_path / "models")
    assert not (tmp_path / "models" / "model.onnx").exists()


def test_downloads_above_the_size_limit_are_refused_before_starting(tmp_path: Path) -> None:
    archive = make_zip(tmp_path)
    spec = spec_for(archive, size_bytes=MAX_DOWNLOAD_BYTES + 1)
    with pytest.raises(ModelFileError, match="500 MB"):
        ensure_model(spec, tmp_path / "models")


def test_archive_members_with_paths_are_rejected(tmp_path: Path) -> None:
    archive = make_zip(tmp_path, member="sub/model.onnx")
    spec = spec_for(archive, archive_member="sub/model.onnx")
    with pytest.raises(ModelFileError, match="Unexpected archive member"):
        ensure_model(spec, tmp_path / "models")


def test_direct_file_download_without_archive(tmp_path: Path) -> None:
    source = tmp_path / "plain.onnx"
    source.write_bytes(PAYLOAD)
    spec = ModelSpec(
        key="plain",
        filename="plain.onnx",
        sha256=PAYLOAD_SHA,
        url=source.as_uri(),
        size_bytes=len(PAYLOAD),
        license="test",
    )
    assert ensure_model(spec, tmp_path / "models").read_bytes() == PAYLOAD


def test_manifest_is_well_formed() -> None:
    for key, spec in MODELS.items():
        assert spec.key == key
        assert len(spec.sha256) == 64
        assert spec.size_bytes < MAX_DOWNLOAD_BYTES
        assert "non-commercial" in spec.license.lower()


# --- onnx helpers --------------------------------------------------------------------------------


def test_missing_model_error_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ModelFileError, match=r"download_models\.py"):
        verify_model_file(tmp_path / "nope.onnx", None)


def test_checksum_mismatch_error_names_both_hashes(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(PAYLOAD)
    with pytest.raises(ModelFileError) as info:
        verify_model_file(model, "a" * 64)
    assert "a" * 64 in str(info.value)
    assert PAYLOAD_SHA in str(info.value)


def test_verify_returns_the_hash_when_it_matches(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(PAYLOAD)
    assert verify_model_file(model, PAYLOAD_SHA) == PAYLOAD_SHA


def test_missing_cuda_falls_back_to_cpu_with_a_log_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="frp.onnx"):
        chosen = resolve_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    # Skip the assertion about the log if this machine genuinely has CUDA.
    if "CUDAExecutionProvider" not in chosen:
        assert chosen == ["CPUExecutionProvider"]
        assert "unavailable" in caplog.text


def test_cuda_only_request_still_yields_a_usable_provider() -> None:
    assert resolve_providers(["CUDAExecutionProvider"])  # never an empty list
