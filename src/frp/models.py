"""Model manifest and checksum-verified downloader (stdlib only).

Checksums were recorded from a download on 2026-10-01 (trust on first use): upstream does not
publish checksums that I verified, so the pins protect against later corruption or substitution,
not against a bad first download. See MODEL_CARD.md for license terms of each model.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from frp.errors import ModelFileError

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES: Final = 500 * 1024 * 1024  # CLAUDE.md: ask before downloading more than this


@dataclass(frozen=True)
class ModelSpec:
    """A model file and where it comes from."""

    key: str  # stable identifier, e.g. "scrfd_10g"
    filename: str  # name inside the models directory
    sha256: str  # of the final .onnx file
    url: str  # direct file, or a zip archive when ``archive_member`` is set
    size_bytes: int  # of the downloaded artifact (file or archive)
    license: str  # short, human-readable; the authoritative text is in MODEL_CARD.md
    archive_member: str | None = None  # file to extract from a zip archive
    download_sha256: str | None = None  # of the downloaded artifact (when it is an archive)


_BUFFALO_L = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
_BUFFALO_SHA = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
_BUFFALO_SIZE = 288_621_354
_INSIGHTFACE_LICENSE = "InsightFace pretrained models: non-commercial research use only (README)"

MODELS: Final[dict[str, ModelSpec]] = {
    "scrfd_10g": ModelSpec(
        key="scrfd_10g",
        filename="det_10g.onnx",
        sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        url=_BUFFALO_L,
        size_bytes=_BUFFALO_SIZE,
        license=_INSIGHTFACE_LICENSE,
        archive_member="det_10g.onnx",
        download_sha256=_BUFFALO_SHA,
    ),
    "arcface_r50": ModelSpec(
        key="arcface_r50",
        filename="w600k_r50.onnx",
        sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        url=_BUFFALO_L,
        size_bytes=_BUFFALO_SIZE,
        license=_INSIGHTFACE_LICENSE,
        archive_member="w600k_r50.onnx",
        download_sha256=_BUFFALO_SHA,
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected_size: int) -> None:
    if expected_size > MAX_DOWNLOAD_BYTES:
        msg = f"{url} is {expected_size:,} bytes, above the 500 MB limit; ask before downloading."
        raise ModelFileError(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "frp-model-downloader"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out, 1 << 20)


def ensure_model(spec: ModelSpec, models_dir: Path, *, force: bool = False) -> Path:
    """Make sure ``spec`` exists and verifies in ``models_dir``; download it if needed."""
    target = models_dir / spec.filename
    if target.is_file() and not force:
        if sha256_of(target) == spec.sha256:
            return target
        logger.warning("%s exists but fails its checksum; re-downloading", target)
    models_dir.mkdir(parents=True, exist_ok=True)
    staging = models_dir / "_download"
    staging.mkdir(exist_ok=True)
    artifact = staging / Path(spec.url).name
    try:
        logger.info("downloading %s (%s bytes)", spec.url, f"{spec.size_bytes:,}")
        _download(spec.url, artifact, spec.size_bytes)
        if spec.download_sha256 is not None and sha256_of(artifact) != spec.download_sha256:
            msg = f"Downloaded archive for {spec.key} failed its checksum; refusing to use it."
            raise ModelFileError(msg)
        if spec.archive_member is not None:
            with zipfile.ZipFile(artifact) as archive:
                member_path = Path(spec.archive_member)
                if member_path.name != spec.archive_member:  # no path tricks inside the zip
                    raise ModelFileError(f"Unexpected archive member: {spec.archive_member}")
                with archive.open(spec.archive_member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
        else:
            shutil.move(artifact, target)
        if sha256_of(target) != spec.sha256:
            target.unlink(missing_ok=True)
            msg = f"{spec.filename} failed its checksum after extraction; removed it."
            raise ModelFileError(msg)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target
