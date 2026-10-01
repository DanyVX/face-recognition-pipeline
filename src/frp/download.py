"""Checksum-verified file download with a hard size cap (stdlib only)."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Final

from frp.errors import ModelFileError

MAX_DOWNLOAD_BYTES: Final = 500 * 1024 * 1024  # CLAUDE.md: ask before downloading more than this
_CHUNK: Final = 1 << 20


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, *, max_bytes: int = MAX_DOWNLOAD_BYTES) -> int:
    """Stream ``url`` into ``dest`` and return the byte count.

    Refuses (before writing) when the server announces more than ``max_bytes`` and aborts if the
    stream exceeds it. Raises ``ModelFileError`` for either case.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "frp-downloader"})  # noqa: S310
    written = 0
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        announced = response.headers.get("Content-Length")
        if announced is not None and int(announced) > max_bytes:
            raise ModelFileError(
                f"{url} is {int(announced):,} bytes, above the {max_bytes:,} byte limit; "
                "ask before downloading."
            )
        with dest.open("wb") as out:
            while block := response.read(_CHUNK):
                written += len(block)
                if written > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise ModelFileError(
                        f"Download from {url} exceeded {max_bytes:,} bytes; aborted."
                    )
                out.write(block)
    return written


def download_verified(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    md5: str | None = None,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Path:
    """Download, then verify the given digests; delete the file and raise on any mismatch."""
    download(url, dest, max_bytes=max_bytes)
    for algorithm, expected in (("sha256", sha256), ("md5", md5)):
        if expected is None:
            continue
        actual = file_digest(dest, algorithm)
        if actual != expected:
            dest.unlink(missing_ok=True)
            raise ModelFileError(
                f"{algorithm} mismatch for {url}: expected {expected}, got {actual}. "
                "The file was deleted."
            )
    return dest


def copy_stream(src_path: Path, dst_path: Path) -> None:
    with src_path.open("rb") as src, dst_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, _CHUNK)
