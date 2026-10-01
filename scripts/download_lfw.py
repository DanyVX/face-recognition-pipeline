"""Download and verify LFW (images + pairs.txt) into ``DATA_DIR/lfw`` (default ``data/lfw``).

Usage: ``uv run python scripts/download_lfw.py [--force]``

The archive is about 173 MB (under the 500 MB approval limit). Digests: SHA-256 as pinned by
scikit-learn for the figshare mirror and the md5 from the official LFW page; both are checked.
LFW's license terms are unverified (see DATA.md): research use only, never redistribute.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile

from frp.config import load_settings
from frp.download import download_verified, file_digest
from frp.errors import FrpError
from frp.lfw import (
    IMAGES_DIRNAME,
    LFW_ARCHIVE_MD5,
    LFW_ARCHIVE_SHA256,
    LFW_ARCHIVE_URL,
    LFW_PAIRS_SHA256,
    LFW_PAIRS_URL,
    PAIRS_FILENAME,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("download_lfw")

    root = load_settings().data_dir / "lfw"
    images = root / IMAGES_DIRNAME
    pairs = root / PAIRS_FILENAME
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.force or not pairs.is_file() or file_digest(pairs) != LFW_PAIRS_SHA256:
            log.info("downloading %s", PAIRS_FILENAME)
            download_verified(LFW_PAIRS_URL, pairs, sha256=LFW_PAIRS_SHA256)
        if args.force or not images.is_dir():
            archive = root / "lfw.tgz"
            log.info("downloading lfw.tgz (about 173 MB)")
            download_verified(
                LFW_ARCHIVE_URL, archive, sha256=LFW_ARCHIVE_SHA256, md5=LFW_ARCHIVE_MD5
            )
            if images.is_dir():
                shutil.rmtree(images)
            log.info("extracting")
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter="data")  # refuses absolute paths and escapes
            archive.unlink()
    except FrpError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    count = sum(1 for _ in images.rglob("*.jpg"))
    print(f"OK  {count} images in {images}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
