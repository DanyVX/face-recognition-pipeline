"""Labeled Faces in the Wild: layout, pair-file parsing and download metadata.

Provenance of the pinned digests is recorded in DATA.md. LFW's license terms are unverified; the
data is for research use and must never be committed or redistributed from this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# figshare mirror used by scikit-learn (digest copied from scikit-learn's source); the original
# is http://vis-www.cs.umass.edu/lfw/lfw.tgz, whose page lists md5 a17d05bd...
LFW_ARCHIVE_URL: Final = "https://ndownloader.figshare.com/files/5976018"
LFW_ARCHIVE_SHA256: Final = "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"
LFW_ARCHIVE_MD5: Final = "a17d05bd522c52d84eca14327a23d494"
LFW_PAIRS_URL: Final = "https://ndownloader.figshare.com/files/5976006"
LFW_PAIRS_SHA256: Final = "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"

IMAGES_DIRNAME: Final = "lfw"
PAIRS_FILENAME: Final = "pairs.txt"


@dataclass(frozen=True)
class LfwPair:
    """One verification pair from ``pairs.txt``."""

    fold: int  # 0-based
    same: bool
    name_a: str
    index_a: int  # 1-based image number within the person's folder
    name_b: str
    index_b: int


def image_path(root: Path, name: str, index: int) -> Path:
    """``root/<name>/<name>_<index:04d>.jpg`` (the LFW layout)."""
    return root / name / f"{name}_{index:04d}.jpg"


def parse_pairs(text: str) -> list[LfwPair]:
    """Parse ``pairs.txt`` (the standard 10-fold, 300 same + 300 different per fold layout)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("pairs file is empty")
    header = lines[0].split()
    if len(header) != 2 or not all(part.isdigit() for part in header):
        raise ValueError(f"bad pairs header: {lines[0]!r}")
    folds, per_class = int(header[0]), int(header[1])
    body = lines[1:]
    if len(body) != folds * per_class * 2:
        raise ValueError(
            f"expected {folds * per_class * 2} pair lines for {folds} folds of "
            f"{per_class}+{per_class}, found {len(body)}"
        )
    pairs: list[LfwPair] = []
    for number, line in enumerate(body, start=2):
        fields = line.split("\t") if "\t" in line else line.split()
        fold = (number - 2) // (per_class * 2)
        try:
            if len(fields) == 3:
                pairs.append(
                    LfwPair(fold, True, fields[0], int(fields[1]), fields[0], int(fields[2]))
                )
            elif len(fields) == 4:
                pairs.append(
                    LfwPair(fold, False, fields[0], int(fields[1]), fields[2], int(fields[3]))
                )
            else:
                raise ValueError(f"{len(fields)} fields")
        except ValueError as exc:
            raise ValueError(f"bad pair at line {number}: {line!r} ({exc})") from exc
    return pairs


def identities(root: Path) -> dict[str, list[Path]]:
    """Map person name -> sorted image paths for an extracted LFW ``root``."""
    result: dict[str, list[Path]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(person_dir.glob("*.jpg"))
        if images:
            result[person_dir.name] = images
    return result
