from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from frp.download import download, download_verified, file_digest
from frp.errors import ModelFileError
from frp.lfw import LfwPair, identities, image_path, parse_pairs

PAYLOAD = b"x" * 5000


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "src.bin"
    path.write_bytes(PAYLOAD)
    return path


# --- download ------------------------------------------------------------------------------------


def test_download_verified_accepts_matching_digests(tmp_path: Path, source: Path) -> None:
    out = download_verified(
        source.as_uri(),
        tmp_path / "out.bin",
        sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        md5=hashlib.md5(PAYLOAD).hexdigest(),  # noqa: S324  # integrity check, not security
    )
    assert out.read_bytes() == PAYLOAD


@pytest.mark.parametrize("algorithm", ["sha256", "md5"])
def test_digest_mismatch_deletes_the_file(tmp_path: Path, source: Path, algorithm: str) -> None:
    target = tmp_path / "out.bin"
    with pytest.raises(ModelFileError, match=algorithm):
        bad = {"sha256": "0" * 64, "md5": "0" * 32}[algorithm]
        download_verified(source.as_uri(), target, **{algorithm: bad})  # type: ignore[arg-type]
    assert not target.exists()


def test_size_cap_is_enforced(tmp_path: Path, source: Path) -> None:
    target = tmp_path / "out.bin"
    with pytest.raises(ModelFileError, match=r"limit|exceeded"):
        download(source.as_uri(), target, max_bytes=1000)
    assert not target.exists()


def test_file_digest_matches_hashlib(source: Path) -> None:
    assert file_digest(source) == hashlib.sha256(PAYLOAD).hexdigest()
    assert file_digest(source, "md5") == hashlib.md5(PAYLOAD).hexdigest()  # noqa: S324


# --- LFW pairs -----------------------------------------------------------------------------------


def pairs_text(folds: int = 2, per_class: int = 2) -> str:
    lines = [f"{folds}\t{per_class}"]
    for fold in range(folds):
        lines += [f"Person_{fold}_{i}\t1\t2" for i in range(per_class)]  # same
        lines += [f"A_{fold}_{i}\t1\tB_{fold}_{i}\t3" for i in range(per_class)]  # different
    return "\n".join(lines) + "\n"


def test_parse_pairs_assigns_folds_and_labels() -> None:
    pairs = parse_pairs(pairs_text(folds=3, per_class=2))
    assert len(pairs) == 12
    assert [p.fold for p in pairs[:4]] == [0, 0, 0, 0]
    assert [p.fold for p in pairs[4:8]] == [1, 1, 1, 1]
    assert [p.same for p in pairs[:4]] == [True, True, False, False]
    first, different = pairs[0], pairs[2]
    assert first == LfwPair(0, True, "Person_0_0", 1, "Person_0_0", 2)
    assert different == LfwPair(0, False, "A_0_0", 1, "B_0_0", 3)


def test_parse_pairs_accepts_the_real_header_shape() -> None:
    text = "10\t300\n" + "".join(
        "".join(f"N{f}_{i}\t1\t2\n" for i in range(300))
        + "".join(f"A{f}_{i}\t1\tB{f}_{i}\t2\n" for i in range(300))
        for f in range(10)
    )
    pairs = parse_pairs(text)
    assert len(pairs) == 6000
    assert sum(p.same for p in pairs) == 3000
    assert {p.fold for p in pairs} == set(range(10))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "garbage\n",
        "2\t2\nonly\tone\tline\n",
        "1\t1\nA\t1\n" + "B\t1\tC\n",
        "1\t1\nA\tx\t2\nB\t1\tC\t2\n",
    ],
)
def test_parse_pairs_rejects_malformed_files(text: str) -> None:
    with pytest.raises(ValueError):
        parse_pairs(text)


def test_image_path_follows_the_lfw_layout(tmp_path: Path) -> None:
    assert (
        image_path(tmp_path, "Abel_Pacheco", 4)
        == tmp_path / "Abel_Pacheco" / "Abel_Pacheco_0004.jpg"
    )


def test_identities_lists_people_with_images(tmp_path: Path) -> None:
    for name, n in (("Zed_One", 2), ("Amy_Two", 1)):
        (tmp_path / name).mkdir()
        for i in range(1, n + 1):
            (tmp_path / name / f"{name}_{i:04d}.jpg").write_bytes(b"x")
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "stray.txt").write_text("x")
    found = identities(tmp_path)
    assert list(found) == ["Amy_Two", "Zed_One"]  # sorted, empty dir and stray file ignored
    assert [p.name for p in found["Zed_One"]] == ["Zed_One_0001.jpg", "Zed_One_0002.jpg"]
