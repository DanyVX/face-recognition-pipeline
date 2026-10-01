"""Index file format: round trip, versioning, corruption, atomic writes, encryption hook."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frp.embed import ModelInfo
from frp.errors import IndexCorruptError, IndexVersionError
from frp.index import FORMAT_VERSION, IndexSnapshot, deserialize, serialize
from frp.index.format import MAGIC, load_snapshot, save_snapshot, write_atomic
from tests.unit.toy_cipher import ToyCipher

DIM = 16
INFO = ModelInfo("test/model", "a" * 64, "pre-v1", DIM)


def unit(n: int, seed: int = 0, dim: int = DIM) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def snapshot(n: int = 20, seed: int = 0) -> IndexSnapshot:
    return IndexSnapshot(
        info=INFO,
        backend="hnsw",
        params={"hnsw_m": 16, "seed": 3},
        ids=np.arange(100, 100 + n, dtype=np.int64),
        vectors=unit(n, seed),
    )


def reseal(body: bytes) -> bytes:
    """Recompute the trailer, simulating corruption that the checksum cannot catch."""
    return body + hashlib.sha256(body).digest()


def split(data: bytes) -> tuple[dict[str, object], bytes]:
    (header_len,) = struct.unpack("<I", data[len(MAGIC) : len(MAGIC) + 4])
    start = len(MAGIC) + 4
    header = json.loads(data[start : start + header_len])
    return header, data[start + header_len : -32]


def rebuild(header: dict[str, object], payload: bytes) -> bytes:
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return reseal(MAGIC + struct.pack("<I", len(raw)) + raw + payload)


# --- round trip ----------------------------------------------------------------------------------


def test_round_trip_preserves_everything() -> None:
    snap = snapshot()
    back = deserialize(serialize(snap))
    assert back.info == INFO and back.backend == "hnsw" and back.params == snap.params
    np.testing.assert_array_equal(back.ids, snap.ids)
    np.testing.assert_array_equal(back.vectors, snap.vectors)  # bit-exact


def test_empty_snapshot_round_trips() -> None:
    empty = IndexSnapshot(INFO, "flat", {}, np.zeros(0, np.int64), np.zeros((0, DIM), np.float32))
    back = deserialize(serialize(empty))
    assert back.ids.shape == (0,) and back.vectors.shape == (0, DIM)


@settings(max_examples=40, deadline=None)
@given(n=st.integers(0, 60), seed=st.integers(0, 10_000))
def test_round_trip_property(n: int, seed: int) -> None:
    snap = snapshot(n, seed)
    back = deserialize(serialize(snap))
    np.testing.assert_array_equal(back.vectors, snap.vectors)
    np.testing.assert_array_equal(back.ids, snap.ids)


def test_header_is_human_inspectable_json_with_version() -> None:
    header, payload = split(serialize(snapshot(5)))
    assert header["format_version"] == FORMAT_VERSION
    assert header["count"] == 5 and header["model"]["dim"] == DIM  # type: ignore[index]
    assert len(payload) == 5 * 8 + 5 * DIM * 4


def test_file_round_trip_via_helpers(tmp_path: Path) -> None:
    path = tmp_path / "idx.frpidx"
    save_snapshot(path, snapshot())
    np.testing.assert_array_equal(load_snapshot(path).ids, snapshot().ids)


# --- versioning ----------------------------------------------------------------------------------


def test_unknown_format_version_is_refused_explicitly() -> None:
    header, payload = split(serialize(snapshot()))
    header["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(IndexVersionError, match="Rebuild the index"):
        deserialize(rebuild(header, payload))


def test_unknown_container_version_is_refused_explicitly() -> None:
    data = bytearray(serialize(snapshot()))
    data[len(MAGIC) - 1] = 2
    with pytest.raises(IndexVersionError):
        deserialize(reseal(bytes(data[:-32])))


# --- corruption ----------------------------------------------------------------------------------


def test_bad_magic_and_short_files_are_corrupt() -> None:
    with pytest.raises(IndexCorruptError, match="magic"):
        deserialize(b"NOTANIDX" + b"\x00" * 100)
    with pytest.raises(IndexCorruptError, match="short"):
        deserialize(b"FRP")
    with pytest.raises(IndexCorruptError):
        deserialize(b"")


def test_every_truncation_is_detected() -> None:
    data = serialize(snapshot(6))
    for cut in range(0, len(data), max(1, len(data) // 60)):
        with pytest.raises(IndexCorruptError):
            deserialize(data[:cut])
    with pytest.raises(IndexCorruptError):
        deserialize(data[:-1])


def test_single_byte_flips_anywhere_are_detected() -> None:
    data = bytearray(serialize(snapshot(6)))
    for position in range(0, len(data), max(1, len(data) // 80)):
        mutated = bytearray(data)
        mutated[position] ^= 0x01
        with pytest.raises((IndexCorruptError, IndexVersionError)):
            deserialize(bytes(mutated))


def test_checksum_valid_but_inconsistent_files_are_rejected() -> None:
    good = serialize(snapshot(6))
    header, payload = split(good)

    wrong_count = {**header, "count": 7}
    with pytest.raises(IndexCorruptError, match="does not match"):
        deserialize(rebuild(wrong_count, payload))

    wrong_dim = {**header, "model": {**header["model"], "dim": DIM + 1}}  # type: ignore[dict-item]
    with pytest.raises(IndexCorruptError, match="does not match"):
        deserialize(rebuild(wrong_dim, payload))

    dup = bytearray(payload)
    dup[8:16] = dup[0:8]  # second id equals first id
    with pytest.raises(IndexCorruptError, match="duplicate"):
        deserialize(rebuild(header, bytes(dup)))

    scaled = np.frombuffer(payload[6 * 8 :], dtype="<f4") * 3.0
    with pytest.raises(IndexCorruptError, match="not L2-normalized"):
        deserialize(rebuild(header, payload[: 6 * 8] + scaled.astype("<f4").tobytes()))

    nan_vecs = np.frombuffer(payload[6 * 8 :], dtype="<f4").copy()
    nan_vecs[3] = np.nan
    with pytest.raises(IndexCorruptError):
        deserialize(rebuild(header, payload[: 6 * 8] + nan_vecs.tobytes()))

    with pytest.raises(IndexCorruptError, match="header"):
        deserialize(reseal(MAGIC + struct.pack("<I", 999_999_999) + b"{}" + b"\x00" * 8))


def test_invalid_json_header_is_corrupt() -> None:
    raw = b"{not json"
    with pytest.raises(IndexCorruptError, match="header"):
        deserialize(reseal(MAGIC + struct.pack("<I", len(raw)) + raw))


def test_refuses_to_write_inconsistent_snapshots() -> None:
    bad = IndexSnapshot(INFO, "flat", {}, np.arange(3, dtype=np.int64), unit(2))
    with pytest.raises(IndexCorruptError):
        serialize(bad)


# --- atomic writes -------------------------------------------------------------------------------


def test_failed_rename_keeps_the_old_file_and_removes_the_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "idx.frpidx"
    save_snapshot(path, snapshot(5, seed=1))
    original = path.read_bytes()

    def boom(self: Path, target: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        save_snapshot(path, snapshot(5, seed=2))
    monkeypatch.undo()
    assert path.read_bytes() == original  # old index intact and loadable
    assert load_snapshot(path).ids.shape == (5,)
    assert [p.name for p in tmp_path.iterdir()] == ["idx.frpidx"]  # no stray temp files


def test_failure_while_writing_leaves_no_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    def boom(fd: int) -> None:
        raise OSError("io error")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="io error"):
        write_atomic(tmp_path / "new.frpidx", b"data")
    assert list(tmp_path.iterdir()) == []


# --- encryption hook -----------------------------------------------------------------------------


def test_encrypted_round_trip_hides_the_vectors() -> None:
    snap = snapshot(8)
    blob = serialize(snap, ToyCipher())
    assert snap.vectors[0].tobytes() not in blob
    assert snap.vectors[0].tobytes() in serialize(snap)  # sanity: visible without the cipher
    back = deserialize(blob, ToyCipher())
    np.testing.assert_array_equal(back.vectors, snap.vectors)
    header, _ = split(blob)
    assert header["encryption"] == {"scheme": "toy-aead"}


def test_wrong_key_missing_key_and_unexpected_cipher_are_rejected() -> None:
    blob = serialize(snapshot(8), ToyCipher())
    with pytest.raises(IndexCorruptError, match="authenticated decryption"):
        deserialize(blob, ToyCipher(b"x" * 32))
    with pytest.raises(IndexCorruptError, match="no key"):
        deserialize(blob)
    with pytest.raises(IndexCorruptError, match="not encrypted"):
        deserialize(serialize(snapshot(8)), ToyCipher())


def test_encrypted_payload_cannot_be_moved_under_a_different_header() -> None:
    cipher = ToyCipher()
    a = serialize(snapshot(8), cipher)
    header, payload = split(a)
    other_header = {**header, "backend": "ivf"}  # header changed, ciphertext reused
    with pytest.raises(IndexCorruptError, match="authenticated decryption"):
        deserialize(rebuild(other_header, payload), cipher)
