"""Encryption key rotation and plaintext-to-encrypted migration.

Rotation procedure (see docs/SECURITY_OPERATIONS.md):
1. add the new key to ``ENCRYPTION_KEYS`` and make it active (old keys stay listed);
2. run ``frp encryption migrate`` - every stored embedding and registered index snapshot is
   re-sealed with the active key;
3. once nothing is sealed with an old key (the command reports the counts), remove the old key.

The same command encrypts a store that was created without encryption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from frp.crypto import blob_key_id
from frp.index.format import load_snapshot, save_snapshot
from frp.store import IdentityStore


@dataclass(frozen=True)
class RotationReport:
    embeddings_resealed: int
    snapshots_resealed: int
    snapshots_missing: int


def rotate_encryption(store: IdentityStore) -> RotationReport:
    """Re-seal all embeddings and registered snapshots with the cipher's active key."""
    cipher = store.cipher
    if cipher is None:
        raise ValueError("Encryption is not configured; nothing to rotate.")
    rows = store.reencrypt()
    resealed = missing = 0
    for raw_path in store.list_snapshots():
        path = Path(raw_path)
        if not path.is_file():
            store.forget_snapshot(raw_path)
            missing += 1
            continue
        save_snapshot(path, load_snapshot(path, cipher), cipher)
        resealed += 1
    return RotationReport(rows, resealed, missing)


def key_usage(store: IdentityStore) -> dict[str, int]:
    """How many stored embedding blobs are sealed with each key id (``plaintext`` if none)."""
    usage: dict[str, int] = {}
    for blob in store.raw_vector_blobs():
        key = "plaintext"
        if blob[:1] == bytes([1]):
            try:
                key = blob_key_id(blob)
            except Exception:
                key = "plaintext"
        usage[key] = usage.get(key, 0) + 1
    return usage
