"""Optional encrypted storage of the raw uploaded images (``STORE_RAW_IMAGES=true``).

Off by default: the default retention of raw biometric inputs is none. When enabled, every image is
sealed with AES-256-GCM (associated data binds it to its identity), written atomically with
owner-only permissions, and registered in the identity database so that erasure can find and
remove it.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from frp.index.format import PayloadCipher
from frp.store import IdentityStore


def _aad(identity_id: str) -> bytes:
    return f"raw|{identity_id}".encode()


class RawImageStore:
    def __init__(self, directory: Path, store: IdentityStore, cipher: PayloadCipher) -> None:
        self.directory = directory
        self._store = store
        self._cipher = cipher
        directory.mkdir(parents=True, exist_ok=True)

    def save(self, identity_id: str, data: bytes) -> str:
        """Encrypt and store one image; returns the opaque file name."""
        filename = f"{uuid.uuid4().hex}.frpimg"
        sealed = self._cipher.encrypt(data, _aad(identity_id))
        handle, tmp_name = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as tmp:
                tmp.write(sealed)
                tmp.flush()
                os.fsync(tmp.fileno())
            Path(tmp_name).replace(self.directory / filename)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        self._store.add_raw_image(identity_id, filename)
        return filename

    def read(self, identity_id: str, filename: str) -> bytes:
        return self._cipher.decrypt((self.directory / filename).read_bytes(), _aad(identity_id))

    def delete_identity(self, identity_id: str) -> int:
        """Overwrite and remove every stored image of ``identity_id``; returns how many."""
        removed = 0
        for filename in self._store.raw_images_of(identity_id):
            path = self.directory / filename
            if path.is_file():
                size = path.stat().st_size
                with path.open("r+b") as handle:  # best-effort overwrite before unlinking
                    handle.write(b"\x00" * size)
                    handle.flush()
                    os.fsync(handle.fileno())
                path.unlink()
                removed += 1
        self._store.forget_raw_images(identity_id)
        return removed

    def files_present(self, identity_id: str) -> list[str]:
        return [f for f in self._store.raw_images_of(identity_id) if (self.directory / f).is_file()]
