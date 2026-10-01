"""A tiny authenticated cipher (XOR stream + HMAC) used only to test the encryption hooks."""

from __future__ import annotations

import hashlib
import hmac


class ToyCipher:
    """A tiny authenticated cipher (XOR stream + HMAC) used only to test the hook."""

    scheme = "toy-aead"

    def __init__(self, key: bytes = b"k" * 32) -> None:
        self.key = key

    def _stream(self, n: int) -> bytes:
        out = b""
        counter = 0
        while len(out) < n:
            out += hashlib.sha256(self.key + counter.to_bytes(8, "little")).digest()
            counter += 1
        return out[:n]

    def encrypt(self, plaintext: bytes, aad: bytes) -> bytes:
        body = bytes(a ^ b for a, b in zip(plaintext, self._stream(len(plaintext)), strict=True))
        tag = hmac.new(self.key, aad + body, hashlib.sha256).digest()
        return body + tag

    def decrypt(self, blob: bytes, aad: bytes) -> bytes:
        body, tag = blob[:-32], blob[-32:]
        if not hmac.compare_digest(tag, hmac.new(self.key, aad + body, hashlib.sha256).digest()):
            raise ValueError("authentication failed")
        return bytes(a ^ b for a, b in zip(body, self._stream(len(body)), strict=True))
