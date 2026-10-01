"""Validation and normalization for identity ids and display names."""

from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Final

from frp.errors import InvalidIdentityError

MAX_NAME_CHARS: Final = 256
_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
# Bidirectional overrides/isolates can make a name render as something else in logs and UIs.
_BIDI_CONTROLS: Final = frozenset("‪‫‬‭‮⁦⁧⁨⁩")


def new_identity_id() -> str:
    """A random, non-guessable identity id (UUID4 hex)."""
    return uuid.uuid4().hex


def validate_identity_id(identity_id: str) -> str:
    """Ids are 1-128 ASCII characters: letters, digits and ``._:@-``, starting alphanumeric."""
    if not isinstance(identity_id, str) or not _ID_PATTERN.fullmatch(identity_id):
        raise InvalidIdentityError(
            "identity_id must be 1-128 characters from A-Z a-z 0-9 . _ : @ - "
            "and start with a letter or digit."
        )
    return identity_id


def normalize_name(name: str, *, max_chars: int = MAX_NAME_CHARS) -> str:
    """NFC-normalize and validate a display name (any script; no control or bidi override chars)."""
    if not isinstance(name, str):
        raise InvalidIdentityError("name must be text.")
    cleaned = unicodedata.normalize("NFC", name).strip()
    if not cleaned:
        raise InvalidIdentityError("name must not be empty.")
    if len(cleaned) > max_chars:
        raise InvalidIdentityError(f"name is too long ({len(cleaned)} > {max_chars} characters).")
    for ch in cleaned:
        category = unicodedata.category(ch)
        if category in ("Cc", "Cs") or ch in _BIDI_CONTROLS:
            raise InvalidIdentityError(
                f"name contains a disallowed character (U+{ord(ch):04X}, category {category})."
            )
    return cleaned
