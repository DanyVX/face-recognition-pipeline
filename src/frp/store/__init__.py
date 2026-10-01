"""Identity store: SQLite metadata and model-tagged embedding blobs."""

from frp.store.sqlite import (
    SCHEMA_VERSION,
    EmbeddingRow,
    IdentityRecord,
    IdentityStore,
    StoreSchemaError,
)
from frp.store.validation import (
    MAX_NAME_CHARS,
    new_identity_id,
    normalize_name,
    validate_identity_id,
)

__all__ = [
    "MAX_NAME_CHARS",
    "SCHEMA_VERSION",
    "EmbeddingRow",
    "IdentityRecord",
    "IdentityStore",
    "StoreSchemaError",
    "new_identity_id",
    "normalize_name",
    "validate_identity_id",
]
