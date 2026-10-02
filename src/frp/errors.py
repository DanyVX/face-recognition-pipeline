"""Exception hierarchy. Every error carries a machine-readable ``code`` (used by the API)."""

from __future__ import annotations

from typing import Any, ClassVar


class FrpError(Exception):
    """Base class for all expected, user-facing errors."""

    code: ClassVar[str] = "INTERNAL_ERROR"
    http_status: ClassVar[int] = 500
    # Optional machine-readable context (for example which uploaded image failed).
    details: dict[str, Any] | None = None


# --- input images -----------------------------------------------------------------------------


class InvalidImageError(FrpError):
    code = "INVALID_IMAGE"
    http_status = 422


class EmptyImageError(InvalidImageError):
    code = "EMPTY_IMAGE"


class UnsupportedImageFormatError(InvalidImageError):
    code = "UNSUPPORTED_IMAGE_FORMAT"
    http_status = 415


class CorruptImageError(InvalidImageError):
    code = "CORRUPT_IMAGE"


class ImageTooLargeError(InvalidImageError):
    code = "IMAGE_TOO_LARGE"
    http_status = 413


class AnimatedImageError(InvalidImageError):
    code = "ANIMATED_IMAGE_REJECTED"


# --- detection / alignment / quality -----------------------------------------------------------


class NoFaceError(FrpError):
    code = "NO_FACE"
    http_status = 422


class MultipleFacesError(FrpError):
    code = "MULTIPLE_FACES"
    http_status = 422


class LowQualityError(FrpError):
    code = "LOW_QUALITY"
    http_status = 422

    def __init__(self, message: str, reasons: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.reasons = reasons


class InvalidLandmarksError(FrpError):
    """Landmarks cannot be used for alignment. ``reason`` is a stable machine-readable token."""

    code = "INVALID_LANDMARKS"
    http_status = 422

    def __init__(self, message: str, reason: str = "INVALID") -> None:
        super().__init__(message)
        self.reason = reason


# --- models / embeddings ---------------------------------------------------------------------


class ModelFileError(FrpError):
    """A model file is missing or fails its checksum."""

    code = "MODEL_UNAVAILABLE"
    http_status = 503


class EmbeddingError(FrpError):
    """The embedder produced an unusable result (NaN/inf, wrong shape)."""

    code = "EMBEDDING_FAILED"


class ModelMismatchError(FrpError):
    """Embeddings from different model versions were mixed."""

    code = "MODEL_MISMATCH"
    http_status = 409


# --- index / store / identities ----------------------------------------------------------------


class VectorIndexError(FrpError):
    """Base for vector-index problems."""

    code = "INDEX_ERROR"


class IndexCorruptError(VectorIndexError):
    """An index file is truncated, tampered with, or internally inconsistent."""

    code = "INDEX_CORRUPT"


class IndexVersionError(VectorIndexError):
    """An index file uses a format version this build cannot read."""

    code = "INDEX_VERSION_UNSUPPORTED"


class DuplicateIdError(VectorIndexError):
    """A vector id was added twice."""

    code = "DUPLICATE_VECTOR_ID"
    http_status = 409


class InvalidIdentityError(FrpError):
    """An identity id or name failed validation."""

    code = "INVALID_IDENTITY"
    http_status = 422


class IdentityNotFoundError(FrpError):
    code = "IDENTITY_NOT_FOUND"
    http_status = 404


class IdentityExistsError(FrpError):
    code = "IDENTITY_EXISTS"
    http_status = 409


class DuplicateIdentityError(FrpError):
    """The face is already enrolled under a different identity (policy: reject)."""

    code = "DUPLICATE_IDENTITY"
    http_status = 409

    def __init__(self, message: str, matches: tuple[tuple[str, float], ...] = ()) -> None:
        super().__init__(message)
        self.matches = matches  # (identity_id, similarity)
