"""Exception hierarchy. Every error carries a machine-readable ``code`` (used by the API)."""

from __future__ import annotations

from typing import ClassVar


class FrpError(Exception):
    """Base class for all expected, user-facing errors."""

    code: ClassVar[str] = "INTERNAL_ERROR"
    http_status: ClassVar[int] = 500


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
    code = "INVALID_LANDMARKS"
    http_status = 422


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
