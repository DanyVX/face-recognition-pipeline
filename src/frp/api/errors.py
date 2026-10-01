"""HTTP-level errors. All carry a machine-readable ``code`` and an HTTP status."""

from __future__ import annotations

from frp.errors import FrpError


class InvalidRequestError(FrpError):
    code = "INVALID_REQUEST"
    http_status = 422


class UnsupportedMediaTypeError(FrpError):
    code = "UNSUPPORTED_MEDIA_TYPE"
    http_status = 415


class UnauthorizedError(FrpError):
    code = "UNAUTHORIZED"
    http_status = 401


class RateLimitedError(FrpError):
    code = "RATE_LIMITED"
    http_status = 429

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PayloadTooLargeError(FrpError):
    code = "PAYLOAD_TOO_LARGE"
    http_status = 413


class UploadTimeoutError(FrpError):
    code = "UPLOAD_TIMEOUT"
    http_status = 408


class RequestTimeoutError(FrpError):
    code = "REQUEST_TIMEOUT"
    http_status = 504


class OverloadedError(FrpError):
    code = "OVERLOADED"
    http_status = 503


class NotReadyError(FrpError):
    code = "NOT_READY"
    http_status = 503


class IdempotencyKeyReusedError(FrpError):
    code = "IDEMPOTENCY_KEY_REUSED"
    http_status = 422
