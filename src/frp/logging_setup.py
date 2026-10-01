"""Logging configuration: plain text or structured JSON, with request ids and no sensitive data.

The JSON formatter writes a fixed set of fields plus a short allow-list of ``extra`` keys. Anything
else passed through ``extra=`` (for example an image, a name or an embedding that someone logs by
mistake) is dropped, so structured logs cannot leak biometric or personal data by accident.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Final

ALLOWED_EXTRAS: Final = ("request_id", "key_id", "action", "route", "status", "identity_id")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ALLOWED_EXTRAS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            # Type and location only: exception messages can contain user data.
            exc_type = record.exc_info[0]
            payload["exception"] = exc_type.__name__ if exc_type else "Exception"
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Replace the root handlers with one stderr handler in the requested format."""
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
