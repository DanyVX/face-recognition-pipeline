"""Per-key token-bucket rate limiting (in process; see LIMITATIONS for multi-node caveats)."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    retry_after: float  # seconds until one token is available (0 when allowed)


class TokenBucketLimiter:
    """Each key owns a bucket of ``burst`` tokens refilled at ``per_minute / 60`` per second."""

    def __init__(
        self,
        per_minute: int,
        burst: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if per_minute < 1 or burst < 1:
            raise ValueError("per_minute and burst must be >= 1")
        self._rate = per_minute / 60.0
        self._burst = float(burst)
        self._clock = clock
        self._max_keys = max_keys
        self._lock = threading.Lock()
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # tokens, last seen

    def check(self, key: str) -> RateDecision:
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.pop(key, (self._burst, now))
            tokens = min(self._burst, tokens + (now - last) * self._rate)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[key] = (tokens, now)
            while len(self._buckets) > self._max_keys:  # drop the least recently used key
                self._buckets.popitem(last=False)
        if allowed:
            return RateDecision(True, 0.0)
        return RateDecision(False, (1.0 - tokens) / self._rate)
