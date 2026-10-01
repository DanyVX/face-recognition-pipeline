"""API building blocks: key store, rate limiter, idempotency store."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from frp.api.auth import ApiKeyStore
from frp.api.errors import IdempotencyKeyReusedError, InvalidRequestError
from frp.api.idempotency import IdempotencyStore
from frp.api.metrics import Metrics
from frp.api.ratelimit import TokenBucketLimiter

# --- API keys ------------------------------------------------------------------------------------


def test_created_keys_verify_and_wrong_ones_do_not(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "auth.db")
    key_id, key = store.create("ci runner")
    assert key.startswith(f"frp_{key_id}_")
    record = store.verify(key)
    assert record is not None and record.key_id == key_id and record.name == "ci runner"
    secret = key.rsplit("_", 1)[1]
    tampered = key[:-1] + ("A" if key[-1] != "A" else "B")
    for bad in (
        None,
        "",
        "nonsense",
        tampered,
        f"frp_{key_id}_{'x' * 43}",
        f"frp_deadbeef_{secret}",
    ):
        assert store.verify(bad) is None, bad


def test_only_a_hash_of_the_secret_is_stored(tmp_path: Path) -> None:
    path = tmp_path / "auth.db"
    store = ApiKeyStore(path)
    _, key = store.create("k")
    store.close()
    secret = key.rsplit("_", 1)[1]
    assert secret.encode() not in path.read_bytes()
    assert key.encode() not in path.read_bytes()


def test_revoked_keys_stop_working_and_listing_shows_state(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "auth.db")
    key_id, key = store.create("a")
    other_id, other = store.create("b")
    assert store.revoke(key_id) is True and store.revoke(key_id) is False  # second call: no-op
    assert store.verify(key) is None and store.verify(other) is not None
    states = {r.key_id: r.revoked for r in store.list()}
    assert states == {key_id: True, other_id: False}
    assert store.revoke("00000000") is False


@pytest.mark.parametrize("name", ["", "   ", "x" * 101])
def test_key_names_are_validated(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        ApiKeyStore(tmp_path / "auth.db").create(name)


def test_keys_are_unique(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "auth.db")
    keys = {store.create(f"k{i}")[1] for i in range(50)}
    assert len(keys) == 50


# --- rate limiting -------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_burst_then_throttle_then_refill() -> None:
    clock = Clock()
    limiter = TokenBucketLimiter(per_minute=60, burst=3, clock=clock)  # 1 token per second
    assert [limiter.check("a").allowed for _ in range(3)] == [True, True, True]
    denied = limiter.check("a")
    assert not denied.allowed and denied.retry_after == pytest.approx(1.0)
    clock.now += 0.5
    assert limiter.check("a").retry_after == pytest.approx(0.5)  # half a token short
    clock.now += 0.5
    assert limiter.check("a").allowed  # one token has accrued
    assert not limiter.check("a").allowed
    clock.now += 100  # a long idle period never exceeds the burst size
    assert [limiter.check("a").allowed for _ in range(4)] == [True, True, True, False]


def test_keys_have_independent_buckets_and_old_keys_are_evicted() -> None:
    clock = Clock()
    limiter = TokenBucketLimiter(per_minute=60, burst=1, clock=clock, max_keys=2)
    assert limiter.check("a").allowed and not limiter.check("a").allowed
    assert limiter.check("b").allowed  # b is unaffected by a
    limiter.check("c")  # three keys with room for two: the least recently used (a) is dropped
    assert limiter.check("a").allowed  # a starts again with a full bucket


def test_limiter_rejects_invalid_configuration() -> None:
    for per_minute, burst in ((0, 1), (1, 0)):
        with pytest.raises(ValueError):
            TokenBucketLimiter(per_minute, burst)


# --- idempotency ---------------------------------------------------------------------------------


def producer_returning(status: int = 201, calls: list[int] | None = None) -> Any:
    def run() -> tuple[int, dict[str, Any], str | None]:
        if calls is not None:
            calls.append(1)
        return status, {"identity_id": "alice", "n": len(calls or [])}, "alice"

    return run


def test_same_key_and_request_replays_without_running_again(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "idem.db", ttl_seconds=60)
    calls: list[int] = []
    first = store.run("client-1", "key-1", "hash-a", producer_returning(calls=calls))
    again = store.run("client-1", "key-1", "hash-a", producer_returning(calls=calls))
    assert first == (201, {"identity_id": "alice", "n": 1}, False)
    assert again == (201, {"identity_id": "alice", "n": 1}, True) and len(calls) == 1


def test_same_key_with_different_content_is_a_conflict(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "idem.db", ttl_seconds=60)
    store.run("c", "k", "hash-a", producer_returning())
    with pytest.raises(IdempotencyKeyReusedError):
        store.run("c", "k", "hash-b", producer_returning())


def test_keys_are_scoped_per_client(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "idem.db", ttl_seconds=60)
    calls: list[int] = []
    store.run("client-1", "k", "h", producer_returning(calls=calls))
    store.run("client-2", "k", "h", producer_returning(calls=calls))
    assert len(calls) == 2  # the other client's key is a different key


def test_failed_requests_are_not_stored_so_a_retry_runs_again(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "idem.db", ttl_seconds=60)
    calls: list[int] = []
    assert store.run("c", "k", "h", producer_returning(status=422, calls=calls))[2] is False
    assert store.run("c", "k", "h", producer_returning(status=201, calls=calls))[2] is False
    assert len(calls) == 2
    assert store.run("c", "k", "h", producer_returning(calls=calls))[2] is True  # now stored


def test_exceptions_from_the_producer_propagate_and_store_nothing(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "idem.db", ttl_seconds=60)

    def boom() -> tuple[int, dict[str, Any], str | None]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        store.run("c", "k", "h", boom)
    assert store.run("c", "k", "h", producer_returning())[2] is False


def test_concurrent_identical_requests_execute_once_and_the_rest_replay(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "idem.db", ttl_seconds=60)
    calls: list[int] = []

    def slow() -> tuple[int, dict[str, Any], str | None]:
        calls.append(1)
        time.sleep(0.2)
        return 201, {"identity_id": "alice"}, "alice"

    results: list[tuple[int, dict[str, Any], bool]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(store.run("c", "same-key", "same-hash", slow))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors and len(calls) == 1
    assert sorted(r[2] for r in results) == [False] + [True] * 5
    assert all(r[1] == {"identity_id": "alice"} for r in results)


def test_records_expire_after_the_ttl(tmp_path: Path) -> None:
    clock = Clock()
    store = IdempotencyStore(tmp_path / "idem.db", ttl_seconds=60, clock=clock)
    calls: list[int] = []
    store.run("c", "k", "h", producer_returning(calls=calls))
    clock.now += 61
    assert store.run("c", "k", "h", producer_returning(calls=calls))[2] is False
    assert len(calls) == 2


def test_key_validation_and_purging(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "idem.db", ttl_seconds=60)
    for bad in ("", "has space", "x" * 129, "bad/char"):
        with pytest.raises(InvalidRequestError):
            store.run("c", bad, "h", producer_returning())
    store.run("c", "k1", "h", producer_returning())
    assert store.purge_identity("alice") == 1 and store.purge_identity("alice") == 0
    assert store.run("c", "k1", "h", producer_returning())[2] is False  # record is gone


# --- metrics -------------------------------------------------------------------------------------


def test_metrics_render_prometheus_text_and_registries_are_isolated() -> None:
    a, b = Metrics(), Metrics()
    a.requests.labels(route="/x", method="GET", status="200").inc()
    a.observe_stage("embed", 0.01)
    text = a.render().decode()
    assert 'frp_http_requests_total{method="GET",route="/x",status="200"} 1.0' in text
    assert 'frp_pipeline_stage_duration_seconds_count{stage="embed"} 1.0' in text
    assert "frp_http_requests_total{" not in b.render().decode()  # no shared global state
