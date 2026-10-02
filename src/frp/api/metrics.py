"""Prometheus metrics. One registry per app instance (so tests can create many apps)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "frp_http_requests_total",
            "HTTP requests by route template, method and status.",
            ["route", "method", "status"],
            registry=self.registry,
        )
        self.request_seconds = Histogram(
            "frp_http_request_duration_seconds",
            "HTTP request duration by route template and method.",
            ["route", "method"],
            buckets=_BUCKETS,
            registry=self.registry,
        )
        self.stage_seconds = Histogram(
            "frp_pipeline_stage_duration_seconds",
            "Pipeline stage duration (decode, detect_align, embed, match).",
            ["stage"],
            buckets=_BUCKETS,
            registry=self.registry,
        )
        self.errors = Counter(
            "frp_errors_total",
            "Errors returned, by machine-readable code.",
            ["code"],
            registry=self.registry,
        )
        self.rate_limited = Counter(
            "frp_rate_limited_total",
            "Requests rejected by the rate limiter.",
            registry=self.registry,
        )
        self.auth_failures = Counter(
            "frp_auth_failures_total",
            "Requests with a missing or invalid API key.",
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "frp_in_flight_requests", "Requests being processed.", registry=self.registry
        )
        self.ready = Gauge(
            "frp_ready",
            "1 when models are loaded and the service accepts work.",
            registry=self.registry,
        )
        self.identities = Gauge(
            "frp_gallery_identities", "Enrolled identities.", registry=self.registry
        )
        self.embeddings = Gauge(
            "frp_gallery_embeddings", "Stored embeddings.", registry=self.registry
        )

    def observe_stage(self, stage: str, seconds: float) -> None:
        self.stage_seconds.labels(stage=stage).observe(seconds)

    def render(self) -> bytes:
        return generate_latest(self.registry)
