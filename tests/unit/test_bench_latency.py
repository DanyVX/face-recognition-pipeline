from __future__ import annotations

import io

import numpy as np
import pytest
from bench.latency import (
    embed_batches,
    make_1080p,
    percentile_stats,
    search_timings,
    stage_timings,
    time_each,
)
from PIL import Image

from frp.align import align_face
from frp.detect import Face
from frp.embed import ArcFaceEmbedder
from tests.conftest import FIXTURES
from tests.synthetic import synthetic_scene
from tests.unit.fake_arcface import FAKE_INFO, FakeArcFaceSession


def test_percentile_stats_are_in_milliseconds_and_ordered() -> None:
    stats = percentile_stats([0.001] * 98 + [0.010, 0.100])
    assert stats["p50_ms"] == pytest.approx(1.0)
    assert (
        stats["p50_ms"]
        <= stats["p95_ms"]
        <= stats["p99_ms"]
        <= stats["max_ms"]
        == pytest.approx(100.0)
    )
    assert stats["n"] == 100


def test_warmup_calls_run_but_are_not_recorded() -> None:
    calls: list[int] = []
    samples = time_each(calls.append, list(range(10)), warmup=3)
    assert len(samples) == 10  # every item timed once...
    assert calls == [0, 1, 2, *range(10)]  # ...and the warm-up items executed an extra time


class FixedDetector:
    def detect(self, image: np.ndarray) -> list[Face]:
        _, landmarks = synthetic_scene()
        return [
            Face(box=np.array([40, 20, 200, 180], dtype=np.float32), landmarks=landmarks, score=0.9)
        ]


def synthetic_png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(synthetic_scene()[0]).save(buf, format="PNG")
    return buf.getvalue()


def test_stage_timings_report_every_stage() -> None:
    embedder = ArcFaceEmbedder(FakeArcFaceSession(), FAKE_INFO)
    out = stage_timings([synthetic_png()] * 8, FixedDetector(), embedder, warmup=2)  # type: ignore[arg-type]
    assert out["n_images"] == out["n_with_face"] == 8
    for stage in ("decode", "detect", "align", "embed_single"):
        assert out[stage]["p50_ms"] > 0 and out[stage]["n"] == 8


def test_embed_batches_report_per_image_cost() -> None:
    embedder = ArcFaceEmbedder(FakeArcFaceSession(), FAKE_INFO, max_batch=8)
    crops = np.stack([align_face(*synthetic_scene()).image] * 4)
    out = embed_batches(embedder, crops, [1, 4, 16], repeats=3, warmup=1)
    assert set(out) == {"1", "4", "16"}
    assert out["16"]["per_image_p50_ms"] == pytest.approx(out["16"]["p50_ms"] / 16)


def test_search_timings_cover_each_gallery_size() -> None:
    out = search_timings(FAKE_INFO, [100, 300], n_queries=10, warmup=2)
    assert set(out) == {"100", "300"} and out["300"]["n"] == 10


def test_the_1080p_scenario_image_has_the_advertised_size() -> None:
    img = Image.open(io.BytesIO(make_1080p(FIXTURES / "astronaut.jpg")))
    assert img.size == (1920, 1080) and img.format == "JPEG"
