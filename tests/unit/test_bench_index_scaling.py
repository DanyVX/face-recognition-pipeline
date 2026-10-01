from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from bench.index_scaling import (
    main,
    percentiles,
    planted_queries,
    recall_at,
    run_benchmark,
    unit_vectors,
)


def test_planted_queries_have_the_requested_cosine() -> None:
    rng = np.random.default_rng(0)
    gallery = unit_vectors(500, 256, rng)
    queries = planted_queries(gallery, 200, 0.6, rng)
    best = (queries @ gallery.T).max(axis=1)
    assert 0.5 < float(np.median(best)) < 0.7
    np.testing.assert_allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-5)


def test_recall_and_percentiles_helpers() -> None:
    a = np.array([[1, 2, 3], [4, 5, 6]])
    e = np.array([[1, 9, 3], [7, 8, 9]])
    assert recall_at(a, e, 1) == 0.5
    assert recall_at(a, e, 3) == pytest.approx((2 / 3 + 0) / 2)
    p = percentiles([0.001] * 99 + [0.1])
    assert p["p50_ms"] == pytest.approx(1.0) and p["p99_ms"] >= p["p95_ms"] >= p["p50_ms"]


def test_tiny_run_reports_all_backends_and_exact_search_is_exact() -> None:
    result = run_benchmark([300], dim=16, n_queries=20, latency_queries=10)
    backends = {entry["backend"]: entry for entry in result["sizes"]["300"]}
    assert set(backends) == {"flat", "hnsw", "ivf"}
    flat = backends["flat"]["sweep"][0]
    assert flat["recall@1_planted"] == 1.0 and flat["recall@10_independent"] == 1.0
    for entry in backends.values():
        assert entry["build_seconds"] >= 0 and entry["first_search_after_delete_seconds"] >= 0
        assert entry["sweep"][0]["latency_single_query_k10"]["p50_ms"] > 0


def test_main_writes_a_result_file_with_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bench.common as common
    import bench.index_scaling as module

    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    original = common.write_result
    monkeypatch.setattr(
        module,
        "write_result",
        lambda name, result, params: original(name, result, params, directory=tmp_path),
    )
    assert main(["--tiny", "--name", "smoke"]) == 0
    doc = json.loads((tmp_path / "smoke.json").read_text(encoding="utf-8"))
    assert doc["params"]["tiny"] is True
    assert {"commit", "dirty"} <= set(doc["env"]["git"])
    assert "200" in doc["result"]["sizes"]
