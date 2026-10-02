from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from bench.precision import agreement, convert_to_fp16, decision_flips
from onnx import TensorProto, helper, numpy_helper


def tiny_model(path: Path) -> Path:
    """x[N,4] @ W[4,3] + b, float32."""
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(rng.normal(size=(4, 3)).astype(np.float32), "W")
    bias = numpy_helper.from_array(rng.normal(size=(3,)).astype(np.float32), "b")
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["x", "W", "b"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])],
        [weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_fp16_conversion_runs_and_stays_close_to_fp32(tmp_path: Path) -> None:
    src = tiny_model(tmp_path / "fp32.onnx")
    dst = convert_to_fp16(src, tmp_path / "sub" / "fp16.onnx")
    assert dst.is_file()
    x = np.random.default_rng(1).normal(size=(5, 4)).astype(np.float32)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    y32 = ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"]).run(
        None, {"x": x}
    )[0]
    session16 = ort.InferenceSession(str(dst), opts, providers=["CPUExecutionProvider"])
    y16 = session16.run(None, {"x": x})[0]
    assert y16.dtype == np.float32 and y16.shape == y32.shape  # float32 inputs/outputs preserved
    np.testing.assert_allclose(y16, y32, rtol=2e-2, atol=2e-2)
    assert not np.array_equal(y16, y32)  # it really is a lower-precision model


def test_agreement_statistics() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(50, 16)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = a + 1e-3 * rng.normal(size=a.shape).astype(np.float32)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    usable = np.ones(50, dtype=bool)
    usable[0] = False
    stats = agreement(a, b, usable)
    assert (
        stats["n"] == 49
        and stats["cosine_min"] > 0.999
        and stats["cosine_mean"] >= stats["cosine_min"]
    )
    assert stats["max_abs_element_diff"] >= stats["mean_abs_element_diff"] > 0
    assert agreement(a, a, usable)["cosine_min"] == pytest.approx(1.0, abs=1e-6)


def test_decision_flips_count_only_pairs_crossing_the_threshold() -> None:
    a = np.array([0.30, 0.50, 0.70, np.nan])
    b = np.array([0.31, 0.49, 0.71, 0.9])
    flips = decision_flips(a, b, 0.5)
    assert flips["pairs_compared"] == 3 and flips["pairs_flipped"] == 1  # 0.50 -> 0.49
    assert flips["max_abs_score_diff"] == pytest.approx(0.01)
