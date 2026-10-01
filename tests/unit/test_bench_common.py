from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from bench.common import load_result, to_jsonable, write_json_atomic, write_result


def test_result_file_contains_env_commit_params_and_result(tmp_path: Path) -> None:
    path = write_result("demo", {"value": np.float32(1.5)}, {"seed": 0}, directory=tmp_path)
    assert path == tmp_path / "demo.json"
    doc = load_result("demo", directory=tmp_path)
    assert doc["name"] == "demo"
    assert doc["params"] == {"seed": 0}
    assert doc["result"] == {"value": 1.5}
    assert {"python", "platform", "ram_total_gb", "git", "packages"} <= set(doc["env"])
    assert {"commit", "dirty"} <= set(doc["env"]["git"])


def test_numpy_and_paths_are_converted_to_plain_json() -> None:
    value = {"a": np.arange(3), "b": np.int64(4), "c": Path("x/y"), "d": (np.float64(0.5), [1])}
    assert to_jsonable(value) == {"a": [0, 1, 2], "b": 4, "c": "x/y", "d": [0.5, [1]]}
    json.dumps(to_jsonable(value))


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    write_json_atomic(target, {"ok": 1})
    write_json_atomic(target, {"ok": 2})  # overwrite
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_failed_write_keeps_the_previous_file_and_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    write_json_atomic(target, {"ok": 1})

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": Unserializable()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]
