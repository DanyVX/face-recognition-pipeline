from __future__ import annotations

import json
from pathlib import Path

from frp.envinfo import collect_env, git_info


def test_collect_env_is_json_serializable_and_complete() -> None:
    info = collect_env()
    json.dumps(info)  # must not raise
    for key in ("timestamp_utc", "python", "platform", "cpu_count_logical", "ram_total_gb", "git"):
        assert key in info
    assert info["ram_total_gb"] > 0


def test_unknown_packages_are_recorded_as_none_not_dropped() -> None:
    packages = collect_env()["packages"]
    assert "faiss-cpu" in packages  # tracked even when not installed (M0 has no ML deps)


def test_git_info_outside_a_repo_reports_a_reason(tmp_path: Path) -> None:
    info = git_info(tmp_path)
    assert info["commit"] is None
    assert info["reason"]
