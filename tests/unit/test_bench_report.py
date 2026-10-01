"""The report renders only what is in the result files and marks everything else TBD."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bench.lfw_verification import leakage_report, verification_report
from bench.open_set import run_once, summarize
from bench.report import (
    README_END,
    README_START,
    TBD,
    Docs,
    build,
    get,
    key_results_table,
    load_results,
    ms,
    pct,
    plot_roc,
    provenance,
    results_markdown,
    update_readme,
)

from frp.embed import ModelInfo
from frp.lfw import LfwPair

ENV = {
    "timestamp_utc": "2026-10-02T00:00:00+00:00",
    "git": {"commit": "abcdef1234567890", "dirty": False},
}


def empty_docs() -> Docs:
    return dict.fromkeys(load_results(Path("nonexistent-dir")))


def verification_doc() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n = 600
    same = np.tile([True, False], n // 2)
    scores = np.where(same, rng.normal(0.65, 0.12, n), rng.normal(0.05, 0.12, n))
    fold = np.repeat(np.arange(10), n // 10)
    result = verification_report(scores, same, fold)
    pairs = [LfwPair(0, True, "A", 1, "A", 2), LfwPair(1, True, "B", 1, "B", 2)]
    result["leakage_check"] = leakage_report(pairs, 10)
    result["model"] = "synthetic/model@abc"
    result["detection"] = {
        "images_total": 1000,
        "images_failed": 7,
        "image_status_counts": {},
        "multi_face_policy": "center",
    }
    return {"name": "lfw_verification", "env": ENV, "params": {}, "result": result}


def open_set_doc() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    rows, names = [], []
    for p in range(40):
        v = rng.normal(size=32) + 0.4 * rng.normal(size=(4, 32))
        rows.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        names += [f"p{p}"] * 4
    emb = np.concatenate(rows).astype(np.float32)
    info = ModelInfo("s", "9" * 64, "none", 32)
    runs = [run_once(emb, np.ones(len(names), bool), names, info, seed=s) for s in range(2)]
    experiments = {"enroll1_max": {"summary": summarize(runs), "runs": runs}}
    return {
        "name": "open_set",
        "env": ENV,
        "params": {},
        "result": {"model": "x", "experiments": experiments},
    }


# --- formatting and provenance -------------------------------------------------------------------


def test_formatters_and_missing_values() -> None:
    assert pct(0.99857) == "99.86%" and pct(None) == TBD
    assert ms(12.345) == "12.3 ms" and ms(None) == TBD
    assert get({"a": {"b": [10, 20]}}, "a", "b", 1) == 20
    assert get({"a": 1}, "a", "x") is None and get(None, "a") is None


def test_provenance_shows_commit_and_dirty_flag() -> None:
    assert provenance({"env": ENV}).startswith("`abcdef12`")
    dirty = {"env": {**ENV, "git": {"commit": "abc", "dirty": True}}}
    assert "uncommitted" in provenance(dirty)
    assert provenance(None) == TBD


# --- everything missing --------------------------------------------------------------------------


def test_with_no_results_every_value_is_tbd_and_nothing_is_invented() -> None:
    docs = empty_docs()
    table = key_results_table(docs)
    body_rows = table.splitlines()[2:]
    assert body_rows and all(TBD in row for row in body_rows)
    text = results_markdown(docs)
    assert text.count(TBD) >= 8
    digits_outside_headings = [
        line
        for line in text.splitlines()
        if any(ch.isdigit() for ch in line) and not line.startswith(("#", "|", "Generated", "file"))
    ]
    assert not [line for line in digits_outside_headings if "%" in line or " ms" in line]


def test_build_with_no_results_writes_markdown_but_no_plots(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n", encoding="utf-8")
    written = build(empty_docs(), docs_dir=tmp_path / "docs", readme=readme)
    assert {p.name for p in written} == {"RESULTS.md", "README.md"}
    assert not (tmp_path / "docs" / "img").exists()
    assert TBD in readme.read_text(encoding="utf-8")


# --- with results --------------------------------------------------------------------------------


def test_table_numbers_are_exactly_the_ones_in_the_result_file() -> None:
    docs = empty_docs()
    ver, ops = verification_doc(), open_set_doc()
    docs["lfw_verification"], docs["open_set"] = ver, ops
    table = key_results_table(docs)
    r = ver["result"]
    assert f"{100 * r['accuracy_mean']:.2f}%" in table
    assert f"{100 * r['accuracy_std']:.2f}%" in table
    assert "7 of 1000 images (0.70%)" in table
    summary = ops["result"]["experiments"]["enroll1_max"]["summary"]
    assert pct(summary["rank1_closed_set"]["mean"]) in table
    assert pct(summary["fnir_at_fpir"]["0.01"]["mean"]) in table
    assert "GPU latency | " + TBD in table  # latency was not run


def test_unresolvable_far_is_flagged_not_reported_as_a_number() -> None:
    docs = empty_docs()
    ver = verification_doc()
    docs["lfw_verification"] = ver
    text = results_markdown(docs)
    floor = ver["result"]["tar_at_far_standard_protocol"]["far_resolution_floor"]
    assert f"{floor:.1e}" in text  # 1/n_impostor is quoted as the resolution limit
    assert (
        "NOT** part of the standard protocol" in text
        or "not** part of the standard protocol" in text.lower()
    )


def test_roc_plot_is_written(tmp_path: Path) -> None:
    docs = empty_docs()
    docs["lfw_verification"] = verification_doc()
    path = plot_roc(docs, tmp_path / "roc.png")
    assert path is not None and path.stat().st_size > 5000
    assert plot_roc(empty_docs(), tmp_path / "none.png") is None


# --- README markers ------------------------------------------------------------------------------


def test_readme_block_is_replaced_in_place(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{README_START}\nold table\n{README_END}\nafter\n", encoding="utf-8")
    assert update_readme(readme, "NEW TABLE") is True
    assert (
        readme.read_text(encoding="utf-8")
        == f"before\n{README_START}\nNEW TABLE\n{README_END}\nafter\n"
    )
    assert update_readme(readme, "NEW TABLE") is False  # idempotent


def test_readme_without_markers_is_an_error(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        update_readme(readme, "x")


def test_the_repository_readme_has_the_markers() -> None:
    text = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    assert README_START in text and README_END in text
