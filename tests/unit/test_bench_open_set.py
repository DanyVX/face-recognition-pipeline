"""Open-set protocol on synthetic embeddings, run through the real Gallery service."""

from __future__ import annotations

import numpy as np
import pytest
from bench.open_set import MISS_SCORE, run_once, summarize

from frp.embed import ModelInfo
from frp.gallery import Aggregation
from frp.index import Backend

INFO = ModelInfo("synthetic/blobs", "9" * 64, "none", 32)


def population(
    n_people: int = 60, per_person: int = 4, singletons: int = 20, seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    rows, names = [], []
    for p in range(n_people + singletons):
        center = rng.normal(size=32)
        count = per_person if p < n_people else 1
        v = center + 0.35 * rng.normal(size=(count, 32))
        rows.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        names += [f"person_{p:03d}"] * count
    return np.concatenate(rows).astype(np.float32), names


def test_protocol_counts_and_basic_sanity() -> None:
    emb, names = population()
    usable = np.ones(len(names), dtype=bool)
    r = run_once(emb, usable, names, INFO, seed=0)
    assert r["n_enrolled_identities"] == 30  # half of the 60 eligible people
    assert r["n_mated_probes"] == 30 * 3  # each enrolled person keeps 3 non-enrollment images
    assert r["n_nonmated_probes"] == 30 * 4 + 20  # unenrolled people plus the 20 singletons
    assert r["n_mated_failed"] == r["n_nonmated_failed"] == r["n_enrollment_failures"] == 0
    assert r["rank1_closed_set"] > 0.95  # easy synthetic identities
    assert 0.0 <= r["fnir_at_fpir"]["0.1"]["fnir"] <= 1.0
    # A tighter false-positive budget cannot give a lower miss rate.
    assert r["fnir_at_fpir"]["0.01"]["fnir"] >= r["fnir_at_fpir"]["0.1"]["fnir"] - 1e-12


def test_splits_are_reproducible_per_seed_and_differ_across_seeds() -> None:
    emb, names = population()
    usable = np.ones(len(names), dtype=bool)
    a = run_once(emb, usable, names, INFO, seed=3)
    b = run_once(emb, usable, names, INFO, seed=3)
    c = run_once(emb, usable, names, INFO, seed=4)
    assert a["fnir_at_fpir"] == b["fnir_at_fpir"] and a["rank1_closed_set"] == b["rank1_closed_set"]
    assert a["enrolled_digest"] == b["enrolled_digest"] != c["enrolled_digest"]


def test_failed_mated_probes_are_misses_and_failed_strangers_are_excluded() -> None:
    emb, names = population()
    usable = np.ones(len(names), dtype=bool)
    clean = run_once(emb, usable, names, INFO, seed=1)
    # Break every image of two people, then everything that is a probe of theirs follows.
    broken = usable.copy()
    for i, name in enumerate(names):
        if name in ("person_005", "person_006", "person_007"):
            broken[i] = False
    r = run_once(emb, broken, names, INFO, seed=1)
    assert r["n_mated_failed"] + r["n_nonmated_failed"] + r["n_enrollment_failures"] > 0
    if r["n_mated_failed"]:
        assert (
            r["rank1_failures_as_misses"] < r["rank1_closed_set"]
        )  # failures hurt the honest number
    assert (
        r["n_nonmated_probes"] <= clean["n_nonmated_probes"]
    )  # failed strangers leave the FPIR base
    assert MISS_SCORE < -1.0  # a miss can never reach any real threshold


def test_closed_set_rank1_hides_open_set_failures() -> None:
    """Look-alike identities: strangers resemble enrolled people, so FNIR at low FPIR is high."""
    rng = np.random.default_rng(5)
    base = rng.normal(size=(20, 32))
    rows, names = [], []
    for p in range(40):  # people 2i and 2i+1 are look-alikes of each other
        center = base[p // 2] + 0.15 * rng.normal(size=32)
        v = center + 0.3 * rng.normal(size=(4, 32))
        rows.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        names += [f"p{p:02d}"] * 4
    emb = np.concatenate(rows).astype(np.float32)
    r = run_once(emb, np.ones(len(names), dtype=bool), names, INFO, seed=0)
    # Measured for this construction (seeds 0-3): closed-set rank-1 is 0.80-0.87, yet at a 1% false
    # positive budget 90-95% of enrolled people are no longer recognised.
    assert r["rank1_closed_set"] > 0.8
    assert r["fnir_at_fpir"]["0.01"]["fnir"] > 0.8
    assert r["fnir_at_fpir"]["0.01"]["fnir"] >= r["fnir_at_fpir"]["0.1"]["fnir"]


@pytest.mark.parametrize("aggregation", [Aggregation.MAX, Aggregation.MEAN])
def test_multi_image_enrollment_and_aggregation_policies(aggregation: Aggregation) -> None:
    emb, names = population(per_person=6)
    r = run_once(
        emb,
        np.ones(len(names), dtype=bool),
        names,
        INFO,
        seed=0,
        n_enroll=3,
        aggregation=aggregation,
    )
    assert r["n_enrolled_identities"] == 30 and r["n_mated_probes"] == 30 * 3
    assert r["aggregation"] == aggregation.value and r["rank1_closed_set"] > 0.9


def test_hnsw_backend_runs_through_the_same_protocol() -> None:
    emb, names = population()
    r = run_once(emb, np.ones(len(names), dtype=bool), names, INFO, seed=0, backend=Backend.HNSW)
    assert r["backend"] == "hnsw" and r["rank1_closed_set"] > 0.9


def test_summary_aggregates_runs() -> None:
    emb, names = population()
    usable = np.ones(len(names), dtype=bool)
    runs = [run_once(emb, usable, names, INFO, seed=s) for s in range(3)]
    s = summarize(runs)
    assert s["n_runs"] == 3 and s["rank1_closed_set"]["std"] >= 0
    assert set(s["fnir_at_fpir"]) == {"0.1", "0.01", "0.001"}
    assert s["fnir_at_fpir"]["0.1"]["resolvable_in_all_runs"] is True
    assert s["fnir_at_fpir"]["0.001"]["resolvable_in_all_runs"] is False  # only ~140 strangers
