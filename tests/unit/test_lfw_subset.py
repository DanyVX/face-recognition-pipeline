from __future__ import annotations

import numpy as np
from bench.lfw_data import ResolvedPairs, drop_unprepared

from frp.lfw import LfwPair, pair_image_paths, relative_image, subset_pairs


def make_pairs() -> list[LfwPair]:
    pairs: list[LfwPair] = []
    for fold in range(3):
        for i in range(5):
            pairs.append(LfwPair(fold, True, f"P{fold}{i}", 1, f"P{fold}{i}", 2))
        for i in range(5):
            pairs.append(LfwPair(fold, False, f"P{fold}{i}", 1, f"Q{fold}{i}", 3))
    return pairs


def test_relative_image_uses_forward_slashes_and_zero_padding() -> None:
    assert relative_image("Abel_Pacheco", 4) == "Abel_Pacheco/Abel_Pacheco_0004.jpg"


def test_subset_takes_the_first_n_of_each_class_in_each_fold() -> None:
    subset = subset_pairs(make_pairs(), 2)
    assert len(subset) == 3 * 2 * 2
    for fold in range(3):
        assert sum(p.fold == fold and p.same for p in subset) == 2
        assert sum(p.fold == fold and not p.same for p in subset) == 2
    assert subset[0].name_a == "P00"  # original order is preserved
    assert subset_pairs(make_pairs(), 100) == make_pairs()  # asking for more returns everything


def test_pair_image_paths_are_unique_and_sorted() -> None:
    paths = pair_image_paths(subset_pairs(make_pairs(), 1))
    assert paths == sorted(set(paths))
    assert "P00/P00_0001.jpg" in paths and "P00/P00_0002.jpg" in paths
    assert "Q00/Q00_0003.jpg" in paths


def test_unprepared_pairs_are_excluded_not_counted_as_failures() -> None:
    pairs = make_pairs()[:4]
    resolved = ResolvedPairs(
        pairs=pairs,
        row_a=np.array([0, 1, -1, 3]),
        row_b=np.array([5, -1, 7, 8]),
        same=np.array([p.same for p in pairs]),
        fold=np.array([p.fold for p in pairs]),
    )
    kept = drop_unprepared(resolved)
    assert len(kept) == 2 and kept.row_a.tolist() == [0, 3] and kept.row_b.tolist() == [5, 8]
    assert kept.pairs == [pairs[0], pairs[3]]
