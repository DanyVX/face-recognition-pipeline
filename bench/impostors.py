"""Cross-identity (impostor) score distributions from *all* image pairs, via fine histograms.

The standard LFW protocol has only 3000 impostor pairs, so FAR below about 3.3e-4 cannot be
measured. This module builds the impostor distribution from every pair of images of different
people (about 87 million for LFW) without storing the scores: a 200 000-bin histogram on [-1, 1]
(bin width 1e-5) is accumulated block by block.

Identities are grouped (``groups``) so that a fold's held-out identities and the training
identities can be separated: histograms are kept per pair of groups, and a fold's training pool is
the sum of every group pair that excludes the held-out group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

BINS = 200_000
LO, HI = -1.0, 1.0


@dataclass
class ScoreHistogram:
    counts: NDArray[np.int64] = field(default_factory=lambda: np.zeros(BINS, dtype=np.int64))

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    @staticmethod
    def _bins(scores: NDArray[np.floating]) -> NDArray[np.intp]:
        idx = np.floor((np.asarray(scores, dtype=np.float64) - LO) / (HI - LO) * BINS)
        return np.clip(idx, 0, BINS - 1).astype(np.intp)

    def add(self, scores: NDArray[np.floating]) -> None:
        if scores.size:
            self.counts += np.bincount(self._bins(scores), minlength=BINS)

    def __iadd__(self, other: ScoreHistogram) -> ScoreHistogram:
        self.counts += other.counts
        return self

    def edge(self, bin_index: int) -> float:
        return LO + bin_index * (HI - LO) / BINS

    def threshold_at_far(self, far: float) -> float:
        """Lowest bin edge ``t`` such that at most ``far`` of the scores are ``>= t``."""
        if not 0.0 < far <= 1.0:
            raise ValueError("far must be in (0, 1]")
        total = self.total
        if total == 0:
            raise ValueError("empty histogram")
        allowed = int(np.floor(far * total))
        above = np.cumsum(self.counts[::-1])[::-1]  # above[b] = scores in bins >= b
        ok = np.flatnonzero(above <= allowed)
        # ``above`` is non-increasing, so ``ok`` is a suffix; its first entry is the lowest bin.
        return self.edge(int(ok[0])) if ok.size else self.edge(BINS)

    def far_at(self, threshold: float) -> float:
        """Fraction of scores ``>= threshold`` (exact for thresholds that are bin edges)."""
        total = self.total
        if total == 0:
            raise ValueError("empty histogram")
        b = int(self._bins(np.array([threshold]))[0])
        if threshold >= HI:
            return 0.0
        return float(self.counts[b:].sum() / total)


def cross_identity_histograms(
    embeddings: NDArray[np.float32],
    identity_ids: NDArray[np.int64],
    groups: NDArray[np.int64],
    *,
    block: int = 1000,
) -> dict[tuple[int, int], ScoreHistogram]:
    """Histogram of cosine scores of different-identity image pairs, per (group, group) pair.

    Only rows with finite embeddings are used. ``identity_ids[i]`` identifies the person of row
    ``i``; ``groups[i]`` is the fold group of that person (``-1`` rows are ignored).
    Each unordered pair of images is counted once.
    """
    usable = np.isfinite(embeddings).all(axis=1) & (groups >= 0)
    group_values = sorted({int(g) for g in groups[usable]})
    rows_of = {g: np.flatnonzero(usable & (groups == g)) for g in group_values}
    result: dict[tuple[int, int], ScoreHistogram] = {}
    for gi, g1 in enumerate(group_values):
        for g2 in group_values[gi:]:
            hist = ScoreHistogram()
            r1, r2 = rows_of[g1], rows_of[g2]
            e2 = embeddings[r2]
            id2 = identity_ids[r2]
            for start in range(0, r1.size, block):
                rows = r1[start : start + block]
                sims = embeddings[rows] @ e2.T
                keep = identity_ids[rows][:, None] != id2[None, :]
                if g1 == g2:
                    keep &= rows[:, None] < r2[None, :]  # each unordered pair once
                hist.add(sims[keep])
            result[(g1, g2)] = hist
    return result


def pool(histograms: dict[tuple[int, int], ScoreHistogram], include: set[int]) -> ScoreHistogram:
    """Sum of every group-pair histogram whose two groups are both in ``include``."""
    merged = ScoreHistogram()
    for (g1, g2), hist in histograms.items():
        if g1 in include and g2 in include:
            merged += hist
    return merged
