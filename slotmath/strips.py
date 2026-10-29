"""Reel-strip construction helpers.

A strip is fully described by its symbol counts; the ORDER only matters for
things that look at adjacent positions (scatter counting within one reel,
stacked symbols). `spread_strip` lays symbols out so that copies of the same
symbol are as far apart as possible (no accidental stacks), which keeps the
per-reel scatter-count distribution close to "at most one scatter visible".
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def spread_strip(counts: Dict[int, int], seed: int = 0) -> np.ndarray:
    """Build a strip from {symbol_id: count} spreading equal symbols evenly.

    Deterministic: the symbol with most copies is placed first on an evenly
    spaced lattice, then the next symbol fills the free slots that are
    farthest from each other, and so on. `seed` only rotates the result.
    """
    L = sum(counts.values())
    strip = -np.ones(L, dtype=np.int64)
    order = sorted(counts.items(), key=lambda kv: -kv[1])
    for sym, c in order:
        free = np.nonzero(strip < 0)[0]
        # choose c free slots as evenly spaced as possible
        picks = np.unique(np.round(np.linspace(0, len(free), c, endpoint=False)).astype(int))
        while len(picks) < c:                     # (rounding collisions)
            extra = np.setdiff1d(np.arange(len(free)), picks)[:c - len(picks)]
            picks = np.sort(np.concatenate([picks, extra]))
        strip[free[picks]] = sym
    assert (strip >= 0).all()
    return np.roll(strip, seed % L)


def strip_from_names(names: Sequence[str], symbols: List[str]) -> np.ndarray:
    return np.array([symbols.index(n) for n in names], dtype=np.int64)


def strip_to_names(strip: np.ndarray, symbols: List[str]) -> List[str]:
    return [symbols[int(s)] for s in strip]


def counts_from_strip(strip: np.ndarray, n_symbols: int) -> np.ndarray:
    return np.bincount(strip, minlength=n_symbols)
