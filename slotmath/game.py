"""Game configuration objects.

Conventions used throughout the package
---------------------------------------
* Every win is expressed in multiples of the TOTAL BET (e.g. 2.5 means 2.5 x bet).
  Modern online slots are specified this way; it keeps the maths independent of
  coin sizes and line counts.
* A reel strip is a 1-D integer array of symbol ids. A spin picks one uniformly
  random *stop* per reel; the visible window is the `rows` consecutive strip
  positions starting at the stop (wrapping around).
* Paylines are given as one row index per reel (0 = top row).
* Line wins pay left-to-right only; the wild substitutes for every symbol except
  the scatter; a line pays the better of (a) the run of the first non-wild
  symbol and (b) the run of pure wilds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# The classic 20-line pattern set for a 5x3 grid (row index per reel).
PAYLINES_20 = np.array(
    [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2],
        [0, 1, 2, 1, 0],
        [2, 1, 0, 1, 2],
        [0, 0, 1, 0, 0],
        [2, 2, 1, 2, 2],
        [1, 0, 0, 0, 1],
        [1, 2, 2, 2, 1],
        [0, 1, 1, 1, 0],
        [2, 1, 1, 1, 2],
        [1, 0, 1, 0, 1],
        [1, 2, 1, 2, 1],
        [0, 1, 0, 1, 0],
        [2, 1, 2, 1, 2],
        [1, 1, 0, 1, 1],
        [1, 1, 2, 1, 1],
        [0, 0, 2, 0, 0],
        [2, 2, 0, 2, 2],
        [0, 2, 0, 2, 0],
    ],
    dtype=np.int64,
)

# The classic 10-line pattern set ("Book of ..." games).
PAYLINES_10 = PAYLINES_20[:10].copy()


@dataclass
class GameConfig:
    """Everything the engine needs to spin and evaluate one game."""

    name: str
    symbols: List[str]                      # index == symbol id
    paytable: np.ndarray                    # shape (n_symbols, 6): pay for 0..5 of a kind (x bet)
    paylines: np.ndarray                    # shape (n_lines, n_reels)
    base_reels: List[np.ndarray]            # one strip per reel (symbol ids)
    rows: int = 3
    wild: Optional[int] = None
    scatter: Optional[int] = None
    scatter_pays: Dict[int, float] = field(default_factory=dict)   # count -> pay (x bet), base game
    fs_reels: Optional[List[np.ndarray]] = None                    # strips used in free spins
    feature: Dict = field(default_factory=dict)                    # free-spins rules (game specific)
    win_cap: Optional[float] = None                                # max win per spin incl. feature (x bet)
    bin_width: float = 0.1                                         # resolution of exact distributions (x bet)

    # ----- convenience -------------------------------------------------
    @property
    def n_reels(self) -> int:
        return len(self.base_reels)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_lines(self) -> int:
        return int(self.paylines.shape[0])

    def symbol_id(self, name: str) -> int:
        return self.symbols.index(name)

    def validate(self) -> None:
        assert self.paytable.shape == (self.n_symbols, 6), "paytable must be (n_symbols, 6)"
        assert self.paylines.shape[1] == self.n_reels
        assert (self.paylines < self.rows).all() and (self.paylines >= 0).all()
        for r in self.base_reels:
            assert r.min() >= 0 and r.max() < self.n_symbols
        if self.fs_reels is not None:
            assert len(self.fs_reels) == self.n_reels
        # all pays must sit on the bin grid so that exact distributions are exact
        for v in np.concatenate([self.paytable.ravel(), np.array(list(self.scatter_pays.values()) + [0.0])]):
            assert abs(round(v / self.bin_width) * self.bin_width - v) < 1e-9, f"pay {v} not a multiple of bin_width"

    def strips_table(self, reels: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Symbol counts per reel, shape (n_reels, n_symbols). Used by PAR sheets."""
        reels = self.base_reels if reels is None else reels
        out = np.zeros((self.n_reels, self.n_symbols), dtype=np.int64)
        for i, strip in enumerate(reels):
            out[i] = np.bincount(strip, minlength=self.n_symbols)
        return out
