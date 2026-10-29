"""Vectorised reel engine (NumPy).

The same evaluation routine is used by the exact analysis (fed with *all* stop
combinations / symbol tuples) and by the Monte-Carlo simulator (fed with random
stops). Sharing one routine is what makes "exact vs simulated" a genuine check
of the model instead of a comparison of two independent implementations.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def windows_from_stops(reels: List[np.ndarray], stops: np.ndarray, rows: int) -> np.ndarray:
    """Visible windows from stop positions.

    reels : list of strips (one per reel), symbol ids
    stops : int array (n, n_reels)
    returns symbol array (n, rows, n_reels), dtype int16
    """
    n, n_reels = stops.shape
    out = np.empty((n, rows, n_reels), dtype=np.int16)
    offs = np.arange(rows)[None, :]
    for i, strip in enumerate(reels):
        idx = (stops[:, i:i + 1] + offs) % len(strip)
        out[:, :, i] = strip[idx]
    return out


def random_stops(reels: List[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """One uniformly random stop per reel - this *is* the RNG model of a reel slot."""
    return np.stack([rng.integers(0, len(strip), size=n) for strip in reels], axis=1)


def _wild_lut(n_symbols: int, wilds: Optional[Sequence[int]]) -> np.ndarray:
    lut = np.zeros(n_symbols, dtype=bool)
    if wilds is not None:
        lut[list(wilds)] = True
    return lut


def line_outcome(
    syms: np.ndarray,
    paytable: np.ndarray,
    wilds: Optional[Sequence[int]] = None,
    symbol_mult: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ONE payline for many spins.

    syms        : (n, n_reels) symbols on the line, left to right
    paytable    : (n_symbols, 6) pays for 0..5 of a kind (x bet)
    wilds       : ids of wild symbols (all variants substitute identically; the
                  pure-wild run pays the paytable row of wilds[0])
    symbol_mult : optional (n_symbols,) multiplier carried by each symbol (1 = none).
                  A winning line is multiplied by the SUM of carried multipliers (>1)
                  on its winning positions, or by 1 if none.
    returns (win, cand, run): win x bet, paying symbol id, number of paying positions
    """
    n, n_reels = syms.shape
    if wilds is None or len(wilds) == 0:
        cand = syms[:, 0]
        match = syms == cand[:, None]
        run = np.cumprod(match, axis=1).sum(axis=1)
        win = paytable[cand, run]
    else:
        lut = _wild_lut(paytable.shape[0], wilds)
        is_wild = lut[syms]
        first_nw = np.argmax(~is_wild, axis=1)            # index of first non-wild (0 if none)
        cand = syms[np.arange(n), first_nw]
        all_wild = is_wild.all(axis=1)
        cand = np.where(all_wild, wilds[0], cand)
        match = (syms == cand[:, None]) | is_wild
        run_c = np.cumprod(match, axis=1).sum(axis=1)
        win_c = paytable[cand, run_c]
        run_w = np.cumprod(is_wild, axis=1).sum(axis=1)
        win_w = paytable[wilds[0], run_w]
        use_w = win_w > win_c
        win = np.where(use_w, win_w, win_c)
        run = np.where(use_w, run_w, run_c)
        cand = np.where(use_w, wilds[0], cand)
    if symbol_mult is not None:
        m = symbol_mult[syms]                                   # (n, n_reels)
        on_win = np.arange(n_reels)[None, :] < run[:, None]
        msum = np.where(on_win & (m > 1), m, 0).sum(axis=1)
        win = win * np.where(msum > 0, msum, 1)
    return win, cand, run


def evaluate_lines(
    window: np.ndarray,
    paytable: np.ndarray,
    paylines: np.ndarray,
    wilds: Optional[Sequence[int]] = None,
    symbol_mult: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Total line win per spin (x bet) over all paylines."""
    n, rows, n_reels = window.shape
    ar = np.arange(n_reels)
    total = np.zeros(n, dtype=np.float64)
    for line in paylines:
        syms = window[:, line, ar]
        win, _, _ = line_outcome(syms, paytable, wilds, symbol_mult)
        total += win
    return total


def count_symbol(window: np.ndarray, symbol: int) -> np.ndarray:
    """Occurrences of `symbol` anywhere in each window (scatter counting)."""
    return (window == symbol).sum(axis=(1, 2))


def reels_containing(window: np.ndarray, symbol: int) -> np.ndarray:
    """Number of reels whose window contains `symbol` (expanding-symbol logic)."""
    return (window == symbol).any(axis=1).sum(axis=1)
