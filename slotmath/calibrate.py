"""Calibration helpers: steer reel-strip composition towards RTP / volatility targets.

Calibration: the paytable and feature rules are fixed by design, and the 
*symbol counts on each reel strip* are the dials that set RTP, hit frequency
and volatility. We use a greedy local search over single-symbol swaps, scored
with cheap exact means (symbol-tuple enumeration) plus a Monte-Carlo estimate
of the free-spin per-spin histogram. The final numbers are then recomputed
exactly by full enumeration.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .engine import count_symbol, evaluate_lines, random_stops, windows_from_stops
from .exact import GridEnumeration


def mc_joint(reels: List[np.ndarray], rows: int, paytable: np.ndarray, paylines: np.ndarray,
             wilds: Sequence[int], scatter: int, bin_width: float, n: int = 2_000_000,
             seed: int = 1, symbol_mult: Optional[np.ndarray] = None, max_bins: int = 200_000) -> GridEnumeration:
    """Monte-Carlo stand-in for exact.enumerate_grid (same output format), for calibration loops."""
    rng = np.random.default_rng(seed)
    max_sc = rows * len(reels)
    hist = np.zeros((max_sc + 1) * max_bins)
    done = 0; mx = 0.0
    while done < n:
        b = min(250_000, n - done)
        stops = random_stops(reels, b, rng)
        win = windows_from_stops(reels, stops, rows)
        lw = evaluate_lines(win, paytable, paylines, wilds, symbol_mult)
        sc = count_symbol(win, scatter)
        idx = sc * max_bins + np.rint(lw / bin_width).astype(np.int64)
        hist += np.bincount(idx, minlength=hist.size)
        mx = max(mx, float(lw.max())); done += b
    joint = hist.reshape(max_sc + 1, max_bins) / n
    last = np.nonzero(joint.sum(axis=0) > 0)[0].max()
    return GridEnumeration(joint=joint[:, : last + 1], bin_width=bin_width, n_combinations=n, max_line_win=mx, extra={})


def local_search(counts: List[Dict[str, int]], score_fn: Callable[[List[Dict[str, int]]], float],
                 allowed: Callable[[int, str, str, List[Dict[str, int]]], bool], symbols: Sequence[str],
                 iters: int = 300, seed: int = 0, verbose: bool = True) -> List[Dict[str, int]]:
    """Greedy local search: move one symbol copy on one reel to another symbol if it improves the score.

    allowed(reel, from_symbol, to_symbol, counts) -> bool enforces design constraints
    (e.g. wilds only on reels 2-4, exactly one scatter per reel, strip length bounds).
    """
    rng = np.random.default_rng(seed)
    best = [dict(c) for c in counts]
    best_score = score_fn(best)
    for it in range(iters):
        i = int(rng.integers(len(best)))
        present = [s for s, c in best[i].items() if c > 0]
        a = present[int(rng.integers(len(present)))]
        b = symbols[int(rng.integers(len(symbols)))]
        if a == b or not allowed(i, a, b, best):
            continue
        cand = [dict(c) for c in best]
        cand[i][a] -= 1
        cand[i][b] = cand[i].get(b, 0) + 1
        sc = score_fn(cand)
        if sc < best_score:
            best, best_score = cand, sc
            if verbose:
                print(f"  iter {it}: reel {i+1} {a}->{b}  score {sc:.6f}")
    return best
