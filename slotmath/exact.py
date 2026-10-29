"""Exact analysis of a base-game spin.

Three levels of exactness are used, each as cheap as the question allows:

1. Line expectation (RTP contribution of paylines).
   Every row of every reel shows a *uniformly random strip position*, so the
   symbol on any one line position has the marginal distribution
   count(symbol on reel) / strip length, independently across reels.
   The expected win of ONE line is therefore a finite sum over all symbol
   5-tuples (n_symbols**5 terms) and every line has the same distribution.
   => line RTP = n_lines * E[single line win]. Cheap and exact.

2. Scatter count distribution.
   Scatters are counted anywhere in the window. Within a reel the rows are
   *not* independent (they are consecutive strip positions), so we take the
   exact per-reel distribution of "scatters visible" by sliding the window
   over the strip, then convolve the five reels (reels are independent).

3. Joint distribution of (scatter count, total line win) for the whole grid.
   Lines overlap, so the *distribution* of the total is not a product; we
   enumerate every stop combination (prod of strip lengths, ~3e7 for 5 reels
   of ~32 stops) in vectorised chunks. All combinations are equiprobable, so
   a histogram of outcomes is the exact distribution. This is exactly what a
   PAR sheet means by "total combinations".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .engine import count_symbol, line_outcome, windows_from_stops, evaluate_lines


# --------------------------------------------------------------------------
# 1. Line expectation via symbol-tuple enumeration
# --------------------------------------------------------------------------
def reel_marginals(reels: List[np.ndarray], n_symbols: int) -> np.ndarray:
    """P(symbol s shows on a given row of reel i) = count/len.  Shape (n_reels, n_symbols)."""
    out = np.zeros((len(reels), n_symbols))
    for i, strip in enumerate(reels):
        out[i] = np.bincount(strip, minlength=n_symbols) / len(strip)
    return out


@dataclass
class LineAnalysis:
    ev: float                        # expected win of ONE line (x bet)
    by_symbol_run: np.ndarray        # (n_symbols, 6) expected contribution per (paying symbol, run)
    prob_symbol_run: np.ndarray      # (n_symbols, 6) probability per (paying symbol, run)
    hit_prob: float                  # P(single line pays > 0)


def single_line_analysis(
    reels: List[np.ndarray],
    paytable: np.ndarray,
    wilds: Optional[Sequence[int]] = None,
    symbol_mult: Optional[np.ndarray] = None,
) -> LineAnalysis:
    n_reels = len(reels)
    n_sym = paytable.shape[0]
    marg = reel_marginals(reels, n_sym)
    # keep only symbols that actually appear on each reel to shrink the enumeration
    present = [np.nonzero(marg[i] > 0)[0] for i in range(n_reels)]
    tuples = np.array(list(product(*present)), dtype=np.int16)          # (K, n_reels)
    prob = np.ones(len(tuples))
    for i in range(n_reels):
        prob *= marg[i, tuples[:, i]]
    win, cand, run = line_outcome(tuples, paytable, wilds, symbol_mult)
    by = np.zeros((n_sym, 6))
    pr = np.zeros((n_sym, 6))
    np.add.at(by, (cand, run), prob * win)
    np.add.at(pr, (cand, run), prob)
    paying = win > 0
    return LineAnalysis(
        ev=float((prob * win).sum()),
        by_symbol_run=by,
        prob_symbol_run=pr,
        hit_prob=float(prob[paying].sum()),
    )


# --------------------------------------------------------------------------
# 2. Scatter count distribution (exact, per-reel sliding window + convolution)
# --------------------------------------------------------------------------
def per_reel_symbol_count_dist(strip: np.ndarray, symbol: int, rows: int) -> np.ndarray:
    """P(k copies of `symbol` visible on this reel), k = 0..rows."""
    L = len(strip)
    is_s = (strip == symbol).astype(np.int64)
    counts = np.array([is_s[(np.arange(rows) + stop) % L].sum() for stop in range(L)])
    return np.bincount(counts, minlength=rows + 1) / L


def scatter_count_distribution(reels: List[np.ndarray], symbol: int, rows: int) -> np.ndarray:
    """P(total scatters in window = k), k = 0..rows*n_reels. Exact via convolution."""
    dist = np.array([1.0])
    for strip in reels:
        dist = np.convolve(dist, per_reel_symbol_count_dist(strip, symbol, rows))
    return dist


def per_reel_presence_prob(reels: List[np.ndarray], symbol: int, rows: int) -> np.ndarray:
    """P(reel i shows `symbol` at least once) for each reel (expanding symbols)."""
    return np.array([1.0 - per_reel_symbol_count_dist(s, symbol, rows)[0] for s in reels])


def reels_containing_distribution(reels: List[np.ndarray], symbol: int, rows: int) -> np.ndarray:
    """P(exactly k reels contain `symbol`), k = 0..n_reels (reels independent)."""
    dist = np.array([1.0])
    for p in per_reel_presence_prob(reels, symbol, rows):
        dist = np.convolve(dist, np.array([1 - p, p]))
    return dist


# --------------------------------------------------------------------------
# 3. Full grid enumeration -> joint histogram of (scatter count, line win)
# --------------------------------------------------------------------------
@dataclass
class GridEnumeration:
    joint: np.ndarray         # (max_scatters+1, n_bins): P(scatter=s, line win = b*bin_width)
    bin_width: float
    n_combinations: int
    max_line_win: float
    extra: Dict               # optional extra histograms (game specific)

    @property
    def win_values(self) -> np.ndarray:
        return np.arange(self.joint.shape[1]) * self.bin_width

    @property
    def line_win_dist(self) -> np.ndarray:
        return self.joint.sum(axis=0)

    @property
    def scatter_dist(self) -> np.ndarray:
        return self.joint.sum(axis=1)

    @property
    def line_ev(self) -> float:
        return float((self.line_win_dist * self.win_values).sum())

    @property
    def line_hit_prob(self) -> float:
        return float(self.line_win_dist[1:].sum())


def enumerate_grid(
    reels: List[np.ndarray],
    rows: int,
    paytable: np.ndarray,
    paylines: np.ndarray,
    wilds: Optional[Sequence[int]],
    scatter: Optional[int],
    bin_width: float,
    symbol_mult: Optional[np.ndarray] = None,
    max_bins: int = 200_000,
    extra_fn=None,
    chunk_reels: int = 2,
) -> GridEnumeration:
    """Enumerate EVERY stop combination and histogram (scatter count, total line win).

    The first `chunk_reels` reels are iterated in Python; the remaining reels are
    expanded with meshgrid and evaluated vectorised. For 5 reels x 32 stops this
    is 1024 chunks of 32768 grids.

    extra_fn(window, line_win, scatter_count) -> dict of name -> (idx_array, n_idx)
    lets a game accumulate additional exact histograms (e.g. "reels containing
    symbol e" for expanding-symbol games) in the same pass.
    """
    n_reels = len(reels)
    lens = [len(r) for r in reels]
    n_comb = int(np.prod(lens))
    max_sc = rows * n_reels if scatter is not None else 0
    hist = np.zeros((max_sc + 1) * max_bins, dtype=np.float64)
    extra_hist: Dict[str, np.ndarray] = {}
    max_win = 0.0

    head = [range(L) for L in lens[:chunk_reels]]
    tail_grids = np.meshgrid(*[np.arange(L) for L in lens[chunk_reels:]], indexing="ij")
    tail = np.stack([g.ravel() for g in tail_grids], axis=1)        # (M, n_reels - chunk_reels)
    M = tail.shape[0]
    stops = np.empty((M, n_reels), dtype=np.int64)
    stops[:, chunk_reels:] = tail

    for head_stops in product(*head):
        stops[:, :chunk_reels] = head_stops
        window = windows_from_stops(reels, stops, rows)
        lw = evaluate_lines(window, paytable, paylines, wilds, symbol_mult)
        sc = count_symbol(window, scatter) if scatter is not None else np.zeros(M, dtype=np.int64)
        b = np.rint(lw / bin_width).astype(np.int64)
        if b.max() >= max_bins:
            raise ValueError("max_bins too small for this game's line wins")
        max_win = max(max_win, float(lw.max()))
        hist += np.bincount(sc * max_bins + b, minlength=hist.size)
        if extra_fn is not None:
            for name, (idx, n_idx) in extra_fn(window, lw, sc).items():
                if name not in extra_hist:
                    extra_hist[name] = np.zeros(n_idx, dtype=np.float64)
                extra_hist[name] += np.bincount(idx, minlength=n_idx)

    joint = hist.reshape(max_sc + 1, max_bins) / n_comb
    # trim trailing zero bins
    last = np.nonzero(joint.sum(axis=0) > 0)[0].max()
    joint = joint[:, : last + 1]
    extra = {k: v / n_comb for k, v in extra_hist.items()}
    return GridEnumeration(joint=joint, bin_width=bin_width, n_combinations=n_comb,
                           max_line_win=max_win, extra=extra)


# --------------------------------------------------------------------------
# Distribution helpers (shared with markov.py)
# --------------------------------------------------------------------------
def shift_dist(dist: np.ndarray, shift_bins: int) -> np.ndarray:
    """Distribution of X + c where c = shift_bins * bin_width (prepend zeros)."""
    if shift_bins == 0:
        return dist
    return np.concatenate([np.zeros(shift_bins), dist])


def scale_dist(dist: np.ndarray, k: int) -> np.ndarray:
    """Distribution of k*X for integer k >= 1 (bin b -> bin k*b). Exact on the grid."""
    if k == 1:
        return dist
    out = np.zeros((len(dist) - 1) * k + 1)
    out[::k] = dist
    return out


def convolve_capped(a: np.ndarray, b: np.ndarray, cap_bins: Optional[int]) -> np.ndarray:
    """Distribution of min(X + Y, cap) for independent X ~ a, Y ~ b (FFT convolution).

    Folding the tail into the cap bin after every addition is exact because
    min(x + y, cap) = min(min(x, cap) + y, cap) for y >= 0.
    """
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    c = np.fft.irfft(fa * fb, nfft)[:n]
    c[c < 0] = 0.0
    if cap_bins is not None and len(c) > cap_bins + 1:
        tail = c[cap_bins + 1:].sum()
        c = c[: cap_bins + 1]
        c[cap_bins] += tail
    return c


def cap_dist(dist: np.ndarray, cap_bins: Optional[int]) -> np.ndarray:
    if cap_bins is None or len(dist) <= cap_bins + 1:
        return dist
    out = dist[: cap_bins + 1].copy()
    out[cap_bins] += dist[cap_bins + 1:].sum()
    return out


def dist_moments(dist: np.ndarray, bin_width: float) -> Tuple[float, float, float]:
    """(mean, variance, std) of a binned distribution (x bet)."""
    v = np.arange(len(dist)) * bin_width
    p = dist / dist.sum()
    mean = float((p * v).sum())
    var = float((p * v * v).sum() - mean ** 2)
    return mean, var, float(np.sqrt(max(var, 0.0)))


def dist_quantile(dist: np.ndarray, bin_width: float, q: float) -> float:
    c = np.cumsum(dist) / dist.sum()
    return float(np.searchsorted(c, q) * bin_width)


def dist_tail_prob(dist: np.ndarray, bin_width: float, threshold: float) -> float:
    """P(X >= threshold)."""
    b = int(np.ceil(threshold / bin_width - 1e-9))
    return float(dist[b:].sum()) if b < len(dist) else 0.0
