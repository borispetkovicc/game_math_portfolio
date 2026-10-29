"""Monte-Carlo simulation harness with confidence intervals.

The simulator is the *independent check* of the exact analysis: it uses the
same engine (reel strips, paylines, paytable) but random stops instead of an
enumeration, and plays the feature as a real player would see it (spin by
spin), instead of solving the Markov chain. If exact and simulated numbers
disagree beyond the confidence interval, one of them has a bug.

Confidence interval: the per-spin payout X_i are i.i.d., so by the CLT
    mean(X) ~ Normal(mu, sigma^2 / N)   for large N,
and a 95% interval is  mean +/- 1.96 * s / sqrt(N)  with s the sample std.
Slot payouts are extremely heavy-tailed (a 5,000x win in a sample of 1e8 spins),
so N must be large for the CLT to kick in; 1e8 spins gives an interval of a
few hundredths of a percentage point of RTP for a medium-high volatility game.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# spin_fn(rng, n) -> (wins (n,) float array in x bet, info dict with optional keys:
#                     'feature_count' int, 'feature_wins' array, 'base_wins' array)
SpinFn = Callable[[np.random.Generator, int], Tuple[np.ndarray, Dict]]


@dataclass
class SimResult:
    name: str
    n_spins: int
    seed: int
    rtp: float
    std: float
    ci95_half: float
    ci95: Tuple[float, float]
    hit_rate: float
    max_win: float
    feature_count: int
    feature_freq_spins: Optional[float]          # one feature every X spins
    feature_rtp: Optional[float]                 # RTP contribution of the feature
    seconds: float
    hist_bin_width: float
    hist: np.ndarray = field(repr=False)        # histogram of per-spin payout (x bet), on bin grid
    extra: Dict = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        d = asdict(self)
        d["hist"] = {"bin_width": self.hist_bin_width, "nonzero": [[int(i), int(c)] for i, c in enumerate(self.hist) if c]}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    def summary(self) -> str:
        lines = [
            f"{self.name}: {self.n_spins:,} spins, seed={self.seed}, {self.seconds:.0f}s",
            f"  RTP        = {100*self.rtp:.4f}%  (95% CI: {100*self.ci95[0]:.4f}% .. {100*self.ci95[1]:.4f}%)",
            f"  std/spin   = {self.std:.4f} x bet",
            f"  hit rate   = {100*self.hit_rate:.3f}%",
            f"  max win    = {self.max_win:.1f} x bet",
        ]
        if self.feature_count:
            lines.append(f"  features   = {self.feature_count:,}  (1 in {self.feature_freq_spins:.1f} spins), feature RTP = {100*self.feature_rtp:.3f}%")
        return "\n".join(lines)


def run_simulation(name: str, spin_fn: SpinFn, n_spins: int, seed: int = 12345,
                   batch: int = 200_000, bin_width: float = 0.1, hist_max: float = 10_000.0,
                   progress: bool = True) -> SimResult:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    s1 = 0.0; s2 = 0.0; hits = 0; mx = 0.0; feats = 0; feat_sum = 0.0
    nb = int(round(hist_max / bin_width)) + 1
    hist = np.zeros(nb, dtype=np.int64)
    done = 0
    next_report = 0.1
    while done < n_spins:
        n = min(batch, n_spins - done)
        wins, info = spin_fn(rng, n)
        s1 += float(wins.sum()); s2 += float((wins * wins).sum())
        hits += int((wins > 0).sum()); mx = max(mx, float(wins.max()))
        b = np.minimum(np.rint(wins / bin_width).astype(np.int64), nb - 1)
        hist += np.bincount(b, minlength=nb)
        feats += int(info.get("feature_count", 0))
        if "feature_wins" in info:
            feat_sum += float(np.sum(info["feature_wins"]))
        done += n
        if progress and done / n_spins >= next_report:
            el = time.time() - t0
            print(f"  [{name}] {done:,}/{n_spins:,} spins, RTP so far {100*s1/done:.3f}%, {el:.0f}s", flush=True)
            next_report += 0.1
    mean = s1 / n_spins
    var = max(s2 / n_spins - mean * mean, 0.0)
    std = float(np.sqrt(var))
    half = 1.96 * std / np.sqrt(n_spins)
    return SimResult(
        name=name, n_spins=n_spins, seed=seed, rtp=mean, std=std, ci95_half=half,
        ci95=(mean - half, mean + half), hit_rate=hits / n_spins, max_win=mx,
        feature_count=feats, feature_freq_spins=(n_spins / feats if feats else None),
        feature_rtp=(feat_sum / n_spins if feats else None), seconds=time.time() - t0,
        hist_bin_width=bin_width, hist=hist,
    )
