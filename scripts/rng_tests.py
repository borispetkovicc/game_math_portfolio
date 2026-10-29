"""Game-level RNG tests on the stop generator used by the simulator (NumPy PCG64).

These are the statistical checks a test lab runs on the *mapping from raw random
numbers to game outcomes* (on top of bit-level batteries such as NIST SP 800-22,
Dieharder or TestU01 on the raw generator):

1. Uniformity of stops per reel          — chi-square with L_i - 1 degrees of freedom
2. Independence between reels            — chi-square on the (reel i stop, reel j stop) contingency table
3. Serial independence within a reel     — chi-square on consecutive-spin pairs (lag 1) and the runs test
4. Symbol frequencies vs the strip        — chi-square on observed symbols in the top row
5. Modulo bias demonstration             — why `raw % L` is not uniform and rejection sampling is

Usage: python scripts/rng_tests.py [--spins 2000000]
A healthy generator gives p-values spread over (0, 1); systematic p < 0.01 across tests = bias.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import kastel_fortune as kf            # noqa: E402
from slotmath.engine import random_stops          # noqa: E402


def chi2_uniform(counts: np.ndarray):
    exp = counts.sum() / len(counts)
    stat = float(((counts - exp) ** 2 / exp).sum())
    return stat, float(stats.chi2.sf(stat, len(counts) - 1))


def chi2_independence(x: np.ndarray, y: np.ndarray, Lx: int, Ly: int):
    table = np.zeros((Lx, Ly))
    np.add.at(table, (x, y), 1)
    stat, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return float(stat), float(p), int(dof)


def runs_test(x: np.ndarray):
    """Wald–Wolfowitz runs test on 'above/below median' of the stop sequence."""
    b = x > np.median(x)
    n1, n2 = int(b.sum()), int((~b).sum())
    runs = 1 + int((b[1:] != b[:-1]).sum())
    mu = 1 + 2 * n1 * n2 / (n1 + n2)
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    cfg = kf.load_config()
    rng = np.random.default_rng(args.seed)
    stops = random_stops(cfg.base_reels, args.spins, rng)
    lens = [len(r) for r in cfg.base_reels]
    print(f"{args.spins:,} spins, strips {lens}\n")

    print("1. Uniformity of stops per reel (chi-square, df = L-1)")
    for i, L in enumerate(lens):
        counts = np.bincount(stops[:, i], minlength=L)
        stat, p = chi2_uniform(counts)
        print(f"   reel {i+1}: chi2 = {stat:7.2f}  (df {L-1})  p = {p:.3f}   max |rel. deviation| = {np.abs(counts/counts.mean()-1).max():.4f}")

    print("\n2. Independence between reels (chi-square on contingency tables)")
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]:
        stat, p, dof = chi2_independence(stops[:, i], stops[:, j], lens[i], lens[j])
        print(f"   reels {i+1}&{j+1}: chi2 = {stat:8.1f} (df {dof})  p = {p:.3f}")

    print("\n3. Serial independence within a reel (lag-1 contingency, runs test)")
    for i in (0, 2, 4):
        stat, p, dof = chi2_independence(stops[:-1, i], stops[1:, i], lens[i], lens[i])
        z, pr = runs_test(stops[:, i])
        print(f"   reel {i+1}: lag-1 chi2 = {stat:8.1f} (df {dof}) p = {p:.3f};  runs test z = {z:+.2f} p = {pr:.3f}")

    print("\n4. Symbol frequencies in the top row vs strip composition (chi-square)")
    for i, strip in enumerate(cfg.base_reels):
        sym = strip[stops[:, i]]
        obs = np.bincount(sym, minlength=cfg.n_symbols)
        expct = np.bincount(strip, minlength=cfg.n_symbols) / len(strip) * args.spins
        m = expct > 0
        stat = float(((obs[m] - expct[m]) ** 2 / expct[m]).sum())
        p = float(stats.chi2.sf(stat, m.sum() - 1))
        print(f"   reel {i+1}: chi2 = {stat:6.2f} (df {m.sum()-1})  p = {p:.3f}")

    print("\n5. Modulo bias: mapping a 32-bit integer to a stop with `raw % L`")
    L = lens[0]
    q, r = divmod(2 ** 32, L)
    print(f"   2^32 = {q} * {L} + {r}: the first {r} stops get {q+1} raw values each, the others {q}")
    print(f"   relative bias (q+1)/q - 1 = {1/q:.2e}  -> invisible in {args.spins:,} spins, but formally non-uniform;")
    print("   labs require rejection sampling (discard raw >= q*L) or 64-bit scaling with a documented bias bound.")
    raw = rng.integers(0, 2 ** 32, size=args.spins, dtype=np.int64)
    counts = np.bincount(raw % L, minlength=L)
    stat, p = chi2_uniform(counts)
    print(f"   chi-square on `raw % {L}` with {args.spins:,} draws: p = {p:.3f}  (detecting a relative bias d needs ~L/d^2 = {L*q*q:.1e} draws)")


if __name__ == "__main__":
    main()
