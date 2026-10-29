"""Overlay the EXACT per-spin payout distribution and the SIMULATED one on log-spaced bins.

Usage: python scripts/plot_exact_vs_sim.py kastel   (or book)
Writes results/<game>_exact_vs_sim.png
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    game = sys.argv[1] if len(sys.argv) > 1 else "kastel"
    rep = pickle.load(open(ROOT / "results" / f"{game}_exact.pkl", "rb"))
    files = sorted(glob.glob(str(ROOT / "results" / f"{game}_sim_*.json")), key=lambda p: json.load(open(p))["n_spins"])
    sim = json.load(open(files[-1]))
    bw = rep.cfg.bin_width
    exact = rep.total_dist
    n = sim["n_spins"]
    hist = np.zeros(max(len(exact), max(i for i, _ in sim["hist"]["nonzero"]) + 1))
    for i, c in sim["hist"]["nonzero"]:
        hist[i] = c
    hist /= n
    values = np.arange(len(hist)) * bw
    edges = np.concatenate([[0.05], np.logspace(np.log10(0.1), np.log10(max(values[-1], 1.0) * 1.05), 60)])
    ex = np.zeros(len(edges) - 1); si = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        m = (values >= lo) & (values < hi)
        ex[k] = exact[: len(values)][m[: len(exact)]].sum() if len(exact) else 0
        si[k] = hist[m].sum()
    mids = np.sqrt(edges[:-1] * edges[1:])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    keep = ex > 0
    ax.plot(mids[keep], ex[keep], color="#111827", lw=2, label="exact (enumeration + Markov chain)")
    keep2 = si > 0
    ax.scatter(mids[keep2], si[keep2], s=18, color="#2563eb", zorder=3, label=f"simulation ({n:,} spins)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("win per spin (x bet), log-spaced bins"); ax.set_ylabel("probability per bin")
    ax.set_title(f"{rep.cfg.name}: exact vs simulated payout distribution", fontsize=10)
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout()
    out = ROOT / "results" / f"{game}_exact_vs_sim.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
