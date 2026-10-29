"""Monte-Carlo simulation of a game, with a 95 % confidence interval and a comparison
against the exact (or public) figures.

Usage:
  python scripts/run_sim.py kastel --spins 1e8 --seed 2024
  python scripts/run_sim.py sugar  --spins 2e7
  python scripts/run_sim.py book   --spins 2e7

Writes results/<game>_sim_<spins>.json and a histogram plot results/<game>_sim_<spins>.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slotmath.simulate import run_simulation        # noqa: E402


def load(game: str):
    if game == "kastel":
        from games import kastel_fortune as g
        cfg = g.load_config()
        ref = json.loads((ROOT / "results" / "kastel_exact.json").read_text()) if (ROOT / "results" / "kastel_exact.json").exists() else {}
        return cfg, g.spin_fn(cfg), ref.get("rtp_exact_capped"), ref.get("hit_rate"), ref.get("std_per_spin"), "exact"
    if game == "sugar":
        from games import sugar_heaven_ref as g
        cfg = g.load_config()
        p = ROOT / "results" / "sugar_exact.json"
        ref = json.loads(p.read_text()) if p.exists() else {}
        return cfg, g.spin_fn(cfg), ref.get("rtp"), ref.get("hit_rate_exact"), None, "exact base + simulated feature"
    if game == "book":
        from games import book_of_arcane_ref as g
        cfg = g.load_config()
        p = ROOT / "results" / "book_exact.json"
        ref = json.loads(p.read_text()) if p.exists() else {}
        return cfg, g.spin_fn(cfg), ref.get("rtp_exact_capped"), ref.get("hit_rate"), ref.get("std_per_spin"), "exact"
    raise SystemExit(f"unknown game {game}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game", choices=["kastel", "sugar", "book"])
    ap.add_argument("--spins", type=float, default=1e7)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--batch", type=int, default=250_000)
    args = ap.parse_args()
    n = int(args.spins)
    cfg, fn, rtp_ref, hit_ref, std_ref, ref_kind = load(args.game)
    hist_max = float(cfg.win_cap)
    res = run_simulation(args.game, fn, n, seed=args.seed, batch=args.batch, bin_width=cfg.bin_width, hist_max=hist_max)
    print("\n" + res.summary())
    if rtp_ref is not None:
        inside = res.ci95[0] <= rtp_ref <= res.ci95[1]
        z = (res.rtp - rtp_ref) / (res.ci95_half / 1.96)
        print(f"\n  reference RTP ({ref_kind}) = {100*rtp_ref:.4f}%  ->  {'INSIDE' if inside else 'OUTSIDE'} the 95% CI  (z = {z:+.2f})")
    if hit_ref is not None:
        print(f"  reference hit rate = {100*hit_ref:.3f}%   simulated = {100*res.hit_rate:.3f}%")
    if std_ref is not None:
        print(f"  reference std/spin = {std_ref:.4f}   simulated = {res.std:.4f}")
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    tag = f"{args.game}_sim_{n:.0e}".replace("+", "")
    res.extra = {"reference_rtp": rtp_ref, "reference_hit_rate": hit_ref, "reference_std": std_ref, "reference_kind": ref_kind}
    res.to_json(str(out / f"{tag}.json"))

    # histogram of per-spin payout (log-log) — the signature of a slot's volatility
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        v = np.arange(len(res.hist)) * res.hist_bin_width
        mask = res.hist > 0
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(v[mask][1:], res.hist[mask][1:] / n, s=6, color="#2563eb", alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("win per spin (x bet)"); ax.set_ylabel("probability (per 0.1x bin)")
        ax.set_title(f"{cfg.name}\nsimulated per-spin payout distribution, {n:,} spins", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout(); fig.savefig(out / f"{tag}.png", dpi=130)
        print(f"  saved results/{tag}.json and results/{tag}.png")
    except Exception as e:                       # plotting is optional
        print(f"  saved results/{tag}.json (plot skipped: {e})")


if __name__ == "__main__":
    main()
