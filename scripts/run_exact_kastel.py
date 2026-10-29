"""Full exact analysis of Kastel Fortune -> results/kastel_exact.json + results/kastel_exact.pkl

Usage: python scripts/run_exact_kastel.py
Runtime: ~3-4 minutes on a laptop (55 million base-game combinations are enumerated).
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import kastel_fortune as kf      # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reels", type=str, default=str(kf.REELS_FILE), help="strips json (e.g. an RTP variant)")
    ap.add_argument("--tag", type=str, default="kastel", help="results file prefix")
    args = ap.parse_args()
    cfg = kf.load_config(Path(args.reels))
    t = time.time()
    rep = kf.exact_analysis(cfg, verbose=True)
    print(f"\nexact analysis done in {time.time()-t:.0f}s\n")
    s = rep.stats
    print(f"RTP (exact, with 1,500x cap) = {100*s['rtp_exact_capped']:.4f}%")
    print(f"RTP (exact, uncapped)        = {100*s['rtp_exact_uncapped']:.4f}%   cap cost = {100*s['cap_cost']:.5f} pp")
    print(f"  lines {100*s['rtp_line']:.3f}% + scatters {100*s['rtp_scatter']:.3f}% + feature {100*s['rtp_feature']:.3f}%")
    print(f"hit rate = {100*s['hit_rate']:.3f}%   std/spin = {s['std_per_spin']:.4f}x   feature 1 in {s['trigger_1_in']:.1f} spins")
    print(f"feature mean | trigger = {s['feature_mean_given_trigger']:.2f}x, median {s['feature_median_given_trigger']:.1f}x, std {s['feature_std_given_trigger']:.2f}x")
    print(f"P(win >= 100x) = 1 in {s['p_win_ge_100x_1_in']:.0f};  P(win >= 1000x) = 1 in {s['p_win_ge_1000x_1_in']:.0f};  P(max win 1500x) = 1 in {s['p_max_win_1_in']:.3g}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"{args.tag}_exact.pkl", "wb") as f:
        pickle.dump(rep, f)
    js = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in s.items()}
    js["total_dist_nonzero"] = {"bin_width": cfg.bin_width,
                                "entries": [[int(i), float(p)] for i, p in enumerate(rep.total_dist) if p > 0]}
    with open(out / f"{args.tag}_exact.json", "w") as f:
        json.dump(js, f, indent=1, default=float)
    print(f"saved {out/(args.tag + '_exact.json')} and {out/(args.tag + '_exact.pkl')}")


if __name__ == "__main__":
    main()
