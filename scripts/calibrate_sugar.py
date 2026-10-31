"""Calibrate the Sugar Heaven reference model to its PUBLIC spec (RTP 96.42 %, hit 31 %).

Usage: python scripts/calibrate_sugar.py [--iters 500]
The free-spin strips are fixed. Their value is estimated by simulation once
(300k features per trigger size). Base strips are tuned by local search on
exact means + a fixed-seed Monte-Carlo hit rate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games import sugar_heaven_ref as sh                     # noqa: E402
from slotmath.calibrate import local_search                 # noqa: E402
from slotmath.engine import count_symbol, evaluate_lines, random_stops, windows_from_stops   # noqa: E402
from slotmath.simulate import run_simulation                # noqa: E402

TARGET = sh.PUBLIC["rtp"]
HIT_BAND = (0.305, 0.315)
TRIG_BAND = (190.0, 240.0)


def mc_hit_rate(cfg, n=200_000, seed=11):
    rng = np.random.default_rng(seed)
    stops = random_stops(cfg.base_reels, n, rng)
    win = windows_from_stops(cfg.base_reels, stops, cfg.rows)
    lw = evaluate_lines(win, cfg.paytable, cfg.paylines, sh.WILDS, sh.SYMBOL_MULT)
    sc = count_symbol(win, sh.S)
    return float(((lw > 0) | (sc >= 3)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--warm", action="store_true", help="start from the saved strips")
    args = ap.parse_args()
    counts = sh.default_counts()
    if args.warm and sh.REELS_FILE.exists():
        saved = sh.load_config()
        counts["base"] = [{sh.SYMBOLS[j]: int(c) for j, c in enumerate(np.bincount(r, minlength=len(sh.SYMBOLS))) if c > 0}
                          for r in saved.base_reels]
    fs_reels = sh.reels_from_counts(counts["fs"])
    cfg0 = sh.build_config(sh.reels_from_counts(counts["base"]), fs_reels)
    t = time.time()
    print("Simulating sticky-wild feature value (300k features per trigger size) ...")
    fs_ev = sh.simulate_feature_ev(cfg0, 300_000)
    for k, (m, h) in fs_ev.items():
        print(f"  {k} scatters -> {sh.FS_SPINS[k]} spins: E[payout] = {m:.2f}x +/- {h:.2f} (95% CI)")
    print(f"  ({time.time()-t:.0f}s)")

    def score(base_counts):
        cfg = sh.build_config(sh.reels_from_counts(base_counts), fs_reels)
        q = sh.quick_rtp(cfg, fs_ev)
        one_in = 1 / q["p_trigger"]
        pen = 0.0
        pen += max(0.0, TRIG_BAND[0] - one_in) * 1e-4 + max(0.0, one_in - TRIG_BAND[1]) * 1e-4
        hit = mc_hit_rate(cfg)
        pen += max(0.0, HIT_BAND[0] - hit) * 0.5 + max(0.0, hit - HIT_BAND[1]) * 0.5
        return abs(q["total"] - TARGET) + pen

    def allowed(i, a, b, cur):
        if a == "S" or b == "S":
            return False
        wild_count = sum(cur[i].get(w, 0) for w in ("W2", "W3", "W5", "W10"))
        if b.startswith("W"):
            if i == 0 or wild_count >= (2 if i == 2 else 1) or b in ("W5", "W10"):
                return False                       # at most one wild (two on reel 3), only x2/x3 in the base game
        if a.startswith("W") and wild_count <= 1 and i in (1, 2, 3):
            return False
        if a.startswith("H") and cur[i][a] - 1 < 2:
            return False
        if a.startswith("L") and cur[i][a] - 1 < 3:
            return False
        if b.startswith("L") and cur[i].get(b, 0) >= 6:
            return False                           # no low symbol more than 6 times per reel
        L = sum(cur[i].values())
        return 34 <= L <= 44

    print(f"\nLocal search on base strips (target RTP {100*TARGET:.2f}%) ...")
    best = local_search(counts["base"], score, allowed, sh.SYMBOLS, iters=args.iters, seed=args.seed)
    cfg = sh.build_config(sh.reels_from_counts(best), fs_reels)
    q = sh.quick_rtp(cfg, fs_ev)
    print("\nResult:")
    for k, v in q.items():
        print(f"  {k:12s} {v:.6f}")
    print(f"  trigger 1 in {1/q['p_trigger']:.1f};  MC hit rate {100*mc_hit_rate(cfg):.2f}%")
    for i, c in enumerate(best):
        print(f"    reel {i+1} (len {sum(c.values())}): {c}")
    sh.save_reels(cfg)
    print(f"Saved strips to {sh.REELS_FILE}")
    res = run_simulation("sugar-quick", sh.spin_fn(cfg), 2_000_000, seed=1, progress=False)
    print(res.summary())


if __name__ == "__main__":
    main()
