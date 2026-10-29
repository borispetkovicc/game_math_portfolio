"""Calibrate Kastel Fortune reel strips to the design targets and save them.

Usage:  python scripts/calibrate_kastel.py [--iters 400] [--target 0.962]

Steps
1. Free-spin strips are fixed by design (they set the feature's "juice").
   Their per-spin outcome histogram is computed EXACTLY by full enumeration,
   and the feature's expected payout by the Markov-chain mean recursion.
2. Base strips are tuned by greedy local search on symbol counts; the score
   uses exact line/scatter means, so the final RTP is exact (not a Monte-Carlo
   estimate). Hit frequency is checked afterwards with a short simulation.
3. The calibrated strips are written to games/kastel_fortune_reels.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games import kastel_fortune as kf                               # noqa: E402
from slotmath.calibrate import local_search                        # noqa: E402
from slotmath.exact import enumerate_grid                          # noqa: E402
from slotmath.simulate import run_simulation                       # noqa: E402

TARGET = 0.9620
TRIG_BAND = (175.0, 200.0)       # acceptable "1 in N spins" for the feature
HIT_BAND = (0.28, 0.34)          # acceptable base-game hit frequency (any win > 0)


def mc_hit_rate(cfg, n: int = 200_000, seed: int = 11) -> float:
    """Base-game hit frequency by simulation with FIXED random numbers (common random numbers
    make the local search stable: every candidate is scored on the same stops)."""
    from slotmath.engine import count_symbol, evaluate_lines, random_stops, windows_from_stops
    rng = np.random.default_rng(seed)
    stops = random_stops(cfg.base_reels, n, rng)
    win = windows_from_stops(cfg.base_reels, stops, cfg.rows)
    lw = evaluate_lines(win, cfg.paytable, cfg.paylines, [cfg.wild])
    sc = count_symbol(win, cfg.scatter)
    return float(((lw > 0) | (sc >= 3)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--target", type=float, default=TARGET)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=str, default=str(kf.REELS_FILE), help="where to write the strips (RTP variants)")
    ap.add_argument("--warm", action="store_true", help="start from the strips in --out's sibling kastel_fortune_reels.json")
    args = ap.parse_args()

    counts = kf.default_counts()
    if args.warm and kf.REELS_FILE.exists():
        saved = kf.load_config()
        counts["base"] = [{kf.SYMBOLS[j]: int(c) for j, c in enumerate(np.bincount(r, minlength=len(kf.SYMBOLS))) if c > 0}
                          for r in saved.base_reels]
    fs_reels = kf.reels_from_counts(counts["fs"])

    t = time.time()
    print("Exact enumeration of free-spin reels ...")
    fs_grid = enumerate_grid(fs_reels, 3, kf.PAYTABLE, kf.build_config(kf.reels_from_counts(counts["base"]), fs_reels).paylines,
                             [kf.W], kf.S, kf.BIN)
    print(f"  {fs_grid.n_combinations:,} combinations in {time.time()-t:.0f}s; per-spin line EV {fs_grid.line_ev:.4f}, hit {fs_grid.line_hit_prob:.4f}")
    fs_ev = kf.fs_feature_ev(kf.build_config(kf.reels_from_counts(counts["base"]), fs_reels), fs_grid)
    print(f"  feature EV by trigger size: { {k: round(v, 2) for k, v in fs_ev.items()} }")

    def score(base_counts):
        cfg = kf.build_config(kf.reels_from_counts(base_counts), fs_reels)
        q = kf.quick_rtp(cfg, fs_ev)
        one_in = 1.0 / q["p_trigger"]
        pen = 0.0
        if one_in < TRIG_BAND[0]:
            pen += (TRIG_BAND[0] - one_in) * 1e-4
        if one_in > TRIG_BAND[1]:
            pen += (one_in - TRIG_BAND[1]) * 1e-4
        hit = mc_hit_rate(cfg)
        if hit < HIT_BAND[0]:
            pen += (HIT_BAND[0] - hit) * 0.5
        if hit > HIT_BAND[1]:
            pen += (hit - HIT_BAND[1]) * 0.5
        return abs(q["total"] - args.target) + pen

    def allowed(i, a, b, cur):
        if a == "S" or b == "S":
            return False                               # exactly one scatter per reel, by design
        if b == "W" and (i not in (1, 2, 3) or cur[i].get("W", 0) >= 2):
            return False                               # wilds only on reels 2-4, at most 2 each
        if a == "W" and cur[i]["W"] <= 1 and i in (1, 2, 3):
            return False                               # keep at least one wild on reels 2-4
        L = sum(cur[i].values())
        if b == "H1" and cur[i].get("H1", 0) >= 4:
            return False                               # top symbol at most 4 per reel
        if cur[i][a] - 1 < 2 and a.startswith("H"):
            return False                               # every premium at least twice per reel
        if cur[i][a] - 1 < 2 and a in ("A", "K", "Q", "J"):
            return False                               # every low at least twice per reel
        if b != "W" and cur[i].get(b, 0) >= 6:
            return False                               # no symbol more than 6 times per reel (no lopsided strips)
        return 34 <= L <= 42

    print(f"\nLocal search on base strips (target RTP {100*args.target:.2f}%) ...")
    best = local_search(counts["base"], score, allowed, kf.SYMBOLS, iters=args.iters, seed=args.seed)
    cfg = kf.build_config(kf.reels_from_counts(best), fs_reels)
    q = kf.quick_rtp(cfg, fs_ev)
    print("\nResult (exact means):")
    for k, v in q.items():
        print(f"  {k:10s} {v:.6f}")
    print(f"  trigger 1 in {1/q['p_trigger']:.1f} spins")
    print("  base counts per reel:")
    for i, c in enumerate(best):
        print(f"    reel {i+1} (len {sum(c.values())}): {c}")

    kf.save_reels(cfg, Path(args.out))
    print(f"\nSaved strips to {args.out}")

    print("\nQuick simulation check (2e6 spins) ...")
    res = run_simulation("kastel-quick", kf.spin_fn(cfg), 2_000_000, seed=1, batch=200_000, progress=False)
    print(res.summary())


if __name__ == "__main__":
    main()
