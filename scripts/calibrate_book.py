"""Calibrate the Book of Arcane reference model to its PUBLIC spec (RTP 96.36 %, hit ~27 %).

Two stages:
1. Reel strips: local search so that line + scatter RTP ~ 48.6 %, hit rate in a band
   around 27 % and feature frequency ~ 1 in 170-200 spins (exact means + fixed-seed MC hit rate).
2. Multiplier weights: after the exact analysis (enumeration + Markov chain), the
   feature RTP is linear in E[M]; the 1x/2x split is solved so the exact RTP is 96.36 %.

Usage: python scripts/calibrate_book.py [--iters 500]   (stage 2 takes ~10 minutes)
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import book_of_arcane_ref as bk                   # noqa: E402
from slotmath.calibrate import local_search                 # noqa: E402
from slotmath.engine import count_symbol, evaluate_lines, random_stops, windows_from_stops   # noqa: E402

TARGET = bk.PUBLIC["rtp"]
BASE_TARGET = 0.486            # line + scatter share of RTP (design choice: ~half from the feature)
HIT_BAND = (0.26, 0.28)
TRIG_BAND = (170.0, 200.0)
REST_WEIGHTS = {3: 0.02, 5: 0.01, 10: 0.006, 20: 0.003, 50: 0.001, 100: 0.0005}


def mc_hit_rate(cfg, n=200_000, seed=11):
    rng = np.random.default_rng(seed)
    stops = random_stops(cfg.base_reels, n, rng)
    win = windows_from_stops(cfg.base_reels, stops, cfg.rows)
    lw = evaluate_lines(win, cfg.paytable, cfg.paylines, [bk.B])
    sc = count_symbol(win, bk.B)
    return float(((lw > 0) | (sc >= 3)).mean())


def initial_weights():
    w = np.zeros(len(bk.MULT_VALUES))
    for m, p in REST_WEIGHTS.items():
        w[list(bk.MULT_VALUES).index(m)] = p
    w[0] = 1.0 - w.sum()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--from-saved", action="store_true", help="skip stage 1, use saved strips")
    args = ap.parse_args()
    w0 = initial_weights()

    def score(counts):
        cfg = bk.build_config(bk.reels_from_counts(counts), w0)
        q = bk.quick_rtp(cfg)
        one_in = 1 / q["p_trigger"]
        pen = max(0.0, TRIG_BAND[0] - one_in) * 1e-4 + max(0.0, one_in - TRIG_BAND[1]) * 1e-4
        hit = mc_hit_rate(cfg)
        pen += max(0.0, HIT_BAND[0] - hit) * 0.5 + max(0.0, hit - HIT_BAND[1]) * 0.5
        return abs(q["line"] + q["scatter"] - BASE_TARGET) + pen

    def allowed(i, a, b, cur):
        if a == "B" or b == "B":
            return False                           # exactly one Book per reel
        if a == "H1" and cur[i][a] - 1 < 1:
            return False
        if a != "H1" and cur[i][a] - 1 < 2:
            return False
        if b == "H1" and cur[i].get("H1", 0) >= 2:
            return False                           # at most 2 top symbols per reel
        if b != "H1" and cur[i].get(b, 0) >= 6:
            return False                           # no symbol more than 6 times per reel (no lopsided strips)
        L = sum(cur[i].values())
        return 32 <= L <= 44

    if args.from_saved:
        cfg = bk.build_config(bk.load_config().base_reels, w0)
    else:
        print("Stage 1: reel strips ...")
        best = local_search(bk.default_counts(), score, allowed, bk.SYMBOLS, iters=args.iters, seed=args.seed)
        cfg = bk.build_config(bk.reels_from_counts(best), w0)
        q = bk.quick_rtp(cfg)
        print(f"  line {100*q['line']:.3f}%  scatter {100*q['scatter']:.3f}%  trigger 1 in {1/q['p_trigger']:.1f}  MC hit {100*mc_hit_rate(cfg):.2f}%")
        for i, c in enumerate(best):
            print(f"    reel {i+1} (len {sum(c.values())}): {c}")
        bk.save_config(cfg)
    if args.skip_exact:
        return

    print("\nStage 2: exact analysis, then solve the multiplier weights ...")
    t = time.time()
    grid = bk.enumerate_base(cfg)
    print(f"  enumeration: {grid.n_combinations:,} combinations in {time.time()-t:.0f}s")
    t = time.time()
    pack = bk.feature_unscaled(cfg, grid)
    print(f"  feature chains in {time.time()-t:.0f}s")
    rep = bk.exact_analysis(cfg, verbose=False, grid=grid, unscaled_pack=pack)
    print(f"  exact RTP with initial weights: {100*rep.stats['rtp_exact_capped']:.4f}%  (E[M] = {rep.stats['expected_multiplier']:.3f})")
    # the solver targets the UNCAPPED RTP; iterate once so that the CAPPED RTP hits the target
    target = TARGET
    for _ in range(3):
        w = bk.solve_mult_weights(cfg, rep, target)
        cfg = bk.build_config(cfg.base_reels, w)
        rep = bk.exact_analysis(cfg, verbose=False, grid=grid, unscaled_pack=pack)
        if abs(rep.stats["rtp_exact_capped"] - TARGET) < 1e-6:
            break
        target += TARGET - rep.stats["rtp_exact_capped"]
    s = rep.stats
    print(f"  solved weights: { {int(m): round(float(p), 6) for m, p in zip(bk.MULT_VALUES, w)} }  E[M] = {s['expected_multiplier']:.4f}")
    print(f"  exact RTP (capped) = {100*s['rtp_exact_capped']:.4f}%   uncapped = {100*s['rtp_exact_uncapped']:.4f}%")
    print(f"  lines {100*s['rtp_line']:.3f}% + scatters {100*s['rtp_scatter']:.3f}% + feature {100*s['rtp_feature']:.3f}%")
    print(f"  hit rate {100*s['hit_rate']:.3f}%  std/spin {s['std_per_spin']:.3f}x  feature 1 in {s['trigger_1_in']:.1f}  feature mean {s['feature_mean']:.1f}x")
    print(f"  P(>=100x) 1 in {s['p_win_ge_100x_1_in']:.0f}; P(>=1000x) 1 in {s['p_win_ge_1000x_1_in']:.0f}; P(>=10000x) 1 in {s['p_win_ge_10000x_1_in']:.3g}; P(100,000x) 1 in {s['p_max_win_1_in']:.3g}")
    bk.save_config(cfg)
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    with open(out / "book_exact.pkl", "wb") as f:
        pickle.dump(rep, f)
    import json
    with open(out / "book_exact.json", "w") as f:
        json.dump({k: v for k, v in s.items()}, f, indent=1, default=float)
    print(f"  saved {bk.REELS_FILE}, results/book_exact.pkl, results/book_exact.json")


if __name__ == "__main__":
    main()
