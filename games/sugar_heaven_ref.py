"""Reference model A — calibrated to the PUBLIC parameters of "Sugar Heaven" (Just Slots).

Public parameters (provider page / reviews): 5x3, 20 paylines, RTP 96.42 % (other
variants 94.40 / 92.45), hit frequency 31 %, high volatility, max win 5,000x,
Multiplier Wilds in the base game, Free Spins 8/9/12 for 3/4/5 scatters with
STICKY wilds, bonus buy 100x (3+ scatters) / 300x (5 scatters).

This is NOT the provider's math (reel strips and paytables are proprietary); it
is an original model whose mechanics and headline numbers match the public spec.
Simplifications: Super Wild enhancers and the random base-game wild drop are
left out; the feature has no retrigger.

Exactness: base game fully exact (enumeration); the sticky-wild feature is
estimated by Monte Carlo with a confidence interval — its state (which cells
hold which multipliers) is too large to enumerate, which is normal practice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from slotmath.engine import count_symbol, evaluate_lines, random_stops, windows_from_stops
from slotmath.exact import (GridEnumeration, dist_moments, enumerate_grid, scatter_count_distribution,
                            single_line_analysis)
from slotmath.game import PAYLINES_20, GameConfig
from slotmath.strips import spread_strip

NAME = "Sugar Heaven (reference model)"
PUBLIC = {"rtp": 0.9642, "hit_rate": 0.31, "volatility": "high", "max_win": 5000.0,
          "free_spins": {3: 8, 4: 9, 5: 12}, "bonus_buy": {100: "3+ scatters", 300: "5 scatters"},
          "source": "https://www.just-slots.com/games/sugar-heaven"}

SYMBOLS = ["W2", "W3", "W5", "W10", "S", "H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4"]
WILDS = [0, 1, 2, 3]
SYMBOL_MULT = np.array([2, 3, 5, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1])
MULT_TO_WILD = {2: 0, 3: 1, 5: 2, 10: 3}
S = 4

PAYTABLE = np.array([
    [0, 0, 0, 1.5, 5.0, 20.0],   # pure-wild line pays like H1 (times the wild multipliers)
    [0, 0, 0, 1.5, 5.0, 20.0],
    [0, 0, 0, 1.5, 5.0, 20.0],
    [0, 0, 0, 1.5, 5.0, 20.0],
    [0, 0, 0, 0.0, 0.0, 0.0],    # scatter: no line pay
    [0, 0, 0, 1.5, 5.0, 20.0],   # H1 cake
    [0, 0, 0, 1.0, 3.0, 12.0],   # H2 cupcake
    [0, 0, 0, 0.8, 2.0, 8.0],    # H3 lollipop
    [0, 0, 0, 0.6, 1.5, 6.0],    # H4 donut
    [0, 0, 0, 0.3, 0.8, 3.0],    # L1 red candy
    [0, 0, 0, 0.2, 0.6, 2.5],    # L2 green candy
    [0, 0, 0, 0.2, 0.5, 2.0],    # L3 blue candy
    [0, 0, 0, 0.2, 0.4, 1.5],    # L4 yellow candy
])
SCATTER_PAYS = {3: 2.0, 4: 10.0, 5: 50.0}
FS_SPINS = {3: 8, 4: 9, 5: 12}
WIN_CAP = 5000.0
BIN = 0.1
REELS_FILE = Path(__file__).with_name("sugar_heaven_ref_reels.json")


def default_counts() -> Dict[str, List[Dict[str, int]]]:
    # Multiplier wilds are rare on purpose: their multipliers ADD on a winning line and, in the
    # feature, they are sticky — a little wild density goes a very long way.
    base = [
        {"S": 1, "H1": 3, "H2": 3, "H3": 4, "H4": 4, "L1": 5, "L2": 5, "L3": 6, "L4": 6},
        {"W2": 1, "S": 1, "H1": 3, "H2": 3, "H3": 4, "H4": 4, "L1": 5, "L2": 5, "L3": 6, "L4": 6},
        {"W2": 1, "W3": 1, "S": 1, "H1": 3, "H2": 3, "H3": 4, "H4": 4, "L1": 5, "L2": 5, "L3": 6, "L4": 6},
        {"W2": 1, "S": 1, "H1": 3, "H2": 3, "H3": 4, "H4": 4, "L1": 5, "L2": 5, "L3": 6, "L4": 6},
        {"S": 1, "H1": 3, "H2": 3, "H3": 4, "H4": 4, "L1": 5, "L2": 5, "L3": 6, "L4": 6},
    ]
    # Free-spin density was tuned so that E[feature | 3 scatters] ~ 98x: the public bonus-buy price
    # of 100x pins the feature's value (a 100x buy with a 128x feature would be a 128 % RTP exploit).
    fs = [
        {"H1": 3, "H2": 3, "H3": 3, "H4": 3, "L1": 3, "L2": 3, "L3": 4, "L4": 4},
        {"W2": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 3, "L1": 3, "L2": 3, "L3": 4, "L4": 4},
        {"W2": 1, "W3": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 3, "L1": 3, "L2": 3, "L3": 4, "L4": 4},
        {"W2": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 3, "L1": 3, "L2": 3, "L3": 4, "L4": 4},
        {"W2": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 3, "L1": 3, "L2": 3, "L3": 4, "L4": 4},
    ]
    return {"base": base, "fs": fs}


def reels_from_counts(counts: List[Dict[str, int]]) -> List[np.ndarray]:
    return [spread_strip({SYMBOLS.index(k): v for k, v in c.items() if v > 0}, seed=5 * i) for i, c in enumerate(counts)]


def build_config(base_reels, fs_reels) -> GameConfig:
    cfg = GameConfig(name=NAME, symbols=SYMBOLS, paytable=PAYTABLE, paylines=PAYLINES_20, base_reels=base_reels,
                     rows=3, wild=WILDS[0], scatter=S, scatter_pays=SCATTER_PAYS, fs_reels=fs_reels,
                     feature={"spins": FS_SPINS}, win_cap=WIN_CAP, bin_width=BIN)
    cfg.validate()
    return cfg


def save_reels(cfg: GameConfig, path: Path = REELS_FILE) -> None:
    path.write_text(json.dumps({"base": [[SYMBOLS[int(s)] for s in r] for r in cfg.base_reels],
                                "fs": [[SYMBOLS[int(s)] for s in r] for r in cfg.fs_reels]}, indent=1))


def load_config(path: Path = REELS_FILE) -> GameConfig:
    d = json.loads(path.read_text())
    return build_config([np.array([SYMBOLS.index(n) for n in r]) for r in d["base"]],
                        [np.array([SYMBOLS.index(n) for n in r]) for r in d["fs"]])


# --------------------------------------------------------------------------
# Feature: sticky multiplier wilds (simulation)
# --------------------------------------------------------------------------
def play_free_spins(cfg: GameConfig, rng: np.random.Generator, n0: np.ndarray) -> np.ndarray:
    """Play one sticky-wild feature per entry of n0 (number of spins). Vectorised."""
    k = len(n0)
    sticky = np.zeros((k, cfg.rows, cfg.n_reels), dtype=np.int64)       # multiplier held per cell (0 = none)
    wild_id_of_mult = np.zeros(11, dtype=np.int16)
    for m, wid in MULT_TO_WILD.items():
        wild_id_of_mult[m] = wid
    total = np.zeros(k)
    n_rem = n0.astype(np.int64).copy()
    while True:
        active = np.nonzero(n_rem > 0)[0]
        if len(active) == 0:
            break
        stops = random_stops(cfg.fs_reels, len(active), rng)
        win = windows_from_stops(cfg.fs_reels, stops, cfg.rows)
        st = sticky[active]
        held = st > 0
        win = np.where(held, wild_id_of_mult[st], win)                   # overlay sticky wilds
        lw = evaluate_lines(win, cfg.paytable, cfg.paylines, WILDS, SYMBOL_MULT)
        total[active] += lw
        landed = SYMBOL_MULT[win]                                         # new wilds become sticky
        sticky[active] = np.where(held, st, np.where(landed > 1, landed, 0))
        n_rem[active] -= 1
    return total


def simulate_feature_ev(cfg: GameConfig, n_features: int = 200_000, seed: int = 7) -> Dict[int, Tuple[float, float]]:
    """Mean feature payout per trigger size with 95% CI half-width, by simulation."""
    rng = np.random.default_rng(seed)
    out = {}
    for sc, n0 in FS_SPINS.items():
        tot = np.concatenate([play_free_spins(cfg, rng, np.full(min(50_000, n_features - i), n0))
                              for i in range(0, n_features, 50_000)])
        out[sc] = (float(tot.mean()), float(1.96 * tot.std() / np.sqrt(len(tot))))
    return out


# --------------------------------------------------------------------------
# Analysis: exact base + simulated feature
# --------------------------------------------------------------------------
def quick_rtp(cfg: GameConfig, fs_ev: Dict[int, Tuple[float, float]]) -> Dict[str, float]:
    la = single_line_analysis(cfg.base_reels, cfg.paytable, WILDS, SYMBOL_MULT)
    sd = scatter_count_distribution(cfg.base_reels, S, cfg.rows)
    line = cfg.n_lines * la.ev
    scat = float(sum(sd[k] * v for k, v in SCATTER_PAYS.items()))
    feat = float(sum(sd[k] * fs_ev[k][0] for k in FS_SPINS))
    feat_ci = float(sum(sd[k] * fs_ev[k][1] for k in FS_SPINS))
    p_trig = float(sum(sd[k] for k in FS_SPINS))
    return {"line": line, "scatter": scat, "feature": feat, "feature_ci": feat_ci,
            "total": line + scat + feat, "p_trigger": p_trig, "line_hit": la.hit_prob}


@dataclass
class Report:
    cfg: GameConfig
    base_grid: GridEnumeration
    fs_ev: Dict[int, Tuple[float, float]]
    stats: Dict


def analysis(cfg: GameConfig, n_features: int = 400_000, verbose: bool = True) -> Report:
    log = print if verbose else (lambda *a, **k: None)
    log("[1/2] base game: full enumeration ...")
    grid = enumerate_grid(cfg.base_reels, cfg.rows, cfg.paytable, cfg.paylines, WILDS, S, cfg.bin_width,
                          symbol_mult=SYMBOL_MULT)
    log(f"      {grid.n_combinations:,} combinations")
    log(f"[2/2] sticky-wild feature: {n_features:,} simulated features per trigger size ...")
    fs_ev = simulate_feature_ev(cfg, n_features)
    sd = grid.scatter_dist
    q = quick_rtp(cfg, fs_ev)
    # hit rate: any line win or any scatter pay (exact from the enumeration)
    hit = float(1.0 - grid.joint[:3, 0].sum())
    stats = {
        "rtp": q["total"], "rtp_ci95_half": q["feature_ci"], "rtp_line": q["line"], "rtp_scatter": q["scatter"],
        "rtp_feature": q["feature"], "p_trigger": q["p_trigger"], "trigger_1_in": 1 / q["p_trigger"],
        "hit_rate_exact": hit, "feature_ev_by_count": {k: v[0] for k, v in fs_ev.items()},
        "feature_ev_ci_by_count": {k: v[1] for k, v in fs_ev.items()},
        "p_trigger_by_count": {k: float(sd[k]) for k in FS_SPINS},
        "bonus_buy_100x_rtp": float(sum(sd[k] * fs_ev[k][0] for k in FS_SPINS) / q["p_trigger"] / 100.0),
        "bonus_buy_300x_rtp": fs_ev[5][0] / 300.0,
        "public": PUBLIC, "base_combinations": grid.n_combinations, "max_line_win_base": grid.max_line_win,
    }
    return Report(cfg=cfg, base_grid=grid, fs_ev=fs_ev, stats=stats)


# --------------------------------------------------------------------------
# Full-game simulator
# --------------------------------------------------------------------------
def spin_fn(cfg: GameConfig):
    sc_pay = np.zeros(cfg.rows * cfg.n_reels + 1)
    for k, v in SCATTER_PAYS.items():
        sc_pay[k] = v
    spins_lut = np.zeros(cfg.rows * cfg.n_reels + 1, dtype=np.int64)
    for k, v in FS_SPINS.items():
        spins_lut[k] = v

    def fn(rng, n):
        stops = random_stops(cfg.base_reels, n, rng)
        win = windows_from_stops(cfg.base_reels, stops, cfg.rows)
        lw = evaluate_lines(win, cfg.paytable, cfg.paylines, WILDS, SYMBOL_MULT)
        sc = count_symbol(win, S)
        total = lw + sc_pay[sc]
        trig = np.nonzero(spins_lut[sc] > 0)[0]
        fw = np.zeros(0)
        if len(trig):
            fw = play_free_spins(cfg, rng, spins_lut[sc[trig]])
            total[trig] += fw
        np.minimum(total, cfg.win_cap, out=total)
        return total, {"feature_count": len(trig), "feature_wins": fw}

    return fn
