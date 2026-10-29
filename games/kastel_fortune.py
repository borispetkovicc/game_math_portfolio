"""KASTEL FORTUNE — an original 5x3, 20-line slot designed and solved exactly.

Design targets
--------------
* RTP 96.20 %, medium-high volatility, hit frequency ~ 27 %
* Free spins (10/15/20 for 3/4/5 scatters) with a multiplier ladder:
  x1 at start, +1 after every spin with a line win, capped at x5
* Retrigger: 3+ scatters inside free spins award +5 spins, at most 4 retriggers
* Max win 1,500x bet (hard cap on the total win of a spin incl. its feature) — chosen from the
  exact tail: P(win >= 1500x) ~ 1 in 5e8, i.e. reachable in a game's lifetime, whereas 5,000x
  would be 1 in 6e16 (an unreachable "advertised" max win is a classic review finding)

Everything about this game is solved EXACTLY (see "exact_analysis"):
the base game by full enumeration of all stop combinations, the feature as an
absorbing Markov chain with a distribution-valued dynamic programme. The
Monte-Carlo simulator ("spin_fn") is the independent check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from slotmath.engine import count_symbol, evaluate_lines, random_stops, windows_from_stops
from slotmath.exact import (GridEnumeration, LineAnalysis, cap_dist, convolve_capped, dist_moments,
                            dist_quantile, dist_tail_prob, enumerate_grid, scatter_count_distribution,
                            shift_dist, single_line_analysis)
from slotmath.game import PAYLINES_20, GameConfig
from slotmath.markov import LadderAnalysis, LadderRules, analyze_ladder_feature, ladder_feature_ev_fast
from slotmath.strips import spread_strip

NAME = "Kastel Fortune"
SYMBOLS = ["W", "S", "H1", "H2", "H3", "H4", "A", "K", "Q", "J"]
SYMBOL_NAMES = {
    "W": "Wild (zlatni ključ)", "S": "Scatter (grb grada)", "H1": "Kastel (tvrđava)",
    "H2": "Vrbas", "H3": "Dajak čamac", "H4": "Lipa", "A": "A", "K": "K", "Q": "Q", "J": "J",
}
W, S = 0, 1

# pays for 0,1,2,3,4,5 of a kind, in multiples of TOTAL bet
PAYTABLE = np.array([
    [0, 0, 0, 0.0, 0.0, 0.0],      # W  substitutes only, no pays of its own: wilds sit on reels 2-4 and a line
                                   #    run must start on reel 1, so a pure-wild run can never occur (an advertised
                                   #    wild pay would be unreachable - a classic certification finding)
    [0, 0, 0, 0.0, 0.0, 0.0],      # S  (never pays on lines; scatter pays are separate)
    [0, 0, 0, 2.0, 8.0, 40.0],     # H1 Kastel
    [0, 0, 0, 1.5, 5.0, 25.0],     # H2 Vrbas
    [0, 0, 0, 1.0, 3.0, 15.0],     # H3 Dajak
    [0, 0, 0, 0.8, 2.0, 10.0],     # H4 Lipa
    [0, 0, 0, 0.4, 1.2, 5.0],      # A
    [0, 0, 0, 0.3, 1.0, 4.0],      # K
    [0, 0, 0, 0.3, 0.8, 3.0],      # Q
    [0, 0, 0, 0.2, 0.6, 2.5],      # J
])
SCATTER_PAYS = {3: 2.0, 4: 10.0, 5: 50.0}          # base game only, x bet
RULES = LadderRules(spins_by_trigger={3: 10, 4: 15, 5: 20}, retrigger_min=3, retrigger_spins=5,
                    max_retriggers=4, mult_start=1, mult_step=1, mult_cap=5)
WIN_CAP = 1500.0
BIN = 0.1
REELS_FILE = Path(__file__).with_name("kastel_fortune_reels.json")


def default_counts() -> Dict[str, List[Dict[str, int]]]:
    """Starting point for calibration: symbol counts per reel (base and free spins)."""
    base = [
        {"W": 0, "S": 1, "H1": 3, "H2": 3, "H3": 4, "H4": 4, "A": 5, "K": 5, "Q": 5, "J": 6},
        {"W": 2, "S": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 4, "A": 5, "K": 5, "Q": 5, "J": 5},
        {"W": 2, "S": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 4, "A": 5, "K": 5, "Q": 5, "J": 5},
        {"W": 2, "S": 1, "H1": 2, "H2": 3, "H3": 3, "H4": 4, "A": 5, "K": 5, "Q": 5, "J": 5},
        {"W": 0, "S": 1, "H1": 2, "H2": 3, "H3": 3, "H4": 4, "A": 5, "K": 5, "Q": 5, "J": 6},
    ]
    # free-spin strips: richer (4 wilds on reels 2-4, fewer low symbols) — fixed by design
    fs = [
        {"W": 0, "S": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 3, "A": 3, "K": 3, "Q": 3, "J": 2},
        {"W": 4, "S": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 3, "A": 3, "K": 3, "Q": 3, "J": 2},
        {"W": 4, "S": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 3, "A": 3, "K": 3, "Q": 3, "J": 2},
        {"W": 4, "S": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 3, "A": 3, "K": 3, "Q": 3, "J": 2},
        {"W": 0, "S": 1, "H1": 3, "H2": 3, "H3": 3, "H4": 3, "A": 3, "K": 3, "Q": 3, "J": 2},
    ]
    return {"base": base, "fs": fs}


def reels_from_counts(counts: List[Dict[str, int]]) -> List[np.ndarray]:
    return [spread_strip({SYMBOLS.index(k): v for k, v in c.items() if v > 0}, seed=7 * i) for i, c in enumerate(counts)]


def build_config(base_reels: List[np.ndarray], fs_reels: List[np.ndarray]) -> GameConfig:
    cfg = GameConfig(
        name=NAME, symbols=SYMBOLS, paytable=PAYTABLE, paylines=PAYLINES_20, base_reels=base_reels,
        rows=3, wild=W, scatter=S, scatter_pays=SCATTER_PAYS, fs_reels=fs_reels,
        feature={"rules": RULES}, win_cap=WIN_CAP, bin_width=BIN,
    )
    cfg.validate()
    return cfg


def save_reels(cfg: GameConfig, path: Path = REELS_FILE) -> None:
    data = {"base": [[SYMBOLS[int(s)] for s in r] for r in cfg.base_reels],
            "fs": [[SYMBOLS[int(s)] for s in r] for r in cfg.fs_reels]}
    path.write_text(json.dumps(data, indent=1))


def load_config(path: Path = REELS_FILE) -> GameConfig:
    data = json.loads(path.read_text())
    base = [np.array([SYMBOLS.index(n) for n in r]) for r in data["base"]]
    fs = [np.array([SYMBOLS.index(n) for n in r]) for r in data["fs"]]
    return build_config(base, fs)


# --------------------------------------------------------------------------
# Exact analysis
# --------------------------------------------------------------------------
@dataclass
class ExactReport:
    cfg: GameConfig
    line: LineAnalysis                  # single-line analysis, base reels
    scatter_dist: np.ndarray            # P(k scatters) base game
    base_grid: GridEnumeration          # full enumeration, base reels
    fs_grid: GridEnumeration            # full enumeration, free-spin reels
    feature: LadderAnalysis
    total_dist: np.ndarray              # exact distribution of a base-game spin's TOTAL win (capped)
    stats: Dict


def quick_rtp(cfg: GameConfig, fs_ev_by_trigger: Dict[int, float]) -> Dict[str, float]:
    """Fast exact RTP decomposition (means only) — used inside the calibration loop."""
    la = single_line_analysis(cfg.base_reels, cfg.paytable, [cfg.wild])
    sd = scatter_count_distribution(cfg.base_reels, cfg.scatter, cfg.rows)
    line_rtp = cfg.n_lines * la.ev
    scat_rtp = sum(sd[k] * v for k, v in cfg.scatter_pays.items() if k < len(sd))
    feat_rtp = sum(sd[k] * fs_ev_by_trigger[k] for k in fs_ev_by_trigger if k < len(sd))
    p_trig = float(sum(sd[k] for k in fs_ev_by_trigger if k < len(sd)))
    return {"line": line_rtp, "scatter": scat_rtp, "feature": feat_rtp,
            "total": line_rtp + scat_rtp + feat_rtp, "p_trigger": p_trig, "line_hit": la.hit_prob}


def fs_feature_ev(cfg: GameConfig, fs_grid: Optional[GridEnumeration] = None) -> Dict[int, float]:
    if fs_grid is None:
        fs_grid = enumerate_grid(cfg.fs_reels, cfg.rows, cfg.paytable, cfg.paylines, [cfg.wild], cfg.scatter, cfg.bin_width)
    return ladder_feature_ev_fast(fs_grid.joint, cfg.bin_width, RULES)


def exact_analysis(cfg: GameConfig, verbose: bool = True, fs_grid: Optional[GridEnumeration] = None,
                   base_grid: Optional[GridEnumeration] = None) -> ExactReport:
    log = print if verbose else (lambda *a, **k: None)
    rules: LadderRules = cfg.feature["rules"]
    cap_bins = int(round(cfg.win_cap / cfg.bin_width))

    log("[1/4] single-line analysis (symbol-tuple enumeration) ...")
    line = single_line_analysis(cfg.base_reels, cfg.paytable, [cfg.wild])
    sd = scatter_count_distribution(cfg.base_reels, cfg.scatter, cfg.rows)

    log("[2/4] full enumeration of free-spin reels ...")
    if fs_grid is None:
        fs_grid = enumerate_grid(cfg.fs_reels, cfg.rows, cfg.paytable, cfg.paylines, [cfg.wild], cfg.scatter, cfg.bin_width)
    log(f"      {fs_grid.n_combinations:,} combinations, max line win in one FS spin = {fs_grid.max_line_win:.1f}x")

    log("[3/4] free-spins Markov chain (distribution-valued DP) ...")
    feature = analyze_ladder_feature(fs_grid.joint, cfg.bin_width, rules, cfg.win_cap)

    log("[4/4] full enumeration of base reels + assembling the total spin distribution ...")
    if base_grid is None:
        base_grid = enumerate_grid(cfg.base_reels, cfg.rows, cfg.paytable, cfg.paylines, [cfg.wild], cfg.scatter, cfg.bin_width)
    total = np.zeros(1)
    for s in range(base_grid.joint.shape[0]):
        row = base_grid.joint[s]
        if row.sum() == 0:
            continue
        if s in rules.spins_by_trigger:
            pay_bins = int(round(cfg.scatter_pays.get(s, 0.0) / cfg.bin_width))
            part = convolve_capped(shift_dist(row, pay_bins), feature.dist_by_trigger[s], cap_bins)
        else:
            part = row
        if len(part) > len(total):
            part = part.copy(); part[: len(total)] += total; total = part
        else:
            total[: len(part)] += part
    total = cap_dist(total, cap_bins)

    # ---- statistics -----------------------------------------------------
    values = np.arange(len(total)) * cfg.bin_width
    rtp_capped = float((total * values).sum())
    line_rtp = cfg.n_lines * line.ev
    scat_rtp = float(sum(sd[k] * v for k, v in cfg.scatter_pays.items()))
    feat_rtp_uncapped = float(sum(sd[k] * feature.ev_by_trigger[k] for k in rules.spins_by_trigger))
    p_trig = float(sum(sd[k] for k in rules.spins_by_trigger))
    mean, var, std = dist_moments(total, cfg.bin_width)
    # feature total distribution conditional on triggering (mixture over trigger sizes)
    L = max(len(d) for d in feature.dist_by_trigger.values())
    fmix = np.zeros(L)
    for k, d in feature.dist_by_trigger.items():
        fmix[: len(d)] += sd[k] / p_trig * d
    f_mean, f_var, f_std = dist_moments(fmix, cfg.bin_width)
    stats = {
        "rtp_exact_capped": rtp_capped,
        "rtp_exact_uncapped": line_rtp + scat_rtp + feat_rtp_uncapped,
        "rtp_line": line_rtp, "rtp_scatter": scat_rtp, "rtp_feature": feat_rtp_uncapped,
        "cap_cost": line_rtp + scat_rtp + feat_rtp_uncapped - rtp_capped,
        "hit_rate": float(1.0 - total[0]),
        "line_hit_rate": base_grid.line_hit_prob,
        "std_per_spin": std, "variance_per_spin": var,
        "p_trigger": p_trig, "trigger_1_in": 1.0 / p_trig,
        "p_trigger_by_count": {k: float(sd[k]) for k in rules.spins_by_trigger},
        "expected_fs_spins_by_count": feature.spins_by_trigger,
        "feature_ev_by_count": feature.ev_by_trigger,
        "feature_mean_given_trigger": f_mean, "feature_std_given_trigger": f_std,
        "feature_median_given_trigger": dist_quantile(fmix, cfg.bin_width, 0.5),
        "feature_p_zero": float(fmix[0]),
        "p_win_ge_100x": dist_tail_prob(total, cfg.bin_width, 100.0),
        "p_win_ge_1000x": dist_tail_prob(total, cfg.bin_width, 1000.0),
        "p_max_win": float(total[cap_bins]) if len(total) > cap_bins else 0.0,
        "max_line_win_single_spin_base": base_grid.max_line_win,
        "max_line_win_single_spin_fs": fs_grid.max_line_win,
        "fs_class_probs": feature.class_probs,
        "fs_per_spin_win_prob": feature.class_probs["win_retrig"] + feature.class_probs["win_noretrig"],
        "fs_per_spin_retrigger_prob": feature.class_probs["win_retrig"] + feature.class_probs["nowin_retrig"],
        "fs_per_spin_mean_unmultiplied": sum(feature.class_mean_win.values()),
        "base_combinations": base_grid.n_combinations, "fs_combinations": fs_grid.n_combinations,
        "scatter_dist": sd.tolist(),
    }
    for k in ("p_win_ge_100x", "p_win_ge_1000x", "p_max_win"):
        stats[k + "_1_in"] = (1.0 / stats[k]) if stats[k] > 0 else float("inf")
    return ExactReport(cfg=cfg, line=line, scatter_dist=sd, base_grid=base_grid, fs_grid=fs_grid,
                       feature=feature, total_dist=total, stats=stats)


# --------------------------------------------------------------------------
# Monte-Carlo simulator (plays the game literally, spin by spin)
# --------------------------------------------------------------------------
def play_free_spins(cfg: GameConfig, rng: np.random.Generator, n0: np.ndarray) -> np.ndarray:
    """Play one free-spins feature for each entry of n0 (initial spins). Vectorised.

    returns total (unmultiplied by nothing else) feature payout per feature (x bet), uncapped.
    """
    rules: LadderRules = cfg.feature["rules"]
    k = len(n0)
    n_rem = n0.astype(np.int64).copy()
    mult = np.full(k, rules.mult_start, dtype=np.int64)
    retrig = np.zeros(k, dtype=np.int64)
    total = np.zeros(k)
    while True:
        active = np.nonzero(n_rem > 0)[0]
        if len(active) == 0:
            break
        stops = random_stops(cfg.fs_reels, len(active), rng)
        win = windows_from_stops(cfg.fs_reels, stops, cfg.rows)
        lw = evaluate_lines(win, cfg.paytable, cfg.paylines, [cfg.wild])
        sc = count_symbol(win, cfg.scatter)
        total[active] += mult[active] * lw
        won = lw > 0
        mult[active[won]] = np.minimum(mult[active[won]] + rules.mult_step, rules.mult_cap)
        can_retrig = (sc >= rules.retrigger_min) & (retrig[active] < rules.max_retriggers)
        n_rem[active[can_retrig]] += rules.retrigger_spins
        retrig[active[can_retrig]] += 1
        n_rem[active] -= 1
    return total


def spin_fn(cfg: GameConfig):
    """Return spin_fn(rng, n) -> (total wins per spin capped, info) for the simulator."""
    rules: LadderRules = cfg.feature["rules"]
    sc_pay = np.zeros(cfg.rows * cfg.n_reels + 1)
    for k, v in cfg.scatter_pays.items():
        sc_pay[k] = v
    spins_lut = np.zeros(cfg.rows * cfg.n_reels + 1, dtype=np.int64)
    for k, v in rules.spins_by_trigger.items():
        spins_lut[k] = v

    def fn(rng: np.random.Generator, n: int):
        stops = random_stops(cfg.base_reels, n, rng)
        win = windows_from_stops(cfg.base_reels, stops, cfg.rows)
        lw = evaluate_lines(win, cfg.paytable, cfg.paylines, [cfg.wild])
        sc = count_symbol(win, cfg.scatter)
        total = lw + sc_pay[sc]
        trig = np.nonzero(spins_lut[sc] > 0)[0]
        feature_wins = np.zeros(0)
        if len(trig):
            feature_wins = play_free_spins(cfg, rng, spins_lut[sc[trig]])
            total[trig] += feature_wins
        if cfg.win_cap is not None:
            np.minimum(total, cfg.win_cap, out=total)
        return total, {"feature_count": len(trig), "feature_wins": feature_wins}

    return fn
