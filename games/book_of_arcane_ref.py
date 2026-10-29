"""Reference model B — calibrated to the PUBLIC parameters of "Book of Arcane 100" (Just Slots).

Public parameters (reviews / casino pages): 5x3, 10 paylines, RTP 96.36 % (other
variants 94.33 / 92.42), hit frequency ~27 %, medium-high/high volatility,
max win 100,000x. The Book is wild AND scatter (pays 2x / 20x / 200x for 3/4/5
anywhere); 3+ Books award 10 free spins; before the feature one regular symbol
becomes the Expanding Symbol and a multiplier 1x-100x is drawn; the expanding
symbol pays on all 10 lines regardless of position; all feature wins are
multiplied; the feature can be retriggered.

This is an original model matching that public spec (strips, paytable and
multiplier weights are ours). Simplifications: the "gamble to upgrade" option
and the random base-game expanding symbol are left out.

Exactness: EVERYTHING is exact here — base game by enumeration, the feature as
an absorbing Markov chain on (spins left, retriggers used), mixed over the
expanding symbol and the multiplier, with the 100,000x cap folded exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from slotmath.engine import count_symbol, evaluate_lines, random_stops, reels_containing, windows_from_stops
from slotmath.exact import (GridEnumeration, cap_dist, convolve_capped, dist_moments, dist_quantile, dist_tail_prob,
                            enumerate_grid, reels_containing_distribution, scatter_count_distribution,
                            shift_dist, single_line_analysis)
from slotmath.game import PAYLINES_10, GameConfig
from slotmath.markov import FixedMultRules, analyze_fixed_feature, absorbing_dp_mean
from slotmath.strips import spread_strip

NAME = "Book of Arcane 100 (reference model)"
PUBLIC = {"rtp": 0.9636, "hit_rate": 0.27, "volatility": "medium-high / high", "max_win": 100_000.0,
          "free_spins": 10, "multiplier_range": "1x-100x", "book_pays": {3: 2, 4: 20, 5: 200},
          "sources": ["https://slotcatalog.com/en/slots/book-of-arcane-100", "https://bookofarcane100.com/"]}

SYMBOLS = ["B", "H1", "H2", "H3", "H4", "A", "K", "Q", "J", "T"]
B = 0
REGULAR = list(range(1, 10))                       # candidates for the expanding symbol (uniform)
PAYTABLE = np.array([
    [0, 0, 0, 0.0, 0.0, 0.0],      # B: wild on lines, pays only as scatter
    [0, 0, 1.0, 10.0, 100.0, 500.0],   # H1 wizard   (pays from 2)
    [0, 0, 0.5, 4.0, 40.0, 200.0],     # H2 witch    (pays from 2)
    [0, 0, 0, 3.0, 30.0, 100.0],       # H3 castle
    [0, 0, 0, 2.0, 15.0, 75.0],        # H4 hat
    [0, 0, 0, 0.5, 4.0, 15.0],         # A
    [0, 0, 0, 0.5, 4.0, 15.0],         # K
    [0, 0, 0, 0.5, 2.5, 10.0],         # Q
    [0, 0, 0, 0.5, 2.5, 10.0],         # J
    [0, 0, 0, 0.5, 2.5, 10.0],         # T
])
SCATTER_PAYS = {3: 2.0, 4: 20.0, 5: 200.0}
RULES = FixedMultRules(initial_spins=10, retrigger_min=3, retrigger_spins=10, max_retriggers=3)
MULT_VALUES = np.array([1, 2, 3, 5, 10, 20, 50, 100])
WIN_CAP = 100_000.0
BIN = 0.1
REELS_FILE = Path(__file__).with_name("book_of_arcane_ref_reels.json")


def default_counts() -> List[Dict[str, int]]:
    return [
        {"B": 1, "H1": 2, "H2": 3, "H3": 3, "H4": 3, "A": 4, "K": 4, "Q": 5, "J": 5, "T": 5},
        {"B": 1, "H1": 2, "H2": 3, "H3": 3, "H4": 3, "A": 4, "K": 4, "Q": 5, "J": 5, "T": 5},
        {"B": 1, "H1": 2, "H2": 3, "H3": 3, "H4": 3, "A": 4, "K": 4, "Q": 5, "J": 5, "T": 5},
        {"B": 1, "H1": 2, "H2": 3, "H3": 3, "H4": 3, "A": 4, "K": 4, "Q": 5, "J": 5, "T": 5},
        {"B": 1, "H1": 2, "H2": 3, "H3": 3, "H4": 3, "A": 4, "K": 4, "Q": 5, "J": 5, "T": 5},
    ]


def default_mult_weights() -> np.ndarray:
    return np.array([0.45, 0.25, 0.14, 0.08, 0.05, 0.02, 0.008, 0.002])


def reels_from_counts(counts) -> List[np.ndarray]:
    return [spread_strip({SYMBOLS.index(k): v for k, v in c.items() if v > 0}, seed=3 * i) for i, c in enumerate(counts)]


def build_config(reels: List[np.ndarray], mult_weights: np.ndarray) -> GameConfig:
    mult_weights = np.asarray(mult_weights, dtype=float)
    assert abs(mult_weights.sum() - 1.0) < 1e-12 and len(mult_weights) == len(MULT_VALUES)
    cfg = GameConfig(name=NAME, symbols=SYMBOLS, paytable=PAYTABLE, paylines=PAYLINES_10, base_reels=reels,
                     rows=3, wild=B, scatter=B, scatter_pays=SCATTER_PAYS, fs_reels=reels,   # same strips in the feature
                     feature={"rules": RULES, "mult_weights": mult_weights}, win_cap=WIN_CAP, bin_width=BIN)
    cfg.validate()
    return cfg


def save_config(cfg: GameConfig, path: Path = REELS_FILE) -> None:
    path.write_text(json.dumps({"reels": [[SYMBOLS[int(s)] for s in r] for r in cfg.base_reels],
                                "mult_values": MULT_VALUES.tolist(),
                                "mult_weights": cfg.feature["mult_weights"].tolist()}, indent=1))


def load_config(path: Path = REELS_FILE) -> GameConfig:
    d = json.loads(path.read_text())
    return build_config([np.array([SYMBOLS.index(n) for n in r]) for r in d["reels"]], np.array(d["mult_weights"]))


def expected_multiplier(cfg: GameConfig) -> float:
    return float((MULT_VALUES * cfg.feature["mult_weights"]).sum())


# --------------------------------------------------------------------------
# Fast (means-only) analysis for calibration
# --------------------------------------------------------------------------
def quick_rtp(cfg: GameConfig) -> Dict[str, float]:
    """Exact line and scatter means; feature mean under a small independence approximation
    (line wins and retriggers assumed independent within a spin) — fixed later by `solve_mult_weights`."""
    la = single_line_analysis(cfg.base_reels, cfg.paytable, [B])
    sd = scatter_count_distribution(cfg.base_reels, B, cfg.rows)
    line = cfg.n_lines * la.ev
    scat = float(sum(sd[k] * v for k, v in SCATTER_PAYS.items()))
    p_trig = float(sd[3:].sum())
    # per-spin unmultiplied feature mean for each expanding symbol
    ev_e = []
    for e in REGULAR:
        rc = reels_containing_distribution(cfg.base_reels, e, cfg.rows)
        exp_mean = float(sum(rc[c] * cfg.n_lines * cfg.paytable[e, c] for c in range(len(rc))))
        ev_e.append(line + exp_mean)
    spin_mean = float(np.mean(ev_e))
    r = RULES

    def nxt(n, rr, retrig):
        n2, r2 = n - 1, rr
        if retrig and rr < r.max_retriggers:
            n2 += r.retrigger_spins; r2 += 1
        return None if n2 == 0 else (n2, r2)

    spins = absorbing_dp_mean((r.initial_spins, 0), lambda s: [(1 - p_trig, 1 - p_trig, nxt(*s, False)), (p_trig, p_trig, nxt(*s, True))])
    feat = p_trig * expected_multiplier(cfg) * spin_mean * spins
    return {"line": line, "scatter": scat, "feature_approx": feat, "total_approx": line + scat + feat,
            "p_trigger": p_trig, "expected_fs_spins": spins, "line_hit": la.hit_prob}


# --------------------------------------------------------------------------
# Exact analysis
# --------------------------------------------------------------------------
def _extra_fn_factory(cfg: GameConfig, nb: int):
    """Per expanding symbol e: joint histogram of (retrigger flag, unmultiplied spin payout bin)."""
    def extra(window, lw, sc):
        retrig = (sc >= RULES.retrigger_min).astype(np.int64)
        out = {}
        for e in REGULAR:
            c = reels_containing(window, e)
            pay = lw + cfg.n_lines * cfg.paytable[e, c]
            b = np.rint(pay / cfg.bin_width).astype(np.int64)
            out[f"e{e}"] = (retrig * nb + b, 2 * nb)
        return out
    return extra


def scale_dist_capped(dist: np.ndarray, m: int, cap_bins: int) -> np.ndarray:
    """Distribution of min(m*X, cap) on the bin grid, without materialising m*len bins."""
    out = np.zeros(cap_bins + 1)
    idx = np.arange(len(dist)) * m
    ok = idx <= cap_bins
    out[idx[ok]] = dist[ok]
    out[cap_bins] += dist[~ok].sum()
    return out


@dataclass
class ExactReport:
    cfg: GameConfig
    grid: GridEnumeration
    feature_dist: np.ndarray            # exact distribution of the (multiplied, capped) feature payout
    feature_unscaled_by_symbol: Dict[int, np.ndarray]
    total_dist: np.ndarray
    stats: Dict


NB_SPIN = 80_000                                 # bins for one spin's unmultiplied payout (<= 8,000x)


def enumerate_base(cfg: GameConfig) -> GridEnumeration:
    """Full enumeration: base joint histogram + per-expanding-symbol feature-spin histograms."""
    return enumerate_grid(cfg.base_reels, cfg.rows, cfg.paytable, cfg.paylines, [B], B, cfg.bin_width,
                          extra_fn=_extra_fn_factory(cfg, NB_SPIN))


def feature_unscaled(cfg: GameConfig, grid: GridEnumeration, verbose: bool = True):
    """Exact UNMULTIPLIED feature distribution for every expanding symbol (independent of the
    multiplier weights, so calibration of the weights does not need to redo this)."""
    log = print if verbose else (lambda *a, **k: None)
    cap_bins = int(round(cfg.win_cap / cfg.bin_width))
    unscaled, ev_unscaled, spins_e = {}, {}, {}
    for e in REGULAR:
        h = grid.extra[f"e{e}"].reshape(2, NB_SPIN)
        last = np.nonzero(h.sum(axis=0) > 0)[0].max()
        d_no, d_re = h[0, : last + 1], h[1, : last + 1]
        dist, ev_bins, spins = analyze_fixed_feature((d_no, d_re), RULES, cap_bins)
        unscaled[e] = dist
        ev_unscaled[e] = ev_bins * cfg.bin_width
        spins_e[e] = spins
        log(f"      symbol {SYMBOLS[e]}: E[unmultiplied feature] = {ev_unscaled[e]:.2f}x, E[spins] = {spins:.3f}")
    return unscaled, ev_unscaled, spins_e


def exact_analysis(cfg: GameConfig, verbose: bool = True, grid: GridEnumeration = None,
                   unscaled_pack=None) -> ExactReport:
    log = print if verbose else (lambda *a, **k: None)
    cap_bins = int(round(cfg.win_cap / cfg.bin_width))
    if grid is None:
        log("[1/3] full enumeration of all stop combinations (base game + per-symbol feature spins) ...")
        grid = enumerate_base(cfg)
    log(f"      {grid.n_combinations:,} combinations")
    if unscaled_pack is None:
        log("[2/3] feature Markov chain for each of the 9 expanding symbols ...")
        unscaled_pack = feature_unscaled(cfg, grid, verbose)
    unscaled, ev_unscaled, spins_e = unscaled_pack

    log("[3/3] mixing over symbol and multiplier, assembling the spin distribution ...")
    w = cfg.feature["mult_weights"]
    feat = np.zeros(cap_bins + 1)
    for e in REGULAR:
        for m, wm in zip(MULT_VALUES, w):
            feat += (wm / len(REGULAR)) * scale_dist_capped(unscaled[e], int(m), cap_bins)
    sd = grid.scatter_dist
    total = np.zeros(1)
    for s in range(grid.joint.shape[0]):
        row = grid.joint[s]
        if row.sum() == 0:
            continue
        if s >= RULES.retrigger_min:
            part = convolve_capped(shift_dist(row, int(round(SCATTER_PAYS.get(s, 0.0) / cfg.bin_width))), feat, cap_bins)
        else:
            part = row
        if len(part) > len(total):
            part = part.copy(); part[: len(total)] += total; total = part
        else:
            total[: len(part)] += part
    total = cap_dist(total, cap_bins)

    values = np.arange(len(total)) * cfg.bin_width
    rtp = float((total * values).sum())
    p_trig = float(sd[3:].sum())
    la = single_line_analysis(cfg.base_reels, cfg.paytable, [B])
    line_rtp = cfg.n_lines * la.ev
    scat_rtp = float(sum(sd[k] * v for k, v in SCATTER_PAYS.items()))
    EM = expected_multiplier(cfg)
    feat_ev_uncapped = EM * float(np.mean(list(ev_unscaled.values())))
    f_mean, f_var, f_std = dist_moments(feat, cfg.bin_width)
    mean, var, std = dist_moments(total, cfg.bin_width)
    stats = {
        "rtp_exact_capped": rtp, "rtp_exact_uncapped": line_rtp + scat_rtp + p_trig * feat_ev_uncapped,
        "rtp_line": line_rtp, "rtp_scatter": scat_rtp, "rtp_feature": p_trig * feat_ev_uncapped,
        "cap_cost": line_rtp + scat_rtp + p_trig * feat_ev_uncapped - rtp,
        "hit_rate": float(1 - total[0]), "std_per_spin": std, "variance_per_spin": var,
        "p_trigger": p_trig, "trigger_1_in": 1 / p_trig,
        "p_trigger_by_count": {k: float(sd[k]) for k in (3, 4, 5)},
        "expected_multiplier": EM, "mult_values": MULT_VALUES.tolist(), "mult_weights": w.tolist(),
        "feature_ev_unscaled_by_symbol": {SYMBOLS[e]: v for e, v in ev_unscaled.items()},
        "feature_expected_spins": float(np.mean(list(spins_e.values()))),
        "feature_mean": f_mean, "feature_std": f_std, "feature_median": dist_quantile(feat, cfg.bin_width, 0.5),
        "feature_p_zero": float(feat[0]),
        "p_win_ge_100x": dist_tail_prob(total, cfg.bin_width, 100.0),
        "p_win_ge_1000x": dist_tail_prob(total, cfg.bin_width, 1000.0),
        "p_win_ge_10000x": dist_tail_prob(total, cfg.bin_width, 10_000.0),
        "p_max_win": float(total[cap_bins]) if len(total) > cap_bins else 0.0,
        "bonus_buy_1000x_rtp": f_mean / 1000.0,
        "base_combinations": grid.n_combinations, "max_line_win_single_spin": grid.max_line_win,
        "public": PUBLIC, "scatter_dist": sd.tolist(),
    }
    for k in ("p_win_ge_100x", "p_win_ge_1000x", "p_win_ge_10000x", "p_max_win"):
        stats[k + "_1_in"] = (1 / stats[k]) if stats[k] > 0 else float("inf")
    return ExactReport(cfg=cfg, grid=grid, feature_dist=feat, feature_unscaled_by_symbol=unscaled,
                       total_dist=total, stats=stats)


def solve_mult_weights(cfg: GameConfig, rep: ExactReport, target: float) -> np.ndarray:
    """Choose multiplier weights so that the exact UNCAPPED RTP equals `target`.

    Feature RTP is linear in E[M]; we keep the shape of the weight vector for
    multipliers >= 3 and move mass between 1x and 2x to hit the required E[M].
    """
    s = rep.stats
    K = s["p_trigger"] * float(np.mean(list(rep.stats["feature_ev_unscaled_by_symbol"].values())))
    need_EM = (target - s["rtp_line"] - s["rtp_scatter"]) / K
    w = cfg.feature["mult_weights"].copy()
    rest = w[2:].copy()
    for _ in range(60):                                       # shrink the big multipliers if infeasible
        EM_rest = float((MULT_VALUES[2:] * rest).sum())
        mass = 1.0 - rest.sum()                               # to split between 1x and 2x
        # w1*1 + w2*2 = need_EM - EM_rest,  w1 + w2 = mass  ->  w2 = need_EM - EM_rest - mass
        w2 = need_EM - EM_rest - mass
        w1 = mass - w2
        if 0.0 <= w1 <= 1.0 and 0.0 <= w2 <= 1.0:
            break
        rest *= 0.9 if w2 < 0 else 1.1
    assert 0 <= w1 <= 1 and 0 <= w2 <= 1, f"target not reachable: w1={w1}, w2={w2}"
    w[0], w[1] = w1, w2
    w[2:] = rest
    return w


# --------------------------------------------------------------------------
# Full-game simulator
# --------------------------------------------------------------------------
def play_free_spins(cfg: GameConfig, rng: np.random.Generator, k: int) -> np.ndarray:
    """Play k features: random expanding symbol and multiplier, retriggers, cap applied by caller."""
    r = RULES
    e = rng.choice(REGULAR, size=k)
    m = rng.choice(MULT_VALUES, size=k, p=cfg.feature["mult_weights"])
    n_rem = np.full(k, r.initial_spins, dtype=np.int64)
    retrig = np.zeros(k, dtype=np.int64)
    total = np.zeros(k)
    while True:
        active = np.nonzero(n_rem > 0)[0]
        if len(active) == 0:
            break
        stops = random_stops(cfg.fs_reels, len(active), rng)
        win = windows_from_stops(cfg.fs_reels, stops, cfg.rows)
        lw = evaluate_lines(win, cfg.paytable, cfg.paylines, [B])
        ea = e[active]
        c = (win == ea[:, None, None]).any(axis=1).sum(axis=1)
        pay = lw + cfg.n_lines * cfg.paytable[ea, c]
        total[active] += pay
        sc = count_symbol(win, B)
        can = (sc >= r.retrigger_min) & (retrig[active] < r.max_retriggers)
        n_rem[active[can]] += r.retrigger_spins
        retrig[active[can]] += 1
        n_rem[active] -= 1
    return total * m


def spin_fn(cfg: GameConfig):
    sc_pay = np.zeros(cfg.rows * cfg.n_reels + 1)
    for k, v in SCATTER_PAYS.items():
        sc_pay[k] = v

    def fn(rng, n):
        stops = random_stops(cfg.base_reels, n, rng)
        win = windows_from_stops(cfg.base_reels, stops, cfg.rows)
        lw = evaluate_lines(win, cfg.paytable, cfg.paylines, [B])
        sc = count_symbol(win, B)
        total = lw + sc_pay[sc]
        trig = np.nonzero(sc >= RULES.retrigger_min)[0]
        fw = np.zeros(0)
        if len(trig):
            fw = play_free_spins(cfg, rng, len(trig))
            total[trig] += fw
        np.minimum(total, cfg.win_cap, out=total)
        return total, {"feature_count": len(trig), "feature_wins": fw}

    return fn
