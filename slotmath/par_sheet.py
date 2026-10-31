"""PAR-sheet generation (Markdown).

A PAR sheet: everything needed for a test lab/regulator to verify the game's RTP from first
principles — reel strips, paytable, every symbol's probability per reel, the
contribution of every winning combination, feature rules and statistics, and
the resulting payout distribution.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .exact import LineAnalysis, dist_quantile, dist_tail_prob, reel_marginals
from .game import GameConfig


def md_table(header: Sequence[str], rows: Sequence[Sequence], align: Optional[str] = None) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |"]
    out.append("|" + "|".join(["---:" if (align or "r")[min(i, len(align or "r") - 1)] == "r" else ":---" for i in range(len(header))]) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def pct(x: float, d: int = 3) -> str:
    return f"{100*x:.{d}f}%"


def one_in(p: float) -> str:
    return "never" if p <= 0 else f"1 in {1/p:,.0f}" if p < 0.01 else f"1 in {1/p:.2f}"


def section_game_summary(cfg: GameConfig, stats: Dict, extra_rows: Sequence[Sequence] = ()) -> str:
    rows = [
        ["Grid", f"{cfg.n_reels} reels x {cfg.rows} rows"],
        ["Paylines", f"{cfg.n_lines} (fixed), pays left to right"],
        ["Bet unit", "all pays in multiples of the total bet"],
        ["Max win cap", f"{cfg.win_cap:,.0f}x" if cfg.win_cap else "none"],
    ] + list(extra_rows)
    return md_table(["Parameter", "Value"], rows, "ll")


def section_paytable(cfg: GameConfig, names: Optional[Dict[str, str]] = None) -> str:
    rows = []
    for i, s in enumerate(cfg.symbols):
        pays = cfg.paytable[i]
        if pays.sum() == 0:
            continue                      # non-paying line symbol (e.g. the scatter)
        label = f"{s} — {names[s]}" if names and s in names else s
        rows.append([label] + [f"{p:g}" if p else "–" for p in pays[2:]])
    txt = md_table(["Symbol", "2 of a kind", "3 of a kind", "4 of a kind", "5 of a kind"], rows, "lrrrr")
    if cfg.scatter_pays:
        txt += "\n\nScatter pays (anywhere on the grid, x total bet): " + ", ".join(f"{k} scatters = {v:g}x" for k, v in sorted(cfg.scatter_pays.items()))
    return txt


def section_strips(cfg: GameConfig, reels: List[np.ndarray], title: str) -> str:
    counts = cfg.strips_table(reels)
    marg = reel_marginals(reels, cfg.n_symbols)
    rows = []
    for i, s in enumerate(cfg.symbols):
        if counts[:, i].sum() == 0:
            continue
        rows.append([s] + [f"{counts[r, i]} ({100*marg[r, i]:.2f}%)" for r in range(cfg.n_reels)])
    rows.append(["**strip length**"] + [f"**{len(r)}**" for r in reels])
    txt = f"**{title}** — count per reel (probability that a given row of that reel shows the symbol):\n\n"
    txt += md_table(["Symbol"] + [f"Reel {r+1}" for r in range(cfg.n_reels)], rows, "l" + "r" * cfg.n_reels)
    txt += "\n\nFull strips (top to bottom, wrapping):\n\n"
    for r, strip in enumerate(reels):
        txt += f"- Reel {r+1}: `" + " ".join(cfg.symbols[int(s)] for s in strip) + "`\n"
    return txt


def section_line_contributions(cfg: GameConfig, la: LineAnalysis) -> str:
    """Per (symbol, count): probability on ONE line, expected pay per line, RTP contribution over all lines."""
    rows = []
    total = 0.0
    for i, s in enumerate(cfg.symbols):
        for k in (2, 3, 4, 5):
            p = la.prob_symbol_run[i, k]
            ev = la.by_symbol_run[i, k]
            if ev <= 0:
                continue
            total += ev * cfg.n_lines
            rows.append([f"{s} x{k}", f"{p:.6e}", one_in(p), f"{cfg.paytable[i, k]:g}", f"{ev*cfg.n_lines:.6f}", pct(ev * cfg.n_lines, 4)])
    rows.append(["**total**", "", f"line hit {pct(la.hit_prob, 3)}", "", f"**{total:.6f}**", f"**{pct(total, 4)}**"])
    return md_table(["Combination", "P (one line)", "", "Pay", "E[win] all lines", "RTP share"], rows, "lrrrrr")


def section_scatter(cfg: GameConfig, sd: np.ndarray) -> str:
    rows = []
    for k, p in enumerate(sd):
        if p <= 0:
            continue
        pay = cfg.scatter_pays.get(k, 0.0)
        rows.append([k, f"{p:.6e}", one_in(p), f"{pay:g}" if pay else "–", f"{p*pay:.6f}" if pay else "–"])
    return md_table(["Scatters", "Probability", "", "Pay", "RTP share"], rows, "rrrrr")


def section_distribution(dist: np.ndarray, bin_width: float, cap: Optional[float], thresholds=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000)) -> str:
    rows = []
    for t in thresholds:
        if cap is not None and t > cap:
            continue
        p = dist_tail_prob(dist, bin_width, t)
        rows.append([f">= {t:,}x", f"{p:.4e}", one_in(p)])
    if cap is not None:
        b = int(round(cap / bin_width))
        p = float(dist[b]) if len(dist) > b else 0.0
        rows.append([f"= max win {cap:,.0f}x", f"{p:.4e}", one_in(p)])
    q = [dist_quantile(dist, bin_width, x) for x in (0.5, 0.9, 0.99, 0.999, 0.9999)]
    txt = md_table(["Win per spin", "Probability", ""], rows, "lrr")
    txt += "\n\nQuantiles of the per-spin win: " + ", ".join(f"q{x}: {v:g}x" for x, v in zip(("50", "90", "99", "99.9", "99.99"), q))
    return txt


def section_exact_vs_sim(exact: Dict[str, float], sim: Optional[Dict]) -> str:
    if not sim:
        return "_Simulation results not available yet — run `scripts/run_sim.py`._"
    rows = []
    half = sim["ci95_half"]
    rtp_e = exact.get("rtp")
    rows.append(["RTP", pct(rtp_e, 4) if rtp_e is not None else "–", f"{pct(sim['rtp'], 4)} ± {100*half:.4f} pp",
                 ("inside CI" if rtp_e is not None and abs(sim["rtp"] - rtp_e) <= half else "OUTSIDE CI") if rtp_e is not None else ""])
    if exact.get("hit") is not None:
        rows.append(["Hit rate", pct(exact["hit"], 3), pct(sim["hit_rate"], 3), ""])
    if exact.get("std") is not None:
        rows.append(["Std dev / spin", f"{exact['std']:.4f}x", f"{sim['std']:.4f}x", ""])
    if exact.get("trigger_1_in") is not None and sim.get("feature_freq_spins"):
        rows.append(["Feature frequency", f"1 in {exact['trigger_1_in']:.1f}", f"1 in {sim['feature_freq_spins']:.1f}", ""])
    if exact.get("feature_rtp") is not None and sim.get("feature_rtp") is not None:
        rows.append(["Feature RTP share", pct(exact["feature_rtp"], 3), pct(sim["feature_rtp"], 3), ""])
    rows.append(["Max win observed", "", f"{sim['max_win']:,.1f}x", ""])
    return md_table(["Quantity", "Exact", f"Simulation ({sim['n_spins']:,} spins, seed {sim['seed']})", ""], rows, "llll")
