"""Generate docs/PAR_SHEET_*.md from the saved exact analyses (+ simulation JSON if present).

Usage: python scripts/make_par_sheets.py
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slotmath import par_sheet as ps                       # noqa: E402
from slotmath.exact import dist_quantile, dist_tail_prob   # noqa: E402


def latest_sim(prefix: str):
    files = sorted(glob.glob(str(ROOT / "results" / f"{prefix}_sim_*.json")), key=lambda p: json.load(open(p))["n_spins"])
    return json.load(open(files[-1])) if files else None


def kastel() -> None:
    from games import kastel_fortune as kf
    rep = pickle.load(open(ROOT / "results" / "kastel_exact.pkl", "rb"))
    cfg, s = rep.cfg, rep.stats
    sim = latest_sim("kastel")
    r = kf.RULES
    md = [f"# PAR sheet — {cfg.name}", f"_Generated {date.today()} by `scripts/make_par_sheets.py` from the exact analysis in `results/kastel_exact.pkl`._", ""]
    md += ["## 1. Game summary", "",
           ps.section_game_summary(cfg, s, [
               ["**RTP (exact, incl. max-win cap)**", f"**{ps.pct(s['rtp_exact_capped'], 4)}**"],
               ["RTP split", f"lines {ps.pct(s['rtp_line'])} + scatter pays {ps.pct(s['rtp_scatter'])} + free spins {ps.pct(s['rtp_feature'])}"],
               ["Cap cost", f"{100*s['cap_cost']:.6f} pp (uncapped RTP {ps.pct(s['rtp_exact_uncapped'], 4)})"],
               ["Hit frequency (any win)", ps.pct(s["hit_rate"])],
               ["Std dev per spin", f"{s['std_per_spin']:.4f}x bet  (volatility index 1.96σ = {1.96*s['std_per_spin']:.2f})"],
               ["Free-spins frequency", f"{ps.one_in(s['p_trigger'])} spins ({ps.pct(s['p_trigger'], 4)})"],
               ["Free spins awarded", ", ".join(f"{k} scatters → {v} spins" for k, v in r.spins_by_trigger.items())],
               ["Multiplier ladder", f"starts x{r.mult_start}, +{r.mult_step} after every winning spin, cap x{r.mult_cap}"],
               ["Retrigger", f"{r.retrigger_min}+ scatters in free spins → +{r.retrigger_spins} spins, max {r.max_retriggers} retriggers"],
               ["Base combinations", f"{s['base_combinations']:,} (all enumerated)"],
               ["Free-spin combinations", f"{s['fs_combinations']:,} (all enumerated)"],
           ]), ""]
    md += ["## 2. Paytable (x total bet)", "", ps.section_paytable(cfg, kf.SYMBOL_NAMES), ""]
    md += ["## 3. Reel strips", "", ps.section_strips(cfg, cfg.base_reels, "Base game"), "", ps.section_strips(cfg, cfg.fs_reels, "Free spins"), ""]
    md += ["## 4. Line-win contributions (base game, exact)", "",
           "Probabilities are for one payline; the RTP share is summed over all 20 lines (every line has the same distribution).", "",
           ps.section_line_contributions(cfg, rep.line), ""]
    md += ["## 5. Scatters (base game, exact)", "", ps.section_scatter(cfg, rep.scatter_dist), ""]
    f = rep.feature
    cp = f.class_probs
    md += ["## 6. Free spins (exact, absorbing Markov chain)", "",
           "One free spin has four outcome classes; their exact probabilities (from full enumeration of the free-spin strips):", "",
           ps.md_table(["Class", "Probability", "E[line win × 1{class}] (unmultiplied)"],
                       [[k, f"{v:.6f}", f"{f.class_mean_win[k]:.6f}x"] for k, v in cp.items()], "lrr"), "",
           f"Per free spin: P(line win) = {ps.pct(s['fs_per_spin_win_prob'])}, P(retrigger) = {ps.pct(s['fs_per_spin_retrigger_prob'])}, E[unmultiplied line win] = {s['fs_per_spin_mean_unmultiplied']:.4f}x.", "",
           "State = (spins remaining, multiplier, retriggers used). Feature value by trigger size:", "",
           ps.md_table(["Trigger", "P(trigger)", "Initial spins", "E[total spins]", "E[payout]", "median", "P(payout = 0)", "P(≥100x)", "P(≥500x)"],
                       [[f"{k} scatters", f"{s['p_trigger_by_count'][k]:.6f}", r.spins_by_trigger[k], f"{f.spins_by_trigger[k]:.3f}",
                         f"{f.ev_by_trigger[k]:.2f}x", f"{dist_quantile(f.dist_by_trigger[k], cfg.bin_width, 0.5):g}x",
                         f"{f.dist_by_trigger[k][0]:.4f}", f"{dist_tail_prob(f.dist_by_trigger[k], cfg.bin_width, 100):.4f}",
                         f"{dist_tail_prob(f.dist_by_trigger[k], cfg.bin_width, 500):.5f}"] for k in r.spins_by_trigger], "lrrrrrrrr"), "",
           f"Averaged over trigger sizes: E[payout | feature] = {s['feature_mean_given_trigger']:.2f}x, std {s['feature_std_given_trigger']:.2f}x, median {s['feature_median_given_trigger']:g}x, P(zero) = {s['feature_p_zero']:.4f}.", ""]
    md += ["## 7. Distribution of the total win of a base-game spin (exact)", "",
           "Includes line wins, scatter pays and the whole free-spins feature it may trigger; the max-win cap is applied.", "",
           ps.section_distribution(rep.total_dist, cfg.bin_width, cfg.win_cap), ""]
    md += ["## 8. Exact vs Monte-Carlo", "",
           ps.section_exact_vs_sim({"rtp": s["rtp_exact_capped"], "hit": s["hit_rate"], "std": s["std_per_spin"],
                                    "trigger_1_in": s["trigger_1_in"], "feature_rtp": s["rtp_feature"]}, sim), ""]
    md += ["## 9. Assumptions and notes", "",
           "- Each reel stops uniformly at random; reels are independent; the visible window is 3 consecutive strip positions.",
           "- Lines pay left to right only, highest win per line; wild substitutes for all symbols except the scatter and has no pays of its own (wilds sit on reels 2-4 only, so a pure-wild run starting on reel 1 cannot occur).",
           "- Companion Excel model: docs/Kastel_Fortune_math_model.xlsx reproduces the RTP decomposition and the free-spins Markov chain with live formulas.",
           "- Scatter pays are added to line wins. Scatters do not pay inside free spins (they only retrigger).",
           "- The multiplier applies to the line wins of the spin on which it is active and climbs *after* a winning spin.",
           "- The max-win cap truncates the total win of a spin including its feature; the exact distribution folds the tail into the cap bin.",
           f"- Max-win reachability: P(hit the {cfg.win_cap:,.0f}x cap) = {ps.one_in(s['p_max_win'])} spins. With a 5,000x cap it would be 1 in ~6e16 — unreachable — which is why the design uses 1,500x.",
           ""]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "PAR_SHEET_kastel_fortune.md").write_text("\n".join(md))
    print("wrote docs/PAR_SHEET_kastel_fortune.md")


def sugar() -> None:
    from games import sugar_heaven_ref as sh
    rep = pickle.load(open(ROOT / "results" / "sugar_exact.pkl", "rb"))
    cfg, s = rep.cfg, rep.stats
    sim = latest_sim("sugar")
    pub = s["public"]
    md = [f"# PAR sheet — {cfg.name}", f"_Generated {date.today()}. Original model fitted to the public spec of Just Slots' Sugar Heaven ({pub['source']}); strips and paytable are ours._", ""]
    md += ["## 1. Game summary", "",
           ps.section_game_summary(cfg, s, [
               ["**RTP**", f"**{ps.pct(s['rtp'], 3)} ± {100*s['rtp_ci95_half']:.3f} pp** (public: {ps.pct(pub['rtp'], 2)})"],
               ["RTP split", f"lines {ps.pct(s['rtp_line'])} (exact) + scatters {ps.pct(s['rtp_scatter'])} (exact) + free spins {ps.pct(s['rtp_feature'])} (simulated, 400k features/trigger size)"],
               ["Hit frequency (exact)", f"{ps.pct(s['hit_rate_exact'], 2)} (public: {ps.pct(pub['hit_rate'], 0)})"],
               ["Volatility", f"high (public: {pub['volatility']}); simulated std/spin see §5"],
               ["Free-spins frequency", f"{ps.one_in(s['p_trigger'])} spins"],
               ["Free spins awarded", ", ".join(f"{k} scatters → {v} spins" for k, v in sh.FS_SPINS.items()) + " (public spec), wilds sticky, no retrigger"],
               ["Multiplier wilds", "W2/W3/W5/W10 carry x2/x3/x5/x10; multipliers on one winning line are added"],
               ["Bonus buy", f"100x → feature RTP {ps.pct(s['bonus_buy_100x_rtp'], 2)};  300x (5 scatters) → {ps.pct(s['bonus_buy_300x_rtp'], 2)}"],
               ["Base combinations", f"{s['base_combinations']:,} (all enumerated)"],
           ]), ""]
    md += ["## 2. Paytable (x total bet)", "", ps.section_paytable(cfg), ""]
    md += ["## 3. Reel strips", "", ps.section_strips(cfg, cfg.base_reels, "Base game"), "", ps.section_strips(cfg, cfg.fs_reels, "Free spins"), ""]
    md += ["## 4. Feature value (simulation, 95% CI)", "",
           ps.md_table(["Trigger", "P(trigger)", "Spins", "E[payout]", "± (95% CI)"],
                       [[f"{k} scatters", f"{s['p_trigger_by_count'][k]:.6f}", sh.FS_SPINS[k], f"{s['feature_ev_by_count'][k]:.2f}x", f"{s['feature_ev_ci_by_count'][k]:.2f}x"] for k in sh.FS_SPINS], "lrrrr"), ""]
    md += ["## 5. Exact/semi-exact vs Monte-Carlo", "",
           ps.section_exact_vs_sim({"rtp": s["rtp"], "hit": s["hit_rate_exact"], "std": None, "trigger_1_in": s["trigger_1_in"], "feature_rtp": s["rtp_feature"]}, sim), ""]
    md += ["## 6. Notes", "",
           "- Base game is exact (full enumeration with the wild multipliers carried by the strips).",
           "- The sticky-wild feature's state (which cells hold which multiplier) is too large to enumerate; it is valued by simulation with a confidence interval — standard practice for sticky/collect features.",
           "- Left out of the model: Super Wild enhancers and the random base-game wild drop of the real game.", ""]
    (ROOT / "docs" / "PAR_SHEET_sugar_heaven_ref.md").write_text("\n".join(md))
    print("wrote docs/PAR_SHEET_sugar_heaven_ref.md")


def book() -> None:
    from games import book_of_arcane_ref as bk
    rep = pickle.load(open(ROOT / "results" / "book_exact.pkl", "rb"))
    cfg, s = rep.cfg, rep.stats
    sim = latest_sim("book")
    pub = s["public"]
    md = [f"# PAR sheet — {cfg.name}", f"_Generated {date.today()}. Original model fitted to the public spec of Just Slots' Book of Arcane 100; strips, paytable and multiplier weights are ours._", ""]
    md += ["## 1. Game summary", "",
           ps.section_game_summary(cfg, s, [
               ["**RTP (exact, incl. cap)**", f"**{ps.pct(s['rtp_exact_capped'], 4)}** (public: {ps.pct(pub['rtp'], 2)})"],
               ["RTP split", f"lines {ps.pct(s['rtp_line'])} + Book scatter pays {ps.pct(s['rtp_scatter'])} + free spins {ps.pct(s['rtp_feature'])}"],
               ["Cap cost", f"{100*s['cap_cost']:.6f} pp"],
               ["Hit frequency (exact)", f"{ps.pct(s['hit_rate'], 2)} (public: ~{ps.pct(pub['hit_rate'], 0)})"],
               ["Std dev per spin", f"{s['std_per_spin']:.3f}x bet (volatility index 1.96σ = {1.96*s['std_per_spin']:.1f})"],
               ["Free-spins frequency", f"{ps.one_in(s['p_trigger'])} spins"],
               ["Free spins", f"{bk.RULES.initial_spins} spins, expanding symbol uniform over the 9 regular symbols, retrigger +{bk.RULES.retrigger_spins} (max {bk.RULES.max_retriggers})"],
               ["Feature multiplier", "drawn once per feature: " + ", ".join(f"x{m}: {100*w:.3f}%" for m, w in zip(s["mult_values"], s["mult_weights"])) + f" → E[M] = {s['expected_multiplier']:.4f}"],
               ["E[feature payout]", f"{s['feature_mean']:.2f}x (median {s['feature_median']:g}x, std {s['feature_std']:.1f}x); expected spins {s['feature_expected_spins']:.3f}"],
               ["Fair bonus-buy price", f"{s['feature_mean']/s['rtp_exact_capped']:.1f}x for the plain feature (buy RTP = game RTP); the real game's 1,000x 'Arcane 100' buy guarantees the wild as expanding symbol, which this model does not include"],
               ["Combinations", f"{s['base_combinations']:,} (all enumerated; the feature uses the same strips)"],
           ]), ""]
    md += ["## 2. Paytable (x total bet, 10 lines)", "", ps.section_paytable(cfg), ""]
    md += ["## 3. Reel strips (base game and free spins)", "", ps.section_strips(cfg, cfg.base_reels, "Strips"), ""]
    md += ["## 4. Expanding symbol: unmultiplied feature value by symbol (exact)", "",
           ps.md_table(["Expanding symbol", "E[feature payout] before multiplier"],
                       [[k, f"{v:.2f}x"] for k, v in s["feature_ev_unscaled_by_symbol"].items()], "lr"), ""]
    md += ["## 5. Distribution of the total win of a base-game spin (exact)", "",
           ps.section_distribution(rep.total_dist, cfg.bin_width, cfg.win_cap), ""]
    md += ["## 6. Exact vs Monte-Carlo", "",
           ps.section_exact_vs_sim({"rtp": s["rtp_exact_capped"], "hit": s["hit_rate"], "std": s["std_per_spin"], "trigger_1_in": s["trigger_1_in"], "feature_rtp": s["rtp_feature"]}, sim), ""]
    md += ["## 7. Notes", "",
           "- The Book is wild on lines and scatter anywhere (pays 2x/20x/200x for 3/4/5 — public spec).",
           "- In free spins the expanding symbol pays `paytable[symbol][number of reels containing it] × 10 lines` in addition to regular line wins; all feature wins are multiplied by the feature multiplier.",
           "- The feature multiplier weights were solved so that the exact RTP equals the public 96.36 % (feature RTP is linear in E[M]).",
           "- Left out: the gamble-to-upgrade option and the random base-game expanding symbol of the real game.", ""]
    (ROOT / "docs" / "PAR_SHEET_book_of_arcane_ref.md").write_text("\n".join(md))
    print("wrote docs/PAR_SHEET_book_of_arcane_ref.md")


if __name__ == "__main__":
    which = sys.argv[1:] or ["kastel", "sugar", "book"]
    for w in which:
        try:
            {"kastel": kastel, "sugar": sugar, "book": book}[w]()
        except FileNotFoundError as e:
            print(f"skipping {w}: {e}")
