"""Unit tests. Run with  `python -m unittest discover -s tests -v`  (or pytest)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slotmath.engine import count_symbol, evaluate_lines, line_outcome, random_stops, windows_from_stops   # noqa: E402
from slotmath.exact import (convolve_capped, enumerate_grid, scale_dist, scatter_count_distribution,          # noqa: E402
                            single_line_analysis)
from slotmath.markov import LadderRules, analyze_ladder_feature, ladder_feature_ev_fast                        # noqa: E402

# tiny test game: W=0 S=1 A=2 B=3
PT = np.array([
    [0, 0, 0, 10.0, 50.0, 200.0],   # W
    [0, 0, 0, 0.0, 0.0, 0.0],       # S
    [0, 0, 0, 2.0, 5.0, 20.0],      # A
    [0, 0, 0, 1.0, 3.0, 10.0],      # B
])
LINES3 = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2]])


class TestLineEvaluation(unittest.TestCase):
    def line(self, syms, mult=None):
        win, cand, run = line_outcome(np.array([syms]), PT, [0], mult)
        return float(win[0]), int(cand[0]), int(run[0])

    def test_three_of_a_kind(self):
        self.assertEqual(self.line([2, 2, 2, 3, 3]), (2.0, 2, 3))

    def test_no_win_when_broken_left(self):
        self.assertEqual(self.line([3, 2, 2, 2, 2])[0], 0.0)       # B then A A A A: B run = 1 -> no pay

    def test_wild_substitution(self):
        self.assertEqual(self.line([2, 0, 2, 0, 2]), (20.0, 2, 5))  # A W A W A = 5 x A

    def test_pure_wild_beats_candidate(self):
        # W W W B B: candidate B run = 5 -> 10x, pure wild run 3 -> 10x: equal, keep candidate
        self.assertEqual(self.line([0, 0, 0, 3, 3])[0], 10.0)
        # W W W B A: B run = 4 -> 3x, pure wild 3 -> 10x: wild wins
        self.assertEqual(self.line([0, 0, 0, 3, 2]), (10.0, 0, 3))

    def test_scatter_breaks_line(self):
        self.assertEqual(self.line([2, 2, 1, 2, 2])[0], 0.0)

    def test_multiplier_sum(self):
        mult = np.array([3, 1, 1, 1])                               # wild carries x3
        self.assertEqual(self.line([2, 0, 2, 3, 3], mult)[0], 2.0 * 3)   # A W A -> 3oak A, one wild x3
        mult2 = np.array([2, 1, 1, 1])
        self.assertEqual(self.line([0, 0, 2, 3, 3], mult2)[0], 2.0 * (2 + 2))   # two wilds x2 each -> x4

    def test_evaluate_lines_sums_lines(self):
        window = np.array([[[2, 2, 2, 3, 3], [3, 3, 3, 3, 2], [1, 1, 1, 1, 1]]], dtype=np.int16)
        self.assertEqual(float(evaluate_lines(window, PT, LINES3, [0])[0]), 3.0 + 2.0)   # row1: B x4 = 3, row0: A x3 = 2


class TestExact(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.reels = [rng.integers(0, 4, size=7) for _ in range(5)]
        for r in self.reels:                                       # make sure every reel has a scatter and a symbol mix
            r[0] = 1

    def test_tuple_ev_equals_enumeration_ev(self):
        la = single_line_analysis(self.reels, PT, [0])
        grid = enumerate_grid(self.reels, 3, PT, LINES3, [0], 1, 0.1)
        self.assertAlmostEqual(la.ev * 3, grid.line_ev, places=10)

    def test_scatter_distribution_equals_enumeration(self):
        sd = scatter_count_distribution(self.reels, 1, 3)
        grid = enumerate_grid(self.reels, 3, PT, LINES3, [0], 1, 0.1)
        g = grid.scatter_dist
        np.testing.assert_allclose(sd[: len(g)], g, atol=1e-12)

    def test_monte_carlo_agrees(self):
        grid = enumerate_grid(self.reels, 3, PT, LINES3, [0], 1, 0.1)
        rng = np.random.default_rng(1)
        n = 300_000
        win = windows_from_stops(self.reels, random_stops(self.reels, n, rng), 3)
        lw = evaluate_lines(win, PT, LINES3, [0])
        sd = count_symbol(win, 1)
        se = lw.std() / np.sqrt(n)
        self.assertLess(abs(lw.mean() - grid.line_ev), 4 * se)
        self.assertLess(abs((sd >= 3).mean() - grid.scatter_dist[3:].sum()), 0.01)

    def test_scale_and_cap(self):
        d = np.array([0.5, 0.25, 0.25])                            # values 0, 1, 2 bins
        np.testing.assert_allclose(scale_dist(d, 3), [0.5, 0, 0, 0.25, 0, 0, 0.25])
        c = convolve_capped(d, d, cap_bins=2)                      # sum of two, capped at 2
        self.assertAlmostEqual(c.sum(), 1.0, places=12)
        self.assertAlmostEqual(c[0], 0.25)                         # both zero
        self.assertAlmostEqual(c[1], 0.25)                         # 0+1 or 1+0
        self.assertAlmostEqual(c[2], 0.5)                          # everything >= 2 folded


class TestMarkov(unittest.TestCase):
    def test_ladder_closed_form(self):
        # every spin wins exactly 1x, never retriggers: payouts 1,2,3,4,5,5,5,5,5,5 = 40x for 10 spins
        joint = np.zeros((16, 11)); joint[0, 10] = 1.0             # scatter count 0, win = 1.0x (bin 10 at 0.1)
        rules = LadderRules({3: 10}, 3, 5, 4, 1, 1, 5)
        ev = ladder_feature_ev_fast(joint, 0.1, rules)
        self.assertAlmostEqual(ev[3], 40.0, places=9)
        an = analyze_ladder_feature(joint, 0.1, rules, cap=None)
        self.assertAlmostEqual(an.ev_by_trigger[3], 40.0, places=9)
        self.assertAlmostEqual(an.spins_by_trigger[3], 10.0, places=9)
        d = an.dist_by_trigger[3]
        self.assertAlmostEqual(d[400], 1.0, places=9)             # all mass at 40.0x

    def test_retrigger_expected_spins(self):
        # no wins, retrigger with prob p each spin, unlimited retriggers would give 10/(1-5p); with a cap of 1
        p = 0.05
        joint = np.zeros((16, 1)); joint[0, 0] = 1 - p; joint[3, 0] = p
        rules = LadderRules({3: 10}, 3, 5, 1, 1, 1, 5)
        an = analyze_ladder_feature(joint, 0.1, rules, cap=None)
        # with max 1 retrigger: E[spins] = 10 + 5 * P(at least one retrigger in 10 spins)
        expected = 10 + 5 * (1 - (1 - p) ** 10)
        self.assertAlmostEqual(an.spins_by_trigger[3], expected, places=9)


if __name__ == "__main__":
    unittest.main()
