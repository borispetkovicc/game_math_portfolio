"""Free-spins features as absorbing Markov chains, solved exactly.

A free-spins feature is a Markov chain whose state carries everything the next
spin depends on (spins remaining, multiplier level, retriggers used, ...). The
"feature over" state (0 spins remaining) is absorbing. We want the
distribution of the TOTAL payout accumulated before absorption.

First-step analysis gives, for every transient state s,

    D(s) = sum_c  [ pay_c(s) ]  (*)  D(next_c(s))

where c runs over the outcome classes of one spin, pay_c(s) is the (sub-)
distribution of that spin's payout given class c (it carries the class
probability p_c, so it integrates to p_c), (*) is convolution (independent
sums) and D(absorbing) = delta_0.

Because every transition strictly decreases a potential function (spins left
plus 5 x retriggers still available), the chain is a finite DAG and (*) can be
evaluated by memoised recursion — no linear system is needed, and the result is
exact up to floating point. The same recursion with means instead of
distributions gives expectations in microseconds (used during calibration).

Payouts live on a grid of width "bin_width" (x bet). Multiplying a payout by an
integer multiplier m maps bin b -> bin m*b exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from .exact import convolve_capped, scale_dist, shift_dist, cap_dist

# A "step" describes one spin from a state: list of (payout sub-distribution, next state)
Step = List[Tuple[np.ndarray, Optional[Hashable]]]
StepFn = Callable[[Hashable], Step]


def absorbing_dp(initial: Hashable, step_fn: StepFn, cap_bins: Optional[int]) -> np.ndarray:
    """Exact distribution of total payout from `initial` until absorption (next=None).

    cap_bins : fold everything above this bin into it (max-win cap). Exact,
               because min(x + y, cap) = min(min(x, cap) + y, cap) for y >= 0.
    """
    memo: Dict[Hashable, np.ndarray] = {}
    delta0 = np.array([1.0])

    def D(state: Optional[Hashable]) -> np.ndarray:
        if state is None:
            return delta0
        if state in memo:
            return memo[state]
        total: Optional[np.ndarray] = None
        for pay, nxt in step_fn(state):
            part = convolve_capped(pay, D(nxt), cap_bins)
            if total is None:
                total = part
            elif len(part) > len(total):
                part[: len(total)] += total
                total = part
            else:
                total[: len(part)] += part
        total = cap_dist(total, cap_bins)
        memo[state] = total
        return total

    return D(initial)


def absorbing_dp_mean(initial: Hashable, step_fn_mean: Callable[[Hashable], List[Tuple[float, float, Optional[Hashable]]]]) -> float:
    """Expected total payout. step_fn_mean(state) -> [(p_c, E[payout * 1{c}], next_c), ...]."""
    memo: Dict[Hashable, float] = {}

    def E(state: Optional[Hashable]) -> float:
        if state is None:
            return 0.0
        if state in memo:
            return memo[state]
        v = 0.0
        for p, m, nxt in step_fn_mean(state):
            v += m + p * E(nxt)
        memo[state] = v
        return v

    return E(initial)


# --------------------------------------------------------------------------
# Multiplier-ladder feature (used by Kastel Fortune)
# --------------------------------------------------------------------------
@dataclass
class LadderRules:
    """Free spins with a multiplier that climbs after every winning spin.

    spins_by_trigger : scatters in base game -> initial free spins
    retrigger_min    : scatters needed inside the feature to retrigger
    retrigger_spins  : extra spins per retrigger
    max_retriggers   : hard cap on retriggers (keeps the chain finite)
    mult_start/step/cap : multiplier ladder; it increases by `step` after a spin
                       with a line win and is applied to the NEXT spins.
    """
    spins_by_trigger: Dict[int, int]
    retrigger_min: int = 3
    retrigger_spins: int = 5
    max_retriggers: int = 4
    mult_start: int = 1
    mult_step: int = 1
    mult_cap: int = 5

    @property
    def max_spins(self) -> int:
        return max(self.spins_by_trigger.values()) + self.retrigger_spins * self.max_retriggers


@dataclass
class LadderAnalysis:
    rules: LadderRules
    bin_width: float
    class_probs: Dict[str, float]                 # per-spin outcome class probabilities
    class_mean_win: Dict[str, float]              # E[line win * 1{class}] per spin (unmultiplied)
    dist_by_trigger: Dict[int, np.ndarray]        # scatter count -> exact distribution of feature total
    ev_by_trigger: Dict[int, float]
    spins_by_trigger: Dict[int, float]            # expected number of free spins


def _ladder_classes(joint: np.ndarray, retrigger_min: int) -> Dict[str, np.ndarray]:
    """Split the per-spin joint histogram into the 4 outcome classes.

    joint : (max_scatters+1, n_bins) exact P(scatter count, line win bin) on FS reels
    returns name -> sub-distribution of the line win (integrates to the class prob)
    """
    nb = joint.shape[1]
    retrig = joint[retrigger_min:].sum(axis=0)
    noretrig = joint[:retrigger_min].sum(axis=0)
    zero = np.zeros(nb)
    classes = {}
    for name, src in (("retrig", retrig), ("noretrig", noretrig)):
        w = src.copy(); w[0] = 0.0                      # winning spins: line win > 0
        nw = zero.copy(); nw[0] = src[0]                # non-winning spins
        classes[f"win_{name}"] = w
        classes[f"nowin_{name}"] = nw
    return classes


def analyze_ladder_feature(joint_fs: np.ndarray, bin_width: float, rules: LadderRules,
                           cap: Optional[float]) -> LadderAnalysis:
    """Exact analysis of a multiplier-ladder free-spins feature.

    joint_fs : exact joint histogram (scatter count, line-win bin) of ONE free spin
               (from exact.enumerate_grid on the free-spins reel strips)
    """
    classes = _ladder_classes(joint_fs, rules.retrigger_min)
    values = np.arange(joint_fs.shape[1]) * bin_width
    cap_bins = None if cap is None else int(round(cap / bin_width))
    p = {k: float(v.sum()) for k, v in classes.items()}
    mean = {k: float((v * values).sum()) for k, v in classes.items()}

    # pre-scale the four class distributions by every multiplier level
    scaled: Dict[Tuple[str, int], np.ndarray] = {}
    for k, v in classes.items():
        for m in range(rules.mult_start, rules.mult_cap + 1, rules.mult_step):
            scaled[(k, m)] = scale_dist(v, m)

    def next_state(n: int, m: int, r: int, win: bool, retrig: bool):
        n2 = n - 1
        r2 = r
        if retrig and r < rules.max_retriggers:
            n2 += rules.retrigger_spins
            r2 += 1
        m2 = min(m + rules.mult_step, rules.mult_cap) if win else m
        return None if n2 == 0 else (n2, m2, r2)

    def step(state):
        n, m, r = state
        out = []
        for name in classes:
            win = name.startswith("win")
            retrig = name.endswith("_retrig")
            out.append((scaled[(name, m)], next_state(n, m, r, win, retrig)))
        return out

    def step_mean(state):
        n, m, r = state
        out = []
        for name in classes:
            win = name.startswith("win")
            retrig = name.endswith("_retrig")
            out.append((p[name], m * mean[name], next_state(n, m, r, win, retrig)))
        return out

    def step_spins(state):   # counts spins instead of money
        n, m, r = state
        out = []
        for name in classes:
            win = name.startswith("win")
            retrig = name.endswith("_retrig")
            out.append((p[name], p[name] * 1.0, next_state(n, m, r, win, retrig)))
        return out

    dist_by, ev_by, spins_by = {}, {}, {}
    for sc, n0 in rules.spins_by_trigger.items():
        s0 = (n0, rules.mult_start, 0)
        dist_by[sc] = absorbing_dp(s0, step, cap_bins)
        ev_by[sc] = absorbing_dp_mean(s0, step_mean)
        spins_by[sc] = absorbing_dp_mean(s0, step_spins)
    return LadderAnalysis(rules=rules, bin_width=bin_width, class_probs=p, class_mean_win=mean,
                          dist_by_trigger=dist_by, ev_by_trigger=ev_by, spins_by_trigger=spins_by)


def ladder_feature_ev_fast(joint_fs: np.ndarray, bin_width: float, rules: LadderRules) -> Dict[int, float]:
    """Expected feature payout per trigger size, means only (for calibration loops)."""
    classes = _ladder_classes(joint_fs, rules.retrigger_min)
    values = np.arange(joint_fs.shape[1]) * bin_width
    p = {k: float(v.sum()) for k, v in classes.items()}
    mean = {k: float((v * values).sum()) for k, v in classes.items()}

    def next_state(n, m, r, win, retrig):
        n2, r2 = n - 1, r
        if retrig and r < rules.max_retriggers:
            n2 += rules.retrigger_spins; r2 += 1
        m2 = min(m + rules.mult_step, rules.mult_cap) if win else m
        return None if n2 == 0 else (n2, m2, r2)

    def step_mean(state):
        n, m, r = state
        return [(p[k], m * mean[k], next_state(n, m, r, k.startswith("win"), k.endswith("_retrig")))
                for k in classes]

    return {sc: absorbing_dp_mean((n0, rules.mult_start, 0), step_mean)
            for sc, n0 in rules.spins_by_trigger.items()}


# --------------------------------------------------------------------------
# Fixed-multiplier feature with retriggers (used by the "Book" reference model)
# --------------------------------------------------------------------------
@dataclass
class FixedMultRules:
    initial_spins: int = 10
    retrigger_min: int = 3
    retrigger_spins: int = 10
    max_retriggers: int = 3


def analyze_fixed_feature(spin_dist_by_retrig: Tuple[np.ndarray, np.ndarray], rules: FixedMultRules,
                          cap_bins: Optional[int]) -> Tuple[np.ndarray, float, float]:
    """Feature whose per-spin payout distribution does not depend on history.

    spin_dist_by_retrig : (sub-distribution of payout when the spin does NOT retrigger,
                           sub-distribution when it DOES); they sum to the full spin distribution.
    returns (exact distribution of UNMULTIPLIED feature total, its mean, expected spins)
    """
    d_no, d_re = spin_dist_by_retrig
    p_no, p_re = float(d_no.sum()), float(d_re.sum())
    values_no = np.arange(len(d_no)); values_re = np.arange(len(d_re))
    m_no = float((d_no * values_no).sum()); m_re = float((d_re * values_re).sum())

    def nxt(n, r, retrig):
        n2, r2 = n - 1, r
        if retrig and r < rules.max_retriggers:
            n2 += rules.retrigger_spins; r2 += 1
        return None if n2 == 0 else (n2, r2)

    def step(state):
        n, r = state
        return [(d_no, nxt(n, r, False)), (d_re, nxt(n, r, True))]

    def step_mean(state):
        n, r = state
        return [(p_no, m_no, nxt(n, r, False)), (p_re, m_re, nxt(n, r, True))]

    def step_spins(state):
        n, r = state
        return [(p_no, p_no, nxt(n, r, False)), (p_re, p_re, nxt(n, r, True))]

    s0 = (rules.initial_spins, 0)
    dist = absorbing_dp(s0, step, cap_bins)
    ev_bins = absorbing_dp_mean(s0, step_mean)
    spins = absorbing_dp_mean(s0, step_spins)
    return dist, ev_bins, spins
