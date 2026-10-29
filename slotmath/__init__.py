"""slotmath — a small, transparent toolkit for slot-game mathematics.

Modules
-------
game      : GameConfig dataclass (reel strips, paylines, paytable, feature rules)
engine    : vectorised NumPy reel/line evaluation shared by exact analysis and simulation
exact     : exact (closed-form / full-enumeration) analysis of a base-game spin
markov    : exact free-spins analysis as an absorbing Markov chain (distribution-valued DP)
simulate  : Monte-Carlo simulation with confidence intervals (the independent check)
par_sheet : PAR-sheet (Probability Accounting Report) generator
"""

__version__ = "0.1.0"
