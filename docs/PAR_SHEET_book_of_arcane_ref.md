# PAR sheet — Book of Arcane 100
Generated 2026-10-10. Original model fitted to the public spec of Just Slots' Book of Arcane 100; strips, paytable and multiplier weights are ours.

## 1. Game summary

| Parameter | Value |
|:---|:---|
| Grid | 5 reels x 3 rows |
| Paylines | 10 (fixed), pays left to right |
| Bet unit | all pays in multiples of the total bet |
| Max win cap | 100,000x |
| **RTP (exact, incl. cap)** | **96.3600%** (public: 96.36%) |
| RTP split | lines 53.217% + Book scatter pays 1.639% + free spins 41.520% |
| Cap cost | 0.015252 pp |
| Hit frequency (exact) | 28.03% (public: ~27%) |
| Std dev per spin | 25.438x bet (volatility index 1.96σ = 49.9) |
| Free-spins frequency | 1 in 181 spins |
| Free spins | 10 spins, expanding symbol uniform over the 9 regular symbols, retrigger +10 (max 3) |
| Feature multiplier | drawn once per feature: x1: 94.060%, x2: 1.890%, x3: 2.000%, x5: 1.000%, x10: 0.600%, x20: 0.300%, x50: 0.100%, x100: 0.050% → E[M] = 1.3084 |
| E[feature payout] | 75.25x (median 37.5x, std 328.4x); expected spins 10.583 |
| Fair bonus-buy price | 78.1x for the plain feature (buy RTP = game RTP); the real game's 1,000x 'Arcane 100' buy guarantees the wild as expanding symbol, which this model does not include |
| Combinations | 52,521,875 (all enumerated; the feature uses the same strips) |

## 2. Paytable (x total bet, 10 lines)

| Symbol | 2 of a kind | 3 of a kind | 4 of a kind | 5 of a kind |
|:---|---:|---:|---:|---:|
| H1 | 1 | 10 | 100 | 500 |
| H2 | 0.5 | 4 | 40 | 200 |
| H3 | – | 3 | 30 | 100 |
| H4 | – | 2 | 15 | 75 |
| A | – | 0.5 | 4 | 15 |
| K | – | 0.5 | 4 | 15 |
| Q | – | 0.5 | 2.5 | 10 |
| J | – | 0.5 | 2.5 | 10 |
| T | – | 0.5 | 2.5 | 10 |

Scatter pays (anywhere on the grid, x total bet): 3 scatters = 2x, 4 scatters = 20x, 5 scatters = 200x

## 3. Reel strips (base game and free spins)

**Strips** — count per reel (probability that a given row of that reel shows the symbol):

| Symbol | Reel 1 | Reel 2 | Reel 3 | Reel 4 | Reel 5 |
|:---|---:|---:|---:|---:|---:|
| B | 1 (2.86%) | 1 (2.86%) | 1 (2.86%) | 1 (2.86%) | 1 (2.86%) |
| H1 | 1 (2.86%) | 1 (2.86%) | 1 (2.86%) | 2 (5.71%) | 2 (5.71%) |
| H2 | 2 (5.71%) | 2 (5.71%) | 2 (5.71%) | 2 (5.71%) | 3 (8.57%) |
| H3 | 2 (5.71%) | 3 (8.57%) | 2 (5.71%) | 2 (5.71%) | 6 (17.14%) |
| H4 | 6 (17.14%) | 2 (5.71%) | 2 (5.71%) | 2 (5.71%) | 2 (5.71%) |
| A | 4 (11.43%) | 6 (17.14%) | 6 (17.14%) | 4 (11.43%) | 2 (5.71%) |
| K | 2 (5.71%) | 6 (17.14%) | 6 (17.14%) | 6 (17.14%) | 4 (11.43%) |
| Q | 5 (14.29%) | 4 (11.43%) | 6 (17.14%) | 6 (17.14%) | 4 (11.43%) |
| J | 6 (17.14%) | 4 (11.43%) | 5 (14.29%) | 6 (17.14%) | 5 (14.29%) |
| T | 6 (17.14%) | 6 (17.14%) | 4 (11.43%) | 4 (11.43%) | 6 (17.14%) |
| **strip length** | **35** | **35** | **35** | **35** | **35** |

Full strips (top to bottom, wrapping):

- Reel 1: `H4 J T Q A H2 H4 J T Q H3 A H4 J T K Q J H4 B T A Q H4 J T H2 H3 A H4 J T Q K H1`
- Reel 2: `J H4 H1 A K T Q J H3 A K T H2 Q J A K T H4 H3 K A Q T J B A K T H2 H3 Q A K T`
- Reel 3: `A K Q J H4 H1 A K Q J T H2 A K Q J H3 T A K Q H4 J K A B Q T J A K Q H2 H3 T`
- Reel 4: `H3 H2 A K Q J T B H4 K Q J A T H1 K Q J H2 A T K Q J H3 H4 Q K A J T H1 K Q J`
- Reel 5: `H3 T H4 H1 Q J H3 T K H2 B A H3 T J K Q H2 H3 T H1 J K H4 H3 T Q J H2 T H3 A K Q J`


## 4. Expanding symbol: unmultiplied feature value by symbol (exact)

| Expanding symbol | E[feature payout] before multiplier |
|:---|---:|
| H1 | 38.28x |
| H2 | 59.72x |
| H3 | 78.62x |
| H4 | 38.35x |
| A | 49.04x |
| K | 63.90x |
| Q | 58.31x |
| J | 66.27x |
| T | 65.32x |

## 5. Distribution of the total win of a base-game spin (exact)

| Win per spin | Probability |  |
|:---|---:|---:|
| >= 1x | 1.2037e-01 | 1 in 8.31 |
| >= 2x | 7.7397e-02 | 1 in 12.92 |
| >= 5x | 1.8438e-02 | 1 in 54.24 |
| >= 10x | 1.4468e-02 | 1 in 69.12 |
| >= 20x | 5.8112e-03 | 1 in 172 |
| >= 50x | 2.9612e-03 | 1 in 338 |
| >= 100x | 1.4049e-03 | 1 in 712 |
| >= 200x | 4.1606e-04 | 1 in 2,403 |
| >= 500x | 8.2280e-05 | 1 in 12,154 |
| >= 1,000x | 3.1937e-05 | 1 in 31,312 |
| >= 2,000x | 1.0737e-05 | 1 in 93,136 |
| >= 5,000x | 2.6596e-06 | 1 in 375,991 |
| >= 10,000x | 7.6244e-07 | 1 in 1,311,587 |
| >= 100,000x | 7.4131e-09 | 1 in 134,896,371 |
| = max win 100,000x | 7.4131e-09 | 1 in 134,896,371 |

Quantiles of the per-spin win: q50: 0x, q90: 1x, q99: 11x, q99.9: 114x, q99.99: 444.5x

## 6. Exact vs Monte-Carlo

| Quantity | Exact | Simulation (200,000,000 spins, seed 2024) |  |
|:---|:---|:---|:---|
| RTP | 96.3600% | 96.3224% ± 0.3642 pp | inside CI |
| Hit rate | 28.031% | 28.025% |  |
| Std dev / spin | 25.4377x | 26.2808x |  |
| Feature frequency | 1 in 181.3 | 1 in 181.4 |  |
| Feature RTP share | 41.520% | 41.683% |  |
| Max win observed |  | 100,000.0x |  |

## 7. Notes

- The Book is wild on lines and scatter anywhere (pays 2x/20x/200x for 3/4/5 — public spec).
- In free spins the expanding symbol pays `paytable[symbol][number of reels containing it] × 10 lines` in addition to regular line wins; all feature wins are multiplied by the feature multiplier.
- The feature multiplier weights were solved so that the exact RTP equals the public 96.36 % (feature RTP is linear in E[M]).
- Left out: the gamble-to-upgrade option and the random base-game expanding symbol of the real game.
