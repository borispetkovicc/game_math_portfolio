# PAR sheet — Sugar Heaven (reference model)
Generated 2026-10-10. Original model fitted to the public spec of Just Slots' Sugar Heaven (https://www.just-slots.com/games/sugar-heaven); strips and paytable are ours.

## 1. Game summary

| Parameter | Value |
|:---|:---|
| Grid | 5 reels x 3 rows |
| Paylines | 20 (fixed), pays left to right |
| Bet unit | all pays in multiples of the total bet |
| Max win cap | 5,000x |
| **RTP** | **96.486% ± 0.229 pp** (public: 96.42%) |
| RTP split | lines 51.163% (exact) + scatters 1.047% (exact) + free spins 44.277% (simulated, 400k features/trigger size) |
| Hit frequency (exact) | 33.08% (public: 31%) |
| Volatility | high (public: high); simulated std/spin see §5 |
| Free-spins frequency | 1 in 226 spins |
| Free spins awarded | 3 scatters → 8 spins, 4 scatters → 9 spins, 5 scatters → 12 spins (public spec), wilds sticky, no retrigger |
| Multiplier wilds | W2/W3/W5/W10 carry x2/x3/x5/x10; multipliers on one winning line are added |
| Bonus buy | 100x → feature RTP 100.02%;  300x (5 scatters) → 105.16% |
| Base combinations | 77,096,604 (all enumerated) |

## 2. Paytable (x total bet)

| Symbol | 2 of a kind | 3 of a kind | 4 of a kind | 5 of a kind |
|:---|---:|---:|---:|---:|
| W2 | – | 1.5 | 5 | 20 |
| W3 | – | 1.5 | 5 | 20 |
| W5 | – | 1.5 | 5 | 20 |
| W10 | – | 1.5 | 5 | 20 |
| H1 | – | 1.5 | 5 | 20 |
| H2 | – | 1 | 3 | 12 |
| H3 | – | 0.8 | 2 | 8 |
| H4 | – | 0.6 | 1.5 | 6 |
| L1 | – | 0.3 | 0.8 | 3 |
| L2 | – | 0.2 | 0.6 | 2.5 |
| L3 | – | 0.2 | 0.5 | 2 |
| L4 | – | 0.2 | 0.4 | 1.5 |

Scatter pays (anywhere on the grid, x total bet): 3 scatters = 2x, 4 scatters = 10x, 5 scatters = 50x

## 3. Reel strips

**Base game** — count per reel (probability that a given row of that reel shows the symbol):

| Symbol | Reel 1 | Reel 2 | Reel 3 | Reel 4 | Reel 5 |
|:---|---:|---:|---:|---:|---:|
| W2 | 0 (0.00%) | 1 (2.63%) | 1 (2.56%) | 1 (2.63%) | 0 (0.00%) |
| W3 | 0 (0.00%) | 0 (0.00%) | 1 (2.56%) | 0 (0.00%) | 0 (0.00%) |
| S | 1 (2.70%) | 1 (2.63%) | 1 (2.56%) | 1 (2.63%) | 1 (2.70%) |
| H1 | 3 (8.11%) | 3 (7.89%) | 3 (7.69%) | 5 (13.16%) | 3 (8.11%) |
| H2 | 2 (5.41%) | 3 (7.89%) | 7 (17.95%) | 4 (10.53%) | 3 (8.11%) |
| H3 | 4 (10.81%) | 3 (7.89%) | 3 (7.69%) | 2 (5.26%) | 2 (5.41%) |
| H4 | 5 (13.51%) | 5 (13.16%) | 5 (12.82%) | 3 (7.89%) | 6 (16.22%) |
| L1 | 5 (13.51%) | 5 (13.16%) | 5 (12.82%) | 5 (13.16%) | 6 (16.22%) |
| L2 | 5 (13.51%) | 5 (13.16%) | 5 (12.82%) | 6 (15.79%) | 6 (16.22%) |
| L3 | 5 (13.51%) | 5 (13.16%) | 5 (12.82%) | 5 (13.16%) | 4 (10.81%) |
| L4 | 7 (18.92%) | 7 (18.42%) | 3 (7.69%) | 6 (15.79%) | 6 (16.22%) |
| **strip length** | **37** | **38** | **39** | **38** | **37** |

Full strips (top to bottom, wrapping):

- Reel 1: `L4 H4 L1 L2 L3 L4 H3 H1 H4 L1 L2 L4 L3 H3 H2 H4 L4 L1 L2 L3 H1 L4 H3 H4 L1 L2 L4 L3 S H1 H4 L1 L4 L2 L3 H3 H2`
- Reel 2: `L4 L2 L3 H3 S L4 H4 L1 L2 L3 L4 H1 H2 H4 L1 L2 L4 L3 H3 W2 H4 L4 L1 L2 L3 H1 H2 L4 H3 H4 L1 L2 L4 L3 H1 H2 H4 L1`
- Reel 3: `W3 H1 H3 H4 H2 L1 L2 L3 L4 S H2 H4 L1 L2 L3 H1 H2 H3 H4 L1 L2 H2 L3 L4 W2 H1 H4 H2 L1 L2 L3 H3 H2 L4 H4 L1 L2 L3 H2`
- Reel 4: `H2 H1 L2 L4 L1 L3 H3 H2 H1 L2 L4 L1 L3 H4 S L2 L4 H1 L1 L3 H2 L2 L4 H4 H1 L1 L3 H3 L2 L4 H2 H1 L1 L3 L2 L4 H4 W2`
- Reel 5: `H1 H4 L2 L1 L3 L4 H2 S H4 L1 L2 L4 H1 L3 H4 L1 L2 L4 H2 H3 H4 L1 L2 L4 L3 H1 H4 L1 L2 L4 H2 L3 H4 L1 L2 L4 H3`


**Free spins** — count per reel (probability that a given row of that reel shows the symbol):

| Symbol | Reel 1 | Reel 2 | Reel 3 | Reel 4 | Reel 5 |
|:---|---:|---:|---:|---:|---:|
| W2 | 0 (0.00%) | 1 (3.70%) | 1 (3.57%) | 1 (3.70%) | 1 (3.70%) |
| W3 | 0 (0.00%) | 0 (0.00%) | 1 (3.57%) | 0 (0.00%) | 0 (0.00%) |
| H1 | 3 (11.54%) | 3 (11.11%) | 3 (10.71%) | 3 (11.11%) | 3 (11.11%) |
| H2 | 3 (11.54%) | 3 (11.11%) | 3 (10.71%) | 3 (11.11%) | 3 (11.11%) |
| H3 | 3 (11.54%) | 3 (11.11%) | 3 (10.71%) | 3 (11.11%) | 3 (11.11%) |
| H4 | 3 (11.54%) | 3 (11.11%) | 3 (10.71%) | 3 (11.11%) | 3 (11.11%) |
| L1 | 3 (11.54%) | 3 (11.11%) | 3 (10.71%) | 3 (11.11%) | 3 (11.11%) |
| L2 | 3 (11.54%) | 3 (11.11%) | 3 (10.71%) | 3 (11.11%) | 3 (11.11%) |
| L3 | 4 (15.38%) | 4 (14.81%) | 4 (14.29%) | 4 (14.81%) | 4 (14.81%) |
| L4 | 4 (15.38%) | 4 (14.81%) | 4 (14.29%) | 4 (14.81%) | 4 (14.81%) |
| **strip length** | **26** | **27** | **28** | **27** | **27** |

Full strips (top to bottom, wrapping):

- Reel 1: `L3 L4 H1 H2 H3 H4 L3 L1 L4 L2 H1 H2 H3 L3 L4 H4 L1 L2 H1 L4 L3 H2 H3 H4 L1 L2`
- Reel 2: `H2 H3 H4 L1 L2 L3 L4 H1 H2 H3 H4 L1 L3 L4 L2 H1 H2 H3 H4 L3 L4 L1 L2 W2 H1 L3 L4`
- Reel 3: `L2 H1 H2 L3 L4 H3 H4 L1 L2 W3 L3 L4 H1 H2 H3 H4 L1 L3 L4 L2 W2 H1 H2 H3 L3 L4 H4 L1`
- Reel 4: `H3 H4 L3 L4 L1 L2 W2 H1 L3 L4 H2 H3 H4 L1 L2 L3 L4 H1 H2 H3 H4 L1 L3 L4 L2 H1 H2`
- Reel 5: `L3 L4 L2 H1 H2 H3 H4 L3 L4 L1 L2 W2 H1 L3 L4 H2 H3 H4 L1 L2 L3 L4 H1 H2 H3 H4 L1`


## 4. Feature value (simulation, 95% CI)

| Trigger | P(trigger) | Spins | E[payout] | ± (95% CI) |
|:---|---:|---:|---:|---:|
| 3 scatters | 0.004241 | 8 | 98.26x | 0.51x |
| 4 scatters | 0.000183 | 9 | 137.27x | 0.67x |
| 5 scatters | 0.000003 | 12 | 315.49x | 1.32x |

## 5. Exact/semi-exact vs Monte-Carlo

| Quantity | Exact | Simulation (50,000,000 spins, seed 2024) |  |
|:---|:---|:---|:---|
| RTP | 96.4861% | 96.2077% ± 0.3650 pp | inside CI |
| Hit rate | 33.079% | 33.084% |  |
| Feature frequency | 1 in 225.9 | 1 in 225.7 |  |
| Feature RTP share | 44.277% | 44.035% |  |
| Max win observed |  | 3,114.5x |  |

## 6. Notes

- Base game is exact (full enumeration with the wild multipliers carried by the strips).
- The sticky-wild feature's state (which cells hold which multiplier) is too large to enumerate; it is valued by simulation with a confidence interval — standard practice for sticky/collect features.
- Left out of the model: Super Wild enhancers and the random base-game wild drop of the real game.
