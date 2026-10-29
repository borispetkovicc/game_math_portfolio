"""Analysis of the Sugar Heaven reference model -> results/sugar_exact.json / .pkl

Base game exact (77 million combinations); sticky-wild feature by simulation
(400k features per trigger size, with CI). Runtime ~5 minutes.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import sugar_heaven_ref as sh      # noqa: E402


def main() -> None:
    cfg = sh.load_config()
    t = time.time()
    rep = sh.analysis(cfg, n_features=400_000, verbose=True)
    s = rep.stats
    print(f"\ndone in {time.time()-t:.0f}s")
    print(f"RTP = {100*s['rtp']:.4f}% +/- {100*s['rtp_ci95_half']:.3f} pp (95% CI from the simulated feature)   public: {100*s['public']['rtp']:.2f}%")
    print(f"  lines {100*s['rtp_line']:.3f}% (exact) + scatters {100*s['rtp_scatter']:.3f}% (exact) + feature {100*s['rtp_feature']:.3f}% (simulated)")
    print(f"hit rate (exact) = {100*s['hit_rate_exact']:.3f}%   public: {100*s['public']['hit_rate']:.0f}%")
    print(f"feature 1 in {s['trigger_1_in']:.1f} spins; E[feature] by scatters: { {k: round(v, 1) for k, v in s['feature_ev_by_count'].items()} }")
    print(f"bonus buy RTP: 100x -> {100*s['bonus_buy_100x_rtp']:.2f}%,  300x (5 scatters) -> {100*s['bonus_buy_300x_rtp']:.2f}%")
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    with open(out / "sugar_exact.pkl", "wb") as f:
        pickle.dump(rep, f)
    with open(out / "sugar_exact.json", "w") as f:
        json.dump(s, f, indent=1, default=float)
    print("saved results/sugar_exact.json / .pkl")


if __name__ == "__main__":
    main()
