#!/usr/bin/env python3
"""Build results/equalization_map.json for the methods currently equalizable
without new code (docs/EQUALIZATION_READINESS.md, answer C).

Budgets eps* in {1, 4, 8} are provisional defaults per Prompt 2 / Stage 2's
suggestion; Stage 2 fixes the final grid. Only two methods are included here
because they are the two with a reusable accountant already available as a
plain library call (RDP for record-level DP-SGD, client-level RDP for
client-level DP) -- see docs/EQUALIZATION_READINESS.md answers C and E.

Usage:
    cd experiments/dp-equalization && uv run python scripts/build_equalization_map.py
"""

from __future__ import annotations

from pathlib import Path

from dp_equalization.accountant import MethodConfig, epsilon
from dp_equalization.equalize import equalize
from dp_equalization.registry import save_equalization_map

# (k, n) = (optimizer steps/round, local train-set size) per silo, R=20.
# experiments/adaptive-clipping/analysis/renyi_group_composition.py:59-65.
HERALD_SILOS = ((155, 1550), (150, 1508), (151, 1518), (152, 1527), (149, 1492))

EPS_STAR_GRID = (1.0, 4.0, 8.0)

TEMPLATES = {
    "fedprox_dpsgd_record_level": MethodConfig(
        method="fedprox_dpsgd_record_level",
        sigma=1.0,  # placeholder; equalize() replaces it
        rounds=20,
        delta=1e-5,
        silos=HERALD_SILOS,
        accountant="rdp",
    ),
    "client_level_dp": MethodConfig(
        method="client_level_dp",
        sigma=1.0,
        rounds=20,
        delta=1e-5,
        accountant="client_level",
    ),
}

OUT_PATH = Path(__file__).resolve().parents[3] / "results" / "equalization_map.json"


def main() -> None:
    entries = []
    for name, template in TEMPLATES.items():
        for eps_star in EPS_STAR_GRID:
            cfg, eps_achieved = equalize(template, eps_star)
            entries.append(
                {
                    "original": template,
                    "eps_star": eps_star,
                    "equalized": cfg,
                    "epsilon_achieved": eps_achieved,
                }
            )
            print(
                f"{name:32s} eps*={eps_star:>5.2f} -> "
                f"sigma={cfg.sigma:.4f}  eps_achieved={eps_achieved:.4f}"
            )

    written = save_equalization_map(entries, OUT_PATH)
    print(f"\nWrote {len(entries)} entries to {written}")


if __name__ == "__main__":
    main()
