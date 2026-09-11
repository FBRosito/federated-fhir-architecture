#!/usr/bin/env python3
"""
plot_convergence.py
--------------------
Convergence-by-round figure (Micro-F1 vs. round) for the 4 final-matrix
configurations (baseline/per_layer x sigma 1.0/2.0) — equivalent to Figure 3
of the original HERALD paper.

Per-seed values are the unweighted mean of micro_f1 across the 5 silos —
the same method analysis.bootstrap_ci.per_seed_value uses, since
RoundLogRecord carries no per-silo example count to weight by. Reusing it
here keeps this figure methodologically consistent with the CI table.

Usage:
    cd experiments/adaptive-clipping && uv run python -m analysis.plot_convergence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from analysis.bootstrap_ci import (  # noqa: E402
    NUM_SILOS,
    SIGMAS,
    STRATEGIES,
    per_seed_value,
)

NUM_ROUNDS = 20
CENTRALIZED_F1 = 0.409
FL_NODP_F1 = 0.308

# Two blues for baseline (dark=sigma1.0, light=sigma2.0), two oranges for
# per_layer (dark=sigma1.0, light=sigma2.0); solid for per_layer, dashed for
# baseline.
_STYLE = {
    ("baseline", 1.0): dict(
        color="#08519c", linestyle="--", label="Baseline (global clip), σ=1.0"
    ),
    ("baseline", 2.0): dict(
        color="#6baed6", linestyle="--", label="Baseline (global clip), σ=2.0"
    ),
    ("per_layer", 1.0): dict(
        color="#d94801", linestyle="-", label="Per-layer clip, σ=1.0"
    ),
    ("per_layer", 2.0): dict(
        color="#fd8d3c", linestyle="-", label="Per-layer clip, σ=2.0"
    ),
}


def load_round_records(
    logs_root: Path, strategy: str, sigma: float
) -> dict[int, dict[int, list[dict]]]:
    """Returns {round: {seed: [records per silo]}} for one (strategy, sigma) config."""
    by_round: dict[int, dict[int, list[dict]]] = {
        r: {} for r in range(1, NUM_ROUNDS + 1)
    }
    for run_dir in sorted(logs_root.glob(f"{strategy}_sigma{sigma}_seed*")):
        if not run_dir.is_dir():
            continue
        seed = int(run_dir.name.rsplit("seed", 1)[1])
        for round_idx in range(1, NUM_ROUNDS + 1):
            records = []
            for silo_id in range(NUM_SILOS):
                round_file = run_dir / str(silo_id) / f"round_{round_idx:03d}.jsonl"
                if not round_file.exists():
                    continue
                with open(round_file) as f:
                    records.append(json.loads(f.readline()))
            if records:
                by_round[round_idx][seed] = records
    return by_round


def config_curve(
    logs_root: Path, strategy: str, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rounds, mean_f1, std_f1) across the 10 seeds, per round."""
    by_round = load_round_records(logs_root, strategy, sigma)
    rounds = np.arange(1, NUM_ROUNDS + 1)
    means = np.full(NUM_ROUNDS, np.nan)
    stds = np.full(NUM_ROUNDS, np.nan)
    for round_idx in rounds:
        seed_values = [
            per_seed_value(records, "micro_f1")
            for records in by_round[round_idx].values()
        ]
        if seed_values:
            means[round_idx - 1] = float(np.mean(seed_values))
            stds[round_idx - 1] = (
                float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
            )
    return rounds, means, stds


def plot_all_configs(logs_root: Path, output_dir: Path) -> None:
    """Plot mean ± std convergence curves for every (strategy, sigma) config."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for strategy in STRATEGIES:
        for sigma in SIGMAS:
            rounds, means, stds = config_curve(logs_root, strategy, sigma)
            style = _STYLE[(strategy, sigma)]
            ax.plot(rounds, means, linewidth=2, **style)
            ax.fill_between(
                rounds, means - stds, means + stds, color=style["color"], alpha=0.15
            )

    ax.axhline(CENTRALIZED_F1, color="gray", linestyle=":", linewidth=1.5)
    ax.text(
        1.1,
        CENTRALIZED_F1 + 0.006,
        f"Centralized (F1={CENTRALIZED_F1})",
        fontsize=8,
        color="gray",
    )
    ax.axhline(FL_NODP_F1, color="gray", linestyle=":", linewidth=1.5)
    ax.text(
        1.1, FL_NODP_F1 + 0.006, f"FL no DP (F1={FL_NODP_F1})", fontsize=8, color="gray"
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Micro-F1")
    ax.set_title("FL Convergence by Round — Global vs. Per-Layer Clipping under DP-SGD")
    ax.set_xlim(1, NUM_ROUNDS)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "convergence_final.png"
    pdf_path = output_dir / "convergence_final.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    print(f"[plot_convergence] wrote {png_path} and {pdf_path}")


def main() -> None:
    """CLI entry point: write the convergence figure for all configs."""
    exp_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=exp_root / "logs")
    parser.add_argument("--output-dir", type=Path, default=exp_root / "figures")
    args = parser.parse_args()

    plot_all_configs(args.logs_dir, args.output_dir)


if __name__ == "__main__":
    main()
