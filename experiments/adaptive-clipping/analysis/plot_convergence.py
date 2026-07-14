#!/usr/bin/env python3
"""
plot_convergence.py
--------------------
Convergence-by-round figure (equivalent to the original HERALD paper's
Figure 3): one line per configuration (baseline_sigma1.0, baseline_sigma2.0,
per_layer_sigma1.0, per_layer_sigma2.0), Micro-F1 mean +/- std over the 10
seeds per round. Saves figures/convergence.png and figures/convergence.pdf
at 300 DPI, matching evaluation/plots.py's savefig convention.

Usage:
    cd experiments/adaptive-clipping && uv run python -m analysis.plot_convergence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

STRATEGIES = ["baseline", "per_layer"]
SIGMAS = [1.0, 2.0]
NUM_SILOS = 5
NUM_ROUNDS = 20
METRIC = "micro_f1"

# Copied from evaluation.plots._COLORS (a private module attribute — copying
# the hex codes locally avoids depending on its mutability contract).
_BASELINE_COLOR = "#1f77b4"   # matches evaluation.plots._COLORS["fedprox"]
_PER_LAYER_COLOR = "#9467bd"  # matches evaluation.plots._COLORS["dp_light"]
_LINE_STYLES = {1.0: "-", 2.0: "--"}


def _require_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_convergence_series(logs_root: Path, strategy: str, sigma: float) -> pd.DataFrame:
    """Long-form DataFrame: round, seed, value — one row per (round, seed),
    value = unweighted mean of `micro_f1` across the run's 5 silos (see
    bootstrap_ci.py's per_seed_value for why this is unweighted)."""
    rows = []
    for run_dir in sorted(logs_root.glob(f"{strategy}_sigma{sigma}_seed*")):
        seed = int(run_dir.name.rsplit("seed", 1)[1])
        for round_idx in range(1, NUM_ROUNDS + 1):
            values = []
            for silo_id in range(NUM_SILOS):
                round_file = run_dir / str(silo_id) / f"round_{round_idx:03d}.jsonl"
                if not round_file.exists():
                    continue
                with open(round_file) as f:
                    record = json.loads(f.readline())
                if METRIC in record:
                    values.append(record[METRIC])
            if values:
                rows.append({"round": round_idx, "seed": seed, "value": sum(values) / len(values)})
    return pd.DataFrame(rows)


def plot_all_configs(logs_root: Path, output_dir: Path) -> None:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))

    for strategy in STRATEGIES:
        color = _BASELINE_COLOR if strategy == "baseline" else _PER_LAYER_COLOR
        for sigma in SIGMAS:
            df = load_convergence_series(logs_root, strategy, sigma)
            if df.empty:
                continue
            agg = df.groupby("round")["value"].agg(["mean", "std"]).reset_index()
            label = f"{strategy}_σ{sigma}"
            ax.plot(agg["round"], agg["mean"], color=color, linestyle=_LINE_STYLES[sigma], label=label)
            ax.fill_between(
                agg["round"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                color=color, alpha=0.15,
            )

    ax.set_xlabel("FL round")
    ax.set_ylabel("Micro-F1")
    ax.set_title("Convergence: baseline vs. per-layer adaptive clipping")
    ax.legend()
    ax.grid(alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "convergence.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_convergence] wrote {output_dir / 'convergence.png'} and convergence.pdf")


def main() -> None:
    exp_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=exp_root / "logs")
    parser.add_argument("--output-dir", type=Path, default=exp_root / "figures")
    args = parser.parse_args()

    plot_all_configs(args.logs_dir, args.output_dir)


if __name__ == "__main__":
    main()
