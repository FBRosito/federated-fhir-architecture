#!/usr/bin/env python3
"""
bootstrap_ci.py
----------------
Bootstrap confidence intervals (B=10000, percentile method) for Micro-F1 and
epsilon at the final round (R=20), across the 4 configurations
(baseline/per_layer x sigma 1.0/2.0).

Usage:
    cd experiments/adaptive-clipping && uv run python -m analysis.bootstrap_ci
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.statistical_analysis import ConfidenceInterval, confidence_interval

STRATEGIES = ["baseline", "per_layer"]
SIGMAS = [1.0, 2.0]
NUM_SILOS = 5
FINAL_ROUND = 20
N_BOOTSTRAP = 10000


def load_final_round_records(
    logs_root: Path, strategy: str, sigma: float
) -> dict[int, list[dict]]:
    """Returns {seed: [round_020 record per silo]} for one (strategy, sigma) config."""
    records_by_seed: dict[int, list[dict]] = {}
    for run_dir in sorted(logs_root.glob(f"{strategy}_sigma{sigma}_seed*")):
        # The glob also matches loose *_server.log / *_silo{N}.log files that
        # sit flat in logs_root next to the run directories — skip those.
        if not run_dir.is_dir():
            continue
        seed = int(run_dir.name.rsplit("seed", 1)[1])
        records = []
        for silo_id in range(NUM_SILOS):
            round_file = run_dir / str(silo_id) / f"round_{FINAL_ROUND:03d}.jsonl"
            if not round_file.exists():
                continue
            with open(round_file) as f:
                records.append(json.loads(f.readline()))
        if records:
            records_by_seed[seed] = records
    return records_by_seed


def per_seed_value(records: list[dict], metric: str) -> float:
    """Mean across the 5 silos for one seed. Unweighted: RoundLogRecord
    (per the approved JSONL schema) does not carry a per-silo example count,
    so a weighted-by-n_examples mean is not derivable from these logs."""
    values = [r[metric] for r in records if metric in r]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def bootstrap_config(
    logs_root: Path,
    strategy: str,
    sigma: float,
    metric: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> ConfidenceInterval:
    """Bootstrap the 95% CI of ``metric`` for one (strategy, sigma) config across seeds."""
    records_by_seed = load_final_round_records(logs_root, strategy, sigma)
    seed_values = [
        per_seed_value(records, metric) for records in records_by_seed.values()
    ]
    return confidence_interval(seed_values, n_bootstrap=n_bootstrap, seed=42)


def check_overlap(ci_a: ConfidenceInterval, ci_b: ConfidenceInterval) -> bool:
    """True if the two 95% CIs overlap."""
    return not (ci_a.ci_upper < ci_b.ci_lower or ci_b.ci_upper < ci_a.ci_lower)


def main() -> None:
    """CLI entry point: compute bootstrap CIs for all configs and write the CSV."""
    exp_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=exp_root / "logs")
    parser.add_argument(
        "--output", type=Path, default=exp_root / "logs" / "bootstrap_ci.csv"
    )
    args = parser.parse_args()

    rows = []
    for strategy in STRATEGIES:
        for sigma in SIGMAS:
            for metric in ("micro_f1", "epsilon"):
                ci = bootstrap_config(args.logs_dir, strategy, sigma, metric)
                rows.append(
                    {
                        "strategy": strategy,
                        "sigma": sigma,
                        "metric": metric,
                        "mean": ci.mean,
                        "ci_lower": ci.ci_lower,
                        "ci_upper": ci.ci_upper,
                        "n_seeds": ci.n,
                    }
                )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[bootstrap_ci] wrote {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
