#!/usr/bin/env python3
"""
compare_strategies.py
----------------------
Paired baseline vs per_layer comparison (same seeds) at each sigma: Wilcoxon
signed-rank test on micro_f1, plus a bootstrap CI on the paired difference
(per_layer - baseline). Bonferroni-corrects across the 2 sigma comparisons.

Usage:
    cd experiments/adaptive-clipping && uv run python -m analysis.compare_strategies
"""

from __future__ import annotations

from pathlib import Path

from analysis.bootstrap_ci import SIGMAS, load_final_round_records, per_seed_value

from evaluation.statistical_analysis import (
    apply_bonferroni,
    confidence_interval,
    wilcoxon_test,
)

METRIC = "micro_f1"


def paired_seed_values(
    logs_root: Path, sigma: float
) -> tuple[list[int], list[float], list[float]]:
    """Returns (seeds, baseline_values, per_layer_values) aligned by seed."""
    baseline_by_seed = load_final_round_records(logs_root, "baseline", sigma)
    per_layer_by_seed = load_final_round_records(logs_root, "per_layer", sigma)
    common_seeds = sorted(set(baseline_by_seed) & set(per_layer_by_seed))
    baseline_values = [
        per_seed_value(baseline_by_seed[s], METRIC) for s in common_seeds
    ]
    per_layer_values = [
        per_seed_value(per_layer_by_seed[s], METRIC) for s in common_seeds
    ]
    return common_seeds, baseline_values, per_layer_values


def main() -> None:
    """CLI entry point: paired per-seed comparison of per-layer vs baseline clipping."""
    exp_root = Path(__file__).resolve().parent.parent
    logs_root = exp_root / "logs"

    comparisons = {}
    diff_cis = {}
    for sigma in SIGMAS:
        seeds, baseline_values, per_layer_values = paired_seed_values(logs_root, sigma)
        key = f"per_layer_sigma{sigma}"
        comparisons[key] = wilcoxon_test(
            baseline_values, per_layer_values, alternative="two-sided"
        )
        diffs = [pl - bl for pl, bl in zip(per_layer_values, baseline_values)]
        diff_cis[key] = confidence_interval(diffs, n_bootstrap=10000, seed=42)
        print(f"sigma={sigma}  n_seeds={len(seeds)}")
        print(
            f"  baseline  {METRIC}: mean={sum(baseline_values)/len(baseline_values):.4f}"
        )
        print(
            f"  per_layer {METRIC}: mean={sum(per_layer_values)/len(per_layer_values):.4f}"
        )
        print(f"  paired difference (per_layer - baseline): {diff_cis[key]}")
        print(f"  Wilcoxon: {comparisons[key]}")
        print()

    corrected = apply_bonferroni(comparisons)
    print("After Bonferroni correction (m=2):")
    for key, stats in corrected.items():
        print(
            f"  {key}: p_adj={stats['p_value_bonferroni']:.6f} "
            f"significant={stats['significant_at_05_bonferroni']}"
        )


if __name__ == "__main__":
    main()
