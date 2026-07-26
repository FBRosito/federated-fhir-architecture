#!/usr/bin/env python3
"""
generate_tables.py
--------------------
Builds the Article 2 results table (Table III) from bootstrap_ci.py's CSV
output and compare_strategies.py's paired Wilcoxon/Bonferroni logic, in both
CSV and LaTeX.

Rows are grouped by sigma (baseline then per_layer within each sigma), with
a separator between the sigma=1.0 and sigma=2.0 blocks, followed by the
Centralized/FL-no-DP reference rows from the original HERALD paper.

"Best trade-off" = the highest Micro-F1 among per_layer configurations whose
95% CI does not overlap the baseline CI at the same sigma (a statistically
supported gain, not just a numerically higher mean).

Usage:
    cd experiments/adaptive-clipping && uv run python -m analysis.bootstrap_ci        # first
    cd experiments/adaptive-clipping && uv run python -m analysis.generate_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bootstrap_ci import SIGMAS, check_overlap, load_final_round_records, per_seed_value
from analysis.compare_strategies import paired_seed_values
from evaluation.statistical_analysis import ConfidenceInterval, apply_bonferroni, wilcoxon_test

CENTRALIZED_F1 = 0.409
FL_NODP_F1 = 0.308


def _f1_std(logs_root: Path, strategy: str, sigma: float) -> float:
    records_by_seed = load_final_round_records(logs_root, strategy, sigma)
    seed_values = [per_seed_value(records, "micro_f1") for records in records_by_seed.values()]
    return float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0


def build_table(ci_csv_path: Path, logs_root: Path) -> pd.DataFrame:
    raw = pd.read_csv(ci_csv_path)

    # Paired comparisons (per_layer vs baseline) at each sigma, reusing
    # compare_strategies.py's own logic so the numbers here can never drift
    # from what that script reports independently.
    comparisons = {}
    for sigma in SIGMAS:
        _, baseline_values, per_layer_values = paired_seed_values(logs_root, sigma)
        comparisons[sigma] = wilcoxon_test(baseline_values, per_layer_values, alternative="two-sided")
    comparisons = apply_bonferroni(comparisons)

    rows = []
    for sigma in SIGMAS:
        group = raw[raw["sigma"] == sigma]
        f1_by_strategy = {s: group[(group["strategy"] == s) & (group["metric"] == "micro_f1")].iloc[0]
                           for s in ("baseline", "per_layer")}
        eps_by_strategy = {s: group[(group["strategy"] == s) & (group["metric"] == "epsilon")].iloc[0]
                            for s in ("baseline", "per_layer")}
        baseline_f1_mean = f1_by_strategy["baseline"]["mean"]

        for strategy in ("baseline", "per_layer"):
            f1_row = f1_by_strategy[strategy]
            eps_row = eps_by_strategy[strategy]
            is_baseline = strategy == "baseline"
            rows.append({
                "Configuration": strategy,
                "sigma": sigma,
                "micro_f1_mean": f1_row["mean"],
                "micro_f1_std": _f1_std(logs_root, strategy, sigma),
                "micro_f1_ci_lower": f1_row["ci_lower"],
                "micro_f1_ci_upper": f1_row["ci_upper"],
                "epsilon_mean": eps_row["mean"],
                "delta_vs_baseline": None if is_baseline else f1_row["mean"] - baseline_f1_mean,
                "p_bonferroni": None if is_baseline else comparisons[sigma]["p_value_bonferroni"],
            })

    df = pd.DataFrame(rows)

    # Best trade-off: highest Micro-F1 among per_layer rows whose CI does not
    # overlap the baseline CI at the same sigma.
    df["best_tradeoff"] = False
    for sigma in SIGMAS:
        block = df[df["sigma"] == sigma]
        baseline_row = block[block["Configuration"] == "baseline"].iloc[0]
        baseline_ci = ConfidenceInterval(
            mean=baseline_row["micro_f1_mean"], std=0.0,
            ci_lower=baseline_row["micro_f1_ci_lower"], ci_upper=baseline_row["micro_f1_ci_upper"], n=10,
        )
        candidates = []
        for idx, row in block[block["Configuration"] == "per_layer"].iterrows():
            row_ci = ConfidenceInterval(
                mean=row["micro_f1_mean"], std=0.0,
                ci_lower=row["micro_f1_ci_lower"], ci_upper=row["micro_f1_ci_upper"], n=10,
            )
            if not check_overlap(row_ci, baseline_ci):
                candidates.append(idx)
        if candidates:
            best_idx = max(candidates, key=lambda i: df.loc[i, "micro_f1_mean"])
            df.loc[best_idx, "best_tradeoff"] = True

    return df


def to_latex(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Configuration & $\sigma$ & Micro-F1 (mean$\pm$std) & $\varepsilon$ (mean) & "
        r"95\% CI & $\Delta$ vs.\ baseline & $p$ (Bonferroni) \\",
        r"\midrule",
    ]

    for sigma_idx, sigma in enumerate(sorted(df["sigma"].unique())):
        if sigma_idx > 0:
            lines.append(r"\midrule")
        block = df[df["sigma"] == sigma]
        for _, row in block.iterrows():
            f1_cell = f"{row['micro_f1_mean']:.4f}$\\pm${row['micro_f1_std']:.4f}"
            if row["best_tradeoff"]:
                f1_cell = r"\textbf{" + f1_cell + "}"
            delta_cell = "—" if pd.isna(row["delta_vs_baseline"]) else f"{row['delta_vs_baseline']:+.4f}"
            p_cell = "—" if pd.isna(row["p_bonferroni"]) else f"{row['p_bonferroni']:.4f}"
            config_label = "Baseline (global clip)" if row["Configuration"] == "baseline" else "Per-layer clip"
            lines.append(
                f"{config_label} & {sigma:.1f} & {f1_cell} & {row['epsilon_mean']:.2f} & "
                f"[{row['micro_f1_ci_lower']:.4f}, {row['micro_f1_ci_upper']:.4f}] & "
                f"{delta_cell} & {p_cell} \\\\"
            )

    lines.append(r"\midrule")
    lines.append(
        f"Centralized$^\\dagger$ & — & {CENTRALIZED_F1:.3f} & — & — & — & — \\\\"
    )
    lines.append(
        f"FL, no DP$^\\dagger$ & — & {FL_NODP_F1:.3f} & — & — & — & — \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\\[2pt]")
    lines.append(
        r"{\footnotesize $^\dagger$Centralized and FL-no-DP figures are taken from the original "
        r"HERALD paper/experiment logs, not re-run in this matrix.}"
    )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    exp_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-csv", type=Path, default=exp_root / "logs" / "bootstrap_ci.csv")
    parser.add_argument("--logs-dir", type=Path, default=exp_root / "logs")
    parser.add_argument("--output-csv", type=Path, default=exp_root / "analysis" / "table_results.csv")
    parser.add_argument("--output-tex", type=Path, default=exp_root / "analysis" / "table_results.tex")
    args = parser.parse_args()

    df = build_table(args.ci_csv, args.logs_dir)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    to_latex(df, args.output_tex)

    print(f"[generate_tables] wrote {args.output_csv} and {args.output_tex}")


if __name__ == "__main__":
    main()
