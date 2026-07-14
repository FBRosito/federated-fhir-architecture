#!/usr/bin/env python3
"""
generate_tables.py
--------------------
Builds the Article 2 summary table (Configuration, sigma, Micro-F1 mean+-std,
epsilon, CI_lower, CI_upper) from bootstrap_ci.py's output CSV, in both CSV
and LaTeX. Highlights the best Micro-F1/epsilon trade-off: the highest
Micro-F1 among configurations whose epsilon CI does not exceed the
baseline's epsilon CI at the same sigma (does not spend more privacy budget).

Usage:
    cd experiments/adaptive-clipping && uv run python -m analysis.bootstrap_ci   # first
    cd experiments/adaptive-clipping && uv run python -m analysis.generate_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.bootstrap_ci import check_overlap
from evaluation.statistical_analysis import ConfidenceInterval


def _ci_from_row(row: pd.Series) -> ConfidenceInterval:
    return ConfidenceInterval(
        mean=row["mean"], std=0.0, ci_lower=row["ci_lower"], ci_upper=row["ci_upper"], n=int(row["n_seeds"]),
    )


def build_table(ci_csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(ci_csv_path)

    rows = []
    for (strategy, sigma), group in raw.groupby(["strategy", "sigma"]):
        f1_row = group[group["metric"] == "micro_f1"].iloc[0]
        eps_row = group[group["metric"] == "epsilon"].iloc[0]
        rows.append({
            "Configuration": strategy,
            "sigma": sigma,
            "micro_f1_mean": f1_row["mean"],
            "micro_f1_ci_lower": f1_row["ci_lower"],
            "micro_f1_ci_upper": f1_row["ci_upper"],
            "epsilon_mean": eps_row["mean"],
            "epsilon_ci_lower": eps_row["ci_lower"],
            "epsilon_ci_upper": eps_row["ci_upper"],
        })
    return pd.DataFrame(rows)


def highlight_best_tradeoff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["best_tradeoff"] = False

    for sigma, group in df.groupby("sigma"):
        baseline_rows = group[group["Configuration"] == "baseline"]
        if baseline_rows.empty:
            continue
        baseline_row = baseline_rows.iloc[0]
        baseline_eps_ci = ConfidenceInterval(
            mean=baseline_row["epsilon_mean"], std=0.0,
            ci_lower=baseline_row["epsilon_ci_lower"], ci_upper=baseline_row["epsilon_ci_upper"],
            n=0,
        )

        candidates = []
        for idx, row in group.iterrows():
            eps_ci = ConfidenceInterval(
                mean=row["epsilon_mean"], std=0.0,
                ci_lower=row["epsilon_ci_lower"], ci_upper=row["epsilon_ci_upper"], n=0,
            )
            not_worse = check_overlap(eps_ci, baseline_eps_ci) or eps_ci.ci_upper <= baseline_eps_ci.ci_upper
            if not_worse:
                candidates.append(idx)

        if candidates:
            best_idx = max(candidates, key=lambda i: df.loc[i, "micro_f1_mean"])
            df.loc[best_idx, "best_tradeoff"] = True

    return df


def to_latex(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & $\sigma$ & Micro-F1 & $\varepsilon$ & CI [lower, upper] \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        f1_cell = f"{row['micro_f1_mean']:.4f}"
        if row["best_tradeoff"]:
            f1_cell = r"\textbf{" + f1_cell + "}"
        lines.append(
            f"{row['Configuration']} & {row['sigma']:.1f} & {f1_cell} & "
            f"{row['epsilon_mean']:.2f} & [{row['micro_f1_ci_lower']:.4f}, {row['micro_f1_ci_upper']:.4f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    exp_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-csv", type=Path, default=exp_root / "logs" / "bootstrap_ci.csv")
    parser.add_argument("--output-csv", type=Path, default=exp_root / "logs" / "table_iii.csv")
    parser.add_argument("--output-tex", type=Path, default=exp_root / "logs" / "table_iii.tex")
    args = parser.parse_args()

    df = build_table(args.ci_csv)
    df = highlight_best_tradeoff(df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    to_latex(df, args.output_tex)

    print(f"[generate_tables] wrote {args.output_csv} and {args.output_tex}")


if __name__ == "__main__":
    main()
