"""
plots.py
--------
Publication figures (Qualis A1 paper).

Plots implemented:
  1. curva_epsilon_vs_f1()  — ε × F1@k for privacy-utility tradeoff analysis
  2. curva_f1_vs_alpha()    — F1@k × α(Dirichlet) for Non-IID heterogeneity
  3. convergence_curves()   — loss/F1 convergence curves by round

Usage:
    from evaluation.plots import curva_epsilon_vs_f1, curva_f1_vs_alpha

    curva_epsilon_vs_f1(
        epsilon_values=[2.1, 4.3, 8.7],
        f1_values=[0.71, 0.69, 0.64],
        output_path="figures/epsilon_vs_f1.pdf",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Color palette consistent with IEEE/JAMIA publications
_COLORS = {
    "centralizado": "#2ca02c",  # green
    "fedprox": "#1f77b4",  # blue
    "fedavg": "#ff7f0e",  # orange
    "dp_light": "#9467bd",  # purple
    "dp_mod": "#8c564b",  # brown
    "dp_strong": "#e377c2",  # pink
}


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError("matplotlib not installed. Run: uv add matplotlib")


def curva_epsilon_vs_f1(
    epsilon_values: list[float],
    f1_values: list[float],
    f1_ci_lower: list[float] | None = None,
    f1_ci_upper: list[float] | None = None,
    baseline_f1: float | None = None,
    metric_name: str = "F1@5",
    output_path: str = "figures/epsilon_vs_f1.pdf",
    title: str = "Privacy-Utility Tradeoff (FedProx + DP-SGD)",
) -> None:
    """
    Plots ε × F1@k curve with confidence interval (shaded area).

    Includes a horizontal line for the centralized baseline (no DP) if provided.

    Args:
        epsilon_values: ε values computed by the RDPAccountant.
        f1_values:      F1@k corresponding to each ε (mean across seeds).
        f1_ci_lower:    Lower bound of the 95% CI (bootstrap).
        f1_ci_upper:    Upper bound of the 95% CI (bootstrap).
        baseline_f1:    F1@k of the centralized baseline (dashed line).
        metric_name:    Metric name on the Y axis.
        output_path:    Output file path (.pdf or .png).
        title:          Plot title.
    """
    plt = _require_matplotlib()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        epsilon_values,
        f1_values,
        "o-",
        color=_COLORS["fedprox"],
        linewidth=2,
        markersize=6,
        label="FedProx + DP-SGD",
    )

    if f1_ci_lower and f1_ci_upper:
        ax.fill_between(
            epsilon_values,
            f1_ci_lower,
            f1_ci_upper,
            alpha=0.2,
            color=_COLORS["fedprox"],
            label="95% CI",
        )

    if baseline_f1 is not None:
        ax.axhline(
            baseline_f1,
            linestyle="--",
            color=_COLORS["centralizado"],
            linewidth=1.5,
            label="Centralised (no DP)",
        )

    ax.set_xlabel("Privacy Budget (ε)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved to: %s", output_path)


def curva_f1_vs_alpha(
    alpha_values: list[float],
    f1_fedprox: list[float],
    f1_fedavg: list[float] | None = None,
    f1_centralizado: float | None = None,
    ci_fedprox: list[tuple[float, float]] | None = None,
    ci_fedavg: list[tuple[float, float]] | None = None,
    metric_name: str = "F1@5",
    output_path: str = "figures/f1_vs_alpha.pdf",
    title: str = "Non-IID Heterogeneity vs Performance",
) -> None:
    """
    Plots F1@k × α(Dirichlet) for FedProx, FedAvg, and centralized baseline.

    Args:
        alpha_values:    α values (e.g. [0.1, 0.5, 1.0]).
        f1_fedprox:      Mean FedProx F1@k for each α.
        f1_fedavg:       Mean FedAvg F1@k for each α (optional).
        f1_centralizado: Centralized baseline F1@k (horizontal line).
        ci_fedprox:      List of (ci_lower, ci_upper) for FedProx.
        ci_fedavg:       List of (ci_lower, ci_upper) for FedAvg.
        metric_name:     Metric name on the Y axis.
        output_path:     Output file path.
        title:           Plot title.
    """
    plt = _require_matplotlib()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        alpha_values,
        f1_fedprox,
        "s-",
        color=_COLORS["fedprox"],
        linewidth=2,
        markersize=7,
        label="FedProx",
    )

    if ci_fedprox:
        lower = [c[0] for c in ci_fedprox]
        upper = [c[1] for c in ci_fedprox]
        ax.fill_between(alpha_values, lower, upper, alpha=0.2, color=_COLORS["fedprox"])

    if f1_fedavg:
        ax.plot(
            alpha_values,
            f1_fedavg,
            "^--",
            color=_COLORS["fedavg"],
            linewidth=2,
            markersize=7,
            label="FedAvg",
        )
        if ci_fedavg:
            lower = [c[0] for c in ci_fedavg]
            upper = [c[1] for c in ci_fedavg]
            ax.fill_between(
                alpha_values, lower, upper, alpha=0.2, color=_COLORS["fedavg"]
            )

    if f1_centralizado is not None:
        ax.axhline(
            f1_centralizado,
            linestyle="--",
            color=_COLORS["centralizado"],
            linewidth=1.5,
            label="Centralised",
        )

    ax.set_xlabel("Dirichlet α (Non-IID heterogeneity ↑)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotations for each α point
    for alpha, f1 in zip(alpha_values, f1_fedprox):
        ax.annotate(
            f"α={alpha}",
            (alpha, f1),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=_COLORS["fedprox"],
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved to: %s", output_path)


def convergence_curves(
    rounds: list[int],
    metrics_by_config: dict[str, list[float]],
    metric_name: str = "F1@5",
    output_path: str = "figures/convergence.pdf",
    title: str = "FL Convergence by Round",
) -> None:
    """
    Plots convergence curves (loss or F1) by federated round.

    Args:
        rounds:              List of round numbers (X axis).
        metrics_by_config:   {"config_name": [val_round1, val_round2, ...]}
        metric_name:         Metric name (Y axis).
        output_path:         Output file path.
        title:               Plot title.
    """
    plt = _require_matplotlib()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    color_cycle = list(_COLORS.values())
    for idx, (config, values) in enumerate(metrics_by_config.items()):
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(rounds[: len(values)], values, linewidth=2, label=config, color=color)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved to: %s", output_path)
