"""
plots.py
--------
Gráficos para publicação (paper Qualis A1).

Plots implementados:
  1. curva_epsilon_vs_f1()  — ε × F1@k para análise privacy-utility tradeoff
  2. curva_f1_vs_alpha()    — F1@k × α(Dirichlet) para heterogeneidade Non-IID
  3. convergence_curves()   — curvas de convergência loss/F1 por round

Uso:
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

import numpy as np

log = logging.getLogger(__name__)

# Paleta de cores consistente com publicações IEEE/JAMIA
_COLORS = {
    "centralizado": "#2ca02c",   # verde
    "fedprox":      "#1f77b4",   # azul
    "fedavg":       "#ff7f0e",   # laranja
    "dp_light":     "#9467bd",   # roxo
    "dp_mod":       "#8c564b",   # marrom
    "dp_strong":    "#e377c2",   # rosa
}


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib não instalado. Execute: uv add matplotlib")


def curva_epsilon_vs_f1(
    epsilon_values: list[float],
    f1_values:      list[float],
    f1_ci_lower:    list[float] | None = None,
    f1_ci_upper:    list[float] | None = None,
    baseline_f1:    float | None = None,
    metric_name:    str = "F1@5",
    output_path:    str = "figures/epsilon_vs_f1.pdf",
    title:          str = "Privacy-Utility Tradeoff (FedProx + DP-SGD)",
) -> None:
    """
    Plota curva ε × F1@k com intervalo de confiança (shaded area).

    Inclui linha horizontal para o baseline centralizado (sem DP) se fornecido.

    Args:
        epsilon_values: Valores de ε calculados pelo RDPAccountant.
        f1_values:      F1@k correspondente a cada ε (média sobre seeds).
        f1_ci_lower:    Limite inferior do CI 95% (bootstrap).
        f1_ci_upper:    Limite superior do CI 95% (bootstrap).
        baseline_f1:    F1@k do baseline centralizado (linha tracejada).
        metric_name:    Nome da métrica no eixo Y.
        output_path:    Caminho do arquivo de saída (.pdf ou .png).
        title:          Título do gráfico.
    """
    plt = _require_matplotlib()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(epsilon_values, f1_values, "o-", color=_COLORS["fedprox"],
            linewidth=2, markersize=6, label="FedProx + DP-SGD")

    if f1_ci_lower and f1_ci_upper:
        ax.fill_between(epsilon_values, f1_ci_lower, f1_ci_upper,
                        alpha=0.2, color=_COLORS["fedprox"], label="95% CI")

    if baseline_f1 is not None:
        ax.axhline(baseline_f1, linestyle="--", color=_COLORS["centralizado"],
                   linewidth=1.5, label="Centralizado (sem DP)")

    ax.set_xlabel("Privacy Budget (ε)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfico salvo em: %s", output_path)


def curva_f1_vs_alpha(
    alpha_values:    list[float],
    f1_fedprox:      list[float],
    f1_fedavg:       list[float] | None = None,
    f1_centralizado: float | None = None,
    ci_fedprox:      list[tuple[float, float]] | None = None,
    ci_fedavg:       list[tuple[float, float]] | None = None,
    metric_name:     str = "F1@5",
    output_path:     str = "figures/f1_vs_alpha.pdf",
    title:           str = "Non-IID Heterogeneity vs Performance",
) -> None:
    """
    Plota F1@k × α(Dirichlet) para FedProx, FedAvg e baseline centralizado.

    Args:
        alpha_values:    Valores de α (ex: [0.1, 0.5, 1.0]).
        f1_fedprox:      F1@k médio do FedProx para cada α.
        f1_fedavg:       F1@k médio do FedAvg para cada α (opcional).
        f1_centralizado: F1@k do baseline centralizado (linha horizontal).
        ci_fedprox:      Lista de (ci_lower, ci_upper) para FedProx.
        ci_fedavg:       Lista de (ci_lower, ci_upper) para FedAvg.
        metric_name:     Nome da métrica no eixo Y.
        output_path:     Caminho do arquivo de saída.
        title:           Título do gráfico.
    """
    plt = _require_matplotlib()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(alpha_values, f1_fedprox, "s-", color=_COLORS["fedprox"],
            linewidth=2, markersize=7, label="FedProx")

    if ci_fedprox:
        lower = [c[0] for c in ci_fedprox]
        upper = [c[1] for c in ci_fedprox]
        ax.fill_between(alpha_values, lower, upper, alpha=0.2, color=_COLORS["fedprox"])

    if f1_fedavg:
        ax.plot(alpha_values, f1_fedavg, "^--", color=_COLORS["fedavg"],
                linewidth=2, markersize=7, label="FedAvg")
        if ci_fedavg:
            lower = [c[0] for c in ci_fedavg]
            upper = [c[1] for c in ci_fedavg]
            ax.fill_between(alpha_values, lower, upper, alpha=0.2, color=_COLORS["fedavg"])

    if f1_centralizado is not None:
        ax.axhline(f1_centralizado, linestyle="--", color=_COLORS["centralizado"],
                   linewidth=1.5, label="Centralizado")

    ax.set_xlabel("Dirichlet α (heterogeneidade Non-IID ↑)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Anotações para cada ponto α
    for alpha, f1 in zip(alpha_values, f1_fedprox):
        ax.annotate(f"α={alpha}", (alpha, f1), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color=_COLORS["fedprox"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfico salvo em: %s", output_path)


def convergence_curves(
    rounds: list[int],
    metrics_by_config: dict[str, list[float]],
    metric_name: str = "F1@5",
    output_path: str = "figures/convergence.pdf",
    title: str = "FL Convergence by Round",
) -> None:
    """
    Plota curvas de convergência (loss ou F1) por round federado.

    Args:
        rounds:              Lista de números de round (eixo X).
        metrics_by_config:   {"config_name": [val_round1, val_round2, ...]}
        metric_name:         Nome da métrica (eixo Y).
        output_path:         Caminho do arquivo de saída.
        title:               Título.
    """
    plt = _require_matplotlib()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    color_cycle = list(_COLORS.values())
    for idx, (config, values) in enumerate(metrics_by_config.items()):
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(rounds[:len(values)], values, linewidth=2, label=config, color=color)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfico salvo em: %s", output_path)
