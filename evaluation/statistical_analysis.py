"""
statistical_analysis.py
-----------------------
Rigor estatístico para publicação Qualis A1.

Funções:
  - confidence_interval(): média ± 1.96·std/√n e bootstrap CI (n=1000 resamples)
  - wilcoxon_test():        Wilcoxon signed-rank, centralizado vs federado
  - summarize_runs():       Consolida resultados de múltiplas seeds em média ± CI

Uso:
    from evaluation.statistical_analysis import summarize_runs, wilcoxon_test

    summary = summarize_runs({"fl_fedprox_alpha0.5": [0.72, 0.71, 0.73]})
    p_val   = wilcoxon_test([0.60, 0.62, 0.61], [0.72, 0.71, 0.73])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Intervalo de confiança para uma métrica."""
    mean:       float
    std:        float
    ci_lower:   float   # percentil 2.5% do bootstrap
    ci_upper:   float   # percentil 97.5% do bootstrap
    n:          int

    @property
    def pm(self) -> float:
        """Margem de erro ±1.96·std/√n (distribuição t para n<30, z para n≥30)."""
        if self.n == 0:
            return float("nan")
        return 1.96 * self.std / max(self.n ** 0.5, 1)

    def __str__(self) -> str:
        return (
            f"{self.mean:.4f} ± {self.pm:.4f} "
            f"[95% CI: {self.ci_lower:.4f}–{self.ci_upper:.4f}] (n={self.n})"
        )


def confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> ConfidenceInterval:
    """
    Calcula média, std e intervalo de confiança 95% via bootstrap.

    Args:
        values:      Lista de métricas (ex: F1@5 nas 3 seeds).
        n_bootstrap: Número de resamples para o bootstrap CI.
        seed:        Semente para reprodutibilidade.

    Returns:
        ConfidenceInterval com mean, std, ci_lower, ci_upper.
    """
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    n = len(arr)

    if n == 0:
        return ConfidenceInterval(
            mean=float("nan"), std=float("nan"),
            ci_lower=float("nan"), ci_upper=float("nan"), n=0
        )

    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    rng      = np.random.default_rng(seed)
    boot_means = [float(np.mean(rng.choice(arr, size=n, replace=True))) for _ in range(n_bootstrap)]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return ConfidenceInterval(mean=mean, std=std, ci_lower=ci_lower, ci_upper=ci_upper, n=n)


def wilcoxon_test(
    baseline: list[float],
    treatment: list[float],
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Wilcoxon signed-rank test para comparar baseline vs tratamento.

    Adequado para amostras pequenas (n=3 seeds) — não assume normalidade.
    Padrão na literatura de FL com poucos runs (He et al., 2020; McMahan et al., 2017).

    Args:
        baseline:    Métricas do baseline centralizado (ou FL sem DP).
        treatment:   Métricas do modelo federado (ou FL com DP).
        alternative: "two-sided" | "greater" | "less".

    Returns:
        {"statistic": W, "p_value": p, "n_pairs": n, "significant_at_05": bool}
    """
    try:
        from scipy.stats import wilcoxon
        if len(baseline) < 2 or len(baseline) != len(treatment):
            log.warning("wilcoxon_test: precisa de n≥2 pares correspondentes.")
            return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": 0, "significant_at_05": False}

        stat, p = wilcoxon(baseline, treatment, alternative=alternative, zero_method="wilcox")
        return {
            "statistic":        round(float(stat), 6),
            "p_value":          round(float(p), 6),
            "n_pairs":          len(baseline),
            "significant_at_05": bool(p < 0.05),
        }
    except ImportError:
        log.warning("scipy não instalado — Wilcoxon test retornará NaN.")
        return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": len(baseline), "significant_at_05": False}
    except Exception as exc:
        log.warning("wilcoxon_test falhou: %s", exc)
        return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": len(baseline), "significant_at_05": False}


def summarize_runs(
    metric_by_config: dict[str, list[float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, ConfidenceInterval]:
    """
    Consolida métricas de múltiplas seeds por configuração.

    Args:
        metric_by_config: {"config_name": [seed42, seed123, seed777], ...}
        n_bootstrap:      Resamples bootstrap para CI.
        seed:             Semente para bootstrap.

    Returns:
        {"config_name": ConfidenceInterval, ...}

    Exemplo:
        summarize_runs({
            "centralizado":           [0.60, 0.62, 0.61],
            "fl_fedprox_alpha0.5":    [0.72, 0.71, 0.73],
            "fl_fedprox_alpha0.5_dp": [0.68, 0.69, 0.67],
        })
    """
    result = {}
    for config, values in metric_by_config.items():
        ci = confidence_interval(values, n_bootstrap=n_bootstrap, seed=seed)
        result[config] = ci
        log.info("%-40s  %s", config, ci)
    return result


def compare_all_vs_baseline(
    metric_by_config: dict[str, list[float]],
    baseline_key: str,
    alternative: str = "two-sided",
) -> dict[str, dict[str, float]]:
    """
    Compara todas as configurações vs baseline com Wilcoxon signed-rank.

    Args:
        metric_by_config: {"config": [values], ...}
        baseline_key:     Chave da configuração baseline (ex: "centralizado").
        alternative:      Alternativa do teste.

    Returns:
        {"config_name": {"statistic": W, "p_value": p, ...}, ...}
    """
    if baseline_key not in metric_by_config:
        raise ValueError(f"Baseline '{baseline_key}' não encontrado em metric_by_config.")

    baseline = metric_by_config[baseline_key]
    results = {}
    for config, values in metric_by_config.items():
        if config == baseline_key:
            continue
        results[config] = wilcoxon_test(baseline, values, alternative=alternative)
        log.info(
            "Wilcoxon %s vs %s: W=%.2f p=%.4f sig=%s",
            config, baseline_key,
            results[config]["statistic"], results[config]["p_value"],
            results[config]["significant_at_05"],
        )
    return results
