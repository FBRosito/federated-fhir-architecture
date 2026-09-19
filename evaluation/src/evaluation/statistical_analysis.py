"""
statistical_analysis.py
-----------------------
Statistical rigor for Qualis A1 publication.

Functions:
  - confidence_interval(): mean ± 1.96·std/√n and bootstrap CI (n=1000 resamples)
  - wilcoxon_test():        Wilcoxon signed-rank, centralized vs federated
  - summarize_runs():       Consolidates results from multiple seeds into mean ± CI

Usage:
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
    """Confidence interval for a metric."""

    mean: float
    std: float
    ci_lower: float  # 2.5th percentile of the bootstrap
    ci_upper: float  # 97.5th percentile of the bootstrap
    n: int

    @property
    def pm(self) -> float:
        """Margin of error ±1.96·std/√n (t-distribution for n<30, z for n≥30)."""
        if self.n == 0:
            return float("nan")
        return 1.96 * self.std / max(self.n**0.5, 1)

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
    Computes mean, std, and 95% confidence interval via bootstrap.

    Args:
        values:      List of metrics (e.g. F1@5 across 3 seeds).
        n_bootstrap: Number of resamples for the bootstrap CI.
        seed:        Seed for reproducibility.

    Returns:
        ConfidenceInterval with mean, std, ci_lower, ci_upper.
    """
    arr = np.array(
        [v for v in values if v is not None and not np.isnan(v)], dtype=float
    )
    n = len(arr)

    if n == 0:
        return ConfidenceInterval(
            mean=float("nan"),
            std=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n=0,
        )

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    rng = np.random.default_rng(seed)
    boot_means = [
        float(np.mean(rng.choice(arr, size=n, replace=True)))
        for _ in range(n_bootstrap)
    ]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return ConfidenceInterval(
        mean=mean, std=std, ci_lower=ci_lower, ci_upper=ci_upper, n=n
    )


def wilcoxon_test(
    baseline: list[float],
    treatment: list[float],
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Wilcoxon signed-rank test to compare baseline vs treatment.

    Suitable for small samples (n=3 seeds) — does not assume normality.
    Standard in FL literature with few runs (He et al., 2020; McMahan et al., 2017).

    Args:
        baseline:    Centralized baseline metrics (or FL without DP).
        treatment:   Federated model metrics (or FL with DP).
        alternative: "two-sided" | "greater" | "less".

    Returns:
        {"statistic": W, "p_value": p, "n_pairs": n, "significant_at_05": bool}
    """
    try:
        from scipy.stats import wilcoxon

        if len(baseline) < 2 or len(baseline) != len(treatment):
            log.warning("wilcoxon_test: requires n≥2 matched pairs.")
            return {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "n_pairs": 0,
                "significant_at_05": False,
            }

        stat, p = wilcoxon(
            baseline, treatment, alternative=alternative, zero_method="wilcox"
        )
        return {
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "n_pairs": len(baseline),
            "significant_at_05": bool(p < 0.05),
        }
    except ImportError:
        log.warning("scipy not installed — Wilcoxon test will return NaN.")
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_pairs": len(baseline),
            "significant_at_05": False,
        }
    except Exception as exc:
        log.warning("wilcoxon_test failed: %s", exc)
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_pairs": len(baseline),
            "significant_at_05": False,
        }


def summarize_runs(
    metric_by_config: dict[str, list[float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, ConfidenceInterval]:
    """
    Consolidates metrics from multiple seeds per configuration.

    Args:
        metric_by_config: {"config_name": [seed42, seed123, seed777], ...}
        n_bootstrap:      Bootstrap resamples for CI.
        seed:             Seed for bootstrap.

    Returns:
        {"config_name": ConfidenceInterval, ...}

    Example:
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
    """Compares all configurations vs baseline with Wilcoxon signed-rank test.

    Args:
        metric_by_config: {"config": [values], ...}
        baseline_key:     Key of the baseline configuration (e.g. "centralizado").
        alternative:      Test alternative hypothesis.

    Returns:
        {"config_name": {"statistic": W, "p_value": p, ...}, ...}
    """
    if baseline_key not in metric_by_config:
        raise ValueError(f"Baseline '{baseline_key}' not found in metric_by_config.")

    baseline = metric_by_config[baseline_key]
    results = {}
    for config, values in metric_by_config.items():
        if config == baseline_key:
            continue
        results[config] = wilcoxon_test(baseline, values, alternative=alternative)
        log.info(
            "Wilcoxon %s vs %s: W=%.2f p=%.4f sig=%s",
            config,
            baseline_key,
            results[config]["statistic"],
            results[config]["p_value"],
            results[config]["significant_at_05"],
        )
    return results


def apply_bonferroni(
    comparisons: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Applies Bonferroni correction to multiple pairwise p-values.

    Corrects for family-wise error rate when testing m simultaneous comparisons:
        p_adj = min(p_raw × m, 1.0)
    Significance threshold becomes α_adj = 0.05 / m.

    Args:
        comparisons: Output of compare_all_vs_baseline() —
                     {"config": {"p_value": float, ...}, ...}

    Returns:
        Same dict with two new keys per config:
          "p_value_bonferroni"          — Bonferroni-adjusted p-value
          "significant_at_05_bonferroni" — bool, adjusted significance
    """
    m = len(comparisons)
    if m == 0:
        return comparisons

    result = {}
    for config, stats in comparisons.items():
        p_raw = stats.get("p_value", float("nan"))
        p_adj = min(p_raw * m, 1.0) if not np.isnan(p_raw) else float("nan")
        result[config] = {
            **stats,
            "p_value_bonferroni": round(p_adj, 6),
            "significant_at_05_bonferroni": (
                bool(p_adj < 0.05) if not np.isnan(p_adj) else False
            ),
        }
        log.info(
            "Bonferroni %s: p_raw=%.4f → p_adj=%.4f (m=%d) sig=%s",
            config,
            p_raw,
            p_adj,
            m,
            result[config]["significant_at_05_bonferroni"],
        )
    return result
