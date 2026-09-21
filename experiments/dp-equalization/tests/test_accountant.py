"""Tests for dp_equalization.accountant.epsilon().

Covers the three properties required by Prompt 2 / Stage 1:
  (i)   epsilon() agrees with tolerance-bounded expectations (sanity checks).
  (ii)  monotonicity: higher sigma -> lower (or equal) epsilon for fixed
        (q/silos, R, delta).
  (iii) regression test reproducing HERALD's known cumulative epsilons
        (305.92 / 55.70 / 17.37 for sigma 0.5/1.0/2.0, R=20, delta=1e-5).
"""

from __future__ import annotations

import pytest

from dp_equalization.accountant import MethodConfig, epsilon

# The 5 (k, n) = (optimizer steps/round, local train-set size) silo pairs
# HERALD's main pipeline used for its FedProx DP-SGD sweep. Copied from
# experiments/adaptive-clipping/analysis/renyi_group_composition.py:59-65
# (same table, verified against experiment_logs/fl_fedprox_alpha0.5_dp*
# _bert_seed42.json in docs/EQUALIZATION_READINESS.md's audit).
HERALD_SILOS = ((155, 1550), (150, 1508), (151, 1518), (152, 1527), (149, 1492))

# experiment_logs/fl_fedprox_alpha0.5_dp{sigma}_bert_seed42.json,
# per_round_eval[-1]["epsilon_cumulative"] (identical across seeds 42/43/44).
KNOWN_HERALD_EPSILON = {0.5: 305.9235, 1.0: 55.7041, 2.0: 17.3668}


def _record_level_config(sigma: float) -> MethodConfig:
    return MethodConfig(
        method="fedprox_dpsgd_record_level",
        sigma=sigma,
        rounds=20,
        delta=1e-5,
        silos=HERALD_SILOS,
        accountant="rdp",
    )


def _client_level_config(sigma: float) -> MethodConfig:
    return MethodConfig(
        method="client_level_dp",
        sigma=sigma,
        rounds=20,
        delta=1e-5,
        accountant="client_level",
    )


class TestBasicBehaviour:
    def test_zero_sigma_returns_inf_for_rdp(self):
        assert epsilon(_record_level_config(0.0)) == float("inf")

    def test_zero_sigma_returns_inf_for_client_level(self):
        assert epsilon(_client_level_config(0.0)) == float("inf")

    def test_unknown_accountant_raises(self):
        cfg = MethodConfig(method="x", sigma=1.0, rounds=20, accountant="bogus")
        with pytest.raises(ValueError):
            epsilon(cfg)

    def test_rdp_without_silos_or_rate_raises(self):
        cfg = MethodConfig(method="x", sigma=1.0, rounds=20)
        with pytest.raises(ValueError):
            epsilon(cfg)

    def test_rdp_with_explicit_sample_rate_and_steps(self):
        cfg = MethodConfig(
            method="explicit",
            sigma=1.0,
            rounds=20,
            sample_rate=0.1,
            steps_per_round=150,
        )
        eps = epsilon(cfg)
        assert eps > 0 and eps < float("inf")


class TestMonotonicity:
    def test_monotonic_in_sigma_record_level(self):
        eps_low_sigma = epsilon(_record_level_config(0.5))
        eps_mid_sigma = epsilon(_record_level_config(1.0))
        eps_high_sigma = epsilon(_record_level_config(2.0))
        assert eps_low_sigma > eps_mid_sigma > eps_high_sigma, (
            "critical invariant: higher noise multiplier must spend less "
            "epsilon for identical (q, R, delta)"
        )

    def test_monotonic_in_sigma_client_level(self):
        eps_low_sigma = epsilon(_client_level_config(0.5))
        eps_mid_sigma = epsilon(_client_level_config(1.0))
        eps_high_sigma = epsilon(_client_level_config(2.0))
        assert eps_low_sigma > eps_mid_sigma > eps_high_sigma


class TestReproducesKnownHeraldEpsilons:
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_reproduces_known_herald_epsilons(self, sigma):
        got = epsilon(_record_level_config(sigma))
        target = KNOWN_HERALD_EPSILON[sigma]
        rel_error = abs(got - target) / target
        assert rel_error < 0.001, (
            f"sigma={sigma}: got eps={got:.4f}, HERALD logged {target} "
            f"(rel_error={rel_error:.4%}) -- accountant call or silo table "
            "has drifted from the logged experiment"
        )
