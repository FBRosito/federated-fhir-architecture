"""
Tests for fl_server.server privacy-budget estimation and per-round config
generation (learning-rate schedule, proximal_mu forwarding).
"""

import math

import pytest

from fl_server.server import estimate_privacy_budget, make_fit_config_fn


class TestEstimatePrivacyBudget:
    def test_should_return_inf_when_noise_multiplier_zero(self):
        assert estimate_privacy_budget(0.0, 5, 2, 4) == float("inf")

    def test_should_return_inf_when_noise_multiplier_negative(self):
        assert estimate_privacy_budget(-1.0, 5, 2, 4) == float("inf")

    def test_should_increase_with_more_rounds(self):
        eps_5 = estimate_privacy_budget(0.9, 5, 2, 4)
        eps_10 = estimate_privacy_budget(0.9, 10, 2, 4)
        assert eps_10 > eps_5, (
            "accumulated epsilon must never decrease as rounds increase "
            "(critical invariant #1)"
        )

    def test_should_decrease_with_higher_noise_multiplier(self):
        eps_low_sigma = estimate_privacy_budget(0.5, 5, 2, 4)
        eps_high_sigma = estimate_privacy_budget(2.0, 5, 2, 4)
        assert eps_high_sigma < eps_low_sigma

    def test_should_not_divide_by_zero_when_total_clients_zero(self):
        # total_clients=0 must not raise ZeroDivisionError (guarded by
        # max(total_clients, 1) in the implementation).
        result = estimate_privacy_budget(0.9, 5, 2, 0)
        assert math.isfinite(result)


class TestFitConfigLearningRateSchedule:
    """Round-boundary (off-by-one) check for the documented LR decay:
    'LR decays to 40% after round 2' — i.e. rounds 1-2 get the full LR,
    round 3 onward gets 40%."""

    def test_should_use_full_lr_through_round_2(self):
        fit_config = make_fit_config_fn(proximal_mu=0.01, num_epochs=1)
        cfg1 = fit_config(1)
        cfg2 = fit_config(2)
        assert cfg1["learning_rate"] == cfg2["learning_rate"]

    def test_should_decay_lr_starting_round_3(self):
        fit_config = make_fit_config_fn(proximal_mu=0.01, num_epochs=1)
        cfg2 = fit_config(2)
        cfg3 = fit_config(3)
        assert cfg3["learning_rate"] == pytest.approx(cfg2["learning_rate"] * 0.4)

    def test_should_forward_proximal_mu_unmodified(self):
        fit_config = make_fit_config_fn(proximal_mu=0.037, num_epochs=1)
        cfg = fit_config(1)
        assert cfg["proximal_mu"] == 0.037

    def test_should_forward_num_epochs(self):
        fit_config = make_fit_config_fn(proximal_mu=0.01, num_epochs=3)
        cfg = fit_config(1)
        assert cfg["num_epochs"] == 3
