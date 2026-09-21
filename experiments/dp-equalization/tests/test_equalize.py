"""Tests for dp_equalization.equalize.equalize()."""

from __future__ import annotations

import pytest

from dp_equalization.accountant import MethodConfig, epsilon
from dp_equalization.equalize import equalize

HERALD_SILOS = ((155, 1550), (150, 1508), (151, 1518), (152, 1527), (149, 1492))

RECORD_LEVEL_TEMPLATE = MethodConfig(
    method="fedprox_dpsgd_record_level",
    sigma=1.0,  # placeholder -- equalize() ignores/replaces this
    rounds=20,
    delta=1e-5,
    silos=HERALD_SILOS,
    accountant="rdp",
)

CLIENT_LEVEL_TEMPLATE = MethodConfig(
    method="client_level_dp",
    sigma=1.0,
    rounds=20,
    delta=1e-5,
    accountant="client_level",
)


class TestHitsTargetWithinTolerance:
    @pytest.mark.parametrize("eps_star", [1.0, 4.0, 8.0])
    def test_record_level(self, eps_star):
        cfg, eps_achieved = equalize(RECORD_LEVEL_TEMPLATE, eps_star)
        assert abs(eps_achieved - eps_star) / eps_star < 0.01
        assert abs(epsilon(cfg) - eps_star) / eps_star < 0.01

    @pytest.mark.parametrize("eps_star", [1.0, 4.0, 8.0])
    def test_client_level(self, eps_star):
        cfg, eps_achieved = equalize(CLIENT_LEVEL_TEMPLATE, eps_star)
        assert abs(eps_achieved - eps_star) / eps_star < 0.01
        assert abs(epsilon(cfg) - eps_star) / eps_star < 0.01


class TestPreservesTemplateFields:
    def test_equalize_only_changes_sigma(self):
        cfg, _ = equalize(RECORD_LEVEL_TEMPLATE, eps_star=4.0)
        assert cfg.method == RECORD_LEVEL_TEMPLATE.method
        assert cfg.rounds == RECORD_LEVEL_TEMPLATE.rounds
        assert cfg.delta == RECORD_LEVEL_TEMPLATE.delta
        assert cfg.silos == RECORD_LEVEL_TEMPLATE.silos
        assert cfg.sigma != RECORD_LEVEL_TEMPLATE.sigma


class TestTighterBudgetNeedsMoreNoise:
    def test_smaller_eps_star_yields_larger_sigma(self):
        cfg_tight, _ = equalize(RECORD_LEVEL_TEMPLATE, eps_star=1.0)
        cfg_loose, _ = equalize(RECORD_LEVEL_TEMPLATE, eps_star=8.0)
        assert cfg_tight.sigma > cfg_loose.sigma


class TestUnreachableTargets:
    def test_raises_when_target_too_small_for_bounds(self):
        with pytest.raises(ValueError):
            equalize(RECORD_LEVEL_TEMPLATE, eps_star=1e-6, sigma_bounds=(1e-3, 10.0))

    def test_raises_when_target_too_large_for_bounds(self):
        with pytest.raises(ValueError):
            equalize(RECORD_LEVEL_TEMPLATE, eps_star=1e6, sigma_bounds=(0.5, 10.0))

    def test_raises_on_nonpositive_eps_star(self):
        with pytest.raises(ValueError):
            equalize(RECORD_LEVEL_TEMPLATE, eps_star=0.0)
