"""
Guards the DP privacy-parameter defaults declared in
ai_client.model_setup.TrainingConfig against silent drift. Per the
bug-hunter-tester git rules, C0 and delta must never change without an
explicit, reviewed decision — a passing baseline here plus a clear failure
message makes any future drift immediately visible in CI.
"""

from ai_client.model_setup import TrainingConfig


class TestTrainingConfigPrivacyDefaults:
    def test_default_clip_norm_is_c0_1_0(self):
        cfg = TrainingConfig()
        assert cfg.max_grad_norm == 1.0, (
            "C0 (max_grad_norm) default drifted from 1.0 — this is a "
            "protected privacy parameter."
        )

    def test_default_target_delta_is_1e_5(self):
        cfg = TrainingConfig()
        assert cfg.target_delta == 1e-5, (
            "target_delta default drifted from 1e-5 — this is a protected "
            "privacy parameter."
        )

    def test_default_noise_multiplier_is_dp_disabled(self):
        # 0.0 means DP-SGD is off by default; a nonzero default would
        # silently change every caller that does not pass noise_multiplier.
        cfg = TrainingConfig()
        assert cfg.noise_multiplier == 0.0

    def test_default_proximal_mu_is_zero(self):
        # FedProx mu defaults to 0 in TrainingConfig; the FL client is
        # responsible for supplying 0.01 from the server round config.
        cfg = TrainingConfig()
        assert cfg.proximal_mu == 0.0
