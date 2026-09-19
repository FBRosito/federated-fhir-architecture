"""
Tests for evaluation.grad_norm_logger — the Fase 0 instrumentation that
records per-round pre-/post-clip gradient norms, clip thresholds, and
cumulative epsilon for the DP-SGD critical path. This module had zero test
coverage prior to this change.
"""

import json

import torch

from evaluation.grad_norm_logger import (
    GradNormLogger,
    group_by_layer,
    is_enabled,
    snapshot_group_norms,
)


def _param(shape=(2,)) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros(shape))


class TestIsEnabled:
    def test_should_default_to_enabled(self, monkeypatch):
        monkeypatch.delenv("LOG_GRAD_NORMS", raising=False)
        assert is_enabled() is True

    def test_should_disable_on_zero(self, monkeypatch):
        monkeypatch.setenv("LOG_GRAD_NORMS", "0")
        assert is_enabled() is False

    def test_should_disable_on_false_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LOG_GRAD_NORMS", "False")
        assert is_enabled() is False

    def test_should_stay_enabled_for_arbitrary_truthy_value(self, monkeypatch):
        monkeypatch.setenv("LOG_GRAD_NORMS", "yes")
        assert is_enabled() is True


class TestGroupByLayer:
    def test_should_group_encoder_layer_pattern(self):
        params = [
            ("bert.encoder.layer.0.attention.self.query.weight", _param()),
            ("bert.encoder.layer.1.attention.self.query.weight", _param()),
        ]
        groups = group_by_layer(params)
        assert "encoder.layer.0" in groups
        assert "encoder.layer.1" in groups

    def test_should_split_lora_a_and_lora_b_within_a_group(self):
        params = [
            ("bert.encoder.layer.0.attention.self.query.lora_A.weight", _param()),
            ("bert.encoder.layer.0.attention.self.query.lora_B.weight", _param()),
        ]
        groups = group_by_layer(params)
        assert "encoder.layer.0.lora_A" in groups
        assert "encoder.layer.0.lora_B" in groups
        # LoRA A/B must not collapse into the same group (each group holds a
        # distinct parameter object, not a shared reference).
        assert groups["encoder.layer.0.lora_A"][0] is not groups["encoder.layer.0.lora_B"][0]

    def test_should_fallback_to_other_for_unmatched_names(self):
        params = [("classifier.head.weight", _param())]
        groups = group_by_layer(params)
        assert "other" in groups

    def test_should_match_gpt_style_h_pattern(self):
        params = [("transformer.h.3.mlp.c_fc.weight", _param())]
        groups = group_by_layer(params)
        assert "h.3" in groups


class TestSnapshotGroupNorms:
    def test_should_omit_groups_with_no_gradient(self):
        p = _param()
        p.grad = None
        norms = snapshot_group_norms([("other.param", p)], attr="grad")
        assert norms == {}, "a group with no grad tensor must be omitted, not zero"

    def test_should_compute_l2_norm_of_present_gradients(self):
        p = _param(shape=(4,))
        p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm = 5.0
        norms = snapshot_group_norms([("other.param", p)], attr="grad")
        assert norms["other"] == 5.0

    def test_should_combine_multiple_params_in_the_same_group(self):
        p1 = torch.nn.Parameter(torch.zeros(2))
        p1.grad = torch.tensor([3.0, 0.0])
        p2 = torch.nn.Parameter(torch.zeros(2))
        p2.grad = torch.tensor([4.0, 0.0])
        norms = snapshot_group_norms(
            [("other.p1", p1), ("other.p2", p2)], attr="grad"
        )
        # L2 combination of two per-tensor norms [3.0, 4.0] -> 5.0
        assert norms["other"] == 5.0


class TestGradNormLoggerJSONL:
    def test_should_noop_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_GRAD_NORMS", "0")
        logger = GradNormLogger(run_id="t1", logs_dir=tmp_path)
        logger.log_round(
            server_round=1,
            partition_id=0,
            grad_norms_pre_clip={"other": 1.0},
            grad_norms_post_clip_noise={"other": 0.5},
            clip_thresholds=1.0,
            learning_rate=1e-4,
            epsilon_cumulative=None,
            noise_multiplier=0.9,
        )
        assert list(tmp_path.iterdir()) == []

    def test_should_write_one_jsonl_record_per_round(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_GRAD_NORMS", "1")
        logger = GradNormLogger(run_id="run_test", logs_dir=tmp_path)
        logger.log_round(
            server_round=1,
            partition_id=2,
            grad_norms_pre_clip={"other": 2.0},
            grad_norms_post_clip_noise={"other": 1.0},
            clip_thresholds=1.0,
            learning_rate=2e-4,
            epsilon_cumulative=0.75,
            noise_multiplier=0.9,
        )
        files = list(tmp_path.glob("grad_norms_run_test.jsonl"))
        assert len(files) == 1
        record = json.loads(files[0].read_text().strip())
        assert record["server_round"] == 1
        assert record["partition_id"] == 2
        assert record["epsilon_cumulative"] == 0.75

    def test_should_append_across_multiple_rounds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_GRAD_NORMS", "1")
        logger = GradNormLogger(run_id="run_multi", logs_dir=tmp_path)
        for rnd in (1, 2):
            logger.log_round(
                server_round=rnd,
                partition_id=0,
                grad_norms_pre_clip={},
                grad_norms_post_clip_noise={},
                clip_thresholds=1.0,
                learning_rate=1e-4,
                epsilon_cumulative=float(rnd),
                noise_multiplier=0.9,
            )
        lines = (tmp_path / "grad_norms_run_multi.jsonl").read_text().splitlines()
        assert len(lines) == 2

    def test_should_sanitize_run_id_used_in_filename(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_GRAD_NORMS", "1")
        logger = GradNormLogger(run_id="weird/run:id!", logs_dir=tmp_path)
        logger.log_round(
            server_round=1,
            partition_id=0,
            grad_norms_pre_clip={},
            grad_norms_post_clip_noise={},
            clip_thresholds=1.0,
            learning_rate=1e-4,
            epsilon_cumulative=None,
            noise_multiplier=0.0,
        )
        produced = list(tmp_path.iterdir())
        assert len(produced) == 1
        assert "/" not in produced[0].name
        assert ":" not in produced[0].name
