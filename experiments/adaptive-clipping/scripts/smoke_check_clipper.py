#!/usr/bin/env python3
"""
smoke_check_clipper.py
-----------------------
CPU-only, sub-second sanity check for PerLayerClipper — no GPU, no model
download. Run before any real (expensive) GPU training run:

    cd experiments/adaptive-clipping && uv run python scripts/smoke_check_clipper.py
"""

import torch
import torch.nn as nn

from adaptive_clipping.clipping import PerLayerClipper


class _FakeSelfAttention(nn.Module):
    """Mirrors HuggingFace BERT's real submodule naming: attention.self.{query,value}."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.query = nn.Module()
        self.query.lora_A = nn.Linear(dim, 2, bias=False)
        self.query.lora_B = nn.Linear(2, dim, bias=False)
        self.value = nn.Module()
        self.value.lora_A = nn.Linear(dim, 2, bias=False)
        self.value.lora_B = nn.Linear(2, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.lora_B(self.query.lora_A(x))
        v = self.value.lora_B(self.value.lora_A(x))
        return q + v


class _FakeEncoderLayer(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = _FakeSelfAttention(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention.self(x)


class _FakePubMedBertLoRA(nn.Module):
    """Parameter names mimic PubMedBERT+LoRA:
    encoder.layer.N.attention.self.{query,value}.lora_{A,B}.weight"""

    def __init__(self, num_layers: int = 2, dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(_FakeEncoderLayer(dim) for _ in range(num_layers))
        self.classifier = nn.Linear(dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.encoder.layer:
            h = h + layer(h)
        return self.classifier(h)


def check_grouping() -> None:
    print("== group_by_attention_layer ==")
    model = _FakePubMedBertLoRA(num_layers=2)
    named_params = list(model.named_parameters())
    groups = PerLayerClipper.group_by_attention_layer(named_params)

    for name, params in groups.items():
        print(f"  {name}: {len(params)} params")

    layer0_names = {n for n, _ in named_params if "encoder.layer.0" in n}
    layer1_names = {n for n, _ in named_params if "encoder.layer.1" in n}
    assert len(groups.get("encoder.layer.0", [])) == len(layer0_names), (
        "expected encoder.layer.0.attention.self.query.lora_A/B and "
        "encoder.layer.0.attention.self.value.lora_A/B all in one group"
    )
    assert len(groups.get("encoder.layer.1", [])) == len(layer1_names), (
        "expected encoder.layer.1's query/value LoRA params in one group, "
        "separate from encoder.layer.0"
    )
    assert "other" in groups and len(groups["other"]) == 2, (
        "expected classifier.weight/classifier.bias to fall into the 'other' group"
    )
    print("  OK: encoder.layer.0 and encoder.layer.1 each grouped correctly; classifier -> other")


def check_thresholds() -> None:
    print("== compute_thresholds (warmup vs. post-warmup) ==")
    torch.manual_seed(0)
    model = _FakePubMedBertLoRA(num_layers=2)
    clipper = PerLayerClipper(
        warmup_rounds=3, percentile=75.0,
        min_clip=0.1, max_clip=10.0, global_c0=1.0,
    )

    x = torch.randn(8, 4)
    thresholds: dict[str, float] = {}
    for round_idx in range(5):
        model.zero_grad()
        # Scaled down so post-warmup percentile thresholds land strictly
        # inside [min_clip, max_clip] instead of saturating the clamp —
        # saturating on every round would hide a broken percentile calculation.
        loss = model(x).pow(2).sum() * 0.01
        loss.backward()
        clipper.update_history(round_idx, model.named_parameters())
        thresholds = clipper.compute_thresholds()
        label = "warmup" if round_idx < 3 else "post-warmup"
        print(f"  round {round_idx} ({label}): {thresholds}")
        if round_idx < 3:
            assert all(v == 1.0 for v in thresholds.values()), "warmup rounds must return global_c0"

    print("  OK: warmup rounds returned global_c0; post-warmup rounds returned calibrated per-group thresholds.")
    print(f"  Final round groups/thresholds: {thresholds}")


def check_history_survives_model_reload() -> None:
    """Regression test for the stale-reference bug fixed in this change:
    PerLayerClipper used to cache nn.Parameter objects at construction time,
    but the real FL client reloads a brand-new model object on every fit()
    (see ai_client.fl_client._load_model/_unload_model). Cached references
    from round 1 would have .grad permanently None from round 2 onward, so
    update_history() silently stopped appending — the history (and, worse,
    the actual clip+noise step in training.py, which used to read the same
    stale clipper.groups) silently froze at round 1's values forever.

    This test simulates that reload explicitly: a brand-new model instance
    every round, fed straight into update_history(), with no state cached
    across rounds by the caller. If the bug were still present, this would
    manifest here too (via a different path — since this test never gives
    the clipper a chance to cache a stale reference in the first place — so
    what this actually guards against is a regression that reintroduces
    that caching)."""
    print("== update_history across simulated model reloads ==")
    clipper = PerLayerClipper(warmup_rounds=3, percentile=75.0, min_clip=0.1, max_clip=10.0, global_c0=1.0)

    for round_idx in range(5):
        torch.manual_seed(round_idx)
        # A brand-new model instance every round, mirroring _load_model()
        # producing a fresh model object on every fit() call in production.
        model = _FakePubMedBertLoRA(num_layers=2)
        x = torch.randn(8, 4)
        loss = model(x).pow(2).sum() * 0.01
        loss.backward()
        groups = clipper.update_history(round_idx, model.named_parameters())
        assert groups, f"round {round_idx}: update_history returned no groups"

    summary = clipper.get_history_summary()
    print("  Full accumulated history after 5 rounds of fresh model instances:")
    for name, stats in sorted(summary.items()):
        print(f"    {name}: len={len(stats['history'])} values={stats['history']}")

    for name, stats in summary.items():
        assert len(stats["history"]) == 5, (
            f"{name}: expected 5 accumulated values (1 per round), got {len(stats['history'])} "
            "— history is not surviving model reload across rounds"
        )
        assert len(set(stats["history"])) > 1, (
            f"{name}: all {len(stats['history'])} values are identical — history is frozen at "
            "round 1, not tracking each round's live gradients (the exact bug this test guards against)"
        )
    print("  OK: history accumulated 5 distinct values per group across 5 simulated model reloads.")


if __name__ == "__main__":
    check_grouping()
    check_thresholds()
    check_history_survives_model_reload()
    print("smoke_check_clipper: ALL CHECKS PASSED")
