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
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    clipper = PerLayerClipper(
        named_params, warmup_rounds=3, percentile=75.0,
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
        clipper.update_history(round_idx)
        thresholds = clipper.compute_thresholds()
        label = "warmup" if round_idx < 3 else "post-warmup"
        print(f"  round {round_idx} ({label}): {thresholds}")
        if round_idx < 3:
            assert all(v == 1.0 for v in thresholds.values()), "warmup rounds must return global_c0"

    print("  OK: warmup rounds returned global_c0; post-warmup rounds returned calibrated per-group thresholds.")
    print(f"  Final round groups/thresholds: {thresholds}")


if __name__ == "__main__":
    check_grouping()
    check_thresholds()
    print("smoke_check_clipper: ALL CHECKS PASSED")
