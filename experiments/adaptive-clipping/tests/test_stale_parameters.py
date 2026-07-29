"""
test_stale_parameters.py
------------------------
Fase 0 regression test for the historical stale-nn.Parameter-reference bug
(Bug 1, see docs/validated_environment.md): the HERALD FL client reloads a
brand-new model object every round via ai_client.fl_client._load_model(), so
any component that captures nn.Parameter references at construction time and
reuses them across rounds ends up with .grad permanently None from round 2
onward — clipping/noise silently stops applying while RDPAccountant keeps
reporting a valid-looking epsilon.

Run from experiments/adaptive-clipping (its own uv-managed venv owns torch,
numpy, pytest — this package is not part of the root uv workspace):

    cd experiments/adaptive-clipping && uv run pytest tests/test_stale_parameters.py -v
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from adaptive_clipping.clipping import (
    PerLayerClipper,
    assert_model_reloaded,
    param_identity_fingerprint,
)


# ── Fixtures mirroring PubMedBERT+LoRA parameter naming ──────────────────────
# (same shape as scripts/smoke_check_clipper.py's fakes, duplicated here so
# this test has no dependency on that script.)

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


def _train_one_step(model: nn.Module, seed: int) -> None:
    """One forward+backward pass, populating .grad for every trainable param."""
    torch.manual_seed(seed)
    model.zero_grad()
    x = torch.randn(8, 4)
    # Scaled down so post-warmup percentile thresholds land inside
    # [min_clip, max_clip] rather than saturating the clamp.
    loss = model(x).pow(2).sum() * 0.01
    loss.backward()


# ── 1. History survives simulated model reload (the original Bug 1 pattern) ─

def test_history_accumulates_across_simulated_model_reloads():
    """A brand-new model instance every round, mirroring _load_model()
    producing a fresh model object on every fit() call in production. If
    PerLayerClipper cached nn.Parameter references (the pre-fix pattern),
    every round after the first would see .grad stay None on the ORIGINAL
    (now-orphaned) parameters, and history would freeze at round 1's values.
    """
    clipper = PerLayerClipper(warmup_rounds=3, percentile=75.0, min_clip=0.1, max_clip=10.0, global_c0=1.0)

    for round_idx in range(1, 6):  # Flower's server_round is 1-indexed
        model = _FakePubMedBertLoRA(num_layers=2)
        _train_one_step(model, seed=round_idx)
        groups = clipper.update_history(round_idx, model.named_parameters())
        assert groups, f"round {round_idx}: update_history returned no groups"

    summary = clipper.get_history_summary()
    assert summary, "expected at least one parameter group in history"
    for name, stats in summary.items():
        assert len(stats["history"]) == 5, (
            f"{name}: expected 5 accumulated values (1 per round), got "
            f"{len(stats['history'])} — history is not surviving model reload "
            f"across rounds (the exact Bug 1 symptom)."
        )
        assert len(set(stats["history"])) > 1, (
            f"{name}: all {len(stats['history'])} values are identical — history is "
            f"frozen, not tracking each round's live gradients (the exact bug this "
            f"test guards against)."
        )


def test_grouping_is_stable_across_reloads():
    """group_by_attention_layer must key on parameter NAME (stable across
    reloads), not on any per-instance identity — otherwise a fresh model's
    parameters would silently fail to join the accumulated history at all."""
    clipper = PerLayerClipper(warmup_rounds=1, percentile=75.0, min_clip=0.1, max_clip=10.0)
    group_keys_per_round = []
    for round_idx in range(1, 4):
        model = _FakePubMedBertLoRA(num_layers=2)
        _train_one_step(model, seed=round_idx)
        groups = clipper.update_history(round_idx, model.named_parameters())
        group_keys_per_round.append(frozenset(groups.keys()))
    assert len(set(group_keys_per_round)) == 1, (
        f"group keys changed across reloads: {group_keys_per_round} — thresholds "
        f"computed per-group would silently stop applying to a group that "
        f"disappears."
    )


# ── 2. Grad-presence guard (the direct, observable symptom of Bug 1) ────────

def test_update_history_raises_when_a_group_has_no_gradient():
    """Simulates the DIRECT symptom of Bug 1: a stale nn.Parameter belonging
    to an already-replaced model has .grad permanently None. Feeds
    update_history() a real model's named_parameters() WITHOUT calling
    backward() first, so every param's .grad is None — this must raise
    rather than silently recording a zero/skipped entry.
    """
    clipper = PerLayerClipper(warmup_rounds=1, percentile=75.0, min_clip=0.1, max_clip=10.0)
    model = _FakePubMedBertLoRA(num_layers=2)
    # No backward() call: every .grad is None, exactly like a stale reference
    # to a deleted model's parameters.
    with pytest.raises(RuntimeError, match="NONE have a .grad"):
        clipper.update_history(1, model.named_parameters())


def test_update_history_ok_when_gradients_present():
    """Sanity counterpart to the above: a normal round (backward() called
    first) must NOT raise."""
    clipper = PerLayerClipper(warmup_rounds=1, percentile=75.0, min_clip=0.1, max_clip=10.0)
    model = _FakePubMedBertLoRA(num_layers=2)
    _train_one_step(model, seed=0)
    groups = clipper.update_history(1, model.named_parameters())  # must not raise
    assert groups


# ── 3. History-length integrity (call-count based, convention-agnostic) ─────

def test_history_length_matches_call_count():
    clipper = PerLayerClipper(warmup_rounds=1, percentile=75.0, min_clip=0.1, max_clip=10.0)
    for i in range(1, 4):
        model = _FakePubMedBertLoRA(num_layers=2)
        _train_one_step(model, seed=i)
        clipper.update_history(i, model.named_parameters())
    for name, stats in clipper.get_history_summary().items():
        assert len(stats["history"]) == 3, (
            f"{name}: expected exactly 3 accumulated values after 3 calls, got "
            f"{len(stats['history'])}."
        )


# ── 4. Model-reload identity fingerprint (client-level guard primitive) ─────

def test_assert_model_reloaded_passes_for_distinct_model_objects():
    """Two genuinely different model instances (the correct, reload-every-
    round behavior) must not raise."""
    model_round1 = _FakePubMedBertLoRA(num_layers=2)
    model_round2 = _FakePubMedBertLoRA(num_layers=2)
    fp1 = param_identity_fingerprint(model_round1.named_parameters())
    assert_model_reloaded(fp1, previous_fingerprint=None, component_name="test")  # first call: no-op
    fp2 = param_identity_fingerprint(model_round2.named_parameters())
    assert_model_reloaded(fp2, previous_fingerprint=fp1, component_name="test")  # different objects: OK


def test_assert_model_reloaded_raises_for_same_model_object_reused():
    """THE regression test for Bug 1's root cause: if a caller (accidentally)
    reuses the SAME model object across rounds instead of reloading — the
    exact precondition that let stale nn.Parameter references go undetected
    — this must raise, unless FL_KEEP_MODEL_IN_VRAM=true opts in explicitly.
    """
    os.environ.pop("FL_KEEP_MODEL_IN_VRAM", None)
    model = _FakePubMedBertLoRA(num_layers=2)  # the SAME instance both "rounds"
    fp_round1 = param_identity_fingerprint(model.named_parameters())
    fp_round2 = param_identity_fingerprint(model.named_parameters())  # same object again
    with pytest.raises(RuntimeError, match="exact same nn.Parameter objects"):
        assert_model_reloaded(fp_round2, previous_fingerprint=fp_round1, component_name="test")


def test_assert_model_reloaded_allows_reuse_when_keep_in_vram_is_true():
    """FL_KEEP_MODEL_IN_VRAM=true is a legitimate opt-in to reusing the same
    model object across rounds (performance mode) — must NOT raise then."""
    os.environ["FL_KEEP_MODEL_IN_VRAM"] = "true"
    try:
        model = _FakePubMedBertLoRA(num_layers=2)
        fp = param_identity_fingerprint(model.named_parameters())
        assert_model_reloaded(fp, previous_fingerprint=fp, component_name="test")  # must not raise
    finally:
        os.environ.pop("FL_KEEP_MODEL_IN_VRAM", None)


# ── 5. Demonstrates this test suite actually catches the pre-fix pattern ────

class _BuggyClipperCachesParamsAtInit:
    """Reimplements PerLayerClipper's PRE-FIX behavior for this one test:
    caches nn.Parameter references at construction time and reuses them on
    every call, ignoring the freshly-passed-in named_parameters(). This is
    the literal Bug 1 pattern this whole file guards against — it exists
    ONLY to prove the tests above would have failed loudly against it,
    without needing to hand-revert the real fix in clipping.py.
    """

    def __init__(self, model: nn.Module) -> None:
        # THE BUG: captured once, at construction, from whatever model
        # happens to exist right now.
        self._cached_named_params = list(model.named_parameters())
        self._history: dict[str, list[float]] = {}

    def update_history(self, round_idx: int, _ignored_named_parameters) -> None:
        # THE BUG: ignores this round's live parameters entirely and reuses
        # the construction-time snapshot instead.
        groups = PerLayerClipper.group_by_attention_layer(
            [(n, p) for n, p in self._cached_named_params if p.requires_grad]
        )
        for name, params in groups.items():
            grads = [p.grad.detach() for p in params if p.grad is not None]
            self._history.setdefault(name, []).append(
                float(torch.norm(torch.stack([g.norm(2) for g in grads]), 2).item())
                if grads else 0.0
            )


def test_buggy_pre_fix_pattern_would_have_frozen_history():
    """Reproduces Bug 1 end-to-end with the pre-fix pattern: constructs the
    buggy clipper against round 1's model, then simulates rounds 2-4 with
    BRAND-NEW model objects (as the real FL client always reloads) fed only
    to backward() — never to the buggy clipper, which stubbornly holds onto
    round 1's now-orphaned parameters. Their .grad goes stale (whatever it
    was after round 1, since nothing ever calls backward() on those exact
    objects again), demonstrating exactly why PerLayerClipper's real
    update_history() raises instead of silently repeating a frozen value.
    """
    model_round1 = _FakePubMedBertLoRA(num_layers=2)
    _train_one_step(model_round1, seed=0)
    buggy = _BuggyClipperCachesParamsAtInit(model_round1)
    buggy.update_history(1, model_round1.named_parameters())

    for round_idx in range(2, 5):
        fresh_model = _FakePubMedBertLoRA(num_layers=2)  # real _load_model() behavior
        _train_one_step(fresh_model, seed=round_idx)
        # THE BUG in action: production code would pass fresh_model.named_parameters()
        # here, but the buggy clipper ignores the argument and reuses round 1's
        # cached (now stale, unchanged) parameters instead.
        buggy.update_history(round_idx, fresh_model.named_parameters())

    for name, history in buggy._history.items():
        assert len(set(history)) == 1, (
            f"expected the buggy pre-fix pattern to freeze '{name}' at a single "
            f"repeated value across rounds 1-4 (proving it never observed rounds "
            f"2-4's live gradients) — got {history}, meaning this reproduction of "
            f"the bug is not faithful."
        )

    # And the real, fixed PerLayerClipper does NOT freeze under the identical
    # round-by-round model sequence:
    real_clipper = PerLayerClipper(warmup_rounds=1, percentile=75.0, min_clip=0.1, max_clip=10.0)
    for round_idx in range(1, 5):
        fresh_model = _FakePubMedBertLoRA(num_layers=2)
        _train_one_step(fresh_model, seed=round_idx)
        real_clipper.update_history(round_idx, fresh_model.named_parameters())
    for name, stats in real_clipper.get_history_summary().items():
        assert len(set(stats["history"])) > 1, (
            f"'{name}': the FIXED PerLayerClipper also froze under the same "
            f"round sequence — the fix would have regressed."
        )
