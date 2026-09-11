"""
clipping.py
-----------
Tracks per-attention-layer gradient-norm history and computes adaptive
clipping thresholds. Contains NO Opacus/PrivacyEngine integration and
performs no clipping itself — training.py owns clip+noise application.
This separation keeps clipping.py importable and unit-testable without
CUDA/model weights (only torch, numpy, re, collections).
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch

# Grouping key: an attention layer is a functional unit of the transformer,
# not an individual LoRA projection — see README.md for the full rationale.
_LAYER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"encoder\.layer\.(\d+)"), "encoder.layer.{}"),
    (re.compile(r"\blayers\.(\d+)"), "layers.{}"),
    (re.compile(r"\bh\.(\d+)"), "h.{}"),
]


def param_identity_fingerprint(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> frozenset[int]:
    """Fase 0 stale-reference guard, reusable by any component that captures
    parameters and survives multiple FL rounds (clippers, accountants,
    client-side aggregators — see docs/validated_environment.md).

    Returns a frozenset of id(p) for every parameter — an identity
    fingerprint, not a value hash. HERALD reloads a brand-new model object
    every round via ai_client.fl_client._load_model() (unless
    FL_KEEP_MODEL_IN_VRAM=true), so a fresh model's parameters get new
    Python object ids every round. Call this once per round on the live
    model's named_parameters() and compare against the previous round's
    fingerprint:

        fp = param_identity_fingerprint(model.named_parameters())
        if fp == self._prev_fp and os.environ.get("FL_KEEP_MODEL_IN_VRAM", "false").lower() != "true":
            raise RuntimeError("model did not reload between rounds — Bug 1 pattern")
        self._prev_fp = fp

    This lives at module level (not inside PerLayerClipper) because the
    "did the model actually reload" check is a property of the CALLER's
    model lifecycle, not of any specific clipper/accountant — see
    AdaptiveClippingClient.fit() in client.py for the real call site.
    """
    return frozenset(id(p) for _, p in named_parameters)


def assert_model_reloaded(
    current_fingerprint: frozenset[int],
    previous_fingerprint: frozenset[int] | None,
    *,
    component_name: str,
) -> None:
    """Raises if `current_fingerprint` is identical to `previous_fingerprint`
    while FL_KEEP_MODEL_IN_VRAM is not 'true' — see param_identity_fingerprint.
    No-ops on the first call (previous_fingerprint is None) or when
    FL_KEEP_MODEL_IN_VRAM=true (same object reuse is then intentional).
    """
    if previous_fingerprint is None:
        return
    keep_in_vram = (
        os.environ.get("FL_KEEP_MODEL_IN_VRAM", "false").strip().lower() == "true"
    )
    if keep_in_vram:
        return
    if current_fingerprint == previous_fingerprint:
        raise RuntimeError(
            f"{component_name}: received the exact same nn.Parameter objects as the "
            f"previous round, but FL_KEEP_MODEL_IN_VRAM is not 'true' — a freshly "
            f"reloaded model is expected every round. This is the historical "
            f"stale-reference bug's signature (Bug 1): either _load_model() did not "
            f"reload, or a cached model/parameter snapshot from a previous round is "
            f"being reused. Set FL_KEEP_MODEL_IN_VRAM=true if reusing the model "
            f"object across rounds is intentional here."
        )


class PerLayerClipper:
    """
    Tracks per-layer gradient norms across FL rounds and computes
    adaptive clipping thresholds calibrated to each LoRA layer's
    historical gradient distribution.

    Holds NO reference to nn.Parameter objects between calls. The FL client
    reloads a brand-new model object on every fit() (see ai_client.fl_client
    _load_model/_unload_model) — any nn.Parameter cached here across rounds
    would belong to a model already deleted, so its .grad is permanently
    None from round 2 onward and the history would silently freeze at
    round 1's values. update_history() therefore takes the current round's
    live named_parameters explicitly, every call.
    """

    def __init__(
        self,
        warmup_rounds: int = 3,
        percentile: float = 75.0,
        min_clip: float = 0.1,
        max_clip: float = 10.0,
        global_c0: float = 1.0,
    ) -> None:
        self.warmup_rounds = warmup_rounds
        self.percentile = percentile
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.global_c0 = global_c0

        self._history: dict[str, list[float]] = defaultdict(list)
        self._current_round = 0

        # Call-count (NOT round_idx) backing the history-length integrity
        # check: production passes 1-indexed server_round while some tests
        # use 0-indexed round numbers, so asserting against a caller-supplied
        # round number would be convention-dependent. Counting calls directly
        # is convention-agnostic and checks exactly the invariant that
        # matters — one appended value per update_history() call, no more,
        # no fewer.
        self._n_updates = 0

    @staticmethod
    def group_by_attention_layer(
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> dict[str, list[torch.nn.Parameter]]:
        """
        Groups LoRA parameters by transformer attention layer index.
        Rationale: an attention layer is a functional unit of the transformer.
        Calibrating clipping at this granularity has cleaner theoretical
        justification than per-projection grouping.

        Grouping key: extracted from parameter name via regex matching
        patterns like 'encoder.layer.N', 'layers.N', 'h.N'.
        Falls back to 'other' for parameters not matching any pattern
        (e.g., classifier head, label attention).

        Stateless — safe to call with a fresh named_parameters() every round.
        """
        groups: dict[str, list[torch.nn.Parameter]] = {}
        for name, param in named_parameters:
            group_key = "other"
            for pattern, template in _LAYER_PATTERNS:
                match = pattern.search(name)
                if match:
                    group_key = template.format(match.group(1))
                    break
            groups.setdefault(group_key, []).append(param)
        return groups

    def update_history(
        self,
        round_idx: int,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> dict[str, list[torch.nn.Parameter]]:
        """Called after backward(), before any clipping, once per optimizer
        step. named_parameters must be the CURRENT round's live model
        parameters — never a reference captured at construction time (see
        class docstring). Grouping is recomputed fresh from these live
        parameters every call and returned, so the clip+noise step can reuse
        the same live groups without touching any state cached on self.

        Fase 0 guards (defense-in-depth against a regression reintroducing
        Bug 1's pattern — see docs/validated_environment.md). Identity/
        fingerprint verification of "did the model actually reload this
        round" is intentionally NOT done here: this class is designed to be
        importable and unit-testable without CUDA/model weights (see module
        docstring), and legitimate unit tests reuse the same fake model
        across simulated rounds to isolate the grouping/threshold math from
        reload behavior. That check instead lives where the reload actually
        happens — see AdaptiveClippingClient.fit() in client.py, which
        compares each round's model parameter identities against the
        previous round's right after self._load_model().

        1. Grad presence: a group where NONE of its parameters carry a
           gradient is the direct, observable symptom of Bug 1 — a stale
           nn.Parameter belongs to a model already replaced by
           _load_model(), so .grad is permanently None and clip+noise
           silently stopped applying to that group.
        2. History-length integrity: this method is called once per optimizer
           step (training.py calls it inside the accumulation loop, so a
           single round may call it many times when DP forces batch=1/
           accum=1). Rather than asserting against round_idx — a
           caller-supplied number whose convention varies (production passes
           Flower's 1-indexed server_round; some tests use 0-indexed round
           numbers) — this tracks its OWN call count and asserts every group
           gains exactly one history entry per call: no more, no fewer. A
           mismatch means a call silently appended zero or multiple times,
           and compute_thresholds() can no longer be trusted.
        """
        self._current_round = round_idx
        self._n_updates += 1
        params_list = [(n, p) for n, p in named_parameters if p.requires_grad]

        groups = self.group_by_attention_layer(params_list)
        for name, params in groups.items():
            grads = [p.grad.detach() for p in params if p.grad is not None]
            if grads and len(grads) < len(params):
                # Partial: some but not all params in the group lack a
                # gradient. Plausible for a genuinely unused sub-branch, but
                # worth surfacing loudly since it is also consistent with a
                # PARTIALLY stale reference set.
                raise RuntimeError(
                    f"PerLayerClipper.update_history(round={round_idx}): group '{name}' "
                    f"has {len(params)} trainable parameter(s) but only {len(grads)} "
                    f"carry a gradient. Expected either all-or-nothing (a group with a "
                    f"parameter that never reaches the forward graph would have NONE)."
                )
            if not grads:
                if not params:
                    continue
                raise RuntimeError(
                    f"PerLayerClipper.update_history(round={round_idx}): group '{name}' "
                    f"has {len(params)} trainable parameter(s) but NONE have a .grad — "
                    f"backward() did not populate gradients for this group this round. "
                    f"This is the direct symptom of the historical stale-reference bug "
                    f"(Bug 1): a cached nn.Parameter belonging to an already-replaced "
                    f"model has .grad permanently None."
                )
            group_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2).item()
            self._history[name].append(float(group_norm))
            if len(self._history[name]) != self._n_updates:
                raise RuntimeError(
                    f"PerLayerClipper history length mismatch for group '{name}': "
                    f"expected {self._n_updates} accumulated value(s) after "
                    f"{self._n_updates} update_history() call(s), got "
                    f"{len(self._history[name])}. A call silently appended zero or "
                    f"multiple times for this group — compute_thresholds() can no "
                    f"longer be trusted."
                )
        return groups

    def compute_thresholds(self) -> dict[str, float]:
        """
        Returns dict mapping layer_name -> C0 for current round.
        During warmup_rounds: returns global C0=1.0 for all layers.
        After warmup: returns percentile of historical norms per layer,
        clamped to [min_clip, max_clip].
        """
        if self._current_round < self.warmup_rounds:
            return {name: self.global_c0 for name in self._history}

        thresholds: dict[str, float] = {}
        for name, values in self._history.items():
            if not values:
                thresholds[name] = self.global_c0
                continue
            p = float(np.percentile(values, self.percentile))
            thresholds[name] = float(np.clip(p, self.min_clip, self.max_clip))
        return thresholds

    def get_history_summary(self) -> dict[str, dict]:
        """
        Returns summary statistics of gradient norm history per layer.
        Used for logging and paper figures.
        Keys: layer_name -> {mean, std, min, max, history: list[float]}
        """
        summary: dict[str, dict] = {}
        for name, values in self._history.items():
            if values:
                summary[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "history": list(values),
                }
            else:
                summary[name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "history": [],
                }
        return summary
