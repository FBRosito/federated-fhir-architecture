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
        """
        self._current_round = round_idx
        groups = self.group_by_attention_layer(
            [(n, p) for n, p in named_parameters if p.requires_grad]
        )
        for name, params in groups.items():
            grads = [p.grad.detach() for p in params if p.grad is not None]
            if not grads:
                continue
            group_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2).item()
            self._history[name].append(float(group_norm))
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
                summary[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "history": []}
        return summary
