"""
clipping.py
-----------
Tracks per-attention-layer gradient-norm history and computes adaptive
clipping thresholds. Contains NO Opacus/PrivacyEngine integration and
performs no clipping itself — training.py owns clip+noise application.
This separation keeps clipping.py importable and unit-testable without
CUDA/model weights (only torch, numpy, re, dataclasses).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

# Grouping key: an attention layer is a functional unit of the transformer,
# not an individual LoRA projection — see README.md for the full rationale.
_LAYER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"encoder\.layer\.(\d+)"), "encoder.layer.{}"),
    (re.compile(r"\blayers\.(\d+)"), "layers.{}"),
    (re.compile(r"\bh\.(\d+)"), "h.{}"),
]


@dataclass
class _LayerNormHistory:
    values: list[float] = field(default_factory=list)


class PerLayerClipper:
    """
    Tracks per-layer gradient norms across FL rounds and computes
    adaptive clipping thresholds calibrated to each LoRA layer's
    historical gradient distribution.
    """

    def __init__(
        self,
        named_parameters: list[tuple[str, torch.nn.Parameter]],
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

        self._groups: dict[str, list[torch.nn.Parameter]] = self.group_by_attention_layer(
            named_parameters
        )
        self._history: dict[str, _LayerNormHistory] = {
            name: _LayerNormHistory() for name in self._groups
        }
        self._current_round = 0

    @staticmethod
    def group_by_attention_layer(
        named_parameters: list[tuple[str, torch.nn.Parameter]],
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

    @property
    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        return self._groups

    def update_history(self, round_idx: int) -> None:
        """Called after backward(), before any clipping, once per optimizer step."""
        self._current_round = round_idx
        for name, params in self._groups.items():
            grads = [p.grad.detach() for p in params if p.grad is not None]
            if not grads:
                continue
            group_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2).item()
            self._history[name].values.append(float(group_norm))

    def compute_thresholds(self) -> dict[str, float]:
        """
        Returns dict mapping layer_name -> C0 for current round.
        During warmup_rounds: returns global C0=1.0 for all layers.
        After warmup: returns percentile of historical norms per layer,
        clamped to [min_clip, max_clip].
        """
        if self._current_round < self.warmup_rounds:
            return {name: self.global_c0 for name in self._groups}

        thresholds: dict[str, float] = {}
        for name, hist in self._history.items():
            if not hist.values:
                thresholds[name] = self.global_c0
                continue
            p = float(np.percentile(hist.values, self.percentile))
            thresholds[name] = float(np.clip(p, self.min_clip, self.max_clip))
        return thresholds

    def get_history_summary(self) -> dict[str, dict]:
        """
        Returns summary statistics of gradient norm history per layer.
        Used for logging and paper figures.
        Keys: layer_name -> {mean, std, min, max, history: list[float]}
        """
        summary: dict[str, dict] = {}
        for name, hist in self._history.items():
            if hist.values:
                summary[name] = {
                    "mean": float(np.mean(hist.values)),
                    "std": float(np.std(hist.values)),
                    "min": float(np.min(hist.values)),
                    "max": float(np.max(hist.values)),
                    "history": list(hist.values),
                }
            else:
                summary[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "history": []}
        return summary
