"""
grad_norm_logger.py
--------------------
Fase 0 instrumentation: per-round JSONL logging of gradient norms, clip
thresholds, effective learning rate, and cumulative epsilon. This is
infrastructure for the Phase 1-2 mechanistic analysis (reproducing Article
2's Table 5 without ad hoc instrumentation) — it does not influence
clipping, noise, or aggregation in any way. Disabling it changes no
training behavior.

Toggle: LOG_GRAD_NORMS (default "1" — on by default in DP experiments per
the Fase 0 spec). Set LOG_GRAD_NORMS=0 to disable.

Grouping matches adaptive_clipping.clipping.PerLayerClipper.group_by_attention_layer
(encoder.layer.N / other), duplicated here rather than imported so that
ai_client and other callers of this module do not take a dependency on the
adaptive-clipping experiment package. LoRA A/B are split within a group when
both are present, and the classifier head (which falls into "other") is
distinguishable from any LoRA A/B also grouped there.

Usage (see ai_client.model_setup_bert.train_bert_one_round and
adaptive_clipping.training for the producer side, which only adds fields to
the existing `metrics` dict — the same pattern already used for dp_epsilon
etc. — and ai_client.fl_client.FHIRFederatedClient.fit / the two experiment
clients for the consumer side, which call log_round() once cumulative
epsilon is known):

    from evaluation.grad_norm_logger import GradNormLogger, is_enabled, snapshot_group_norms

    if is_enabled():
        pre_clip = snapshot_group_norms(model.named_parameters(), attr="grad")
        ... clip + noise ...
        post_clip = snapshot_group_norms(model.named_parameters(), attr="grad")
        metrics["grad_norms_pre_clip"] = json.dumps(pre_clip)
        metrics["grad_norms_post_clip_noise"] = json.dumps(post_clip)

    # later, in fit(), once epsilon_cumulative is known:
    GradNormLogger(run_id=...).log_round(
        server_round=..., partition_id=..., grad_norms_pre_clip=...,
        grad_norms_post_clip_noise=..., clip_thresholds=..., learning_rate=...,
        epsilon_cumulative=..., noise_multiplier=...,
    )
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import torch

_LAYER_PATTERNS: list[tuple["re.Pattern[str]", str]] = [
    (re.compile(r"encoder\.layer\.(\d+)"), "encoder.layer.{}"),
    (re.compile(r"\blayers\.(\d+)"), "layers.{}"),
    (re.compile(r"\bh\.(\d+)"), "h.{}"),
]


def is_enabled() -> bool:
    """LOG_GRAD_NORMS default is ON ("1") — the Fase 0 spec asks for this
    instrumentation to default to enabled in DP experiments; set
    LOG_GRAD_NORMS=0/false/no/off to disable."""
    return os.environ.get("LOG_GRAD_NORMS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def group_by_layer(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, list[torch.nn.Parameter]]:
    """Groups by transformer attention layer (encoder.layer.N / layers.N /
    h.N, falling back to "other"), further split by .lora_A / .lora_B when
    the parameter name contains either — so a group with both a LoRA A and
    LoRA B projection reports two separate norms, per the Fase 0 spec."""
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, param in named_parameters:
        group_key = "other"
        for pattern, template in _LAYER_PATTERNS:
            match = pattern.search(name)
            if match:
                group_key = template.format(match.group(1))
                break
        lname = name.lower()
        if "lora_a" in lname:
            group_key = f"{group_key}.lora_A"
        elif "lora_b" in lname:
            group_key = f"{group_key}.lora_B"
        groups.setdefault(group_key, []).append(param)
    return groups


def snapshot_group_norms(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    attr: str = "grad",
) -> dict[str, float]:
    """L2 norm per group. attr="grad" (default) reads the current gradient —
    call once right before clip_grad_norm_ for the pre-clip snapshot, and
    once more right before optimizer.step() (after clip + DP noise, when
    active) for the post-clip/noise snapshot. Groups with no tensor for
    every one of their parameters (e.g. attr="grad" before any backward())
    are omitted rather than reported as zero, so a missing group is visibly
    different from a genuinely near-zero gradient.
    """
    groups = group_by_layer(named_parameters)
    norms: dict[str, float] = {}
    for name, params in groups.items():
        tensors = [
            getattr(p, attr) for p in params if getattr(p, attr, None) is not None
        ]
        if not tensors:
            continue
        norms[name] = float(
            torch.norm(torch.stack([t.detach().norm(2) for t in tensors]), 2).item()
        )
    return norms


class GradNormLogger:
    """Appends one JSON line per round to
    experiments/logs/grad_norms_<run_id>.jsonl. A new instance may be
    created per round (fit() is called fresh every round, same as the
    model) — all instances sharing a run_id append to the same file.
    """

    def __init__(
        self,
        run_id: str | None = None,
        logs_dir: str | Path | None = None,
    ) -> None:
        self.enabled = is_enabled()
        if not self.enabled:
            return
        run_id = (
            run_id
            or os.environ.get("ADAPTIVE_EXPERIMENT_TAG")
            or os.environ.get("FL_EXPERIMENT_TAG")
            or f"run_{int(time.time())}"
        )
        # Sanitize: run_id ends up in a filename.
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
        if logs_dir is not None:
            base = Path(logs_dir)
        elif os.environ.get("GRAD_NORM_LOGS_DIR"):
            base = Path(os.environ["GRAD_NORM_LOGS_DIR"])
        else:
            # Anchor on this module's own location rather than the current
            # working directory: client processes for different experiment
            # packages run with different cwds (experiments/article3,
            # experiments/adaptive-clipping, ...), so a relative
            # "experiments/logs" would land in a different, wrong place
            # depending on which experiment launched it. This file always
            # lives at <repo_root>/evaluation/src/evaluation/, so parents[3]
            # is <repo_root> regardless of the caller's cwd.
            repo_root = Path(__file__).resolve().parents[3]
            base = repo_root / "experiments" / "logs"
        self.path = base / f"grad_norms_{safe_run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_round(
        self,
        *,
        server_round: int,
        partition_id: int | None,
        grad_norms_pre_clip: dict[str, float],
        grad_norms_post_clip_noise: dict[str, float],
        clip_thresholds: dict[str, float] | float,
        learning_rate: float,
        epsilon_cumulative: float | None,
        noise_multiplier: float,
        extra: dict | None = None,
    ) -> None:
        """Writes one JSONL record. No-op if LOG_GRAD_NORMS disabled."""
        if not self.enabled:
            return
        record = {
            "server_round": server_round,
            "partition_id": partition_id,
            "grad_norms_pre_clip": grad_norms_pre_clip,
            "grad_norms_post_clip_noise": grad_norms_post_clip_noise,
            "clip_thresholds": clip_thresholds,
            "learning_rate": learning_rate,
            "epsilon_cumulative": epsilon_cumulative,
            "noise_multiplier": noise_multiplier,
            "wall_clock": time.time(),
        }
        if extra:
            record.update(extra)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
