"""
dual_training.py
-----------------
Dual-LoRA (HERALD-PFL) training and evaluation for the Article 3 experiment.

Extends ai_client's production BERT training function rather than
reimplementing it: dual_train_bert_one_round() calls
ai_client.model_setup_bert.train_bert_one_round() TWICE per round —

    1. adapter="default" (global, r=8), DP-SGD as configured — identical to
       the dp_lora/ffa_lora strategies' training path.
    2. adapter="local" (r=4), plain SGD, no DP, on the same silo's data —
       with the classifier head temporarily frozen so this second pass
       cannot leak a non-private gradient into the head parameters that
       get_bert_parameters() transmits to the server every round regardless
       of which adapter is active (get_peft_model_state_dict/
       set_peft_model_state_dict default to adapter_name="default", so the
       LoRA weights are already isolated by adapter name — but the
       classifier head is a plain nn.Linear outside any adapter and is
       therefore NOT isolated by that mechanism, hence the explicit freeze).

Why the local adapter needs an explicit on-disk checkpoint: ai_client's
FHIRFederatedClient reloads the whole model from scratch inside every single
fit() call (FL_KEEP_MODEL_IN_VRAM defaults to false, and no article3 script
sets it) — see model_setup_bert.load_bert_model(). Without persistence, the
"local" adapter would be randomly re-initialized every round and never
accumulate any silo-specific signal, defeating the entire premise of a
personalization component. local_adapter_path() below gives each silo a
fixed on-disk location; dual_train_bert_one_round() loads it (if present) at
the start of the round and overwrites it at the end.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from ai_client.model_setup_bert import (
    BertTrainingConfig,
    PLMICDModel,
    train_bert_one_round,
)
from evaluation.icd_metrics import ICD10Metrics, compute_icd_metrics

log = logging.getLogger(__name__)


def local_adapter_path(logs_root: Path, experiment_tag: str, silo_id: int) -> Path:
    """Path where a silo's frozen local LoRA adapter checkpoint is stored."""
    return Path(logs_root) / experiment_tag / str(silo_id) / "local_adapter.pt"


def _load_local_adapter(model: PLMICDModel, ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return False
    device = next(model.parameters()).device
    local_state = torch.load(ckpt_path, map_location=device)
    set_peft_model_state_dict(model.encoder, local_state, adapter_name="local")
    return True


def _save_local_adapter(model: PLMICDModel, ckpt_path: Path) -> None:
    local_state = get_peft_model_state_dict(model.encoder, adapter_name="local")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(local_state, ckpt_path)


def dual_train_bert_one_round(
    model: PLMICDModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Any],
    label_index: dict[str, int],
    train_cfg: BertTrainingConfig,
    local_ckpt_path: Path,
    max_length: int,
) -> tuple[list[np.ndarray], int, dict[str, float]]:
    """Global DP-SGD pass (adapter="default") + local no-DP pass (adapter="local"),
    with the local-adapter checkpoint loaded before and saved after. Returns
    the same (parameters, n_examples, metrics) shape as train_bert_one_round —
    parameters come from the global pass only (get_bert_parameters() already
    excludes the local adapter by name, see module docstring)."""
    loaded = _load_local_adapter(model, local_ckpt_path)
    log.info(
        "Dual LoRA: local adapter %s from %s",
        "loaded" if loaded else "starting fresh (no checkpoint yet)",
        local_ckpt_path,
    )

    # 1. Global step — default adapter, DP-SGD as configured.
    model.encoder.set_adapter("default")
    updated_params, n_examples, metrics = train_bert_one_round(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        label_index=label_index,
        train_cfg=train_cfg,
        max_length=max_length,
    )

    # 2. Local step — local adapter, no DP, head frozen so this pass cannot
    #    leak a non-private gradient into the (always-transmitted) head.
    model.encoder.set_adapter("local")
    head_params = [*model.label_attention.parameters(), *model.classifier.parameters()]
    prev_requires_grad = [p.requires_grad for p in head_params]
    for p in head_params:
        p.requires_grad = False

    local_cfg = replace(train_cfg, noise_multiplier=0.0, proximal_mu=0.0)
    train_bert_one_round(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        label_index=label_index,
        train_cfg=local_cfg,
        max_length=max_length,
    )  # side effect only — local adapter's weights update in place

    for p, rg in zip(head_params, prev_requires_grad):
        p.requires_grad = rg

    _save_local_adapter(model, local_ckpt_path)

    # 3. Restore default active — hygiene for the next evaluate()/fit() call.
    model.encoder.set_adapter("default")

    metrics["lora_mode"] = "dual"
    return updated_params, n_examples, metrics


def evaluate_bert_model_fused(
    model: PLMICDModel,
    dataloader: DataLoader,
    device: torch.device,
    k_list: list[int] | None = None,
) -> tuple[ICD10Metrics, ICD10Metrics, ICD10Metrics]:
    """Runs two forward-only passes (adapter="default", adapter="local") over
    the SAME dataloader (must not shuffle, so batches line up example-for-
    example across passes) and returns (fused, global_only, local_only)
    ICD10Metrics — fused = 0.5*logits_default + 0.5*logits_local, matching
    HERALD-PFL's late-fusion inference."""
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _run_pass(adapter_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        model.encoder.set_adapter(adapter_name)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with torch.amp.autocast(
                    "cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()
                ):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                all_logits.append(out["logits"].float().cpu())
                all_labels.append(labels.float().cpu())
        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

    logits_global, labels_t = _run_pass("default")
    logits_local, _ = _run_pass("local")
    model.encoder.set_adapter("default")  # restore

    logits_fused = 0.5 * logits_global + 0.5 * logits_local
    labels_np = labels_t.numpy()

    loss_fn = torch.nn.BCEWithLogitsLoss()

    def _to_metrics(logits: torch.Tensor) -> ICD10Metrics:
        avg_loss = loss_fn(logits, labels_t).item()
        scores_np = 1.0 / (1.0 + np.exp(-logits.numpy()))
        m = compute_icd_metrics(labels_np, scores_np, k_list=k_list)
        m.avg_loss = round(avg_loss, 6)
        return m

    fused_metrics = _to_metrics(logits_fused)
    global_metrics = _to_metrics(logits_global)
    local_metrics = _to_metrics(logits_local)
    log.info(
        "Dual LoRA eval: fused F1=%.4f | global-only F1=%.4f | local-only F1=%.4f",
        fused_metrics.micro_f1,
        global_metrics.micro_f1,
        local_metrics.micro_f1,
    )
    return fused_metrics, global_metrics, local_metrics
