"""
model_setup_bert.py
-------------------
PubMedBERT backend for multi-label ICD-10 coding (Experiment A).

Architecture follows PLM-ICD (Huang et al., NAACL 2022 Findings):
  - Encoder: PubMedBERT (Gu et al., ACM Trans. Healthcare 2021)
  - Per-label attention: one attention vector per ICD-10 code
  - Classification head: sigmoid per code → BCEWithLogitsLoss

Only the LoRA adapters (r=8 on q/v of the encoder) are exchanged in
federation — the PubMedBERT base weights remain frozen across all silos.

Base model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
  - 110M parameters, fp32/bf16, ~440 MB VRAM → multiple silos in parallel.
  - No 4-bit quantisation (small model; fits comfortably in 12 GB VRAM).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

log = logging.getLogger(__name__)

PUBMEDBERT_MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DEFAULT_MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))

# BERT encoder modules that will receive LoRA adapters.
# q/v are sufficient for adaptation without modifying full attention layers.
BERT_LORA_TARGET_MODULES = ["query", "value"]


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class BertLoRAConfig:
    r:            int   = 8
    lora_alpha:   int   = 16
    lora_dropout: float = 0.05
    bias:         str   = "none"
    target_modules: list[str] = field(default_factory=lambda: BERT_LORA_TARGET_MODULES)


@dataclass
class BertTrainingConfig:
    learning_rate:        float = 2e-4
    weight_decay:         float = 0.01
    num_epochs:           int   = 1
    batch_size:           int   = 8
    gradient_accum_steps: int   = 8
    max_grad_norm:        float = 1.0
    warmup_ratio:         float = 0.1
    use_amp:              bool  = True
    proximal_mu:          float = 0.0
    noise_multiplier:     float = 0.0
    target_delta:         float = 1e-5
    dp_subsample_rate:    float = 0.1


# ── Model: PubMedBERT + per-label attention + sigmoid ─────────────────────────

class PLMICDModel(nn.Module):
    """
    PubMedBERT encoder with per-label attention and sigmoid head,
    following the PLM-ICD architecture (Huang et al., 2022).

    For each ICD-10 code c, the attention is:
        α_c = softmax(U_c · H^T)
        v_c = α_c · H
        logit_c = W_c · v_c + b_c

    where H ∈ R^{L×d} are BERT hidden states and U_c ∈ R^d is the
    trainable attention vector for code c.
    """

    def __init__(
        self,
        encoder: PreTrainedModel,
        num_labels: int,
        hidden_size: int = 768,
    ) -> None:
        super().__init__()
        self.encoder    = encoder
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        # Per-label attention vectors: U ∈ R^{num_labels × hidden_size}
        self.label_attention = nn.Linear(hidden_size, num_labels, bias=False)
        # Per-label classification: W ∈ R^{num_labels × hidden_size}, b ∈ R^{num_labels}
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # H: [batch, seq_len, hidden_size]
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Attn scores: [batch, seq_len, num_labels]
        attn_scores = self.label_attention(H)

        # Mask padding tokens before softmax
        pad_mask = (attention_mask == 0).unsqueeze(-1)  # [batch, seq_len, 1]
        attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))

        # α: [batch, num_labels, seq_len]
        alpha = torch.softmax(attn_scores.transpose(1, 2), dim=-1)

        # Per-label representation: v ∈ [batch, num_labels, hidden_size]
        v = torch.bmm(alpha, H)

        # Logits: [batch, num_labels] via linear classifier
        logits = (v * self.classifier.weight.unsqueeze(0)).sum(-1) + self.classifier.bias

        output: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # BCEWithLogitsLoss — numerically stable; sigmoid not required beforehand
            loss_fn = nn.BCEWithLogitsLoss()
            output["loss"] = loss_fn(logits, labels.float())

        return output


# ── Model loading ──────────────────────────────────────────────────────────────

def load_bert_model(
    model_name: str = PUBMEDBERT_MODEL_ID,
    num_labels: int = 50,
    lora_cfg: BertLoRAConfig | None = None,
) -> tuple[PLMICDModel, PreTrainedTokenizerBase]:
    """
    Loads PubMedBERT, applies LoRA (r=8), and builds the PLMICDModel.

    Args:
        model_name:  HuggingFace ID of the BERT encoder.
        num_labels:  Number of ICD-10 labels (50 for MIMIC-IV-50, ~3k for full).
        lora_cfg:    LoRA configuration (uses BertLoRAConfig() if None).

    Returns:
        (model, tokenizer)
    """
    if lora_cfg is None:
        lora_cfg = BertLoRAConfig()

    hf_token = os.getenv("HF_TOKEN")

    log.info("Loading PubMedBERT tokeniser: %s", model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    except Exception as _e:
        log.warning("HF Hub unavailable (%s). Loading tokeniser from local cache.", _e)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=hf_token, use_fast=True, local_files_only=True,
        )

    log.info("Loading PubMedBERT encoder: %s", model_name)
    try:
        encoder = AutoModel.from_pretrained(
            model_name, token=hf_token, torch_dtype=torch.float32,
        )
    except Exception as _e:
        log.warning("HF Hub unavailable (%s). Loading encoder from local cache.", _e)
        encoder = AutoModel.from_pretrained(
            model_name, token=hf_token, torch_dtype=torch.float32, local_files_only=True,
        )

    peft_config = LoraConfig(
        task_type      = TaskType.FEATURE_EXTRACTION,
        r              = lora_cfg.r,
        lora_alpha     = lora_cfg.lora_alpha,
        lora_dropout   = lora_cfg.lora_dropout,
        bias           = lora_cfg.bias,
        target_modules = lora_cfg.target_modules,
    )
    encoder = get_peft_model(encoder, peft_config)
    encoder.print_trainable_parameters()

    hidden_size = encoder.config.hidden_size
    model = PLMICDModel(encoder=encoder, num_labels=num_labels, hidden_size=hidden_size)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(
        "PLMICDModel: %d labels | trainable=%d (%.3f%% of total=%d)",
        num_labels, trainable, 100 * trainable / max(total, 1), total,
    )
    return model, tokenizer


# ── Dataset ────────────────────────────────────────────────────────────────────

class ICD10MultiLabelDataset(Dataset):
    """
    Dataset for multi-label ICD-10 coding.

    Each example is tokenised; the label is a binary vector of length
    `num_labels` with 1 at the indices of codes present in the admission.
    """

    def __init__(
        self,
        texts: list[str],
        label_vectors: list[list[int]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        assert len(texts) == len(label_vectors)
        self.items: list[dict[str, torch.Tensor]] = []

        for text, lv in zip(texts, label_vectors):
            enc = tokenizer(
                text,
                max_length     = max_length,
                truncation     = True,
                padding        = "max_length",
                return_tensors = "pt",
            )
            self.items.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(lv, dtype=torch.float32),
            })

        log.info("ICD10MultiLabelDataset: %d examples, max_length=%d", len(self.items), max_length)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def build_bert_dataset(
    examples: list[Any],   # list[TrainingExample] — lazy import to avoid circular dep
    label_index: dict[str, int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
    num_labels: int = 0,
) -> ICD10MultiLabelDataset:
    """
    Builds an ICD10MultiLabelDataset from TrainingExamples from fhir_consumer.

    Args:
        examples:    List of TrainingExample with `clinical_text` and `all_icd10_codes`.
        label_index: ICD-10 code → label vector index mapping.
        tokenizer:   PubMedBERT tokeniser.
        max_length:  Maximum tokenisation length.
        num_labels:  Label vector size. If 0, uses len(label_index).
                     Must equal the model's num_labels for shape compatibility.
    """
    num_labels = num_labels if num_labels > 0 else len(label_index)
    max_length = min(max_length, 512)  # PubMedBERT/BERT hardcap: max_position_embeddings=512
    texts: list[str] = []
    label_vectors: list[list[int]] = []

    for ex in examples:
        texts.append(ex.clinical_text or ex.to_prompt())
        lv = [0] * num_labels
        codes = ex.all_icd10_codes if ex.all_icd10_codes else [ex.icd10_code]
        for code in codes:
            if code in label_index and label_index[code] < num_labels:
                lv[label_index[code]] = 1
        label_vectors.append(lv)

    return ICD10MultiLabelDataset(texts, label_vectors, tokenizer, max_length)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_bert_one_round(
    model: PLMICDModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Any],
    label_index: dict[str, int],
    train_cfg: BertTrainingConfig | None = None,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
) -> tuple[list[np.ndarray], int, dict[str, float]]:
    """
    Executes one federated training round for the PubMedBERT backend.

    Returns only the LoRA weights of the encoder + attention/classifier layers
    for aggregation by the Flower server.

    Returns:
        (bert_parameters, num_examples, metrics)
    """
    if not examples:
        log.warning("No examples for BERT training — round skipped.")
        return get_bert_parameters(model), 0, {"train_loss": 0.0}

    if train_cfg is None:
        train_cfg = BertTrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # FedProx: capture global reference before any update
    proximal_mu = train_cfg.proximal_mu
    if proximal_mu > 0.0:
        _global_ref = [
            p.detach().float().clone()
            for p in model.parameters()
            if p.requires_grad
        ]
        log.info("FedProx BERT active: μ=%.4f | %d trainable tensors.", proximal_mu, len(_global_ref))
    else:
        _global_ref = None

    dp_active = train_cfg.noise_multiplier > 0.0
    # DP requires batch_size=1 and accum_steps=1 so each clip_grad_norm_ call sees
    # exactly one sample's gradient — making batch-level clipping equivalent to
    # per-sample clipping. This is the same constraint applied in model_setup.py (LLM).
    effective_batch_size  = 1 if dp_active else train_cfg.batch_size
    effective_accum_steps = 1 if dp_active else train_cfg.gradient_accum_steps

    if dp_active:
        log.info(
            "BERT DP-SGD active: σ=%.2f | C=%.4f | δ=%s | batch=1 | accum=1 (per-sample clipping)",
            train_cfg.noise_multiplier, train_cfg.max_grad_norm, train_cfg.target_delta,
        )
        try:
            from opacus.accountants import RDPAccountant
            dp_accountant = RDPAccountant()
        except ImportError:
            log.warning("opacus not installed — ε accounting disabled for BERT.")
            dp_accountant = None
    else:
        dp_accountant = None

    dataset = build_bert_dataset(examples, label_index, tokenizer, max_length,
                                num_labels=model.num_labels)
    n_total = max(len(dataset), 1)
    if dp_active:
        # WOR subsampling: draw k = q*n fresh indices each call (no fixed seed)
        # so RDP amplification-by-subsampling holds. dp_sample_rate reflects actual q.
        q = train_cfg.dp_subsample_rate
        k = max(1, int(n_total * q))
        dp_sample_rate = k / n_total
        rng_sub = np.random.default_rng()  # OS entropy — fresh subset every round
        subset_indices = rng_sub.choice(n_total, size=k, replace=False).tolist()
        sampler = SubsetRandomSampler(subset_indices)
        dataloader = DataLoader(
            dataset,
            batch_size = effective_batch_size,
            sampler    = sampler,
            drop_last  = False,
            pin_memory = torch.cuda.is_available(),
        )
        log.info("BERT DP subsampling: k=%d/%d samples (q=%.4f)", k, n_total, dp_sample_rate)
    else:
        dp_sample_rate = 1.0
        dataloader = DataLoader(
            dataset,
            batch_size = effective_batch_size,
            shuffle    = True,
            drop_last  = False,
            pin_memory = torch.cuda.is_available(),
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = train_cfg.learning_rate,
        weight_decay = train_cfg.weight_decay,
        betas        = (0.9, 0.95),
        eps          = 1e-8,
    )
    total_steps  = max(1, len(dataloader) * train_cfg.num_epochs // effective_accum_steps)
    warmup_steps = max(1, int(total_steps * train_cfg.warmup_ratio))
    scheduler    = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps),
        eta_min=train_cfg.learning_rate * 0.1,
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler    = torch.amp.GradScaler("cuda", enabled=train_cfg.use_amp and amp_dtype == torch.float16)

    model.train()
    cumulative_loss = 0.0
    global_step     = 0
    optimizer.zero_grad()

    for epoch in range(train_cfg.num_epochs):
        epoch_loss    = 0.0
        valid_batches = 0
        accum_count   = 0

        for step, batch in enumerate(dataloader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=train_cfg.use_amp):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ce_loss = out["loss"]
                if proximal_mu > 0.0 and _global_ref is not None:
                    prox_term = sum(
                        (p.float() - ref).pow(2).sum()
                        for p, ref in zip(
                            (q for q in model.parameters() if q.requires_grad),
                            _global_ref,
                        )
                    )
                    loss_scaled = (ce_loss + (proximal_mu / 2) * prox_term) / effective_accum_steps
                else:
                    loss_scaled = ce_loss / effective_accum_steps

            raw_loss = ce_loss.item()
            if not math.isfinite(raw_loss):
                log.warning("Non-finite loss (%.4g) at step %d — batch skipped.", raw_loss, step)
                continue

            scaler.scale(loss_scaled).backward()
            epoch_loss    += raw_loss
            valid_batches += 1
            accum_count   += 1

            if accum_count % effective_accum_steps == 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    train_cfg.max_grad_norm,
                )
                if dp_active and math.isfinite(float(total_norm)):
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.requires_grad and p.grad is not None:
                                noise = torch.normal(
                                    mean=0.0,
                                    std=train_cfg.max_grad_norm * train_cfg.noise_multiplier,
                                    size=p.grad.shape,
                                    device=p.grad.device,
                                    dtype=p.grad.dtype,
                                )
                                p.grad.add_(noise)
                    if dp_accountant is not None:
                        dp_accountant.step(
                            noise_multiplier=train_cfg.noise_multiplier,
                            sample_rate=dp_sample_rate,
                        )
                if math.isfinite(float(total_norm)):
                    scaler.step(optimizer)
                else:
                    log.warning("Non-finite grad norm at step %d — skipped.", global_step)
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0
                if global_step >= warmup_steps:
                    scheduler.step()
                global_step += 1
                log.info(
                    "BERT Epoch %d/%d | step %d | loss=%.4f | lr=%.2e",
                    epoch + 1, train_cfg.num_epochs, global_step,
                    epoch_loss / max(valid_batches, 1),
                    optimizer.param_groups[0]["lr"],
                )

        if accum_count > 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), train_cfg.max_grad_norm
            )
            if dp_active and math.isfinite(float(total_norm)):
                with torch.no_grad():
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            noise = torch.normal(
                                mean=0.0,
                                std=train_cfg.max_grad_norm * train_cfg.noise_multiplier,
                                size=p.grad.shape,
                                device=p.grad.device,
                                dtype=p.grad.dtype,
                            )
                            p.grad.add_(noise)
                if dp_accountant is not None:
                    dp_accountant.step(
                        noise_multiplier=train_cfg.noise_multiplier,
                        sample_rate=dp_sample_rate,
                    )
            if math.isfinite(float(total_norm)):
                scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        avg_epoch_loss = epoch_loss / max(valid_batches, 1)
        log.info("BERT Epoch %d/%d completed — loss=%.4f", epoch + 1, train_cfg.num_epochs, avg_epoch_loss)
        cumulative_loss += avg_epoch_loss

    avg_loss = cumulative_loss / max(train_cfg.num_epochs, 1)
    metrics  = {"train_loss": round(avg_loss, 6)}

    if dp_active and dp_accountant is not None:
        try:
            epsilon = dp_accountant.get_epsilon(delta=train_cfg.target_delta)
            metrics["dp_epsilon"]          = round(float(epsilon), 4)
            metrics["dp_delta"]            = train_cfg.target_delta
            metrics["dp_noise_multiplier"] = train_cfg.noise_multiplier
            metrics["dp_sample_rate"]      = dp_sample_rate
            metrics["dp_steps_this_round"] = global_step
            log.info(
                "BERT DP: ε=%.4f | δ=%s | σ=%.2f | steps=%d",
                epsilon, train_cfg.target_delta, train_cfg.noise_multiplier, global_step,
            )
        except Exception as exc:
            log.warning("Error computing BERT DP ε: %s", exc)

    log.info("BERT round completed — loss=%.4f", avg_loss)
    return get_bert_parameters(model), len(examples), metrics


# ── Flower parameter utilities ─────────────────────────────────────────────────

def get_bert_parameters(model: PLMICDModel) -> list[np.ndarray]:
    """
    Extracts trainable weights from PLMICDModel as a list of NumPy arrays.

    Includes: encoder LoRA adapters + label_attention + classifier.
    """
    lora_state = get_peft_model_state_dict(model.encoder)
    lora_params = [v.detach().float().cpu().numpy() for v in lora_state.values()]
    head_params = [
        p.detach().float().cpu().numpy()
        for p in [*model.label_attention.parameters(), *model.classifier.parameters()]
    ]
    return lora_params + head_params


def set_bert_parameters(model: PLMICDModel, parameters: list[np.ndarray]) -> None:
    """Applies aggregated weights (LoRA + head) to PLMICDModel."""
    lora_state = get_peft_model_state_dict(model.encoder)
    n_lora = len(lora_state)

    new_lora_state = {
        k: torch.tensor(v, dtype=lora_state[k].dtype)
        for k, v in zip(lora_state.keys(), parameters[:n_lora])
    }
    set_peft_model_state_dict(model.encoder, new_lora_state)

    head_tensors = parameters[n_lora:]
    head_params  = [*model.label_attention.parameters(), *model.classifier.parameters()]
    if len(head_params) != len(head_tensors):
        raise ValueError(
            f"Head shape mismatch: model has {len(head_params)} tensors, "
            f"received {len(head_tensors)}."
        )
    with torch.no_grad():
        for param, arr in zip(head_params, head_tensors):
            param.copy_(torch.tensor(arr, dtype=param.dtype))

    log.info(
        "BERT weights updated: %d LoRA tensors + %d head tensors.",
        n_lora, len(head_params),
    )
