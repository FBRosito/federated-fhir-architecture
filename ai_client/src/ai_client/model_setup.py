"""
model_setup.py
--------------
Configures an open-source LLM (Llama-3.2 by default) with optional NF4 4-bit
quantisation via BitsAndBytesConfig, applies LoRA adapters (PEFT) on top of
frozen base weights for the ICD-10 code extraction task, and exposes a
standard PyTorch training function compatible with the Flower federated loop.

Flow:
    1. load_quantized_model()   → base model in NF4 / fp16 / bf16 (controlled by
                                  MODEL_BASE_PRECISION, default nf4)
    2. apply_lora()             → injects trainable LoRA adapters
    3. build_dataset()          → tokenises examples from fhir_consumer
    4. train_one_round()        → PyTorch loop + returns LoRA weights for Flower

Hardware requirements:
    - NVIDIA GPU with CUDA ≥ 12.4 and ≥ 7 GB VRAM (target: 6-7 GB with NF4 + LoRA).
    - bitsandbytes ≥ 0.43 installed in the CUDA environment.

Environment variables:
    MODEL_NAME           HuggingFace model ID (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    HF_TOKEN             HuggingFace access token (required for gated models)
    MAX_SEQ_LEN          Maximum sequence length for tokenisation (default: 512)
    MODEL_BASE_PRECISION Base model precision: nf4 (default, ~500 MB) | fp16 | bf16 (~2.4 GB)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ai_client.fhir_consumer import TrainingExample

log = logging.getLogger(__name__)


def _log_vram(label: str) -> None:
    """Logs PyTorch allocated and reserved VRAM on CUDA device 0."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    reserv = torch.cuda.memory_reserved(0) / 1024**3
    log.info("VRAM [%s] — allocated: %.2f GB | reserved: %.2f GB", label, alloc, reserv)


# ── Constants and defaults ────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))

# MODEL_BASE_PRECISION controls how the base model is loaded into VRAM.
# TurboQuant is orthogonal — it compresses LoRA deltas during FL transmission,
# not model weights, and has no effect on VRAM regardless of precision chosen here.
#   nf4  — BitsAndBytes NF4 4-bit double quant (~500 MB for 1B, ~4.2 GB for 8B)
#   fp16 — full float16, no quantisation (~2.4 GB for 1B; may improve gradient quality)
#   bf16 — full bfloat16, no quantisation (~2.4 GB for 1B; preferred on Ampere+)
_BASE_PRECISION = os.getenv("MODEL_BASE_PRECISION", "nf4").lower()
if _BASE_PRECISION not in {"nf4", "fp16", "bf16"}:
    log.warning(
        "Unknown MODEL_BASE_PRECISION=%r — falling back to 'nf4'", _BASE_PRECISION
    )
    _BASE_PRECISION = "nf4"

# Llama-3 attention and FFN modules that will receive LoRA adapters.
# q/v suffice for extraction tasks; include k/o/gate/up/down for higher
# expressiveness (at the cost of more trainable parameters).
LLAMA3_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ── Configurations ────────────────────────────────────────────────────────────


@dataclass
class QuantizationConfig:
    """NF4 4-bit quantisation parameters via bitsandbytes.

    NF4 with double quantisation reduces Llama-3.1-8B from ~16 GB (fp16) to
    ~4.2 GB of base weights. With LoRA + activations total usage is ~6-7 GB,
    within the 12 GB budget of an RTX 3060.
    """

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True  # double quantisation — saves ~0.4 GB
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class LoRAAdapterConfig:
    """LoRA adapter hyperparameters."""

    r: int = 16  # decomposition rank (capacity ↑ with r ↑)
    lora_alpha: int = 32  # effective scale = lora_alpha / r = 2.0
    lora_dropout: float = 0.05
    bias: str = "none"  # "none" | "all" | "lora_only"
    use_rslora: bool = False  # RSLoRA: normalises scale by sqrt(r)
    target_modules: list[str] = field(
        default_factory=lambda: LLAMA3_LORA_TARGET_MODULES
    )


@dataclass
class TrainingConfig:
    """Federated training loop configuration."""

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 1  # epochs per federated round
    batch_size: int = 2  # small due to VRAM constraint
    gradient_accum_steps: int = 8  # effective batch = batch_size × accum_steps = 16
    max_grad_norm: float = 1.0  # DP clipping threshold C₀ — data-independent constant
    warmup_ratio: float = 0.1  # fraction of steps used for warmup
    use_amp: bool = True  # mixed-precision (bf16 if supported)
    noise_multiplier: float = 0.0  # DP-SGD: 0.0 = no DP; >0 activates Gaussian noise
    target_delta: float = 1e-5  # DP: target δ for ε computation via RDP
    dp_subsample_rate: float = 0.1  # DP: fraction of dataset sampled per round (q≪1)
    calibrate_grad_norm: bool = False  # calibration mode: logs LoRA norms without DP
    # FedProx: (μ/2)||LoRA_local - LoRA_global||² applied only on LoRA adapters.
    # Mathematically valid: base weights are frozen and identical across silos,
    # so ||w_full_local - w_full_global||² = ||LoRA_local - LoRA_global||².
    # μ should be calibrated to the LoRA adapter scale (~1e-3): use μ ∈ {0.01, 0.1, 1.0}.
    proximal_mu: float = 0.0


# ── Quantised model loading ───────────────────────────────────────────────────


def build_bnb_config(cfg: QuantizationConfig) -> BitsAndBytesConfig:
    """Builds the NF4 4-bit BitsAndBytesConfig with double quantisation."""
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_quantized_model(
    model_name: str = DEFAULT_MODEL_NAME,
    quant_cfg: QuantizationConfig | None = None,
    device_map: str | dict = "cuda:0",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads model and tokeniser with precision controlled by MODEL_BASE_PRECISION.

    - nf4  (default): NF4 4-bit via BitsAndBytes; base weights frozen.
    - fp16 / bf16: full-precision load, no BitsAndBytes; higher VRAM, better gradients.

    Args:
        model_name: HuggingFace ID or local path.
        quant_cfg:  Used only when MODEL_BASE_PRECISION=nf4 (ignored otherwise).
        device_map: Device mapping passed to from_pretrained.

    Returns:
        (model, tokenizer) ready for `apply_lora()`.
    """
    hf_token = os.getenv("HF_TOKEN")

    log.info("Loading tokeniser: %s", model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
            padding_side="right",
        )
    except Exception as _e:
        # HF Hub may return 500 when checking is_base_mistral even with cached model.
        # Fallback to local cache avoids failure due to transient API unavailability.
        log.warning("HF Hub unavailable (%s). Loading tokeniser from local cache.", _e)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
            padding_side="right",
            local_files_only=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _kwargs: dict = dict(
        device_map=device_map, token=hf_token, attn_implementation="eager"
    )

    if _BASE_PRECISION == "nf4":
        quant_cfg = quant_cfg or QuantizationConfig()
        bnb_config = build_bnb_config(quant_cfg)
        log.info(
            "Loading model — NF4 4-bit (BitsAndBytes double quant): %s", model_name
        )
        _kwargs["quantization_config"] = bnb_config
    else:
        dtype = torch.float16 if _BASE_PRECISION == "fp16" else torch.bfloat16
        log.info(
            "Loading model — %s full precision (no quantisation, VRAM ~2× NF4): %s",
            _BASE_PRECISION.upper(),
            model_name,
        )
        _kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **_kwargs)
    except Exception as _e:
        log.warning("HF Hub unavailable (%s). Loading model from local cache.", _e)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True, **_kwargs
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "Base parameters — total: %d M | trainable before LoRA: %d | precision: %s",
        total // 1_000_000,
        trainable,
        _BASE_PRECISION,
    )
    _log_vram("after load_quantized_model")

    return model, tokenizer


# ── LoRA adapter injection ────────────────────────────────────────────────────


def apply_lora(
    model: PreTrainedModel,
    lora_cfg: LoRAAdapterConfig | None = None,
) -> PreTrainedModel:
    """
    Prepares the model for k-bit training and injects LoRA adapters.

    Steps:
        1. `prepare_model_for_kbit_training` enables gradient checkpointing and
           converts LayerNorms to float32 (required for gradient stability with
           quantised weights).
        2. `LoraConfig` defines the adapter hyperparameters.
        3. `get_peft_model` freezes base weights and adds trainable LoRA modules
           to the specified attention and FFN projections.

    Args:
        model:    Model loaded via `load_quantized_model`.
        lora_cfg: LoRA hyperparameters (uses LoRAAdapterConfig() if None).

    Returns:
        PeftModel with only the LoRA adapters marked as trainable.
    """
    if lora_cfg is None:
        lora_cfg = LoRAAdapterConfig()

    # prepare_model_for_kbit_training freezes base params, enables gradient
    # checkpointing, and casts LayerNorms to fp32. Safe for all precisions —
    # the fp32 cast is a no-op on non-quantised models.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        use_rslora=lora_cfg.use_rslora,
        target_modules=lora_cfg.target_modules,
        # modules_to_save is not needed: the Llama-3.1-8B vocabulary is not
        # expanded in this project. Including embed_tokens/lm_head here causes
        # PEFT to save full trainable copies of those layers (~4 GB in float32),
        # causing OOM on the 12 GB GPU and WSL crash.
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Trainable LoRA parameters: %d (%.4f%% of total)",
        trainable,
        100 * trainable / sum(p.numel() for p in model.parameters()),
    )
    _log_vram("after apply_lora")

    return model


# ── Dataset and tokenisation ──────────────────────────────────────────────────


class ClinicalICD10Dataset(Dataset):
    """
    PyTorch dataset for ICD-10 extraction fine-tuning.

    Each example is formatted as an instruction→response prompt in Alpaca style
    and tokenised with truncation/padding to `max_length` tokens.
    Loss is computed **only over response tokens** (instruction tokens receive
    label = -100 to be ignored in cross-entropy).
    """

    # Delimiter separating instruction from response in the prompt.
    # WARNING: Do NOT translate — this is part of the prompt format the model
    # was fine-tuned on; changing it would break generation.
    RESPONSE_SEPARATOR = "### Resposta:\n"

    def __init__(
        self,
        examples: list[TrainingExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        raw_items = [self._tokenize(ex) for ex in examples]
        self.items = [item for item in raw_items if item is not None]
        n_dropped = len(raw_items) - len(self.items)
        if n_dropped:
            log.warning(
                "Dataset: %d/%d examples dropped — response truncated by "
                "max_length=%d. Increase MAX_SEQ_LEN or reduce clinical text.",
                n_dropped,
                len(raw_items),
                max_length,
            )
        log.info(
            "Dataset created: %d examples, max_length=%d", len(self.items), max_length
        )

    def _tokenize(self, example: TrainingExample) -> dict[str, torch.Tensor]:
        full_prompt = example.to_prompt()

        # Tokenise the full prompt
        full_enc = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Locate where the response starts to mask the instruction prefix.
        # Uses add_special_tokens=True to include BOS in the count — full_enc
        # also has BOS, so prefix_len covers BOS + instruction tokens.
        prefix = full_prompt.split(self.RESPONSE_SEPARATOR)[0] + self.RESPONSE_SEPARATOR
        prefix_ids = self.tokenizer(
            prefix,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        prefix_len = min(len(prefix_ids), self.max_length)

        # Labels: -100 on the instruction part, real ids on the response part
        labels = input_ids.clone()
        labels[:prefix_len] = -100
        # Also ignore padding
        labels[attention_mask == 0] = -100

        # Truncation can cut all response tokens; no active tokens → outputs.loss=NaN
        if (labels != -100).sum() == 0:
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def build_dataset(
    examples: list[TrainingExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
) -> ClinicalICD10Dataset:
    """Builds the ClinicalICD10Dataset from fhir_consumer examples."""
    return ClinicalICD10Dataset(examples, tokenizer, max_length)


# ── PyTorch training loop ─────────────────────────────────────────────────────


def train_one_round(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[TrainingExample],
    train_cfg: TrainingConfig | None = None,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
) -> tuple[list[np.ndarray], int, dict[str, float]]:
    """
    Executes one federated training round on local FHIR examples.

    Returns only the **LoRA adapter weights** (not the quantised base weights),
    which the Flower client sends to the server for aggregation.

    Args:
        model:      PeftModel returned by `apply_lora`.
        tokenizer:  Tokeniser matching the model.
        examples:   Training examples from `fetch_training_examples`.
        train_cfg:  Training hyperparameters (uses TrainingConfig() if None).
        max_length: Maximum tokenisation length.

    Returns:
        Tuple `(lora_parameters, num_examples, metrics)` in the format expected
        by Flower's `NumPyClient.fit()`:
          - lora_parameters: list of NumPy arrays with updated LoRA weights.
          - num_examples:    number of examples used in training.
          - metrics:         dict with "train_loss" and "train_perplexity".
    """
    if not examples:
        log.warning("No examples for training — round skipped.")
        return (
            get_lora_parameters(model),
            0,
            {"train_loss": 0.0, "train_perplexity": 1.0},
        )

    _log_vram("start of train_one_round")

    if train_cfg is None:
        train_cfg = TrainingConfig()

    # Diagnostic: average lora_B norm to verify initial state
    _lora_state_diag = get_peft_model_state_dict(model)
    _lora_b_norms = [
        v.float().norm().item() for k, v in _lora_state_diag.items() if "lora_B" in k
    ]
    _lora_b_norm_ini = sum(_lora_b_norms) / max(len(_lora_b_norms), 1)
    log.info(
        "Average lora_B norm (start): %.6f | tensors=%d",
        _lora_b_norm_ini,
        len(_lora_b_norms),
    )
    del _lora_state_diag, _lora_b_norms

    # DP-SGD: with active noise, force batch_size=1 and gradient_accum_steps=1
    # so each optimizer step corresponds to 1 subsampled example — necessary
    # condition for amplification-by-subsampling to be valid in the RDP accountant.
    dp_active = train_cfg.noise_multiplier > 0.0
    calibrating = train_cfg.calibrate_grad_norm
    effective_batch_size = 1 if (dp_active or calibrating) else train_cfg.batch_size
    effective_accum_steps = (
        1 if (dp_active or calibrating) else train_cfg.gradient_accum_steps
    )

    if calibrating:
        log.info(
            "CALIBRATION MODE grad norm: 1 round without DP, collecting LoRA norms."
        )
        dp_accountant = None
        dp_sample_rate = 1.0
        _calib_norms: list[float] = []
    elif dp_active:
        log.info(
            "DP-SGD active: σ=%.2f | C=%.4f | q=%.3f | batch=1 | accum=1",
            train_cfg.noise_multiplier,
            train_cfg.max_grad_norm,
            train_cfg.dp_subsample_rate,
        )
        try:
            from opacus.accountants import RDPAccountant

            dp_accountant = RDPAccountant()
        except ImportError:
            log.warning("opacus not installed — ε accounting disabled.")
            dp_accountant = None
    else:
        dp_accountant = None
        dp_sample_rate = 0.0

    # FedProx: capture global LoRA weights as an immutable reference before any update.
    # These weights are what the server sent (set_lora_parameters was already called by caller).
    proximal_mu = train_cfg.proximal_mu
    if proximal_mu > 0.0:
        _global_ref = [
            p.detach().float().clone() for p in model.parameters() if p.requires_grad
        ]
        log.info(
            "FedProx active: μ=%.4f | %d LoRA tensors as global reference.",
            proximal_mu,
            len(_global_ref),
        )
    else:
        _global_ref = None

    dataset = build_dataset(examples, tokenizer, max_length)
    if len(dataset) == 0:
        log.warning(
            "train_one_round: empty dataset after filtering (max_length=%d). "
            "Increase MAX_SEQ_LEN. Returning unchanged weights.",
            max_length,
        )
        params = get_lora_parameters(model)
        return params, 0, {"loss": float("nan"), "perplexity": float("nan")}
    n_total = max(len(dataset), 1)

    if dp_active or calibrating:
        # Explicit subsampling per round: each round uses k = q·n distinct samples.
        # q = dp_subsample_rate during DP; q=1.0 during calibration (full dataset).
        q = train_cfg.dp_subsample_rate if dp_active else 1.0
        k = max(1, int(n_total * q))
        dp_sample_rate = k / n_total
        rng_sub = np.random.default_rng()  # OS entropy — fresh randomness every round
        subset_indices = rng_sub.choice(n_total, size=k, replace=False).tolist()
        from torch.utils.data import SubsetRandomSampler

        sampler = SubsetRandomSampler(subset_indices)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        log.info("DP subsampling: k=%d/%d samples (q=%.4f)", k, n_total, dp_sample_rate)
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    total_steps = len(dataloader) * train_cfg.num_epochs // effective_accum_steps
    warmup_steps = max(1, int(total_steps * train_cfg.warmup_ratio))
    # eta_min = 10% of initial LR — ensures LR never reaches 0 in a single round,
    # which would cause the last optimizer step to produce a null update.
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=train_cfg.learning_rate * 0.1,
    )

    # AMP: bfloat16 on Ampere+; float16 as fallback
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=train_cfg.use_amp and amp_dtype == torch.float16
    )

    model.train()
    cumulative_loss = 0.0
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(train_cfg.num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        # Count of valid accumulated batches since the last optimizer.step().
        # Separate from the step index so NaN batches don't misalign the window.
        accum_count = 0

        for step, batch in enumerate(dataloader):
            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=train_cfg.use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                if proximal_mu > 0.0 and _global_ref is not None:
                    prox_term = sum(
                        (p.float() - ref).pow(2).sum()
                        for p, ref in zip(
                            (q for q in model.parameters() if q.requires_grad),
                            _global_ref,
                        )
                    )
                    loss_scaled = (
                        outputs.loss + (proximal_mu / 2) * prox_term
                    ) / effective_accum_steps
                else:
                    loss_scaled = outputs.loss / effective_accum_steps

            raw_loss = outputs.loss.item()
            if not math.isfinite(raw_loss):
                log.warning(
                    "Non-finite loss (%.4g) at step %d — batch skipped.", raw_loss, step
                )
                continue

            scaler.scale(loss_scaled).backward()
            epoch_loss += raw_loss
            valid_batches += 1
            accum_count += 1

            # Weight update every `effective_accum_steps` VALID batches
            if accum_count % effective_accum_steps == 0:
                scaler.unscale_(optimizer)

                # Calibration: measure raw norm BEFORE any clipping.
                # clip_grad_norm_ modifies gradients in-place, so the measurement
                # must happen first — otherwise we always record min(true_norm, C₀).
                if calibrating:
                    raw_norm = torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        float("inf"),  # inf → no modification, pure measurement
                    )
                    if math.isfinite(float(raw_norm)) and float(raw_norm) > 0:
                        _calib_norms.append(float(raw_norm))

                total_norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    train_cfg.max_grad_norm,
                )

                # DP-SGD: after clipping, inject Gaussian noise calibrated to C·σ on each gradient
                if dp_active and math.isfinite(float(total_norm)):
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.requires_grad and p.grad is not None:
                                noise = torch.normal(
                                    mean=0.0,
                                    std=train_cfg.max_grad_norm
                                    * train_cfg.noise_multiplier,
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
                    log.warning(
                        "Non-finite grad norm (%.4g) at global_step %d — optimizer step skipped.",
                        float(total_norm),
                        global_step,
                    )
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0

                if global_step >= warmup_steps:
                    scheduler.step()
                global_step += 1

                log.info(
                    "Epoch %d/%d | opt_step %d | loss=%.4f | lr=%.2e",
                    epoch + 1,
                    train_cfg.num_epochs,
                    global_step,
                    epoch_loss / max(valid_batches, 1),
                    optimizer.param_groups[0]["lr"],
                )

        # Flush residual gradients at epoch end (incomplete accumulation window)
        if accum_count > 0:
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
                                std=train_cfg.max_grad_norm
                                * train_cfg.noise_multiplier,
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
        log.info(
            "Epoch %d/%d completed — average loss: %.4f",
            epoch + 1,
            train_cfg.num_epochs,
            avg_epoch_loss,
        )
        cumulative_loss += avg_epoch_loss

    avg_loss = cumulative_loss / max(train_cfg.num_epochs, 1)
    perplexity = float(torch.exp(torch.tensor(avg_loss)).item())

    # Calibration: save norm distribution and recommend max_grad_norm
    if calibrating and _calib_norms:
        p50 = float(np.percentile(_calib_norms, 50))
        p75 = float(np.percentile(_calib_norms, 75))
        p95 = float(np.percentile(_calib_norms, 95))
        log.info(
            "CALIBRATION grad norm LoRA — p50=%.6f | p75=%.6f | p95=%.6f | n=%d",
            p50,
            p75,
            p95,
            len(_calib_norms),
        )
        log.info("RECOMMENDATION: use max_grad_norm=%.6f (75th percentile)", p75)
        import json
        import pathlib

        out = pathlib.Path(
            os.getenv("FL_CALIB_OUTPUT", "/tmp/grad_norm_calibration.json")
        )
        out.write_text(
            json.dumps(
                {
                    "p50": p50,
                    "p75": p75,
                    "p95": p95,
                    "recommended_max_grad_norm": p75,
                    "n_steps": len(_calib_norms),
                },
                indent=2,
            )
        )
        log.info("Calibration saved to: %s", out)

    _lora_state_end = get_peft_model_state_dict(model)
    _lora_b_end = [
        v.float().norm().item() for k, v in _lora_state_end.items() if "lora_B" in k
    ]
    _lora_b_norm_end = sum(_lora_b_end) / max(len(_lora_b_end), 1)
    _drift_ratio = _lora_b_norm_end / max(_lora_b_norm_ini, 1e-9)
    log.info(
        "Average lora_B norm (end): %.6f | drift_ratio=%.2fx",
        _lora_b_norm_end,
        _drift_ratio,
    )
    del _lora_state_end, _lora_b_end

    metrics = {
        "train_loss": round(avg_loss, 6),
        "train_perplexity": round(perplexity, 4),
        "lora_b_norm_end": round(_lora_b_norm_end, 6),
        "lora_b_drift_ratio": round(_drift_ratio, 4),
    }

    if dp_active and dp_accountant is not None:
        try:
            epsilon = dp_accountant.get_epsilon(delta=train_cfg.target_delta)
            metrics["dp_epsilon"] = round(float(epsilon), 4)
            metrics["dp_delta"] = train_cfg.target_delta
            metrics["dp_noise_multiplier"] = train_cfg.noise_multiplier
            metrics["dp_sample_rate"] = dp_sample_rate
            metrics["dp_steps_this_round"] = global_step
            log.info(
                "DP: ε=%.4f | δ=%s | σ=%.2f | steps=%d",
                epsilon,
                train_cfg.target_delta,
                train_cfg.noise_multiplier,
                global_step,
            )
        except Exception as exc:
            log.warning("Error computing DP ε: %s", exc)

    log.info("Round completed — loss=%.4f | perplexity=%.2f", avg_loss, perplexity)
    _log_vram("end of train_one_round")

    return get_lora_parameters(model), len(examples), metrics


# ── Continuous training (centralised baseline) ────────────────────────────────


def train_continuous(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[TrainingExample],
    num_epochs: int,
    learning_rate: float = 5e-5,
    batch_size: int = 1,
    gradient_accum_steps: int = 64,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Trains for num_epochs without resetting the optimizer between epochs.

    Unlike train_one_round() (which recreates the optimizer on each federated
    call), here AdamW preserves 1st and 2nd moment estimates between epochs —
    identical behaviour to standard centralised fine-tuning.

    Returns a list of per-epoch metrics for round-by-round comparison with FL.
    """
    if not examples:
        log.warning("train_continuous: no examples — aborting.")
        return []

    dataset = build_dataset(examples, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    total_steps = max(1, len(dataloader) * num_epochs // gradient_accum_steps)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=learning_rate * 0.1,
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    per_epoch_metrics: list[dict[str, Any]] = []
    global_step = 0
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        valid_batches = 0
        accum_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()
            ):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss_scaled = outputs.loss / gradient_accum_steps

            raw_loss = outputs.loss.item()
            if not math.isfinite(raw_loss):
                log.warning(
                    "Non-finite loss (%.4g) at step %d — batch skipped.", raw_loss, step
                )
                continue

            scaler.scale(loss_scaled).backward()
            epoch_loss += raw_loss
            valid_batches += 1
            accum_count += 1

            if accum_count % gradient_accum_steps == 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm
                )
                if math.isfinite(float(total_norm)):
                    scaler.step(optimizer)
                else:
                    log.warning(
                        "Non-finite grad norm at global_step %d — optimizer step skipped.",
                        global_step,
                    )
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0
                if global_step >= warmup_steps:
                    scheduler.step()
                global_step += 1
                log.info(
                    "Epoch %d/%d | opt_step %d | loss=%.4f | lr=%.2e",
                    epoch,
                    num_epochs,
                    global_step,
                    epoch_loss / max(valid_batches, 1),
                    optimizer.param_groups[0]["lr"],
                )

        # Flush residual gradients at epoch end
        if accum_count > 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm
            )
            if math.isfinite(float(total_norm)):
                scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(valid_batches, 1)
        metrics = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "train_perplexity": round(
                float(torch.exp(torch.tensor(avg_loss)).item()), 4
            ),
            "opt_steps_total": global_step,
        }
        log.info(
            "Epoch %d/%d completed — loss=%.4f | ppl=%.2f | accumulated_opt_steps=%d",
            epoch,
            num_epochs,
            avg_loss,
            metrics["train_perplexity"],
            global_step,
        )
        per_epoch_metrics.append(metrics)

    return per_epoch_metrics


# ── Flower parameter utilities ────────────────────────────────────────────────


def get_lora_parameters(model: PreTrainedModel) -> list[np.ndarray]:
    """
    Extracts only the LoRA adapter weights as a list of NumPy arrays.

    This is the representation exchanged between Flower client and server during
    federated aggregation (FedAvg or similar). The quantised base weights are
    **not** included — only the LoRA deltas.

    Returns:
        Ordered list of NumPy arrays corresponding to each LoRA tensor.
    """
    lora_state = get_peft_model_state_dict(model)
    return [v.detach().float().cpu().numpy() for v in lora_state.values()]


def set_lora_parameters(model: PreTrainedModel, parameters: list[np.ndarray]) -> None:
    """
    Applies aggregated LoRA weights (received from the Flower server) to the local model.

    Called in `NumPyClient.configure_fit()` or `NumPyClient.evaluate()` before
    any inference or subsequent training round.

    Args:
        model:      PeftModel with LoRA adapters.
        parameters: List of NumPy arrays in the same format as `get_lora_parameters`.
    """
    lora_state = get_peft_model_state_dict(model)
    keys = list(lora_state.keys())

    if len(keys) != len(parameters):
        raise ValueError(
            f"Parameter mismatch: model has {len(keys)} LoRA tensors, "
            f"but received {len(parameters)}."
        )

    new_state = {
        k: torch.tensor(v, dtype=lora_state[k].dtype) for k, v in zip(keys, parameters)
    }
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, new_state)
    log.info(
        "LoRA weights updated with aggregated parameters from server (%d tensors).",
        len(keys),
    )
