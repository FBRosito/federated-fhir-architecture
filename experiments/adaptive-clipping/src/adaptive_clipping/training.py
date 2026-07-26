"""
training.py
------------
Per-layer-clipping variants of HERALD's training loops.

ai_client.model_setup_bert.train_bert_one_round() and
ai_client.model_setup.train_one_round() implement DP-SGD manually — a single
global L2 clip (torch.nn.utils.clip_grad_norm_) over all trainable
parameters, followed by a manual Gaussian noise loop. Opacus's
PrivacyEngine is never wired into that path; opacus.accountants.RDPAccountant
is used purely for (epsilon, delta) bookkeeping via .step(noise_multiplier, sample_rate)
and never sees clip thresholds. Because that clip+noise logic is inlined in
free functions we cannot subclass or monkeypatch without modifying files
outside experiments/adaptive-clipping/, this module reimplements the two
training loops verbatim except at the clip+noise step, which is replaced by
a per-attention-layer version using adaptive_clipping.clipping.PerLayerClipper.

Sensitivity accounting: per-layer thresholds returned by
PerLayerClipper.compute_thresholds() are rescaled so that the vector
sensitivity of the whole gradient, sqrt(sum(C_l**2) for C_l in thresholds),
equals train_cfg.max_grad_norm (the same C0 the baseline pipeline uses).
This bounds the CLIP correctly, but dp_accountant.step() is still called
with a single scalar noise_multiplier while the actual noise std added per
group is normalized_threshold_l * noise_multiplier — i.e. non-isotropic
across the parameter vector. RDP for a Gaussian mechanism is not a function
of the global L2 sensitivity alone when the per-coordinate noise variance
is non-uniform: an adversary whose worst-case perturbation is spread across
every low-threshold group (each saturating its own C_l simultaneously)
produces a larger Rényi divergence than the scalar accountant reports,
because the noise-to-sensitivity ratio the accountant assumes (std/C0) does
not hold group-by-group when std is itself scaled by C_l. Confirmed
empirically: dp_accountant.step()'s reported epsilon is bit-identical
between "baseline" and "per_layer" runs at the same sigma, despite the
per_layer path adding noise 18x-427x smaller on the encoder groups — see
the audit that motivated this fix. Do not treat epsilon_cumulative from the
per_layer path as an accurate privacy guarantee until this is corrected.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np
import torch
from peft import get_peft_model_state_dict
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from adaptive_clipping.clipping import PerLayerClipper
from ai_client.fhir_consumer import TrainingExample
from ai_client.model_setup import DEFAULT_MAX_SEQ_LEN as LLM_DEFAULT_MAX_SEQ_LEN
from ai_client.model_setup import TrainingConfig, build_dataset, get_lora_parameters
from ai_client.model_setup_bert import DEFAULT_MAX_SEQ_LEN as BERT_DEFAULT_MAX_SEQ_LEN
from ai_client.model_setup_bert import (
    BertTrainingConfig,
    PLMICDModel,
    build_bert_dataset,
    get_bert_parameters,
)

log = logging.getLogger(__name__)


def _normalized_thresholds(clipper: PerLayerClipper, max_grad_norm: float) -> dict[str, float]:
    thresholds = clipper.compute_thresholds()
    vector_norm = math.sqrt(sum(c**2 for c in thresholds.values()))
    scale = max_grad_norm / max(vector_norm, 1e-8)
    return {name: c * scale for name, c in thresholds.items()}


def _clip_and_noise_per_layer(
    groups: dict[str, list[torch.nn.Parameter]],
    normalized_thresholds: dict[str, float],
    dp_active: bool,
    noise_multiplier: float,
) -> float:
    """Clips each attention-layer group to its own (normalized) threshold,
    then — if DP is active — adds Gaussian noise to each group's gradients
    with std = normalized_threshold * noise_multiplier. Returns the combined
    total norm (post-clip), mirroring the scalar `total_norm` the baseline
    pipeline uses for its finite-value guard.

    `groups` must be the dict returned by this round's clipper.update_history()
    call (current round's live parameters) — never clipper.groups, which no
    longer exists precisely because a cached copy goes stale after one round
    (see clipping.py's PerLayerClipper docstring)."""
    total_norm_sq = 0.0
    for name, params in groups.items():
        if not params:
            continue
        group_norm = torch.nn.utils.clip_grad_norm_(params, normalized_thresholds[name])
        total_norm_sq += float(min(float(group_norm), normalized_thresholds[name])) ** 2
    total_norm = math.sqrt(total_norm_sq)

    if dp_active and math.isfinite(total_norm):
        with torch.no_grad():
            for name, params in groups.items():
                std_l = normalized_thresholds[name] * noise_multiplier
                for p in params:
                    if p.grad is not None:
                        noise = torch.normal(
                            mean=0.0,
                            std=std_l,
                            size=p.grad.shape,
                            device=p.grad.device,
                            dtype=p.grad.dtype,
                        )
                        p.grad.add_(noise)
    return total_norm


# ── BERT backend ────────────────────────────────────────────────────────────

def adaptive_train_bert_one_round(
    model: PLMICDModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Any],
    label_index: dict[str, int],
    train_cfg: BertTrainingConfig | None,
    clipper: PerLayerClipper,
    server_round: int,
    max_length: int = BERT_DEFAULT_MAX_SEQ_LEN,
) -> tuple[list[np.ndarray], int, dict[str, float]]:
    """Per-layer-clipping variant of ai_client.model_setup_bert.train_bert_one_round.
    See module docstring for the sensitivity-accounting justification."""
    if not examples:
        log.warning("No examples for BERT training — round skipped.")
        return get_bert_parameters(model), 0, {"train_loss": 0.0}

    if train_cfg is None:
        train_cfg = BertTrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    proximal_mu = train_cfg.proximal_mu
    if proximal_mu > 0.0:
        _global_ref = [p.detach().float().clone() for p in model.parameters() if p.requires_grad]
        log.info("FedProx BERT active: μ=%.4f | %d trainable tensors.", proximal_mu, len(_global_ref))
    else:
        _global_ref = None

    dp_active = train_cfg.noise_multiplier > 0.0
    effective_batch_size = 1 if dp_active else train_cfg.batch_size
    effective_accum_steps = 1 if dp_active else train_cfg.gradient_accum_steps

    if dp_active:
        log.info(
            "BERT per-layer DP-SGD active: σ=%.2f | C0=%.4f | δ=%s | batch=1 | accum=1",
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
        q = train_cfg.dp_subsample_rate
        k = max(1, int(n_total * q))
        dp_sample_rate = k / n_total
        rng_sub = np.random.default_rng()
        subset_indices = rng_sub.choice(n_total, size=k, replace=False).tolist()
        sampler = SubsetRandomSampler(subset_indices)
        dataloader = DataLoader(
            dataset, batch_size=effective_batch_size, sampler=sampler,
            drop_last=False, pin_memory=torch.cuda.is_available(),
        )
        log.info("BERT DP subsampling: k=%d/%d samples (q=%.4f)", k, n_total, dp_sample_rate)
    else:
        dp_sample_rate = 1.0
        dataloader = DataLoader(
            dataset, batch_size=effective_batch_size, shuffle=True,
            drop_last=False, pin_memory=torch.cuda.is_available(),
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95), eps=1e-8,
    )
    total_steps = max(1, len(dataloader) * train_cfg.num_epochs // effective_accum_steps)
    warmup_steps = max(1, int(total_steps * train_cfg.warmup_ratio))
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps),
        eta_min=train_cfg.learning_rate * 0.1,
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.use_amp and amp_dtype == torch.float16)

    model.train()
    cumulative_loss = 0.0
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(train_cfg.num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        accum_count = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=train_cfg.use_amp):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ce_loss = out["loss"]
                if proximal_mu > 0.0 and _global_ref is not None:
                    prox_term = sum(
                        (p.float() - ref).pow(2).sum()
                        for p, ref in zip(
                            (q for q in model.parameters() if q.requires_grad), _global_ref,
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
            epoch_loss += raw_loss
            valid_batches += 1
            accum_count += 1

            if accum_count % effective_accum_steps == 0:
                scaler.unscale_(optimizer)
                groups = clipper.update_history(server_round, model.named_parameters())
                normalized_thresholds = _normalized_thresholds(clipper, train_cfg.max_grad_norm)
                total_norm = _clip_and_noise_per_layer(
                    groups, normalized_thresholds, dp_active, train_cfg.noise_multiplier,
                )
                if dp_active and math.isfinite(total_norm) and dp_accountant is not None:
                    dp_accountant.step(
                        noise_multiplier=train_cfg.noise_multiplier, sample_rate=dp_sample_rate,
                    )
                if math.isfinite(total_norm):
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
                    epoch_loss / max(valid_batches, 1), optimizer.param_groups[0]["lr"],
                )

        if accum_count > 0:
            scaler.unscale_(optimizer)
            groups = clipper.update_history(server_round, model.named_parameters())
            normalized_thresholds = _normalized_thresholds(clipper, train_cfg.max_grad_norm)
            total_norm = _clip_and_noise_per_layer(
                groups, normalized_thresholds, dp_active, train_cfg.noise_multiplier,
            )
            if dp_active and math.isfinite(total_norm) and dp_accountant is not None:
                dp_accountant.step(
                    noise_multiplier=train_cfg.noise_multiplier, sample_rate=dp_sample_rate,
                )
            if math.isfinite(total_norm):
                scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        avg_epoch_loss = epoch_loss / max(valid_batches, 1)
        log.info("BERT Epoch %d/%d completed — loss=%.4f", epoch + 1, train_cfg.num_epochs, avg_epoch_loss)
        cumulative_loss += avg_epoch_loss

    avg_loss = cumulative_loss / max(train_cfg.num_epochs, 1)
    metrics = {
        "train_loss": round(avg_loss, 6),
        "clipping_strategy": "per_layer",
        "per_layer_thresholds": json.dumps(normalized_thresholds),
        "per_layer_norms": json.dumps(clipper.get_history_summary()),
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
                "BERT per-layer DP: ε=%.4f | δ=%s | σ=%.2f | steps=%d",
                epsilon, train_cfg.target_delta, train_cfg.noise_multiplier, global_step,
            )
        except Exception as exc:
            log.warning("Error computing BERT DP ε: %s", exc)

    log.info("BERT per-layer round completed — loss=%.4f", avg_loss)
    return get_bert_parameters(model), len(examples), metrics


# ── LLM backend ─────────────────────────────────────────────────────────────

def adaptive_train_one_round(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[TrainingExample],
    train_cfg: TrainingConfig | None,
    clipper: PerLayerClipper,
    server_round: int,
    max_length: int = LLM_DEFAULT_MAX_SEQ_LEN,
) -> tuple[list[np.ndarray], int, dict[str, float]]:
    """Per-layer-clipping variant of ai_client.model_setup.train_one_round.
    See module docstring for the sensitivity-accounting justification."""
    if not examples:
        log.warning("No examples for training — round skipped.")
        return get_lora_parameters(model), 0, {"train_loss": 0.0, "train_perplexity": 1.0}

    if train_cfg is None:
        train_cfg = TrainingConfig()

    _lora_state_diag = get_peft_model_state_dict(model)
    _lora_b_norms = [v.float().norm().item() for k, v in _lora_state_diag.items() if "lora_B" in k]
    _lora_b_norm_ini = sum(_lora_b_norms) / max(len(_lora_b_norms), 1)
    log.info("Average lora_B norm (start): %.6f | tensors=%d", _lora_b_norm_ini, len(_lora_b_norms))
    del _lora_state_diag, _lora_b_norms

    dp_active = train_cfg.noise_multiplier > 0.0
    calibrating = train_cfg.calibrate_grad_norm
    effective_batch_size = 1 if (dp_active or calibrating) else train_cfg.batch_size
    effective_accum_steps = 1 if (dp_active or calibrating) else train_cfg.gradient_accum_steps

    if calibrating:
        log.info("CALIBRATION MODE grad norm: 1 round without DP, collecting LoRA norms.")
        dp_accountant = None
        dp_sample_rate = 1.0
        _calib_norms: list[float] = []
    elif dp_active:
        log.info(
            "Per-layer DP-SGD active: σ=%.2f | C0=%.4f | q=%.3f | batch=1 | accum=1",
            train_cfg.noise_multiplier, train_cfg.max_grad_norm, train_cfg.dp_subsample_rate,
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

    proximal_mu = train_cfg.proximal_mu
    if proximal_mu > 0.0:
        _global_ref = [p.detach().float().clone() for p in model.parameters() if p.requires_grad]
        log.info("FedProx active: μ=%.4f | %d LoRA tensors as global reference.", proximal_mu, len(_global_ref))
    else:
        _global_ref = None

    dataset = build_dataset(examples, tokenizer, max_length)
    if len(dataset) == 0:
        log.warning(
            "adaptive_train_one_round: empty dataset after filtering (max_length=%d). Returning unchanged weights.",
            max_length,
        )
        return get_lora_parameters(model), 0, {"loss": float("nan"), "perplexity": float("nan")}
    n_total = max(len(dataset), 1)

    if dp_active or calibrating:
        q = train_cfg.dp_subsample_rate if dp_active else 1.0
        k = max(1, int(n_total * q))
        dp_sample_rate = k / n_total
        rng_sub = np.random.default_rng()
        subset_indices = rng_sub.choice(n_total, size=k, replace=False).tolist()
        sampler = SubsetRandomSampler(subset_indices)
        dataloader = DataLoader(
            dataset, batch_size=1, sampler=sampler, drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        log.info("DP subsampling: k=%d/%d samples (q=%.4f)", k, n_total, dp_sample_rate)
    else:
        dataloader = DataLoader(
            dataset, batch_size=effective_batch_size, shuffle=True,
            drop_last=False, pin_memory=torch.cuda.is_available(),
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95), eps=1e-8,
    )

    total_steps = len(dataloader) * train_cfg.num_epochs // effective_accum_steps
    warmup_steps = max(1, int(total_steps * train_cfg.warmup_ratio))
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps),
        eta_min=train_cfg.learning_rate * 0.1,
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.use_amp and amp_dtype == torch.float16)

    model.train()
    cumulative_loss = 0.0
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(train_cfg.num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        accum_count = 0

        for step, batch in enumerate(dataloader):
            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=train_cfg.use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                if proximal_mu > 0.0 and _global_ref is not None:
                    prox_term = sum(
                        (p.float() - ref).pow(2).sum()
                        for p, ref in zip(
                            (q for q in model.parameters() if q.requires_grad), _global_ref,
                        )
                    )
                    loss_scaled = (outputs.loss + (proximal_mu / 2) * prox_term) / effective_accum_steps
                else:
                    loss_scaled = outputs.loss / effective_accum_steps

            raw_loss = outputs.loss.item()
            if not math.isfinite(raw_loss):
                log.warning("Non-finite loss (%.4g) at step %d — batch skipped.", raw_loss, step)
                continue

            scaler.scale(loss_scaled).backward()
            epoch_loss += raw_loss
            valid_batches += 1
            accum_count += 1

            if accum_count % effective_accum_steps == 0:
                scaler.unscale_(optimizer)

                if calibrating:
                    raw_norm = torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), float("inf"),
                    )
                    if math.isfinite(float(raw_norm)) and float(raw_norm) > 0:
                        _calib_norms.append(float(raw_norm))

                groups = clipper.update_history(server_round, model.named_parameters())
                normalized_thresholds = _normalized_thresholds(clipper, train_cfg.max_grad_norm)
                total_norm = _clip_and_noise_per_layer(
                    groups, normalized_thresholds, dp_active, train_cfg.noise_multiplier,
                )
                if dp_active and math.isfinite(total_norm) and dp_accountant is not None:
                    dp_accountant.step(
                        noise_multiplier=train_cfg.noise_multiplier, sample_rate=dp_sample_rate,
                    )

                if math.isfinite(total_norm):
                    scaler.step(optimizer)
                else:
                    log.warning(
                        "Non-finite grad norm (%.4g) at global_step %d — optimizer step skipped.",
                        total_norm, global_step,
                    )
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0

                if global_step >= warmup_steps:
                    scheduler.step()
                global_step += 1

                log.info(
                    "Epoch %d/%d | opt_step %d | loss=%.4f | lr=%.2e",
                    epoch + 1, train_cfg.num_epochs, global_step,
                    epoch_loss / max(valid_batches, 1), optimizer.param_groups[0]["lr"],
                )

        if accum_count > 0:
            scaler.unscale_(optimizer)
            groups = clipper.update_history(server_round, model.named_parameters())
            normalized_thresholds = _normalized_thresholds(clipper, train_cfg.max_grad_norm)
            total_norm = _clip_and_noise_per_layer(
                groups, normalized_thresholds, dp_active, train_cfg.noise_multiplier,
            )
            if dp_active and math.isfinite(total_norm) and dp_accountant is not None:
                dp_accountant.step(
                    noise_multiplier=train_cfg.noise_multiplier, sample_rate=dp_sample_rate,
                )
            if math.isfinite(total_norm):
                scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        avg_epoch_loss = epoch_loss / max(valid_batches, 1)
        log.info("Epoch %d/%d completed — average loss: %.4f", epoch + 1, train_cfg.num_epochs, avg_epoch_loss)
        cumulative_loss += avg_epoch_loss

    avg_loss = cumulative_loss / max(train_cfg.num_epochs, 1)
    perplexity = float(torch.exp(torch.tensor(avg_loss)).item())

    if calibrating and _calib_norms:
        p50 = float(np.percentile(_calib_norms, 50))
        p75 = float(np.percentile(_calib_norms, 75))
        p95 = float(np.percentile(_calib_norms, 95))
        log.info("CALIBRATION grad norm LoRA — p50=%.6f | p75=%.6f | p95=%.6f | n=%d", p50, p75, p95, len(_calib_norms))
        log.info("RECOMMENDATION: use max_grad_norm=%.6f (75th percentile)", p75)

    _lora_state_end = get_peft_model_state_dict(model)
    _lora_b_end = [v.float().norm().item() for k, v in _lora_state_end.items() if "lora_B" in k]
    _lora_b_norm_end = sum(_lora_b_end) / max(len(_lora_b_end), 1)
    _drift_ratio = _lora_b_norm_end / max(_lora_b_norm_ini, 1e-9)
    log.info("Average lora_B norm (end): %.6f | drift_ratio=%.2fx", _lora_b_norm_end, _drift_ratio)
    del _lora_state_end, _lora_b_end

    metrics = {
        "train_loss": round(avg_loss, 6),
        "train_perplexity": round(perplexity, 4),
        "lora_b_norm_end": round(_lora_b_norm_end, 6),
        "lora_b_drift_ratio": round(_drift_ratio, 4),
        "clipping_strategy": "per_layer",
        "per_layer_thresholds": json.dumps(normalized_thresholds),
        "per_layer_norms": json.dumps(clipper.get_history_summary()),
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
                "Per-layer DP: ε=%.4f | δ=%s | σ=%.2f | steps=%d",
                epsilon, train_cfg.target_delta, train_cfg.noise_multiplier, global_step,
            )
        except Exception as exc:
            log.warning("Error computing DP ε: %s", exc)

    log.info("Per-layer round completed — loss=%.4f | perplexity=%.2f", avg_loss, perplexity)
    return get_lora_parameters(model), len(examples), metrics
