"""
fl_client.py
------------
Flower FL client for the federated learning pipeline over FHIR data.

Inherits from NumPyClient and orchestrates:
    - Lazy loading of quantized model + LoRA adapters (model_setup.py)
    - Fetching Conditions and DocumentReferences from FHIR R4 (fhir_consumer.py)
    - Local training for one round (train_one_round from model_setup.py)
    - Local evaluation with perplexity, ICD-10 accuracy, or summarization metrics

Round flow:
    1. Server sends global LoRA weights (NDArrays)
    2. `fit`:  apply weights → train one round → return updated weights + metrics
    3. `evaluate`: apply weights → compute loss/metrics → return evaluation metrics

Train/eval split:
    Examples are split 80/20 (stratified by ICD-10 code) on first call.
    The split is fixed for consistency across rounds.

Environment variables:
    FHIR_SERVER_URL       Base URL of the FHIR R4 server (default: http://localhost:8080/fhir)
    FL_SERVER_ADDRESS     Flower server address (default: fl_server:9091)
    ETL_PARTITION_ID      Non-IID partition to consume (-1 = all, default: -1)
    MODEL_NAME            HuggingFace model ID for the base model
    MAX_SEQ_LEN           Maximum tokenization length
    FL_PROXIMAL_MU        FedProx proximal coefficient μ (can be overridden by server config)
    FL_EVAL_ACCURACY      If "true", compute ICD-10 extraction accuracy during evaluation
    MODEL_BACKEND         "llm" | "bert" | "llm-summarization"
    FL_SAVE_CHECKPOINT    If set, saves LoRA weights to this directory after the final round
"""

from __future__ import annotations

import contextlib
import fcntl
import gc
import logging
import math
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context, NDArrays, Scalar

from ai_client.fhir_consumer import TrainingExample, fetch_training_examples
from ai_client.model_setup import (
    LoRAAdapterConfig,
    QuantizationConfig,
    TrainingConfig,
    apply_lora,
    build_dataset,
    get_lora_parameters,
    load_quantized_model,
    set_lora_parameters,
    train_one_round,
)

log = logging.getLogger(__name__)

# ── Environment defaults ──────────────────────────────────────────────────────

_FHIR_URL     = os.getenv("FHIR_SERVER_URL",   "http://localhost:8080/fhir")
_FL_ADDRESS   = os.getenv("FL_SERVER_ADDRESS",  "fl_server:9091")
_PARTITION_ID = int(os.getenv("ETL_PARTITION_ID", "-1"))
_MODEL_NAME   = os.getenv("MODEL_NAME",          "meta-llama/Llama-3.1-8B")
_MAX_SEQ_LEN  = int(os.getenv("MAX_SEQ_LEN",     "512"))
_EVAL_ACCURACY = os.getenv("FL_EVAL_ACCURACY",   "false").lower() == "true"
# Number of beams for evaluation with beam search (Recall@k, F1@k).
_TOP_K         = int(os.getenv("FL_TOP_K", "5"))
# DP-SGD: Gaussian noise multiplier (0.0 = no DP).
_NOISE_MULTIPLIER = float(os.getenv("FL_NOISE_MULTIPLIER", "0.0"))
# DP: target δ for ε calculation via RDP accountant.
_TARGET_DELTA     = float(os.getenv("FL_TARGET_DELTA", "1e-5"))
# DP: dataset subsampling rate per round (amplification-by-subsampling).
_DP_SUBSAMPLE_RATE = float(os.getenv("FL_DP_SUBSAMPLE_RATE", "0.1"))
# Calibration mode: no DP, logs empirical max_grad_norm of LoRA adapters.
_CALIBRATE_GRAD_NORM = os.getenv("FL_CALIBRATE_GRAD_NORM", "false").lower() == "true"
# Gradient clipping threshold C₀ — read from env so calibration output feeds back in.
_MAX_GRAD_NORM = float(os.getenv("FL_MAX_GRAD_NORM", "1.0"))

# Path to shared lock file between silos via Docker volume.
_GPU_LOCK_PATH = os.getenv("GPU_LOCK_PATH", "/var/gpu_sync/gpu.lock")
# When True, disables the lock so silos run in parallel on the GPU.
# Safe for models <3B (1B uses ~4-5 GB peak; two silos fit within 12 GB).
_PARALLEL_GPU  = os.getenv("FL_PARALLEL_GPU", "false").lower() == "true"

# Fast-dev mode: keeps model in VRAM between calls within the same round.
_KEEP_MODEL            = os.getenv("FL_KEEP_MODEL_IN_VRAM", "false").lower() == "true"
# Fast-dev mode: caps training examples (0 = no limit).
_MAX_EXAMPLES          = int(os.getenv("FL_MAX_EXAMPLES", "0"))
# Overrides gradient_accum_steps (0 = use per-model default).
_GRADIENT_ACCUM_STEPS  = int(os.getenv("FL_GRADIENT_ACCUM_STEPS", "0"))
# Batch size per step; smaller models (<3B) support 8+ without OOM.
_BATCH_SIZE            = int(os.getenv("FL_BATCH_SIZE", "8"))

# Model backend:
#   "llm"              — Llama-3.x with LoRA (default; Experiment B: discharge summary)
#   "bert"             — PubMedBERT with per-label attention (Experiment A: ICD-10 coding)
#   "llm-summarization"— Explicit alias for the generative LLM backend
_MODEL_BACKEND = os.getenv("MODEL_BACKEND", "llm").lower()
# ICD-10 benchmark for the BERT backend (top50 | full | none)
_BERT_BENCHMARK = os.getenv("BERT_BENCHMARK", "full")

# If set, save LoRA adapter weights to this directory after the final training round.
# Used by post_eval.py for LLM-as-judge evaluation.
_CHECKPOINT_DIR = os.getenv("FL_SAVE_CHECKPOINT", "")

# TurboQuant LoRA delta compression (two-stage: Lloyd-Max + QJL residual).
# Reduces per-round communication: Llama-3.2-1B r=16 from ~32 MB to ~8 MB at 4-bit.
# Inner products are preserved in expectation — FedProx convergence is maintained.
_LORA_COMPRESS  = os.getenv("FL_LORA_COMPRESS", "false").lower() == "true"
_COMPRESS_BITS  = int(os.getenv("FL_COMPRESS_BITS", "4"))


@contextlib.contextmanager
def _gpu_lock():
    """
    Serializes GPU access between silos via POSIX flock on a shared Docker volume.

    When FL_PARALLEL_GPU=true, the lock is disabled and silos run in parallel.
    Ensures the lock is released even if an exception occurs.
    """
    if _PARALLEL_GPU:
        yield
        return
    os.makedirs(os.path.dirname(_GPU_LOCK_PATH), exist_ok=True)
    with open(_GPU_LOCK_PATH, "w") as lock_file:
        log.info("Waiting for GPU lock (%s)...", _GPU_LOCK_PATH)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        log.info("GPU lock acquired.")
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            log.info("GPU lock released.")

# ── Stratified train/eval split ───────────────────────────────────────────────

def _stratified_split(
    examples: list[TrainingExample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """
    Splits examples into train/eval while preserving the proportion of each
    ICD-10 code — important given the Non-IID nature of the data.

    Args:
        examples:    List of TrainingExample (or SummarizationExample, duck-typed).
        train_ratio: Fraction assigned to training.
        seed:        Random seed for reproducibility.

    Returns:
        (train_examples, eval_examples)
    """
    rng = np.random.default_rng(seed)
    by_code: dict[str, list] = defaultdict(list)
    for ex in examples:
        code = getattr(ex, "icd10_code", "unknown")
        by_code[code].append(ex)

    train, eval_ = [], []
    for code, group in by_code.items():
        shuffled = list(rng.permutation(group))  # type: ignore[arg-type]
        n_train = max(1, math.floor(len(shuffled) * train_ratio))
        train.extend(shuffled[:n_train])
        eval_.extend(shuffled[n_train:])

    # Fallback: when all groups have only 1 sample (e.g. 20 unique ICD-10 codes with
    # 1 example each), stratified split puts nothing in eval_.
    # In that case, do a random global split to ensure at least one eval example —
    # without this the FL server receives n_examples=0, causing divide-by-zero in
    # aggregate_evaluate, which terminates with GrpcBridgeClosed.
    if not eval_ and train:
        rng.shuffle(train)
        n_move = max(1, math.ceil(len(train) * (1 - train_ratio)))
        eval_  = train[:n_move]
        train  = train[n_move:]

    log.info(
        "Train/eval split: %d/%d examples across %d unique ICD-10 codes.",
        len(train), len(eval_), len(by_code),
    )
    return train, eval_


# ── Local evaluation ──────────────────────────────────────────────────────────

ICD10_PATTERN = re.compile(r"\b([A-Z]\d{2}[A-Z0-9]{0,4}(?:\.\d{1,2})?)\b")
RESPONSE_SEP  = "### Resposta:\n"
_DEBUG_SAMPLES = 10


def _extract_icd_codes(text: str) -> set[str]:
    """Extracts all ICD-10 codes from generated text."""
    return {c.upper() for c in ICD10_PATTERN.findall(text.upper())}


def _evaluate_local(
    model,
    tokenizer,
    examples: list[TrainingExample],
    max_length: int,
    compute_accuracy: bool = False,
    top_k: int = 5,
) -> tuple[float, int, dict[str, float]]:
    """
    Computes loss/perplexity and optionally ICD-10 extraction metrics on the
    local evaluation set.

    Accuracy metrics use beam search with top_k beams and compare against all
    admission codes (all_icd10_codes) when available.

    Args:
        model:            PeftModel in eval mode.
        tokenizer:        Corresponding tokenizer.
        examples:         Evaluation examples.
        max_length:       Maximum tokenization length.
        compute_accuracy: If True, generate predictions and compute metrics.
        top_k:            Number of beams for beam search (precision/recall@k).

    Returns:
        (avg_loss, num_examples, metrics_dict)
    """
    if not examples:
        return 0.0, 0, {"eval_loss": 0.0, "eval_perplexity": 1.0}

    dataset    = build_dataset(examples, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device     = next(model.parameters()).device

    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    labels         = labels,
                )

            num_active = (labels != -100).sum().item()
            if num_active > 0:
                total_loss   += outputs.loss.item() * num_active
                total_tokens += num_active

    avg_loss   = total_loss / max(total_tokens, 1)
    perplexity = float(torch.exp(torch.tensor(avg_loss)).item())
    metrics    = {
        "eval_loss":       round(avg_loss, 6),
        "eval_perplexity": round(perplexity, 4),
    }

    # ── Beam search extraction metrics ───────────────────────────────────────
    if compute_accuracy:
        hits_at_1  = 0
        hits_at_k  = 0
        recall_sum = 0.0
        prec_sum   = 0.0
        f1_sum     = 0.0

        for i, ex in enumerate(examples):
            # Ground truth: all admission codes (fallback: primary code only)
            gt_codes = (
                {c.upper() for c in ex.all_icd10_codes if c}
                if ex.all_icd10_codes
                else {ex.icd10_code.upper()}
            )

            prefix = ex.to_prompt().split(RESPONSE_SEP)[0] + RESPONSE_SEP
            enc = tokenizer(
                prefix,
                return_tensors      = "pt",
                truncation          = True,
                max_length          = max_length - 20,
                add_special_tokens  = True,
            ).to(device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens       = 20,
                    do_sample            = False,
                    num_beams            = top_k,
                    num_return_sequences = top_k,
                    pad_token_id         = tokenizer.pad_token_id,
                    eos_token_id         = tokenizer.eos_token_id,
                )

            input_len = enc["input_ids"].shape[1]
            beams: list[set[str]] = [
                _extract_icd_codes(tokenizer.decode(seq[input_len:], skip_special_tokens=True))
                for seq in gen_ids
            ]
            top1_codes = beams[0] if beams else set()
            topk_codes = set().union(*beams) if beams else set()

            hit_at_1 = bool(top1_codes & gt_codes)
            hit_at_k = bool(topk_codes & gt_codes)
            recall_k = len(topk_codes & gt_codes) / max(len(gt_codes), 1)
            prec_k   = len(topk_codes & gt_codes) / max(len(topk_codes), 1)
            f1_k     = (
                2 * prec_k * recall_k / max(prec_k + recall_k, 1e-9)
            )

            hits_at_1  += int(hit_at_1)
            hits_at_k  += int(hit_at_k)
            recall_sum += recall_k
            prec_sum   += prec_k
            f1_sum     += f1_k

            if i < _DEBUG_SAMPLES:
                log.info(
                    "GEN[%d] gt=%s | top1=%s | top%d=%s | R@k=%.2f F1@k=%.2f",
                    i, sorted(gt_codes)[:3], sorted(top1_codes)[:2],
                    top_k, sorted(topk_codes)[:3], recall_k, f1_k,
                )

        n = max(len(examples), 1)
        metrics["eval_icd10_accuracy"]    = round(hits_at_1 / n, 4)
        metrics[f"eval_acc_at_{top_k}"]   = round(hits_at_k / n, 4)
        metrics[f"eval_recall_at_{top_k}"]= round(recall_sum / n, 4)
        metrics[f"eval_prec_at_{top_k}"]  = round(prec_sum / n, 4)
        metrics[f"eval_f1_at_{top_k}"]    = round(f1_sum / n, 4)

        log.info(
            "ICD-10 metrics (n=%d): acc@1=%.2f%% | acc@%d=%.2f%% | "
            "R@%d=%.2f%% | P@%d=%.2f%% | F1@%d=%.2f%%",
            n,
            metrics["eval_icd10_accuracy"] * 100,
            top_k, metrics[f"eval_acc_at_{top_k}"] * 100,
            top_k, metrics[f"eval_recall_at_{top_k}"] * 100,
            top_k, metrics[f"eval_prec_at_{top_k}"] * 100,
            top_k, metrics[f"eval_f1_at_{top_k}"] * 100,
        )

    model.train()
    return avg_loss, len(examples), metrics


def _evaluate_summarization(
    model,
    tokenizer,
    examples: list,
    max_length: int,
    max_gen_tokens: int = 256,
) -> tuple[float, int, dict[str, float]]:
    """
    Generates discharge summaries and computes ROUGE-1/2/L + BERTScore(PubMedBERT).

    Called when MODEL_BACKEND=llm-summarization. Expects examples to be
    SummarizationExample objects with a `reference_summary` attribute and a
    `to_inference_prompt()` method.

    Args:
        model:          PeftModel in eval mode.
        tokenizer:      Corresponding tokenizer.
        examples:       SummarizationExample list from fhir_consumer_summarization.
        max_length:     Maximum input tokenization length.
        max_gen_tokens: Maximum new tokens to generate per summary.

    Returns:
        (proxy_loss, num_examples, metrics_dict)
        proxy_loss = 1.0 - rouge_l  (so Flower can track convergence via loss decrease)
    """
    from evaluation.summarization_metrics import evaluate_summaries

    if not examples:
        return 1.0, 0, {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0,
                        "bertscore_f1": 0.0, "eval_loss": 1.0}

    device = next(model.parameters()).device
    model.eval()
    predictions: list[str] = []
    references:  list[str] = []

    with torch.no_grad():
        for ex in examples:
            prompt = ex.to_inference_prompt()
            enc = tokenizer(
                prompt,
                return_tensors     = "pt",
                truncation         = True,
                max_length         = max_length - max_gen_tokens,
                add_special_tokens = True,
            ).to(device)

            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens = max_gen_tokens,
                    do_sample      = False,
                    num_beams      = 4,
                    pad_token_id   = tokenizer.pad_token_id,
                    eos_token_id   = tokenizer.eos_token_id,
                )

            input_len = enc["input_ids"].shape[1]
            gen_text  = tokenizer.decode(
                gen_ids[0][input_len:], skip_special_tokens=True
            ).strip()
            predictions.append(gen_text)
            references.append(ex.reference_summary)

    summ_result = evaluate_summaries(predictions, references, compute_qags=False)
    flat = summ_result.to_flat_dict()

    # proxy loss for Flower convergence tracking: higher ROUGE-L → lower loss
    proxy_loss = round(1.0 - flat.get("rouge_l", 0.0), 6)
    flat["eval_loss"] = proxy_loss

    log.info(
        "Summarization eval (n=%d): ROUGE-1=%.4f | ROUGE-2=%.4f | ROUGE-L=%.4f | "
        "BERTScore-F1=%.4f",
        len(examples),
        flat.get("rouge_1", 0.0),
        flat.get("rouge_2", 0.0),
        flat.get("rouge_l", 0.0),
        flat.get("bertscore_f1", 0.0),
    )
    return proxy_loss, len(examples), flat


def _compute_cumulative_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    total_steps: int,
    delta: float,
) -> float:
    """Recompute RDP ε for total_steps identical (σ, q) steps across all FL rounds."""
    if total_steps <= 0 or noise_multiplier <= 0:
        return float("inf")
    try:
        from opacus.accountants import RDPAccountant
        acc = RDPAccountant()
        for _ in range(total_steps):
            acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        return float(acc.get_epsilon(delta=delta))
    except Exception as exc:
        log.warning("Cumulative ε computation failed: %s", exc)
        return float("inf")


# ── NumPyClient ───────────────────────────────────────────────────────────────

class FHIRFederatedClient(NumPyClient):
    """
    Federated client that:
        1. Consumes FHIR resources (Condition + DocumentReference) from FHIR R4.
        2. Loads the quantized model with LoRA adapters (lazy, only on first call
           to avoid OOM during parameter negotiation).
        3. Executes local training with global weights received from the server.
        4. Reports perplexity, ICD-10 accuracy, or summarization metrics.

    Args:
        fhir_url:     Base URL of the FHIR R4 server.
        model_name:   HuggingFace model ID (e.g. meta-llama/...).
        partition_id: Non-IID partition to consume (-1 = all).
        max_length:   Maximum tokenization length.
        quant_cfg:    Quantization config (uses QuantizationConfig() if None).
        lora_cfg:     LoRA adapter config (uses LoRAAdapterConfig() if None).
    """

    def __init__(
        self,
        fhir_url:     str = _FHIR_URL,
        model_name:   str = _MODEL_NAME,
        partition_id: int = _PARTITION_ID,
        max_length:   int = _MAX_SEQ_LEN,
        quant_cfg: QuantizationConfig | None = None,
        lora_cfg:  LoRAAdapterConfig | None = None,
    ) -> None:
        self.fhir_url     = fhir_url
        self.model_name   = model_name
        self.partition_id = partition_id
        self.max_length   = max_length
        self.quant_cfg    = quant_cfg or QuantizationConfig()
        self.lora_cfg     = lora_cfg  or LoRAAdapterConfig()

        # Model and tokenizer — not cached between calls.
        # Loaded by _load_model() and unloaded by _unload_model()
        # inside the _gpu_lock() block at each fit/evaluate.
        self._model     = None
        self._tokenizer = None

        # Examples loaded once and split for the entire session
        self._train_examples: list | None = None
        self._eval_examples:  list | None = None

        # Backend: "llm" (generative Llama) or "bert" (PubMedBERT multi-label)
        # or "llm-summarization" (Llama for discharge summary generation)
        self._backend = _MODEL_BACKEND
        # Label index for the BERT backend (built on first _ensure_data call)
        self._label_index: dict[str, int] | None = None

        # Cumulative DP accounting across all FL rounds.
        # dp_steps is the total optimizer steps taken so far (with DP active).
        # σ, q, δ are captured from the first DP round and assumed constant.
        self._dp_total_steps:      int   = 0
        self._dp_noise_multiplier: float = 0.0
        self._dp_sample_rate:      float = 1.0
        self._dp_target_delta:     float = 1e-5

    # ── Model lifecycle management ────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Loads the model into VRAM, selecting the correct backend.

        Backend "llm" / "llm-summarization": Llama-3.x NF4 4-bit + LoRA.
        Backend "bert": PubMedBERT fp32/bf16 + lightweight LoRA (r=8).

        MUST be called exclusively inside a `_gpu_lock()` block.
        """
        if _KEEP_MODEL and self._model is not None:
            log.info("Model already in VRAM (FL_KEEP_MODEL_IN_VRAM) — reusing.")
            return

        if self._backend == "bert":
            from ai_client.model_setup_bert import BertLoRAConfig, load_bert_model
            # num_labels derived from the shared global label_index.json, filtered
            # by BERT_BENCHMARK so FL and centralised evaluate on identical label spaces.
            # top50: indices 0-49 in the global file (globally most frequent codes).
            # full:  all codes in the global file.
            _lp = os.getenv("BERT_LABEL_INDEX_PATH", "")
            if _lp and os.path.exists(_lp):
                import json as _json
                with open(_lp) as _f:
                    _full_idx = _json.load(_f)
                num_labels = 50 if _BERT_BENCHMARK == "top50" else len(_full_idx)
            else:
                num_labels = 50  # fallback for smoke tests without build-mimic
            log.info("Loading PubMedBERT (backend=bert, num_labels=%d)...", num_labels)
            model, tokenizer = load_bert_model(
                model_name = self.model_name if self.model_name != _MODEL_NAME
                             else "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                num_labels = num_labels,
            )
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            model.to(device)
        else:
            log.info("Loading model '%s' (NF4 4-bit, backend=%s)...", self.model_name, self._backend)
            model, tokenizer = load_quantized_model(
                model_name = self.model_name,
                quant_cfg  = self.quant_cfg,
            )
            model = apply_lora(model, self.lora_cfg)

        self._model     = model
        self._tokenizer = tokenizer
        log.info("Model loaded into VRAM (backend=%s).", self._backend)

    def _unload_model(self) -> None:
        """
        Removes the model from VRAM and clears the CUDA cache.

        Must be called at the end of each fit/evaluate, still inside the
        `_gpu_lock()` block, to free VRAM before releasing the lock.
        """
        if _KEEP_MODEL:
            log.info("FL_KEEP_MODEL_IN_VRAM=true — keeping model in VRAM.")
            return
        del self._model
        del self._tokenizer
        self._model     = None
        self._tokenizer = None
        gc.collect()               # force Python GC to release cyclic references
        torch.cuda.synchronize()   # wait for all pending CUDA operations
        torch.cuda.empty_cache()   # return reserved VRAM to the driver
        log.info("Model unloaded from VRAM.")

    def _ensure_data(self) -> None:
        """Fetches FHIR examples and performs the train/eval split on first call.

        Retries with exponential backoff when the FHIR R4 server is not yet
        ready (ConnectError) or when FHIR returned 0 examples (ETL still loading
        data). Waits up to 10 minutes total.

        For MODEL_BACKEND=llm-summarization, fetches SummarizationExample objects
        (with reference_summary for ROUGE/BERTScore) instead of TrainingExample.
        SummarizationExample.to_prompt() is duck-type compatible with build_dataset().
        """
        if self._train_examples is not None:
            return

        import time as _time
        _MAX_WAIT   = 600   # seconds — 10 min total
        _BASE_DELAY = 10    # seconds — initial wait
        waited      = 0

        while True:
            log.info("Fetching FHIR data from %s (partition=%d)...", self.fhir_url, self.partition_id)

            if self._backend == "llm-summarization":
                from ai_client.fhir_consumer_summarization import fetch_summarization_examples
                examples, stats = fetch_summarization_examples(self.fhir_url)
            else:
                examples, stats = fetch_training_examples(self.fhir_url)

            if stats.warnings:
                for w in stats.warnings:
                    log.warning("FHIR consumer: %s", w)

            # No examples may mean: HAPI still initializing, or ETL still loading,
            # or genuinely no data. Retry within the time budget.
            if not examples and waited < _MAX_WAIT:
                delay = min(_BASE_DELAY * (2 ** (waited // _BASE_DELAY)), 60)
                log.warning(
                    "FHIR returned 0 examples — waiting %ds before retry "
                    "(total waited: %ds/%ds)...", delay, waited, _MAX_WAIT,
                )
                _time.sleep(delay)
                waited += delay
                continue

            break

        if not examples:
            log.warning("No examples available — client will operate without local data.")
            self._train_examples = []
            self._eval_examples  = []
            return

        # Filter by partition if specified
        if self.partition_id >= 0:
            # Filter by the "partition_id=N" tag written by ETL into the Condition note.
            # Fallback: if note is empty (untagged data), include the example for
            # backward compatibility with older dataset versions.
            tag = f"partition_id={self.partition_id}"
            examples = [
                ex for ex in examples
                if tag in ex.partition_note or not ex.partition_note
            ]
            log.info("After partition %d filter: %d examples.", self.partition_id, len(examples))

        if _MAX_EXAMPLES > 0 and len(examples) > _MAX_EXAMPLES:
            import random as _random
            # Fixed seed for reproducibility across runs on the same partition.
            # Ensures the same subset is sampled regardless of when _ensure_data
            # is called, preventing unstable train/eval splits between rounds.
            _random.seed(42 + self.partition_id)
            _random.shuffle(examples)
            examples = examples[:_MAX_EXAMPLES]
            log.info("FL_MAX_EXAMPLES: capped at %d examples.", _MAX_EXAMPLES)

        self._train_examples, self._eval_examples = _stratified_split(examples)

        # BERT backend: load the global label_index.json and filter by BERT_BENCHMARK.
        # top50 → keep only the 50 globally most frequent codes (index < 50).
        # full  → keep all codes.
        # This ensures FL silos and centralised_baseline use the same code→column mapping.
        if self._backend == "bert" and self._label_index is None:
            _lp = os.getenv("BERT_LABEL_INDEX_PATH", "")
            if _lp and os.path.exists(_lp):
                import json as _json
                with open(_lp) as _f:
                    _full_idx = _json.load(_f)
                if _BERT_BENCHMARK == "top50":
                    self._label_index = {k: v for k, v in _full_idx.items() if v < 50}
                else:
                    self._label_index = _full_idx
                log.info("BERT label index: %d labels (benchmark=%s).", len(self._label_index), _BERT_BENCHMARK)
            else:
                from ai_client.fhir_consumer_bert import build_label_index
                all_examples = self._train_examples + self._eval_examples
                self._label_index = build_label_index(all_examples, benchmark=_BERT_BENCHMARK)
                log.info("BERT label index (local fallback): %d ICD-10 labels.", len(self._label_index))

    # ── NumPyClient interface ─────────────────────────────────────────────────

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """
        Returns initial LoRA adapter weights as a list of NDArrays.

        Called by the server at round 0 to obtain initial parameters.
        Loads and unloads the model inside the GPU lock to avoid
        leaving VRAM occupied during parameter negotiation.

        When FL_LORA_COMPRESS=true, parameters are TurboQuant-compressed
        before transmission (each array serialized as uint8 pickle bytes).
        """
        with _gpu_lock():
            self._load_model()
            if self._backend == "bert":
                from ai_client.model_setup_bert import get_bert_parameters
                params = get_bert_parameters(self._model)
            else:
                params = get_lora_parameters(self._model)
            log.info(
                "get_parameters: %d tensors | total size %.2f MB (backend=%s)",
                len(params),
                sum(a.nbytes for a in params) / 1024**2,
                self._backend,
            )
            self._unload_model()

        if _LORA_COMPRESS and self._backend != "bert":
            from ai_client.turbocompress import compress_parameters
            params = compress_parameters(params, n_bits=_COMPRESS_BITS)

        return params

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Receives global weights, runs one local training round, and returns
        the updated weights for the server to aggregate with DP.

        Args:
            parameters: Global LoRA weights sent by the server.
            config:     Round configuration (server_round, proximal_mu, lr, etc.).

        Returns:
            (updated_parameters, num_train_examples, metrics)
        """
        server_round = int(config.get("server_round", 0))
        proximal_mu  = float(config.get("proximal_mu", 0.01))
        learning_rate = float(config.get("learning_rate", 2e-4))
        num_epochs    = int(config.get("num_epochs", 1))

        log.info(
            "fit — round %d | lr=%.2e | μ=%.4f | epochs=%d",
            server_round, learning_rate, proximal_mu, num_epochs,
        )

        # Decompress incoming global parameters if TurboQuant is enabled
        if _LORA_COMPRESS and self._backend != "bert":
            from ai_client.turbocompress import decompress_parameters
            parameters = decompress_parameters(parameters)

        # Fetch FHIR data outside the GPU lock — does not use VRAM
        self._ensure_data()

        if not self._train_examples:
            log.warning("No training data — returning weights without update.")
            # Need to load model just to get weight structure
            with _gpu_lock():
                self._load_model()
                if self._backend == "bert":
                    from ai_client.model_setup_bert import get_bert_parameters
                    empty_params = get_bert_parameters(self._model)
                else:
                    empty_params = get_lora_parameters(self._model)
                self._unload_model()
            # Return num_examples=1 (not 0) to avoid divide-by-zero in FedProx
            # aggregation on the server (total_examples=0 → NaN weights →
            # silent crash in DPAdaptiveClipping → clients freeze).
            return empty_params, 1, {"train_loss": float("nan")}

        train_cfg = TrainingConfig(
            learning_rate        = learning_rate,
            num_epochs           = num_epochs,
            gradient_accum_steps = _GRADIENT_ACCUM_STEPS if _GRADIENT_ACCUM_STEPS > 0 else 8,
            batch_size           = _BATCH_SIZE,
            noise_multiplier     = _NOISE_MULTIPLIER,
            max_grad_norm        = _MAX_GRAD_NORM,
            target_delta         = _TARGET_DELTA,
            dp_subsample_rate    = _DP_SUBSAMPLE_RATE,
            calibrate_grad_norm  = _CALIBRATE_GRAD_NORM,
            proximal_mu          = proximal_mu,
        )

        # Load, train, and unload within the lock to serialize VRAM usage
        # between silos (NF4 4-bit model occupies ~6-7 GB)
        with _gpu_lock():
            self._load_model()

            if self._backend == "bert":
                from ai_client.model_setup_bert import (
                    BertTrainingConfig,
                    get_bert_parameters,
                    set_bert_parameters,
                    train_bert_one_round,
                )
                set_bert_parameters(self._model, parameters)
                bert_cfg = BertTrainingConfig(
                    learning_rate        = learning_rate,
                    num_epochs           = num_epochs,
                    gradient_accum_steps = _GRADIENT_ACCUM_STEPS if _GRADIENT_ACCUM_STEPS > 0 else 8,
                    batch_size           = _BATCH_SIZE,
                    proximal_mu          = proximal_mu,
                    noise_multiplier     = _NOISE_MULTIPLIER,
                    max_grad_norm        = _MAX_GRAD_NORM,
                    target_delta         = _TARGET_DELTA,
                    dp_subsample_rate    = _DP_SUBSAMPLE_RATE,
                )
                updated_params, n_examples, metrics = train_bert_one_round(
                    model        = self._model,
                    tokenizer    = self._tokenizer,
                    examples     = self._train_examples,
                    label_index  = self._label_index or {},
                    train_cfg    = bert_cfg,
                    max_length   = self.max_length,
                )
            else:
                set_lora_parameters(self._model, parameters)
                updated_params, n_examples, metrics = train_one_round(
                    model      = self._model,
                    tokenizer  = self._tokenizer,
                    examples   = self._train_examples,
                    train_cfg  = train_cfg,
                    max_length = self.max_length,
                )

            # Local evaluation with post-training weights (model still in VRAM)
            if self._eval_examples:
                if self._backend == "bert":
                    import torch
                    from torch.utils.data import DataLoader
                    from ai_client.model_setup_bert import build_bert_dataset
                    device = next(self._model.parameters()).device
                    eval_ds = build_bert_dataset(
                        self._eval_examples, self._label_index or {}, self._tokenizer,
                        self.max_length, num_labels=self._model.num_labels,
                    )
                    eval_dl = DataLoader(eval_ds, batch_size=_BATCH_SIZE, shuffle=False)
                    self._model.eval()
                    total_loss = 0.0
                    with torch.no_grad():
                        for batch in eval_dl:
                            input_ids  = batch["input_ids"].to(device)
                            attn_mask  = batch["attention_mask"].to(device)
                            labels_b   = batch["labels"].to(device)
                            out = self._model(input_ids=input_ids, attention_mask=attn_mask, labels=labels_b)
                            total_loss += out["loss"].item()
                    local_loss = total_loss / max(len(eval_dl), 1)
                    metrics["local_eval_loss"] = round(local_loss, 6)
                    log.info("Local BERT eval (post-training): bce_loss=%.4f", local_loss)
                elif self._backend == "llm-summarization":
                    local_loss, _n_eval, local_metrics = _evaluate_summarization(
                        model      = self._model,
                        tokenizer  = self._tokenizer,
                        examples   = self._eval_examples,
                        max_length = self.max_length,
                    )
                    metrics["local_eval_loss"]    = local_loss
                    metrics["local_rouge_1"]      = local_metrics.get("rouge_1", 0.0)
                    metrics["local_rouge_l"]      = local_metrics.get("rouge_l", 0.0)
                    metrics["local_bertscore_f1"] = local_metrics.get("bertscore_f1", 0.0)
                else:
                    local_loss, _n_eval, local_metrics = _evaluate_local(
                        model            = self._model,
                        tokenizer        = self._tokenizer,
                        examples         = self._eval_examples,
                        max_length       = self.max_length,
                        compute_accuracy = _EVAL_ACCURACY,
                        top_k            = _TOP_K,
                    )
                    metrics["local_eval_loss"]       = round(local_loss, 6)
                    metrics["local_eval_perplexity"] = local_metrics.get("eval_perplexity", 1.0)
                    for k, v in local_metrics.items():
                        if k.startswith("eval_icd10") or k.startswith("eval_acc_") \
                                or k.startswith("eval_recall_") or k.startswith("eval_prec_") \
                                or k.startswith("eval_f1_"):
                            metrics[f"local_{k}"] = v
                    log.info(
                        "Local eval (post-training): loss=%.4f | ppl=%.2f%s",
                        local_loss,
                        local_metrics.get("eval_perplexity", 1.0),
                        f" | acc@1={local_metrics['eval_icd10_accuracy']:.2%}"
                        if "eval_icd10_accuracy" in local_metrics else "",
                    )

            # Save LoRA checkpoint on the final round for post-training evaluation
            num_rounds = int(os.getenv("FL_NUM_ROUNDS", "5"))
            if _CHECKPOINT_DIR and server_round >= num_rounds:
                try:
                    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
                    self._model.save_pretrained(_CHECKPOINT_DIR)
                    self._tokenizer.save_pretrained(_CHECKPOINT_DIR)
                    log.info("Checkpoint saved to %s (round %d/%d)", _CHECKPOINT_DIR, server_round, num_rounds)
                except Exception as exc:
                    log.warning("Checkpoint save failed: %s", exc)

            self._unload_model()

        metrics["server_round"]  = float(server_round)
        metrics["partition_id"]  = float(self.partition_id)
        metrics["proximal_mu"]   = proximal_mu

        # Rename dp_epsilon → epsilon_spent for experiment JSON schema consistency
        if "dp_epsilon" in metrics:
            metrics["epsilon_spent"] = metrics.pop("dp_epsilon")

        # Accumulate optimizer steps and recompute cumulative ε across all rounds.
        # epsilon_spent is per-round (fresh accountant each round); epsilon_cumulative
        # is the true privacy cost reported in the paper (RDP composition over all rounds).
        dp_steps_this_round = int(metrics.pop("dp_steps_this_round", 0))
        if dp_steps_this_round > 0:
            self._dp_total_steps += dp_steps_this_round
            if self._dp_noise_multiplier == 0.0:
                self._dp_noise_multiplier = float(metrics.get("dp_noise_multiplier", 0.0))
                self._dp_sample_rate      = float(metrics.get("dp_sample_rate", 1.0))
                self._dp_target_delta     = float(metrics.get("dp_delta", 1e-5))
            epsilon_cum = _compute_cumulative_epsilon(
                self._dp_noise_multiplier,
                self._dp_sample_rate,
                self._dp_total_steps,
                self._dp_target_delta,
            )
            metrics["epsilon_cumulative"] = round(epsilon_cum, 4)
            metrics["dp_total_steps"]     = self._dp_total_steps
            log.info(
                "DP cumulative: ε=%.4f (total_steps=%d across %d rounds so far)",
                epsilon_cum, self._dp_total_steps, server_round,
            )

        log.info(
            "fit round %d | silo=%d | exemplos=%d | loss=%.4f | ppl=%.2f | "
            "lora_b_norm_end=%.6f | drift_ratio=%.2fx | "
            "local_eval_loss=%.4f | local_eval_ppl=%.2f%s",
            server_round, self.partition_id, n_examples,
            metrics.get("train_loss", float("nan")),
            metrics.get("train_perplexity", float("nan")),
            metrics.get("lora_b_norm_end", float("nan")),
            metrics.get("lora_b_drift_ratio", float("nan")),
            metrics.get("local_eval_loss", float("nan")),
            metrics.get("local_eval_perplexity", float("nan")),
            f" | local_acc={metrics['local_eval_icd10_accuracy']:.2%}"
            if "local_eval_icd10_accuracy" in metrics else "",
        )

        # Compress outgoing LoRA delta before gRPC transmission
        if _LORA_COMPRESS and self._backend != "bert":
            from ai_client.turbocompress import compress_parameters
            updated_params = compress_parameters(updated_params, n_bits=_COMPRESS_BITS)

        return updated_params, n_examples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """
        Evaluates the global model on the local validation set.

        Args:
            parameters: Global LoRA weights to evaluate.
            config:     Round configuration.

        Returns:
            (loss, num_eval_examples, metrics)
            The returned `loss` is used by the server as the global model quality
            metric — increasing loss across rounds may indicate divergence.

        For MODEL_BACKEND=bert: returns ICD-10 multi-label metrics (micro_f1, AUC-ROC, P@k).
        For MODEL_BACKEND=llm-summarization: returns ROUGE-1/2/L + BERTScore.
        For MODEL_BACKEND=llm: returns eval_loss + perplexity + optional ICD-10 accuracy.
        """
        server_round     = int(config.get("server_round", 0))
        compute_accuracy = bool(config.get("compute_accuracy", _EVAL_ACCURACY))

        log.info("evaluate — round %d | compute_accuracy=%s", server_round, compute_accuracy)

        # Decompress incoming global parameters if TurboQuant is enabled
        if _LORA_COMPRESS and self._backend != "bert":
            from ai_client.turbocompress import decompress_parameters
            parameters = decompress_parameters(parameters)

        self._ensure_data()

        if not self._eval_examples:
            log.warning("No evaluation data — returning null metrics.")
            return 0.0, 1, {"eval_loss": 0.0, "eval_perplexity": 1.0}

        with _gpu_lock():
            self._load_model()

            if self._backend == "bert":
                import torch
                from torch.utils.data import DataLoader
                from ai_client.model_setup_bert import (
                    build_bert_dataset,
                    set_bert_parameters,
                )
                from evaluation.icd_metrics import evaluate_bert_model
                set_bert_parameters(self._model, parameters)
                device = next(self._model.parameters()).device
                eval_ds = build_bert_dataset(
                    self._eval_examples, self._label_index or {}, self._tokenizer,
                    self.max_length, num_labels=self._model.num_labels,
                )
                eval_dl = DataLoader(eval_ds, batch_size=_BATCH_SIZE, shuffle=False)
                bert_result = evaluate_bert_model(self._model, eval_dl, device)
                loss     = bert_result.avg_loss
                n_examples = len(eval_ds)
                metrics  = bert_result.to_flat_dict()

            elif self._backend == "llm-summarization":
                set_lora_parameters(self._model, parameters)
                loss, n_examples, metrics = _evaluate_summarization(
                    model      = self._model,
                    tokenizer  = self._tokenizer,
                    examples   = self._eval_examples,
                    max_length = self.max_length,
                )

            else:
                set_lora_parameters(self._model, parameters)
                loss, n_examples, metrics = _evaluate_local(
                    model            = self._model,
                    tokenizer        = self._tokenizer,
                    examples         = self._eval_examples,
                    max_length       = self.max_length,
                    compute_accuracy = compute_accuracy,
                    top_k            = _TOP_K,
                )

            self._unload_model()

        metrics["server_round"] = float(server_round)
        metrics["partition_id"] = float(self.partition_id)

        log.info(
            "evaluate round %d | silo=%d | n=%d | loss=%.4f%s",
            server_round, self.partition_id, n_examples, loss,
            f" | ppl={metrics['eval_perplexity']:.2f}" if "eval_perplexity" in metrics
            else f" | rouge_l={metrics.get('rouge_l', 0.0):.4f}" if "rouge_l" in metrics
            else f" | micro_f1={metrics.get('micro_f1', 0.0):.4f}" if "micro_f1" in metrics
            else "",
        )
        return loss, n_examples, metrics

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """Reports client metadata to the server (optional)."""
        return {
            "model_name":   self.model_name,
            "partition_id": float(self.partition_id),
            "fhir_url":     self.fhir_url,
            "has_gpu":      float(torch.cuda.is_available()),
            "gpu_name":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }


# ── ClientApp (Flower 1.x modern API) ────────────────────────────────────────

def client_fn(context: Context) -> fl.client.Client:
    """
    ClientApp factory. The Flower runtime calls this function for each
    simulated client (or once per real node).

    Hyperparameters are read from `context.run_config` with fallback
    to environment variables.
    """
    run_cfg      = context.run_config if hasattr(context, "run_config") else {}
    partition_id = int(context.node_config.get("partition-id", _PARTITION_ID))

    client = FHIRFederatedClient(
        fhir_url     = str(run_cfg.get("fhir_url",    _FHIR_URL)),
        model_name   = str(run_cfg.get("model_name",  _MODEL_NAME)),
        partition_id = partition_id,
        max_length   = int(run_cfg.get("max_length",  _MAX_SEQ_LEN)),
    )
    return client.to_client()


# ClientApp — entry point for `flwr run` / SuperNode
app = ClientApp(client_fn=client_fn)


# ── Legacy entry point (start_numpy_client) ───────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )

    log.info("=== FL Client starting ===")
    log.info(
        "FHIR: %s | Flower server: %s | Partition: %d | Model: %s | Backend: %s",
        _FHIR_URL, _FL_ADDRESS, _PARTITION_ID, _MODEL_NAME, _MODEL_BACKEND,
    )
    log.info(
        "DP: σ=%.2f | δ=%s | q=%.3f | calibrate=%s",
        _NOISE_MULTIPLIER, _TARGET_DELTA, _DP_SUBSAMPLE_RATE, _CALIBRATE_GRAD_NORM,
    )
    if _NOISE_MULTIPLIER > 0:
        log.info(
            "DP-SGD active (client-side, Opacus): gradient clipping + Gaussian noise "
            "applied ONLY to LoRA adapter layers. Base model weights excluded via "
            "requires_grad=False. ε will be reported per round as 'epsilon_spent'."
        )
    else:
        log.info("DP-SGD disabled (FL_NOISE_MULTIPLIER=0).")
    if _LORA_COMPRESS:
        log.info(
            "TurboQuant LoRA compression ENABLED: %d-bit Stage-1 (Lloyd-Max) + "
            "1-bit QJL residual. Applies to get_parameters, fit, evaluate.",
            _COMPRESS_BITS,
        )
    else:
        log.info("TurboQuant compression disabled (FL_LORA_COMPRESS=false).")

    client = FHIRFederatedClient(
        fhir_url     = _FHIR_URL,
        model_name   = _MODEL_NAME,
        partition_id = _PARTITION_ID,
        max_length   = _MAX_SEQ_LEN,
    )

    start_client(
        server_address         = _FL_ADDRESS,
        client                 = client.to_client(),
        grpc_max_message_length = 512 * 1024 * 1024,   # 512 MB — LLM weights
        insecure               = True,                  # TLS should be enabled in production
    )


if __name__ == "__main__":
    main()
