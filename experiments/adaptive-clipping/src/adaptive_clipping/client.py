"""
client.py
---------
AdaptiveClippingClient — extends ai_client.fl_client.FHIRFederatedClient to
compare per-layer adaptive DP-SGD clipping ("per_layer") against the
original global-clip baseline ("baseline") without touching any file
outside experiments/adaptive-clipping/.

fit() is the only override point available: NumPyClient exposes no finer
hook, and the clip+noise logic HERALD's fit() eventually calls into lives
inside free functions (train_bert_one_round / train_one_round), not methods
— see training.py's module docstring for the full rationale. Everything in
fit() that is not the training call itself (FHIR data fetch, GPU lock,
model load/unload, local post-training evaluation, checkpoint saving, DP
epsilon bookkeeping) is copied verbatim from the parent's fit() so the two
clipping strategies are comparable on every other axis. evaluate(),
get_parameters(), and get_properties() are inherited unchanged.

Environment variables (beyond everything ai_client.fl_client already reads):
    ADAPTIVE_CLIPPING_STRATEGY   "baseline" | "per_layer" (default: baseline)
    ADAPTIVE_WARMUP_ROUNDS       FL rounds before per-layer thresholds activate (default: 3)
    ADAPTIVE_PERCENTILE          Percentile used to derive per-layer thresholds (default: 75.0)
    ADAPTIVE_MIN_CLIP            Lower clamp for per-layer thresholds (default: 0.1)
    ADAPTIVE_MAX_CLIP            Upper clamp for per-layer thresholds (default: 10.0)
    FL_SEED                      Seeds torch/numpy/random at process start (not seeded by
                                  ai_client.fl_client.main(), only by centralized_baseline.py)
"""

from __future__ import annotations

import logging
import os
import random

import flwr as fl
import numpy as np
import torch
from flwr.client import ClientApp, start_client
from flwr.common import Context, NDArrays, Scalar

from adaptive_clipping.clipping import (
    PerLayerClipper,
    assert_model_reloaded,
    param_identity_fingerprint,
)
from adaptive_clipping.training import adaptive_train_bert_one_round, adaptive_train_one_round
from evaluation.grad_norm_logger import GradNormLogger
from evaluation.metrics_logger import GPUTimer
from ai_client.fl_client import (
    _BATCH_SIZE,
    _CALIBRATE_GRAD_NORM,
    _CHECKPOINT_DIR,
    _COMPRESS_BITS,
    _DP_SUBSAMPLE_RATE,
    _EVAL_ACCURACY,
    _FHIR_URL,
    _FL_ADDRESS,
    _GRADIENT_ACCUM_STEPS,
    _LORA_COMPRESS,
    _MAX_GRAD_NORM,
    _MAX_SEQ_LEN,
    _MODEL_NAME,
    _NOISE_MULTIPLIER,
    _PARTITION_ID,
    _TARGET_DELTA,
    _TOP_K,
    FHIRFederatedClient,
    _compute_cumulative_epsilon,
    _evaluate_local,
    _evaluate_summarization,
    _gpu_lock,
    verify_effective_learning_rate,
)
from ai_client.model_setup import TrainingConfig, get_lora_parameters, set_lora_parameters

log = logging.getLogger(__name__)

_ADAPTIVE_CLIPPING_STRATEGY = os.getenv("ADAPTIVE_CLIPPING_STRATEGY", "baseline").lower()
_ADAPTIVE_WARMUP_ROUNDS = int(os.getenv("ADAPTIVE_WARMUP_ROUNDS", "3"))
_ADAPTIVE_PERCENTILE = float(os.getenv("ADAPTIVE_PERCENTILE", "75.0"))
_ADAPTIVE_MIN_CLIP = float(os.getenv("ADAPTIVE_MIN_CLIP", "0.1"))
_ADAPTIVE_MAX_CLIP = float(os.getenv("ADAPTIVE_MAX_CLIP", "10.0"))
_FL_SEED = int(os.getenv("FL_SEED", "42"))


class AdaptiveClippingClient(FHIRFederatedClient):
    """FHIRFederatedClient with a swappable clipping strategy for fit()."""

    def __init__(
        self,
        *args,
        clipping_strategy: str = "baseline",
        warmup_rounds: int = 3,
        percentile: float = 75.0,
        min_clip: float = 0.1,
        max_clip: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.clipping_strategy = clipping_strategy
        self._clipper_kwargs = dict(
            warmup_rounds=warmup_rounds, percentile=percentile,
            min_clip=min_clip, max_clip=max_clip,
        )
        self._clipper: PerLayerClipper | None = None
        # Fase 0 stale-reference guard: fingerprints self._model's parameter
        # identities right after each _load_model() call and compares against
        # the previous round's, so a regression where the model silently
        # fails to reload (the Bug 1 pattern) is caught immediately instead
        # of producing a frozen clip-threshold history. See
        # adaptive_clipping.clipping.assert_model_reloaded.
        self._prev_model_fingerprint: frozenset[int] | None = None

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        server_round = int(config.get("server_round", 0))
        proximal_mu = float(config.get("proximal_mu", 0.01))
        learning_rate = float(config.get("learning_rate", 2e-4))
        num_epochs = int(config.get("num_epochs", 1))
        verify_effective_learning_rate(server_round, learning_rate)

        log.info(
            "fit — round %d | lr=%.2e | μ=%.4f | epochs=%d | strategy=%s",
            server_round, learning_rate, proximal_mu, num_epochs, self.clipping_strategy,
        )

        if _LORA_COMPRESS and self._backend != "bert":
            from ai_client.turbocompress import decompress_parameters
            parameters = decompress_parameters(parameters)

        self._ensure_data()

        if not self._train_examples:
            log.warning("No training data — returning weights without update.")
            with _gpu_lock():
                self._load_model()
                if self._backend == "bert":
                    from ai_client.model_setup_bert import get_bert_parameters
                    empty_params = get_bert_parameters(self._model)
                else:
                    empty_params = get_lora_parameters(self._model)
                self._unload_model()
            return empty_params, 1, {"train_loss": float("nan")}

        train_cfg = TrainingConfig(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            gradient_accum_steps=_GRADIENT_ACCUM_STEPS if _GRADIENT_ACCUM_STEPS > 0 else 8,
            batch_size=_BATCH_SIZE,
            noise_multiplier=_NOISE_MULTIPLIER,
            max_grad_norm=_MAX_GRAD_NORM,
            target_delta=_TARGET_DELTA,
            dp_subsample_rate=_DP_SUBSAMPLE_RATE,
            calibrate_grad_norm=_CALIBRATE_GRAD_NORM,
            proximal_mu=proximal_mu,
        )

        with _gpu_lock():
            self._load_model()

            # Fase 0 stale-reference guard: verify _load_model() actually gave
            # us a new model object this round before doing anything else with
            # it. See clipping.assert_model_reloaded / the class docstring.
            _fp = param_identity_fingerprint(self._model.named_parameters())
            assert_model_reloaded(
                _fp, self._prev_model_fingerprint,
                component_name=f"AdaptiveClippingClient.fit(round={server_round}, silo={self.partition_id})",
            )
            self._prev_model_fingerprint = _fp

            with GPUTimer() as _timer:
                if self._backend == "bert":
                    from ai_client.model_setup_bert import (
                        BertTrainingConfig,
                        set_bert_parameters,
                        train_bert_one_round,
                    )
                    set_bert_parameters(self._model, parameters)
                    bert_cfg = BertTrainingConfig(
                        learning_rate=learning_rate,
                        num_epochs=num_epochs,
                        gradient_accum_steps=_GRADIENT_ACCUM_STEPS if _GRADIENT_ACCUM_STEPS > 0 else 8,
                        batch_size=_BATCH_SIZE,
                        proximal_mu=proximal_mu,
                        noise_multiplier=_NOISE_MULTIPLIER,
                        max_grad_norm=_MAX_GRAD_NORM,
                        target_delta=_TARGET_DELTA,
                        dp_subsample_rate=_DP_SUBSAMPLE_RATE,
                    )
                    if self.clipping_strategy == "per_layer":
                        if self._clipper is None:
                            self._clipper = PerLayerClipper(**self._clipper_kwargs)
                        updated_params, n_examples, metrics = adaptive_train_bert_one_round(
                            model=self._model, tokenizer=self._tokenizer, examples=self._train_examples,
                            label_index=self._label_index or {}, train_cfg=bert_cfg,
                            clipper=self._clipper, server_round=server_round, max_length=self.max_length,
                        )
                    else:
                        updated_params, n_examples, metrics = train_bert_one_round(
                            model=self._model, tokenizer=self._tokenizer, examples=self._train_examples,
                            label_index=self._label_index or {}, train_cfg=bert_cfg, max_length=self.max_length,
                        )
                        metrics["clipping_strategy"] = "baseline"
                else:
                    set_lora_parameters(self._model, parameters)
                    if self.clipping_strategy == "per_layer":
                        if self._clipper is None:
                            self._clipper = PerLayerClipper(**self._clipper_kwargs)
                        updated_params, n_examples, metrics = adaptive_train_one_round(
                            model=self._model, tokenizer=self._tokenizer, examples=self._train_examples,
                            train_cfg=train_cfg, clipper=self._clipper, server_round=server_round,
                            max_length=self.max_length,
                        )
                    else:
                        from ai_client.model_setup import train_one_round
                        updated_params, n_examples, metrics = train_one_round(
                            model=self._model, tokenizer=self._tokenizer, examples=self._train_examples,
                            train_cfg=train_cfg, max_length=self.max_length,
                        )
                        metrics["clipping_strategy"] = "baseline"

            metrics["wall_clock_seconds"] = _timer.elapsed_s

            if self._eval_examples:
                if self._backend == "bert":
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
                            input_ids = batch["input_ids"].to(device)
                            attn_mask = batch["attention_mask"].to(device)
                            labels_b = batch["labels"].to(device)
                            out = self._model(input_ids=input_ids, attention_mask=attn_mask, labels=labels_b)
                            total_loss += out["loss"].item()
                    local_loss = total_loss / max(len(eval_dl), 1)
                    metrics["local_eval_loss"] = round(local_loss, 6)
                    log.info("Local BERT eval (post-training): bce_loss=%.4f", local_loss)
                elif self._backend == "llm-summarization":
                    local_loss, _n_eval, local_metrics = _evaluate_summarization(
                        model=self._model, tokenizer=self._tokenizer, examples=self._eval_examples,
                        max_length=self.max_length,
                    )
                    metrics["local_eval_loss"] = local_loss
                    metrics["local_rouge_1"] = local_metrics.get("rouge_1", 0.0)
                    metrics["local_rouge_l"] = local_metrics.get("rouge_l", 0.0)
                    metrics["local_bertscore_f1"] = local_metrics.get("bertscore_f1", 0.0)
                else:
                    local_loss, _n_eval, local_metrics = _evaluate_local(
                        model=self._model, tokenizer=self._tokenizer, examples=self._eval_examples,
                        max_length=self.max_length, compute_accuracy=_EVAL_ACCURACY, top_k=_TOP_K,
                    )
                    metrics["local_eval_loss"] = round(local_loss, 6)
                    metrics["local_eval_perplexity"] = local_metrics.get("eval_perplexity", 1.0)
                    for k, v in local_metrics.items():
                        if k.startswith("eval_icd10") or k.startswith("eval_acc_") \
                                or k.startswith("eval_recall_") or k.startswith("eval_prec_") \
                                or k.startswith("eval_f1_"):
                            metrics[f"local_{k}"] = v
                    log.info(
                        "Local eval (post-training): loss=%.4f | ppl=%.2f%s",
                        local_loss, local_metrics.get("eval_perplexity", 1.0),
                        f" | acc@1={local_metrics['eval_icd10_accuracy']:.2%}"
                        if "eval_icd10_accuracy" in local_metrics else "",
                    )

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

        metrics["server_round"] = float(server_round)
        metrics["partition_id"] = float(self.partition_id)
        metrics["proximal_mu"] = proximal_mu

        if "dp_epsilon" in metrics:
            metrics["epsilon_spent"] = metrics.pop("dp_epsilon")

        dp_steps_this_round = int(metrics.pop("dp_steps_this_round", 0))
        if dp_steps_this_round > 0:
            self._dp_total_steps += dp_steps_this_round
            if self._dp_noise_multiplier == 0.0:
                self._dp_noise_multiplier = float(metrics.get("dp_noise_multiplier", 0.0))
                self._dp_sample_rate = float(metrics.get("dp_sample_rate", 1.0))
                self._dp_target_delta = float(metrics.get("dp_delta", 1e-5))
            epsilon_cum = _compute_cumulative_epsilon(
                self._dp_noise_multiplier, self._dp_sample_rate,
                self._dp_total_steps, self._dp_target_delta,
            )
            metrics["epsilon_cumulative"] = round(epsilon_cum, 4)
            metrics["dp_total_steps"] = self._dp_total_steps
            log.info(
                "DP cumulative: ε=%.4f (total_steps=%d across %d rounds so far)",
                epsilon_cum, self._dp_total_steps, server_round,
            )

        # Fase 0 instrumentation: write the per-round grad-norm JSONL record
        # (see evaluation.grad_norm_logger).
        if "grad_norms_pre_clip" in metrics:
            import json as _json
            GradNormLogger().log_round(
                server_round=server_round,
                partition_id=self.partition_id,
                grad_norms_pre_clip=_json.loads(metrics.pop("grad_norms_pre_clip")),
                grad_norms_post_clip_noise=_json.loads(metrics.pop("grad_norms_post_clip_noise")),
                clip_thresholds=_json.loads(metrics.pop("clip_threshold"))
                    if self.clipping_strategy == "per_layer" else metrics.pop("clip_threshold", None),
                learning_rate=metrics.pop("effective_lr", learning_rate),
                epsilon_cumulative=metrics.get("epsilon_cumulative"),
                noise_multiplier=_NOISE_MULTIPLIER,
                extra={"clipping_strategy": self.clipping_strategy},
            )

        log.info(
            "fit round %d | silo=%d | strategy=%s | exemplos=%d | loss=%.4f | "
            "local_eval_loss=%.4f",
            server_round, self.partition_id, self.clipping_strategy, n_examples,
            metrics.get("train_loss", float("nan")), metrics.get("local_eval_loss", float("nan")),
        )

        if _LORA_COMPRESS and self._backend != "bert":
            from ai_client.turbocompress import compress_parameters
            updated_params = compress_parameters(updated_params, n_bits=_COMPRESS_BITS)

        return updated_params, n_examples, metrics

    # evaluate(), get_parameters(), get_properties() inherited unchanged from
    # FHIRFederatedClient — the clipping strategy only affects local training.


def client_fn(context: Context) -> fl.client.Client:
    run_cfg = context.run_config if hasattr(context, "run_config") else {}
    partition_id = int(context.node_config.get("partition-id", _PARTITION_ID))

    client = AdaptiveClippingClient(
        fhir_url=str(run_cfg.get("fhir_url", _FHIR_URL)),
        model_name=str(run_cfg.get("model_name", _MODEL_NAME)),
        partition_id=partition_id,
        max_length=int(run_cfg.get("max_length", _MAX_SEQ_LEN)),
        clipping_strategy=str(run_cfg.get("clipping_strategy", _ADAPTIVE_CLIPPING_STRATEGY)),
        warmup_rounds=int(run_cfg.get("warmup_rounds", _ADAPTIVE_WARMUP_ROUNDS)),
        percentile=float(run_cfg.get("percentile", _ADAPTIVE_PERCENTILE)),
        min_clip=float(run_cfg.get("min_clip", _ADAPTIVE_MIN_CLIP)),
        max_clip=float(run_cfg.get("max_clip", _ADAPTIVE_MAX_CLIP)),
    )
    return client.to_client()


app = ClientApp(client_fn=client_fn)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # ai_client.fl_client.main() does not seed torch/numpy/random — without
    # this, the 10-seeds-per-config design of the experiment matrix would
    # have no effect on FL training (only centralized_baseline.py seeds today).
    torch.manual_seed(_FL_SEED)
    np.random.seed(_FL_SEED)
    random.seed(_FL_SEED)

    log.info("=== Adaptive Clipping FL Client starting ===")
    log.info(
        "FHIR: %s | Flower server: %s | Partition: %d | strategy=%s | seed=%d",
        _FHIR_URL, _FL_ADDRESS, _PARTITION_ID, _ADAPTIVE_CLIPPING_STRATEGY, _FL_SEED,
    )
    if _ADAPTIVE_CLIPPING_STRATEGY == "per_layer":
        log.info(
            "Per-layer clipping: warmup_rounds=%d | percentile=%.1f | min_clip=%.2f | max_clip=%.2f",
            _ADAPTIVE_WARMUP_ROUNDS, _ADAPTIVE_PERCENTILE, _ADAPTIVE_MIN_CLIP, _ADAPTIVE_MAX_CLIP,
        )

    client = AdaptiveClippingClient(
        fhir_url=_FHIR_URL,
        model_name=_MODEL_NAME,
        partition_id=_PARTITION_ID,
        max_length=_MAX_SEQ_LEN,
        clipping_strategy=_ADAPTIVE_CLIPPING_STRATEGY,
        warmup_rounds=_ADAPTIVE_WARMUP_ROUNDS,
        percentile=_ADAPTIVE_PERCENTILE,
        min_clip=_ADAPTIVE_MIN_CLIP,
        max_clip=_ADAPTIVE_MAX_CLIP,
    )

    start_client(
        server_address=_FL_ADDRESS,
        client=client.to_client(),
        grpc_max_message_length=512 * 1024 * 1024,
        insecure=True,
    )


if __name__ == "__main__":
    main()
