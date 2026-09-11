"""
client.py
---------
DualLoraClient — extends ai_client.fl_client.FHIRFederatedClient with the
HERALD-PFL dual-adapter (global r=8 DP-SGD + local r=4 no-DP) training and
fused-logit evaluation, for the Article 3 "dual_lora" strategy. BERT-only
(the LLM backends are out of scope for this experiment).

Like experiments/adaptive-clipping/src/adaptive_clipping/client.py, fit()
and evaluate() are the only override points; everything not specific to the
dual-adapter mechanism (FHIR data fetch, GPU lock, model load/unload,
checkpoint saving, DP epsilon bookkeeping) is copied from the parent class
so dual_lora is comparable to dp_lora/ffa_lora on every other axis.

Environment variables (beyond everything ai_client.fl_client already reads):
    FL_LORA_MODE                 must be "dual" for this client to be meaningful
                                  (also read by ai_client.model_setup_bert.load_bert_model)
    ADAPTIVE_EXPERIMENT_TAG      experiment tag — same variable name run_dual_lora.sh
                                  already exports for the round-JSONL log directory,
                                  reused here for the local-adapter checkpoint path
    ADAPTIVE_LOGS_DIR            logs root (default: "logs")
    FL_SEED                      seeds torch/numpy/random at process start
"""

from __future__ import annotations

import logging
import os
import random

import flwr as fl
import numpy as np
import torch
from article3.dual_training import (
    _load_local_adapter,
    dual_train_bert_one_round,
    evaluate_bert_model_fused,
    local_adapter_path,
)
from flwr.client import ClientApp, start_client
from flwr.common import Context, NDArrays, Scalar

from ai_client.fl_client import (
    _BATCH_SIZE,
    _CHECKPOINT_DIR,
    _DP_SUBSAMPLE_RATE,
    _FHIR_URL,
    _FL_ADDRESS,
    _GRADIENT_ACCUM_STEPS,
    _MAX_GRAD_NORM,
    _MAX_SEQ_LEN,
    _MODEL_NAME,
    _NOISE_MULTIPLIER,
    _PARTITION_ID,
    _TARGET_DELTA,
    FHIRFederatedClient,
    _compute_cumulative_epsilon,
    _gpu_lock,
    verify_effective_learning_rate,
)
from evaluation.grad_norm_logger import GradNormLogger
from evaluation.metrics_logger import GPUTimer

log = logging.getLogger(__name__)

_EXPERIMENT_TAG = os.getenv("ADAPTIVE_EXPERIMENT_TAG", "dual_lora_run")
_LOGS_DIR = os.getenv("ADAPTIVE_LOGS_DIR", "logs")
_FL_SEED = int(os.getenv("FL_SEED", "42"))


class DualLoraClient(FHIRFederatedClient):
    """FHIRFederatedClient with HERALD-PFL dual-adapter training/evaluation. BERT-only."""

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Run one dual-adapter (global + local LoRA) training round.

        Returns:
            Flower ``fit`` triple: (updated parameters, num examples, metrics).
        """
        server_round = int(config.get("server_round", 0))
        proximal_mu = float(config.get("proximal_mu", 0.01))
        learning_rate = float(config.get("learning_rate", 2e-4))
        num_epochs = int(config.get("num_epochs", 1))
        verify_effective_learning_rate(server_round, learning_rate)

        log.info(
            "fit — round %d | lr=%.2e | μ=%.4f | epochs=%d | lora_mode=dual",
            server_round,
            learning_rate,
            proximal_mu,
            num_epochs,
        )

        self._ensure_data()

        from ai_client.model_setup_bert import BertTrainingConfig, get_bert_parameters

        if not self._train_examples:
            log.warning("No training data — returning weights without update.")
            with _gpu_lock():
                self._load_model()
                empty_params = get_bert_parameters(self._model)
                self._unload_model()
            return empty_params, 1, {"train_loss": float("nan")}

        bert_cfg = BertTrainingConfig(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            gradient_accum_steps=(
                _GRADIENT_ACCUM_STEPS if _GRADIENT_ACCUM_STEPS > 0 else 8
            ),
            batch_size=_BATCH_SIZE,
            proximal_mu=proximal_mu,
            noise_multiplier=_NOISE_MULTIPLIER,
            max_grad_norm=_MAX_GRAD_NORM,
            target_delta=_TARGET_DELTA,
            dp_subsample_rate=_DP_SUBSAMPLE_RATE,
        )
        ckpt_path = local_adapter_path(_LOGS_DIR, _EXPERIMENT_TAG, self.partition_id)

        with _gpu_lock():
            self._load_model()

            from ai_client.model_setup_bert import set_bert_parameters

            set_bert_parameters(self._model, parameters)

            with GPUTimer() as _timer:
                updated_params, n_examples, metrics = dual_train_bert_one_round(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    examples=self._train_examples,
                    label_index=self._label_index or {},
                    train_cfg=bert_cfg,
                    local_ckpt_path=ckpt_path,
                    max_length=self.max_length,
                )
            metrics["wall_clock_seconds"] = _timer.elapsed_s

            num_rounds = int(os.getenv("FL_NUM_ROUNDS", "5"))
            if _CHECKPOINT_DIR and server_round >= num_rounds:
                # NOTE: PLMICDModel is a plain nn.Module (not a HF PreTrainedModel),
                # so it has no save_pretrained() — reuse the same numpy-array
                # representation get_bert_parameters()/set_bert_parameters() already
                # use everywhere else, so cross_silo_eval.py can reload it with
                # set_bert_parameters(fresh_model, params) exactly like a client
                # receiving aggregated weights from the server.
                try:
                    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
                    np.savez(
                        os.path.join(_CHECKPOINT_DIR, "final_global_params.npz"),
                        *updated_params,
                    )
                    log.info(
                        "Final global params saved to %s (round %d/%d)",
                        _CHECKPOINT_DIR,
                        server_round,
                        num_rounds,
                    )
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
                self._dp_noise_multiplier = float(
                    metrics.get("dp_noise_multiplier", 0.0)
                )
                self._dp_sample_rate = float(metrics.get("dp_sample_rate", 1.0))
                self._dp_target_delta = float(metrics.get("dp_delta", 1e-5))
            epsilon_cum = _compute_cumulative_epsilon(
                self._dp_noise_multiplier,
                self._dp_sample_rate,
                self._dp_total_steps,
                self._dp_target_delta,
            )
            metrics["epsilon_cumulative"] = round(epsilon_cum, 4)
            metrics["dp_total_steps"] = self._dp_total_steps
            log.info(
                "DP cumulative: ε=%.4f (total_steps=%d across %d rounds so far)",
                epsilon_cum,
                self._dp_total_steps,
                server_round,
            )

        # Fase 0 instrumentation: write the per-round grad-norm JSONL record
        # (see evaluation.grad_norm_logger). Reflects the global (DP-SGD)
        # adapter pass — dual_train_bert_one_round() returns that pass's
        # metrics; the local (no-DP) pass is side-effect only.
        if "grad_norms_pre_clip" in metrics:
            import json as _json

            GradNormLogger().log_round(
                server_round=server_round,
                partition_id=self.partition_id,
                grad_norms_pre_clip=_json.loads(metrics.pop("grad_norms_pre_clip")),
                grad_norms_post_clip_noise=_json.loads(
                    metrics.pop("grad_norms_post_clip_noise")
                ),
                clip_thresholds=metrics.pop("clip_threshold", None),
                learning_rate=metrics.pop("effective_lr", learning_rate),
                epsilon_cumulative=metrics.get("epsilon_cumulative"),
                noise_multiplier=_NOISE_MULTIPLIER,
                extra={"lora_mode": "dual"},
            )

        log.info(
            "fit round %d | silo=%d | lora_mode=dual | exemplos=%d | loss=%.4f",
            server_round,
            self.partition_id,
            n_examples,
            metrics.get("train_loss", float("nan")),
        )

        return updated_params, n_examples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the global adapter on this silo's held-out data.

        Returns:
            Flower ``evaluate`` triple: (loss, num examples, metrics).
        """
        server_round = int(config.get("server_round", 0))
        log.info("evaluate — round %d | lora_mode=dual", server_round)

        self._ensure_data()

        if not self._eval_examples:
            log.warning("No evaluation data — returning null metrics.")
            return 0.0, 1, {"eval_loss": 0.0}

        ckpt_path = local_adapter_path(_LOGS_DIR, _EXPERIMENT_TAG, self.partition_id)

        with _gpu_lock():
            self._load_model()

            from torch.utils.data import DataLoader

            from ai_client.model_setup_bert import (
                build_bert_dataset,
                set_bert_parameters,
            )

            set_bert_parameters(self._model, parameters)
            _load_local_adapter(self._model, ckpt_path)

            device = next(self._model.parameters()).device
            eval_ds = build_bert_dataset(
                self._eval_examples,
                self._label_index or {},
                self._tokenizer,
                self.max_length,
                num_labels=self._model.num_labels,
            )
            eval_dl = DataLoader(eval_ds, batch_size=_BATCH_SIZE, shuffle=False)
            fused_metrics, global_metrics, local_metrics = evaluate_bert_model_fused(
                self._model,
                eval_dl,
                device,
            )
            n_examples = len(eval_ds)

            self._unload_model()

        loss = fused_metrics.avg_loss
        metrics = fused_metrics.to_flat_dict()
        metrics["micro_f1_global"] = global_metrics.micro_f1
        metrics["micro_f1_local"] = local_metrics.micro_f1
        metrics["server_round"] = float(server_round)
        metrics["partition_id"] = float(self.partition_id)

        log.info(
            "evaluate round %d | silo=%d | n=%d | fused micro_f1=%.4f",
            server_round,
            self.partition_id,
            n_examples,
            metrics.get("micro_f1", 0.0),
        )
        return loss, n_examples, metrics


def client_fn(context: Context) -> fl.client.Client:
    """ClientApp factory: build this silo's dual-LoRA client from run/node config."""
    run_cfg = context.run_config if hasattr(context, "run_config") else {}
    partition_id = int(context.node_config.get("partition-id", _PARTITION_ID))

    client = DualLoraClient(
        fhir_url=str(run_cfg.get("fhir_url", _FHIR_URL)),
        model_name=str(run_cfg.get("model_name", _MODEL_NAME)),
        partition_id=partition_id,
        max_length=int(run_cfg.get("max_length", _MAX_SEQ_LEN)),
    )
    return client.to_client()


app = ClientApp(client_fn=client_fn)


def main() -> None:
    """Legacy ``start_numpy_client`` entry point (SuperNode uses the ClientApp)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    torch.manual_seed(_FL_SEED)
    np.random.seed(_FL_SEED)
    random.seed(_FL_SEED)

    log.info("=== Dual-LoRA FL Client starting ===")
    log.info(
        "FHIR: %s | Flower server: %s | Partition: %d | seed=%d",
        _FHIR_URL,
        _FL_ADDRESS,
        _PARTITION_ID,
        _FL_SEED,
    )

    client = DualLoraClient(
        fhir_url=_FHIR_URL,
        model_name=_MODEL_NAME,
        partition_id=_PARTITION_ID,
        max_length=_MAX_SEQ_LEN,
    )

    start_client(
        server_address=_FL_ADDRESS,
        client=client.to_client(),
        grpc_max_message_length=512 * 1024 * 1024,
        insecure=True,
    )


if __name__ == "__main__":
    main()
