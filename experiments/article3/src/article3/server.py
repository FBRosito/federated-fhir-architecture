"""
server.py
---------
Thin wrapper around fl_server.server's FedProx/FedAvg strategy builder,
analogous to experiments/adaptive-clipping/src/adaptive_clipping/server.py
(same buffering pattern, since fit_metrics_aggregation_fn runs before
evaluate_metrics_aggregation_fn within a round and micro_f1/micro_f1_global/
micro_f1_local only exist in evaluate()'s metrics while epsilon_cumulative/
wall_clock_seconds only exist in fit()'s). No aggregation, DP, or FedProx
logic is modified — only the fit_metrics_fn/eval_metrics_fn callbacks passed
to build_base_strategy() are swapped, same as adaptive_clipping does.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flwr.server import ServerConfig, start_server

from article3.logging_utils import RoundLogger, RoundLogRecord
from fl_server.server import (
    FRACTION_EVAL,
    FRACTION_FIT,
    MIN_CLIENTS,
    NOISE_MULTIPLIER,
    NUM_ROUNDS,
    PROXIMAL_MU,
    ROUND_TIMEOUT,
    SERVER_ADDRESS,
    STRATEGY_NAME,
    aggregate_eval_metrics,
    aggregate_fit_metrics,
    build_base_strategy,
)

log = logging.getLogger(__name__)

_EXPERIMENT_TAG = os.getenv("ADAPTIVE_EXPERIMENT_TAG", "dual_lora_run")
_LOGS_DIR = Path(os.getenv("ADAPTIVE_LOGS_DIR", "logs"))
_LORA_MODE = os.getenv("FL_LORA_MODE", "dual").lower()
_GLOBAL_C0 = float(os.getenv("FL_MAX_GRAD_NORM", "1.0"))
_TARGET_DELTA = float(os.getenv("FL_TARGET_DELTA", "1e-5"))

_fit_buffer: dict[tuple[int, int], dict] = {}


def _buffer_fit_metrics(metrics: list[tuple[int, dict]]) -> dict:
    numeric_metrics = [
        (n, {k: v for k, v in m.items() if isinstance(v, (int, float))})
        for n, m in metrics
    ]
    aggregated = aggregate_fit_metrics(numeric_metrics)
    for _n_examples, client_metrics in metrics:
        server_round = int(client_metrics.get("server_round", 0))
        silo_id = int(client_metrics.get("partition_id", -1))
        _fit_buffer[(server_round, silo_id)] = client_metrics
    return aggregated


def _flush_eval_metrics(metrics: list[tuple[int, dict]]) -> dict:
    aggregated = aggregate_eval_metrics(metrics)
    for _n_examples, client_metrics in metrics:
        server_round = int(client_metrics.get("server_round", 0))
        silo_id = int(client_metrics.get("partition_id", -1))
        fit_metrics = _fit_buffer.pop((server_round, silo_id), {})

        record = RoundLogRecord(
            round=server_round,
            silo_id=silo_id,
            sigma=NOISE_MULTIPLIER,
            lora_mode=str(fit_metrics.get("lora_mode", _LORA_MODE)),
            global_c0=_GLOBAL_C0,
            micro_f1=float(client_metrics.get("micro_f1", 0.0)),
            micro_f1_global=float(client_metrics.get("micro_f1_global", 0.0)),
            micro_f1_local=float(client_metrics.get("micro_f1_local", 0.0)),
            epsilon=float(fit_metrics.get("epsilon_cumulative", fit_metrics.get("epsilon_spent", 0.0))),
            delta=_TARGET_DELTA,
            wall_clock_seconds=float(fit_metrics.get("wall_clock_seconds", 0.0)),
        )
        logger = RoundLogger(experiment_tag=_EXPERIMENT_TAG, silo_id=silo_id, logs_root=_LOGS_DIR)
        logger.log_round(record)
    return aggregated


def build_dual_lora_strategy(
    *,
    strategy_name: str = STRATEGY_NAME,
    min_clients: int = MIN_CLIENTS,
    fraction_fit: float = FRACTION_FIT,
    fraction_eval: float = FRACTION_EVAL,
    proximal_mu: float = PROXIMAL_MU,
):
    return build_base_strategy(
        strategy_name=strategy_name,
        min_clients=min_clients,
        fraction_fit=fraction_fit,
        fraction_eval=fraction_eval,
        proximal_mu=proximal_mu,
        fit_metrics_fn=_buffer_fit_metrics,
        eval_metrics_fn=_flush_eval_metrics,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    log.info("=== Dual-LoRA FL Server starting ===")
    log.info(
        "Config: strategy=%s | lora_mode=%s | rounds=%d | min_clients=%d | σ=%.2f | tag=%s",
        STRATEGY_NAME, _LORA_MODE, NUM_ROUNDS, MIN_CLIENTS, NOISE_MULTIPLIER, _EXPERIMENT_TAG,
    )

    strategy = build_dual_lora_strategy()

    start_server(
        server_address=SERVER_ADDRESS,
        config=ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=ROUND_TIMEOUT),
        strategy=strategy,
        grpc_max_message_length=512 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
