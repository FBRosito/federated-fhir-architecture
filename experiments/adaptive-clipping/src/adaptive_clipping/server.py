"""
server.py
---------
Thin wrapper around fl_server.server's FedProx/FedAvg strategy builder. Adds
per-round, per-silo JSONL logging (GradientNormLogger) of the clipping
thresholds/norms each client reported during fit() and the micro_f1 each
client reported during evaluate() — no aggregation, DP, or FedProx logic is
modified. build_base_strategy() is reused unchanged (build_strategy() itself
hardcodes aggregate_fit_metrics/aggregate_eval_metrics internally and can't
take a custom fit_metrics_fn without editing fl_server/server.py, which is
out of scope); only its fit_metrics_fn/eval_metrics_fn callbacks are swapped.

Flower calls fit_metrics_aggregation_fn before evaluate_metrics_aggregation_fn
within a round, but per_layer_thresholds/per_layer_norms only exist in fit()'s
metrics and micro_f1 only exists in evaluate()'s — so this module buffers each
silo's fit metrics keyed by (round, silo_id) and merges them in when that
silo's evaluate metrics for the same round arrive, before writing one
RoundLogRecord.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from adaptive_clipping.logging_utils import GradientNormLogger, RoundLogRecord
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import Strategy

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

_EXPERIMENT_TAG = os.getenv("ADAPTIVE_EXPERIMENT_TAG", "adaptive_clipping_run")
_LOGS_DIR = Path(os.getenv("ADAPTIVE_LOGS_DIR", "logs"))
_CLIPPING_STRATEGY = os.getenv("ADAPTIVE_CLIPPING_STRATEGY", "baseline").lower()
_GLOBAL_C0 = float(os.getenv("FL_MAX_GRAD_NORM", "1.0"))
_TARGET_DELTA = float(os.getenv("FL_TARGET_DELTA", "1e-5"))

# Buffers each silo's fit()-phase metrics until that (round, silo) pair's
# evaluate()-phase metrics arrive, so a single RoundLogRecord can carry both
# per_layer_thresholds/per_layer_norms (fit) and micro_f1 (evaluate).
_fit_buffer: dict[tuple[int, int], dict] = {}


def _json_field(metrics: dict, key: str) -> dict:
    raw = metrics.get(key)
    if not raw:
        return {}
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError):
        return {}


def _buffer_fit_metrics(metrics: list[tuple[int, dict]]) -> dict:
    # aggregate_fit_metrics (fl_server) does a plain float(value) weighted
    # average — it can't handle the string/JSON fields clients attach here
    # (clipping_strategy, per_layer_thresholds, per_layer_norms). Those are
    # read back from the raw per-client dicts via _fit_buffer below, not
    # from this aggregate, so it's safe to drop non-numeric values here.
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

        strategy = str(fit_metrics.get("clipping_strategy", _CLIPPING_STRATEGY))
        record = RoundLogRecord(
            round=server_round,
            silo_id=silo_id,
            sigma=NOISE_MULTIPLIER,
            clipping_strategy=strategy,
            global_c0=_GLOBAL_C0,
            per_layer_thresholds=_json_field(fit_metrics, "per_layer_thresholds"),
            per_layer_norms=_json_field(fit_metrics, "per_layer_norms"),
            micro_f1=float(client_metrics.get("micro_f1", 0.0)),
            epsilon=float(
                fit_metrics.get(
                    "epsilon_cumulative", fit_metrics.get("epsilon_spent", 0.0)
                )
            ),
            delta=_TARGET_DELTA,
            wall_clock_seconds=float(fit_metrics.get("wall_clock_seconds", 0.0)),
        )
        logger = GradientNormLogger(
            experiment_tag=_EXPERIMENT_TAG, silo_id=silo_id, logs_root=_LOGS_DIR
        )
        logger.log_round(record)
    return aggregated


def build_adaptive_strategy(
    *,
    strategy_name: str = STRATEGY_NAME,
    min_clients: int = MIN_CLIENTS,
    fraction_fit: float = FRACTION_FIT,
    fraction_eval: float = FRACTION_EVAL,
    proximal_mu: float = PROXIMAL_MU,
) -> Strategy:
    """Builds a plain FedProx/FedAvg strategy — client-side DP only, same
    aggregation as the base HERALD server — with fit/eval metrics wrapped
    for per-round JSONL logging."""
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
    """Legacy ``start_server`` entry point for the adaptive-clipping experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    log.info("=== Adaptive Clipping FL Server starting ===")
    log.info(
        "Config: strategy=%s | clipping=%s | rounds=%d | min_clients=%d | σ=%.2f | tag=%s",
        STRATEGY_NAME,
        _CLIPPING_STRATEGY,
        NUM_ROUNDS,
        MIN_CLIENTS,
        NOISE_MULTIPLIER,
        _EXPERIMENT_TAG,
    )

    strategy = build_adaptive_strategy()

    start_server(
        server_address=SERVER_ADDRESS,
        config=ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=ROUND_TIMEOUT),
        strategy=strategy,
        grpc_max_message_length=512 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
