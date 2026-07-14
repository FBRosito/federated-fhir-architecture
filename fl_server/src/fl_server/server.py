"""
server.py
---------
Flower orchestrator for the Federated Learning pipeline over FHIR data.

Strategy: FedProx (more robust than FedAvg for Non-IID data).
This server is a CLEAN aggregator — Differential Privacy is applied CLIENT-SIDE
by each silo via Opacus (gradient clipping + Gaussian noise on LoRA layers only,
before weights leave the edge node). The server receives already-privatized LoRA
deltas and aggregates them with FedProx weighted averaging.

Why FedProx for Non-IID?
    Our data is deliberately heterogeneous (partitions by medical specialty).
    FedProx introduces a proximal term μ||w - w_global||² that penalises
    clients that deviate too far from the global model, stabilising convergence.
    The proximal term applies only to LoRA parameters {B_l, A_l}: since base
    weights W_0 are frozen (requires_grad=False), ||θ_total - θ_t||² reduces
    to ||θ_LoRA - θ_LoRA_t||² by construction.

Why client-side DP (not server-side)?
    Client-side DP (local DP) ensures each patient's gradient is privatized
    before leaving the edge node — a strictly stronger guarantee than
    server-side post-aggregation noise. The FL server only sees noised deltas
    and cannot reconstruct individual patient records.

Threat model mitigated:
    - Gradient Inversion Attacks (Zhu et al. 2019): DP noise on each LoRA
      gradient and low-rank structure (rank r ≪ d) prevent exact reconstruction.
    - Membership Inference: (ε, δ)-DP guarantee computed via RDP accountant
      on the client; ε_spent is reported per round in experiment JSON logs.

Environment variables:
    FL_SERVER_ADDRESS      gRPC listen address (default: [::]:9091)
    FL_NETWORK_MODE        "simulated" (insecure gRPC) | anything else = "real" (TLS)
    FL_CA_CERT_PATH        CA cert — 1st element of the TLS certificate chain (real mode)
    FL_SERVER_CERT_PATH    Server certificate (real mode)
    FL_SERVER_KEY_PATH     Server private key (real mode)
    FL_NUM_ROUNDS          Federated training rounds (default: 5)
    FL_MIN_CLIENTS         Minimum clients to start each round (default: 2)
    FL_STRATEGY            "fedprox" | "fedavg" (default: fedprox)
    FL_PROXIMAL_MU         FedProx proximal coefficient μ (default: 0.01)
    FL_NOISE_MULTIPLIER    DP noise multiplier σ/C (default: 0.9)
    FL_INITIAL_CLIP_NORM   Initial clipping norm (default: 1.0)
    FL_CLIP_NORM_TARGET_Q  Adaptive clipping target quantile (default: 0.5)
    FL_FRACTION_FIT        Fraction of clients used per training round (default: 1.0)
    FL_FRACTION_EVAL       Fraction of clients used per eval round (default: 1.0)
    FL_NUM_EPOCHS          Local training epochs per round (default: 1)
"""

from __future__ import annotations

import logging
import math
import os

import flwr as fl
import numpy as np
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, start_server
from flwr.server.strategy import FedAvg, FedProx, Strategy
from flwr.server.strategy import (
    DifferentialPrivacyServerSideAdaptiveClipping as DPAdaptiveClipping,
)
from flwr.server.strategy import (
    DifferentialPrivacyServerSideFixedClipping as DPFixedClipping,
)

log = logging.getLogger(__name__)

# ── Configuration from environment ───────────────────────────────────────────

def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))

def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))

def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


SERVER_ADDRESS      = _env_str("FL_SERVER_ADDRESS", "[::]:9091")
NETWORK_MODE        = _env_str("FL_NETWORK_MODE", "real").strip().lower()
IS_SIMULATED        = NETWORK_MODE == "simulated"
CA_CERT_PATH        = _env_str("FL_CA_CERT_PATH", "")
SERVER_CERT_PATH    = _env_str("FL_SERVER_CERT_PATH", "")
SERVER_KEY_PATH     = _env_str("FL_SERVER_KEY_PATH", "")
ROUND_TIMEOUT       = _env_float("FL_ROUND_TIMEOUT", 3600.0)   # 1h — waits for clients
NUM_ROUNDS          = _env_int("FL_NUM_ROUNDS", 5)
MIN_CLIENTS         = _env_int("FL_MIN_CLIENTS", 2)
STRATEGY_NAME       = _env_str("FL_STRATEGY", "fedprox").lower()
PROXIMAL_MU         = _env_float("FL_PROXIMAL_MU", 0.01)
NOISE_MULTIPLIER    = _env_float("FL_NOISE_MULTIPLIER", 0.9)
INITIAL_CLIP_NORM   = _env_float("FL_INITIAL_CLIP_NORM", 1.0)
CLIP_NORM_TARGET_Q  = _env_float("FL_CLIP_NORM_TARGET_Q", 0.5)
FRACTION_FIT        = _env_float("FL_FRACTION_FIT", 1.0)
FRACTION_EVAL       = _env_float("FL_FRACTION_EVAL", 1.0)
# Learning rate sent to clients in fit_config; overridable per round.
# In fast-dev mode with few examples, use smaller values (e.g. 5e-6) to avoid
# destructive fine-tuning that overwrites the base model's pre-trained knowledge.
LEARNING_RATE       = _env_float("FL_LEARNING_RATE", 5e-5)
EVAL_ACCURACY       = os.getenv("FL_EVAL_ACCURACY", "false").lower() == "true"
NUM_EPOCHS          = _env_int("FL_NUM_EPOCHS", 1)

# ── Privacy budget estimation (Gaussian Mechanism approximation) ──────────────

def estimate_privacy_budget(
    noise_multiplier: float,
    num_rounds: int,
    num_clients_per_round: int,
    total_clients: int,
    delta: float = 1e-5,
) -> float:
    """
    Estimates ε spend via simple sequential composition of the Gaussian mechanism
    over training rounds.

    WARNING: This is a conservative upper bound. For exact accounting use
    `autodp` (Rényi DP) or `opacus` (PRV accountant).

    Args:
        noise_multiplier:        σ / C (noise-to-sensitivity ratio).
        num_rounds:              Number of federated rounds.
        num_clients_per_round:   Clients sampled per round.
        total_clients:           Total clients in the pool.
        delta:                   DP failure probability.

    Returns:
        Estimated total ε consumed.
    """
    if noise_multiplier <= 0:
        return float("inf")

    sampling_rate = num_clients_per_round / max(total_clients, 1)
    # Per-round bound via approximate analytical Gaussian mechanism
    epsilon_per_round = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier
    # Simple sequential composition (conservative): ε_total ≈ T * ε_round * q
    epsilon_total = num_rounds * epsilon_per_round * sampling_rate
    return round(epsilon_total, 4)


# ── NaN filter for aggregation ────────────────────────────────────────────────

def _has_nan(parameters) -> bool:
    return any(not np.all(np.isfinite(a)) for a in parameters_to_ndarrays(parameters))


class NaNSafeMixin:
    """Drops clients with NaN/Inf weights before delegating aggregation."""
    def aggregate_fit(self, server_round, results, failures):
        valid = [(c, r) for c, r in results if not _has_nan(r.parameters)]
        dropped = len(results) - len(valid)
        if dropped:
            log.warning(
                "Round %d: %d/%d clients dropped (NaN/Inf in weights).",
                server_round, dropped, len(results),
            )
        if not valid:
            log.error("Round %d: all clients have NaN — round skipped.", server_round)
            return None, {}
        return super().aggregate_fit(server_round, valid, failures)


class NaNSafeFedAvg(NaNSafeMixin, FedAvg): pass
class NaNSafeFedProx(NaNSafeMixin, FedProx): pass


class PartialResultsDPStrategy(DPAdaptiveClipping):
    """Adaptive DP strategy tolerant of partial client failures.

    Flower's native implementation aborts the entire aggregation on any gRPC
    failure (line `if failures: return None, {}`). This subclass filters NaN
    and failures BEFORE passing to the DP mechanism, allowing rounds with
    absent or corrupted clients to still produce a valid aggregated model.

    Trade-off: with fewer clients than num_sampled_clients, the DP noise level
    is conservatively calibrated to the original count — the (ε,δ) guarantee
    remains valid (excess noise never violates DP), but may be noisier than
    necessary when all clients are present.
    """

    def aggregate_fit(self, server_round, results, failures):
        # Filter NaN/Inf weights from received results
        valid = [(c, r) for c, r in results if not _has_nan(r.parameters)]
        nan_dropped = len(results) - len(valid)
        if nan_dropped:
            log.warning(
                "Round %d: %d/%d results with NaN/Inf dropped before DP.",
                server_round, nan_dropped, len(results),
            )
        if failures:
            log.warning(
                "Round %d: %d gRPC failure(s) ignored — "
                "proceeding with %d valid result(s).",
                server_round, len(failures), len(valid),
            )
        if not valid:
            log.error(
                "Round %d: no valid results after filters — round skipped.",
                server_round,
            )
            return None, {}
        # Pass failures=[] so DPAdaptiveClipping does not abort
        return super().aggregate_fit(server_round, valid, [])


# ── Strategy builders ─────────────────────────────────────────────────────────

def build_base_strategy(
    strategy_name: str,
    min_clients: int,
    fraction_fit: float,
    fraction_eval: float,
    proximal_mu: float,
    fit_metrics_fn,
    eval_metrics_fn,
) -> Strategy:
    """Instantiates FedProx or FedAvg according to `strategy_name`."""
    common_kwargs = dict(
        fraction_fit                  = fraction_fit,
        fraction_evaluate             = fraction_eval,
        min_fit_clients               = min_clients,
        min_evaluate_clients          = max(1, min_clients // 2),
        min_available_clients         = min_clients,
        on_fit_config_fn              = make_fit_config_fn(proximal_mu, NUM_EPOCHS),
        on_evaluate_config_fn         = make_eval_config_fn(),
        fit_metrics_aggregation_fn    = fit_metrics_fn,
        evaluate_metrics_aggregation_fn = eval_metrics_fn,
        accept_failures               = True,
        # initial_parameters=None: Flower will request initial parameters
        # from the first available client in round 0
        initial_parameters            = None,
    )

    if strategy_name == "fedprox":
        log.info("Base strategy: NaNSafeFedProx (μ=%.4f)", proximal_mu)
        return NaNSafeFedProx(proximal_mu=proximal_mu, **common_kwargs)

    log.info("Base strategy: NaNSafeFedAvg")
    return NaNSafeFedAvg(**common_kwargs)


def wrap_with_dp(
    base_strategy: Strategy,
    num_sampled_clients: int,
    noise_multiplier: float,
    initial_clip_norm: float,
    target_quantile: float,
) -> PartialResultsDPStrategy:
    """
    Wraps the base strategy with server-side adaptive clipping DP.

    Uses PartialResultsDPStrategy to tolerate partial client failures:
    DP does not abort when a client fails or sends NaN weights.

    Mechanism:
        1. Upon receiving each client's parameters, the server clips them
           to L2 norm C_t (updated adaptively each round).
        2. After aggregation (FedAvg/FedProx), adds Gaussian noise
           N(0, σ²I) with σ = noise_multiplier × C_t to the aggregated vector.
        3. C_t is adjusted so that fraction `target_quantile` of client norms
           falls below the threshold — no prior knowledge of the update
           distribution required.

    Args:
        base_strategy:       Already-configured base strategy.
        num_sampled_clients: Clients sampled per round (affects σ_count).
        noise_multiplier:    σ / C — noise-to-sensitivity ratio.
        initial_clip_norm:   Initial clipping norm C_0.
        target_quantile:     Clipping target quantile (0.5 = median).

    Returns:
        Strategy with adaptive DP configured.
    """
    # clipped_count_stddev: std dev of noise on the count of clipped clients —
    # must be large enough for the effective noise_multiplier of the counting
    # mechanism to be achievable. Safe rule of thumb: sqrt(num_sampled_clients)
    clipped_count_stddev = max(1.0, math.sqrt(num_sampled_clients))

    dp_strategy = PartialResultsDPStrategy(
        strategy              = base_strategy,
        noise_multiplier      = noise_multiplier,
        num_sampled_clients   = num_sampled_clients,
        initial_clipping_norm = initial_clip_norm,
        target_clipped_quantile = target_quantile,
        clip_norm_lr          = 0.2,
        clipped_count_stddev  = clipped_count_stddev,
    )
    log.info(
        "Adaptive DP configured: σ=%.2f, C_0=%.2f, q_target=%.2f, "
        "σ_count=%.2f, clients/round=%d",
        noise_multiplier, initial_clip_norm, target_quantile,
        clipped_count_stddev, num_sampled_clients,
    )
    return dp_strategy


# ── Per-round configuration callbacks ────────────────────────────────────────

def make_fit_config_fn(proximal_mu: float, num_epochs: int = 1):
    """
    Returns a function that generates the training configuration sent to each
    client at the start of each `fit` round.

    The `proximal_mu` field is forwarded to the client so it can apply the
    correct proximal term even without knowing the global server configuration.
    """
    def fit_config(server_round: int) -> dict[str, Scalar]:
        # Base LR from FL_LEARNING_RATE; later rounds use 40% of initial value.
        # With few examples (fast-dev), set FL_LEARNING_RATE=5e-6 in docker-compose
        # or Makefile to prevent LoRA from destroying the base model's knowledge.
        lr = LEARNING_RATE if server_round <= 2 else LEARNING_RATE * 0.4
        config: dict[str, Scalar] = {
            "server_round":  server_round,
            "proximal_mu":   proximal_mu,
            "learning_rate": lr,
            "num_epochs":    num_epochs,
        }
        log.debug("fit_config round %d: %s", server_round, config)
        return config
    return fit_config


def make_eval_config_fn():
    """Returns a function that generates the evaluation configuration per round."""
    def eval_config(server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
            "compute_accuracy": EVAL_ACCURACY,
        }
    return eval_config


# ── Metrics aggregation ───────────────────────────────────────────────────────

def aggregate_fit_metrics(
    metrics: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    """
    Aggregates training metrics reported by clients (weighted average
    by the number of examples used by each client).
    """
    if not metrics:
        return {}
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    for n, m in metrics:
        weight = n / total_examples
        for key, value in m.items():
            aggregated[key] = aggregated.get(key, 0.0) + float(value) * weight

    log.info(
        "Aggregated fit metrics (%d clients): %s",
        len(metrics),
        {k: f"{v:.4f}" for k, v in aggregated.items()},
    )
    return {k: round(v, 6) for k, v in aggregated.items()}


def aggregate_eval_metrics(
    metrics: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    """Aggregates evaluation metrics (weighted average by examples)."""
    if not metrics:
        return {}
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    for n, m in metrics:
        weight = n / total_examples
        for key, value in m.items():
            aggregated[key] = aggregated.get(key, 0.0) + float(value) * weight

    log.info(
        "Aggregated eval metrics (%d clients): %s",
        len(metrics),
        {k: f"{v:.4f}" for k, v in aggregated.items()},
    )
    return {k: round(v, 6) for k, v in aggregated.items()}


# ── Full strategy builder ─────────────────────────────────────────────────────

def build_strategy(
    strategy_name: str  = STRATEGY_NAME,
    min_clients: int    = MIN_CLIENTS,
    num_rounds: int     = NUM_ROUNDS,
    fraction_fit: float = FRACTION_FIT,
    fraction_eval: float = FRACTION_EVAL,
    proximal_mu: float  = PROXIMAL_MU,
    noise_multiplier: float = NOISE_MULTIPLIER,
    initial_clip_norm: float = INITIAL_CLIP_NORM,
    target_quantile: float  = CLIP_NORM_TARGET_Q,
) -> Strategy:
    """
    Builds the base strategy (FedProx|FedAvg) without server-side DP.

    DP is applied CLIENT-SIDE by each silo (Opacus, LoRA layers only).
    The server receives already-privatized LoRA deltas.
    FL_NOISE_MULTIPLIER is logged here for documentation; it has no effect
    on server aggregation — it controls client-side noise in fl_client.py.
    """
    num_sampled = max(1, round(min_clients * fraction_fit))

    strategy = build_base_strategy(
        strategy_name  = strategy_name,
        min_clients    = min_clients,
        fraction_fit   = fraction_fit,
        fraction_eval  = fraction_eval,
        proximal_mu    = proximal_mu,
        fit_metrics_fn = aggregate_fit_metrics,
        eval_metrics_fn = aggregate_eval_metrics,
    )

    if noise_multiplier > 0:
        log.info(
            "Client-side DP active: σ=%.2f (applied in each silo via Opacus). "
            "Server aggregates already-privatized LoRA deltas without additional noise.",
            noise_multiplier,
        )
        eps = estimate_privacy_budget(
            noise_multiplier        = noise_multiplier,
            num_rounds              = num_rounds,
            num_clients_per_round   = num_sampled,
            total_clients           = min_clients,
            delta                   = 1e-5,
        )
        log.info(
            "Estimated privacy budget (δ=1e-5): ε ≈ %.4f "
            "over %d rounds with σ=%.2f [conservative upper bound — "
            "exact ε reported per-round by clients via RDP accountant]",
            eps, num_rounds, noise_multiplier,
        )
        if eps > 10.0:
            log.warning(
                "ε=%.2f is high — weak privacy. Consider increasing "
                "FL_NOISE_MULTIPLIER or reducing FL_NUM_ROUNDS.",
                eps,
            )
    else:
        log.info("FL_NOISE_MULTIPLIER=0 — DP disabled.")

    return strategy


# ── ServerApp (Flower 1.x modern API) ────────────────────────────────────────

def server_fn(context) -> ServerAppComponents:
    """
    ServerApp factory function. Called by the Flower runtime on startup.

    Reads hyperparameters from `context.run_config` (defined in the run's
    `pyproject.toml` if present) with fallback to environment variables.
    """
    run_cfg = context.run_config if hasattr(context, "run_config") else {}

    strategy = build_strategy(
        strategy_name    = str(run_cfg.get("strategy",        STRATEGY_NAME)),
        min_clients      = int(run_cfg.get("min_clients",     MIN_CLIENTS)),
        num_rounds       = int(run_cfg.get("num_rounds",      NUM_ROUNDS)),
        fraction_fit     = float(run_cfg.get("fraction_fit",  FRACTION_FIT)),
        fraction_eval    = float(run_cfg.get("fraction_eval", FRACTION_EVAL)),
        proximal_mu      = float(run_cfg.get("proximal_mu",   PROXIMAL_MU)),
        noise_multiplier = float(run_cfg.get("noise_multiplier", NOISE_MULTIPLIER)),
        initial_clip_norm = float(run_cfg.get("initial_clip_norm", INITIAL_CLIP_NORM)),
        target_quantile  = float(run_cfg.get("target_quantile", CLIP_NORM_TARGET_Q)),
    )

    config = ServerConfig(
        num_rounds    = int(run_cfg.get("num_rounds", NUM_ROUNDS)),
        round_timeout = float(run_cfg.get("round_timeout", 300.0)),
    )

    return ServerAppComponents(strategy=strategy, config=config)


# ServerApp — entry point for `flwr run` / SuperLink
app = ServerApp(server_fn=server_fn)


# ── TLS certificates (FL_NETWORK_MODE=real) ──────────────────────────────────

def _load_server_certificates() -> tuple[bytes, bytes, bytes] | None:
    if not (CA_CERT_PATH and SERVER_CERT_PATH and SERVER_KEY_PATH):
        return None
    with open(CA_CERT_PATH, "rb") as f:
        ca_cert = f.read()
    with open(SERVER_CERT_PATH, "rb") as f:
        server_cert = f.read()
    with open(SERVER_KEY_PATH, "rb") as f:
        server_key = f.read()
    return ca_cert, server_cert, server_key


# ── Legacy entry point (start_server) ────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )

    log.info("=== FL Server starting ===")
    log.info(
        "Config: strategy=%s | rounds=%d | epochs=%d | min_clients=%d | "
        "σ=%.2f | C_0=%.2f | μ=%.4f",
        STRATEGY_NAME, NUM_ROUNDS, NUM_EPOCHS, MIN_CLIENTS,
        NOISE_MULTIPLIER, INITIAL_CLIP_NORM, PROXIMAL_MU,
    )
    log.info(
        "Privacy architecture: DP applied CLIENT-SIDE (Opacus on LoRA layers). "
        "Server is a clean FedProx aggregator — no server-side noise injection. "
        "Per-round ε reported by clients via RDP accountant as 'epsilon_spent'."
    )

    strategy = build_strategy()

    log.info("FL network mode: %s", "SIMULATED (insecure loopback)" if IS_SIMULATED else "REAL (TLS)")

    if IS_SIMULATED:
        history = start_server(
            server_address         = SERVER_ADDRESS,
            config                 = ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=ROUND_TIMEOUT),
            strategy               = strategy,
            grpc_max_message_length = 512 * 1024 * 1024,  # 512 MB — LLM weights are large
        )
    else:
        history = start_server(
            server_address         = SERVER_ADDRESS,
            config                 = ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=ROUND_TIMEOUT),
            strategy               = strategy,
            grpc_max_message_length = 512 * 1024 * 1024,  # 512 MB — LLM weights are large
            certificates            = _load_server_certificates(),
        )

    _print_training_summary(history, NUM_ROUNDS)


def _print_training_summary(history, num_rounds: int) -> None:
    """Prints a detailed summary of all rounds at the end of federated training."""
    SEP = "=" * 72
    log.info(SEP)
    log.info("FEDERATED TRAINING FINAL SUMMARY")
    completed = len(history.losses_distributed) if history.losses_distributed else 0
    log.info("Rounds completed : %d / %d", completed, num_rounds)
    log.info(SEP)

    if history.losses_distributed:
        log.info("── Global loss (evaluate) per round ──")
        for rnd, loss in sorted(history.losses_distributed):
            log.info("  Round %2d | loss=%.6f", rnd, loss)

    if history.metrics_distributed_fit:
        all_rounds   = sorted({r for vals in history.metrics_distributed_fit.values() for r, _ in vals})
        metric_keys  = sorted(history.metrics_distributed_fit.keys())
        col_w        = max(22, max(len(k) for k in metric_keys) + 2)
        header       = "  ".join(f"{k:>{col_w}}" for k in metric_keys)
        log.info("── Training metrics (fit) per round ──")
        log.info("  %-8s  %s", "Round", header)
        for rnd in all_rounds:
            row = {k: next((v for r, v in history.metrics_distributed_fit[k] if r == rnd), float("nan"))
                   for k in metric_keys}
            vals_str = "  ".join(f"{float(row[k]):>{col_w}.6f}" for k in metric_keys)
            log.info("  %-8d  %s", rnd, vals_str)

    if history.metrics_distributed:
        all_rounds  = sorted({r for vals in history.metrics_distributed.values() for r, _ in vals})
        metric_keys = sorted(history.metrics_distributed.keys())
        col_w       = max(22, max(len(k) for k in metric_keys) + 2)
        header      = "  ".join(f"{k:>{col_w}}" for k in metric_keys)
        log.info("── Evaluation metrics per round ──")
        log.info("  %-8s  %s", "Round", header)
        for rnd in all_rounds:
            row = {k: next((v for r, v in history.metrics_distributed[k] if r == rnd), float("nan"))
                   for k in metric_keys}
            vals_str = "  ".join(f"{float(row[k]):>{col_w}.6f}" for k in metric_keys)
            log.info("  %-8d  %s", rnd, vals_str)

    if history.losses_distributed and len(history.losses_distributed) >= 2:
        losses = [v for _, v in sorted(history.losses_distributed)]
        delta  = losses[-1] - losses[0]
        trend  = "decreasing (converging)" if delta < 0 else "increasing (diverging)"
        log.info(
            "── Loss trend: %s (Δ=%.4f | round1=%.4f | last=%.4f)",
            trend, delta, losses[0], losses[-1],
        )

    log.info(SEP)


if __name__ == "__main__":
    main()
