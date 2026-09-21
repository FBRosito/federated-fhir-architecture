"""Privacy-cost accounting for the equalization operator E.

``epsilon(config)`` is a pure function of (sigma, sample_rate/silos, rounds,
delta) -> eps, computed via Opacus's RDP accountant without running any
training. It wraps the two accountant callables identified as reusable
library calls in docs/EQUALIZATION_READINESS.md (answer E):

  - ``ai_client.fl_client._compute_cumulative_epsilon`` (record-level RDP,
    Opacus ``RDPAccountant``) for methods that clip+noise per optimizer step
    (DP-FedAvg, DP-FedProx).
  - ``client_level_dp.accountant.client_level_epsilon`` (client-level RDP,
    one Gaussian release per FL round on the aggregate) for client-level DP.

Both are reused as-is rather than reimplemented, so paper epsilon numbers
never drift from the numbers the training code itself reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_client.fl_client import _compute_cumulative_epsilon
from client_level_dp.accountant import client_level_epsilon

Accountant = Literal["rdp", "client_level"]


@dataclass(frozen=True)
class MethodConfig:
    """A reparameterizable DP training configuration for one FL method.

    Record-level methods (accountant="rdp") are accounted either as a single
    (sample_rate, steps_per_round) pair, or, when ``silos`` is given, as the
    mean of per-silo RDP accounting over ``silos`` -- (k, n) optimizer-steps
    and local-dataset-size pairs for each silo. HERALD's main pipeline
    reports epsilon_cumulative as this per-silo mean (evidence:
    experiments/adaptive-clipping/analysis/renyi_group_composition.py:59-65
    lists the same 5 (k, n) pairs used here; averaging RDPAccountant over
    them reproduces the archived experiment_logs/fl_fedprox_alpha0.5_dp*
    _bert_seed42.json epsilon_cumulative values to <0.02% relative error --
    see tests/test_accountant.py::test_reproduces_known_herald_epsilons).

    Client-level methods (accountant="client_level") noise the aggregate
    once per round; ``sigma`` and ``rounds`` are the only accounting inputs
    (clipping norm C0/C_silo affects sensitivity/utility, not epsilon).
    """

    method: str
    sigma: float
    rounds: int
    delta: float = 1e-5
    sample_rate: float | None = None
    steps_per_round: int | None = None
    silos: tuple[tuple[int, int], ...] | None = None
    accountant: Accountant = "rdp"


def epsilon(config: MethodConfig) -> float:
    """Privacy cost of ``config``, without running any training."""
    if config.accountant == "client_level":
        eps, _alpha = client_level_epsilon(config.sigma, config.rounds, config.delta)
        return eps

    if config.accountant != "rdp":
        raise ValueError(f"unknown accountant {config.accountant!r}")

    if config.silos is not None:
        per_silo = [
            _compute_cumulative_epsilon(
                config.sigma, k / n, k * config.rounds, config.delta
            )
            for k, n in config.silos
        ]
        return sum(per_silo) / len(per_silo)

    if config.sample_rate is None or config.steps_per_round is None:
        raise ValueError(
            "rdp accountant needs either `silos` or both `sample_rate` and "
            "`steps_per_round`"
        )
    total_steps = config.steps_per_round * config.rounds
    return _compute_cumulative_epsilon(
        config.sigma, config.sample_rate, total_steps, config.delta
    )
