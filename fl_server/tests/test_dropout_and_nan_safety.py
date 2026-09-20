"""
Tests for client dropout / NaN-weight handling in fl_server.server strategies.

Covers invariant checklist items:
  - Client dropout mid-round.
  - Aggregation with a silo having zero samples / all-NaN results.
"""

import numpy as np
from flwr.common import FitRes, Status, ndarrays_to_parameters
from flwr.common.typing import Code

from fl_server.server import (
    NaNSafeFedAvg,
    NaNSafeFedProx,
    _has_nan,
    build_base_strategy,
    wrap_with_dp,
)


def _fit_res(values, num_examples=10, metrics=None):
    return FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters([np.array(values, dtype=np.float64)]),
        num_examples=num_examples,
        metrics=metrics or {},
    )


class TestHasNan:
    def test_should_detect_nan(self):
        assert _has_nan(ndarrays_to_parameters([np.array([1.0, float("nan")])]))

    def test_should_detect_inf(self):
        assert _has_nan(ndarrays_to_parameters([np.array([float("inf"), 0.0])]))

    def test_should_not_flag_finite_values(self):
        assert not _has_nan(ndarrays_to_parameters([np.array([1.0, 2.0, -3.5])]))


class TestNaNSafeMixinDropout:
    def test_should_drop_nan_client_and_aggregate_remaining(self):
        strategy = NaNSafeFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )
        good = (None, _fit_res([2.0, 4.0], num_examples=10))
        bad = (None, _fit_res([float("nan"), 1.0], num_examples=10))

        params, metrics = strategy.aggregate_fit(1, [good, bad], [])

        assert params is not None, "the healthy client's result should still aggregate"

    def test_should_skip_round_when_all_clients_nan(self):
        strategy = NaNSafeFedProx(
            proximal_mu=0.01,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )
        bad1 = (None, _fit_res([float("nan")], num_examples=10))
        bad2 = (None, _fit_res([float("inf")], num_examples=10))

        params, metrics = strategy.aggregate_fit(1, [bad1, bad2], [])

        assert params is None
        assert metrics == {}

    def test_should_handle_client_dropout_mid_round_via_failures_list(self):
        # A client dropping mid-round shows up in `failures`, not `results`.
        # accept_failures=True (as configured in build_base_strategy) means
        # the strategy must still aggregate on the remaining results.
        strategy = build_base_strategy(
            strategy_name="fedavg",
            min_clients=2,
            fraction_fit=1.0,
            fraction_eval=1.0,
            proximal_mu=0.01,
            fit_metrics_fn=lambda m: {},
            eval_metrics_fn=lambda m: {},
        )
        good = (None, _fit_res([1.0, 1.0], num_examples=5))
        params, _ = strategy.aggregate_fit(1, [good], [RuntimeError("client dropped")])
        assert params is not None


class TestPartialResultsDPStrategyDropout:
    """PartialResultsDPStrategy must not abort the whole round on partial
    gRPC failures, unlike Flower's native DPAdaptiveClipping."""

    def test_should_forward_empty_failures_list_to_tolerate_dropout(self, monkeypatch):
        captured = {}

        def fake_super_aggregate_fit(self, server_round, results, failures):
            captured["results"] = results
            captured["failures"] = failures
            return "AGGREGATED", {"ok": True}

        from flwr.server.strategy import (
            DifferentialPrivacyServerSideAdaptiveClipping as DPAdaptiveClipping,
        )

        monkeypatch.setattr(
            DPAdaptiveClipping, "aggregate_fit", fake_super_aggregate_fit
        )

        base = build_base_strategy(
            strategy_name="fedavg",
            min_clients=2,
            fraction_fit=1.0,
            fraction_eval=1.0,
            proximal_mu=0.01,
            fit_metrics_fn=lambda m: {},
            eval_metrics_fn=lambda m: {},
        )
        dp_strategy = wrap_with_dp(
            base,
            num_sampled_clients=2,
            noise_multiplier=0.9,
            initial_clip_norm=1.0,
            target_quantile=0.5,
        )

        good = (None, _fit_res([1.0], num_examples=5))
        result = dp_strategy.aggregate_fit(
            1, [good], failures=[RuntimeError("dropped mid-round")]
        )

        assert result == ("AGGREGATED", {"ok": True})
        # The whole point of PartialResultsDPStrategy: failures must NOT reach
        # the wrapped DPAdaptiveClipping (which aborts on any failures).
        assert captured["failures"] == []
        assert len(captured["results"]) == 1

    def test_should_skip_round_when_no_valid_results_remain(self, monkeypatch):
        from flwr.server.strategy import (
            DifferentialPrivacyServerSideAdaptiveClipping as DPAdaptiveClipping,
        )

        def fail_if_called(self, *a, **kw):
            raise AssertionError("must not delegate when no valid results remain")

        monkeypatch.setattr(DPAdaptiveClipping, "aggregate_fit", fail_if_called)

        base = build_base_strategy(
            strategy_name="fedavg",
            min_clients=2,
            fraction_fit=1.0,
            fraction_eval=1.0,
            proximal_mu=0.01,
            fit_metrics_fn=lambda m: {},
            eval_metrics_fn=lambda m: {},
        )
        dp_strategy = wrap_with_dp(
            base,
            num_sampled_clients=2,
            noise_multiplier=0.9,
            initial_clip_norm=1.0,
            target_quantile=0.5,
        )
        bad = (None, _fit_res([float("nan")], num_examples=5))
        params, metrics = dp_strategy.aggregate_fit(1, [bad], [])
        assert params is None
        assert metrics == {}
