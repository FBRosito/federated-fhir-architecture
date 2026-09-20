"""
Bug-hunting tests for fl_server.server metric aggregation.

Target invariant: FedProx aggregation weighted by n_k/n (see
.claude/agents/bug-hunter-tester.md, invariant 4) — this must hold for every
metric key that is aggregated, and a single client's degenerate report
(e.g. the "no local data" placeholder in ai_client.fl_client.fit(), which
returns num_examples=1 and train_loss=nan) must not corrupt metrics
contributed by the other, legitimate clients.
"""

import math

from fl_server.server import aggregate_eval_metrics, aggregate_fit_metrics


class TestAggregateFitMetricsNaNPoisoning:
    """BUG: a single client's NaN metric poisons the weighted average for
    every other client's value of that same key, even when its weight
    (n_examples) is tiny relative to the round total.

    fl_server/src/fl_server/server.py:400-404 does:
        aggregated[key] = aggregated.get(key, 0.0) + float(value) * weight
    float('nan') * weight is nan, and 0.0 + nan is nan, permanently
    poisoning `aggregated[key]` regardless of how small `weight` is or how
    many other clients contributed a finite value for that key.
    """

    def test_should_ignore_nan_client_contribution_when_computing_weighted_average(
        self,
    ):
        # Two healthy clients with real training data, one degenerate
        # "no local data" client (this is exactly the shape ai_client.fl_client
        # .fit() returns for a silo with zero examples: num_examples=1,
        # train_loss=nan — see ai_client/src/ai_client/fl_client.py:894-897).
        metrics = [
            (1000, {"train_loss": 0.5}),
            (1000, {"train_loss": 0.7}),
            (1, {"train_loss": float("nan")}),
        ]
        result = aggregate_fit_metrics(metrics)

        # Expected (correct) behavior: the degenerate client's negligible
        # weight (1 / 2001) should not be able to blow up the reported
        # round-level train_loss for 2000 real examples.
        assert "train_loss" in result
        assert math.isfinite(result["train_loss"]), (
            "aggregate_fit_metrics returned a non-finite train_loss "
            f"({result['train_loss']!r}) because one low-weight client "
            "reported NaN — this silently corrupts the reported metrics for "
            "the whole round even though 2000/2001 examples were healthy."
        )

    def test_should_ignore_nan_client_contribution_in_eval_metrics_too(self):
        metrics = [
            (500, {"eval_loss": 1.2}),
            (1, {"eval_loss": float("nan")}),
        ]
        result = aggregate_eval_metrics(metrics)
        assert math.isfinite(result["eval_loss"]), (
            "aggregate_eval_metrics propagates NaN from a single negligible-"
            "weight client into the round's aggregated eval_loss."
        )


class TestAggregateMetricsPartialKeyWeighting:
    """BUG: when only some clients report a given metric key (e.g. a
    BERT-backend client reporting 'micro_f1' alongside an LLM-backend client
    that does not), the weight denominator is `total_examples` across *all*
    clients, not just those that reported the key — so the reported average
    is silently scaled down and does not equal a true n_k/n weighted mean
    over the clients that actually reported it.
    """

    def test_should_weight_only_by_clients_reporting_the_metric(self):
        metrics = [
            (100, {"only_client_a": 10.0, "shared": 4.0}),
            (100, {"shared": 4.0}),
        ]
        result = aggregate_fit_metrics(metrics)

        # "only_client_a" is reported by a client carrying half the total
        # weight (100 / 200). A correct n_k/n weighted average *restricted to
        # the clients reporting this key* should equal the client's own
        # value (10.0), since it is the sole reporter.
        assert result["only_client_a"] == 10.0, (
            f"expected only_client_a == 10.0 (its sole reporter's value), "
            f"got {result['only_client_a']} — the denominator silently "
            "includes clients that never reported this key, deflating the "
            "weighted average."
        )
        # The key both clients share should be an unaffected 4.0.
        assert result["shared"] == 4.0


class TestAggregateMetricsEdgeCases:
    def test_should_return_empty_dict_when_no_metrics(self):
        assert aggregate_fit_metrics([]) == {}
        assert aggregate_eval_metrics([]) == {}

    def test_should_return_empty_dict_when_total_examples_zero(self):
        # Zero-sample silo dropout mid-round (invariant: aggregation with a
        # silo having zero samples must not raise ZeroDivisionError).
        assert aggregate_fit_metrics([(0, {"train_loss": 1.0})]) == {}

    def test_should_weight_by_num_examples(self):
        metrics = [
            (100, {"train_loss": 1.0}),
            (300, {"train_loss": 2.0}),
        ]
        result = aggregate_fit_metrics(metrics)
        # 100/400 * 1.0 + 300/400 * 2.0 = 0.25 + 1.5 = 1.75
        assert result["train_loss"] == 1.75
