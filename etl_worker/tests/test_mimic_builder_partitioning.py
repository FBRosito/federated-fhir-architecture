"""
Tests for the Non-IID Dirichlet partitioning and chronological temporal
split logic in etl_worker/mimic_builder.py (used to build the federated
silos consumed by ai_client / fl_server).

Covers bug-hunting checklist items:
  - Dirichlet sum-to-n (partition counts must exactly cover every row).
  - Chronological train/val/test split (temporal leakage risk).
"""

import numpy as np
import pandas as pd
import pytest


def _make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    admittime = pd.date_range("2105-01-01", periods=n, freq="D")
    chapters = rng.choice(list("IJZ"), size=n)
    return pd.DataFrame(
        {
            "admittime": admittime,
            "icd_chapter": chapters,
        }
    )


class TestDirichletPartitionSumToN:
    def test_should_assign_every_row_to_a_silo(self, mimic_builder):
        df = _make_df(n=150)
        out = mimic_builder._dirichlet_partition(df, n_silos=3, alpha=0.5, seed=42)
        assert (out["partition_id"] >= 0).all(), "no row should be left unassigned (-1)"

    def test_should_conserve_total_row_count_per_chapter(self, mimic_builder):
        df = _make_df(n=300)
        out = mimic_builder._dirichlet_partition(df, n_silos=4, alpha=0.3, seed=7)
        for chapter, group in df.groupby("icd_chapter"):
            assigned = out.loc[group.index, "partition_id"]
            assert len(assigned) == len(
                group
            ), f"chapter {chapter}: partitioning must not drop or duplicate rows"

    def test_should_conserve_total_row_count_overall(self, mimic_builder):
        df = _make_df(n=97)  # deliberately not divisible by n_silos
        out = mimic_builder._dirichlet_partition(df, n_silos=3, alpha=0.1, seed=1)
        assert len(out) == len(df)
        counts_by_silo = out["partition_id"].value_counts().sum()
        assert counts_by_silo == len(df)

    def test_should_be_deterministic_given_fixed_seed(self, mimic_builder):
        df = _make_df(n=120)
        out1 = mimic_builder._dirichlet_partition(df, n_silos=3, alpha=0.2, seed=42)
        out2 = mimic_builder._dirichlet_partition(df, n_silos=3, alpha=0.2, seed=42)
        assert out1["partition_id"].tolist() == out2["partition_id"].tolist()


class TestTemporalSplitPercentile:
    def test_should_split_approximately_80_10_10_globally(self, mimic_builder):
        df = _make_df(n=1000)
        splits = mimic_builder._assign_temporal_split_percentile(df)
        counts = splits.value_counts()
        n = len(df)
        assert counts["train"] == int(0.80 * n)
        assert counts["val"] == int(0.90 * n) - int(0.80 * n)
        assert counts["test"] == n - int(0.90 * n)

    def test_should_put_earliest_admissions_in_train_and_latest_in_test(
        self, mimic_builder
    ):
        df = _make_df(n=100)
        splits = mimic_builder._assign_temporal_split_percentile(df)
        ordered = df.loc[df.sort_values("admittime").index]
        # First 80 chronologically must all be "train".
        assert (splits.loc[ordered.index[:80]] == "train").all()
        # Last 10 chronologically must all be "test".
        assert (splits.loc[ordered.index[-10:]] == "test").all()


class TestTemporalSplitVsNonIIDPartitionOrdering:
    """BUG: etl_worker/mimic_builder.py:load_base_data computes the
    chronological temporal_split (line 425, `_assign_temporal_split_percentile`)
    on the FULL, un-partitioned dataframe, and only afterwards applies Dirichlet
    Non-IID partitioning by ICD-10 chapter (line 490, `_dirichlet_partition`).

    Because real MIMIC-IV data has ICD-10 chapters whose incidence drifts over
    time (e.g. coding practice changes, seasonal disease patterns), and
    Dirichlet partitioning at low alpha deliberately concentrates each ICD-10
    chapter into one or two dominant silos, a silo dominated by a
    time-correlated chapter ends up with a temporal_split composition that
    deviates arbitrarily far from the intended 80/10/10 split — the global
    percentile split gives no per-silo guarantee at all. In the extreme
    (all of a chapter's admissions fall in the tail of the time range) a
    silo can end up with 0 test examples, or its 'test' split can consist
    entirely of near-duplicate admissions from the same narrow time window,
    which is a temporal-leakage-adjacent risk in the opposite direction
    (over-representation) of what the split is meant to prevent.
    """

    def test_should_keep_per_silo_split_close_to_global_ratio_but_currently_does_not(
        self, mimic_builder
    ):
        n = 600
        # Chapter "I" admissions are all in the FIRST 20% of the time range
        # (which the global percentile split marks almost entirely "train");
        # chapter "J" spans the full time range as a control group.
        admittime = pd.date_range("2105-01-01", periods=n, freq="D")
        chapters = ["I"] * (n // 5) + ["J"] * (n - n // 5)
        df = pd.DataFrame({"admittime": admittime, "icd_chapter": chapters})

        # Compute the split exactly as load_base_data does: BEFORE partitioning.
        df["temporal_split"] = mimic_builder._assign_temporal_split_percentile(df)

        # Now Non-IID partition with low alpha so chapter "I" is concentrated
        # into a single dominant silo, reproducing production settings
        # (e.g. --dirichlet-alpha 0.1, the documented "highly Non-IID" value).
        # seed=3 is chosen because it deterministically drives chapter "I"
        # entirely into one silo for this alpha/n_silos combination — the
        # bug does not depend on the seed, just on any assignment where a
        # time-correlated chapter ends up isolated in a silo.
        partitioned = mimic_builder._dirichlet_partition(
            df, n_silos=2, alpha=0.05, seed=3
        )

        # Find whichever silo chapter "I" was concentrated into.
        chapter_i_rows = partitioned[partitioned["icd_chapter"] == "I"]
        dominant_silo = chapter_i_rows["partition_id"].mode()[0]
        silo_df = partitioned[partitioned["partition_id"] == dominant_silo]

        test_fraction = (silo_df["temporal_split"] == "test").mean()

        # Expected (correct) behavior for a proper per-silo split: each silo's
        # test fraction should stay close to the global 10% target regardless
        # of which chapters Dirichlet assigned to it.
        assert test_fraction == pytest.approx(0.10, abs=0.03), (
            f"silo {dominant_silo} (dominated by chapter 'I', which is "
            f"concentrated in the earliest 20% of admittimes) has a 'test' "
            f"fraction of {test_fraction:.3f}, far from the intended 10% — "
            "the chronological split is computed globally BEFORE Non-IID "
            "Dirichlet partitioning, so per-silo train/val/test ratios are "
            "not guaranteed and can become degenerate for any silo whose "
            "dominant chapter clusters in time."
        )
