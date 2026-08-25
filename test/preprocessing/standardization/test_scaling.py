"""Spec for FeatureScaling / FeatureScalings -- learned statistics as objects."""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)
from oop_ml.preprocessing.standardization.scaling import FeatureScaling, FeatureScalings


def make_scalings() -> FeatureScalings:
    return FeatureScalings(
        [FeatureScaling("x1", 1.4, 2.0), FeatureScaling("x2", 0.0, 0.5)]
    )


class TestFeatureScaling:
    def test_carries_its_name_and_statistics(self):
        scaling = FeatureScaling("age", 40.0, 5.0)

        assert scaling.name == "age"
        assert scaling.mean == pytest.approx(40.0)
        assert scaling.standard_deviation == pytest.approx(5.0)

    def test_name_is_stripped(self):
        assert FeatureScaling("  age  ", 1.0, 1.0).name == "age"

    def test_empty_name_raises(self):
        with pytest.raises(InvalidValuesError):
            FeatureScaling("", 1.0, 1.0)

    def test_zero_spread_raises(self):
        with pytest.raises(AllSameValuesError):
            FeatureScaling("constant", 7.0, 0.0)

    def test_negative_spread_raises(self):
        with pytest.raises(AllSameValuesError):
            FeatureScaling("age", 1.0, -2.0)

    def test_standardize_centres_and_rescales(self):
        scaling = FeatureScaling("age", 10.0, 2.0)

        np.testing.assert_allclose(
            scaling.standardize(np.array([8.0, 10.0, 14.0])), [-1.0, 0.0, 2.0]
        )

    def test_restore_undoes_standardize(self):
        scaling = FeatureScaling("age", 10.0, 2.0)
        original = np.array([8.0, 10.0, 14.0])

        np.testing.assert_allclose(
            scaling.restore(scaling.standardize(original)), original
        )

    def test_equal_when_name_and_statistics_match(self):
        assert FeatureScaling("age", 1.0, 2.0) == FeatureScaling("age", 1.0, 2.0)

    def test_unequal_when_statistics_differ(self):
        assert FeatureScaling("age", 1.0, 2.0) != FeatureScaling("age", 1.0, 3.0)


class TestFeatureScalings:
    def test_reads_a_scaling_by_name(self):
        assert make_scalings()["x1"].mean == pytest.approx(1.4)

    def test_scaling_for_matches_subscript(self):
        scalings = make_scalings()

        assert scalings.scaling_for("x2") == scalings["x2"]

    def test_unknown_feature_raises(self):
        with pytest.raises(InvalidValuesError):
            make_scalings()["nope"]

    def test_membership_is_by_name(self):
        assert "x1" in make_scalings()
        assert "nope" not in make_scalings()

    def test_iterates_scaling_objects(self):
        assert all(isinstance(scaling, FeatureScaling) for scaling in make_scalings())

    def test_counts_its_features(self):
        scalings = make_scalings()

        assert scalings.n_features == 2
        assert len(scalings) == 2

    def test_empty_raises(self):
        with pytest.raises(EmptyValuesError):
            FeatureScalings([])

    def test_duplicate_names_raise(self):
        with pytest.raises(NonUniqueFeaturesError):
            FeatureScalings(
                [FeatureScaling("x1", 0.0, 1.0), FeatureScaling("x1", 1.0, 2.0)]
            )
