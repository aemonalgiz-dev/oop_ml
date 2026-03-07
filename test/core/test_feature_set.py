"""Spec for FeatureSet -- the cross-column invariants a model can rely on."""

import pytest

from oop_ml.core.feature import Feature
from oop_ml.core.feature_set import FeatureSet
from oop_ml.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    TooFewValuesError,
)
from test.fixtures import EXACT_PLANE


def make_feature_set() -> FeatureSet:
    return FeatureSet(EXACT_PLANE.input_features)


class TestConstruction:
    def test_keeps_columns_in_the_order_supplied(self):
        assert [feature.name for feature in make_feature_set()] == ["x1", "x2"]

    def test_reports_the_shared_sample_count(self):
        assert make_feature_set().n_samples == 5

    def test_reports_the_column_count(self):
        feature_set = make_feature_set()

        assert feature_set.n_features == 2
        assert len(feature_set) == 2

    def test_iterates_the_feature_objects_themselves(self):
        assert all(isinstance(feature, Feature) for feature in make_feature_set())


class TestInvariants:
    def test_no_features_raises(self):
        with pytest.raises(EmptyValuesError):
            FeatureSet([])

    def test_duplicate_names_raise(self):
        with pytest.raises(NonUniqueFeaturesError):
            FeatureSet(
                [
                    Feature(name="age", values=[1, 2, 3]),
                    Feature(name="age", values=[4, 5, 6]),
                ]
            )

    def test_duplicate_name_appears_in_the_message(self):
        with pytest.raises(NonUniqueFeaturesError, match="age"):
            FeatureSet(
                [
                    Feature(name="age", values=[1, 2, 3]),
                    Feature(name="age", values=[4, 5, 6]),
                ]
            )

    def test_mismatched_lengths_raise(self):
        with pytest.raises(NonEqualArrayLengthError):
            FeatureSet(
                [
                    Feature(name="age", values=[1, 2, 3, 4]),
                    Feature(name="price", values=[1, 2, 3]),
                ]
            )

    def test_mismatch_names_both_columns(self):
        with pytest.raises(NonEqualArrayLengthError, match="price.*age"):
            FeatureSet(
                [
                    Feature(name="age", values=[1, 2, 3, 4]),
                    Feature(name="price", values=[1, 2, 3]),
                ]
            )

    def test_a_constant_column_is_structurally_fine(self):
        # Holding one is legal; only *fitting* on one is not.
        feature_set = FeatureSet(
            [
                Feature(name="age", values=[1, 2, 3]),
                Feature(name="constant", values=[7, 7, 7]),
            ]
        )

        assert feature_set.n_features == 2


class TestCheckColumnsVary:
    def test_passes_when_every_column_varies(self):
        make_feature_set().check_columns_vary()

    def test_constant_column_raises(self):
        feature_set = FeatureSet(
            [
                Feature(name="age", values=[1, 2, 3]),
                Feature(name="constant", values=[7, 7, 7]),
            ]
        )

        with pytest.raises(AllSameValuesError):
            feature_set.check_columns_vary()


class TestColumnLookup:
    def test_returns_the_named_column(self):
        assert make_feature_set().column("x2").name == "x2"

    def test_unknown_name_raises(self):
        with pytest.raises(InvalidValuesError):
            make_feature_set().column("nope")


class TestCheckAlignedWith:
    def test_matching_length_passes(self):
        make_feature_set().check_aligned_with(EXACT_PLANE.target_feature)

    def test_mismatched_length_raises(self):
        with pytest.raises(NonEqualArrayLengthError):
            make_feature_set().check_aligned_with(Feature(name="y", values=[1, 2]))


class TestCheckSupportsParameterCount:
    def test_enough_samples_passes(self):
        # 5 rows, 2 features plus an intercept = 3 parameters
        make_feature_set().check_supports_parameter_count(3)

    def test_exactly_enough_samples_passes(self):
        make_feature_set().check_supports_parameter_count(5)

    def test_too_few_samples_raises(self):
        # 3 rows cannot determine 4 parameters
        feature_set = FeatureSet(
            [
                Feature(name="x1", values=[1, 2, 3]),
                Feature(name="x2", values=[4, 5, 7]),
                Feature(name="x3", values=[2, 9, 4]),
            ]
        )

        with pytest.raises(TooFewValuesError):
            feature_set.check_supports_parameter_count(4)
