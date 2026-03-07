"""Spec for Dataset / DataSplit -- features and their target kept together."""

import numpy as np
import pytest

from oop_ml.data.feature import Feature
from oop_ml.exceptions import (
    EmptyValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
)
from oop_ml.model_selection.dataset import Dataset, DataSplit
from test.fixtures import EXACT_PLANE


def make_dataset() -> Dataset:
    return Dataset(EXACT_PLANE.input_features, EXACT_PLANE.target_feature)


class TestConstruction:
    def test_reports_its_shape(self):
        dataset = make_dataset()

        assert dataset.n_samples == 5
        assert dataset.n_features == 2
        assert len(dataset) == 5

    def test_keeps_the_features_in_order(self):
        assert [feature.name for feature in make_dataset().input_features] == [
            "x1",
            "x2",
        ]

    def test_keeps_the_target(self):
        assert make_dataset().target_feature.name == "y"


class TestValidation:
    def test_no_features_raises(self):
        with pytest.raises(EmptyValuesError):
            Dataset([], EXACT_PLANE.target_feature)

    def test_duplicate_names_raise(self):
        first, second = EXACT_PLANE.input_features

        with pytest.raises(NonUniqueFeaturesError):
            Dataset([first, Feature("x1", second.values)], EXACT_PLANE.target_feature)

    def test_misaligned_features_raise(self):
        with pytest.raises(NonEqualArrayLengthError):
            Dataset(
                [Feature("x1", [1, 2, 3]), Feature("x2", [4, 5])],
                Feature("y", [1, 2, 3]),
            )

    def test_target_of_a_different_length_raises(self):
        with pytest.raises(NonEqualArrayLengthError):
            Dataset(EXACT_PLANE.input_features, Feature("y", [1, 2]))

    def test_a_constant_column_is_allowed(self):
        # A held-out fold can legitimately contain one; the model's own fit is
        # what rejects it.
        dataset = Dataset(
            [Feature("x1", [1, 2, 3]), Feature("flat", [7, 7, 7])],
            Feature("y", [1, 2, 3]),
        )

        assert dataset.n_features == 2


class TestSelectRows:
    @pytest.mark.parametrize(
        ("row_indices", "expected_first", "expected_target"),
        [
            ([0, 1], [1.0, 1.0], [6.0, 9.0]),
            ([4, 3, 2], [3.0, 0.0, 2.0], [7.0, 4.0, 11.0]),
            ([2], [2.0], [11.0]),
        ],
        ids=["first two", "reordered", "single row"],
    )
    def test_subsets_every_column_identically(
        self, row_indices, expected_first, expected_target
    ):
        subset = make_dataset().select_rows(row_indices)

        np.testing.assert_allclose(subset.input_features[0].values, expected_first)
        np.testing.assert_allclose(subset.target_feature.values, expected_target)

    def test_preserves_names(self):
        subset = make_dataset().select_rows([0, 1, 2])

        assert [feature.name for feature in subset.input_features] == ["x1", "x2"]
        assert subset.target_feature.name == "y"

    def test_reports_the_new_row_count(self):
        assert make_dataset().select_rows([0, 2, 4]).n_samples == 3

    def test_leaves_the_original_untouched(self):
        dataset = make_dataset()

        dataset.select_rows([0, 1])

        assert dataset.n_samples == 5


class TestDataSplit:
    def make_split(self) -> DataSplit:
        dataset = make_dataset()

        return DataSplit(dataset.select_rows([0, 1, 2]), dataset.select_rows([3, 4]))

    def test_names_the_halves_rather_than_ordering_them(self):
        split = self.make_split()

        assert split.training.n_samples == 3
        assert split.testing.n_samples == 2

    def test_reports_the_total(self):
        assert self.make_split().n_samples == 5
