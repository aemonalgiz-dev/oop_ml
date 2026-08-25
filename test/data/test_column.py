"""Spec for Column -- the library's single coercion and validation boundary."""

import numpy as np
import pytest

from oop_ml.data.column import Column
from oop_ml.data.feature import Feature
from oop_ml.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    TooFewValuesError,
)
from oop_ml.validation import ValueRole


def make_column(values: list[float] | None = None) -> Column:
    return Column(
        values if values is not None else [1, 2, 3, 4], ValueRole.INPUT_VALUES
    )


class TestConstruction:
    def test_coerces_sequence_to_float_array(self):
        column = make_column([1, 2, 3])

        assert isinstance(column.values, np.ndarray)
        assert column.values.dtype == np.float64

    def test_reports_its_role(self):
        assert make_column().role is ValueRole.INPUT_VALUES

    def test_n_samples_and_len_agree(self):
        column = make_column([1, 2, 3])

        assert column.n_samples == 3
        assert len(column) == 3


class TestValidation:
    def test_empty_raises(self):
        with pytest.raises(EmptyValuesError):
            Column([], ValueRole.INPUT_VALUES)

    @pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
    def test_non_finite_raises(self, invalid_value):
        with pytest.raises(InvalidValuesError):
            Column([1.0, 2.0, invalid_value], ValueRole.INPUT_VALUES)

    def test_two_dimensional_raises(self):
        with pytest.raises(InvalidValuesError):
            Column([[1, 2], [3, 4]], ValueRole.INPUT_VALUES)  # type: ignore[arg-type]

    def test_error_names_the_role(self):
        with pytest.raises(EmptyValuesError, match="input_values"):
            Column([], ValueRole.INPUT_VALUES)


class TestImmutability:
    def test_values_are_read_only(self):
        column = make_column()

        with pytest.raises(ValueError):
            column.values[0] = 99.0

    def test_callers_array_stays_writable(self):
        original = np.array([1.0, 2.0, 3.0])
        Column(original, ValueRole.INPUT_VALUES)

        original[0] = 99.0

        assert original[0] == 99.0


class TestOf:
    def test_wraps_raw_input(self):
        assert isinstance(Column.of([1, 2, 3], ValueRole.INPUT_VALUES), Column)

    def test_returns_an_existing_column_unchanged(self):
        column = make_column()

        assert Column.of(column, ValueRole.TARGET_VALUES) is column

    def test_existing_column_keeps_its_own_role(self):
        column = make_column()

        assert Column.of(column, ValueRole.TARGET_VALUES).role is ValueRole.INPUT_VALUES

    def test_takes_the_column_from_anything_carrying_one(self):
        # Feature satisfies HasColumn structurally, so it is accepted anywhere a
        # column is wanted -- and hands over its own, without re-validating.
        feature = Feature("age", [1, 2, 3])

        assert Column.of(feature, ValueRole.INPUT_VALUES) is feature.column


class TestStatistics:
    def test_mean(self):
        assert make_column([1, 2, 3, 4]).mean == pytest.approx(2.5)

    def test_deviations_are_distances_from_the_mean(self):
        np.testing.assert_allclose(
            make_column([1, 2, 3]).deviations,
            [-1.0, 0.0, 1.0],
        )

    def test_sum_of_squared_deviations(self):
        assert make_column([1, 2, 3]).sum_of_squared_deviations == pytest.approx(2.0)

    def test_sum_of_squared_deviations_is_zero_for_a_constant_column(self):
        assert make_column([7, 7, 7]).sum_of_squared_deviations == pytest.approx(0.0)


class TestGuards:
    def test_variance_check_passes_when_values_differ(self):
        make_column([1, 2, 3]).check_has_variance()

    def test_variance_check_raises_when_constant(self):
        with pytest.raises(AllSameValuesError):
            make_column([2, 2, 2]).check_has_variance()

    def test_min_length_passes_when_long_enough(self):
        make_column([1, 2]).check_min_length(2)

    def test_min_length_raises_when_too_short(self):
        with pytest.raises(TooFewValuesError):
            make_column([1]).check_min_length(2)

    def test_equal_length_passes_when_aligned(self):
        make_column([1, 2, 3]).check_equal_length(make_column([4, 5, 6]))

    def test_equal_length_raises_when_unaligned(self):
        with pytest.raises(NonEqualArrayLengthError):
            make_column([1, 2, 3]).check_equal_length(make_column([4, 5]))
