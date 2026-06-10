"""Spec for the three value objects a Markov chain answers with.

Each one is a table bound to state names, and each refuses in its constructor
whatever would make it describe something that is not what its name says: a
distribution that does not sum to one, a count that is not a whole number of at
least zero, a row of transitions that goes nowhere. So a caller holding one
never has to check it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.sequences.transitions import (
    StateDistribution,
    TransitionCounts,
    TransitionMatrix,
)

STATES = ("sunny", "rainy")
COUNTS = [[9, 1], [1, 1]]
TABLE = [[0.9, 0.1], [0.5, 0.5]]


class TestStateDistribution:
    def test_it_is_addressable_by_state_name(self) -> None:
        distribution = StateDistribution(STATES, [0.25, 0.75])

        assert distribution["rainy"] == pytest.approx(0.75)
        assert distribution.probability_of("sunny") == pytest.approx(0.25)
        assert type(distribution["rainy"]) is float

    def test_it_knows_its_states_in_order(self) -> None:
        distribution = StateDistribution(STATES, [0.25, 0.75])

        assert distribution.states == STATES
        assert distribution.n_states == 2
        assert len(distribution) == 2

    def test_it_reads_as_an_array(self) -> None:
        assert np.array_equal(
            np.asarray(StateDistribution(STATES, [0.25, 0.75])), [0.25, 0.75]
        )

    def test_its_values_are_frozen(self) -> None:
        distribution = StateDistribution(STATES, [0.25, 0.75])

        with pytest.raises(ValueError):
            distribution.values[0] = 1.0

    def test_the_caller_s_array_is_copied_rather_than_frozen_underneath_them(
        self,
    ) -> None:
        probabilities = np.array([0.25, 0.75])

        StateDistribution(STATES, probabilities)
        probabilities[0] = 0.5

        assert probabilities[0] == 0.5

    def test_a_point_mass_is_certain_of_one_state(self) -> None:
        certain = StateDistribution.concentrated_on(STATES, "rainy")

        assert np.array_equal(np.asarray(certain), [0.0, 1.0])

    def test_it_can_be_put_into_another_order_by_name(self) -> None:
        reordered = StateDistribution(STATES, [0.25, 0.75]).in_order_of(
            ("rainy", "sunny")
        )

        assert reordered.states == ("rainy", "sunny")
        assert np.array_equal(np.asarray(reordered), [0.75, 0.25])

    def test_another_order_over_different_states_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            StateDistribution(STATES, [0.25, 0.75]).in_order_of(("sunny", "snowy"))

    def test_an_unknown_state_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            StateDistribution(STATES, [0.25, 0.75])["snowy"]

    @pytest.mark.parametrize(
        "probabilities",
        [
            pytest.param([0.5, 0.6], id="sums above one"),
            pytest.param([0.2, 0.2], id="sums below one"),
            pytest.param([-0.5, 1.5], id="negative"),
            pytest.param([math.nan, 1.0], id="not a number"),
            pytest.param([math.inf, 0.0], id="infinite"),
        ],
    )
    def test_numbers_that_are_not_a_distribution_are_refused(
        self, probabilities: list[float]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            StateDistribution(STATES, probabilities)

    def test_one_number_per_state_is_required(self) -> None:
        with pytest.raises(ShapeMismatchError):
            StateDistribution(STATES, [1.0])

    def test_no_states_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            StateDistribution([], [])

    def test_a_repeated_state_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            StateDistribution(["sunny", "sunny"], [0.5, 0.5])

    def test_a_single_string_of_states_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            StateDistribution("ab", [0.5, 0.5])  # type: ignore[arg-type]


class TestTransitionCounts:
    def test_a_count_is_addressable_by_both_names(self) -> None:
        counts = TransitionCounts(STATES, COUNTS)

        assert counts.count_of("sunny", "sunny") == 9
        assert counts.count_of("rainy", "sunny") == 1
        assert type(counts.count_of("sunny", "rainy")) is int

    def test_it_totals_what_leaves_each_state_and_what_it_holds(self) -> None:
        counts = TransitionCounts(STATES, COUNTS)

        assert counts.leaving("sunny") == 10
        assert counts.leaving("rainy") == 2
        assert counts.n_transitions == 12

    def test_it_names_the_states_never_left(self) -> None:
        counts = TransitionCounts(("open", "closed"), [[0, 1], [0, 0]])

        assert counts.never_left == ("closed",)

    def test_its_values_are_whole_numbers_and_frozen(self) -> None:
        counts = TransitionCounts(STATES, COUNTS)

        assert counts.values.dtype == np.int64
        with pytest.raises(ValueError):
            counts.values[0, 0] = 3

    def test_the_transition_matrix_divides_each_row_by_its_total(self) -> None:
        table = TransitionCounts(STATES, COUNTS).transition_matrix(0.0)

        assert np.allclose(np.asarray(table), TABLE)

    def test_smoothing_adds_the_same_count_to_every_cell(self) -> None:
        """``(9 + 2) / (10 + 4)`` and ``(1 + 2) / (10 + 4)`` on the first row."""
        table = TransitionCounts(STATES, COUNTS).transition_matrix(2.0)

        assert np.allclose(np.asarray(table)[0], [11.0 / 14.0, 3.0 / 14.0])

    def test_a_row_with_no_counts_is_refused_at_a_smoothing_of_zero(self) -> None:
        counts = TransitionCounts(("open", "closed"), [[0, 1], [0, 0]])

        with pytest.raises(TooFewValuesError, match="closed"):
            counts.transition_matrix(0.0)

    @pytest.mark.parametrize("smoothing", [-0.5, math.nan, math.inf])
    def test_a_smoothing_that_is_not_a_finite_count_is_refused(
        self, smoothing: float
    ) -> None:
        with pytest.raises(InvalidValuesError):
            TransitionCounts(STATES, COUNTS).transition_matrix(smoothing)

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([[1, -1], [0, 1]], id="negative"),
            pytest.param([[1.5, 0], [0, 1]], id="fractional"),
            pytest.param([[math.nan, 0], [0, 1]], id="not a number"),
        ],
    )
    def test_numbers_that_are_not_counts_are_refused(
        self, values: list[list[float]]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            TransitionCounts(STATES, values)

    def test_a_table_that_is_not_one_row_and_column_per_state_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            TransitionCounts(STATES, [[1, 2, 3], [4, 5, 6]])

    def test_an_unknown_state_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            TransitionCounts(STATES, COUNTS).count_of("sunny", "snowy")


class TestTransitionMatrix:
    def test_a_probability_is_addressable_by_both_names(self) -> None:
        table = TransitionMatrix(STATES, TABLE)

        assert table.probability_of("sunny", "rainy") == pytest.approx(0.1)
        assert type(table.probability_of("sunny", "rainy")) is float

    def test_a_row_is_the_distribution_of_where_that_state_goes_next(self) -> None:
        row = TransitionMatrix(STATES, TABLE)["rainy"]

        assert isinstance(row, StateDistribution)
        assert row.states == STATES
        assert np.allclose(np.asarray(row), [0.5, 0.5])

    def test_it_iterates_its_rows_in_state_order(self) -> None:
        rows = [np.asarray(row) for row in TransitionMatrix(STATES, TABLE)]

        assert len(rows) == 2
        assert np.allclose(rows[0], [0.9, 0.1])
        assert np.allclose(rows[1], [0.5, 0.5])

    def test_its_values_are_frozen(self) -> None:
        table = TransitionMatrix(STATES, TABLE)

        with pytest.raises(ValueError):
            table.values[0, 0] = 1.0

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([[0.9, 0.2], [0.5, 0.5]], id="a row summing above one"),
            pytest.param([[1.2, -0.2], [0.5, 0.5]], id="negative"),
            pytest.param([[math.nan, 1.0], [0.5, 0.5]], id="not a number"),
            pytest.param([[0.0, 0.0], [0.5, 0.5]], id="a row going nowhere"),
        ],
    )
    def test_a_table_that_is_not_a_transition_matrix_is_refused(
        self, values: list[list[float]]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            TransitionMatrix(STATES, values)

    def test_a_table_that_is_not_square_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            TransitionMatrix(STATES, [[1.0], [1.0]])
