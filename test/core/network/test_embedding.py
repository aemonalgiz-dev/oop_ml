"""Spec for the lookup table, and most of it is about the two checks it adds.

This is the layer this package recorded as impossible, so the tests that carry
the module are the ones about what arrives rather than about the arithmetic.
A position of ``2.9999`` truncates to 2 and a position of ``-1`` wraps to the
last row of the table, and under the coercion every other layer performs both of
those are finite float64 values that no scan can object to. So the fixtures here
include both, by value, and assert a typed refusal rather than an answer.

The backward pass has one claim worth a check of its own and it is the
accumulation. A position appearing twice in one sequence was read twice, so its
row collects a contribution from each appearance, and an implementation that
assigns rather than adds reports one of them and looks entirely healthy: right
shape, finite numbers, a loss that falls. The fixture therefore repeats a
position deliberately, and the finite difference is what settles it, since it
knows nothing about how the sum was written.

The table is built by counting rather than drawn, so every row names itself and
a lookup that reached one position too far along answers a different number
instead of a plausible one.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.embedding import Embedding
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.shape import LayerShape

#: The step a central difference takes on either side.
NUDGE = 1e-6

N_POSITIONS = 5
N_DIMENSIONS = 3
N_STEPS = 4


def counting_table() -> np.ndarray:
    """A table whose every entry names its own position and column."""
    return np.arange(N_POSITIONS * N_DIMENSIONS, dtype=np.float64).reshape(
        N_POSITIONS, N_DIMENSIONS
    )


def counting_layer() -> Embedding:
    """The fixture layer, carrying the counting table."""
    return Embedding(
        n_positions=N_POSITIONS, n_dimensions=N_DIMENSIONS, n_steps=N_STEPS
    ).with_parameters(counting_table())


def repeating_positions() -> np.ndarray:
    """Two sequences, one of which reads position 1 three times.

    The repetition is the point. Row 1 of the table is read three times in the
    first sequence and once in the second, so its slope is the sum of four
    contributions, and a backward pass that assigns rather than accumulates
    reports a quarter of the answer with nothing raised.
    """
    return np.array([[1.0, 1.0, 4.0, 1.0], [0.0, 2.0, 1.0, 3.0]])


def summary_of(layer: Embedding, positions: np.ndarray, weights: np.ndarray) -> float:
    """A scalar reading of the answer whose slope is ``weights`` exactly.

    ``sum(weights * outputs)`` differentiates to ``weights`` at every output, so
    one block serves as both the summary's definition and the arriving block the
    backward pass is handed, with no loss object in between.
    """
    return float(np.sum(weights * layer.respond_to(positions).outputs))


def measured_table_slopes(
    layer: Embedding, positions: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every table entry's slope, by nudging it and watching the summary move."""
    measured = np.zeros_like(layer.table)
    for row in range(layer.table.shape[0]):
        for column in range(layer.table.shape[1]):
            raised = np.array(layer.table, copy=True)
            lowered = np.array(layer.table, copy=True)
            raised[row, column] += NUDGE
            lowered[row, column] -= NUDGE
            measured[row, column] = (
                summary_of(layer.with_parameters(raised), positions, arriving)
                - summary_of(layer.with_parameters(lowered), positions, arriving)
            ) / (2.0 * NUDGE)
    return measured


class TestConstruction:
    def test_it_reads_positions_and_answers_vectors(self) -> None:
        layer = Embedding(n_positions=50, n_dimensions=8, n_steps=6)

        assert layer.shape == LayerShape(n_inputs=(6,), n_outputs=(6, 8))

    def test_the_table_holds_one_row_per_position(self) -> None:
        layer = Embedding(n_positions=50, n_dimensions=8, n_steps=6)

        assert layer.table.shape == (50, 8)

    def test_the_table_is_frozen(self) -> None:
        layer = Embedding(n_positions=5, n_dimensions=2, n_steps=3)

        with pytest.raises(ValueError):
            layer.table[0, 0] = 99.0

    def test_the_seed_settles_the_draw(self) -> None:
        first = Embedding(n_positions=6, n_dimensions=4, n_steps=2, random_seed=7)
        second = Embedding(n_positions=6, n_dimensions=4, n_steps=2, random_seed=7)

        assert np.array_equal(first.table, second.table)

    def test_different_seeds_draw_different_tables(self) -> None:
        first = Embedding(n_positions=6, n_dimensions=4, n_steps=2, random_seed=1)
        second = Embedding(n_positions=6, n_dimensions=4, n_steps=2, random_seed=2)

        assert not np.array_equal(first.table, second.table)

    @pytest.mark.parametrize("setting", ["n_positions", "n_dimensions", "n_steps"])
    def test_a_setting_of_zero_is_refused(self, setting: str) -> None:
        arguments = {"n_positions": 5, "n_dimensions": 3, "n_steps": 4}
        arguments[setting] = 0

        with pytest.raises(InvalidValuesError):
            Embedding(**arguments)

    @pytest.mark.parametrize("setting", ["n_positions", "n_dimensions", "n_steps"])
    def test_a_fractional_setting_is_refused(self, setting: str) -> None:
        arguments: dict[str, object] = {
            "n_positions": 5,
            "n_dimensions": 3,
            "n_steps": 4,
        }
        arguments[setting] = 2.5

        with pytest.raises(InvalidValuesError):
            Embedding(**arguments)  # type: ignore[arg-type]

    def test_it_says_its_geometry_in_its_repr(self) -> None:
        layer = Embedding(n_positions=5, n_dimensions=3, n_steps=4)

        assert repr(layer) == ("Embedding(n_positions=5, n_dimensions=3, n_steps=4)")


class TestTheForwardPass:
    def test_it_answers_one_vector_per_step(self) -> None:
        response = counting_layer().respond_to(repeating_positions())

        assert response.outputs.shape == (2, N_STEPS, N_DIMENSIONS)

    def test_each_step_gets_the_row_it_named(self) -> None:
        """The counting table makes a lookup one position out a visible failure."""
        response = counting_layer().respond_to(np.array([[0.0, 2.0, 4.0, 1.0]]))

        assert np.array_equal(
            response.outputs[0],
            [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0], [12.0, 13.0, 14.0], [3.0, 4.0, 5.0]],
        )

    def test_a_repeated_position_answers_the_same_row_twice(self) -> None:
        response = counting_layer().respond_to(np.array([[3.0, 3.0, 0.0, 0.0]]))

        assert np.array_equal(response.outputs[0, 0], response.outputs[0, 1])
        assert np.array_equal(response.outputs[0, 2], response.outputs[0, 3])

    def test_the_scores_and_the_outputs_are_the_same_block(self) -> None:
        """Nothing is bent here, so a pre-activation value is the value."""
        response = counting_layer().respond_to(repeating_positions())

        assert np.array_equal(response.scores, response.outputs)

    def test_it_keeps_the_positions_it_read(self) -> None:
        """The backward pass needs them, and only the forward pass knows them."""
        positions = repeating_positions()

        response = counting_layer().respond_to(positions)

        assert np.array_equal(response.inputs, positions)

    def test_the_answer_is_frozen(self) -> None:
        response = counting_layer().respond_to(repeating_positions())

        with pytest.raises(ValueError):
            response.outputs[0, 0, 0] = 99.0

    def test_a_list_of_whole_numbers_is_accepted(self) -> None:
        """Positions arrive as integers far more often than as floats."""
        response = counting_layer().respond_to([[0, 1, 2, 3]])  # type: ignore[arg-type]

        assert np.array_equal(response.outputs[0, 1], [3.0, 4.0, 5.0])


class TestTheRefusalsThisLayerAddsAtTheBoundary:
    """The hole this layer was recorded as unable to close, closed.

    Each of these is a value that the ordinary coercion accepts happily and
    that names a row other than the one the caller meant.
    """

    def test_a_fractional_position_is_refused(self) -> None:
        """2.9999 truncates to 2, and nothing about it is non-finite."""
        with pytest.raises(InvalidValuesError):
            counting_layer().respond_to(np.array([[0.0, 2.9999, 1.0, 3.0]]))

    def test_a_position_just_above_a_whole_number_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            counting_layer().respond_to(np.array([[0.0, 1.0, 2.0000001, 3.0]]))

    def test_a_negative_position_is_refused(self) -> None:
        """-1 indexes the last row of the table, which is a real answer."""
        with pytest.raises(InvalidValuesError):
            counting_layer().respond_to(np.array([[-1.0, 1.0, 2.0, 3.0]]))

    def test_a_position_past_the_end_of_the_table_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            counting_layer().respond_to(np.array([[0.0, 1.0, float(N_POSITIONS), 3.0]]))

    def test_the_last_position_in_the_table_is_accepted(self) -> None:
        """The refusal is at the end of the table, not one row before it."""
        response = counting_layer().respond_to(
            np.array([[0.0, 1.0, float(N_POSITIONS - 1), 3.0]])
        )

        assert np.array_equal(response.outputs[0, 2], [12.0, 13.0, 14.0])

    def test_a_non_finite_position_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            counting_layer().respond_to(np.array([[0.0, np.nan, 2.0, 3.0]]))

    def test_an_infinite_position_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            counting_layer().respond_to(np.array([[0.0, np.inf, 2.0, 3.0]]))

    def test_a_block_that_is_not_numbers_at_all_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            counting_layer().respond_to([[1.0, 2.0], [3.0]])  # type: ignore[arg-type]

    def test_the_shape_checks_still_run_after_the_coercion(self) -> None:
        """Overriding the read must not take over more than the coercion."""
        with pytest.raises(ShapeMismatchError):
            counting_layer().respond_to(np.array([[0.0, 1.0, 2.0]]))

    def test_a_block_of_no_rows_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            counting_layer().respond_to(np.zeros((0, N_STEPS)))

    def test_an_already_arranged_block_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            counting_layer().respond_to(np.zeros((2, N_STEPS, N_DIMENSIONS)))


class TestTheBackwardPass:
    def test_only_the_rows_that_were_looked_up_have_any_slope(self) -> None:
        """Position 2 is never read here, so its row is owed exactly nothing."""
        layer = counting_layer()
        positions = np.array([[0.0, 1.0, 3.0, 4.0]])
        response = layer.respond_to(positions)

        correction = layer.correction_for(response, np.ones((1, N_STEPS, N_DIMENSIONS)))

        assert correction.gradient is not None
        assert np.array_equal(correction.gradient.weights[2], np.zeros(N_DIMENSIONS))
        assert np.count_nonzero(correction.gradient.weights) == 4 * N_DIMENSIONS

    def test_a_position_read_twice_collects_both_contributions(self) -> None:
        """The accumulation, asserted as a number rather than as an intention."""
        layer = counting_layer()
        response = layer.respond_to(np.array([[1.0, 1.0, 1.0, 0.0]]))

        correction = layer.correction_for(response, np.ones((1, N_STEPS, N_DIMENSIONS)))

        assert correction.gradient is not None
        assert np.array_equal(
            correction.gradient.weights[1], np.full(N_DIMENSIONS, 3.0)
        )
        assert np.array_equal(correction.gradient.weights[0], np.ones(N_DIMENSIONS))

    def test_the_gradient_describes_the_whole_table(self) -> None:
        layer = counting_layer()
        response = layer.respond_to(repeating_positions())

        correction = layer.correction_for(response, np.ones((2, N_STEPS, N_DIMENSIONS)))

        assert correction.gradient is not None
        assert correction.gradient.weights.shape == (N_POSITIONS, N_DIMENSIONS)

    def test_the_bias_block_is_zero_and_the_right_height(self) -> None:
        """An embedding has no biases; the block exists so the pairing holds."""
        layer = counting_layer()
        response = layer.respond_to(repeating_positions())

        correction = layer.correction_for(response, np.ones((2, N_STEPS, N_DIMENSIONS)))

        assert correction.gradient is not None
        assert np.array_equal(correction.gradient.biases, np.zeros(N_POSITIONS))

    def test_nothing_is_passed_down(self) -> None:
        """A position is a name, not a measurement, so it has no slope."""
        layer = counting_layer()
        response = layer.respond_to(repeating_positions())

        correction = layer.correction_for(response, np.ones((2, N_STEPS, N_DIMENSIONS)))

        assert correction.passed_down.shape == (2, N_STEPS)
        assert np.array_equal(correction.passed_down, np.zeros((2, N_STEPS)))

    def test_it_reports_that_it_learns(self) -> None:
        """Unlike flattening, this layer really does have something to learn."""
        layer = counting_layer()
        response = layer.respond_to(repeating_positions())

        correction = layer.correction_for(response, np.ones((2, N_STEPS, N_DIMENSIONS)))

        assert correction.learns

    def test_arriving_slopes_of_the_wrong_arrangement_are_refused(self) -> None:
        layer = counting_layer()
        response = layer.respond_to(repeating_positions())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, N_STEPS)))

    def test_arriving_slopes_for_the_wrong_number_of_rows_are_refused(self) -> None:
        layer = counting_layer()
        response = layer.respond_to(repeating_positions())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((3, N_STEPS, N_DIMENSIONS)))

    def test_slopes_that_are_not_numbers_are_refused(self) -> None:
        layer = counting_layer()
        response = layer.respond_to(repeating_positions())

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, [[1.0, 2.0], [3.0]])  # type: ignore[arg-type]


class TestTheGradientCheck:
    """Nudge each table entry, watch the summary move, compare with the claim.

    The only thing standing behind the accumulation. It runs the layer forward
    and knows nothing about how the backward step was written.
    """

    def test_the_table_slopes_match_a_finite_difference(self) -> None:
        layer = Embedding(
            n_positions=N_POSITIONS, n_dimensions=N_DIMENSIONS, n_steps=N_STEPS
        )
        positions = repeating_positions()
        arriving = np.random.default_rng(3).normal(size=(2, N_STEPS, N_DIMENSIONS))
        response = layer.respond_to(positions)

        claimed = layer.correction_for(response, arriving).gradient
        measured = measured_table_slopes(layer, positions, arriving)

        assert claimed is not None
        assert np.allclose(claimed.weights, measured, atol=1e-7)

    def test_the_check_holds_when_a_position_is_never_read(self) -> None:
        layer = Embedding(
            n_positions=N_POSITIONS, n_dimensions=N_DIMENSIONS, n_steps=N_STEPS
        )
        positions = np.array([[0.0, 0.0, 3.0, 3.0]])
        arriving = np.random.default_rng(5).normal(size=(1, N_STEPS, N_DIMENSIONS))
        response = layer.respond_to(positions)

        claimed = layer.correction_for(response, arriving).gradient
        measured = measured_table_slopes(layer, positions, arriving)

        assert claimed is not None
        assert np.allclose(claimed.weights, measured, atol=1e-7)


class TestStepping:
    def test_a_step_moves_only_the_rows_that_were_read(self) -> None:
        layer = counting_layer()
        response = layer.respond_to(np.array([[0.0, 0.0, 0.0, 0.0]]))
        correction = layer.correction_for(response, np.ones((1, N_STEPS, N_DIMENSIONS)))

        stepped = layer.stepped_by(correction.gradient, learning_rate=0.5)

        assert np.array_equal(stepped.table[0], counting_table()[0] - 2.0)
        assert np.array_equal(stepped.table[1:], counting_table()[1:])

    def test_a_step_does_not_touch_the_layer_it_came_from(self) -> None:
        layer = counting_layer()
        response = layer.respond_to(np.array([[0.0, 0.0, 0.0, 0.0]]))
        correction = layer.correction_for(response, np.ones((1, N_STEPS, N_DIMENSIONS)))

        layer.stepped_by(correction.gradient, learning_rate=0.5)

        assert np.array_equal(layer.table, counting_table())

    def test_a_step_keeps_the_shape(self) -> None:
        layer = counting_layer()
        gradient = LayerGradient(
            weights=np.ones((N_POSITIONS, N_DIMENSIONS)), biases=np.zeros(N_POSITIONS)
        )

        assert layer.stepped_by(gradient, learning_rate=0.1).shape == layer.shape

    def test_stepping_by_nothing_is_refused(self) -> None:
        """This layer has parameters, so it cannot answer with itself."""
        with pytest.raises(ShapeMismatchError):
            counting_layer().stepped_by(None, learning_rate=0.1)

    def test_a_gradient_of_the_wrong_shape_is_refused(self) -> None:
        gradient = LayerGradient(weights=np.ones((2, 2)), biases=np.zeros(2))

        with pytest.raises(ShapeMismatchError):
            counting_layer().stepped_by(gradient, learning_rate=0.1)

    def test_a_replacement_table_of_the_wrong_shape_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            counting_layer().with_parameters(np.ones((N_POSITIONS + 1, N_DIMENSIONS)))
