"""Spec for the flattening layer, whose whole job is to move nothing.

Three of these are worth reading rather than skimming.

The first is the gradient check. A relabelling has a trivial derivative, which
makes it exactly the kind of layer whose backward pass nobody bothers to check,
and exactly the kind that can be silently wrong: reshape one direction in C
order and the other in Fortran order and every shape still conforms, every
number is finite, training still descends, and the blame arrives at the wrong
pixel forever. So the oracle here is the definition of a derivative. Nudge one
input value by a hair, measure how a scalar summary of the outputs actually
moved, and compare against what ``correction_for`` claims. That number is built
from forward passes alone and knows nothing about how the backward step was
written.

The summary is deliberately not linear. With a fixed block of arriving slopes
the claim and the measurement would both reduce to the same block of constants,
and a mapping error would have to be caught by position alone; with
``sum(sin(outputs))`` the slope at each output is ``cos`` of the value that
actually landed there, so a mismatched mapping disagrees in the *values* as
well as the positions.

The second is ordering. A flatten tested on zeros, or on any block whose values
repeat, passes while shuffling every number it touches. The fixtures here count
upward, so each value names its own position.

The third is the refusal of a wrongly arranged block. It is the argument for
the layer existing at all: ``(2, 3, 2)``, ``(2, 2, 3)``, ``(1, 12, 1)`` and
``(12,)`` all hold the same twelve numbers, and a layer that accepted any of
them would be exactly the count comparison this package refused to write into
``LayerShape.follows``.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.flatten import Flatten
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.shape import LayerShape

ARRANGEMENT = (2, 3, 2)
FLAT_WIDTH = 12


def counting_block(n_rows: int = 2) -> np.ndarray:
    """Rows of consecutive numbers, so every value names its own position.

    Row ``i`` runs from ``i * 12 + 1`` to ``i * 12 + 12``, arranged
    ``(2, 3, 2)`` in C order. A transposed reshape, a reversed row order, or a
    Fortran ordering all move at least one of those numbers somewhere it can be
    seen.
    """
    total = n_rows * FLAT_WIDTH
    return np.arange(1.0, total + 1.0).reshape(n_rows, *ARRANGEMENT)


class TestShape:
    def test_it_answers_with_the_count_of_what_it_reads(self) -> None:
        assert Flatten((8, 13, 13)).shape == LayerShape(
            n_inputs=(8, 13, 13), n_outputs=1352
        )

    def test_the_answering_side_is_one_dimensional(self) -> None:
        assert Flatten(ARRANGEMENT).shape.answers == (FLAT_WIDTH,)

    def test_the_reading_side_keeps_the_arrangement(self) -> None:
        assert Flatten(ARRANGEMENT).shape.reads == ARRANGEMENT

    def test_a_dense_layer_can_follow_it(self) -> None:
        """The point of the layer: the join now holds by exact equality."""
        bridge = Flatten((8, 26, 26))

        assert LayerShape(n_inputs=5408, n_outputs=3).follows(bridge.shape)

    def test_a_dense_layer_cannot_follow_the_arrangement_directly(self) -> None:
        """Which is why the bridge has to be stated rather than inferred."""
        picture = LayerShape(n_inputs=(1, 28, 28), n_outputs=(8, 26, 26))

        assert not LayerShape(n_inputs=5408, n_outputs=3).follows(picture)

    def test_a_bare_integer_is_the_one_dimensional_case(self) -> None:
        assert Flatten(7).shape == LayerShape(n_inputs=7, n_outputs=7)

    def test_an_extent_of_zero_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Flatten((8, 0, 13))

    def test_a_fractional_extent_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Flatten((8, 2.5))  # type: ignore[list-item]

    def test_it_says_both_sides_in_its_repr(self) -> None:
        assert repr(Flatten(ARRANGEMENT)) == ("Flatten(reads=(2, 3, 2), answers=(12,))")


class TestTheForwardPass:
    def test_it_answers_one_row_per_observation(self) -> None:
        response = Flatten(ARRANGEMENT).respond_to(counting_block(4))

        assert response.outputs.shape == (4, FLAT_WIDTH)

    def test_it_preserves_the_order_of_the_values(self) -> None:
        """C order, last extent fastest, which is what the backward step undoes."""
        response = Flatten(ARRANGEMENT).respond_to(counting_block(2))

        assert np.array_equal(
            response.outputs[0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        assert np.array_equal(
            response.outputs[1],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        )

    def test_each_row_stays_with_its_own_observation(self) -> None:
        """A block filled in visiting order rather than row order fails here."""
        block = np.stack([np.full(ARRANGEMENT, 5.0), np.full(ARRANGEMENT, -5.0)])

        response = Flatten(ARRANGEMENT).respond_to(block)

        assert np.array_equal(response.outputs[0], np.full(FLAT_WIDTH, 5.0))
        assert np.array_equal(response.outputs[1], np.full(FLAT_WIDTH, -5.0))

    def test_the_scores_and_the_outputs_are_the_same_block(self) -> None:
        """There is no activation here, so a pre-activation value is the value."""
        response = Flatten(ARRANGEMENT).respond_to(counting_block())

        assert np.array_equal(response.scores, response.outputs)

    def test_it_keeps_the_block_it_read(self) -> None:
        """The backward walk reads this rather than reconstructing it."""
        block = counting_block()

        response = Flatten(ARRANGEMENT).respond_to(block)

        assert np.array_equal(response.inputs, block)

    def test_the_answer_is_frozen(self) -> None:
        """It shares a buffer with the inputs, so the sharing must be unwritable."""
        response = Flatten(ARRANGEMENT).respond_to(counting_block())

        with pytest.raises(ValueError):
            response.outputs[0, 0] = 99.0

    def test_a_nested_list_is_accepted(self) -> None:
        block = counting_block(1).tolist()

        response = Flatten(ARRANGEMENT).respond_to(block)  # type: ignore[arg-type]

        assert response.outputs.shape == (1, FLAT_WIDTH)

    def test_the_one_dimensional_case_is_the_identity(self) -> None:
        rows = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        assert np.array_equal(Flatten(3).respond_to(rows).outputs, rows)


class TestTheForwardPassRefusals:
    def test_a_differently_arranged_block_of_the_same_size_is_refused(self) -> None:
        """(1, 12, 1) holds the same twelve numbers and is not the same input.

        The whole argument for this layer, asserted directly: equal counts are
        not an agreement, so a count comparison would have let this through.
        """
        with pytest.raises(ShapeMismatchError):
            Flatten(ARRANGEMENT).respond_to(np.ones((2, 1, 12, 1)))

    def test_a_transposed_arrangement_is_refused(self) -> None:
        """(2, 2, 3) is (2, 3, 2) with two extents swapped, and just as wrong."""
        with pytest.raises(ShapeMismatchError):
            Flatten(ARRANGEMENT).respond_to(np.ones((2, 2, 2, 3)))

    def test_an_already_flat_block_is_refused(self) -> None:
        """Equal counts, wrong number of dimensions, and still not an agreement."""
        with pytest.raises(ShapeMismatchError):
            Flatten(ARRANGEMENT).respond_to(np.ones((2, FLAT_WIDTH)))

    def test_a_block_with_too_many_dimensions_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            Flatten(ARRANGEMENT).respond_to(np.ones((2, 1, 2, 3, 2)))

    def test_a_block_of_no_rows_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            Flatten(ARRANGEMENT).respond_to(np.ones((0, *ARRANGEMENT)))

    def test_a_non_finite_entry_is_refused(self) -> None:
        """A relabelling propagates a nan perfectly, so it has to stop here."""
        block = counting_block()
        block[1, 0, 2, 1] = np.nan

        with pytest.raises(InvalidValuesError):
            Flatten(ARRANGEMENT).respond_to(block)

    def test_an_infinite_entry_is_refused(self) -> None:
        block = counting_block()
        block[0, 1, 1, 0] = np.inf

        with pytest.raises(InvalidValuesError):
            Flatten(ARRANGEMENT).respond_to(block)

    def test_a_block_that_is_not_numbers_at_all_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Flatten(ARRANGEMENT).respond_to([[1.0, 2.0], [3.0]])  # type: ignore[arg-type]


class TestTheBackwardPass:
    def test_the_round_trip_restores_the_arrangement(self) -> None:
        """Flatten, then un-flatten, and the original block comes back."""
        layer = Flatten(ARRANGEMENT)
        block = counting_block(3)
        response = layer.respond_to(block)

        correction = layer.correction_for(response, response.outputs)

        assert correction.passed_down.shape == (3, *ARRANGEMENT)
        assert np.array_equal(correction.passed_down, block)

    def test_it_returns_each_slope_to_its_own_position(self) -> None:
        """Distinguishable slopes, so a shuffled mapping cannot survive."""
        layer = Flatten(ARRANGEMENT)
        response = layer.respond_to(counting_block(1))
        arriving = np.arange(100.0, 100.0 + FLAT_WIDTH).reshape(1, FLAT_WIDTH)

        correction = layer.correction_for(response, arriving)

        assert np.array_equal(correction.passed_down, arriving.reshape(1, *ARRANGEMENT))

    def test_it_has_nothing_to_learn(self) -> None:
        """None rather than a block of zeros, which would be a small lie."""
        layer = Flatten(ARRANGEMENT)
        response = layer.respond_to(counting_block())

        correction = layer.correction_for(response, np.ones((2, FLAT_WIDTH)))

        assert correction.gradient is None
        assert not correction.learns

    def test_arriving_slopes_of_the_wrong_width_are_refused(self) -> None:
        layer = Flatten(ARRANGEMENT)
        response = layer.respond_to(counting_block())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, FLAT_WIDTH + 1)))

    def test_arriving_slopes_for_the_wrong_number_of_rows_are_refused(self) -> None:
        layer = Flatten(ARRANGEMENT)
        response = layer.respond_to(counting_block(2))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((3, FLAT_WIDTH)))

    def test_still_arranged_slopes_are_refused(self) -> None:
        """What arrives from above is a row, because that is what this layer answers."""
        layer = Flatten(ARRANGEMENT)
        response = layer.respond_to(counting_block())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, *ARRANGEMENT)))

    def test_slopes_that_are_not_numbers_are_refused(self) -> None:
        layer = Flatten(ARRANGEMENT)
        response = layer.respond_to(counting_block())

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, [[1.0, 2.0], [3.0]])  # type: ignore[arg-type]


class TestTheGradientCheck:
    """Nudge each input value, watch a summary of the outputs, compare.

    This is the test the backward pass exists to pass. It knows nothing about
    reshapes; it only runs the layer forward, many times.
    """

    def summary_of(self, layer: Flatten, block: np.ndarray) -> float:
        """A scalar reading of the outputs, deliberately not linear.

        ``sum(sin(outputs))`` has slope ``cos(outputs)`` at each output, so the
        arriving block below is a genuine derivative taken at the values that
        actually landed in those positions rather than a block of constants
        that would agree with any mapping.
        """
        return float(np.sum(np.sin(layer.respond_to(block).outputs)))

    def numeric_slopes(self, layer: Flatten, block: np.ndarray) -> np.ndarray:
        """How the summary really moves when each input value is nudged."""
        step = 1e-6
        measured = np.empty_like(block)

        for position in np.ndindex(block.shape):
            moved = []
            for direction in (+step, -step):
                nudged = np.array(block, copy=True)
                nudged[position] += direction
                moved.append(self.summary_of(layer, nudged))
            measured[position] = (moved[0] - moved[1]) / (2.0 * step)

        return measured

    def test_the_slopes_passed_down_match_a_finite_difference(self) -> None:
        layer = Flatten(ARRANGEMENT)
        block = np.random.default_rng(11).normal(size=(3, *ARRANGEMENT))
        response = layer.respond_to(block)
        arriving = np.cos(response.outputs)

        claimed = layer.correction_for(response, arriving).passed_down
        measured = self.numeric_slopes(layer, block)

        assert claimed.shape == block.shape
        assert np.allclose(claimed, measured, atol=1e-7)

    def test_the_check_also_holds_on_a_lopsided_arrangement(self) -> None:
        """Equal extents can hide a transposition; unequal ones cannot."""
        layer = Flatten((1, 4, 2))
        block = np.random.default_rng(19).normal(size=(2, 1, 4, 2))
        response = layer.respond_to(block)
        arriving = np.cos(response.outputs)

        claimed = layer.correction_for(response, arriving).passed_down
        measured = self.numeric_slopes(layer, block)

        assert np.allclose(claimed, measured, atol=1e-7)


class TestStepping:
    def test_a_step_answers_with_the_very_same_layer(self) -> None:
        """Nothing in it can move, and it is immutable, so there is no copy to make."""
        layer = Flatten(ARRANGEMENT)

        assert layer.stepped_by(None, learning_rate=0.1) is layer

    def test_a_gradient_it_never_asked_for_moves_nothing(self) -> None:
        layer = Flatten(ARRANGEMENT)
        gradient = LayerGradient(weights=np.ones((2, 3)), biases=np.ones(2))

        assert layer.stepped_by(gradient, learning_rate=0.5) is layer

    def test_a_step_leaves_its_shape_alone(self) -> None:
        layer = Flatten(ARRANGEMENT)

        assert layer.stepped_by(None, learning_rate=0.1).shape == layer.shape
