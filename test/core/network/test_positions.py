"""Spec for the fixed pattern that tells a step where it sits.

Two of these carry the module.

The first is the claim the layer exists for, and it is asserted directly rather
than described: two identical measurements at two different steps come out of
this layer differing. Everything downstream of it reads a set of steps rather
than a sequence, so without this the third word of a sentence and the tenth are
the same evidence, and a spec that only checked shapes would pass for a layer
that added nothing at all.

The second is the finite-difference check. This layer learns nothing, so the
whole of its backward pass is the claim that adding a constant does not change a
slope, which is the kind of derivation that is obviously right and occasionally
written down wrong -- scaled by the pattern, say, or with the pattern subtracted
back off. The check knows none of that; it runs the layer forward many times.

The pattern's own numbers are pinned twice over. Once at step zero, where every
sine is zero and every cosine is one whatever the wavelength, and once against
hand-worked values on a four-dimensional fixture at a wavelength of one hundred,
where the two frequencies are exactly ``1.0`` and ``0.1`` and so a pair read from
the wrong column is a different number rather than a near one.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.positions import PositionalEncoding
from oop_ml.core.network.shape import LayerShape

#: The step a central difference takes on either side.
NUDGE = 1e-6

N_STEPS = 3
N_DIMENSIONS = 4
WAVELENGTH = 100.0


def worked_layer() -> PositionalEncoding:
    """The fixture whose pattern is worked by hand below."""
    return PositionalEncoding(reads=(N_STEPS, N_DIMENSIONS), wavelength=WAVELENGTH)


def sequence_block(n_rows: int = 2) -> np.ndarray:
    """Rows of a sequence, every value distinct so nothing cancels."""
    total = n_rows * N_STEPS * N_DIMENSIONS
    return (np.arange(total, dtype=np.float64) / 10.0 - 1.0).reshape(
        n_rows, N_STEPS, N_DIMENSIONS
    )


def summary_of(
    layer: PositionalEncoding, block: np.ndarray, weights: np.ndarray
) -> float:
    """A scalar reading of the answer whose slope is ``weights`` exactly."""
    return float(np.sum(weights * layer.respond_to(block).outputs))


def measured_input_slopes(
    layer: PositionalEncoding, block: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every input value's slope, by nudging it and watching the summary move."""
    measured = np.zeros_like(block)
    for index in np.ndindex(block.shape):
        raised = np.array(block, copy=True)
        lowered = np.array(block, copy=True)
        raised[index] += NUDGE
        lowered[index] -= NUDGE
        measured[index] = (
            summary_of(layer, raised, arriving) - summary_of(layer, lowered, arriving)
        ) / (2.0 * NUDGE)
    return measured


class TestConstruction:
    def test_it_answers_with_what_it_reads(self) -> None:
        layer = PositionalEncoding(reads=(12, 8))

        assert layer.shape == LayerShape(n_inputs=(12, 8), n_outputs=(12, 8))

    def test_it_moves_no_join(self) -> None:
        """So it can be dropped into a sound stack without making it unsound."""
        layer = PositionalEncoding(reads=(12, 8))

        assert layer.shape.reads == layer.shape.answers

    def test_the_pattern_is_one_vector_per_step(self) -> None:
        assert PositionalEncoding(reads=(12, 8)).pattern.shape == (12, 8)

    def test_the_pattern_is_frozen(self) -> None:
        with pytest.raises(ValueError):
            PositionalEncoding(reads=(12, 8)).pattern[0, 0] = 99.0

    def test_a_bare_integer_is_refused_by_name(self) -> None:
        """It is what a dense layer's width looks like, so it earns a sentence."""
        with pytest.raises(ShapeMismatchError):
            PositionalEncoding(reads=8)  # type: ignore[arg-type]

    def test_three_extents_are_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            PositionalEncoding(reads=(1, 12, 8))

    def test_an_extent_of_zero_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            PositionalEncoding(reads=(0, 8))

    def test_a_fractional_extent_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            PositionalEncoding(reads=(12, 2.5))  # type: ignore[arg-type]

    @pytest.mark.parametrize("wavelength", [0.0, -10.0, np.nan, np.inf])
    def test_an_unusable_wavelength_is_refused(self, wavelength: float) -> None:
        with pytest.raises(InvalidValuesError):
            PositionalEncoding(reads=(12, 8), wavelength=wavelength)

    def test_a_wavelength_of_one_is_permitted(self) -> None:
        """Degenerate rather than invalid: every pair carries the same wave."""
        layer = PositionalEncoding(reads=(3, 4), wavelength=1.0)

        assert np.allclose(layer.pattern[:, 0], layer.pattern[:, 2])

    def test_it_says_its_geometry_in_its_repr(self) -> None:
        assert repr(worked_layer()) == (
            "PositionalEncoding(reads=(3, 4), wavelength=100.0)"
        )


class TestThePattern:
    def test_the_first_step_is_all_sines_of_zero_and_cosines_of_zero(self) -> None:
        """True at every wavelength, so it pins the interleaving on its own."""
        pattern = PositionalEncoding(reads=(4, 6), wavelength=37.0).pattern

        assert np.allclose(pattern[0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    def test_the_worked_row_at_step_one(self) -> None:
        """Frequencies 1.0 and 0.1 exactly, at a wavelength of one hundred."""
        pattern = worked_layer().pattern

        assert np.allclose(
            pattern[1], [np.sin(1.0), np.cos(1.0), np.sin(0.1), np.cos(0.1)]
        )

    def test_the_worked_row_at_step_two(self) -> None:
        pattern = worked_layer().pattern

        assert np.allclose(
            pattern[2], [np.sin(2.0), np.cos(2.0), np.sin(0.2), np.cos(0.2)]
        )

    def test_the_wavelengths_are_a_geometric_series(self) -> None:
        """Which is what makes a wide layer describe both near and far order."""
        layer = PositionalEncoding(reads=(2, 6), wavelength=1000.0)

        assert np.allclose(
            layer.pattern[1, 0::2],
            [np.sin(1.0), np.sin(1.0 / 10.0), np.sin(1.0 / 100.0)],
        )

    def test_an_odd_width_leaves_the_last_column_a_sine(self) -> None:
        """The cosine has no column to land in, and is dropped rather than wrapped."""
        layer = PositionalEncoding(reads=(2, 3), wavelength=100.0)

        assert np.allclose(
            layer.pattern[1],
            [np.sin(1.0), np.cos(1.0), np.sin(100.0 ** (-2.0 / 3.0))],
        )

    def test_no_two_steps_share_a_pattern(self) -> None:
        """The whole claim: a step's position is recoverable from what is added."""
        pattern = PositionalEncoding(reads=(20, 8)).pattern

        for first in range(pattern.shape[0]):
            for second in range(first + 1, pattern.shape[0]):
                assert not np.allclose(pattern[first], pattern[second])

    def test_the_pattern_is_bounded(self) -> None:
        """Sines and cosines, so it cannot swamp the measurements it is added to."""
        pattern = PositionalEncoding(reads=(50, 16)).pattern

        assert np.all(np.abs(pattern) <= 1.0)


class TestTheForwardPass:
    def test_it_adds_the_pattern_to_every_step(self) -> None:
        layer = worked_layer()
        block = sequence_block()

        response = layer.respond_to(block)

        assert np.allclose(response.outputs, block + layer.pattern)

    def test_every_row_gets_the_same_pattern(self) -> None:
        """The pattern is about position within a sequence, not about which row."""
        layer = worked_layer()
        block = np.zeros((3, N_STEPS, N_DIMENSIONS))

        response = layer.respond_to(block)

        assert np.allclose(response.outputs[0], response.outputs[1])
        assert np.allclose(response.outputs[0], response.outputs[2])

    def test_two_identical_steps_come_out_different(self) -> None:
        """The reason the layer exists, asserted rather than described."""
        layer = worked_layer()
        repeated = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (1, N_STEPS, 1))

        response = layer.respond_to(repeated)

        assert not np.allclose(response.outputs[0, 0], response.outputs[0, 1])

    def test_the_scores_and_the_outputs_are_the_same_block(self) -> None:
        response = worked_layer().respond_to(sequence_block())

        assert np.array_equal(response.scores, response.outputs)

    def test_it_keeps_the_block_it_read(self) -> None:
        block = sequence_block()

        response = worked_layer().respond_to(block)

        assert np.allclose(response.inputs, block)

    def test_the_answer_is_frozen(self) -> None:
        response = worked_layer().respond_to(sequence_block())

        with pytest.raises(ValueError):
            response.outputs[0, 0, 0] = 99.0

    def test_a_wrongly_arranged_block_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            worked_layer().respond_to(np.zeros((2, N_DIMENSIONS, N_STEPS)))

    def test_a_flat_block_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            worked_layer().respond_to(np.zeros((2, N_STEPS * N_DIMENSIONS)))

    def test_a_block_of_no_rows_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            worked_layer().respond_to(np.zeros((0, N_STEPS, N_DIMENSIONS)))

    def test_a_non_finite_entry_is_refused(self) -> None:
        block = sequence_block()
        block[1, 2, 0] = np.nan

        with pytest.raises(InvalidValuesError):
            worked_layer().respond_to(block)


class TestTheBackwardPass:
    def test_it_passes_the_arriving_block_straight_down(self) -> None:
        """Adding a constant changes no slope, so nothing here is arithmetic."""
        layer = worked_layer()
        response = layer.respond_to(sequence_block())
        arriving = np.random.default_rng(2).normal(size=(2, N_STEPS, N_DIMENSIONS))

        correction = layer.correction_for(response, arriving)

        assert np.allclose(correction.passed_down, arriving)

    def test_it_has_nothing_to_learn(self) -> None:
        layer = worked_layer()
        response = layer.respond_to(sequence_block())

        correction = layer.correction_for(response, np.ones((2, N_STEPS, N_DIMENSIONS)))

        assert correction.gradient is None
        assert not correction.learns

    def test_arriving_slopes_of_the_wrong_arrangement_are_refused(self) -> None:
        layer = worked_layer()
        response = layer.respond_to(sequence_block())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, N_STEPS * N_DIMENSIONS)))

    def test_arriving_slopes_for_the_wrong_number_of_rows_are_refused(self) -> None:
        layer = worked_layer()
        response = layer.respond_to(sequence_block(2))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((3, N_STEPS, N_DIMENSIONS)))

    def test_slopes_that_are_not_numbers_are_refused(self) -> None:
        layer = worked_layer()
        response = layer.respond_to(sequence_block())

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, [[1.0, 2.0], [3.0]])  # type: ignore[arg-type]


class TestTheGradientCheck:
    def test_the_slopes_passed_down_match_a_finite_difference(self) -> None:
        layer = worked_layer()
        block = np.random.default_rng(13).normal(size=(2, N_STEPS, N_DIMENSIONS))
        arriving = np.random.default_rng(17).normal(size=(2, N_STEPS, N_DIMENSIONS))
        response = layer.respond_to(block)

        claimed = layer.correction_for(response, arriving).passed_down
        measured = measured_input_slopes(layer, block, arriving)

        assert claimed.shape == block.shape
        assert np.allclose(claimed, measured, atol=1e-7)


class TestStepping:
    def test_a_step_answers_with_the_very_same_layer(self) -> None:
        layer = worked_layer()

        assert layer.stepped_by(None, learning_rate=0.1) is layer

    def test_a_gradient_it_never_asked_for_moves_nothing(self) -> None:
        layer = worked_layer()
        gradient = LayerGradient(weights=np.ones((2, 3)), biases=np.ones(2))

        assert layer.stepped_by(gradient, learning_rate=0.5) is layer

    def test_a_step_leaves_the_pattern_alone(self) -> None:
        layer = worked_layer()

        stepped = layer.stepped_by(None, learning_rate=0.1)

        assert np.array_equal(stepped.pattern, layer.pattern)
