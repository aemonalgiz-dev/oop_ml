"""Spec for attention, where three of the tests could each hide a working bug.

The finite-difference check carries this module for the reason
``test_convolution.py`` and ``test_recurrent.py`` both set out, and attention
adds three ways to be wrong that all produce finite, plausible, correctly
shaped numbers and a network that trains.

The first is the squashing's slope. Each step's attention along the sequence
adds to one, so raising one share lowers the rest, and the slope is a whole
matrix per step rather than a number per entry. Written as though the shares
were independent it loses the subtracted term, and what is left still points
downhill often enough to train.

The second is the pair of blocks that reach the same comparison. A query and a
key both feed one score, so both collect a slope from it, and the two are
transposes of each other's arithmetic. Swapping them is undetectable by shape
and, on a fixture where the sequence is short and the heads are one, almost
undetectable by eye.

The third is the heads. Splitting the width into heads and putting it back is a
reshape and a transpose in each direction, and getting either the wrong way
round mixes one head's answer into another's slice. Every head still answers,
the shapes all agree, and the layer learns something.

So the checks are run over the weights, the biases and the inputs, with one head
and with three, and both with and without a step being allowed to read ahead,
since masking changes which entries of the squashing are alive.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    MLLibError,
    ShapeMismatchError,
)
from oop_ml.core.network.attention import AttentionResponse, MultiHeadAttention
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.layer import LayerResponse
from oop_ml.core.network.shape import LayerShape

#: The step a central difference takes on either side.
NUDGE = 1e-6

#: One head and several, since the reshape only has room to be wrong above one.
HEAD_COUNTS = [1, 2, 3]


def four_step_sequence() -> np.ndarray:
    """Two rows of a four-step sequence over six numbers, no two alike."""
    return np.linspace(-0.8, 0.9, 2 * 4 * 6).reshape(2, 4, 6)


def spread_parameters(layer: MultiHeadAttention) -> MultiHeadAttention:
    """The same layer with parameters no two of which coincide.

    A fixed progression rather than a draw, so a failure is reproducible and no
    projection's rows are a copy of another's. A query band read where the key
    band was meant then answers differently rather than the same.
    """
    rows, columns = layer.weights.shape
    count = rows * columns
    weights = (np.arange(count, dtype=np.float64) / count - 0.5).reshape(rows, columns)
    biases = np.arange(rows, dtype=np.float64) / rows - 0.25
    return layer.with_parameters(weights, biases)


def summary_of(
    layer: MultiHeadAttention, rows: np.ndarray, weights: np.ndarray
) -> float:
    """A scalar reading of the answer, whose slope is ``weights`` exactly."""
    return float(np.sum(weights * layer.respond_to(rows).outputs))


def measured_weight_slopes(
    layer: MultiHeadAttention, rows: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every projection weight's slope, by nudging it and watching the summary."""
    measured = np.zeros_like(layer.weights)
    for row in range(layer.weights.shape[0]):
        for column in range(layer.weights.shape[1]):
            raised = np.array(layer.weights, copy=True)
            lowered = np.array(layer.weights, copy=True)
            raised[row, column] += NUDGE
            lowered[row, column] -= NUDGE
            measured[row, column] = (
                summary_of(layer.with_parameters(raised, layer.biases), rows, arriving)
                - summary_of(
                    layer.with_parameters(lowered, layer.biases), rows, arriving
                )
            ) / (2.0 * NUDGE)
    return measured


def measured_bias_slopes(
    layer: MultiHeadAttention, rows: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every bias's slope, the same way."""
    measured = np.zeros_like(layer.biases)
    for position in range(layer.biases.shape[0]):
        raised = np.array(layer.biases, copy=True)
        lowered = np.array(layer.biases, copy=True)
        raised[position] += NUDGE
        lowered[position] -= NUDGE
        measured[position] = (
            summary_of(layer.with_parameters(layer.weights, raised), rows, arriving)
            - summary_of(layer.with_parameters(layer.weights, lowered), rows, arriving)
        ) / (2.0 * NUDGE)
    return measured


def measured_input_slopes(
    layer: MultiHeadAttention, rows: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every input value's slope, by nudging the sequence itself."""
    measured = np.zeros_like(rows)
    for index in np.ndindex(rows.shape):
        raised = np.array(rows, copy=True)
        lowered = np.array(rows, copy=True)
        raised[index] += NUDGE
        lowered[index] -= NUDGE
        measured[index] = (
            summary_of(layer, raised, arriving) - summary_of(layer, lowered, arriving)
        ) / (2.0 * NUDGE)
    return measured


class TestConstruction:
    def test_a_step_keeps_its_width(self) -> None:
        layer = MultiHeadAttention(reads=(10, 8), n_heads=2)

        assert layer.shape == LayerShape(n_inputs=(10, 8), n_outputs=(10, 8))

    @pytest.mark.parametrize("n_heads", HEAD_COUNTS)
    def test_the_heads_split_the_width_between_them(self, n_heads: int) -> None:
        layer = MultiHeadAttention(reads=(4, 6), n_heads=n_heads)

        assert layer.head_size == 6 // n_heads
        assert layer.n_heads * layer.head_size == 6

    def test_the_four_projections_stack_into_one_block(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6), n_heads=2)

        assert layer.weights.shape == (4 * 6, 6)
        assert layer.biases.shape == (4 * 6,)

    def test_a_width_the_heads_cannot_split_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError, match="divide"):
            MultiHeadAttention(reads=(4, 7), n_heads=2)

    def test_a_bare_width_is_refused_by_name(self) -> None:
        with pytest.raises(ShapeMismatchError, match="n_steps, n_model"):
            MultiHeadAttention(reads=6)  # type: ignore[arg-type]

    def test_a_picture_arrangement_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError, match="extents"):
            MultiHeadAttention(reads=(1, 4, 4))

    @pytest.mark.parametrize(
        ("reads", "n_heads"), [((0, 6), 1), ((4, 0), 1), ((4, 6), 0), ((4, 6), -2)]
    )
    def test_a_setting_below_its_floor_is_refused(
        self, reads: tuple[int, int], n_heads: int
    ) -> None:
        with pytest.raises(InvalidValuesError):
            MultiHeadAttention(reads=reads, n_heads=n_heads)

    def test_a_seed_fixes_the_starting_projections(self) -> None:
        first = MultiHeadAttention(reads=(4, 6), random_seed=5)
        again = MultiHeadAttention(reads=(4, 6), random_seed=5)
        other = MultiHeadAttention(reads=(4, 6), random_seed=6)

        assert np.array_equal(first.weights, again.weights)
        assert not np.array_equal(first.weights, other.weights)

    def test_the_projections_cannot_be_edited_from_outside(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))

        with pytest.raises(ValueError):
            layer.weights[0, 0] = 1.0


class TestWhoReadsWhom:
    def test_every_step_reads_shares_that_add_to_one(self) -> None:
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6), n_heads=2))

        attention = layer.respond_to(four_step_sequence()).attention

        assert np.allclose(attention.sum(axis=-1), 1.0)

    def test_the_attention_block_names_every_pair_of_steps_in_every_head(
        self,
    ) -> None:
        layer = MultiHeadAttention(reads=(4, 6), n_heads=3)

        attention = layer.respond_to(four_step_sequence()).attention

        assert attention.shape == (2, 3, 4, 4)

    def test_a_step_reads_nothing_after_itself_when_it_may_not_look_ahead(
        self,
    ) -> None:
        layer = spread_parameters(
            MultiHeadAttention(reads=(4, 6), n_heads=2, looks_ahead=False)
        )

        attention = layer.respond_to(four_step_sequence()).attention

        above_the_diagonal = np.triu(np.ones((4, 4), dtype=bool), k=1)
        assert np.all(attention[:, :, above_the_diagonal] == 0.0)

    def test_the_shares_still_add_to_one_under_the_mask(self) -> None:
        """The reason the stopping happens before the squashing and not after.

        Zeroing the weights afterwards would leave the early steps adding to
        less than one, which shrinks their answers without anything raising.
        """
        layer = spread_parameters(
            MultiHeadAttention(reads=(4, 6), n_heads=2, looks_ahead=False)
        )

        attention = layer.respond_to(four_step_sequence()).attention

        assert np.allclose(attention.sum(axis=-1), 1.0)

    def test_the_first_step_reads_only_itself_when_it_may_not_look_ahead(
        self,
    ) -> None:
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6), looks_ahead=False))

        attention = layer.respond_to(four_step_sequence()).attention

        assert np.allclose(attention[:, :, 0, 0], 1.0)

    def test_a_masked_layer_ignores_what_comes_after_a_step(self) -> None:
        """The claim the mask exists for, asserted on the answer rather than the
        weights. Changing a later step must not move an earlier step's answer.
        """
        layer = spread_parameters(
            MultiHeadAttention(reads=(4, 6), n_heads=2, looks_ahead=False)
        )
        sequence = four_step_sequence()
        altered = np.array(sequence, copy=True)
        altered[:, 3, :] += 5.0

        before = layer.respond_to(sequence).outputs
        after = layer.respond_to(altered).outputs

        assert np.allclose(before[:, :3, :], after[:, :3, :])
        assert not np.allclose(before[:, 3, :], after[:, 3, :])

    def test_an_unmasked_layer_does_not_ignore_it(self) -> None:
        """The control for the test above, so that one cannot pass vacuously."""
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6), n_heads=2))
        sequence = four_step_sequence()
        altered = np.array(sequence, copy=True)
        altered[:, 3, :] += 5.0

        before = layer.respond_to(sequence).outputs
        after = layer.respond_to(altered).outputs

        assert not np.allclose(before[:, 0, :], after[:, 0, :])

    def test_the_answer_is_the_gathered_values_through_the_last_projection(
        self,
    ) -> None:
        """Worked from the definition rather than from the implementation."""
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6), n_heads=2))
        response = layer.respond_to(four_step_sequence())

        gathered = response.attention @ response.values
        merged = np.transpose(gathered, (0, 2, 1, 3)).reshape(2, 4, 6)
        expected = merged @ layer.weights[18:24].T + layer.biases[18:24]

        assert np.allclose(response.outputs, expected)


class TestTheForwardPass:
    def test_a_block_of_the_wrong_arrangement_is_refused(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((2, 5, 6)))

    def test_a_row_block_is_refused_where_a_sequence_is_wanted(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((2, 24)))

    def test_an_empty_block_is_refused(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))

        with pytest.raises(EmptyValuesError):
            layer.respond_to(np.zeros((0, 4, 6)))

    def test_a_non_finite_value_is_refused_where_it_enters(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))
        sequence = four_step_sequence()
        sequence[0, 0, 0] = np.nan

        with pytest.raises(InvalidValuesError):
            layer.respond_to(sequence)

    def test_a_very_large_score_does_not_answer_not_a_number(self) -> None:
        """The subtraction inside the squashing, asserted rather than assumed."""
        layer = MultiHeadAttention(reads=(4, 6), n_heads=2)
        enormous = np.full((2, 4, 6), 400.0)

        answered = layer.respond_to(enormous)

        assert np.all(np.isfinite(answered.outputs))
        assert np.allclose(answered.attention.sum(axis=-1), 1.0)


class TestTheBackwardPass:
    """The finite-difference checks, which stand behind the whole derivation."""

    @pytest.mark.parametrize("n_heads", HEAD_COUNTS)
    @pytest.mark.parametrize("looks_ahead", [True, False])
    def test_every_weight_slope_matches_a_finite_difference(
        self, n_heads: int, looks_ahead: bool
    ) -> None:
        layer = spread_parameters(
            MultiHeadAttention(reads=(4, 6), n_heads=n_heads, looks_ahead=looks_ahead)
        )
        sequence = four_step_sequence()
        arriving = np.linspace(0.2, 1.1, 2 * 4 * 6).reshape(2, 4, 6)

        correction = layer.correction_for(layer.respond_to(sequence), arriving)
        measured = measured_weight_slopes(layer, sequence, arriving)

        assert correction.gradient is not None
        assert np.allclose(correction.gradient.weights, measured, atol=1e-6)

    @pytest.mark.parametrize("n_heads", HEAD_COUNTS)
    @pytest.mark.parametrize("looks_ahead", [True, False])
    def test_every_bias_slope_matches_a_finite_difference(
        self, n_heads: int, looks_ahead: bool
    ) -> None:
        layer = spread_parameters(
            MultiHeadAttention(reads=(4, 6), n_heads=n_heads, looks_ahead=looks_ahead)
        )
        sequence = four_step_sequence()
        arriving = np.linspace(0.2, 1.1, 2 * 4 * 6).reshape(2, 4, 6)

        correction = layer.correction_for(layer.respond_to(sequence), arriving)
        measured = measured_bias_slopes(layer, sequence, arriving)

        assert correction.gradient is not None
        assert np.allclose(correction.gradient.biases, measured, atol=1e-6)

    @pytest.mark.parametrize("n_heads", HEAD_COUNTS)
    @pytest.mark.parametrize("looks_ahead", [True, False])
    def test_every_input_slope_matches_a_finite_difference(
        self, n_heads: int, looks_ahead: bool
    ) -> None:
        layer = spread_parameters(
            MultiHeadAttention(reads=(4, 6), n_heads=n_heads, looks_ahead=looks_ahead)
        )
        sequence = four_step_sequence()
        arriving = np.linspace(0.2, 1.1, 2 * 4 * 6).reshape(2, 4, 6)

        correction = layer.correction_for(layer.respond_to(sequence), arriving)
        measured = measured_input_slopes(layer, sequence, arriving)

        assert np.allclose(correction.passed_down, measured, atol=1e-6)

    def test_no_blame_reaches_a_step_from_before_it_under_the_mask(self) -> None:
        """A step nobody may read gets blame from itself alone.

        The last step is read by nothing, so what the loss wants from it comes
        entirely through its own answer. Under an unmasked layer it collects
        from every step, and the contrast is the test.
        """
        sequence = four_step_sequence()
        arriving = np.zeros((2, 4, 6))
        arriving[:, 0, :] = 1.0

        masked = spread_parameters(
            MultiHeadAttention(reads=(4, 6), n_heads=2, looks_ahead=False)
        )
        open_layer = spread_parameters(
            MultiHeadAttention(reads=(4, 6), n_heads=2, looks_ahead=True)
        )

        masked_blame = masked.correction_for(
            masked.respond_to(sequence), arriving
        ).passed_down
        open_blame = open_layer.correction_for(
            open_layer.respond_to(sequence), arriving
        ).passed_down

        assert np.allclose(masked_blame[:, 1:, :], 0.0)
        assert not np.allclose(open_blame[:, 1:, :], 0.0)

    def test_the_passed_down_block_is_shaped_like_the_sequence_read(self) -> None:
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6), n_heads=2))
        sequence = four_step_sequence()

        correction = layer.correction_for(
            layer.respond_to(sequence), np.ones((2, 4, 6))
        )

        assert correction.passed_down.shape == sequence.shape

    def test_a_response_from_another_kind_of_layer_is_refused(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))
        sequence = four_step_sequence()
        plain = LayerResponse(sequence, np.zeros((2, 4, 6)), np.zeros((2, 4, 6)))

        with pytest.raises(ShapeMismatchError, match="workings"):
            layer.correction_for(plain, np.ones((2, 4, 6)))

    def test_an_arriving_block_of_the_wrong_arrangement_is_refused(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))
        response = layer.respond_to(four_step_sequence())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, 4, 9)))


class TestSteppingBy:
    def test_a_step_moves_against_the_slope(self) -> None:
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6), n_heads=2))
        gradient = LayerGradient(
            np.ones_like(layer.weights), np.ones_like(layer.biases)
        )

        stepped = layer.stepped_by(gradient, 0.25)

        assert np.allclose(stepped.weights, layer.weights - 0.25)

    def test_a_step_leaves_the_layer_it_came_from_alone(self) -> None:
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6), n_heads=2))
        before = np.array(layer.weights, copy=True)

        layer.stepped_by(
            LayerGradient(np.ones_like(layer.weights), np.ones_like(layer.biases)),
            0.25,
        )

        assert np.array_equal(layer.weights, before)

    def test_a_step_keeps_the_head_count_and_the_mask(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6), n_heads=3, looks_ahead=False)

        stepped = layer.stepped_by(
            LayerGradient(np.zeros_like(layer.weights), np.zeros_like(layer.biases)),
            0.1,
        )

        assert stepped.n_heads == 3
        assert stepped.looks_ahead is False
        assert stepped.shape == layer.shape

    def test_stepping_by_nothing_is_refused(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))

        with pytest.raises(ShapeMismatchError, match="needs a gradient"):
            layer.stepped_by(None, 0.1)

    def test_a_gradient_of_the_wrong_arrangement_is_refused(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))

        with pytest.raises(ShapeMismatchError):
            layer.stepped_by(LayerGradient(np.ones((2, 2)), np.ones(2)), 0.1)

    def test_a_descending_step_lowers_the_summary(self) -> None:
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6), n_heads=2))
        sequence = four_step_sequence()
        arriving = np.full((2, 4, 6), 0.2)

        before = summary_of(layer, sequence, arriving)
        correction = layer.correction_for(layer.respond_to(sequence), arriving)
        after = summary_of(
            layer.stepped_by(correction.gradient, 0.005), sequence, arriving
        )

        assert after < before


class TestTheResponse:
    def test_two_responses_of_different_kinds_are_not_equal_either_way_round(
        self,
    ) -> None:
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6)))
        sequence = four_step_sequence()
        attended = layer.respond_to(sequence)
        plain = LayerResponse(
            sequence, np.array(attended.outputs), np.array(attended.outputs)
        )

        assert attended != plain
        assert plain != attended

    def test_the_workings_cannot_be_edited_through_the_response(self) -> None:
        layer = spread_parameters(MultiHeadAttention(reads=(4, 6)))

        response = layer.respond_to(four_step_sequence())

        with pytest.raises(ValueError):
            response.attention[0, 0, 0, 0] = 1.0

    def test_it_is_the_kind_of_response_that_carries_the_workings(self) -> None:
        layer = MultiHeadAttention(reads=(4, 6))

        assert isinstance(layer.respond_to(four_step_sequence()), AttentionResponse)


class TestEveryRefusalIsThisLibrarysOwn:
    def test_a_bare_width_does_not_escape_as_a_builtin_failure(self) -> None:
        with pytest.raises(MLLibError):
            MultiHeadAttention(reads=6)  # type: ignore[arg-type]
