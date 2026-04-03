"""Spec for the convolution, and one test in it carries the whole module.

The one that matters is the finite-difference check, for the reason
``test_backward.py`` already sets out: almost every wrong backward pass still
runs. A convolution adds three of its own ways to be wrong, and all three
produce finite, plausible, correctly shaped numbers.

The first is forgetting to accumulate the kernel gradient. A kernel entry is
used at every output position, so its slope is a sum over all of them; a body
that assigns instead of adding trains on whichever window happened to be last
and descends anyway, just towards something else.

The second is the blame passed downward. An input value covered by several
windows collects a contribution from each, so the accumulation has to run over
output positions. Written the other way round, the terms lost at the border and
under a stride greater than one are exactly the ones nobody checks by eye.

The third is the padding. Blame accumulates into the bordered block and the
border is then stripped off; an off-by-one there shifts every gradient by one
pixel, which trains a model that works slightly less well forever and raises
nothing ever.

So the check is run three ways -- over the kernels, over the biases, and over
the inputs -- and twice over, once with padding and once with a stride, since
those are the two configurations where the bookkeeping stops being trivial. It
knows nothing about how the gradient was derived; it only runs the layer
forward many times and watches.

The forward numbers are worked by hand rather than generated. A 3x3 picture
under a 2x2 kernel has four output positions and every one of them is a
different number here, so a transposed block, a reversed sweep or a filter
answering in the wrong channel all fail rather than averaging out.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    MLLibError,
    ShapeMismatchError,
)
from oop_ml.core.network.activation import HyperbolicTangent, Identity, RectifiedLinear
from oop_ml.core.network.convolution import Conv2d
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.shape import LayerShape

#: One picture, one channel, the numbers 1 to 9 in reading order.
COUNTING_PICTURE = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])

#: The step a central difference takes on either side.
NUDGE = 1e-6


def hand_worked_layer() -> Conv2d:
    """Two filters over a 1x3x3 picture, both with kernels chosen on paper.

    The first filter is ``[[1, 2], [3, 4]]`` with a bias of ``0.5``, so no two
    of its four answers coincide. The second reads only the bottom right of its
    window, with a bias of ``-1.0``, so it answers a different number in a
    different channel and a block filled in visiting order cannot pass.
    """
    layer = Conv2d(reads=(1, 3, 3), n_filters=2, kernel_size=2, activation=Identity())
    kernels = np.array(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[0.0, 0.0], [0.0, 1.0]]],
        ]
    )
    return layer.with_parameters(kernels, np.array([0.5, -1.0]))


def summary_of(layer: Conv2d, rows: np.ndarray, weights: np.ndarray) -> float:
    """A scalar reading of the whole answer, whose slope is ``weights`` exactly.

    ``sum(weights * outputs)`` differentiates to ``weights`` at every output,
    so the same block is both the scalar's definition and the arriving block a
    backward pass is handed. That is what makes the finite difference and the
    claim comparable without a loss object in between.
    """
    return float(np.sum(weights * layer.respond_to(rows).outputs))


class TestConstruction:
    def test_its_shape_names_both_arrangements(self) -> None:
        layer = Conv2d(
            reads=(1, 28, 28), n_filters=8, kernel_size=3, activation=Identity()
        )

        assert layer.shape == LayerShape(n_inputs=(1, 28, 28), n_outputs=(8, 26, 26))

    @pytest.mark.parametrize(
        ("height", "width", "kernel_size", "stride", "padding", "answers"),
        [
            (28, 28, 3, 1, 0, (26, 26)),
            (28, 28, 5, 1, 2, (28, 28)),
            (28, 28, 3, 2, 0, (13, 13)),
            (32, 32, 3, 1, 1, (32, 32)),
            (7, 7, 3, 2, 1, (4, 4)),
            (5, 5, 2, 2, 0, (2, 2)),
            (10, 6, 3, 3, 0, (3, 2)),
            (1, 1, 1, 1, 0, (1, 1)),
        ],
    )
    def test_the_output_extents_are_the_arithmetic(
        self,
        height: int,
        width: int,
        kernel_size: int,
        stride: int,
        padding: int,
        answers: tuple[int, int],
    ) -> None:
        """``(extent - kernel + 2 * padding) // stride + 1``, per spatial axis."""
        layer = Conv2d(
            reads=(1, height, width),
            n_filters=4,
            kernel_size=kernel_size,
            activation=Identity(),
            stride=stride,
            padding=padding,
        )

        assert layer.shape.answers == (4, *answers)

    def test_the_filter_count_is_the_answer_s_channel_count(self) -> None:
        layer = Conv2d(
            reads=(3, 9, 9), n_filters=6, kernel_size=3, activation=Identity()
        )

        assert layer.shape.answers[0] == 6
        assert layer.n_filters == 6

    def test_the_kernels_read_every_input_channel(self) -> None:
        layer = Conv2d(
            reads=(3, 9, 9), n_filters=6, kernel_size=4, activation=Identity()
        )

        assert layer.kernels.shape == (6, 3, 4, 4)
        assert layer.bias_vector.shape == (6,)

    def test_the_parameter_count_owes_nothing_to_the_picture_s_size(self) -> None:
        """Weight sharing, stated as a fact about two shapes rather than prose."""
        small = Conv2d(
            reads=(1, 8, 8), n_filters=4, kernel_size=3, activation=Identity()
        )
        large = Conv2d(
            reads=(1, 64, 64), n_filters=4, kernel_size=3, activation=Identity()
        )

        assert small.kernels.shape == large.kernels.shape
        assert small.bias_vector.shape == large.bias_vector.shape

    def test_the_kernels_are_scaled_by_the_fan_in(self) -> None:
        """He initialisation: a spread of ``sqrt(2 / fan_in)``, not of one."""
        layer = Conv2d(
            reads=(4, 10, 10), n_filters=32, kernel_size=3, activation=Identity()
        )

        expected = float(np.sqrt(2.0 / (4 * 3 * 3)))
        assert float(np.std(layer.kernels)) == pytest.approx(expected, rel=0.1)

    def test_the_biases_start_at_zero(self) -> None:
        layer = Conv2d(
            reads=(2, 6, 6), n_filters=3, kernel_size=3, activation=Identity()
        )

        assert np.allclose(layer.bias_vector, 0.0)

    def test_the_same_seed_gives_the_same_layer(self) -> None:
        first = Conv2d(
            reads=(2, 6, 6),
            n_filters=3,
            kernel_size=3,
            activation=Identity(),
            random_seed=17,
        )
        second = Conv2d(
            reads=(2, 6, 6),
            n_filters=3,
            kernel_size=3,
            activation=Identity(),
            random_seed=17,
        )

        assert np.array_equal(first.kernels, second.kernels)

    def test_a_different_seed_gives_a_different_layer(self) -> None:
        first = Conv2d(
            reads=(2, 6, 6),
            n_filters=3,
            kernel_size=3,
            activation=Identity(),
            random_seed=17,
        )
        second = Conv2d(
            reads=(2, 6, 6),
            n_filters=3,
            kernel_size=3,
            activation=Identity(),
            random_seed=18,
        )

        assert not np.array_equal(first.kernels, second.kernels)

    def test_the_learned_arrays_are_frozen(self) -> None:
        layer = Conv2d(
            reads=(1, 5, 5), n_filters=2, kernel_size=2, activation=Identity()
        )

        with pytest.raises(ValueError):
            layer.kernels[0, 0, 0, 0] = 1.0
        with pytest.raises(ValueError):
            layer.bias_vector[0] = 1.0


class TestRefusedConfigurations:
    def test_a_window_wider_than_the_picture_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            Conv2d(reads=(1, 3, 3), n_filters=1, kernel_size=5, activation=Identity())

    def test_the_refusal_names_the_arithmetic(self) -> None:
        """Every term in it was chosen by the caller, so every term is quoted."""
        with pytest.raises(ShapeMismatchError) as refusal:
            Conv2d(reads=(1, 3, 3), n_filters=1, kernel_size=5, activation=Identity())

        assert "(3 - 5 + 2 * 0) // 1 + 1 = -1" in str(refusal.value)

    def test_a_stride_that_outruns_the_picture_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            Conv2d(
                reads=(1, 4, 4),
                n_filters=1,
                kernel_size=6,
                activation=Identity(),
                stride=3,
            )

    def test_only_one_axis_needs_to_fail(self) -> None:
        """A 2x9 picture takes a 4-wide window across and not down."""
        with pytest.raises(ShapeMismatchError):
            Conv2d(reads=(1, 2, 9), n_filters=1, kernel_size=4, activation=Identity())

    @pytest.mark.parametrize("kernel_size", [0, -1])
    def test_a_kernel_below_one_is_refused(self, kernel_size: int) -> None:
        with pytest.raises(InvalidValuesError):
            Conv2d(
                reads=(1, 5, 5),
                n_filters=1,
                kernel_size=kernel_size,
                activation=Identity(),
            )

    @pytest.mark.parametrize("stride", [0, -2])
    def test_a_stride_below_one_is_refused(self, stride: int) -> None:
        with pytest.raises(InvalidValuesError):
            Conv2d(
                reads=(1, 5, 5),
                n_filters=1,
                kernel_size=2,
                activation=Identity(),
                stride=stride,
            )

    def test_a_negative_padding_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Conv2d(
                reads=(1, 5, 5),
                n_filters=1,
                kernel_size=2,
                activation=Identity(),
                padding=-1,
            )

    def test_a_padding_of_zero_is_allowed(self) -> None:
        """Zero is the ordinary case, which is why its floor is not one."""
        layer = Conv2d(
            reads=(1, 5, 5),
            n_filters=1,
            kernel_size=2,
            activation=Identity(),
            padding=0,
        )

        assert layer.padding == 0

    @pytest.mark.parametrize("n_filters", [0, -3])
    def test_a_filter_count_below_one_is_refused(self, n_filters: int) -> None:
        with pytest.raises(InvalidValuesError):
            Conv2d(
                reads=(1, 5, 5),
                n_filters=n_filters,
                kernel_size=2,
                activation=Identity(),
            )

    @pytest.mark.parametrize("reads", [(5, 5), (1, 5, 5, 5), (5,)])
    def test_a_picture_that_is_not_three_extents_is_refused(
        self, reads: tuple[int, ...]
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            Conv2d(reads=reads, n_filters=1, kernel_size=2, activation=Identity())

    @pytest.mark.parametrize(
        "reads", [5, None, 3.5], ids=["a bare width", "nothing", "a fraction"]
    )
    def test_reads_that_is_not_a_sequence_of_extents_is_refused_by_name(
        self, reads: object
    ) -> None:
        """The refusal has to be one of this library's own, not builtins'.

        ``tuple(5)`` raises ``TypeError: 'int' object is not iterable``, which
        is not an ``MLLibError`` and so escapes the hierarchy every failure here
        is supposed to stay inside. The bare width is the mistake worth naming,
        because it is exactly what a dense layer's width looks like and a reader
        moving between the two will write it.
        """
        with pytest.raises(MLLibError):
            Conv2d(
                reads=reads,  # type: ignore[arg-type]
                n_filters=1,
                kernel_size=2,
                activation=Identity(),
            )

    @pytest.mark.parametrize("extent", [0, -4])
    def test_an_extent_below_one_is_refused(self, extent: int) -> None:
        with pytest.raises(InvalidValuesError):
            Conv2d(
                reads=(1, extent, 5), n_filters=1, kernel_size=1, activation=Identity()
            )

    def test_a_boolean_stride_is_refused(self) -> None:
        """``True`` is an ``int`` in Python, and a stride of one by accident."""
        with pytest.raises(InvalidValuesError):
            Conv2d(
                reads=(1, 5, 5),
                n_filters=1,
                kernel_size=2,
                activation=Identity(),
                stride=True,
            )

    def test_a_fractional_kernel_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Conv2d(
                reads=(1, 5, 5),
                n_filters=1,
                kernel_size=2.5,  # type: ignore[arg-type]
                activation=Identity(),
            )


class TestTheForwardPassWorkedByHand:
    def test_the_first_filter_answers_the_numbers_computed_on_paper(self) -> None:
        """``[[1,2],[3,4]]`` over ``1..9``, plus a bias of ``0.5``."""
        answer = hand_worked_layer().respond_to(COUNTING_PICTURE).outputs

        assert np.allclose(answer[0, 0], [[37.5, 47.5], [67.5, 77.5]])

    def test_the_second_filter_answers_in_its_own_channel(self) -> None:
        """It reads the bottom right of each window and subtracts one."""
        answer = hand_worked_layer().respond_to(COUNTING_PICTURE).outputs

        assert np.allclose(answer[0, 1], [[4.0, 5.0], [7.0, 8.0]])

    def test_the_answer_has_the_shape_the_layer_promised(self) -> None:
        layer = hand_worked_layer()

        answer = layer.respond_to(COUNTING_PICTURE).outputs

        assert answer.shape == (1, *layer.shape.answers)

    def test_a_second_row_is_answered_independently(self) -> None:
        """Two pictures in one block, and neither leaks into the other."""
        block = np.concatenate([COUNTING_PICTURE, COUNTING_PICTURE * 2.0])

        answer = hand_worked_layer().respond_to(block).outputs

        assert np.allclose(answer[0, 0], [[37.5, 47.5], [67.5, 77.5]])
        assert np.allclose(answer[1, 0], [[74.5, 94.5], [134.5, 154.5]])

    def test_a_stride_skips_positions_and_answers_the_windows_it_keeps(self) -> None:
        """A 3x3 window of ones over ``0..24``, two apart, is four block sums."""
        layer = Conv2d(
            reads=(1, 5, 5),
            n_filters=1,
            kernel_size=3,
            activation=Identity(),
            stride=2,
        ).with_parameters(np.ones((1, 1, 3, 3)), np.zeros(1))
        picture = np.arange(25.0).reshape(1, 1, 5, 5)

        answer = layer.respond_to(picture).outputs

        assert np.allclose(answer[0, 0], [[54.0, 72.0], [144.0, 162.0]])

    def test_padding_pads_with_zeros(self) -> None:
        """A 2x2 window of ones over ``[[1,2],[3,4]]`` with one row of border.

        Every number in the answer is a partial sum of the picture, and only a
        border of exact zeros produces this particular nine. Replicating the
        edge, or wrapping it round, changes eight of them.
        """
        layer = Conv2d(
            reads=(1, 2, 2),
            n_filters=1,
            kernel_size=2,
            activation=Identity(),
            padding=1,
        ).with_parameters(np.ones((1, 1, 2, 2)), np.zeros(1))
        picture = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])

        answer = layer.respond_to(picture).outputs

        assert answer.shape == (1, 1, 3, 3)
        assert np.allclose(
            answer[0, 0], [[1.0, 3.0, 2.0], [4.0, 10.0, 6.0], [3.0, 7.0, 4.0]]
        )

    def test_a_filter_answers_the_same_pattern_wherever_it_sits(self) -> None:
        """Translation equivariance, which is the whole reason for the sweep."""
        layer = Conv2d(
            reads=(1, 5, 5), n_filters=1, kernel_size=2, activation=Identity()
        ).with_parameters(np.array([[[[1.0, 2.0], [3.0, 4.0]]]]), np.zeros(1))
        here = np.zeros((1, 1, 5, 5))
        here[0, 0, 1, 1] = 1.0
        moved = np.zeros((1, 1, 5, 5))
        moved[0, 0, 2, 2] = 1.0

        first = layer.respond_to(here).outputs
        second = layer.respond_to(moved).outputs

        assert np.allclose(second[0, 0, 1:, 1:], first[0, 0, :-1, :-1])

    def test_the_scores_and_the_outputs_disagree_once_a_bend_is_applied(self) -> None:
        """A layer whose bend was never applied returns the scores twice."""
        layer = Conv2d(
            reads=(1, 3, 3), n_filters=1, kernel_size=2, activation=RectifiedLinear()
        ).with_parameters(np.full((1, 1, 2, 2), -1.0), np.zeros(1))

        response = layer.respond_to(COUNTING_PICTURE)

        assert np.all(response.scores < 0.0)
        assert np.allclose(response.outputs, 0.0)

    def test_the_response_carries_the_block_that_was_read(self) -> None:
        response = hand_worked_layer().respond_to(COUNTING_PICTURE)

        assert np.allclose(response.inputs, COUNTING_PICTURE)


class TestRefusedBlocks:
    def test_a_block_of_the_wrong_arrangement_is_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((1, 9)))

    def test_a_picture_of_the_wrong_extents_is_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((2, 1, 4, 4)))

    def test_a_block_of_no_rows_is_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(EmptyValuesError):
            layer.respond_to(np.zeros((0, 1, 3, 3)))

    def test_a_non_finite_value_is_refused_where_it_enters(self) -> None:
        """One ``nan`` reaches every window that covers it and then the stack."""
        layer = hand_worked_layer()
        poisoned = np.array(COUNTING_PICTURE, copy=True)
        poisoned[0, 0, 1, 1] = np.nan

        with pytest.raises(InvalidValuesError):
            layer.respond_to(poisoned)

    def test_a_block_that_is_not_numbers_at_all_is_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(InvalidValuesError):
            layer.respond_to("a picture")  # type: ignore[arg-type]

    def test_parameters_of_the_wrong_shape_are_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(ShapeMismatchError):
            layer.with_parameters(np.zeros((2, 1, 3, 3)), np.zeros(2))

    def test_biases_of_the_wrong_length_are_refused(self) -> None:
        layer = hand_worked_layer()

        with pytest.raises(ShapeMismatchError):
            layer.with_parameters(np.zeros((2, 1, 2, 2)), np.zeros(3))


class TestTheCorrectionsShape:
    def test_the_kernel_gradient_arrives_flattened_to_two_dimensions(self) -> None:
        """``LayerGradient`` is ``(n_neurons, n_inputs)``, so a filter is a row."""
        layer = Conv2d(
            reads=(2, 5, 5), n_filters=3, kernel_size=2, activation=Identity()
        )
        rows = np.random.default_rng(0).normal(size=(2, 2, 5, 5))
        response = layer.respond_to(rows)

        correction = layer.correction_for(response, np.ones((2, 3, 4, 4)))

        assert correction.gradient is not None
        assert correction.gradient.weights.shape == (3, 2 * 2 * 2)
        assert correction.gradient.biases.shape == (3,)

    def test_the_blame_passed_down_has_the_shape_of_what_was_read(self) -> None:
        layer = Conv2d(
            reads=(2, 5, 5),
            n_filters=3,
            kernel_size=3,
            activation=Identity(),
            padding=1,
        )
        rows = np.random.default_rng(0).normal(size=(2, 2, 5, 5))
        response = layer.respond_to(rows)

        correction = layer.correction_for(response, np.ones((2, 3, 5, 5)))

        assert correction.passed_down.shape == rows.shape

    def test_a_bias_slope_is_the_blame_summed_over_every_position(self) -> None:
        """One bias per filter, so every position it answered at contributes."""
        layer = Conv2d(
            reads=(1, 4, 4), n_filters=2, kernel_size=2, activation=Identity()
        )
        rows = np.zeros((3, 1, 4, 4))
        response = layer.respond_to(rows)

        correction = layer.correction_for(response, np.ones((3, 2, 3, 3)))

        assert correction.gradient is not None
        assert np.allclose(correction.gradient.biases, 3 * 3 * 3)

    def test_an_arriving_block_of_the_wrong_shape_is_refused(self) -> None:
        layer = hand_worked_layer()
        response = layer.respond_to(COUNTING_PICTURE)

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((1, 2, 3, 3)))

    def test_an_arriving_block_that_is_not_numbers_is_refused(self) -> None:
        layer = hand_worked_layer()
        response = layer.respond_to(COUNTING_PICTURE)

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, "blame")  # type: ignore[arg-type]


class TestTheGradientCheck:
    """Nudge one number, watch the summary, compare against the claim.

    Run over the kernels, the biases and the inputs in turn, because they are
    three separate derivations that share only the ``delta`` block, and a
    mistake in any one of them leaves the other two right.
    """

    def padded_layer(self) -> Conv2d:
        """Two channels, two filters, a border of one, and a smooth bend.

        ``HyperbolicTangent`` rather than a rectifier because a central
        difference across a kink measures the average of two different slopes
        and reports a disagreement that is the test's fault.
        """
        return Conv2d(
            reads=(2, 4, 4),
            n_filters=2,
            kernel_size=2,
            activation=HyperbolicTangent(),
            padding=1,
            random_seed=5,
        )

    def strided_layer(self) -> Conv2d:
        """One channel, a 3x3 window two apart, so windows overlap by one."""
        return Conv2d(
            reads=(1, 5, 5),
            n_filters=2,
            kernel_size=3,
            activation=HyperbolicTangent(),
            stride=2,
            random_seed=6,
        )

    def batch_for(self, layer: Conv2d, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Two pictures, and the summary weights that are also the blame."""
        generator = np.random.default_rng(seed)
        rows = generator.normal(size=(2, *layer.shape.reads))
        weights = generator.normal(size=(2, *layer.shape.answers))
        return rows, weights

    def claimed_kernel_gradient(
        self, layer: Conv2d, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        correction = layer.correction_for(layer.respond_to(rows), weights)
        assert correction.gradient is not None
        return correction.gradient.weights.reshape(layer.kernels.shape)

    def measured_kernel_gradient(
        self, layer: Conv2d, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(layer.kernels)
        for filter_index in range(layer.kernels.shape[0]):
            for channel in range(layer.kernels.shape[1]):
                for kernel_row in range(layer.kernel_size):
                    for kernel_column in range(layer.kernel_size):
                        moved = []
                        for direction in (+NUDGE, -NUDGE):
                            kernels = np.array(layer.kernels, copy=True)
                            kernels[
                                filter_index, channel, kernel_row, kernel_column
                            ] += direction
                            moved.append(
                                summary_of(
                                    layer.with_parameters(kernels, layer.bias_vector),
                                    rows,
                                    weights,
                                )
                            )
                        measured[filter_index, channel, kernel_row, kernel_column] = (
                            moved[0] - moved[1]
                        ) / (2.0 * NUDGE)
        return measured

    def measured_bias_gradient(
        self, layer: Conv2d, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(layer.bias_vector)
        for filter_index in range(layer.n_filters):
            moved = []
            for direction in (+NUDGE, -NUDGE):
                biases = np.array(layer.bias_vector, copy=True)
                biases[filter_index] += direction
                moved.append(
                    summary_of(
                        layer.with_parameters(layer.kernels, biases), rows, weights
                    )
                )
            measured[filter_index] = (moved[0] - moved[1]) / (2.0 * NUDGE)
        return measured

    def measured_input_gradient(
        self, layer: Conv2d, rows: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        measured = np.zeros_like(rows)
        channels, height, width = layer.shape.reads
        for row in range(rows.shape[0]):
            for channel in range(channels):
                for picture_row in range(height):
                    for picture_column in range(width):
                        moved = []
                        for direction in (+NUDGE, -NUDGE):
                            nudged = np.array(rows, copy=True)
                            nudged[row, channel, picture_row, picture_column] += (
                                direction
                            )
                            moved.append(summary_of(layer, nudged, weights))
                        measured[row, channel, picture_row, picture_column] = (
                            moved[0] - moved[1]
                        ) / (2.0 * NUDGE)
        return measured

    def test_every_kernel_slope_matches_a_finite_difference(self) -> None:
        layer = self.padded_layer()
        rows, weights = self.batch_for(layer, seed=11)

        claimed = self.claimed_kernel_gradient(layer, rows, weights)
        measured = self.measured_kernel_gradient(layer, rows, weights)

        assert np.allclose(claimed, measured, atol=1e-7)

    def test_every_bias_slope_matches_a_finite_difference(self) -> None:
        layer = self.padded_layer()
        rows, weights = self.batch_for(layer, seed=11)

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_bias_gradient(layer, rows, weights)

        assert correction.gradient is not None
        assert np.allclose(correction.gradient.biases, measured, atol=1e-7)

    def test_every_input_slope_matches_a_finite_difference(self) -> None:
        """The half that travels downward, and the half padding can shift."""
        layer = self.padded_layer()
        rows, weights = self.batch_for(layer, seed=11)

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    def test_the_kernel_slopes_hold_under_a_stride(self) -> None:
        layer = self.strided_layer()
        rows, weights = self.batch_for(layer, seed=12)

        claimed = self.claimed_kernel_gradient(layer, rows, weights)
        measured = self.measured_kernel_gradient(layer, rows, weights)

        assert np.allclose(claimed, measured, atol=1e-7)

    def test_the_input_slopes_hold_under_a_stride(self) -> None:
        """Overlapping windows, so a value that is missing a term shows here."""
        layer = self.strided_layer()
        rows, weights = self.batch_for(layer, seed=12)

        correction = layer.correction_for(layer.respond_to(rows), weights)
        measured = self.measured_input_gradient(layer, rows, weights)

        assert np.allclose(correction.passed_down, measured, atol=1e-7)

    def test_the_check_holds_through_a_rectifier_too(self) -> None:
        """A different bend means a different ``delta``, and nothing else.

        A central difference across a rectifier's kink measures the average of
        two different slopes and reports a disagreement the layer is not
        responsible for, so the fixture has to keep every score clear of zero.
        Measured on this one, the smallest absolute score is 0.00246, against a
        nudge that moves a score by at most 1.2e-06, so no score changes sign
        and the kink is never crossed.
        """
        layer = Conv2d(
            reads=(1, 4, 4),
            n_filters=2,
            kernel_size=2,
            activation=RectifiedLinear(),
            random_seed=9,
        )
        rows, weights = self.batch_for(layer, seed=13)

        claimed = self.claimed_kernel_gradient(layer, rows, weights)
        measured = self.measured_kernel_gradient(layer, rows, weights)

        assert np.allclose(claimed, measured, atol=1e-6)


class TestStepping:
    def test_a_step_moves_against_the_slope(self) -> None:
        layer = Conv2d(
            reads=(1, 4, 4),
            n_filters=2,
            kernel_size=2,
            activation=HyperbolicTangent(),
            random_seed=3,
        )
        rows = np.random.default_rng(4).normal(size=(2, 1, 4, 4))
        correction = layer.correction_for(layer.respond_to(rows), np.ones((2, 2, 3, 3)))
        assert correction.gradient is not None

        stepped = layer.stepped_by(correction.gradient, learning_rate=0.1)

        expected = layer.kernels - 0.1 * correction.gradient.weights.reshape(
            layer.kernels.shape
        )
        assert np.allclose(stepped.kernels, expected)
        assert np.allclose(
            stepped.bias_vector, layer.bias_vector - 0.1 * correction.gradient.biases
        )

    def test_a_step_lowers_the_summary_it_was_asked_to_lower(self) -> None:
        """The point of the whole exercise, on one small step."""
        layer = Conv2d(
            reads=(1, 5, 5),
            n_filters=2,
            kernel_size=3,
            activation=HyperbolicTangent(),
            random_seed=8,
        )
        generator = np.random.default_rng(21)
        rows = generator.normal(size=(3, *layer.shape.reads))
        weights = generator.normal(size=(3, *layer.shape.answers))

        correction = layer.correction_for(layer.respond_to(rows), weights)
        stepped = layer.stepped_by(correction.gradient, learning_rate=0.05)

        assert summary_of(stepped, rows, weights) < summary_of(layer, rows, weights)

    def test_the_original_layer_is_untouched(self) -> None:
        """Training does not mutate, which is what keeps the arrays frozen."""
        layer = Conv2d(
            reads=(1, 4, 4), n_filters=2, kernel_size=2, activation=Identity()
        )
        before = np.array(layer.kernels, copy=True)
        rows = np.random.default_rng(5).normal(size=(2, 1, 4, 4))
        correction = layer.correction_for(layer.respond_to(rows), np.ones((2, 2, 3, 3)))

        layer.stepped_by(correction.gradient, learning_rate=0.5)

        assert np.array_equal(layer.kernels, before)

    def test_stepping_by_nothing_is_refused(self) -> None:
        """A convolution has parameters, so a missing gradient is a mistake."""
        layer = Conv2d(
            reads=(1, 4, 4), n_filters=1, kernel_size=2, activation=Identity()
        )

        with pytest.raises(ShapeMismatchError):
            layer.stepped_by(None, learning_rate=0.1)

    def test_a_gradient_of_the_wrong_width_is_refused(self) -> None:
        layer = Conv2d(
            reads=(1, 4, 4), n_filters=2, kernel_size=2, activation=Identity()
        )

        with pytest.raises(ShapeMismatchError):
            layer.stepped_by(
                LayerGradient(weights=np.zeros((2, 9)), biases=np.zeros(2)),
                learning_rate=0.1,
            )

    def test_the_stepped_layer_keeps_its_geometry(self) -> None:
        layer = Conv2d(
            reads=(1, 5, 5),
            n_filters=2,
            kernel_size=3,
            activation=HyperbolicTangent(),
            stride=2,
            padding=1,
        )
        rows = np.random.default_rng(6).normal(size=(2, 1, 5, 5))
        correction = layer.correction_for(
            layer.respond_to(rows), np.ones((2, *layer.shape.answers))
        )

        stepped = layer.stepped_by(correction.gradient, learning_rate=0.1)

        assert stepped.shape == layer.shape
        assert stepped.stride == 2
        assert stepped.padding == 1
        assert stepped.activation == layer.activation
