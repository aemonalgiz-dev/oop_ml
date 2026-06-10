"""Spec for nearest-neighbour upsampling, the mirror image of pooling.

The forward pass is repetition: every value is copied over a ``factor`` by
``factor`` block, so ``[[1, 2], [3, 4]]`` at a factor of 2 becomes a four by
four grid of four constant quarters. The backward pass is the transpose of
that, which is a sum: each input value was copied to ``factor ** 2`` positions,
so its slope is the total of the slopes arriving at all of them.

Three oracles, none of which reads the implementation. A plain Python loop
written from the definition, ``out[i, j] = in[i // factor, j // factor]``, for
the forward pass and its transpose. A finite-difference check, which is
computed from forward passes alone. And the adjoint identity, ``<up(x), y> ==
<x, up_transpose(y)>`` for arbitrary blocks, which holds for the correct
backward pass of any linear layer and for no other.

The relation to pooling is asserted rather than described. An average pool
whose window and stride equal the factor undoes the forward pass exactly, and
the same average pool, scaled by ``factor ** 2``, is exactly the backward pass.
"""

from collections.abc import Callable

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    MLLibError,
    ShapeMismatchError,
)
from oop_ml.core.network.pooling import AveragePool2d, MaxPool2d
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.upsampling import NearestUpsample2d

# Worked on paper. At a factor of 2 each value fills a two by two quarter.
HAND_WORKED_GRID = np.array([[1.0, 2.0], [3.0, 4.0]])
HAND_WORKED_REPEATED = np.array(
    [
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [3.0, 3.0, 4.0, 4.0],
        [3.0, 3.0, 4.0, 4.0],
    ]
)

# Worked on paper for the way back. Each two by two quarter of 1..16 sums to
# the value its source position is owed:
#   1 + 2 + 5 + 6 = 14      3 + 4 + 7 + 8 = 22
#   9 + 10 + 13 + 14 = 46   11 + 12 + 15 + 16 = 54
ARRIVING_ONE_TO_SIXTEEN = np.arange(1.0, 17.0).reshape(4, 4)
HAND_WORKED_SUMMED = np.array([[14.0, 22.0], [46.0, 54.0]])


def one_picture(grid: np.ndarray) -> np.ndarray:
    """One row, one channel, wrapping a bare grid as a block a layer reads."""
    return grid.reshape(1, 1, *grid.shape)


def repeated_by_loop(block: np.ndarray, factor: int) -> np.ndarray:
    """The definition of the forward pass, one output position at a time."""
    n_rows, channels, height, width = block.shape
    answer = np.zeros((n_rows, channels, height * factor, width * factor))
    for row in range(n_rows):
        for channel in range(channels):
            for out_row in range(height * factor):
                for out_column in range(width * factor):
                    answer[row, channel, out_row, out_column] = block[
                        row, channel, out_row // factor, out_column // factor
                    ]
    return answer


def summed_by_loop(arriving: np.ndarray, factor: int) -> np.ndarray:
    """The transpose of the definition, adding each arriving value to its source."""
    n_rows, channels, out_height, out_width = arriving.shape
    owed = np.zeros((n_rows, channels, out_height // factor, out_width // factor))
    for row in range(n_rows):
        for channel in range(channels):
            for out_row in range(out_height):
                for out_column in range(out_width):
                    owed[row, channel, out_row // factor, out_column // factor] += (
                        arriving[row, channel, out_row, out_column]
                    )
    return owed


def small_distinct_block(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Distinct values spaced 0.1 apart and centred on zero.

    Centred and small so a weighted summary stays near zero, since a central
    difference multiplies the summary's rounding by half the reciprocal of its
    step.
    """
    generator = np.random.default_rng(seed)
    n_values = int(np.prod(shape))
    values = generator.permutation(np.arange(n_values, dtype=np.float64))
    return ((values - n_values / 2.0) * 0.1).reshape(shape)


def finite_difference_slopes(
    layer: NearestUpsample2d, block: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """How ``sum(outputs * weights)`` moves when each input value is nudged."""
    step = 1e-6
    measured = np.empty_like(block)
    for position in np.ndindex(*block.shape):
        moved = []
        for direction in (+step, -step):
            nudged = block.copy()
            nudged[position] += direction
            outputs = np.asarray(layer.respond_to(nudged).outputs)
            moved.append(float(np.sum(outputs * weights)))
        measured[position] = (moved[0] - moved[1]) / (2.0 * step)
    return measured


class TestTheShapeArithmetic:
    """Every extent is known at construction, so the whole shape is."""

    def test_a_factor_of_two_doubles_each_spatial_axis(self) -> None:
        layer = NearestUpsample2d(reads=(1, 14, 14), factor=2)

        assert layer.shape == LayerShape(n_inputs=(1, 14, 14), n_outputs=(1, 28, 28))

    def test_the_channel_count_is_carried_through_untouched(self) -> None:
        layer = NearestUpsample2d(reads=(8, 13, 13), factor=2)

        assert layer.shape.answers == (8, 26, 26)

    def test_the_two_spatial_axes_are_each_multiplied(self) -> None:
        layer = NearestUpsample2d(reads=(3, 2, 5), factor=3)

        assert layer.shape.answers == (3, 6, 15)

    def test_a_factor_of_one_answers_what_it_read(self) -> None:
        layer = NearestUpsample2d(reads=(2, 4, 7), factor=1)

        assert layer.shape.answers == (2, 4, 7)

    def test_the_element_count_grows_by_the_square_of_the_factor(self) -> None:
        layer = NearestUpsample2d(reads=(2, 4, 4), factor=3)

        assert layer.shape.n_inputs == 32
        assert layer.shape.n_outputs == 32 * 9

    def test_the_default_factor_is_two(self) -> None:
        layer = NearestUpsample2d(reads=(1, 4, 4))

        assert layer.factor == 2

    def test_a_numpy_whole_number_is_a_factor(self) -> None:
        """Read off a computation, a factor is routinely a ``numpy.int64``."""
        layer = NearestUpsample2d(reads=(1, 4, 4), factor=np.int64(3))  # type: ignore[arg-type]

        assert layer.factor == 3
        assert layer.shape.answers == (1, 12, 12)

    def test_it_joins_the_pooling_layer_it_mirrors_in_both_directions(self) -> None:
        """Pool then grow, or grow then pool, and every join holds."""
        pool = MaxPool2d(reads=(4, 8, 8), window=2, stride=2)
        grow = NearestUpsample2d(reads=(4, 4, 4), factor=2)

        assert grow.shape.follows(pool.shape)
        assert AveragePool2d(reads=(4, 8, 8), window=2, stride=2).shape.follows(
            grow.shape
        )


class TestConstructionRefusals:
    @pytest.mark.parametrize(
        "reads", [5, None, 3.5], ids=["a bare width", "nothing", "a fraction"]
    )
    def test_reads_that_is_not_a_sequence_of_extents_is_refused_by_name(
        self, reads: object
    ) -> None:
        """``tuple(5)`` raises a bare ``TypeError``, which is not the library's.

        A bare width is the natural mistake, since it is what a dense layer's
        ``reads`` looks like, and the pooling layers once let it escape.
        """
        with pytest.raises(MLLibError):
            NearestUpsample2d(reads=reads, factor=2)  # type: ignore[arg-type]

    def test_a_bare_width_is_refused_as_a_shape_mistake(self) -> None:
        with pytest.raises(ShapeMismatchError):
            NearestUpsample2d(reads=5, factor=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "reads", [(), (8,), (8, 8), (1, 1, 8, 8)], ids=["none", "one", "two", "four"]
    )
    def test_a_picture_that_is_not_three_extents_is_refused(
        self, reads: tuple[int, ...]
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            NearestUpsample2d(reads=reads, factor=2)

    @pytest.mark.parametrize(
        "reads",
        [(0, 4, 4), (1, 0, 4), (1, 4, 0), (-1, 4, 4)],
        ids=["no channels", "no height", "no width", "negative channels"],
    )
    def test_an_extent_below_one_is_refused(self, reads: tuple[int, int, int]) -> None:
        with pytest.raises(InvalidValuesError):
            NearestUpsample2d(reads=reads, factor=2)

    @pytest.mark.parametrize(
        "reads",
        [(1, 4.5, 4), (1, 4, 4.5), (True, 4, 4)],
        ids=["fractional height", "fractional width", "boolean channels"],
    )
    def test_an_extent_that_is_not_a_whole_number_is_refused(
        self, reads: tuple[object, object, object]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            NearestUpsample2d(reads=reads, factor=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("factor", [0, -2], ids=["zero", "negative"])
    def test_a_factor_below_one_is_refused(self, factor: int) -> None:
        """A factor of zero answers an empty picture, which is an absent layer."""
        with pytest.raises(InvalidValuesError):
            NearestUpsample2d(reads=(1, 4, 4), factor=factor)

    @pytest.mark.parametrize(
        "factor", [2.5, True, "2"], ids=["fraction", "bool", "text"]
    )
    def test_a_factor_that_is_not_a_whole_number_is_refused(
        self, factor: object
    ) -> None:
        """``True`` indexes as 1 and would otherwise pass as a factor of one."""
        with pytest.raises(InvalidValuesError):
            NearestUpsample2d(reads=(1, 4, 4), factor=factor)  # type: ignore[arg-type]


class TestTheForwardPass:
    def test_it_matches_a_grid_worked_by_hand(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        assert np.array_equal(response.outputs[0, 0], HAND_WORKED_REPEATED)

    def test_a_factor_of_three_fills_a_three_by_three_block(self) -> None:
        layer = NearestUpsample2d(reads=(1, 1, 1), factor=3)

        response = layer.respond_to(np.array([[[[5.0]]]]))

        assert np.array_equal(response.outputs[0, 0], np.full((3, 3), 5.0))

    def test_it_matches_the_definition_written_as_a_loop(self) -> None:
        """Several rows, several channels, a rectangle and an odd factor."""
        layer = NearestUpsample2d(reads=(3, 2, 5), factor=3)
        block = np.random.default_rng(4).normal(size=(2, 3, 2, 5))

        response = layer.respond_to(block)

        assert np.array_equal(response.outputs, repeated_by_loop(block, 3))

    def test_a_factor_of_one_hands_back_the_block_it_read(self) -> None:
        layer = NearestUpsample2d(reads=(2, 3, 3), factor=1)
        block = np.random.default_rng(5).normal(size=(2, 2, 3, 3))

        assert np.array_equal(layer.respond_to(block).outputs, block)

    def test_channels_are_repeated_independently(self) -> None:
        layer = NearestUpsample2d(reads=(2, 2, 2), factor=2)
        block = np.stack([HAND_WORKED_GRID, -HAND_WORKED_GRID])[np.newaxis]

        response = layer.respond_to(block)

        assert np.array_equal(response.outputs[0, 0], HAND_WORKED_REPEATED)
        assert np.array_equal(response.outputs[0, 1], -HAND_WORKED_REPEATED)

    def test_rows_are_repeated_independently(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        block = np.stack(
            [one_picture(HAND_WORKED_GRID)[0], one_picture(2.0 * HAND_WORKED_GRID)[0]]
        )

        response = layer.respond_to(block)

        assert np.array_equal(response.outputs[1, 0], 2.0 * HAND_WORKED_REPEATED)

    def test_the_answer_has_the_shape_the_layer_promised(self) -> None:
        layer = NearestUpsample2d(reads=(2, 3, 4), factor=2)

        response = layer.respond_to(np.zeros((5, 2, 3, 4)))

        assert response.outputs.shape == (5, *layer.shape.answers)

    def test_scores_and_outputs_are_the_same_block(self) -> None:
        """No activation, so there is no pre-activation value to keep apart."""
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        assert response.scores is response.outputs

    def test_the_response_carries_the_block_that_was_read(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        assert np.array_equal(response.inputs, one_picture(HAND_WORKED_GRID))

    def test_the_answer_is_frozen(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        with pytest.raises(ValueError):
            response.outputs[0, 0, 0, 0] = 99.0

    def test_the_callers_own_block_is_not_frozen_underneath_them(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        block = one_picture(HAND_WORKED_GRID.copy())

        layer.respond_to(block)
        block[0, 0, 0, 0] = 99.0

        assert block[0, 0, 0, 0] == 99.0

    def test_changing_one_answer_cannot_reach_its_neighbours(self) -> None:
        """The repeated block is a copy, not a broadcast view of one number.

        A broadcast view would make the four entries of a quarter one memory
        location, and the frozen flag is the only thing standing between that
        and a write that moves four answers at once.
        """
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        outputs = np.asarray(layer.respond_to(one_picture(HAND_WORKED_GRID)).outputs)

        assert outputs.flags.c_contiguous
        assert not np.shares_memory(outputs[0, 0, 0, 0:1], outputs[0, 0, 0, 1:2])


class TestForwardRefusals:
    @pytest.mark.parametrize(
        "block",
        [
            pytest.param(np.zeros((2, 1, 2, 3)), id="wrong width"),
            pytest.param(np.zeros((2, 2, 2, 2)), id="wrong channel count"),
            pytest.param(np.zeros((1, 2, 2)), id="three dimensions"),
        ],
    )
    def test_a_block_arranged_some_other_way_is_refused(
        self, block: np.ndarray
    ) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(block)

    def test_a_block_with_no_rows_is_refused(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)

        with pytest.raises(EmptyValuesError):
            layer.respond_to(np.zeros((0, 1, 2, 2)))

    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_entry_is_refused(self, poison: float) -> None:
        """Repetition would copy a ``nan`` to four places without a word."""
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        poisoned = one_picture(HAND_WORKED_GRID.copy())
        poisoned[0, 0, 1, 1] = poison

        with pytest.raises(InvalidValuesError):
            layer.respond_to(poisoned)


class TestTheBackwardPass:
    """Each input is owed the total of the slopes at every copy of it."""

    def test_it_matches_a_grid_worked_by_hand(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        correction = layer.correction_for(
            response, one_picture(ARRIVING_ONE_TO_SIXTEEN)
        )

        assert np.array_equal(correction.passed_down[0, 0], HAND_WORKED_SUMMED)

    def test_it_matches_the_transpose_written_as_a_loop(self) -> None:
        layer = NearestUpsample2d(reads=(3, 2, 5), factor=3)
        generator = np.random.default_rng(6)
        response = layer.respond_to(generator.normal(size=(2, 3, 2, 5)))
        arriving = generator.normal(size=(2, 3, 6, 15))

        correction = layer.correction_for(response, arriving)

        assert np.allclose(correction.passed_down, summed_by_loop(arriving, 3))

    def test_the_block_passed_down_is_the_shape_the_layer_read(self) -> None:
        layer = NearestUpsample2d(reads=(3, 2, 4), factor=2)
        response = layer.respond_to(np.zeros((2, 3, 2, 4)))

        correction = layer.correction_for(response, np.ones((2, 3, 4, 8)))

        assert correction.passed_down.shape == (2, 3, 2, 4)

    def test_the_total_blame_equals_the_total_arriving(self) -> None:
        """Every arriving slope belongs to exactly one source, so none is lost."""
        generator = np.random.default_rng(7)
        layer = NearestUpsample2d(reads=(2, 3, 3), factor=3)
        response = layer.respond_to(generator.normal(size=(4, 2, 3, 3)))
        arriving = generator.normal(size=(4, 2, 9, 9))

        correction = layer.correction_for(response, arriving)

        assert float(np.sum(correction.passed_down)) == pytest.approx(
            float(np.sum(arriving)), rel=1e-12
        )

    def test_the_backward_pass_does_not_depend_on_what_was_read(self) -> None:
        """The layer is linear, so its derivative is the same everywhere."""
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        arriving = one_picture(ARRIVING_ONE_TO_SIXTEEN)

        upright = layer.correction_for(
            layer.respond_to(one_picture(HAND_WORKED_GRID)), arriving
        )
        other = layer.correction_for(
            layer.respond_to(one_picture(-7.0 * HAND_WORKED_GRID)), arriving
        )

        assert np.array_equal(upright.passed_down, other.passed_down)

    def test_it_reports_no_gradient(self) -> None:
        """None rather than a block of zeros, which would claim it learns."""
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        correction = layer.correction_for(response, np.ones((1, 1, 4, 4)))

        assert correction.gradient is None
        assert correction.learns is False


class TestBackwardRefusals:
    def test_an_arriving_block_of_the_wrong_shape_is_refused(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((1, 1, 2, 2)))

    def test_an_arriving_block_with_the_wrong_row_count_is_refused(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, 1, 4, 4)))

    def test_an_arriving_block_that_is_not_numeric_is_refused(self) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, "not a slope")  # type: ignore[arg-type]

    def test_a_response_from_a_differently_shaped_layer_is_refused(self) -> None:
        wide = NearestUpsample2d(reads=(1, 3, 3), factor=2)
        narrow = NearestUpsample2d(reads=(1, 2, 2), factor=2)
        response = wide.respond_to(np.zeros((1, 1, 3, 3)))

        with pytest.raises(ShapeMismatchError):
            narrow.correction_for(response, np.ones((1, 1, 4, 4)))


class TestTheGradientCheck:
    """Nudge each input, watch a weighted summary, compare with the claim.

    The layer is linear, so the central difference is exact up to the rounding
    of the summary itself, and every geometry below should agree to far better
    than the tolerance.
    """

    @pytest.mark.parametrize(
        ("reads", "factor", "n_rows", "seed"),
        [
            pytest.param((1, 2, 2), 2, 1, 31, id="one picture"),
            pytest.param((2, 3, 2), 3, 2, 33, id="rectangle, factor three"),
            pytest.param((3, 2, 2), 1, 2, 35, id="factor one"),
            pytest.param((1, 3, 3), 2, 3, 37, id="three rows"),
        ],
    )
    def test_the_backward_pass_matches_a_finite_difference(
        self, reads: tuple[int, int, int], factor: int, n_rows: int, seed: int
    ) -> None:
        layer = NearestUpsample2d(reads=reads, factor=factor)
        block = small_distinct_block((n_rows, *reads), seed=seed)
        weights = np.random.default_rng(seed + 1).normal(
            size=(n_rows, *layer.shape.answers)
        )

        claimed = layer.correction_for(layer.respond_to(block), weights).passed_down
        measured = finite_difference_slopes(layer, block, weights)

        assert np.allclose(claimed, measured, atol=1e-8)

    @pytest.mark.parametrize("seed", [41, 43, 47])
    def test_the_backward_pass_is_the_adjoint_of_the_forward(self, seed: int) -> None:
        """``<up(x), y> == <x, up_transpose(y)>`` for any two blocks.

        What makes a backward pass correct for a linear layer, stated as one
        equation about inner products rather than as a derivative.
        """
        generator = np.random.default_rng(seed)
        layer = NearestUpsample2d(reads=(2, 3, 4), factor=2)
        block = generator.normal(size=(3, 2, 3, 4))
        arriving = generator.normal(size=(3, 2, 6, 8))

        forward = np.asarray(layer.respond_to(block).outputs)
        backward = layer.correction_for(layer.respond_to(block), arriving).passed_down

        assert float(np.sum(forward * arriving)) == pytest.approx(
            float(np.sum(block * backward)), rel=1e-12
        )


class TestItMirrorsAveragePooling:
    """Stated as equations about the two layers, rather than as an analogy."""

    @pytest.mark.parametrize("factor", [1, 2, 3])
    def test_an_average_pool_of_the_same_size_undoes_it(self, factor: int) -> None:
        grow = NearestUpsample2d(reads=(2, 3, 2), factor=factor)
        shrink = AveragePool2d(
            reads=(2, 3 * factor, 2 * factor), window=factor, stride=factor
        )
        block = np.random.default_rng(factor).normal(size=(2, 2, 3, 2))

        recovered = shrink.respond_to(grow.respond_to(block).outputs).outputs

        assert np.allclose(recovered, block)

    @pytest.mark.parametrize("factor", [2, 3])
    def test_its_backward_pass_is_that_average_pool_times_the_block_area(
        self, factor: int
    ) -> None:
        """A sum over a block is its mean times how many positions it holds."""
        grow = NearestUpsample2d(reads=(1, 2, 3), factor=factor)
        shrink = AveragePool2d(
            reads=(1, 2 * factor, 3 * factor), window=factor, stride=factor
        )
        generator = np.random.default_rng(50 + factor)
        arriving = generator.normal(size=(2, 1, 2 * factor, 3 * factor))

        summed = grow.correction_for(
            grow.respond_to(np.zeros((2, 1, 2, 3))), arriving
        ).passed_down
        averaged = np.asarray(shrink.respond_to(arriving).outputs)

        assert np.allclose(summed, averaged * factor * factor)


def transposed_convolution(
    picture: np.ndarray, kernel: np.ndarray, stride: int
) -> np.ndarray:
    """A transposed convolution written as a scatter, for the comparison only.

    Each input value is multiplied by the kernel and added into the output at
    ``stride`` times its own position, which is the definition.
    """
    height, width = picture.shape
    side = kernel.shape[0]
    output = np.zeros(((height - 1) * stride + side, (width - 1) * stride + side))
    for row in range(height):
        for column in range(width):
            output[
                row * stride : row * stride + side,
                column * stride : column * stride + side,
            ] += picture[row, column] * kernel
    return output


class TestWhyRepetitionRatherThanATransposedConvolution:
    """The measurement the module docstring rests on, pinned.

    A picture of ones through a transposed convolution whose weights are all
    ones counts how many kernel placements land on each output position. At a
    kernel of 3 and a stride of 2 the count is uneven, 1, 2 and 4 in a repeating
    pattern, so before any learning the layer already paints a checkerboard
    that its weights would have to learn away. Repetition has no overlap at all.
    """

    def interior(self, picture: np.ndarray, border: int) -> np.ndarray:
        return picture[border:-border, border:-border]

    def test_a_kernel_of_three_at_stride_two_is_uneven(self) -> None:
        counts = transposed_convolution(np.ones((4, 4)), np.ones((3, 3)), stride=2)

        assert sorted(set(self.interior(counts, 2).ravel().tolist())) == [
            1.0,
            2.0,
            4.0,
        ]

    def test_a_kernel_the_stride_divides_is_even_away_from_the_border(self) -> None:
        """The artefact is about divisibility, and saying so is the honest claim."""
        counts = transposed_convolution(np.ones((4, 4)), np.ones((4, 4)), stride=2)

        assert sorted(set(self.interior(counts, 2).ravel().tolist())) == [4.0]

    def test_repetition_of_a_constant_picture_is_constant(self) -> None:
        layer = NearestUpsample2d(reads=(1, 4, 4), factor=2)

        outputs = np.asarray(layer.respond_to(np.ones((1, 1, 4, 4))).outputs)

        assert sorted(set(outputs.ravel().tolist())) == [1.0]


class TestItHasNothingToLearn:
    @pytest.mark.parametrize("learning_rate", [0.0, 0.1, 1000.0])
    def test_a_step_answers_with_the_very_same_layer(
        self, learning_rate: float
    ) -> None:
        layer = NearestUpsample2d(reads=(1, 2, 2), factor=2)

        assert layer.stepped_by(None, learning_rate=learning_rate) is layer


class TestValueSemantics:
    def test_two_layers_configured_alike_are_equal(self) -> None:
        assert NearestUpsample2d(reads=(1, 4, 4), factor=2) == NearestUpsample2d(
            reads=(1, 4, 4), factor=2
        )

    def test_a_different_factor_is_a_different_layer(self) -> None:
        assert NearestUpsample2d(reads=(1, 4, 4), factor=2) != NearestUpsample2d(
            reads=(1, 4, 4), factor=3
        )

    def test_it_is_not_equal_to_the_pooling_layer_it_mirrors(self) -> None:
        assert NearestUpsample2d(reads=(1, 4, 4), factor=2) != AveragePool2d(
            reads=(1, 4, 4), window=2, stride=2
        )

    def test_it_defers_to_anything_that_is_not_an_upsampling_layer(self) -> None:
        assert NearestUpsample2d(reads=(1, 4, 4)).__eq__(object()) is NotImplemented

    def test_it_is_hashable(self) -> None:
        layer = NearestUpsample2d(reads=(1, 4, 4), factor=2)

        assert len({layer, NearestUpsample2d(reads=(1, 4, 4), factor=2)}) == 1

    def test_its_repr_names_its_configuration(self) -> None:
        text = repr(NearestUpsample2d(reads=(3, 4, 4), factor=2))

        assert text == "NearestUpsample2d(reads=(3, 4, 4), factor=2)"


# Kept so that a reader can see the oracle functions are what they claim to be,
# independently of the layer: the loop and its transpose satisfy the adjoint
# identity on their own.
def test_the_two_loop_oracles_are_adjoint_to_each_other() -> None:
    generator = np.random.default_rng(60)
    block = generator.normal(size=(1, 2, 2, 3))
    arriving = generator.normal(size=(1, 2, 4, 6))
    forward: Callable[[np.ndarray, int], np.ndarray] = repeated_by_loop

    assert float(np.sum(forward(block, 2) * arriving)) == pytest.approx(
        float(np.sum(block * summed_by_loop(arriving, 2))), rel=1e-12
    )
