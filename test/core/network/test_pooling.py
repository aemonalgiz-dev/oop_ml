"""Spec for the pooling family, where the backward pass is the whole difficulty.

Two layers are specified here and they share almost everything.
:class:`~oop_ml.core.network.pooling.MaxPool2d` keeps the largest value in each
window and :class:`~oop_ml.core.network.pooling.AveragePool2d` keeps the mean,
and beneath that one difference sit the same extents, the same sweep, the same
refusals and the same absence of parameters. So the file is in three parts. What
only a maximum does, what only a mean does, and what every pooling layer must do
whichever it is. The third part is parametrized over both classes, because a
claim about a family that is only ever asserted of one member is a claim nobody
is keeping.

The forward halves are easy to get right and easy to test. The largest value in
a window and the mean of a window are both numbers a person can compute on
paper, and every hand-worked grid below was computed that way before the
implementation existed.

The backward halves are the opposite. Sharing a window's arriving value out over
the positions that earned it is a routing rather than an arithmetic, and every
wrong routing still runs. Send the blame to the wrong corner of the window and
every shape still conforms. Assign instead of accumulate and the gradient is
merely smaller, entirely finite, and a network built on it still descends --
more slowly, and towards somewhere else. Drop the ``1 / n`` from a mean and the
gradient is four times too large on a two-by-two window, which is a learning
rate nobody chose. Recompute a maximum's winner with a different tie rule than
the forward pass used and nothing raises at all.

So the finite-difference check carries this file, and it is run over both
classes on the same four geometries. It is the oracle because it is computed
from forward passes alone and knows nothing about how either backward pass was
derived. The overlapping-window fixture is what separates ``+=`` from ``=``, and
the hand-worked version of it is written on a grid whose centre belongs to all
four windows so that an assignment is off by a factor rather than by a rounding.

The shares are the family's own invariant
-----------------------------------------
Every pooling answer here is a weighted mean of its window, and
:meth:`~oop_ml.core.network.pooling.Pool2d.shares_of` is those weights. Two
things follow that the base class deliberately asserts nothing about, so this
file asserts them of every subclass instead.

The weights sum to one, which is what stops a pooling layer scaling the gradient
on its way past. And the weights recover the answer, so that
``sum(shares * window)`` is what ``summarise(window)`` already returned, 5.0 for
a maximum and 2.75 for a mean on the same little window. That second claim is
what ties the two methods of a subclass to each other rather than leaving them
two independent opinions about the same window.

The two claims are independent, which is why both are made. A mean that routed
everything to its largest position has shares that still sum to one and recovers
5.0 where the answer was 2.75. A mean that dropped its ``1 / n`` recovers 11.0
and has shares summing to 4.0. Neither break is caught by both claims, and
neither is caught by the shape claim at all.

Summing to one is exact in arithmetic and only nearly exact in float64, and the
spec says the true thing rather than the tidy one. Measured over window sides 1
to 32, sixteen of them leave ``numpy.sum`` of a mean's shares one or two units
in the last place away from one. Side 7 gives 0.9999999999999999 and side 31
gives 1.0000000000000004, a worst deviation of 4.4e-16. What is tested is
therefore one to within a few of those units, and side 7 is one of the windows
the family check is parametrized over so that the inexact case is actually run.

Total blame is conserved, and that sum is the reason
----------------------------------------------------
Because every window hands out exactly its own arriving value and no more, the
whole block passed down sums to the whole block arriving. That holds whether or
not the windows overlap, which is worth stating plainly since overlapping looks
like it should manufacture blame. Measured on both classes at strides of 1, 2
and 3, the two totals agree to the last bit. What overlapping changes is where
the total lands rather than what it is, and a value read by four windows really
does collect four contributions.

The tie, which the oracle cannot arbitrate
------------------------------------------
This applies to :class:`~oop_ml.core.network.pooling.MaxPool2d` alone, since a
mean prefers no position and has no tie to break.

At an exact tie the derivative does not exist, the one-sided differences
disagree, and a finite-difference check has no opinion. That case is pinned
directly against the convention the implementation commits to, which is
``numpy.argmax``, first in row-major order within the window. It is a decision
being recorded, not a fact being verified, which is exactly why it needs a test
written by hand.

That the oracle discriminates, measured rather than assumed
-----------------------------------------------------------
Broken backward passes were written and run against these fixtures. For the
maximum, on the overlapping fixture, worst absolute disagreement with the finite
differences: the real one 1.9e-10, an assigning one 2.9, one routing to each
window's top-left corner 3.8, and one dividing and modulo-ing the flat winner
the wrong way round 1.9. The tolerance there is 1e-8, so there are two orders of
magnitude of headroom beneath it and ten of clearance above.

The same exercise was repeated for the family check, over its five fixtures and
both classes, which is ten cases. Worst honest disagreement 1.2e-09. An assigning
mean 0.462 on the overlapping fixture. A mean that dropped its ``1 / n`` between
0.905 and 2.26. A mean routing like a maximum between 0.905 and 2.26. A maximum
sharing like a mean between 0.905 and 2.26. Either class routing to each window's
top-left corner between 0.996 and 3.02. An assignment is invisible wherever the
windows are disjoint, where it agrees with the honest pass to the last bit, which
is the entire reason the overlapping fixture is in the list.

That honest floor is ten times higher than the max-only checks above, because
these blocks are larger and a central difference multiplies the summary's own
rounding by 5e5. The family check is held to 1e-7 rather than 1e-8 for that
reason, which leaves eighty times of room beneath it and six orders of magnitude
above it, and 1e-7 is already what ``test_convolution.py`` uses for the same
arithmetic.

Then each break was installed over the real class and the whole file run against
it, which is the measurement that matters, since a number is only evidence if
some assertion is reading it. Failing tests out of 287, none of the seven passing
clean: an assigning sweep 12, a mean that dropped its ``1 / n`` 28, a mean routing
like a maximum 20, a maximum sharing like a mean 25, both classes routing to the
top-left corner 35, a maximum alone doing so 17, and a mean whose forward pass was
a maximum 19.

That run is also what settled the two shares claims. The routing break leaves the
sum at one and is caught only by the recovery claim; the unnormalised break is
caught by both. Neither claim covers for the other, which is why the file makes
them separately rather than folding them into one.

Two smaller claims were measured the same way. A mean's routing does not depend on
what its window held, so negating the block leaves the blame identical, and under
max-style routing the two blocks disagree by exactly 4.0 on the hand-worked
fixture. And a mean starves nothing, so all sixteen positions of a four-by-four
receive blame where a maximum leaves exactly four non-zero, which is also the
count a mean that routed like one would report.
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
from oop_ml.core.network.pooling import AveragePool2d, MaxPool2d, Pool2d
from oop_ml.core.network.shape import LayerShape

# Worked on paper. Windows of 2 at stride 2 cut this into four quarters:
#   [[1, 3], [5, 6]] -> 6      [[2, 4], [7, 8]] -> 8
#   [[9, 2], [4, 4]] -> 9      [[1, 0], [3, 3]] -> 3
HAND_WORKED_GRID = np.array(
    [
        [1.0, 3.0, 2.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 2.0, 1.0, 0.0],
        [4.0, 4.0, 3.0, 3.0],
    ]
)
HAND_WORKED_POOLED = np.array([[6.0, 8.0], [9.0, 3.0]])

# Where each of those four maxima sits in the grid above.
HAND_WORKED_WINNERS = ((1, 1), (1, 3), (2, 0), (3, 2))

# The same four quarters of the same grid, worked on paper again, and this time
# the whole of each rather than the largest of it:
#   (1 + 3 + 5 + 6) / 4 = 3.75      (2 + 4 + 7 + 8) / 4 = 5.25
#   (9 + 2 + 4 + 4) / 4 = 4.75      (1 + 0 + 3 + 3) / 4 = 1.75
# No entry coincides with the maxima above, which is what makes one fixture
# serve both layers and still tell them apart.
HAND_WORKED_AVERAGED = np.array([[3.75, 5.25], [4.75, 1.75]])

# A 3x3 grid whose four 2x2 windows at stride 1 all share the centre. Worked on
# paper for both layers, and they disagree about every one of the four:
#   [[1, 3], [5, 6]] -> max 6, mean 3.75    [[3, 0], [6, 2]] -> max 6, mean 2.75
#   [[5, 6], [0, 0]] -> max 6, mean 2.75    [[6, 2], [0, 7]] -> max 7, mean 3.75
OVERLAPPING_GRID = np.array([[1.0, 3.0, 0.0], [5.0, 6.0, 2.0], [0.0, 0.0, 7.0]])
OVERLAPPING_AVERAGED = np.array([[3.75, 2.75], [2.75, 3.75]])

# One strong response and three nothings. A maximum reports 8.0 and reports it
# whatever the other three are; a mean reports 2.0 and would report 8.0 only if
# all four were 8.0. The most direct way to ask a layer which one it is.
SPIKED_GRID = np.array([[0.0, 0.0], [0.0, 8.0]])

# Every concrete pooling layer. A claim about the family is parametrized over
# this rather than asserted of whichever subclass was written first, and the
# ids are the class names so a failure says which layer broke it.
POOLING_CLASSES = [
    pytest.param(MaxPool2d, id="MaxPool2d"),
    pytest.param(AveragePool2d, id="AveragePool2d"),
]


def one_picture(grid: np.ndarray) -> np.ndarray:
    """One row, one channel, wrapping a bare grid as a block a layer reads."""
    return grid.reshape(1, 1, *grid.shape)


def summary_of(outputs: np.ndarray, weights: np.ndarray) -> float:
    """A scalar reading of a whole block, so a derivative has something to be of.

    A weighted sum rather than a plain one, because a plain sum makes every
    arriving value 1.0 and would let a backward pass that ignored ``arriving``
    entirely pass the gradient check.
    """
    return float(np.sum(np.asarray(outputs) * weights))


def largest_in(window: np.ndarray) -> float:
    """The definition of a maximum, in a plain loop that borrows no numpy.

    Written from what a maximum *is* rather than from what the implementation
    does, which is the whole difference between an oracle and a restatement of
    the body. It walks the window in row-major order and replaces the running
    best only on a strict improvement, so it keeps the first of any tie and
    therefore pins the tie convention as well as the value.
    """
    values = [value for row in window.tolist() for value in row]
    best = values[0]
    for value in values[1:]:
        if value > best:
            best = value
    return float(best)


def mean_of(window: np.ndarray) -> float:
    """The definition of a mean, totalled in a plain loop and divided once."""
    values = [value for row in window.tolist() for value in row]
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


# What each layer's answer is, derived from the definition rather than from the
# layer. The family tests below read this instead of calling ``summarise`` twice
# and comparing it with itself.
ANSWER_BY_DEFINITION: dict[type[Pool2d], Callable[[np.ndarray], float]] = {
    MaxPool2d: largest_in,
    AveragePool2d: mean_of,
}


def distinct_values_block(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Distinct values, shuffled, spaced 0.1 apart and centred on zero.

    Distinct so that no window ties. Spaced far wider than the 1e-6 nudge so
    that no nudge can change a winner. Centred and scaled small so that the
    summary stays near zero: a central difference divides by 2e-6, which
    multiplies the summary's own rounding by 5e5, and a block of values in the
    hundreds would put that amplified noise at the tolerance itself.
    """
    generator = np.random.default_rng(seed)
    n_values = int(np.prod(shape))
    values = generator.permutation(np.arange(n_values, dtype=np.float64))
    return ((values - n_values / 2.0) * 0.1).reshape(shape)


def finite_difference_slopes(
    layer: Pool2d, block: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """How the summary really moves when each input value is nudged.

    Typed on the base rather than on one subclass, because the family check runs
    it over both of them and it is the same oracle either way. It only ever asks
    the layer to respond, so it cannot inherit a mistake from the backward pass
    it is judging.
    """
    step = 1e-6
    measured = np.empty_like(block)

    for position in np.ndindex(*block.shape):
        moved = []
        for direction in (+step, -step):
            nudged = block.copy()
            nudged[position] += direction
            moved.append(summary_of(layer.respond_to(nudged).outputs, weights))
        measured[position] = (moved[0] - moved[1]) / (2.0 * step)

    return measured


class TestTheShapeArithmetic:
    """Every extent is known at construction, so the whole shape is."""

    def test_two_by_two_at_stride_two_halves_each_spatial_axis(self) -> None:
        layer = MaxPool2d(reads=(1, 28, 28), window=2, stride=2)

        assert layer.shape == LayerShape(n_inputs=(1, 28, 28), n_outputs=(1, 14, 14))

    def test_the_channel_count_is_carried_through_untouched(self) -> None:
        """Two channels are two detectors, and one says nothing about the other."""
        layer = MaxPool2d(reads=(8, 26, 26), window=2, stride=2)

        assert layer.shape.answers[0] == 8
        assert layer.shape.answers == (8, 13, 13)

    def test_an_odd_extent_loses_the_remainder(self) -> None:
        """``(5 - 2) // 2 + 1`` is 2, and the fifth row is never in a window."""
        layer = MaxPool2d(reads=(1, 5, 5), window=2, stride=2)

        assert layer.shape.answers == (1, 2, 2)

    def test_stride_one_barely_shrinks_anything(self) -> None:
        layer = MaxPool2d(reads=(3, 5, 5), window=2, stride=1)

        assert layer.shape.answers == (3, 4, 4)

    def test_the_two_spatial_axes_are_computed_separately(self) -> None:
        """``(7 - 2) // 3 + 1`` is 2 and ``(4 - 2) // 3 + 1`` is 1."""
        layer = MaxPool2d(reads=(1, 7, 4), window=2, stride=3)

        assert layer.shape.answers == (1, 2, 1)

    def test_a_window_the_size_of_the_picture_answers_with_one_number(self) -> None:
        layer = MaxPool2d(reads=(4, 6, 6), window=6, stride=6)

        assert layer.shape.answers == (4, 1, 1)

    def test_the_element_count_falls_by_the_pooling_factor(self) -> None:
        layer = MaxPool2d(reads=(2, 8, 8), window=2, stride=2)

        assert layer.shape.n_inputs == 128
        assert layer.shape.n_outputs == 32

    def test_two_pools_chain(self) -> None:
        """The shape agreement is decidable before a single row arrives."""
        first = MaxPool2d(reads=(3, 8, 8), window=2, stride=2)
        second = MaxPool2d(reads=(3, 4, 4), window=2, stride=2)

        assert second.shape.follows(first.shape)

    def test_the_defaults_are_two_and_two(self) -> None:
        layer = MaxPool2d(reads=(1, 8, 8))

        assert layer.window == 2
        assert layer.stride == 2


class TestConstructionRefusals:
    def test_a_window_below_one_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            MaxPool2d(reads=(1, 8, 8), window=0, stride=2)

    def test_a_negative_window_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            MaxPool2d(reads=(1, 8, 8), window=-2, stride=2)

    def test_a_stride_below_one_is_refused(self) -> None:
        """A stride of zero never advances, so the window never moves."""
        with pytest.raises(InvalidValuesError):
            MaxPool2d(reads=(1, 8, 8), window=2, stride=0)

    def test_a_negative_stride_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            MaxPool2d(reads=(1, 8, 8), window=2, stride=-1)

    def test_a_fractional_window_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            MaxPool2d(reads=(1, 8, 8), window=2.5, stride=2)  # type: ignore[arg-type]

    def test_a_boolean_window_is_refused(self) -> None:
        """``True`` indexes as 1 and would otherwise pass as a window of one."""
        with pytest.raises(InvalidValuesError):
            MaxPool2d(reads=(1, 8, 8), window=True, stride=2)  # type: ignore[arg-type]

    def test_a_window_taller_than_the_picture_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            MaxPool2d(reads=(1, 3, 8), window=4, stride=2)

    def test_a_window_wider_than_the_picture_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            MaxPool2d(reads=(1, 8, 3), window=4, stride=2)

    def test_two_extents_are_not_a_picture(self) -> None:
        with pytest.raises(ShapeMismatchError):
            MaxPool2d(reads=(8, 8), window=2, stride=2)

    def test_four_extents_are_not_a_picture_either(self) -> None:
        with pytest.raises(ShapeMismatchError):
            MaxPool2d(reads=(1, 1, 8, 8), window=2, stride=2)

    def test_a_zero_extent_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            MaxPool2d(reads=(0, 8, 8), window=2, stride=2)

    def test_a_fractional_extent_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            MaxPool2d(reads=(1, 8.5, 8), window=2, stride=2)  # type: ignore[arg-type]

    def test_a_stride_larger_than_the_window_is_allowed(self) -> None:
        """It skips values rather than reusing them, which is legal if unusual."""
        layer = MaxPool2d(reads=(1, 8, 8), window=2, stride=4)

        assert layer.shape.answers == (1, 2, 2)


class TestForwardRefusals:
    def test_a_three_dimensional_block_is_refused(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(HAND_WORKED_GRID.reshape(1, 4, 4))

    def test_the_wrong_channel_count_is_refused(self) -> None:
        layer = MaxPool2d(reads=(3, 4, 4), window=2, stride=2)

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((2, 2, 4, 4)))

    def test_the_wrong_spatial_extents_are_refused(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(np.zeros((2, 1, 4, 6)))

    def test_a_block_with_no_rows_is_refused(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        with pytest.raises(EmptyValuesError):
            layer.respond_to(np.zeros((0, 1, 4, 4)))

    def test_a_non_finite_entry_is_refused(self) -> None:
        """A ``nan`` compares false against everything and would win nothing."""
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        poisoned = one_picture(HAND_WORKED_GRID.copy())
        poisoned[0, 0, 2, 2] = np.nan

        with pytest.raises(InvalidValuesError):
            layer.respond_to(poisoned)

    def test_an_infinite_entry_is_refused(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        poisoned = one_picture(HAND_WORKED_GRID.copy())
        poisoned[0, 0, 0, 0] = np.inf

        with pytest.raises(InvalidValuesError):
            layer.respond_to(poisoned)

    def test_a_block_that_is_not_numeric_is_refused(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        with pytest.raises(InvalidValuesError):
            layer.respond_to("not a picture")  # type: ignore[arg-type]

    def test_nested_lists_are_accepted(self) -> None:
        """Coercing at the boundary is what makes a small example writable."""
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID).tolist())

        assert np.allclose(response.outputs[0, 0], HAND_WORKED_POOLED)


class TestTheForwardPass:
    def test_it_matches_a_grid_worked_by_hand(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        assert np.allclose(response.outputs[0, 0], HAND_WORKED_POOLED)

    def test_the_answer_has_the_shape_the_layer_promised(self) -> None:
        layer = MaxPool2d(reads=(2, 6, 6), window=3, stride=3)

        response = layer.respond_to(np.zeros((5, 2, 6, 6)))

        assert response.outputs.shape == (5, *layer.shape.answers)

    def test_channels_are_pooled_independently(self) -> None:
        """The second channel is the first negated, so mixing them is visible."""
        layer = MaxPool2d(reads=(2, 4, 4), window=2, stride=2)
        block = np.stack([HAND_WORKED_GRID, -HAND_WORKED_GRID])[np.newaxis]

        response = layer.respond_to(block)

        assert np.allclose(response.outputs[0, 0], HAND_WORKED_POOLED)
        assert np.allclose(
            response.outputs[0, 1],
            np.array([[-1.0, -2.0], [-2.0, -0.0]]),
        )

    def test_rows_are_pooled_independently(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        block = np.stack(
            [one_picture(HAND_WORKED_GRID)[0], one_picture(-HAND_WORKED_GRID)[0]]
        )

        response = layer.respond_to(block)

        assert np.allclose(response.outputs[0, 0], HAND_WORKED_POOLED)
        assert np.allclose(
            response.outputs[1, 0], np.array([[-1.0, -2.0], [-2.0, -0.0]])
        )

    def test_overlapping_windows_reuse_values(self) -> None:
        """Worked by hand: at stride 1 the 6 wins three of the four windows."""
        grid = np.array([[1.0, 3.0, 0.0], [5.0, 6.0, 2.0], [0.0, 0.0, 7.0]])
        layer = MaxPool2d(reads=(1, 3, 3), window=2, stride=1)

        response = layer.respond_to(one_picture(grid))

        assert np.allclose(response.outputs[0, 0], [[6.0, 6.0], [6.0, 7.0]])

    def test_scores_and_outputs_are_the_same_block(self) -> None:
        """No activation, so there is no pre-activation value to keep apart."""
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        assert response.scores is response.outputs

    def test_the_response_carries_the_block_that_was_read(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        assert np.allclose(response.inputs, one_picture(HAND_WORKED_GRID))

    def test_the_answer_is_frozen(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        with pytest.raises(ValueError):
            response.outputs[0, 0, 0, 0] = 99.0

    def test_the_callers_own_block_is_not_frozen_underneath_them(self) -> None:
        """``already_checked`` freezes what it is given, so it is given a copy."""
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        block = one_picture(HAND_WORKED_GRID.copy())

        layer.respond_to(block)
        block[0, 0, 0, 0] = 99.0

        assert block[0, 0, 0, 0] == 99.0


class TestTheBackwardRouting:
    """The whole arriving value to the winner, and zero to everyone else."""

    def test_each_windows_value_lands_on_its_own_winner(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))
        arriving = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)

        correction = layer.correction_for(response, arriving)

        expected = np.zeros((4, 4))
        for value, (winning_row, winning_column) in zip(
            (1.0, 2.0, 3.0, 4.0), HAND_WORKED_WINNERS, strict=True
        ):
            expected[winning_row, winning_column] = value
        assert np.allclose(correction.passed_down[0, 0], expected)

    def test_every_other_position_gets_exactly_zero(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        correction = layer.correction_for(response, np.ones((1, 1, 2, 2)))

        assert int(np.count_nonzero(correction.passed_down)) == 4

    def test_the_block_passed_down_is_the_shape_the_layer_read(self) -> None:
        layer = MaxPool2d(reads=(3, 6, 6), window=3, stride=3)
        response = layer.respond_to(
            np.arange(2 * 3 * 6 * 6, dtype=float).reshape(2, 3, 6, 6)
        )

        correction = layer.correction_for(response, np.ones((2, 3, 2, 2)))

        assert correction.passed_down.shape == (2, 3, 6, 6)

    def test_the_total_blame_is_conserved_when_windows_do_not_overlap(self) -> None:
        """Every arriving value goes somewhere, and nothing is invented."""
        generator = np.random.default_rng(5)
        layer = MaxPool2d(reads=(2, 6, 6), window=2, stride=2)
        response = layer.respond_to(generator.normal(size=(3, 2, 6, 6)))
        arriving = generator.normal(size=(3, 2, 3, 3))

        correction = layer.correction_for(response, arriving)

        assert float(np.sum(correction.passed_down)) == pytest.approx(
            float(np.sum(arriving))
        )

    def test_a_pooling_layer_reports_no_gradient(self) -> None:
        """None rather than a block of zeros, which would claim it learns."""
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        correction = layer.correction_for(response, np.ones((1, 1, 2, 2)))

        assert correction.gradient is None
        assert correction.learns is False

    def test_channels_are_routed_independently(self) -> None:
        layer = MaxPool2d(reads=(2, 4, 4), window=2, stride=2)
        block = np.stack([HAND_WORKED_GRID, np.zeros((4, 4))])[np.newaxis]
        response = layer.respond_to(block)
        arriving = np.zeros((1, 2, 2, 2))
        arriving[0, 0] = 1.0

        correction = layer.correction_for(response, arriving)

        assert float(np.sum(correction.passed_down[0, 0])) == pytest.approx(4.0)
        assert float(np.sum(correction.passed_down[0, 1])) == pytest.approx(0.0)


class TestOverlappingWindowsAccumulate:
    """The test that separates ``+=`` from ``=``, and it is the whole point."""

    def test_a_position_winning_four_windows_receives_all_four_values(self) -> None:
        """The centre is the maximum of every 2x2 window at stride 1."""
        grid = np.array(
            [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]],
        )
        layer = MaxPool2d(reads=(1, 3, 3), window=2, stride=1)
        response = layer.respond_to(one_picture(grid))
        arriving = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)

        correction = layer.correction_for(response, arriving)

        expected = np.zeros((3, 3))
        expected[1, 1] = 10.0
        assert np.allclose(correction.passed_down[0, 0], expected)

    def test_an_assignment_would_have_left_only_the_last_window(self) -> None:
        """Pinned as a number, so the failure mode is named rather than implied."""
        grid = np.array(
            [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]],
        )
        layer = MaxPool2d(reads=(1, 3, 3), window=2, stride=1)
        response = layer.respond_to(one_picture(grid))
        arriving = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)

        correction = layer.correction_for(response, arriving)

        centre = float(correction.passed_down[0, 0, 1, 1])
        assert centre == pytest.approx(10.0)
        # 4.0 is the last window's own arriving value, which is what an
        # assignment would have left sitting there.
        assert centre != pytest.approx(4.0)

    def test_one_position_collects_every_window_it_won(self) -> None:
        """What overlap changes is the concentration, not the total.

        This test used to be called ``test_total_blame_exceeds_the_arriving_
        total_when_windows_overlap``, and its name claimed something its
        assertion did not check and that is in fact false. The shares in every
        window sum to one, so the blame leaving a pooling layer always equals
        the blame arriving, whatever the geometry -- measured on both classes
        at strides 1, 2 and 3, agreeing to the last bit. The total was never
        going to exceed anything.

        The real claim is where the blame lands. The centre of this grid wins
        all four windows and is owed all four arriving values, so it holds the
        entire 4.0 while its neighbours hold nothing. An implementation that
        assigns instead of accumulating leaves the centre holding 1.0, which
        the total alone cannot see, because that version routes the missing 3.0
        nowhere at all and its total is 1.0 rather than 4.0. Asserting the
        position is what distinguishes the two claims, so both are made.
        """
        grid = np.array(
            [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]],
        )
        layer = MaxPool2d(reads=(1, 3, 3), window=2, stride=1)
        response = layer.respond_to(one_picture(grid))

        correction = layer.correction_for(response, np.ones((1, 1, 2, 2)))

        assert float(correction.passed_down[0, 0, 1, 1]) == pytest.approx(4.0)
        assert float(np.sum(correction.passed_down)) == pytest.approx(4.0)


class TestTheTieConvention:
    """No derivative exists here, so this records a decision rather than a fact."""

    def test_a_wholly_constant_window_pays_its_top_left_corner(self) -> None:
        grid = np.full((2, 2), 7.0)
        layer = MaxPool2d(reads=(1, 2, 2), window=2, stride=2)
        response = layer.respond_to(one_picture(grid))

        correction = layer.correction_for(response, np.array([[[[3.0]]]]))

        assert np.allclose(correction.passed_down[0, 0], [[3.0, 0.0], [0.0, 0.0]])

    def test_the_earlier_position_in_row_major_order_takes_everything(self) -> None:
        """``(0, 1)`` flattens to index 1 and ``(1, 0)`` to 2, so ``(0, 1)`` wins."""
        grid = np.array([[1.0, 9.0], [9.0, 2.0]])
        layer = MaxPool2d(reads=(1, 2, 2), window=2, stride=2)
        response = layer.respond_to(one_picture(grid))

        correction = layer.correction_for(response, np.array([[[[1.0]]]]))

        assert np.allclose(correction.passed_down[0, 0], [[0.0, 1.0], [0.0, 0.0]])

    def test_the_top_row_beats_the_bottom_row(self) -> None:
        grid = np.array([[4.0, 1.0], [4.0, 1.0]])
        layer = MaxPool2d(reads=(1, 2, 2), window=2, stride=2)
        response = layer.respond_to(one_picture(grid))

        correction = layer.correction_for(response, np.array([[[[1.0]]]]))

        assert np.allclose(correction.passed_down[0, 0], [[1.0, 0.0], [0.0, 0.0]])

    def test_a_tie_still_answers_the_right_maximum_forwards(self) -> None:
        """Which position won is undecided; what the window is worth is not."""
        grid = np.array([[4.0, 1.0], [4.0, 1.0]])
        layer = MaxPool2d(reads=(1, 2, 2), window=2, stride=2)

        assert float(layer.respond_to(one_picture(grid)).outputs[0, 0, 0, 0]) == 4.0


class TestBackwardRefusals:
    def test_an_arriving_block_of_the_wrong_shape_is_refused(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((1, 1, 3, 3)))

    def test_an_arriving_block_with_the_wrong_row_count_is_refused(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, 1, 2, 2)))

    def test_an_arriving_block_that_is_not_numeric_is_refused(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, "not a slope")  # type: ignore[arg-type]

    def test_a_response_from_a_differently_shaped_layer_is_refused(self) -> None:
        """A pairing mistake, and it would route real numbers to arbitrary places."""
        wide = MaxPool2d(reads=(1, 6, 6), window=2, stride=2)
        narrow = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        response = wide.respond_to(np.zeros((1, 1, 6, 6)))

        with pytest.raises(ShapeMismatchError):
            narrow.correction_for(response, np.ones((1, 1, 2, 2)))


class TestTheGradientCheck:
    """Nudge each input by a hair, watch a scalar summary, compare the claim.

    The oracle for the routing. It is computed from forward passes alone, so a
    backward pass that agrees with it is right for reasons that owe nothing to
    how it was derived.

    Every fixture here uses distinct integer values, which matters twice. It
    guarantees no ties, so the derivative exists everywhere. And it guarantees
    that a nudge of 1e-6 cannot change which position wins a window, since the
    nearest two values in any window are at least 1.0 apart -- so the summary is
    exactly linear across the nudge and the central difference is exact rather
    than approximate.
    """

    def measured_slopes(
        self, layer: MaxPool2d, block: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """How the summary really moves when each input value is nudged.

        The body sits at module scope as :func:`finite_difference_slopes`, moved
        there unchanged once the family check below needed the same oracle. The
        four tests in this class are what they always were.
        """
        return finite_difference_slopes(layer, block, weights)

    def distinct_block(self, shape: tuple[int, ...], seed: int) -> np.ndarray:
        """The fixture builder, at module scope as :func:`distinct_values_block`.

        Moved for the same reason and with the same body. The prose explaining
        the spacing and the scaling went with it.
        """
        return distinct_values_block(shape, seed)

    def test_it_matches_a_finite_difference_on_disjoint_windows(self) -> None:
        shape = (2, 2, 4, 4)
        layer = MaxPool2d(reads=(2, 4, 4), window=2, stride=2)
        block = self.distinct_block(shape, seed=11)
        weights = np.random.default_rng(12).normal(size=(2, 2, 2, 2))

        claimed = layer.correction_for(layer.respond_to(block), weights).passed_down
        measured = self.measured_slopes(layer, block, weights)

        assert np.allclose(claimed, measured, atol=1e-8)

    def test_it_matches_a_finite_difference_on_overlapping_windows(self) -> None:
        """The one that matters: nine windows over sixteen values, at stride 1."""
        shape = (1, 1, 4, 4)
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=1)
        block = self.distinct_block(shape, seed=13)
        weights = np.random.default_rng(14).normal(size=(1, 1, 3, 3))

        claimed = layer.correction_for(layer.respond_to(block), weights).passed_down
        measured = self.measured_slopes(layer, block, weights)

        assert np.allclose(claimed, measured, atol=1e-8)
        # A value winning several windows must show a slope larger than any one
        # of them, or the accumulation is not happening at all.
        assert float(np.max(np.abs(claimed))) > float(np.max(np.abs(weights)))

    def test_it_matches_a_finite_difference_with_a_non_square_window(self) -> None:
        """A stride of 3 and a window of 2 skips values; the skipped ones score 0."""
        shape = (1, 2, 7, 4)
        layer = MaxPool2d(reads=(2, 7, 4), window=2, stride=3)
        block = self.distinct_block(shape, seed=15)
        weights = np.random.default_rng(16).normal(size=(1, 2, 2, 1))

        claimed = layer.correction_for(layer.respond_to(block), weights).passed_down
        measured = self.measured_slopes(layer, block, weights)

        assert np.allclose(claimed, measured, atol=1e-8)

    def test_it_matches_a_finite_difference_over_several_rows(self) -> None:
        """Rows must not bleed into each other, and only a multi-row check says so."""
        shape = (3, 1, 4, 4)
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        block = self.distinct_block(shape, seed=17)
        weights = np.random.default_rng(18).normal(size=(3, 1, 2, 2))

        claimed = layer.correction_for(layer.respond_to(block), weights).passed_down
        measured = self.measured_slopes(layer, block, weights)

        assert np.allclose(claimed, measured, atol=1e-8)


class TestItHasNothingToLearn:
    def test_a_step_answers_with_the_very_same_layer(self) -> None:
        """Not a copy. It is immutable and unchanged, so there is nothing to build."""
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        assert layer.stepped_by(None, learning_rate=0.1) is layer

    def test_a_large_learning_rate_changes_nothing_either(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)

        assert layer.stepped_by(None, learning_rate=1000.0) is layer

    def test_a_stepped_layer_still_answers_identically(self) -> None:
        layer = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        before = np.asarray(layer.respond_to(one_picture(HAND_WORKED_GRID)).outputs)

        stepped = layer.stepped_by(None, learning_rate=0.5)

        # ``stepped_by`` promises a ``Layer``; narrowing is what lets the test
        # ask it to pool something, and it is an assertion worth making anyway.
        assert isinstance(stepped, MaxPool2d)
        after = np.asarray(stepped.respond_to(one_picture(HAND_WORKED_GRID)).outputs)

        assert np.array_equal(before, after)


class TestValueSemantics:
    def test_two_layers_configured_alike_are_equal(self) -> None:
        assert MaxPool2d(reads=(1, 8, 8), window=2, stride=2) == MaxPool2d(
            reads=(1, 8, 8), window=2, stride=2
        )

    def test_a_different_window_and_stride_is_a_different_layer(self) -> None:
        """Both answer with ``(1, 2, 2)``, and they are still not the same layer."""
        wider_window = MaxPool2d(reads=(1, 5, 5), window=3, stride=2)
        longer_stride = MaxPool2d(reads=(1, 5, 5), window=2, stride=3)

        assert wider_window.shape == longer_stride.shape
        assert wider_window != longer_stride

    def test_it_defers_to_anything_that_is_not_a_pooling_layer(self) -> None:
        assert MaxPool2d(reads=(1, 8, 8)).__eq__(object()) is NotImplemented

    def test_it_is_hashable(self) -> None:
        layer = MaxPool2d(reads=(1, 8, 8), window=2, stride=2)

        assert len({layer, MaxPool2d(reads=(1, 8, 8), window=2, stride=2)}) == 1

    def test_its_repr_names_its_configuration(self) -> None:
        text = repr(MaxPool2d(reads=(3, 8, 8), window=2, stride=2))

        assert text == "MaxPool2d(reads=(3, 8, 8), window=2, stride=2)"


class TestTheAverageForwardPass:
    """The mean of a window, on grids worked out before the layer was asked.

    Every number here was computed on paper from the same fixtures the maximum
    is tested against, which is what lets one grid ask both layers a question
    and get two different answers back.
    """

    def test_it_matches_a_grid_worked_by_hand(self) -> None:
        layer = AveragePool2d(reads=(1, 4, 4), window=2, stride=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        assert np.allclose(response.outputs[0, 0], HAND_WORKED_AVERAGED)

    def test_no_quarter_of_that_grid_agrees_with_its_own_maximum(self) -> None:
        """The fixture is only worth having if the two layers part on all of it."""
        assert not np.any(np.isclose(HAND_WORKED_AVERAGED, HAND_WORKED_POOLED))

    def test_a_lone_spike_is_diluted_by_the_rest_of_its_window(self) -> None:
        """The obvious case. One 8 among three zeros is a mean of 2, not of 8."""
        layer = AveragePool2d(reads=(1, 2, 2), window=2, stride=2)

        response = layer.respond_to(one_picture(SPIKED_GRID))

        assert float(response.outputs[0, 0, 0, 0]) == pytest.approx(2.0)

    def test_a_three_by_three_window_divides_by_nine(self) -> None:
        """The top left 3x3 of the hand-worked grid sums to 36, so it means 4."""
        layer = AveragePool2d(reads=(1, 4, 4), window=3, stride=3)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        assert response.outputs.shape == (1, 1, 1, 1)
        assert float(response.outputs[0, 0, 0, 0]) == pytest.approx(4.0)

    def test_overlapping_windows_average_what_each_of_them_covers(self) -> None:
        """Four windows over nine values, and the centre is counted four times."""
        layer = AveragePool2d(reads=(1, 3, 3), window=2, stride=1)

        response = layer.respond_to(one_picture(OVERLAPPING_GRID))

        assert np.allclose(response.outputs[0, 0], OVERLAPPING_AVERAGED)

    def test_a_constant_window_averages_to_that_constant(self) -> None:
        """A mean cannot leave the range of what it read, pinned at one end of it."""
        layer = AveragePool2d(reads=(1, 2, 2), window=2, stride=2)

        response = layer.respond_to(one_picture(np.full((2, 2), 7.0)))

        assert float(response.outputs[0, 0, 0, 0]) == pytest.approx(7.0)

    def test_negatives_drag_the_mean_below_every_positive_it_read(self) -> None:
        """A maximum would answer 4.0 here and can never answer a negative at all."""
        grid = np.array([[4.0, -6.0], [-8.0, 2.0]])
        layer = AveragePool2d(reads=(1, 2, 2), window=2, stride=2)

        response = layer.respond_to(one_picture(grid))

        assert float(response.outputs[0, 0, 0, 0]) == pytest.approx(-2.0)

    def test_channels_are_averaged_independently(self) -> None:
        """A mean is linear, so the negated channel is the negated answer exactly.

        That is a stronger statement than the maximum's own version of this
        test can make, since negating a block moves which value wins each window
        and the answers have to be written out by hand there.
        """
        layer = AveragePool2d(reads=(2, 4, 4), window=2, stride=2)
        block = np.stack([HAND_WORKED_GRID, -HAND_WORKED_GRID])[np.newaxis]

        response = layer.respond_to(block)

        assert np.allclose(response.outputs[0, 0], HAND_WORKED_AVERAGED)
        assert np.allclose(response.outputs[0, 1], -HAND_WORKED_AVERAGED)

    def test_rows_are_averaged_independently(self) -> None:
        layer = AveragePool2d(reads=(1, 4, 4), window=2, stride=2)
        block = np.stack(
            [one_picture(HAND_WORKED_GRID)[0], one_picture(HAND_WORKED_GRID * 2.0)[0]]
        )

        response = layer.respond_to(block)

        assert np.allclose(response.outputs[0, 0], HAND_WORKED_AVERAGED)
        assert np.allclose(response.outputs[1, 0], HAND_WORKED_AVERAGED * 2.0)


class TestTheAverageBackwardSharing:
    """A quarter each on a two-by-two, and no position is ever starved."""

    def test_every_position_receives_its_own_windows_value_over_four(self) -> None:
        """Worked on paper. Each quarter holds one arriving value divided by four."""
        layer = AveragePool2d(reads=(1, 4, 4), window=2, stride=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))
        arriving = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)

        correction = layer.correction_for(response, arriving)

        expected = np.array(
            [
                [0.25, 0.25, 0.50, 0.50],
                [0.25, 0.25, 0.50, 0.50],
                [0.75, 0.75, 1.00, 1.00],
                [0.75, 0.75, 1.00, 1.00],
            ]
        )
        assert np.allclose(correction.passed_down[0, 0], expected)

    def test_nothing_is_starved(self) -> None:
        """Sixteen non-zeros where the maximum leaves four, on the same block."""
        layer = AveragePool2d(reads=(1, 4, 4), window=2, stride=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        correction = layer.correction_for(response, np.ones((1, 1, 2, 2)))

        assert int(np.count_nonzero(correction.passed_down)) == 16

    def test_the_sharing_owes_nothing_to_what_the_window_held(self) -> None:
        """A mean has no winner, so the values read change nothing on the way back.

        The sharpest statement of how the two layers differ, and it needs no
        oracle. Negating the block moves a maximum's winner in every window and
        leaves a mean's shares exactly where they were. A mean that routed like
        a maximum disagrees with itself here by 4.0.
        """
        layer = AveragePool2d(reads=(1, 4, 4), window=2, stride=2)
        arriving = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)

        upright = layer.correction_for(
            layer.respond_to(one_picture(HAND_WORKED_GRID)), arriving
        )
        negated = layer.correction_for(
            layer.respond_to(one_picture(-HAND_WORKED_GRID)), arriving
        )

        assert np.allclose(upright.passed_down, negated.passed_down)

    def test_an_overlapped_centre_collects_a_quarter_from_all_four_windows(
        self,
    ) -> None:
        """Worked on paper, and the test that separates ``+=`` from ``=`` by hand.

        Every window hands each of its four positions a quarter of its own
        value. The centre belongs to all four windows, so it collects
        ``(1 + 2 + 3 + 4) / 4``, and the corners belong to one window each.
        """
        layer = AveragePool2d(reads=(1, 3, 3), window=2, stride=1)
        response = layer.respond_to(one_picture(OVERLAPPING_GRID))
        arriving = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)

        correction = layer.correction_for(response, arriving)

        expected = np.array(
            [
                [0.25, 0.75, 0.50],
                [1.00, 2.50, 1.50],
                [0.75, 1.75, 1.00],
            ]
        )
        assert np.allclose(correction.passed_down[0, 0], expected)

    def test_an_assignment_would_have_left_the_centre_at_one(self) -> None:
        """Pinned as a number, so the failure mode is named rather than implied."""
        layer = AveragePool2d(reads=(1, 3, 3), window=2, stride=1)
        response = layer.respond_to(one_picture(OVERLAPPING_GRID))
        arriving = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)

        correction = layer.correction_for(response, arriving)

        centre = float(correction.passed_down[0, 0, 1, 1])
        assert centre == pytest.approx(2.5)
        # 1.0 is the last window's own 4.0 shared four ways, which is what an
        # assignment would have left sitting there.
        assert centre != pytest.approx(1.0)

    def test_a_three_by_three_window_hands_out_ninths(self) -> None:
        """The denominator is the window's area rather than a hard-coded four."""
        layer = AveragePool2d(reads=(1, 3, 3), window=3, stride=3)
        response = layer.respond_to(one_picture(OVERLAPPING_GRID))

        correction = layer.correction_for(response, np.array([[[[9.0]]]]))

        assert np.allclose(correction.passed_down[0, 0], np.full((3, 3), 1.0))

    def test_channels_are_shared_independently(self) -> None:
        layer = AveragePool2d(reads=(2, 4, 4), window=2, stride=2)
        block = np.stack([HAND_WORKED_GRID, np.zeros((4, 4))])[np.newaxis]
        response = layer.respond_to(block)
        arriving = np.zeros((1, 2, 2, 2))
        arriving[0, 0] = 1.0

        correction = layer.correction_for(response, arriving)

        assert float(np.sum(correction.passed_down[0, 0])) == pytest.approx(4.0)
        assert float(np.sum(correction.passed_down[0, 1])) == pytest.approx(0.0)


class TestThePoolingFamilyContract:
    """What every pooling layer must do, asserted of every pooling layer.

    :class:`~oop_ml.core.network.pooling.Pool2d` owns the geometry, the sweep,
    the refusals, the absence of parameters and the algebra of the shares, and a
    subclass supplies two functions of one window. So these claims belong to the
    base and are made here of both subclasses at once, which is what stops the
    next kind of pooling inheriting a contract nobody checks it against.

    The shares get four separate claims because they fail in four separate ways
    and no one of them catches the others. See the module docstring for the two
    broken means that make that concrete.
    """

    WINDOWS = [
        pytest.param(np.array([[1.0, 5.0], [3.0, 2.0]]), id="ordinary"),
        pytest.param(np.array([[7.0, 7.0], [7.0, 7.0]]), id="constant"),
        pytest.param(np.array([[-1.0, -5.0], [-3.0, -2.0]]), id="all negative"),
        pytest.param(np.array([[0.0]]), id="one position"),
        pytest.param(
            np.array([[9.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
            id="three by three",
        ),
        pytest.param(np.arange(16.0).reshape(4, 4), id="four by four"),
        # Side 7 is one of the sixteen sides in 1 to 32 where a mean's shares do
        # not sum to exactly one in float64. It is here so that the inexact case
        # is run rather than only described.
        pytest.param(np.arange(49.0).reshape(7, 7), id="seven by seven"),
    ]

    def layer_reading(self, pooling: type[Pool2d], window: np.ndarray) -> Pool2d:
        """A layer whose one window is exactly the array being asked about."""
        side = int(window.shape[0])
        return pooling(reads=(1, side, side), window=side, stride=side)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("window", WINDOWS)
    def test_the_shares_sum_to_one(
        self, pooling: type[Pool2d], window: np.ndarray
    ) -> None:
        """Which is what stops a pooling layer scaling the gradient going past.

        Not an exact equality, because it is not exactly true. Measured over
        window sides 1 to 32, sixteen of them leave a mean's shares one or two
        units in the last place from one, worst 4.4e-16. A few of those units is
        the claim that holds, and it is still four thousand million million
        times tighter than the 4.0 a mean that dropped its ``1 / n`` produces.
        """
        layer = self.layer_reading(pooling, window)

        total = float(np.sum(layer.shares_of(window)))

        assert total == pytest.approx(1.0, abs=1e-15)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("window", WINDOWS)
    def test_the_shares_have_the_window_s_shape(
        self, pooling: type[Pool2d], window: np.ndarray
    ) -> None:
        """One share per position, so the backward pass can add it in place."""
        layer = self.layer_reading(pooling, window)

        assert layer.shares_of(window).shape == window.shape

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("window", WINDOWS)
    def test_no_share_is_negative(
        self, pooling: type[Pool2d], window: np.ndarray
    ) -> None:
        """A negative share sends a position blame of the wrong sign.

        Raising the input would then lower the answer, which no pooling layer
        here does. Note that it is a claim about the shares and not about the
        window, so the all-negative fixture is the one that matters.
        """
        layer = self.layer_reading(pooling, window)

        assert bool(np.all(layer.shares_of(window) >= 0.0))

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("window", WINDOWS)
    def test_the_answer_is_what_the_definition_says_it_is(
        self, pooling: type[Pool2d], window: np.ndarray
    ) -> None:
        """Against a plain Python loop written from the definition, not the body."""
        layer = self.layer_reading(pooling, window)

        assert layer.summarise(window) == pytest.approx(
            ANSWER_BY_DEFINITION[pooling](window)
        )

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("window", WINDOWS)
    def test_the_shares_recover_the_answer(
        self, pooling: type[Pool2d], window: np.ndarray
    ) -> None:
        """``sum(shares * window)`` is the answer, which is what a weight means.

        The claim that ties a subclass's two methods to each other. Both are
        compared against the same independent loop rather than against one
        another, so a subclass that got both of them wrong the same way still
        fails.
        """
        layer = self.layer_reading(pooling, window)

        weighted = float(np.sum(layer.shares_of(window) * window))

        assert weighted == pytest.approx(ANSWER_BY_DEFINITION[pooling](window))

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize(
        ("height", "width", "window", "stride", "answers"),
        [
            (28, 28, 2, 2, (14, 14)),
            (5, 5, 2, 2, (2, 2)),
            (5, 5, 2, 1, (4, 4)),
            (7, 4, 2, 3, (2, 1)),
            (6, 6, 6, 6, (1, 1)),
            (8, 8, 2, 4, (2, 2)),
            (9, 4, 3, 2, (4, 1)),
            (1, 1, 1, 1, (1, 1)),
        ],
    )
    def test_the_output_extents_are_the_arithmetic(
        self,
        pooling: type[Pool2d],
        height: int,
        width: int,
        window: int,
        stride: int,
        answers: tuple[int, int],
    ) -> None:
        """``(extent - window) // stride + 1``, per spatial axis and per class."""
        layer = pooling(reads=(3, height, width), window=window, stride=stride)

        assert layer.shape.answers == (3, *answers)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_the_channel_count_is_carried_through_untouched(
        self, pooling: type[Pool2d]
    ) -> None:
        """Pooling shrinks the two spatial axes and mixes no detectors."""
        layer = pooling(reads=(8, 26, 26), window=2, stride=2)

        assert layer.shape == LayerShape(n_inputs=(8, 26, 26), n_outputs=(8, 13, 13))

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_the_defaults_are_two_and_two(self, pooling: type[Pool2d]) -> None:
        layer = pooling(reads=(1, 8, 8))

        assert layer.window == 2
        assert layer.stride == 2

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_scores_and_outputs_are_the_same_block(self, pooling: type[Pool2d]) -> None:
        """No activation anywhere in the family, so no pre-activation value."""
        layer = pooling(reads=(1, 4, 4), window=2, stride=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        assert response.scores is response.outputs

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_the_answer_is_frozen(self, pooling: type[Pool2d]) -> None:
        layer = pooling(reads=(1, 4, 4), window=2, stride=2)

        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        with pytest.raises(ValueError):
            response.outputs[0, 0, 0, 0] = 99.0

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_it_reports_no_gradient(self, pooling: type[Pool2d]) -> None:
        """None rather than a block of zeros, which would claim it learns."""
        layer = pooling(reads=(1, 4, 4), window=2, stride=2)
        response = layer.respond_to(one_picture(HAND_WORKED_GRID))

        correction = layer.correction_for(response, np.ones((1, 1, 2, 2)))

        assert correction.gradient is None
        assert correction.learns is False

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("learning_rate", [0.0, 0.1, 1000.0])
    def test_a_step_answers_with_the_very_same_layer(
        self, pooling: type[Pool2d], learning_rate: float
    ) -> None:
        """Identity, not an equal copy. There is nothing in it to move."""
        layer = pooling(reads=(1, 4, 4), window=2, stride=2)

        assert layer.stepped_by(None, learning_rate=learning_rate) is layer

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_a_stepped_layer_still_answers_identically(
        self, pooling: type[Pool2d]
    ) -> None:
        layer = pooling(reads=(1, 4, 4), window=2, stride=2)
        before = np.asarray(layer.respond_to(one_picture(HAND_WORKED_GRID)).outputs)

        stepped = layer.stepped_by(None, learning_rate=0.5)

        assert isinstance(stepped, Pool2d)
        after = np.asarray(stepped.respond_to(one_picture(HAND_WORKED_GRID)).outputs)
        assert np.array_equal(before, after)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize(
        ("window", "stride"),
        [
            pytest.param(2, 2, id="disjoint"),
            pytest.param(2, 1, id="overlapping by one"),
            pytest.param(3, 1, id="overlapping by two"),
            pytest.param(2, 3, id="skipping values"),
        ],
    )
    def test_the_total_blame_equals_the_total_arriving(
        self, pooling: type[Pool2d], window: int, stride: int
    ) -> None:
        """A corollary of the shares summing to one, and it holds when they overlap.

        Overlapping looks like it should manufacture blame, and it does not.
        Each window hands out its own arriving value and no more, so the two
        totals agree whatever the stride does. What overlapping changes is where
        the total lands, which the hand-worked routing tests pin separately.
        """
        generator = np.random.default_rng(3)
        layer = pooling(reads=(2, 7, 7), window=window, stride=stride)
        response = layer.respond_to(generator.normal(size=(3, 2, 7, 7)))
        arriving = generator.normal(size=(3, *layer.shape.answers))

        correction = layer.correction_for(response, arriving)

        assert float(np.sum(correction.passed_down)) == pytest.approx(
            float(np.sum(arriving)), rel=1e-12
        )

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_the_block_passed_down_is_the_shape_the_layer_read(
        self, pooling: type[Pool2d]
    ) -> None:
        layer = pooling(reads=(3, 6, 6), window=3, stride=3)
        response = layer.respond_to(
            np.arange(2 * 3 * 6 * 6, dtype=float).reshape(2, 3, 6, 6)
        )

        correction = layer.correction_for(response, np.ones((2, 3, 2, 2)))

        assert correction.passed_down.shape == (2, 3, 6, 6)


class TestConstructionRefusalsForEveryPoolingLayer:
    """The refusals live on the base, so both subclasses have to make all of them.

    The class-only versions of these above test one subclass, which was the only
    subclass when they were written. A refusal that a second subclass quietly
    stopped making is exactly the kind of thing a family test exists to catch.
    """

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize(
        "reads", [5, None, 3.5], ids=["a bare width", "nothing", "a fraction"]
    )
    def test_reads_that_is_not_a_sequence_of_extents_is_refused_by_name(
        self, pooling: type[Pool2d], reads: object
    ) -> None:
        """The refusal has to be one of this library's own, not builtins'.

        ``tuple(5)`` raises ``TypeError: 'int' object is not iterable``, which
        is not an ``MLLibError`` and so escapes the hierarchy every failure here
        is supposed to stay inside. The bare width is the mistake worth
        naming, because it is exactly what a dense layer's ``reads`` looks like
        and a reader moving between the two will write it.
        """
        with pytest.raises(MLLibError):
            pooling(reads=reads, window=2, stride=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("window", [0, -2], ids=["zero", "negative"])
    def test_a_window_below_one_is_refused(
        self, pooling: type[Pool2d], window: int
    ) -> None:
        """A window of zero selects nothing, so it is an absent layer not a small one."""
        with pytest.raises(InvalidValuesError):
            pooling(reads=(1, 8, 8), window=window, stride=2)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("stride", [0, -1], ids=["zero", "negative"])
    def test_a_stride_below_one_is_refused(
        self, pooling: type[Pool2d], stride: int
    ) -> None:
        """A stride of zero never advances, so the window never moves."""
        with pytest.raises(InvalidValuesError):
            pooling(reads=(1, 8, 8), window=2, stride=stride)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("window", [2.5, True], ids=["fractional", "boolean"])
    def test_a_window_that_is_not_a_whole_number_is_refused(
        self, pooling: type[Pool2d], window: object
    ) -> None:
        """``True`` indexes as 1 and would otherwise pass as a window of one."""
        with pytest.raises(InvalidValuesError):
            pooling(reads=(1, 8, 8), window=window, stride=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("stride", [1.5, True], ids=["fractional", "boolean"])
    def test_a_stride_that_is_not_a_whole_number_is_refused(
        self, pooling: type[Pool2d], stride: object
    ) -> None:
        """The same guard on the other count, which nothing tested before."""
        with pytest.raises(InvalidValuesError):
            pooling(reads=(1, 8, 8), window=2, stride=stride)  # type: ignore[arg-type]

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize(
        "reads", [(1, 3, 8), (1, 8, 3)], ids=["taller than", "wider than"]
    )
    def test_a_window_that_does_not_fit_the_picture_is_refused(
        self, pooling: type[Pool2d], reads: tuple[int, int, int]
    ) -> None:
        """Only one axis has to fail. A window that does not fit yields no positions."""
        with pytest.raises(ShapeMismatchError):
            pooling(reads=reads, window=4, stride=2)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize(
        "reads",
        [(), (8,), (8, 8), (1, 1, 8, 8)],
        ids=["none", "one", "two", "four"],
    )
    def test_a_picture_that_is_not_three_extents_is_refused(
        self, pooling: type[Pool2d], reads: tuple[int, ...]
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            pooling(reads=reads, window=2, stride=2)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize(
        "reads",
        [(0, 8, 8), (1, 0, 8), (1, 8, 0), (-1, 8, 8)],
        ids=["no channels", "no height", "no width", "negative channels"],
    )
    def test_an_extent_below_one_is_refused(
        self, pooling: type[Pool2d], reads: tuple[int, int, int]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            pooling(reads=reads, window=1, stride=1)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize(
        "reads",
        [(1, 8.5, 8), (1, 8, 8.5), (True, 8, 8)],
        ids=["fractional height", "fractional width", "boolean channels"],
    )
    def test_an_extent_that_is_not_a_whole_number_is_refused(
        self, pooling: type[Pool2d], reads: tuple[object, object, object]
    ) -> None:
        with pytest.raises(InvalidValuesError):
            pooling(reads=reads, window=2, stride=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_a_stride_larger_than_the_window_is_allowed(
        self, pooling: type[Pool2d]
    ) -> None:
        """It skips values rather than reusing them, which is legal if unusual."""
        layer = pooling(reads=(1, 8, 8), window=2, stride=4)

        assert layer.shape.answers == (1, 2, 2)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_a_window_the_size_of_the_picture_is_allowed(
        self, pooling: type[Pool2d]
    ) -> None:
        """The boundary the refusal above sits on, so the guard is ``>`` not ``>=``."""
        layer = pooling(reads=(4, 6, 6), window=6, stride=6)

        assert layer.shape.answers == (4, 1, 1)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize(
        "block",
        [
            pytest.param(np.zeros((2, 1, 4, 6)), id="wrong width"),
            pytest.param(np.zeros((2, 3, 4, 4)), id="wrong channel count"),
            pytest.param(np.zeros((1, 4, 4)), id="three dimensions"),
        ],
    )
    def test_a_block_arranged_some_other_way_is_refused(
        self, pooling: type[Pool2d], block: np.ndarray
    ) -> None:
        """Named specifically rather than as a family of errors, per the charter."""
        layer = pooling(reads=(1, 4, 4), window=2, stride=2)

        with pytest.raises(ShapeMismatchError):
            layer.respond_to(block)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_a_block_with_no_rows_is_refused(self, pooling: type[Pool2d]) -> None:
        """Emptiness is its own refusal, not a shape that happens to be wrong."""
        layer = pooling(reads=(1, 4, 4), window=2, stride=2)

        with pytest.raises(EmptyValuesError):
            layer.respond_to(np.zeros((0, 1, 4, 4)))

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_entry_is_refused(
        self, pooling: type[Pool2d], poison: float
    ) -> None:
        """A ``nan`` compares false against everything and would win nothing."""
        layer = pooling(reads=(1, 4, 4), window=2, stride=2)
        poisoned = one_picture(HAND_WORKED_GRID.copy())
        poisoned[0, 0, 2, 2] = poison

        with pytest.raises(InvalidValuesError):
            layer.respond_to(poisoned)


class TestTheTwoAreNotInterchangeable:
    """Identical geometry, and still not the same layer.

    ``Pool2d.__eq__`` narrows on ``type(self) is type(other)`` rather than on
    ``isinstance``, and this is the class that says why. A maximum and a mean
    configured alike agree about every number that configures them and disagree
    about the first block either of them reads, so treating them as
    interchangeable would put the wrong layer into a cache, a set or a stack.
    """

    def geometry_alike(self) -> tuple[MaxPool2d, AveragePool2d]:
        """One maximum and one mean, configured identically."""
        return (
            MaxPool2d(reads=(1, 8, 8), window=2, stride=2),
            AveragePool2d(reads=(1, 8, 8), window=2, stride=2),
        )

    def test_they_agree_about_every_number_that_configures_them(self) -> None:
        """So the inequality below is about their types and about nothing else."""
        maximum, mean = self.geometry_alike()

        assert maximum.shape == mean.shape
        assert maximum.window == mean.window
        assert maximum.stride == mean.stride

    def test_a_maximum_and_a_mean_of_one_geometry_are_not_equal(self) -> None:
        maximum, mean = self.geometry_alike()

        assert maximum != mean
        # Both operands defer, so the fallback is identity and the answer is a
        # real ``False`` rather than the sentinel leaking out to the caller.
        assert (maximum == mean) is False

    def test_each_defers_rather_than_answering_for_the_other(self) -> None:
        """``NotImplemented`` is *returned*, so Python retries the other operand.

        Which then defers in turn, and the comparison falls back to identity and
        comes out false. Raising the sentinel instead would kill that fallback,
        and this assertion is what an editor's quick-fix would break -- see the
        eight comparison methods the project notes record.
        """
        maximum, mean = self.geometry_alike()

        assert maximum.__eq__(mean) is NotImplemented
        assert mean.__eq__(maximum) is NotImplemented

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_it_defers_to_anything_that_is_not_a_pooling_layer(
        self, pooling: type[Pool2d]
    ) -> None:
        assert pooling(reads=(1, 8, 8)).__eq__(object()) is NotImplemented

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_two_of_one_class_configured_alike_are_equal(
        self, pooling: type[Pool2d]
    ) -> None:
        """The other half of the rule, which is what makes the narrowing a narrowing."""
        assert pooling(reads=(2, 8, 8), window=3, stride=1) == pooling(
            reads=(2, 8, 8), window=3, stride=1
        )

    def test_a_set_keeps_both_of_them(self) -> None:
        """The claim that actually matters, and it needs only ``__eq__`` to be false."""
        maximum, mean = self.geometry_alike()

        assert len({maximum, mean}) == 2

    def test_a_dictionary_keyed_on_them_does_not_collapse(self) -> None:
        """A registry of layers is the shape of code this protects."""
        maximum, mean = self.geometry_alike()
        registry = {maximum: "largest", mean: "mean"}

        assert registry[MaxPool2d(reads=(1, 8, 8), window=2, stride=2)] == "largest"
        assert registry[AveragePool2d(reads=(1, 8, 8), window=2, stride=2)] == "mean"

    def test_their_hashes_differ(self) -> None:
        """The class itself is one of the four things hashed, so they land apart.

        A collision would be a coincidence rather than a design, and the set
        above is the claim that would survive one. This is here because a hash
        built from the geometry alone would put every pooling layer of one shape
        in a single bucket, which is a real cost rather than a wrong answer.
        """
        maximum, mean = self.geometry_alike()

        assert hash(maximum) != hash(mean)

    @pytest.mark.parametrize(
        ("pooling", "name"),
        [(MaxPool2d, "MaxPool2d"), (AveragePool2d, "AveragePool2d")],
    )
    def test_each_repr_names_its_own_class(
        self, pooling: type[Pool2d], name: str
    ) -> None:
        text = repr(pooling(reads=(3, 8, 8), window=2, stride=2))

        assert text == f"{name}(reads=(3, 8, 8), window=2, stride=2)"


class TestMaximumAndMeanDisagree:
    """The tests that would fail if one of the two were quietly the other.

    Everything else in this file would pass if ``AveragePool2d`` were a second
    name for ``MaxPool2d``, or if the shared sweep called the wrong subclass's
    methods. These are the ones that would not.

    Forwards they part on any window that is not constant. Backwards they part
    on every window there is, including the constant one, and that is the
    difference that matters. A maximum sends the whole arriving value to one
    position and starves the rest, while a mean pays everybody.
    """

    def test_they_answer_the_same_spike_differently(self) -> None:
        """8.0 against 2.0 on one window, which is as plain as it gets."""
        maximum = MaxPool2d(reads=(1, 2, 2), window=2, stride=2)
        mean = AveragePool2d(reads=(1, 2, 2), window=2, stride=2)
        picture = one_picture(SPIKED_GRID)

        assert float(maximum.respond_to(picture).outputs[0, 0, 0, 0]) == 8.0
        assert float(mean.respond_to(picture).outputs[0, 0, 0, 0]) == 2.0

    def test_they_disagree_about_every_window_of_the_hand_worked_grid(self) -> None:
        maximum = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        mean = AveragePool2d(reads=(1, 4, 4), window=2, stride=2)
        picture = one_picture(HAND_WORKED_GRID)

        largest = np.asarray(maximum.respond_to(picture).outputs)
        averaged = np.asarray(mean.respond_to(picture).outputs)

        assert not np.any(np.isclose(largest, averaged))

    def test_they_answer_a_constant_window_alike(self) -> None:
        """The one window where the two agree, which is what the next test needs."""
        picture = one_picture(np.full((2, 2), 7.0))
        maximum = MaxPool2d(reads=(1, 2, 2), window=2, stride=2)
        mean = AveragePool2d(reads=(1, 2, 2), window=2, stride=2)

        assert float(maximum.respond_to(picture).outputs[0, 0, 0, 0]) == 7.0
        assert float(mean.respond_to(picture).outputs[0, 0, 0, 0]) == 7.0

    def test_and_still_share_that_constant_window_out_differently(self) -> None:
        """Agreeing forwards is not agreeing. The backward pass is the real split.

        On the window where the two answers are indistinguishable, the maximum
        pays its top-left corner everything and the mean pays all four corners a
        quarter each. A forward-only spec would have called these two layers the
        same layer here.
        """
        picture = one_picture(np.full((2, 2), 7.0))
        arriving = np.array([[[[3.0]]]])
        maximum = MaxPool2d(reads=(1, 2, 2), window=2, stride=2)
        mean = AveragePool2d(reads=(1, 2, 2), window=2, stride=2)

        by_maximum = maximum.correction_for(maximum.respond_to(picture), arriving)
        by_mean = mean.correction_for(mean.respond_to(picture), arriving)

        assert np.allclose(by_maximum.passed_down[0, 0], [[3.0, 0.0], [0.0, 0.0]])
        assert np.allclose(by_mean.passed_down[0, 0], np.full((2, 2), 0.75))

    def test_the_maximum_starves_three_of_four_and_the_mean_starves_none(self) -> None:
        """Counted rather than described, over a whole block of sixteen values."""
        picture = one_picture(HAND_WORKED_GRID)
        arriving = np.ones((1, 1, 2, 2))
        maximum = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        mean = AveragePool2d(reads=(1, 4, 4), window=2, stride=2)

        by_maximum = maximum.correction_for(maximum.respond_to(picture), arriving)
        by_mean = mean.correction_for(mean.respond_to(picture), arriving)

        assert int(np.count_nonzero(by_maximum.passed_down)) == 4
        assert int(np.count_nonzero(by_mean.passed_down)) == 16

    def test_only_the_maximum_changes_its_routing_when_the_window_does(self) -> None:
        """A maximum's shares read the data and a mean's do not, stated as a test.

        Negating the block moves the winner of every window here, so the
        maximum's blame moves with it. The mean's does not move at all, and a
        mean that routed like a maximum disagrees with itself by 4.0 on this
        fixture.
        """
        arriving = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)
        upright = one_picture(HAND_WORKED_GRID)
        negated = one_picture(-HAND_WORKED_GRID)
        maximum = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        mean = AveragePool2d(reads=(1, 4, 4), window=2, stride=2)

        assert not np.allclose(
            maximum.correction_for(maximum.respond_to(upright), arriving).passed_down,
            maximum.correction_for(maximum.respond_to(negated), arriving).passed_down,
        )
        assert np.allclose(
            mean.correction_for(mean.respond_to(upright), arriving).passed_down,
            mean.correction_for(mean.respond_to(negated), arriving).passed_down,
        )

    def test_they_route_differently_and_still_hand_out_the_same_total(self) -> None:
        """Both sets of shares sum to one, so both totals are the arriving total."""
        picture = one_picture(HAND_WORKED_GRID)
        arriving = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)
        maximum = MaxPool2d(reads=(1, 4, 4), window=2, stride=2)
        mean = AveragePool2d(reads=(1, 4, 4), window=2, stride=2)

        by_maximum = maximum.correction_for(maximum.respond_to(picture), arriving)
        by_mean = mean.correction_for(mean.respond_to(picture), arriving)

        assert not np.allclose(by_maximum.passed_down, by_mean.passed_down)
        assert float(np.sum(by_maximum.passed_down)) == pytest.approx(10.0)
        assert float(np.sum(by_mean.passed_down)) == pytest.approx(10.0)


class TestTheGradientCheckHoldsForEveryPoolingLayer:
    """The same oracle, the same five geometries, both classes.

    Running one fixture through both layers is what makes a disagreement
    attributable to the layer rather than to the fixture, and it is the only
    coverage ``AveragePool2d``'s backward pass has that owes nothing to how it
    was derived.

    The tolerance is 1e-7 rather than the 1e-8 used above, and the reason is the
    fixtures rather than the layers. These blocks are larger, a central
    difference multiplies the summary's own rounding by 5e5, and the measured
    honest floor across all ten cases is 1.2e-09. Against a smallest measured
    break of 0.462 that leaves eighty times of room beneath the tolerance and
    six orders of magnitude above it. See the module docstring for the seven
    broken passes those numbers came from.
    """

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    @pytest.mark.parametrize(
        ("reads", "window", "stride", "n_rows", "seed"),
        [
            pytest.param((2, 4, 4), 2, 2, 2, 21, id="disjoint, two channels"),
            pytest.param((1, 4, 4), 2, 1, 1, 23, id="overlapping at stride one"),
            pytest.param((2, 7, 4), 2, 3, 1, 25, id="rectangular, skipping values"),
            pytest.param((1, 4, 4), 2, 2, 3, 27, id="three rows"),
        ],
    )
    def test_the_routing_matches_a_finite_difference(
        self,
        pooling: type[Pool2d],
        reads: tuple[int, int, int],
        window: int,
        stride: int,
        n_rows: int,
        seed: int,
    ) -> None:
        """Nudge each input by a hair, watch a scalar summary, compare the claim.

        The overlapping row is the one an assignment fails, by 1.69 for the
        maximum and 0.462 for the mean. The other three cannot see an assignment
        at all, since a position belonging to one window is written once either
        way, and they are here for the rest of the bookkeeping.
        """
        layer = pooling(reads=reads, window=window, stride=stride)
        block = distinct_values_block((n_rows, *reads), seed=seed)
        weights = np.random.default_rng(seed + 1).normal(
            size=(n_rows, *layer.shape.answers)
        )

        claimed = layer.correction_for(layer.respond_to(block), weights).passed_down
        measured = finite_difference_slopes(layer, block, weights)

        assert np.allclose(claimed, measured, atol=1e-7)

    @pytest.mark.parametrize("pooling", POOLING_CLASSES)
    def test_the_check_holds_where_the_window_is_the_whole_picture(
        self, pooling: type[Pool2d]
    ) -> None:
        """One window, one answer, and no sweep at all -- the degenerate geometry."""
        layer = pooling(reads=(2, 3, 3), window=3, stride=3)
        block = distinct_values_block((2, 2, 3, 3), seed=29)
        weights = np.random.default_rng(30).normal(size=(2, 2, 1, 1))

        claimed = layer.correction_for(layer.respond_to(block), weights).passed_down
        measured = finite_difference_slopes(layer, block, weights)

        assert np.allclose(claimed, measured, atol=1e-7)
