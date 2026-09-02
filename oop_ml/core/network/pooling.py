"""Summarising each window down to one number, and sharing the blame back out.

What pooling is for
-------------------
A convolution answers with a bank of feature maps that are almost as large as
the picture it read, and most of what they hold is repetition: a detector that
fires on an edge fires again one pixel along. Pooling is the layer that says so.
It slides a window over each map and keeps a single number per window, which
does two things at once. It shrinks the map, so the layers above it read
something smaller, and it makes the answer insensitive to where inside the
window the evidence actually sat, which is the small translation tolerance a
picture classifier wants.

Only the spatial extents shrink. Channels are pooled independently and never
mixed, because two channels are two different detectors and the largest response
of one says nothing about the other. So ``(8, 26, 26)`` pools to ``(8, 13, 13)``
and the leading 8 is carried through untouched.

The forward pass has no activation, which is worth saying explicitly for a
reader arriving from :class:`~oop_ml.core.network.layer.DenseLayer`. There is no
weighted sum to bend, so there is no pre-activation value distinct from the
answer, and :attr:`~oop_ml.core.network.layer.LayerResponse.scores` and
:attr:`~oop_ml.core.network.layer.LayerResponse.outputs` are the same block.
Not merely equal, the same array object.

The one thing two pooling layers disagree about
------------------------------------------------
Everything above is shared, and so is the geometry: the same extent arithmetic,
the same sweep, the same refusals, the same absence of parameters. What
:class:`MaxPool2d` and :class:`AveragePool2d` actually differ by is a single
function, ``d output / d input``, evaluated inside one window.

That is why :class:`Pool2d` holds the sweep and a subclass supplies only
:meth:`Pool2d.shares_of`, which answers what fraction of a window's answer each
position in it is responsible for. Written that way the whole distinction is two
lines of arithmetic and it reads as the derivative it is::

    maximum   the winner is responsible for all of it        [[0, 0], [1, 0]]
    average   every position is equally responsible          [[.25, .25], [.25, .25]]

Both sum to one, and that is not a coincidence to be checked once. Each output
is a weighted mean of its window with weights summing to one, so the shares are
those same weights, and a pooling layer whose shares did not sum to one would be
scaling the gradient on its way past. The base asserts nothing about it; the
spec does, for every subclass, which is the right place for a claim about a
family rather than about a body.

To one within float64, which is worth stating precisely because the spec has to
choose a tolerance. An average over a window side of 7 sums to
0.9999999999999999, and 16 of the first 32 sides are off by an ulp or two, worst
4.4e-16 at side 31. That is the reciprocal not being representable rather than
anything wrong, and the effect on a gradient is far below the arithmetic already
in it. The spec checks to 1e-15 and includes side 7 among its windows, so the
inexact case is actually exercised rather than avoided.

Why the backward pass of a maximum is a routing rather than an arithmetic
-------------------------------------------------------------------------
A maximum is a *selection*: over the region where the winner stays the winner,
the output simply is that input, so the derivative of the output with respect to
the winning position is exactly 1 and with respect to every other position in
the window is exactly 0. The whole arriving value goes to the position that won,
and the rest of the window gets nothing.

An average has no such asymmetry. Every position enters the answer with the same
coefficient, so every position gets the same share, and no position is ever
starved of gradient. That is the practical difference between them and it is
worth stating in those terms rather than as a preference: max pooling trains
only the winners, average pooling trains everything in the window a little.
Neither has parameters, so both report ``gradient=None`` rather than a block of
zeros, which would be a small lie about having something to learn.

Accumulate, never assign
------------------------
When the stride is smaller than the window the windows overlap, and one input
position contributes to several of them. At ``window=2, stride=1`` an interior
position belongs to four windows, so its share of the blame is the sum of four
contributions. The backward pass therefore adds into the block it is building.
Writing ``=`` instead of ``+=`` keeps only whichever window was visited last,
which produces a smaller gradient of exactly the right shape, full of plausible
finite numbers, on a layer that still trains. Nothing but a gradient check finds
it, which is why the overlapping case has a test of its own rather than riding
along with the disjoint one.

The tie, said honestly
----------------------
This applies to :class:`MaxPool2d` alone; an average has no tie to break.

When two positions in a window hold the same maximum, the derivative does not
exist. The maximum of two crossing lines has a corner at the crossing, and the
slope approaching from one side is 1 for the first position and 0 for the
second, while from the other side it is the reverse. There is no single answer,
so an implementation does not compute one, it *chooses* one.

This one chooses the first position in row-major order within the window: top
row before bottom, left before right. That is what :func:`numpy.argmax` returns
for a tie, and matching it is the point. The forward and backward passes have to
agree about who won, and the cheapest way to guarantee that is for both to ask
the same function the same question rather than for one of them to apply a rule
of its own. So a tied position that is not the first receives zero, and the
first receives the whole arriving value.

The alternative is to split the arriving value evenly between the tied
positions, which is a legitimate subgradient and is what a symmetric reading of
the corner suggests. It is not what any established implementation does, and it
buys nothing: exact ties in float64 data are essentially confined to constant
regions and to hand-built fixtures, and on a constant region every choice is as
defensible as every other. Note that a finite-difference check cannot arbitrate
this, because the one-sided differences genuinely disagree there. The tests pin
the convention directly instead, on a hand-built tie, which is the only way to
pin a decision that no oracle can settle.

Why the winner is recomputed rather than remembered
---------------------------------------------------
The forward pass knows which position won every window, and could hand a mask or
a block of indices to the backward pass.
:class:`~oop_ml.core.network.layer.LayerResponse` has room for the inputs, the
scores and the outputs, and nothing else, so carrying that would mean widening
the type every layer shares for the benefit of this one. It already carries the
block this layer read, and re-scanning it applies the same tie rule to the same
frozen numbers, so the two passes cannot disagree about a winner even in
principle.

The cost is a second scan of every window. Measured on ``(32, 8, 26, 26)`` with
``window=2, stride=2``, median of five runs, the forward pass takes 0.0692 s and
the backward pass 0.0965 s. The backward pass is doing the same scan plus an
index division and an accumulation, and the ratio of 1.39 is roughly that.
Recording the winners forward would move work rather than remove it, and it is
an optimisation to measure rather than a correction.

:class:`~oop_ml.core.network.dropout.Dropout` reaches the opposite conclusion
about the same question, and the two are worth reading together. Its mask cannot
be recomputed at all, because it was drawn at random, so it *has* to be carried
on the response. A pooling winner is a function of frozen numbers the response
already holds, so carrying it would be storing a derivable fact.

What the shared sweep costs, measured rather than waved at
-----------------------------------------------------------
Folding both layers into one sweep is not free, and the number is worth having
before deciding it was worth it. :meth:`Pool2d.shares_of` returns an array, so
the backward pass allocates one small block per window where a max-specific
version indexes a single position directly. Measured on ``(8, 4, 26, 26)`` at
``window=2, stride=2``, median of five, the backward pass costs 35.8 ms through
``shares_of`` against 16.4 ms written directly, a ratio of 2.18x; on an
overlapping ``(4, 8, 14, 14)`` at ``stride=1`` it is 2.06x. The two agree
exactly, which is what makes the comparison a comparison.

That cost is accepted here for the reason the next section gives: this whole
module is already 32x off a vectorised implementation, so a 2.18x on top of it
does not change what kind of code this is, and the optimisation pass that closes
the 32x closes this along with it. What is bought is that the difference between
the two layers is two lines of arithmetic that read as the derivatives they are,
rather than two near-identical hundred-line sweeps that have to be diffed to see
where they part.

On the loops
------------
Both passes are plain nested Python loops over rows, channels and window
positions, which is the definition written down. That is deliberate and it is
slow: the forward pass above is 0.0692 s where a reshape-and-reduce over the
same block is 0.0022 s, 32x quicker, and on that block the two agree exactly.
This library's rule is that correctness comes first and speed comes in a
separate pass that measures both ends. The reshape trick is also not general --
it holds only when the stride equals the window and the extents divide evenly --
so the replacement is a real piece of work rather than a one-line substitution,
and it has to carry the tie convention with it.
"""

from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import Sequence

import numpy as np

from oop_ml.core.exceptions import (
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray

SPATIAL_ARRANGEMENT = 3
"""How many extents one picture has here: channels, height, width.

A *block* of pictures is one dimension wider than that, since the rows come
first, which is why both entry points check against ``SPATIAL_ARRANGEMENT + 1``.
"""


def _as_whole_number(value: object, role: str) -> int:
    """Read one configured count as a whole number, in the library's own words.

    ``operator.index`` rather than ``isinstance(value, int)`` for the reason
    :mod:`oop_ml.core.network.shape` gives: an extent is routinely read off a
    numpy computation, and a ``numpy.int64`` is a whole number by every standard
    except ``isinstance``. ``bool`` is excluded ahead of it, since ``True``
    indexes as 1 and would otherwise slip through as a window of one.

    Parameters
    ----------
    value:
        The candidate count.
    role:
        What it is, for the message.

    Returns
    -------
    int
        The same number, as a builtin ``int``.

    Raises
    ------
    InvalidValuesError
        If the value is a bool, or is not a whole number at all.
    """
    if isinstance(value, bool):
        raise InvalidValuesError(f"{role} must be a whole number, not a bool")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise InvalidValuesError(f"{role} must be a whole number") from None


class Pool2d(Layer):
    """The geometry, the sweep and the refusals every pooling layer shares.

    A subclass supplies :meth:`summarise` and :meth:`shares_of` and nothing
    else. Both are functions of one window, so neither has to know about rows,
    channels, strides or blocks, and the two of them together are the whole of
    what one kind of pooling is.

    Parameters
    ----------
    reads:
        ``(channels, height, width)``, the arrangement of one input. Three
        extents exactly: this layer pools two spatial axes and carries the
        channel axis through, so a shape with any other number of extents is not
        a picture it can read.
    window:
        The side of the square window, at least 1 and no larger than either
        spatial extent.
    stride:
        How far the window moves between positions, at least 1. Smaller than
        ``window`` means the windows overlap, which is legal and is what makes
        the backward pass accumulate.

    Raises
    ------
    InvalidValuesError
        If ``window`` or ``stride`` is not a whole number, or is below 1, or if
        any extent in ``reads`` is not a whole number of at least 1. A window of
        zero selects nothing and a stride of zero never advances, so neither is
        a degenerate layer, it is an absent one.
    ShapeMismatchError
        If ``reads`` does not hold exactly three extents, or if the window is
        larger than the height or the width. A window that does not fit yields
        no positions at all, so the layer would answer with nothing.

    Notes
    -----
    The output extents are ``(extent - window) // stride + 1`` on each spatial
    axis, and the channel count is unchanged. Every term is known here, so the
    whole shape is settled at construction, which is the claim this package
    opened with.

    No pooling layer has parameters. :meth:`correction_for` answers with
    ``gradient=None`` and :meth:`stepped_by` answers with ``self``, and those
    live here rather than in each subclass because neither is a thing a kind of
    pooling gets to have an opinion about.
    """

    __slots__ = ("_shape", "_stride", "_window")

    def __init__(self, reads: Sequence[int], window: int = 2, stride: int = 2) -> None:
        # Guarded, because ``tuple(5)`` raises a bare TypeError from builtins
        # and every failure in this library is one of its own. A bare integer is
        # the natural mistake here -- it is what a dense layer's width looks
        # like -- and it deserves the message that says so rather than
        # "'int' object is not iterable".
        try:
            extents = tuple(reads)
        except TypeError:
            raise ShapeMismatchError(
                "a pooling layer reads (channels, height, width), got "
                f"{reads!r}, which is not a sequence of extents"
            ) from None
        if len(extents) != SPATIAL_ARRANGEMENT:
            raise ShapeMismatchError(
                "a pooling layer reads (channels, height, width), got "
                f"{len(extents)} extents"
            )

        channels = _as_whole_number(extents[0], "the channel count")
        height = _as_whole_number(extents[1], "the height")
        width = _as_whole_number(extents[2], "the width")
        for extent, role in (
            (channels, "channels"),
            (height, "height"),
            (width, "width"),
        ):
            if extent < 1:
                raise InvalidValuesError(
                    f"a pooling layer's {role} must be at least 1, got {extent}"
                )

        window_side = _as_whole_number(window, "the window")
        stride_length = _as_whole_number(stride, "the stride")
        if window_side < 1:
            raise InvalidValuesError(
                f"a pooling window must be at least 1 wide, got {window_side}"
            )
        if stride_length < 1:
            raise InvalidValuesError(
                f"a pooling stride must be at least 1, got {stride_length}"
            )
        if window_side > height or window_side > width:
            raise ShapeMismatchError(
                f"a window of {window_side} does not fit in a picture "
                f"{height} by {width}"
            )

        self._window = window_side
        self._stride = stride_length
        self._shape = LayerShape(
            n_inputs=(channels, height, width),
            n_outputs=(
                channels,
                (height - window_side) // stride_length + 1,
                (width - window_side) // stride_length + 1,
            ),
        )

    @property
    def shape(self) -> LayerShape:
        """``(channels, height, width)`` in, ``(channels, pooled, pooled)`` out."""
        return self._shape

    @property
    def window(self) -> int:
        """The side of the square window."""
        return self._window

    @property
    def stride(self) -> int:
        """How far the window moves between positions."""
        return self._stride

    @abstractmethod
    def summarise(self, window: FloatArray) -> float:
        """The one number this kind of pooling keeps for a window.

        Parameters
        ----------
        window:
            ``(window, window)``, a view into the block being pooled. It is a
            view rather than a copy, so it must not be written to.

        Returns
        -------
        float
            The window's answer.
        """

    @abstractmethod
    def shares_of(self, window: FloatArray) -> FloatArray:
        """What fraction of this window's answer each position is responsible for.

        The derivative of :meth:`summarise` with respect to each entry of the
        window, which for every pooling layer here is also a set of weights
        summing to one. See the module docstring for why those are the same
        thing and why the sum matters.

        Parameters
        ----------
        window:
            ``(window, window)``, the same view :meth:`summarise` was given.

        Returns
        -------
        FloatArray
            ``(window, window)``, summing to one.
        """

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """One number per window, given a block already checked.

        Parameters
        ----------
        inputs:
            ``(n_rows, channels, height, width)``, already known by
            :meth:`respond_to` to be the right arrangement, non-empty and
            finite. Nothing is re-checked here.
        purpose:
            Ignored. What a window summarises to does not depend on why it was
            asked for.

        Returns
        -------
        LayerResponse
            ``scores`` and ``outputs`` are the same array object, because there
            is no activation and therefore no pre-activation value to keep
            apart. The response is wrapped by
            :meth:`~oop_ml.core.network.layer.LayerResponse.already_checked`,
            which is sound because this method allocated the block and has
            shared it with nobody, and because the shapes here are four
            dimensional where the checking constructor expects two.

        Notes
        -----
        The definition, written as the four nested loops it is: for each row,
        for each channel, for each window position, whatever :meth:`summarise`
        says. The window is a view, so no copy is made per position.
        """
        n_rows = inputs.shape[0]
        channels, out_height, out_width = self._shape.answers

        outputs = np.empty((n_rows, channels, out_height, out_width))
        for row in range(n_rows):
            for channel in range(channels):
                for out_row in range(out_height):
                    top = out_row * self._stride
                    for out_column in range(out_width):
                        left = out_column * self._stride
                        outputs[row, channel, out_row, out_column] = self.summarise(
                            inputs[
                                row,
                                channel,
                                top : top + self._window,
                                left : left + self._window,
                            ]
                        )

        # Pooling applies no bend, so the score and the answer are one and the
        # same block. Saying that here keeps LayerResponse honest rather than
        # inventing a pre-activation that does not exist.
        return LayerResponse.already_checked(
            inputs=inputs, scores=outputs, outputs=outputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """Share each window's arriving value out over the positions that earned it.

        Parameters
        ----------
        response:
            What this layer did on the way up. Only its ``inputs`` are read,
            since the shares are recomputed from the block that was pooled
            rather than remembered.
        arriving:
            ``(n_rows, channels, pooled_height, pooled_width)``, the slope of
            the loss at this layer's outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is ``(n_rows, channels, height, width)``, and
            ``gradient`` is ``None`` because no pooling layer has anything to
            learn.

        Raises
        ------
        InvalidValuesError
            If ``arriving`` cannot be read as a float array.
        ShapeMismatchError
            If the response was not produced by a layer of this shape, or if
            ``arriving`` does not describe this layer's outputs for the rows the
            response holds. Both are pairing mistakes rather than data
            mistakes, and either would otherwise route real numbers to
            arbitrary positions.

        Notes
        -----
        One line does the work, and it is the chain rule with nothing else in
        it: this window's arriving value times each position's share of the
        answer.

        The accumulation is ``+=`` rather than ``=``, and that is the line to
        read twice. At ``stride < window`` the windows overlap and one position
        contributes to several of them, so its share is a sum. See the module
        docstring for what an assignment would silently do instead.
        """
        arriving = self._checked_arriving(response, arriving)

        n_rows = response.inputs.shape[0]
        channels, out_height, out_width = self._shape.answers

        passed_down = np.zeros_like(response.inputs)
        for row in range(n_rows):
            for channel in range(channels):
                for out_row in range(out_height):
                    top = out_row * self._stride
                    for out_column in range(out_width):
                        left = out_column * self._stride
                        window = response.inputs[
                            row,
                            channel,
                            top : top + self._window,
                            left : left + self._window,
                        ]
                        # Accumulate: with a stride below the window, one input
                        # contributes to several windows and is owed all of them.
                        passed_down[
                            row,
                            channel,
                            top : top + self._window,
                            left : left + self._window,
                        ] += float(
                            arriving[row, channel, out_row, out_column]
                        ) * self.shares_of(window)

        return LayerCorrection(passed_down=passed_down, gradient=None)

    def stepped_by(self, gradient: LayerGradient | None, learning_rate: float) -> Layer:
        """This same layer, because there is nothing in it to move.

        Parameters
        ----------
        gradient:
            Ignored. :meth:`correction_for` always answers ``None`` here, and a
            caller who has one anyway got it from somewhere else.
        learning_rate:
            Ignored, for the same reason.

        Returns
        -------
        Layer
            ``self``. Not a copy: the layer is immutable and unchanged, so
            returning it is correct rather than a shortcut, and it is what the
            base class asks a parameterless layer to do.
        """
        return self

    # Value semantics, which a dense layer deliberately does not have. A pooling
    # layer is pure configuration -- three extents, a window and a stride, all
    # of them settled at construction and none of them learned -- so two of them
    # configured alike really are interchangeable, in the way two
    # :class:`~oop_ml.core.network.shape.LayerShape` objects are.
    #
    # ``type(self) is type(other)`` rather than ``isinstance``: a max pool and
    # an average pool of identical geometry answer differently on the first
    # block either of them reads, so they are not interchangeable and must not
    # compare equal.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pool2d) or type(self) is not type(other):
            return NotImplemented
        return (
            self._shape == other._shape
            and self._window == other._window
            and self._stride == other._stride
        )

    def __hash__(self) -> int:
        return hash((type(self), self._shape, self._window, self._stride))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(reads={self._shape.reads!r}, "
            f"window={self._window!r}, stride={self._stride!r})"
        )


class MaxPool2d(Pool2d):
    """Keeps the largest value in each window of each channel.

    The classic pooling layer, and the one that makes a picture classifier
    tolerant of small shifts: the answer says a feature was found somewhere in
    this window and deliberately does not say where.

    See :class:`Pool2d` for the parameters, the refusals and the shape
    arithmetic, all of which are shared. What is specific to this class is the
    two methods below and the tie convention the module docstring sets out.
    """

    __slots__ = ()

    def summarise(self, window: FloatArray) -> float:
        """The largest value in the window."""
        return float(window.max())

    def shares_of(self, window: FloatArray) -> FloatArray:
        """One at the winning position, zero everywhere else.

        The derivative of a maximum. Over the region where the winner stays the
        winner, the answer *is* that input, so its slope is exactly 1 and every
        other position's is exactly 0.

        ``numpy.argmax`` reads the window flattened in row-major order and
        returns the first index on a tie, which is the convention the module
        docstring commits to. The forward pass asks the same function about the
        same frozen numbers, so the two cannot disagree about who won.
        """
        shares = np.zeros_like(window)
        winning_row, winning_column = np.unravel_index(
            int(np.argmax(window)), window.shape
        )
        shares[winning_row, winning_column] = 1.0
        return shares


class AveragePool2d(Pool2d):
    """Keeps the mean of each window of each channel.

    The same geometry as :class:`MaxPool2d` and a different opinion about what
    a window is worth. A maximum reports the strongest evidence in the window
    and discards the rest; a mean reports how much evidence there was on
    average, so a window with one strong response and a window with several
    moderate ones can come out alike.

    Which is better is a question about the data rather than about the layer.
    The difference that is not a matter of taste is on the way back: a maximum
    sends the whole arriving value to one position and starves the other three
    of a two-by-two window, while a mean gives every position a quarter. Every
    input in the window therefore receives gradient at every step, which is
    what makes average pooling the gentler of the two on a deep stack.

    See :class:`Pool2d` for the parameters, the refusals and the shape
    arithmetic. There is no tie to break here, since no position is preferred.
    """

    __slots__ = ()

    def summarise(self, window: FloatArray) -> float:
        """The mean of the window."""
        return float(window.mean())

    def shares_of(self, window: FloatArray) -> FloatArray:
        """The same fraction everywhere, one over the number of positions.

        The derivative of a mean. Each entry enters the answer with coefficient
        ``1 / n``, so each is responsible for exactly that much of it, and a
        two-by-two window hands each of its four positions a quarter.
        """
        return np.full_like(window, 1.0 / window.size)
