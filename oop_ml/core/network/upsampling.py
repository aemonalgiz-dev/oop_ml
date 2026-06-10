"""Growing a picture back by repeating each value over a square block.

What it is for
--------------
A network that answers once per picture can pool its way down and stop. A
network that answers once per *pixel*, marking which pixels belong to an object
or rebuilding the picture it was handed, has to get back up to the size it
started at after pooling took it down. This layer is the cheapest way back:
every value is copied over a ``factor`` by ``factor`` block, so ``(8, 13, 13)``
at a factor of 2 answers ``(8, 26, 26)``. Channels are carried through
untouched, as pooling carries them.

It is the mirror image of pooling, and the relation is exact rather than an
analogy. An average pool whose window and stride equal the factor undoes this
layer completely, since every block it averages holds one repeated value. The
spec asserts both directions of that, on the forward pass and the backward.

Why it has nothing to learn
---------------------------
Repetition has no weights and no bend, so :meth:`NearestUpsample2d.correction_for`
answers ``gradient=None`` and :meth:`NearestUpsample2d.stepped_by` answers with
the layer itself, as pooling and flattening do. ``scores`` and ``outputs`` are
the same block, because there is no pre-activation value to keep apart.

The backward pass is a sum
--------------------------
Every input value was copied to ``factor ** 2`` output positions, so it reaches
the loss by that many routes, and its slope is the total of the slopes arriving
at all of them. That is the transpose of repetition, and it is the exact
derivative: the layer is linear, so the derivative is the same matrix wherever
it is taken, and the backward pass does not read the block that went up at
all. Arranged as ``(rows, channels, height, factor, width, factor)`` the
arriving block sums over the two ``factor`` axes in one reduction.

The spec checks that three ways which owe nothing to the implementation: a plain
loop written from the definition, a finite-difference check, and the adjoint
identity ``<up(x), y> == <x, up_transpose(y)>``, which is what makes a backward
pass correct for a linear layer.

Why repetition rather than a transposed convolution
---------------------------------------------------
A transposed convolution is the learned way back up, and at a stride below its
kernel size neighbouring kernel placements overlap. Whether that overlap is
even depends on whether the stride divides the kernel size. Measured on a four
by four picture of ones and a kernel of ones, which counts how many placements
land on each output position: a kernel of 3 at a stride of 2 gives interior
counts of 1, 2 and 4 in a repeating pattern, a checkerboard the layer paints
before it has learned anything and whose weights would have to learn it away.
A kernel of 4 at a stride of 2 gives 4 everywhere away from the border, and a
kernel of 2 gives 1 everywhere. So the artefact is about divisibility, and it
is avoidable by choosing the kernel; what this layer offers instead is that
there is no overlap to choose about. Repeating the same picture of ones gives
ones everywhere, at any factor. The spec pins all four of those counts.

That is all that is claimed. Whether a network trained with one or the other
produces visibly better pictures is a question about training, and nothing
here has trained one.

Why the answer is a copy and not a broadcast view
-------------------------------------------------
Repetition can be written as a broadcast view in which every entry of a block
is the same memory location, which costs nothing to build. The answer is
frozen, so nothing could write through it, but the next layer's coercion would
copy it anyway, and a view whose strides are zero is a surprise to hand anyone.
So the forward pass makes one contiguous copy. Measured, median of five, on
``(32, 8, 13, 13)`` at a factor of 2: forward 2.0 ms and backward 3.0 ms,
against 76 ms and 104 ms for the plain loops the spec uses as oracles.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray

PICTURE_EXTENTS = 3
"""How many extents one picture has here: channels, height, width."""


class NearestUpsample2d(Layer):
    """Repeats every value over a ``factor`` by ``factor`` block of each channel.

    Parameters
    ----------
    reads:
        ``(channels, height, width)``, the arrangement of one input. Three
        extents exactly, since this layer grows two spatial axes and carries
        the channel axis through.
    factor:
        How many times each spatial axis grows, a whole number of at least 1.
        A factor of 1 answers what it read.

    Raises
    ------
    ShapeMismatchError
        If ``reads`` is not a sequence of extents at all, a bare width being
        the natural mistake, or does not hold exactly three.
    InvalidValuesError
        If an extent is not a whole number of at least 1, or ``factor`` is not
        a whole number of at least 1. The extents are read by
        :class:`~oop_ml.core.network.shape.LayerShape`, which is the one object
        in this package that reads an extent.

    Notes
    -----
    The answer is ``(channels, height * factor, width * factor)``, and every
    term is known here, so the whole shape is settled at construction.
    """

    __slots__ = ("_factor", "_shape")

    def __init__(self, reads: Sequence[int], factor: int = 2) -> None:
        # Guarded, because ``tuple(5)`` raises a bare TypeError from builtins,
        # and every failure in this library is one of its own. The pooling
        # layers once let exactly this escape.
        try:
            extents: tuple[int, ...] = tuple(reads)
        except TypeError:
            raise ShapeMismatchError(
                "an upsampling layer reads (channels, height, width), got "
                f"{reads!r}, which is not a sequence of extents"
            ) from None
        if len(extents) != PICTURE_EXTENTS:
            raise ShapeMismatchError(
                "an upsampling layer reads (channels, height, width), got "
                f"{len(extents)} extents"
            )

        channels, height, width = LayerShape(n_inputs=extents, n_outputs=extents).reads
        growth = self._whole_factor(factor)

        self._factor = growth
        self._shape = LayerShape(
            n_inputs=(channels, height, width),
            n_outputs=(channels, height * growth, width * growth),
        )

    @staticmethod
    def _whole_factor(factor: object) -> int:
        """Read the factor, refusing a bool, a fraction, or one below 1.

        Raises
        ------
        InvalidValuesError
            If ``factor`` is not a whole number of at least 1.
        """
        if isinstance(factor, bool):
            raise InvalidValuesError(
                "an upsampling factor is a whole number, not a bool"
            )
        try:
            whole = operator.index(factor)  # type: ignore[arg-type]
        except TypeError:
            raise InvalidValuesError(
                f"an upsampling factor is a whole number, got {factor!r}"
            ) from None
        if whole < 1:
            raise InvalidValuesError(f"an upsampling factor is at least 1, got {whole}")
        return whole

    @property
    def shape(self) -> LayerShape:
        """What it reads, and the same with each spatial extent times the factor."""
        return self._shape

    @property
    def factor(self) -> int:
        """How many times each spatial axis grows."""
        return self._factor

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """Every value copied over its block, given a block already checked.

        Parameters
        ----------
        inputs:
            ``(n_rows, channels, height, width)``, already known by
            :meth:`respond_to` to be the right arrangement, non-empty and
            finite.
        purpose:
            Ignored. Repetition is the same whatever the pass is for.

        Returns
        -------
        LayerResponse
            ``scores`` and ``outputs`` are the same array object, since there
            is no activation. Wrapped by
            :meth:`~oop_ml.core.network.layer.LayerResponse.already_checked`,
            which is sound because the block was allocated here and shared with
            nobody.
        """
        n_rows, channels, height, width = inputs.shape
        growth = self._factor

        repeated = np.broadcast_to(
            inputs[:, :, :, np.newaxis, :, np.newaxis],
            (n_rows, channels, height, growth, width, growth),
        )
        outputs = np.ascontiguousarray(
            repeated.reshape(n_rows, channels, height * growth, width * growth),
            dtype=np.float64,
        )

        return LayerResponse.already_checked(
            inputs=inputs, scores=outputs, outputs=outputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """Hand each input the total of the slopes arriving at its copies.

        Parameters
        ----------
        response:
            What this layer did on the way up. Only its shape is read, since a
            linear layer's derivative does not depend on what went through it.
        arriving:
            ``(n_rows, channels, height * factor, width * factor)``, the slope
            of the loss at this layer's outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is ``(n_rows, channels, height, width)``, and
            ``gradient`` is ``None`` because nothing here is learned.

        Raises
        ------
        InvalidValuesError
            If ``arriving`` cannot be read as a float array.
        ShapeMismatchError
            If the response came from a layer of another shape, or ``arriving``
            does not describe this layer's outputs for the rows the response
            holds.
        """
        block = self._checked_arriving(response, arriving)

        n_rows, channels, _, _ = block.shape
        _, height, width = self._shape.reads
        growth = self._factor

        passed_down = block.reshape(
            n_rows, channels, height, growth, width, growth
        ).sum(axis=(3, 5))

        return LayerCorrection(passed_down=passed_down, gradient=None)

    def stepped_by(self, gradient: LayerGradient | None, learning_rate: float) -> Layer:
        """This same layer, because there is nothing in it to move.

        Parameters
        ----------
        gradient:
            Ignored. :meth:`correction_for` always answers ``None`` here.
        learning_rate:
            Ignored, for the same reason.

        Returns
        -------
        Layer
            ``self``, which is correct rather than a shortcut: the layer is
            immutable and unchanged.
        """
        return self

    # Value semantics, as a pooling layer has: pure configuration, settled at
    # construction and never learned, so two configured alike are
    # interchangeable. ``type(self) is type(other)`` for the reason Pool2d
    # gives, so a subclass that answered differently could not compare equal.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NearestUpsample2d) or type(self) is not type(other):
            return NotImplemented
        return self._shape == other._shape and self._factor == other._factor

    def __hash__(self) -> int:
        return hash((type(self), self._shape, self._factor))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(reads={self._shape.reads!r}, "
            f"factor={self._factor!r})"
        )
