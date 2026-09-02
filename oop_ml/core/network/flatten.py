"""The join between a picture and a row, written down as a layer of its own.

Why this exists rather than a looser shape rule
------------------------------------------------
A convolutional stack answers with something arranged, say ``(8, 26, 26)``, and
a dense layer reads a row, say ``(5408,)``. Those two hold exactly the same
5408 numbers, and the cheap way to let them meet is to weaken
:meth:`~oop_ml.core.network.shape.LayerShape.follows` until it compares element
*counts* instead of extents.

That is the wrong repair, and the reason is worth keeping. Equal counts is a
far weaker statement than equal arrangement. ``(8, 26, 26)``, ``(26, 8, 26)``,
``(4, 52, 26)`` and ``(5408,)`` all pass a count comparison, and only one of
them is what the layer beneath actually produced. Three of the four are genuine
arrangement errors, of exactly the kind that surfaces as a model which trains,
converges, and quietly reads the picture sideways. Once counts alone decide,
the chain of integer equalities this package opened with stops proving that the
network fits together and proves only that it holds the right *amount* of data.

So the extents keep having to match exactly, and the one place where a row and
a picture are genuinely interchangeable gets said out loud instead. That is
this layer. :class:`Flatten` reads ``(8, 26, 26)``, answers ``(5408,)``, and
carries the conversion in its own shape, so a stack containing one still
validates by strict extent equality at every join, and a stack missing one is
refused at construction rather than discovered in a matrix multiply. The bridge
is a thing a network states it has, not a hole in the rule that would have
caught its absence.

Why it computes nothing, and why that is not a reason to skip it
-----------------------------------------------------------------
Flattening is a relabelling of the same numbers. No weight, no bias, no bend,
and therefore no gradient: :meth:`Flatten.correction_for` answers ``None``
rather than a zero-filled :class:`~oop_ml.core.network.gradient.LayerGradient`,
which would be a small lie about having something to learn, and
:meth:`Flatten.stepped_by` answers with the layer itself.

The backward pass is the forward pass run the other way and nothing more. A
relabelling's derivative is the identity: the slope of the loss at output
position ``k`` *is* the slope at whichever input position ``k`` came from, so
passing the blame down is the same reshape read in reverse. What has to hold is
that the two reshapes agree about which position is which, which is why both
use numpy's default C ordering (the last extent varying fastest, so one
channel's values stay contiguous) rather than one of them choosing Fortran
order and quietly undoing the correspondence.

Why the reshape is a view
-------------------------
Both directions use ``reshape``, which hands back a view of the same buffer
whenever the block is contiguous, and both blocks are frozen before anyone else
can see them, so nothing can write through the sharing. Copying instead would
be the single most expensive thing in a convolutional stack for no gain, since
the block being flattened is the largest intermediate the network holds.
Measured on numpy 2.5.1, the reshape against the same reshape forced to copy::

    block                  size      view         copy
    (32, 8, 13, 13)     0.35 MB    0.177 us      8.375 us       47x
    (256, 8, 13, 13)    2.77 MB    0.174 us    613.467 us     3526x
    (512, 16, 26, 26)  44.30 MB    0.209 us   9546.409 us    45677x

The view is flat in the size of the block and the copy is linear in it, which
is what makes this a decision rather than a micro-optimisation.

Where the layer does spend
--------------------------
At its own boundary, coercing the block it is handed and scanning it for
non-finite entries: at 256 by 8 by 13 by 13 that measured 479 us for the
coercion and 58 us for the scan, against 0.174 us for the reshape they guard.
The whole cost of this layer is establishing that its input is trustworthy.
That is the trade :mod:`~oop_ml.core.network.blocks` records, and it is paid
here for a sharper reason: a relabelling propagates a ``nan`` perfectly, so a
flatten that skipped the scan would launder a poisoned block into the dense
half of the network without a word.

The coercion and the scan used to be copied here rather than imported, on the
argument that two private helpers in a sibling module are not an interface.
That held while there were two copies. By the time there were three of one and
two of the other, all byte-identical, the copies had become the interface in
everything but name, and :mod:`~oop_ml.core.network.blocks` is that interface
said out loud.
"""

from __future__ import annotations

from collections.abc import Sequence

from oop_ml.core.exceptions import (
    ShapeMismatchError,
)
from oop_ml.core.network.blocks import as_block
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray


class Flatten(Layer):
    """Reads an arranged block, answers with the same numbers in one row.

    Parameters
    ----------
    reads:
        The extents of one input, arrangement included, such as
        ``(8, 13, 13)``. A bare integer is accepted for the one dimensional
        case, where flattening is the identity; the layer is still worth having
        there, because it states a join a caller would otherwise have to carry
        in their head.

    Raises
    ------
    InvalidValuesError
        If the extents are empty, or hold anything that is not a whole number
        of at least one. The refusal comes from
        :class:`~oop_ml.core.network.shape.LayerShape`, which is deliberately
        the only object in this package that reads an extent.

    Notes
    -----
    The shape is built by handing ``reads`` to ``LayerShape`` twice. The first
    construction exists only to validate the extents and hand them back as a
    tuple, and its answering side is a throwaway copy of its reading side; the
    second pairs those validated extents with the count they imply, which is
    what this layer actually answers with. Deriving that count here instead
    would mean a second implementation of "what is a valid extent", and the
    guard belongs to the one object that already owns it.
    """

    __slots__ = ("_shape",)

    def __init__(self, reads: int | Sequence[int]) -> None:
        validated_arrangement = LayerShape(n_inputs=reads, n_outputs=reads)
        self._shape = LayerShape(
            n_inputs=validated_arrangement.reads,
            n_outputs=validated_arrangement.n_inputs,
        )

    @property
    def shape(self) -> LayerShape:
        """The arrangement read, paired with the single row answered with.

        ``Flatten((8, 13, 13)).shape`` reads ``(8, 13, 13)`` and answers
        ``(1352,)``, so both of its joins are settled by exact extent equality
        like every other join in a stack.
        """
        return self._shape

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """The forward pass, given a block already known to be arranged right.

        Parameters
        ----------
        inputs:
            ``(n_rows, *shape.reads)``, validated by :meth:`respond_to`.
        purpose:
            Ignored. A reshape is a reshape.

        Returns
        -------
        LayerResponse
            ``scores`` and ``outputs`` are the *same block*, because this layer
            has no activation and so nothing separates a pre-activation value
            from a post-activation one. A backward pass that asks for the
            scores gets the outputs, which is the honest answer rather than a
            duplicated array pretending a bend happened.

        Notes
        -----
        One ``reshape`` in C order, so row ``i`` of the answer is observation
        ``i`` read with its last extent varying fastest. The result is a view
        of the inputs wherever numpy can make it one, and
        :meth:`LayerResponse.already_checked` then freezes both, so the sharing
        is unobservable.
        """
        outputs = inputs.reshape(inputs.shape[0], self._shape.n_outputs)

        return LayerResponse.already_checked(
            inputs=inputs, scores=outputs, outputs=outputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """Put the arriving slopes back into the arrangement they came from.

        Parameters
        ----------
        response:
            What this layer did on the way up. Only its row count is read,
            since a relabelling's backward step does not depend on the values
            that went through it.
        arriving:
            ``(n_rows, *shape.answers)``, the slope of the loss at this
            layer's flattened outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is ``(n_rows, *shape.reads)``, the same slopes
            returned to their positions, and ``gradient`` is ``None``, because
            this layer holds no parameters for a gradient to describe.

        Raises
        ------
        InvalidValuesError
            If the arriving block cannot be read as a float array.
        ShapeMismatchError
            If the arriving block is not one slope per row per output of this
            layer.

        Notes
        -----
        There is no arithmetic here, and that is the whole content of the
        method. Output position ``k`` of a row holds precisely the value that
        input position ``k`` of that row held, so the slope of the loss at the
        one is the slope at the other, and the backward step is the forward
        reshape read in the opposite direction. Both use C ordering, which is
        what makes them inverses rather than two unrelated rearrangements.

        The arriving block is *not* scanned for non-finite entries, where the
        forward block is. A slope that has overflowed was produced above this
        layer, and the layer that produced it is where saying so belongs; this
        one would be reporting a symptom one join late, at the price of a scan
        on every backward pass. That is the reasoning
        :class:`~oop_ml.core.network.layer.LayerResponse` already records for
        not scanning what a forward pass emits.
        """
        arriving = self._checked_arriving(response, arriving)

        arriving_block = as_block(arriving, "a flattening layer's arriving slopes")

        expected = (response.n_rows, self._shape.n_outputs)
        if arriving_block.shape != expected:
            raise ShapeMismatchError(
                f"a flattening layer is handed a {expected} block of arriving "
                f"slopes, got {arriving_block.shape}"
            )

        return LayerCorrection(
            passed_down=arriving_block.reshape(response.n_rows, *self._shape.reads),
            gradient=None,
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> Flatten:
        """This same layer, because there is nothing in it that can move.

        Parameters
        ----------
        gradient:
            ``None`` for this layer, since :meth:`correction_for` never
            produces anything else. A gradient handed here describes parameters
            this layer does not have, and moves nothing.
        learning_rate:
            How far a parameter would move. Unused, for the same reason.

        Returns
        -------
        Flatten
            ``self``. Training does not mutate anywhere in this package, and
            answering with the identical object is not a shortcut around that
            rule but the strongest available statement of it: an unchanged
            immutable layer has nothing a copy could differ in.
        """
        return self

    def __repr__(self) -> str:
        return f"Flatten(reads={self._shape.reads!r}, answers={self._shape.answers!r})"
