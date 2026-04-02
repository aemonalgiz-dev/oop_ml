"""How wide a layer is on each side, as one object rather than loose numbers.

Why a shape is a tuple and not a number
----------------------------------------
A dense layer reads a row of ``n`` numbers, so one integer describes it. A
convolutional layer reads a picture: so many channels, so many rows, so many
columns, and collapsing that to a single count throws away the arrangement the
layer is entirely about. So a shape here is a *tuple of extents* per side, and
a dense layer is simply the one-dimensional case::

    dense        (784,)        -> (3,)
    convolution  (1, 28, 28)   -> (8, 26, 26)
    pooling      (8, 26, 26)   -> (8, 13, 13)
    flatten      (8, 13, 13)   -> (1352,)

``n_inputs`` and ``n_outputs`` remain, as the *element counts*: the product of
each side's extents. For a dense layer those are the extents themselves, which
is why nothing about the one-dimensional case had to change, and why
``LayerShape(4, 8)`` still means what it always did.

Why this is a pairing and not two loose values
-----------------------------------------------
What a layer reads and what it answers are not interchangeable, and code that
carries them separately eventually passes them in the wrong order. That is the
tuple this library does not write: a pair whose meaning depends on position is
an object nobody has got round to naming.

Naming it buys more than safety. "Do these two layers fit together" is a
sentence about two shapes, so it belongs to the shape rather than to whichever
loop is assembling a network, and :meth:`LayerShape.follows` is that sentence.

Why this matters more once convolution exists
----------------------------------------------
A dense layer's output width is a number somebody chose. A convolution's is
*computed*, from the kernel size, the stride and the padding::

    out = (extent - kernel + 2 * padding) // stride + 1

Get any of those wrong and the mistake surfaces as a shape error somewhere
downstream, often several layers later, in the middle of a training run. Every
term is known at construction, so the answer is too, and the whole chain can be
settled before a single row is read. That is the claim this package opened
with, and convolution is where it stops being a nicety.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from math import prod

from oop_ml.core.exceptions import InvalidValuesError


def _as_extents(value: int | Sequence[int], role: str) -> tuple[int, ...]:
    """Read one side of a shape as a tuple of extents.

    A bare integer is the one-dimensional case, so ``LayerShape(4, 8)`` keeps
    working and keeps meaning what it did.

    Any whole number is accepted, not only a builtin ``int``. Extents are
    routinely read off a numpy computation, and a ``numpy.int64`` is an extent
    by every standard except ``isinstance``, so the guard is ``operator.index``.
    ``bool`` is excluded ahead of it, since ``True`` indexes as 1 and would
    otherwise slip through as an extent of one by accident.
    """
    candidates: tuple[object, ...] = (
        tuple(value) if isinstance(value, (list, tuple)) else (value,)
    )

    if not candidates:
        raise InvalidValuesError(f"{role} needs at least one extent")

    extents: list[int] = []
    for extent in candidates:
        if isinstance(extent, bool):
            raise InvalidValuesError(f"{role} must be whole numbers, not bools")
        try:
            whole = operator.index(extent)  # type: ignore[arg-type]
        except TypeError:
            raise InvalidValuesError(f"{role} must be whole numbers") from None
        if whole < 1:
            raise InvalidValuesError(f"{role} extents must be at least 1, got {whole}")
        extents.append(whole)

    return tuple(extents)


class LayerShape:
    """What a layer reads and what it answers with.

    Parameters
    ----------
    n_inputs:
        The extents of one input. An integer for a row of numbers, a tuple for
        anything arranged, such as ``(channels, height, width)``. The name is
        kept from the one-dimensional era, where the extent and the count were
        the same number.
    n_outputs:
        The extents of one output, in the same form.

    Raises
    ------
    InvalidValuesError
        If either side is empty, or holds anything that is not a whole number
        of at least one. An extent of zero is not a degenerate layer, it is an
        absent one, and it would make every downstream shape agree with it
        vacuously.
    """

    __slots__ = ("_answers", "_reads")

    def __init__(
        self, n_inputs: int | Sequence[int], n_outputs: int | Sequence[int]
    ) -> None:
        self._reads = _as_extents(n_inputs, "n_inputs")
        self._answers = _as_extents(n_outputs, "n_outputs")

    @property
    def reads(self) -> tuple[int, ...]:
        """The extents of one input, arrangement included."""
        return self._reads

    @property
    def answers(self) -> tuple[int, ...]:
        """The extents of one output, arrangement included."""
        return self._answers

    @property
    def n_inputs(self) -> int:
        """How many numbers one input holds, whatever their arrangement."""
        return prod(self._reads)

    @property
    def n_outputs(self) -> int:
        """How many numbers one output holds, whatever their arrangement."""
        return prod(self._answers)

    def follows(self, previous: LayerShape) -> bool:
        """Whether a layer of this shape can read what ``previous`` answers.

        The whole of a network's shape agreement, asked one join at a time.

        The extents must match exactly, not merely the counts. A layer
        answering ``(8, 26, 26)`` and one reading ``(5408,)`` hold the same
        number of values and still do not join, because a convolution cannot
        read a row and a dense layer cannot read a picture. Bridging those is
        what a flattening layer is for, and letting the counts alone decide
        would let an arrangement error pass as an agreement.

        Parameters
        ----------
        previous:
            The shape of the layer immediately beneath this one.

        Returns
        -------
        bool
            True when ``previous`` answers with exactly what this shape reads.
        """
        return previous.answers == self._reads

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LayerShape):
            return NotImplemented
        return self._reads == other._reads and self._answers == other._answers

    def __hash__(self) -> int:
        return hash((self._reads, self._answers))

    def __repr__(self) -> str:
        return f"LayerShape(n_inputs={self._reads!r}, n_outputs={self._answers!r})"
