"""The sweep: a small grid of weights carried across a large picture.

The one operation underneath everything else
--------------------------------------------
Template matching slides a picture and scores the fit. An edge operator slides
three columns of weights and answers where the brightness changes. The oriented
histogram slides nothing but is built on the answers of two such operators. A
rectangle feature is a sum over a box, which is a sweep whose weights happen to
be flat. Written separately these look like four methods; written once they are
one method asked four questions, and that is what this module is.

Correlation, not convolution, and the difference matters exactly once
---------------------------------------------------------------------
:func:`swept` multiplies the weights by the pixels *in the order they are
written* and sums. Convolution flips the weights end over end first. For a
symmetric grid the two agree and nobody notices; for the gradient operators
they differ in sign, so a convolution answers that brightness rises to the left
where a correlation says it rises to the right. Signal processing says
convolution because it wants the algebra to be associative; vision says
correlation because it wants the answer to point the way the reader expects.
This package says correlation, and the sign of every gradient here is stated
against that choice rather than left for the caller to discover.

What happens at the edge, which is a real decision
--------------------------------------------------
A three-wide grid centred on the first column needs a column that does not
exist. Every option invents something:

* ``KEEP_VALID`` invents nothing and answers a smaller picture, which is
  honest and makes the answer a different size from the question.
* ``EXTEND`` repeats the edge pixel outward, so a flat border stays flat and
  an edge operator answers zero there, which is usually what is wanted.
* ``WRAP`` takes the pixel from the opposite side, which is right for a
  picture that genuinely tiles and wrong for a photograph, where it invents an
  edge between the left and right sides that nothing in the scene produced.
* ``PAD_WITH_ZERO`` treats the outside as black, which puts a bright edge
  around any picture that is not already dark at its border.

There is no neutral option, so the enum is closed and a caller has to pick
one. ``EXTEND`` is the default because its invention is the smallest: it
asserts only that the scene continues as it was, where zero-padding asserts
that the world outside the frame is black.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import numpy as np

from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.types import FloatArray


class EdgeRule(StrEnum):
    """What a sweep reads where the weights hang off the picture.

    A closed enum rather than a string, for the reason
    :class:`~oop_ml.core.distance.metric.DistanceMetric` gives: a wrong value
    should be a type error rather than a plausible answer computed from
    something nobody chose.
    """

    KEEP_VALID = "keep_valid"
    """Answer only where the weights fit, so the answer is smaller."""

    EXTEND = "extend"
    """Repeat the edge pixel outward."""

    WRAP = "wrap"
    """Take the pixel from the opposite side."""

    PAD_WITH_ZERO = "pad_with_zero"
    """Treat everything outside as zero."""


PadMode = Literal["edge", "wrap", "constant"]
"""The spellings numpy accepts, narrowed from ``str``.

Written as a literal rather than a plain string because ``np.pad``
is overloaded on this parameter, so a ``str`` matches none of its
overloads and a typo would only be caught when a picture reached it.
"""

NUMPY_PAD_MODE: dict[EdgeRule, PadMode] = {
    EdgeRule.EXTEND: "edge",
    EdgeRule.WRAP: "wrap",
    EdgeRule.PAD_WITH_ZERO: "constant",
}
"""How each rule is spelled for the padding call, written down once."""


def checked_weights(weights: FloatArray | list[list[float]]) -> FloatArray:
    """The weights of a sweep, validated.

    Raises
    ------
    InvalidValuesError
        If the weights are not a finite two-dimensional grid, or either side
        is even, which would leave the grid with no centre pixel to answer at.
    """
    grid = np.asarray(weights, dtype=np.float64)
    if grid.ndim != 2:
        raise InvalidValuesError(
            f"a sweep's weights are a two-dimensional grid, got shape {grid.shape}"
        )
    if grid.size == 0:
        raise InvalidValuesError("a sweep needs at least one weight")
    if not np.all(np.isfinite(grid)):
        raise InvalidValuesError("a sweep's weights are finite")
    if grid.shape[0] % 2 == 0 or grid.shape[1] % 2 == 0:
        raise InvalidValuesError(
            f"a sweep's weights have a centre pixel, so each side is odd; got "
            f"shape {grid.shape}. An even side leaves the answer half a pixel "
            f"away from where it was computed"
        )
    return grid


def swept(
    picture: Picture,
    weights: FloatArray | list[list[float]],
    edge_rule: EdgeRule = EdgeRule.EXTEND,
) -> Picture:
    """Carry ``weights`` across ``picture`` and answer the weighted sums.

    Correlation rather than convolution; see the module docstring for why, and
    for what each edge rule invents.

    Parameters
    ----------
    picture:
        What is swept across.
    weights:
        The grid carried, odd on both sides.
    edge_rule:
        What is read where the grid hangs off. Under ``KEEP_VALID`` the answer
        is smaller than the picture by one less than each side of the grid.

    Raises
    ------
    InvalidValuesError
        If the weights are not a valid grid.
    ShapeMismatchError
        If the grid is larger than the picture, so that no position fits.
    """
    grid = checked_weights(weights)
    height, width = grid.shape
    if height > picture.height or width > picture.width:
        raise ShapeMismatchError(
            f"a {height} by {width} grid does not fit inside a "
            f"{picture.height} by {picture.width} picture, so there is no "
            f"position to answer at"
        )

    half_height, half_width = height // 2, width // 2
    if edge_rule is EdgeRule.KEEP_VALID:
        padded = picture.values
    else:
        padded = np.pad(
            picture.values,
            ((half_height, half_height), (half_width, half_width)),
            mode=NUMPY_PAD_MODE[edge_rule],
        )

    windows = np.lib.stride_tricks.sliding_window_view(padded, (height, width))
    return Picture(np.einsum("ijkl,kl->ij", windows, grid))


def windows_of(picture: Picture, height: int, width: int) -> FloatArray:
    """Every ``height`` by ``width`` patch of ``picture``, as one array.

    Indexed ``[row, column, y, x]``, where ``row`` and ``column`` are the
    patch's top-left pixel. A view rather than a copy, so asking for every
    patch of a large picture costs nothing until something reads them.

    Raises
    ------
    InvalidValuesError
        If either side is below one.
    ShapeMismatchError
        If the patch is larger than the picture.
    """
    if height < 1 or width < 1:
        raise InvalidValuesError(
            f"a patch has at least one pixel on each side, got {height} by {width}"
        )
    if height > picture.height or width > picture.width:
        raise ShapeMismatchError(
            f"a {height} by {width} patch does not fit inside a "
            f"{picture.height} by {picture.width} picture"
        )
    return np.lib.stride_tricks.sliding_window_view(picture.values, (height, width))
