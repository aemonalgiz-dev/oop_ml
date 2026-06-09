"""A picture, and the one place raw pixels are checked.

Why a type rather than an array
-------------------------------
Every method in this package sweeps something small across something large,
and every one of them has the same three ways to be handed nonsense: an array
that is not two-dimensional, an array holding a value that is not a number,
and an array with no pixels in it. Written as guards they would be repeated in
each method and forgotten in one of them. Written as a type they are
unrepresentable, which is the argument
:class:`~oop_ml.core.data.column.Column` already makes for a column of
observations.

The invariant is deliberately narrow. A picture here is two-dimensional and
finite, and that is all. It is *not* required to lie in any particular range,
because the range is a convention rather than a fact: a photograph read from a
file arrives in 0 to 255, the same photograph divided through arrives in 0 to
1, and a gradient of either is signed and lies in neither. A type that insisted
on one of those would refuse the output of half this package.

Greyscale, and why that is not a simplification here
----------------------------------------------------
A picture is one number per position rather than three. That is not laziness
about colour; it is what every method in this package actually reads. Template
matching, the gradient operators, the oriented histogram and the rectangle
features are all defined on brightness, and the usual first step of a colour
implementation is to collapse the channels into brightness anyway. Colour
belongs where it changes an answer, which is a different set of methods.
:meth:`Picture.from_channels` does the collapsing, with the weights written
down, so a caller with a colour array has one obvious way in.

What is frozen and why
----------------------
The buffer is copied and then frozen, and the copy matters as much as the
freeze: without it we would be write-protecting an array the caller still
holds, so their next write would raise from a line that looks unrelated. Same
rule as everywhere else here, pinned by the encapsulation spec.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.types import FloatArray, array_for_protocol

BRIGHTNESS_WEIGHTS: tuple[float, float, float] = (0.2126, 0.7152, 0.0722)
"""How much each of red, green and blue contributes to brightness.

The luminance weights of Rec. 709, which is the standard the sRGB pictures
almost everything produces are written against. They are not equal because the
eye is not equally sensitive to the three: green carries most of the apparent
brightness and blue almost none, so an unweighted mean of the channels makes
blue regions look lighter than they are and green ones darker.
"""


class Picture:
    """An immutable, finite, two-dimensional ``float64`` grid of brightness.

    Parameters
    ----------
    values:
        The pixels, row by row from the top. Coerced to a finite
        two-dimensional ``float64`` array, then copied, then frozen.

    Raises
    ------
    InvalidValuesError
        If the values are not coercible, not two-dimensional, or not finite.
    EmptyValuesError
        If either side has no extent.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Any) -> None:
        try:
            grid = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError(
                f"a picture is a grid of numbers, and these could not be read as "
                f"one: {error}"
            ) from error

        if grid.ndim != 2:
            raise InvalidValuesError(
                f"a picture has a height and a width, so its values are "
                f"two-dimensional; got {grid.ndim} dimension(s) with shape "
                f"{grid.shape}"
            )
        if grid.size == 0:
            raise EmptyValuesError(
                f"a picture needs at least one pixel, got shape {grid.shape}"
            )
        if not np.all(np.isfinite(grid)):
            raise InvalidValuesError(
                "a picture holds finite brightness values, and these hold a "
                "missing or infinite one"
            )

        self._values = grid.copy()
        self._values.setflags(write=False)

    @classmethod
    def from_channels(cls, channels: Any) -> Picture:
        """Collapse a ``(height, width, 3)`` colour array into brightness.

        Weighted by :data:`BRIGHTNESS_WEIGHTS` rather than averaged, because
        the three channels do not contribute equally to how light a pixel
        looks.

        Raises
        ------
        InvalidValuesError
            If the array is not three-dimensional with exactly three channels.
        """
        grid = np.asarray(channels, dtype=np.float64)
        if grid.ndim != 3 or grid.shape[2] != 3:
            raise InvalidValuesError(
                f"a colour picture is (height, width, 3); got shape {grid.shape}"
            )
        return cls(grid @ np.asarray(BRIGHTNESS_WEIGHTS, dtype=np.float64))

    @property
    def values(self) -> FloatArray:
        """The pixels, frozen. Copy it before writing to it."""
        return self._values

    @property
    def height(self) -> int:
        """How many rows of pixels."""
        return int(self._values.shape[0])

    @property
    def width(self) -> int:
        """How many columns of pixels."""
        return int(self._values.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)``."""
        return (self.height, self.width)

    @property
    def n_pixels(self) -> int:
        """How many pixels in all."""
        return int(self._values.size)

    @property
    def brightest(self) -> float:
        """The largest value present."""
        return float(self._values.max())

    @property
    def darkest(self) -> float:
        """The smallest value present."""
        return float(self._values.min())

    def patch_at(self, row: int, column: int, height: int, width: int) -> Picture:
        """The rectangle whose top-left pixel is ``(row, column)``.

        Raises
        ------
        InvalidValuesError
            If the rectangle is not positive, or does not fit inside.
        """
        if height < 1 or width < 1:
            raise InvalidValuesError(
                f"a patch has at least one pixel on each side, got {height} by {width}"
            )
        if row < 0 or column < 0:
            raise InvalidValuesError(
                f"a patch starts inside the picture, got row {row} and column {column}"
            )
        if row + height > self.height or column + width > self.width:
            raise InvalidValuesError(
                f"a {height} by {width} patch at row {row}, column {column} runs "
                f"past a {self.height} by {self.width} picture"
            )
        return Picture(self._values[row : row + height, column : column + width])

    def rescaled_to_unit(self) -> Picture:
        """The same picture shifted and stretched to run from zero to one.

        A picture with no variation at all cannot be stretched, and comes back
        as zeros rather than as a division by nothing.
        """
        span = self.brightest - self.darkest
        if span == 0.0:
            return Picture(np.zeros_like(self._values))
        return Picture((self._values - self.darkest) / span)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._values, dtype, copy)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Picture):
            return NotImplemented
        return bool(np.array_equal(self._values, other._values))

    def __hash__(self) -> int:
        return hash((self.shape, self._values.tobytes()))

    def __repr__(self) -> str:
        return f"Picture(height={self.height}, width={self.width})"
