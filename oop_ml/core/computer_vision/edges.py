"""Where the brightness changes, and which way it is changing.

Why an edge is the thing classical vision was built on
------------------------------------------------------
Raw brightness is almost useless for recognising anything, because it moves
with the lighting. Turn the lamp up and every pixel of a scene changes; the
*differences* between neighbouring pixels scale but their pattern does not,
and their directions do not move at all. So a description built from edges
survives a change in lighting that a description built from pixels does not,
and that single fact is why the next three decades of the subject were spent
on gradients rather than on brightness.

A derivative of something sampled
---------------------------------
Brightness is only known at whole pixels, so its rate of change has to be
estimated. The cheapest estimate is the difference between the two neighbours,
``(right - left) / 2``, which is :attr:`GradientOperator.CENTRAL_DIFFERENCE`.
It is exact for a straight ramp and hopeless on a real photograph, because a
single stray pixel moves it as much as a real edge does.

The named operators all repair that the same way, by averaging across the
edge while differencing along it. They differ only in the weights of that
average:

* :attr:`GradientOperator.PREWITT` averages the three rows equally.
* :attr:`GradientOperator.SOBEL` weights the centre row twice the others,
  which is a small smoothing and is why it is the one everybody reaches for.
* :attr:`GradientOperator.SCHARR` uses 3 and 10, chosen by Scharr so that the
  *direction* the operator reports is as close to rotation-independent as a
  three-wide grid allows. Sobel's directions drift by around a degree on a
  diagonal edge and Scharr's by far less, which is worth having when the
  direction is the thing being histogrammed later.

A closed enum rather than a class per operator, for the reason
:class:`~oop_ml.core.distance.metric.DistanceMetric` gives: none of them takes
a parameter, so there is nothing for an object to hold.

Two answers, not one
--------------------
Sweeping twice gives a horizontal rate of change and a vertical one, and
neither alone is the edge. Together they are a vector per pixel, whose length
says how sharply the brightness is changing and whose angle says which way.
:class:`GradientField` is that pair kept together, because handing back two
pictures and letting the caller pair them is exactly the tuple-of-different-
things this library refuses everywhere else.

The direction is perpendicular to the edge
------------------------------------------
This catches people out and so is stated rather than assumed. The gradient
points the way brightness *increases*, which is across the edge, not along it.
A vertical edge with a bright right side has a gradient pointing right, at
zero radians. The edge itself runs up and down. :attr:`GradientField.direction`
is the gradient's own angle; the edge's angle is that turned by a quarter
circle, and :attr:`GradientField.edge_direction` says so rather than leaving
the reader to work it out.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from oop_ml.core.computer_vision.filtering import EdgeRule, swept
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.types import FloatArray

QUARTER_CIRCLE = np.pi / 2
"""A right angle in radians, for turning a gradient into the edge it crosses."""


class GradientOperator(StrEnum):
    """Which estimate of the rate of change of brightness to sweep."""

    CENTRAL_DIFFERENCE = "central_difference"
    """The bare difference of the two neighbours. No smoothing at all."""

    PREWITT = "prewitt"
    """Difference across, equal average along."""

    SOBEL = "sobel"
    """Difference across, centre-weighted average along. The usual choice."""

    SCHARR = "scharr"
    """Sobel's shape with weights chosen so the reported angle is steadier."""


HORIZONTAL_WEIGHTS: dict[GradientOperator, list[list[float]]] = {
    GradientOperator.CENTRAL_DIFFERENCE: [[-0.5, 0.0, 0.5]],
    GradientOperator.PREWITT: [
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ],
    GradientOperator.SOBEL: [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ],
    GradientOperator.SCHARR: [
        [-3.0, 0.0, 3.0],
        [-10.0, 0.0, 10.0],
        [-3.0, 0.0, 3.0],
    ],
}
"""The grid that answers how fast brightness rises to the right.

Positive on the right, so a bright region to the right of a dark one gives a
positive answer, which is the correlation convention
:mod:`oop_ml.core.computer_vision.filtering` commits to.
"""


def weights_of(operator: GradientOperator, vertical: bool) -> list[list[float]]:
    """One operator's grid, for the horizontal or the vertical direction.

    The vertical grid is the horizontal one turned on its side, which is not a
    coincidence: the two directions are the same estimate asked about the other
    axis, so writing the second by hand would be a chance to write it wrong.
    """
    horizontal = HORIZONTAL_WEIGHTS[operator]
    if not vertical:
        return horizontal
    return [list(row) for row in np.asarray(horizontal, dtype=np.float64).T]


class GradientField:
    """The rate of change of brightness at every pixel, in both directions.

    Parameters
    ----------
    horizontal:
        How fast brightness rises to the right.
    vertical:
        How fast brightness rises downward.

    Raises
    ------
    ShapeMismatchError
        If the two do not describe the same picture.
    """

    __slots__ = ("_horizontal", "_vertical")

    def __init__(self, horizontal: Picture, vertical: Picture) -> None:
        if horizontal.shape != vertical.shape:
            raise ShapeMismatchError(
                f"the two directions of one gradient field describe the same "
                f"picture, got {horizontal.shape} across and {vertical.shape} down"
            )
        self._horizontal = horizontal
        self._vertical = vertical

    @classmethod
    def of(
        cls,
        picture: Picture,
        operator: GradientOperator = GradientOperator.SOBEL,
        edge_rule: EdgeRule = EdgeRule.EXTEND,
    ) -> GradientField:
        """Sweep ``operator`` across ``picture`` in both directions."""
        return cls(
            swept(picture, weights_of(operator, vertical=False), edge_rule),
            swept(picture, weights_of(operator, vertical=True), edge_rule),
        )

    @property
    def horizontal(self) -> Picture:
        """How fast brightness rises to the right, per pixel."""
        return self._horizontal

    @property
    def vertical(self) -> Picture:
        """How fast brightness rises downward, per pixel."""
        return self._vertical

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the picture this describes."""
        return self._horizontal.shape

    @property
    def magnitude(self) -> Picture:
        """How sharply the brightness is changing, per pixel.

        The length of the gradient vector, so it is never negative and is zero
        exactly where the brightness is flat in both directions.
        """
        return Picture(np.hypot(self._horizontal.values, self._vertical.values))

    @property
    def direction(self) -> Picture:
        """Which way the brightness *rises*, in radians, per pixel.

        Measured from pointing right, turning towards pointing down, and lying
        in ``(-pi, pi]``. Where the brightness is flat the answer is zero,
        which is a convention rather than a fact, since a flat region has no
        direction at all.
        """
        return Picture(np.arctan2(self._vertical.values, self._horizontal.values))

    @property
    def edge_direction(self) -> Picture:
        """Which way the *edge* runs, which is across the gradient.

        The gradient turned by a right angle, folded back into ``(-pi, pi]``.
        Exposed because confusing the two is the commonest mistake made with
        this object, and a reader should not have to remember which one they
        were handed.
        """
        turned = self.direction.values + QUARTER_CIRCLE
        return Picture((turned + np.pi) % (2 * np.pi) - np.pi)

    def strongest_at_least(self, threshold: float) -> Picture:
        """One where the change is at least ``threshold`` sharp, zero elsewhere.

        The crudest possible edge decision, kept because it is the one every
        reader tries first and because what it gets wrong is instructive: a
        single threshold either keeps the noise in the flat regions or loses
        the faint half of a real edge, and no value does neither.

        Raises
        ------
        InvalidValuesError
            If the threshold is negative, which no magnitude can be below.
        """
        if threshold < 0.0:
            raise InvalidValuesError(
                f"a magnitude is never negative, so a threshold below zero keeps "
                f"every pixel; got {threshold}"
            )
        return Picture((self.magnitude.values >= threshold).astype(np.float64))

    def as_vectors(self) -> FloatArray:
        """The field as ``(height, width, 2)``, across then down.

        For a caller that wants the pair per pixel rather than two pictures.
        """
        return np.stack([self._horizontal.values, self._vertical.values], axis=-1)

    def __eq__(self, other: object) -> bool:
        # Both halves of the guard, which is the house idiom rather than
        # belt and braces. The exact-type check is what keeps a subclass
        # carrying one more thing from comparing equal to a plain field, the
        # rule ``LayerResponse`` had to learn; the isinstance is what narrows
        # ``object`` so the attribute reads below are checkable.
        if not isinstance(other, GradientField) or type(self) is not type(other):
            return NotImplemented
        return (
            self._horizontal == other._horizontal and self._vertical == other._vertical
        )

    def __hash__(self) -> int:
        return hash((self._horizontal, self._vertical))

    def __repr__(self) -> str:
        height, width = self.shape
        return f"GradientField(height={height}, width={width})"
