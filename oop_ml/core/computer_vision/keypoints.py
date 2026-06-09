"""The few places worth describing, and what surrounds each of them.

Describing less of the picture, on purpose
------------------------------------------
Everything before this module answers something at every pixel. That is the
right shape for a question about the picture as a whole, and the wrong shape
for the question two photographs of one scene actually pose: which position in
this one is which position in that one. Answering it pixel by pixel is both
enormous and hopeless, because most pixels are indistinguishable from their
neighbours and nothing can be said about where they went.

So the method changes. Find the few positions that *can* be located, describe
what surrounds each of them, and match the descriptions. A picture stops being
a grid of brightness and becomes a short list of places, which is small enough
to compare against every other picture's list.

Why a corner, which is the whole argument
------------------------------------------
The test of whether a position can be located is whether moving a window away
from it changes what the window sees. Three cases, and they are the family:

* On flat ground, moving the window in any direction changes nothing. The
  position is unlocatable in both directions.
* On an edge, moving the window *across* the edge changes everything and
  moving it *along* the edge changes nothing. So an edge can be located in one
  direction and not in the other, and a patch cut from the middle of one is
  identical to a patch cut from anywhere else along its length. This is the
  aperture problem, and it is measured rather than asserted in the spec: on the
  left side of a nine-wide square, a five-wide patch centred on the middle of
  that edge is at distance **exactly 0.0** from the patch at five different
  positions along it, while the patch centred on the corner is at distance 0.0
  from exactly one position, its own.
* On a corner, moving the window in either direction changes what it sees. The
  position is pinned, which is the property being hunted.

That is a statement about the *spread of gradient directions* inside a window,
and the object that carries it is the structure tensor.

The structure tensor, and its two numbers
------------------------------------------
Take the gradient field the edges module already answers, form the three
products ``horizontal^2``, ``vertical^2`` and ``horizontal * vertical``, and
average each over a window. That gives a two-by-two symmetric matrix per pixel,
:class:`StructureTensor`, whose eigenvalues say how much the brightness changes
along the two directions in which it changes most and least. Flat ground gives
two small eigenvalues, an edge gives one large and one zero, a corner gives two
large ones.

The eigenvalues are never computed as eigenvalues. A two-by-two symmetric
matrix is settled entirely by its determinant and its trace -- the eigenvalues
are ``(trace +- sqrt(trace^2 - 4 determinant)) / 2`` -- so both measures here
read exactly those two pictures and nothing else.

Two scores from the same two numbers
-------------------------------------
:class:`SmallestEigenvalueMeasure` answers the smaller eigenvalue directly,
which is Shi and Tomasi's rule. :class:`HarrisMeasure` answers
``determinant - sensitivity * trace^2``, which is Harris and Stephens'
approximation to the same idea, invented to avoid the square root when a square
root was expensive.

The smaller eigenvalue is the default, for two reasons that are facts rather
than taste. It takes no parameter, where Harris takes a sensitivity nobody has
a principled value for -- the literature says "between 0.04 and 0.06" and that
is the whole of the guidance. And it is in the units of a squared gradient, so
a threshold on it means the same thing on two pictures, where Harris' response
is a squared gradient *to the fourth power* and its numbers do not transfer.

Harris is kept because it is what the world calls this, and because its failure
is instructive: at a sensitivity of a quarter or above the response cannot be
positive anywhere, since ``determinant <= trace^2 / 4`` holds for every real
symmetric matrix. :class:`HarrisMeasure` refuses that value at construction
rather than answering a picture with no corners in it.

One corner, one keypoint
-------------------------
A corner is a few pixels across, so the response around it is a blob and the
naive answer is a dozen keypoints where the scene has one. :meth:`
CornerResponse.peaks` takes the strongest first and refuses any later candidate
within ``separation`` pixels of one already taken. Because it works downward
from the strongest, everything it keeps is automatically the strongest within
that radius, so the local-maximum test comes free rather than being a second
pass.

What a descriptor is for, and what this one is not
---------------------------------------------------
A keypoint's position is not enough to match on, since two pictures of one
scene put the same corner in different places -- that is the thing being
solved. What travels is what *surrounds* the corner, so
:class:`PatchDescriptor` is the square of brightness around it, with its mean
removed and its length divided out. Those two steps buy real invariance and
the spec pins both: adding ten to every pixel of the scene moves the
descriptor by 1.2e-15 and multiplying every pixel by three moves it by
2.0e-16, which is rounding rather than a change, so turning the lamp up does
not break a match.

What it does **not** buy is said plainly here rather than implied. This
descriptor is not scale invariant and it is not rotation invariant. There is
no scale space, no orientation assignment, and no interpolation, which are the
three things that separate this from SIFT. Measured, a quarter turn of the
picture -- the one rotation that is exact and needs no interpolation -- moves
every keypoint to exactly where it should be, with strengths that agree to the
last bit, and leaves the corresponding descriptors **1.443376** apart on a
scale whose maximum is 2.0. That is the same distance a square's top-left
corner sits from its own top-right corner, which is the honest reading: after a
quarter turn the descriptor does not recognise a corner as itself, it merely
recognises it as some other corner. The keypoints are a property of the scene;
the descriptions are a property of the frame.

The ratio test, and why a distance alone is not enough
-------------------------------------------------------
A nearest neighbour always exists. Asking whether the nearest descriptor is
*near* answers with a number that depends on the picture's contrast, its noise
and its patch size, and no threshold on it transfers. Lowe's ratio test asks a
question that needs no units instead: is the nearest much nearer than the
second nearest? A real match is distinctive, so its runner-up is far behind; a
coincidence is one of many equally good answers, so its runner-up is right
behind it. Measured on a square photographed twice, a genuine match wins at a
distance of 0.0 against a runner-up at 1.443376, so its ratio is **0.0**; the
same corner matched into a scene holding two identical squares wins at 0.0
against a runner-up at 0.0, a ratio of **1.0**, and with a faint texture over
that scene to break the exact tie the two numbers become 0.050116 and 0.051100,
a ratio of **0.9807**. At Lowe's threshold of 0.8 the first is kept and both of
the others are refused, which is the method declining to answer rather than
answering one of two indistinguishable corners at random.

Where this stops working
-------------------------
On a photograph taken from further away, because there is no scale space. On a
photograph taken with the camera tilted, because there is no orientation. On
anything with repeated structure -- a brick wall, a window grid, a keyboard --
where the ratio test correctly refuses every match and the method answers
nothing rather than answering wrongly, which is the better of the two failures
but is still a failure. And on a scene with no corners at all: a picture of a
horizon has one long edge and this module will find nothing on it, correctly.

What surprised the measuring
-----------------------------
The order this family is usually described in does not hold, and the honest
version is more interesting. "Corners score above edges, which score above flat
ground" is true of the gradient *energy* and false of every corner measure. On
a nine-wide bright square, the tensor's trace reads 11.555556 at a corner,
10.666667 in the middle of an edge and 0.0 on flat ground: the energy separates
the square from its background by everything and a corner from an edge by 8%,
which is to say it cannot tell them apart at all. The determinant reads
30.222222 at the corner and **exactly 0.0** at the edge, because along a
straight edge every gradient in the window points the same way and the matrix
is exactly rank one. So the smaller eigenvalue is 4.0 at the corner and exactly
zero at the edge -- the same number flat ground gives.

Harris does separate an edge from flat ground, and in the direction nobody
expects: its response reads 24.880988 at the corner, ``-4.551111`` at the edge
and 0.0 on flat ground, so an edge scores *below* empty sky, because an edge is
where its ``- sensitivity * trace^2`` term finally has something to bite on.
Both measures are therefore saying the same thing in their own words, and it is
the thing this family exists for: an edge is no more locatable than flat
ground, and a measure that ranked it higher would be placing keypoints exactly
where they cannot be pinned.

The second surprise is how much of the work the suppression does. On that one
square, 32 pixels score above a threshold of 0.5, which is four corners
reported as eight pixels each; a separation of 2 takes it to 4, and every
larger separation up to 7 answers 4 as well. The blob is not a detail of the
method, it is most of the method's raw output.

The third was a bug, found by the spec and worth keeping because the wrong
version answers something plausible. The border exclusion was written the
obvious way, as a filter on the *candidates* before the suppression ran, and
that quietly promotes the pixel next to an excluded corner: on a square whose
corners sit two pixels from the frame, at a border of 3 it answers three
shoulder pixels at ``(3, 3)``, ``(3, 9)`` and ``(9, 3)`` scoring 3.5556 in
place of the corners at 4.0 it was asked to leave out. The count is right, the
strengths are plausible, and every position is wrong by a pixel. The border
now drops what is *reported* while a keypoint inside it still suppresses its
own neighbourhood first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from numpy.typing import DTypeLike
from pydantic import BaseModel, ConfigDict, Field

from oop_ml.core.computer_vision.edges import GradientField, GradientOperator
from oop_ml.core.computer_vision.filtering import EdgeRule, swept, windows_of
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.types import FloatArray, array_for_protocol

HARRIS_SENSITIVITY_CEILING = 0.25
"""The sensitivity at which Harris' response can no longer be positive.

For any real symmetric matrix the determinant is at most a quarter of the
squared trace, since ``a * b <= ((a + b) / 2)^2`` for the two eigenvalues. So
``determinant - sensitivity * trace^2`` is at most
``trace^2 * (0.25 - sensitivity)``, which is zero or below from a quarter
upward, whatever the picture contains.
"""

DEFAULT_HARRIS_SENSITIVITY = 0.04
"""The lower end of the range Harris and Stephens suggested, and the usual pick."""


class CornerMeasure(BaseModel, ABC):
    """How one pixel's structure tensor is turned into a single score.

    A class hierarchy rather than a closed enum, for the reason
    :class:`~oop_ml.core.kernel.functions.Kernel` gives and
    :class:`~oop_ml.core.distance.metric.DistanceMetric` does not: one of these
    takes a parameter. An enum member paired with a keyword argument that is
    read by one member and ignored by the other is the magic-string problem
    with a hat on.

    Every measure reads the determinant and the trace and nothing else, because
    those two settle a two-by-two symmetric matrix completely.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def score(self, tensor: StructureTensor) -> Picture:
        """The corner score at every pixel of ``tensor``."""
        return Picture(
            self._scored(tensor.determinant.values, tensor.trace.values),
        )

    @abstractmethod
    def _scored(self, determinant: FloatArray, trace: FloatArray) -> FloatArray:
        """Score every pixel from the two numbers that describe its matrix."""

    @property
    @abstractmethod
    def description(self) -> str:
        """A short phrase naming the measure, for a repr and for a report."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.description})"


class SmallestEigenvalueMeasure(CornerMeasure):
    """The smaller of the two eigenvalues, which is Shi and Tomasi's rule.

    It is the direct reading of the question: the window is pinned in both
    directions exactly as far as the *weaker* of the two directions is pinned,
    so the weaker one is the score. Never negative, in the units of a squared
    gradient, and takes no parameter, which is why it is this package's
    default.
    """

    def _scored(self, determinant: FloatArray, trace: FloatArray) -> FloatArray:
        gap = np.maximum(trace * trace - 4.0 * determinant, 0.0)
        return (trace - np.sqrt(gap)) / 2.0

    @property
    def description(self) -> str:
        return "the smaller eigenvalue"


class HarrisMeasure(CornerMeasure):
    """``determinant - sensitivity * trace^2``, which is Harris and Stephens'.

    An approximation to the smaller eigenvalue that avoids the square root, at
    a time when a square root per pixel was worth avoiding. It is large and
    positive where both eigenvalues are large, near zero on flat ground, and
    **negative** on an edge, where one eigenvalue is large and the other is not.
    That signed behaviour is a genuine difference from the smaller eigenvalue
    rather than a rounding of it, and the spec pins both numbers.

    Parameters
    ----------
    sensitivity:
        How heavily a large trace is punished. Must lie below
        :data:`HARRIS_SENSITIVITY_CEILING`, above which no picture can produce
        a positive response.

    Raises
    ------
    pydantic.ValidationError
        If the sensitivity is not above zero and below a quarter.
    """

    sensitivity: float = Field(
        default=DEFAULT_HARRIS_SENSITIVITY,
        gt=0.0,
        lt=HARRIS_SENSITIVITY_CEILING,
    )

    def _scored(self, determinant: FloatArray, trace: FloatArray) -> FloatArray:
        return determinant - self.sensitivity * trace * trace

    @property
    def description(self) -> str:
        return f"sensitivity={self.sensitivity}"


class StructureTensor:
    """The averaged products of the gradient directions, per pixel.

    Three pictures standing for the two-by-two symmetric matrix

    .. code-block:: text

        [ horizontal_squared   cross              ]
        [ cross                vertical_squared   ]

    at every position. Kept together as one object rather than handed back as
    three pictures, for the reason :class:`~oop_ml.core.computer_vision.edges.
    GradientField` gives about its two: a caller who has to pair them can pair
    them wrongly.

    The averaging window is what makes this a statement about a neighbourhood
    rather than about a pixel. Without it every matrix is the outer product of
    one gradient vector with itself, which is rank one everywhere, so the
    smaller eigenvalue is zero everywhere and no corner exists.

    Parameters
    ----------
    horizontal_squared:
        The average of the squared horizontal gradient over the window.
    vertical_squared:
        The same for the vertical gradient.
    cross:
        The average of the product of the two.

    Raises
    ------
    ShapeMismatchError
        If the three do not describe the same picture.
    """

    __slots__ = ("_cross", "_horizontal_squared", "_vertical_squared")

    def __init__(
        self,
        horizontal_squared: Picture,
        vertical_squared: Picture,
        cross: Picture,
    ) -> None:
        if not (horizontal_squared.shape == vertical_squared.shape == cross.shape):
            raise ShapeMismatchError(
                f"the three parts of one structure tensor describe the same "
                f"picture, got {horizontal_squared.shape}, "
                f"{vertical_squared.shape} and {cross.shape}"
            )
        self._horizontal_squared = horizontal_squared
        self._vertical_squared = vertical_squared
        self._cross = cross

    @classmethod
    def of(
        cls,
        picture: Picture,
        window_side: int = 3,
        operator: GradientOperator = GradientOperator.SOBEL,
        edge_rule: EdgeRule = EdgeRule.EXTEND,
    ) -> StructureTensor:
        """Build the tensor of ``picture`` by averaging over a square window.

        The window is flat rather than a bell, which is a real choice and not
        an oversight. A Gaussian window is the usual weighting and is a little
        steadier, because a flat square window is not the same in every
        direction; a flat one is a sweep of equal weights, which is the
        operation this package is built from, and it makes the answer at a
        pixel exactly an average of what the window covers rather than an
        average of something nobody wrote down.

        Parameters
        ----------
        picture:
            What to describe.
        window_side:
            How many pixels wide the averaging window is. Odd, so it has a
            centre pixel to answer at.
        operator:
            Which gradient estimate to build the products from.
        edge_rule:
            What the sweeps read where they hang off the picture.

        Raises
        ------
        InvalidValuesError
            If the window side is below one or is even.
        ShapeMismatchError
            If the window does not fit inside the picture.
        """
        if window_side < 1:
            raise InvalidValuesError(
                f"an averaging window has at least one pixel on each side, "
                f"got {window_side}"
            )
        field = GradientField.of(picture, operator, edge_rule)
        horizontal = field.horizontal.values
        vertical = field.vertical.values
        weights = np.full((window_side, window_side), 1.0 / (window_side * window_side))
        return cls(
            swept(Picture(horizontal * horizontal), weights, edge_rule),
            swept(Picture(vertical * vertical), weights, edge_rule),
            swept(Picture(horizontal * vertical), weights, edge_rule),
        )

    @property
    def horizontal_squared(self) -> Picture:
        """The averaged squared horizontal gradient, per pixel."""
        return self._horizontal_squared

    @property
    def vertical_squared(self) -> Picture:
        """The averaged squared vertical gradient, per pixel."""
        return self._vertical_squared

    @property
    def cross(self) -> Picture:
        """The averaged product of the two gradients, per pixel."""
        return self._cross

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the picture this describes."""
        return self._horizontal_squared.shape

    @property
    def determinant(self) -> Picture:
        """The product of the two eigenvalues, per pixel.

        Near zero wherever the window sees only one gradient direction, which
        is flat ground and every straight edge.
        """
        return Picture(
            self._horizontal_squared.values * self._vertical_squared.values
            - self._cross.values * self._cross.values
        )

    @property
    def trace(self) -> Picture:
        """The sum of the two eigenvalues, per pixel.

        The total gradient energy in the window, which is large on an edge and
        on a corner alike and is exactly the quantity that cannot tell them
        apart.
        """
        return Picture(self._horizontal_squared.values + self._vertical_squared.values)

    def response(self, measure: CornerMeasure | None = None) -> CornerResponse:
        """Score every pixel, defaulting to :class:`SmallestEigenvalueMeasure`."""
        chosen = measure if measure is not None else SmallestEigenvalueMeasure()
        return CornerResponse(chosen.score(self), chosen)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructureTensor) or type(self) is not type(other):
            return NotImplemented
        return (
            self._horizontal_squared == other._horizontal_squared
            and self._vertical_squared == other._vertical_squared
            and self._cross == other._cross
        )

    def __hash__(self) -> int:
        return hash((self._horizontal_squared, self._vertical_squared, self._cross))

    def __repr__(self) -> str:
        height, width = self.shape
        return f"StructureTensor(height={height}, width={width})"


class Keypoint:
    """One position worth describing, and how strongly it was preferred.

    Parameters
    ----------
    row:
        Which row of the picture, from the top.
    column:
        Which column, from the left.
    strength:
        The corner measure's score there. Carried because the whole point of a
        keypoint is that it was chosen over its neighbours, and a caller
        comparing two of them should not have to go back to the response.

    Raises
    ------
    InvalidValuesError
        If the position is outside a picture, or the strength is not finite.
    """

    __slots__ = ("_column", "_row", "_strength")

    def __init__(self, row: int, column: int, strength: float) -> None:
        if row < 0 or column < 0:
            raise InvalidValuesError(
                f"a keypoint sits inside a picture, got row {row} and column {column}"
            )
        if not np.isfinite(strength):
            raise InvalidValuesError(
                f"a keypoint's strength is a finite score, got {strength}"
            )
        self._row = int(row)
        self._column = int(column)
        self._strength = float(strength)

    @property
    def row(self) -> int:
        """Which row of the picture, from the top."""
        return self._row

    @property
    def column(self) -> int:
        """Which column of the picture, from the left."""
        return self._column

    @property
    def strength(self) -> float:
        """The corner measure's score at this position."""
        return self._strength

    def shifted_by(self, rows: int, columns: int) -> Keypoint:
        """The same keypoint moved, keeping its strength.

        Exists because the claim that keypoints belong to the scene rather
        than to the frame is checked by shifting a picture and shifting the
        expectation to match.
        """
        return Keypoint(self._row + rows, self._column + columns, self._strength)

    def is_further_than(self, separation: int, other: Keypoint) -> bool:
        """Whether ``other`` is at least ``separation`` pixels away.

        Measured as the larger of the two axis gaps rather than as a straight
        line, because the thing being suppressed is a square blob of response
        around one corner.
        """
        return (
            max(abs(self._row - other.row), abs(self._column - other.column))
            >= separation
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Keypoint) or type(self) is not type(other):
            return NotImplemented
        return (
            self._row == other._row
            and self._column == other._column
            and self._strength == other._strength
        )

    def __hash__(self) -> int:
        return hash((self._row, self._column, self._strength))

    def __repr__(self) -> str:
        return (
            f"Keypoint(row={self._row}, column={self._column}, "
            f"strength={self._strength:.6g})"
        )


class Keypoints:
    """The keypoints found in one picture, strongest first when peaks made them.

    Iterable and countable rather than a holder of a list, which is this
    library's rule everywhere: a collection that hands out its container has
    handed out the ability to change it behind the collection's back.

    Parameters
    ----------
    keypoints:
        The positions, in the order they should be read.
    """

    __slots__ = ("_keypoints",)

    def __init__(self, keypoints: Iterable[Keypoint]) -> None:
        self._keypoints = tuple(keypoints)

    @property
    def strongest(self) -> Keypoint:
        """The highest-scoring keypoint.

        Raises
        ------
        EmptyValuesError
            If nothing was found, which is a real answer for a picture with no
            corners in it and so is refused here rather than silently.
        """
        if not self._keypoints:
            raise EmptyValuesError(
                "no keypoints were found, so there is no strongest one"
            )
        return max(self._keypoints, key=lambda keypoint: keypoint.strength)

    def __iter__(self) -> Iterator[Keypoint]:
        return iter(self._keypoints)

    def __len__(self) -> int:
        return len(self._keypoints)

    def __getitem__(self, index: int) -> Keypoint:
        return self._keypoints[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Keypoints) or type(self) is not type(other):
            return NotImplemented
        return self._keypoints == other._keypoints

    def __hash__(self) -> int:
        return hash(self._keypoints)

    def __repr__(self) -> str:
        return f"Keypoints(n_keypoints={len(self._keypoints)})"


class CornerResponse:
    """A corner measure's score at every pixel, and the measure that made it.

    The measure travels with the scores because the numbers are meaningless
    without it: a reading of ``-40.96`` is an edge under Harris and impossible
    under the smaller eigenvalue, which is never negative.

    Parameters
    ----------
    scores:
        One score per pixel.
    measure:
        What produced them.
    """

    __slots__ = ("_measure", "_scores")

    def __init__(self, scores: Picture, measure: CornerMeasure) -> None:
        self._scores = scores
        self._measure = measure

    @classmethod
    def of(
        cls,
        picture: Picture,
        measure: CornerMeasure | None = None,
        window_side: int = 3,
        operator: GradientOperator = GradientOperator.SOBEL,
        edge_rule: EdgeRule = EdgeRule.EXTEND,
    ) -> CornerResponse:
        """Score ``picture`` from its own gradients, in one call.

        ``measure`` defaults to :class:`SmallestEigenvalueMeasure`; see the
        module docstring for why that rather than Harris.
        """
        return StructureTensor.of(picture, window_side, operator, edge_rule).response(
            measure
        )

    @property
    def scores(self) -> Picture:
        """The score at every pixel."""
        return self._scores

    @property
    def measure(self) -> CornerMeasure:
        """What produced the scores."""
        return self._measure

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the picture this describes."""
        return self._scores.shape

    def at(self, row: int, column: int) -> float:
        """The score at one position.

        Raises
        ------
        InvalidValuesError
            If the position is outside the picture.
        """
        height, width = self.shape
        if not (0 <= row < height and 0 <= column < width):
            raise InvalidValuesError(
                f"row {row}, column {column} is outside a {height} by {width} response"
            )
        return float(self._scores.values[row, column])

    def peaks(
        self,
        minimum_strength: float = 0.0,
        separation: int = 3,
        limit: int | None = None,
        border: int = 0,
    ) -> Keypoints:
        """The strong local peaks, strongest first.

        Candidates are every pixel scoring **strictly above**
        ``minimum_strength``, which is why the default of zero drops flat
        ground: flat ground scores exactly zero under the smaller eigenvalue.
        They are then taken in order of strength, and a candidate is refused if
        an already-taken keypoint lies within ``separation`` pixels of it. That
        ordering is what makes a separate local-maximum test unnecessary --
        anything kept was the strongest thing within its own radius, or
        something stronger would have taken it out.

        Ties are broken by the lower row and then the lower column, so two
        pixels with the identical score answer the same way on every run.

        Parameters
        ----------
        minimum_strength:
            Scores at or below this are not candidates.
        separation:
            The fewest pixels, along the wider of the two axes, that two kept
            keypoints may be apart. One means no suppression at all.
        limit:
            Stop after this many. ``None`` keeps every peak.
        border:
            Do not report keypoints within this many pixels of the picture's
            edge, where a sweep read invented pixels and where a patch will not
            fit. A keypoint inside the border still takes part in the
            suppression before it is dropped, which is not a detail: filtering
            the candidates first instead, measured on a square whose corners
            sit two pixels from the frame, answers a shoulder pixel at
            ``(3, 3)`` scoring 3.5556 in place of the corner at ``(2, 2)``
            scoring 4.0. Excluding a corner must not promote the pixel beside
            it.

        Raises
        ------
        InvalidValuesError
            If the separation is below one, the limit is below one, or the
            border is negative.
        """
        if separation < 1:
            raise InvalidValuesError(
                f"two keypoints are at least one pixel apart, got a separation "
                f"of {separation}"
            )
        if limit is not None and limit < 1:
            raise InvalidValuesError(
                f"a limit of {limit} asks for no keypoints at all; pass None to "
                f"ask for every peak"
            )
        if border < 0:
            raise InvalidValuesError(
                f"a border is a count of pixels to leave out, got {border}"
            )

        scores = self._scores.values
        height, width = self.shape
        rows, columns = np.nonzero(scores > minimum_strength)
        if rows.size == 0:
            return Keypoints(())
        strengths = scores[rows, columns]
        order = np.lexsort((columns, rows, -strengths))

        taken: list[Keypoint] = []
        reported: list[Keypoint] = []
        for position in order:
            candidate = Keypoint(
                int(rows[position]),
                int(columns[position]),
                float(strengths[position]),
            )
            if not all(candidate.is_further_than(separation, kept) for kept in taken):
                continue
            taken.append(candidate)
            if (
                border <= candidate.row < height - border
                and border <= candidate.column < width - border
            ):
                reported.append(candidate)
                if limit is not None and len(reported) == limit:
                    break
        return Keypoints(reported)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CornerResponse) or type(self) is not type(other):
            return NotImplemented
        return self._scores == other._scores and self._measure == other._measure

    def __repr__(self) -> str:
        height, width = self.shape
        return (
            f"CornerResponse(height={height}, width={width}, measure={self._measure!r})"
        )


class PatchDescriptor:
    """The square of brightness around a keypoint, centred and normalised.

    The two steps are the whole of the invariance. Subtracting the patch's own
    mean removes a change in exposure; dividing by the patch's length removes a
    change in contrast. What is left is a unit vector whose distance to another
    unit vector is a pure statement about pattern, running from ``0.0`` for an
    identical patch to ``2.0`` for its exact negative.

    Not scale invariant and not rotation invariant. See the module docstring;
    this is stated in both places because a reader who assumes otherwise will
    get a plausible answer rather than an error.

    A patch with no variation at all -- flat ground -- has nothing to normalise
    and comes back as zeros, the same choice
    :meth:`~oop_ml.core.computer_vision.picture.Picture.rescaled_to_unit`
    makes. Every such descriptor is zero distance from every other, which is
    correct and is exactly why keypoints are hunted at corners.

    Parameters
    ----------
    keypoint:
        Where the patch was taken from.
    values:
        The centred, normalised patch, read row by row.

    Raises
    ------
    InvalidValuesError
        If the values are not a finite one-dimensional run.
    EmptyValuesError
        If there are no values.
    """

    __slots__ = ("_keypoint", "_values")

    def __init__(self, keypoint: Keypoint, values: Any) -> None:
        run = np.asarray(values, dtype=np.float64)
        if run.ndim != 1:
            raise InvalidValuesError(
                f"a descriptor is one run of numbers, got shape {run.shape}"
            )
        if run.size == 0:
            raise EmptyValuesError("a descriptor needs at least one number")
        if not np.all(np.isfinite(run)):
            raise InvalidValuesError("a descriptor holds finite numbers")
        self._keypoint = keypoint
        self._values = run.copy()
        self._values.setflags(write=False)

    @classmethod
    def of(cls, picture: Picture, keypoint: Keypoint, side: int = 5) -> PatchDescriptor:
        """Describe the ``side`` by ``side`` patch centred on ``keypoint``.

        Raises
        ------
        InvalidValuesError
            If the side is not odd and positive.
        ShapeMismatchError
            If the patch runs off the picture. It is refused rather than
            padded, because a padded patch describes pixels the scene does not
            contain and would match another picture's padding.
        """
        if side < 1 or side % 2 == 0:
            raise InvalidValuesError(
                f"a patch is centred on its keypoint, so its side is odd and "
                f"positive; got {side}"
            )
        half = side // 2
        top, left = keypoint.row - half, keypoint.column - half
        height, width = picture.shape
        if top < 0 or left < 0 or top + side > height or left + side > width:
            raise ShapeMismatchError(
                f"a {side} by {side} patch centred on row {keypoint.row}, "
                f"column {keypoint.column} runs off a {height} by {width} "
                f"picture; keep keypoints at least {half} pixels from the "
                f"border, which is what the border argument of peaks is for"
            )
        patch = windows_of(picture, side, side)[top, left].reshape(-1)
        centred = patch - patch.mean()
        length = float(np.linalg.norm(centred))
        if length == 0.0:
            return cls(keypoint, np.zeros_like(centred))
        return cls(keypoint, centred / length)

    @property
    def keypoint(self) -> Keypoint:
        """Where the patch was taken from."""
        return self._keypoint

    @property
    def values(self) -> FloatArray:
        """The centred, normalised patch, frozen. Copy it before writing."""
        return self._values

    @property
    def n_values(self) -> int:
        """How many numbers describe the patch."""
        return int(self._values.size)

    @property
    def is_flat(self) -> bool:
        """Whether the patch had no variation to normalise."""
        return not bool(np.any(self._values))

    def distance_to(self, other: PatchDescriptor) -> float:
        """How far this description is from another, as a straight line.

        Raises
        ------
        ShapeMismatchError
            If the two describe patches of different sizes.
        """
        if self.n_values != other.n_values:
            raise ShapeMismatchError(
                f"two descriptors compared to each other describe patches of "
                f"one size, got {self.n_values} numbers and {other.n_values}"
            )
        return float(np.linalg.norm(self._values - other.values))

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._values, dtype, copy)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PatchDescriptor) or type(self) is not type(other):
            return NotImplemented
        return self._keypoint == other._keypoint and bool(
            np.array_equal(self._values, other._values)
        )

    def __hash__(self) -> int:
        return hash((self._keypoint, self._values.tobytes()))

    def __repr__(self) -> str:
        return (
            f"PatchDescriptor(row={self._keypoint.row}, "
            f"column={self._keypoint.column}, n_values={self.n_values})"
        )


class DescriptorMatch:
    """One descriptor's nearest neighbour, kept with what it beat.

    The runner-up is part of the match rather than a diagnostic beside it,
    because the ratio between the two is the only thing that says whether the
    match means anything, and a match object that reported the distance alone
    would invite exactly the threshold that does not transfer between pictures.

    Parameters
    ----------
    first:
        The descriptor that was asked about.
    second:
        The nearest descriptor in the other set.
    distance:
        How far apart those two are.
    runner_up_distance:
        How far the second nearest was.

    Raises
    ------
    InvalidValuesError
        If either distance is negative or not finite, or the runner-up is
        nearer than the winner, which no ordering can produce.
    """

    __slots__ = ("_distance", "_first", "_runner_up_distance", "_second")

    def __init__(
        self,
        first: PatchDescriptor,
        second: PatchDescriptor,
        distance: float,
        runner_up_distance: float,
    ) -> None:
        if not (np.isfinite(distance) and np.isfinite(runner_up_distance)):
            raise InvalidValuesError(
                f"a match's distances are finite, got {distance} and "
                f"{runner_up_distance}"
            )
        if distance < 0.0 or runner_up_distance < 0.0:
            raise InvalidValuesError(
                f"a distance is never negative, got {distance} and {runner_up_distance}"
            )
        if runner_up_distance < distance:
            raise InvalidValuesError(
                f"the runner-up is not nearer than the winner, got a winner at "
                f"{distance} and a runner-up at {runner_up_distance}"
            )
        self._first = first
        self._second = second
        self._distance = float(distance)
        self._runner_up_distance = float(runner_up_distance)

    @property
    def first(self) -> PatchDescriptor:
        """The descriptor that was asked about."""
        return self._first

    @property
    def second(self) -> PatchDescriptor:
        """The nearest descriptor in the other set."""
        return self._second

    @property
    def distance(self) -> float:
        """How far apart the two are."""
        return self._distance

    @property
    def runner_up_distance(self) -> float:
        """How far the second nearest was."""
        return self._runner_up_distance

    @property
    def ratio(self) -> float:
        """The winner's distance over the runner-up's, in ``[0, 1]``.

        Near zero for a match that is distinctive and near one for a match that
        had an equally good alternative. Two equally good answers at zero
        distance -- two identical patches in the other picture -- answer one,
        because that is what the number means, and dividing zero by zero would
        answer nothing at all.
        """
        if self._runner_up_distance == 0.0:
            return 1.0
        return self._distance / self._runner_up_distance

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DescriptorMatch) or type(self) is not type(other):
            return NotImplemented
        return (
            self._first == other._first
            and self._second == other._second
            and self._distance == other._distance
            and self._runner_up_distance == other._runner_up_distance
        )

    def __hash__(self) -> int:
        return hash(
            (self._first, self._second, self._distance, self._runner_up_distance)
        )

    def __repr__(self) -> str:
        return (
            f"DescriptorMatch(row={self._first.keypoint.row}, "
            f"column={self._first.keypoint.column}, ratio={self.ratio:.4g})"
        )


class DescriptorMatches:
    """The matches found between two sets of descriptors.

    Iterable and countable, and never handing out its container, which is the
    rule :class:`Keypoints` follows for the same reason.
    """

    __slots__ = ("_matches",)

    def __init__(self, matches: Iterable[DescriptorMatch]) -> None:
        self._matches = tuple(matches)

    @property
    def most_distinctive(self) -> DescriptorMatch:
        """The match whose runner-up was furthest behind it.

        Raises
        ------
        EmptyValuesError
            If nothing matched.
        """
        if not self._matches:
            raise EmptyValuesError("no descriptors matched, so there is no best one")
        return min(self._matches, key=lambda match: match.ratio)

    def __iter__(self) -> Iterator[DescriptorMatch]:
        return iter(self._matches)

    def __len__(self) -> int:
        return len(self._matches)

    def __getitem__(self, index: int) -> DescriptorMatch:
        return self._matches[index]

    def __repr__(self) -> str:
        return f"DescriptorMatches(n_matches={len(self._matches)})"


class Descriptors:
    """The descriptions of one picture's keypoints.

    Parameters
    ----------
    descriptors:
        The descriptions, in the order their keypoints were found.

    Raises
    ------
    ShapeMismatchError
        If they do not all describe patches of one size, which would make the
        set uncomparable against anything including itself.
    """

    __slots__ = ("_descriptors",)

    def __init__(self, descriptors: Iterable[PatchDescriptor]) -> None:
        held = tuple(descriptors)
        sizes = {descriptor.n_values for descriptor in held}
        if len(sizes) > 1:
            raise ShapeMismatchError(
                f"one set of descriptors describes patches of one size, got "
                f"sizes {sorted(sizes)}"
            )
        self._descriptors = held

    @classmethod
    def of(cls, picture: Picture, keypoints: Keypoints, side: int = 5) -> Descriptors:
        """Describe every keypoint's patch.

        Raises
        ------
        ShapeMismatchError
            If any keypoint's patch runs off the picture.
        """
        return cls(
            PatchDescriptor.of(picture, keypoint, side) for keypoint in keypoints
        )

    @property
    def keypoints(self) -> Keypoints:
        """The positions these describe, in the same order."""
        return Keypoints(descriptor.keypoint for descriptor in self._descriptors)

    def matched_to(
        self, other: Descriptors, maximum_ratio: float = 0.8
    ) -> DescriptorMatches:
        """Each of these matched against its nearest in ``other``.

        A match is kept only when the nearest is enough nearer than the second
        nearest, which is Lowe's ratio test; see the module docstring for why a
        distance alone will not do.

        Parameters
        ----------
        other:
            The descriptors to search in.
        maximum_ratio:
            The largest winner-over-runner-up ratio still counted as a match.
            0.8 is Lowe's own figure. One keeps everything, which is to say it
            turns the test off.

        Raises
        ------
        InvalidValuesError
            If the ratio is not above zero and at most one.
        TooFewValuesError
            If ``other`` holds fewer than two descriptors, since with one there
            is no runner-up and the test cannot be asked.
        ShapeMismatchError
            If the two sets describe patches of different sizes.
        """
        if not 0.0 < maximum_ratio <= 1.0:
            raise InvalidValuesError(
                f"a ratio test compares a winner against a runner-up, so its "
                f"threshold lies in (0, 1]; got {maximum_ratio}"
            )
        if len(other) < 2:
            raise TooFewValuesError(
                f"the ratio test needs a runner-up to compare against, so it "
                f"needs at least two descriptors to search in; got {len(other)}"
            )
        if not self._descriptors:
            return DescriptorMatches(())

        mine = np.stack([descriptor.values for descriptor in self._descriptors])
        theirs = np.stack([descriptor.values for descriptor in other])
        if mine.shape[1] != theirs.shape[1]:
            raise ShapeMismatchError(
                f"two sets of descriptors compared to each other describe "
                f"patches of one size, got {mine.shape[1]} numbers and "
                f"{theirs.shape[1]}"
            )

        distances = np.linalg.norm(mine[:, None, :] - theirs[None, :, :], axis=2)
        ordering = np.argsort(distances, axis=1)

        kept: list[DescriptorMatch] = []
        for position, descriptor in enumerate(self._descriptors):
            winner = int(ordering[position, 0])
            runner_up = int(ordering[position, 1])
            match = DescriptorMatch(
                descriptor,
                other[winner],
                float(distances[position, winner]),
                float(distances[position, runner_up]),
            )
            if match.ratio <= maximum_ratio:
                kept.append(match)
        return DescriptorMatches(kept)

    def __iter__(self) -> Iterator[PatchDescriptor]:
        return iter(self._descriptors)

    def __len__(self) -> int:
        return len(self._descriptors)

    def __getitem__(self, index: int) -> PatchDescriptor:
        return self._descriptors[index]

    def __repr__(self) -> str:
        return f"Descriptors(n_descriptors={len(self._descriptors)})"
