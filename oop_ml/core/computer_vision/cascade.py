"""Reject almost everything cheaply, and spend the effort on what is left.

What this module is, and what it is not
---------------------------------------
This is the mechanism behind Viola and Jones' detector: the integral image, the
rectangle features it makes affordable, and the cascade of boosted stages that
turns a very expensive classifier into a very cheap one for almost every window
it is shown. It is *not* a shipped face detector. There is no trained
twenty-five-stage classifier here and no six-thousand-feature front end; the
fixtures in the spec are hand-built patterns a few pixels across, and the point
of them is that every number the method claims can be measured on something
small enough to check by hand. A face detector is this mechanism plus a corpus
of five thousand faces and a week of fitting, and the corpus is the part that is
missing rather than the idea.

Three separable ideas, three objects
------------------------------------
**The integral image** (:class:`IntegralImage`) is a table whose entry at
``(row, column)`` is the sum of every pixel above and to the left of it. Given
that table, the sum over *any* rectangle is::

    table[bottom, right] - table[top, right] - table[bottom, left] + table[top, left]

Four lookups and three additions, whatever the rectangle's size. A rectangle of
one pixel costs four lookups and a rectangle of three hundred and sixty thousand
pixels costs four lookups -- measured, the second takes 1.06 times the first
where summing the pixels directly takes 136 times as much -- and that single
fact is what makes everything after it possible. Without it a detector reading
rectangle sums at every position and every scale would be summing the same
pixels over and over, and the cost would be the area of the rectangle rather
than a constant.
:meth:`IntegralImage.corners_for` names the four lookups as objects rather than
performing them, so the constant cost is something a reader can count rather
than something this docstring asserts.

**The rectangle features** (:class:`RectangleFeature`) are the only thing the
detector ever reads. A feature is the sum over some bright cells minus the sum
over some dark ones, in five arrangements
(:class:`FeatureArrangement`): two cells side by side or stacked, three in a
row or in a column, and four in a chequer. Each is a statement about *contrast*
between neighbouring boxes -- brighter here than there -- which is the thing
that survives a change in lighting, for the reason
:mod:`oop_ml.core.computer_vision.edges` gives for gradients. With one measured
caveat: only three of the five arrangements have signs that cancel, so the two
three-cell ones read the window's overall brightness as well as its structure.
:data:`CELL_SIGNS` says what that costs and why the original detector
normalises a window's variance first.

A feature is fixed by its arrangement, its position and its cell size, and
nothing else, so a window admits an enormous number of them.
:meth:`FeatureBank.count_available` says how many, and the number is the reason
a cascade has to exist at all: a twenty-four by twenty-four window holds 576
pixels and admits 162,336 features, 281.8 times its own pixel count. Evaluating
all of them at every window position of even a modest picture is not slow, it is
impossible.

**The cascade** (:class:`HaarCascade`) is the answer to that. A stage is a
weighted vote of :class:`ThresholdRule` -- one feature, one threshold, one
direction, which is the simplest classifier that can be written down -- chosen
by boosting, in the shape
:class:`~oop_ml.numpy.classification.ensembles.adaboost_classifier.AdaBoostClassifier`
already uses. A window is accepted only if it survives every stage, and a stage
is *calibrated* rather than merely fitted: its acceptance threshold is lowered
until it keeps every positive it was trained on. That trade is deliberate and
is the whole design. A stage that keeps all the positives also keeps a good
share of the negatives, so no stage is any good on its own; what each stage does
is throw away a fraction of what reaches it, and the fractions multiply.

Why the ordering is the contribution
------------------------------------
The first stage has a handful of rules and runs on every window. The last stage
has the same handful and runs on the few windows that got that far, so the cost
approaches the *first* stage's cost times the number of windows however many
stages follow it, and a detector of six thousand rules ends up costing what a
detector of ten rules costs.

:meth:`HaarCascade.scan` measures that rather than claiming it, and the measured
number on a small cascade is worth reading beside the famous one. A three-stage
cascade of five rules saves only 2.3 times, because the saving is bounded by how
many rules there are to skip and there are five. The ratio that is not bounded
is against the *feature bank*: measured on a 66 by 66 scene, 6,657 rule
evaluations against the 31,290,600 it would take to measure every feature of a
twelve by twelve window at every position.

Where it fails, which is not a detail
-------------------------------------
The arrangements are axis-aligned and signed. A feature that says "bright above,
dark below" says nothing whatever about "dark above, bright below" -- it answers
with the opposite sign, and a rule thresholding it in one direction rejects the
mirrored pattern outright. The same goes for rotation: the vertical
two-rectangle feature and the horizontal one are different features, and a
cascade fitted on one has never seen the other. So a target photographed upside
down, or lit from the other side, is not merely detected less often; it is not
detected at all, and the spec measures that rather than mentioning it. Real
implementations answer this by fitting one cascade per pose, which is a
statement about how much the method does not generalise.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Classifier
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import Probabilities
from oop_ml.core.exceptions import (
    DivergenceError,
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    ShapeMismatchError,
)
from oop_ml.core.types import FloatArray, IndexArray, MaskArray

LOOKUPS_PER_RECTANGLE = 4
"""How many table entries one rectangle sum reads, whatever its size.

Written down as a name because it is the whole claim of the integral image and
because :meth:`IntegralImage.corners_for` hands back exactly this many objects,
so the claim is checkable rather than merely stated.
"""

DEFAULT_ACCEPTANCE_SHARE = 0.5
"""The share of the total voice a plain boosted vote demands.

Half, which is what AdaBoost's own decision rule comes to once the votes are
written as shares. A cascade stage only ever *lowers* it, because letting a
negative through costs a later stage a little work and dropping a positive
costs the detector the target entirely.
"""

ACCEPTANCE_TOLERANCE = 1e-9
"""How far below its threshold a confidence may sit and still be accepted.

Two routes reach one feature value: the table built over a single window-sized
picture during fitting, and the table built over a whole scene during a scan.
The pixels are the same and the arithmetic is the same, but the running sums
form at different magnitudes, so the two answers can part in the last bits. A
calibrated threshold sits exactly *on* the confidence of the hardest positive,
which is precisely where that difference decides an inequality, so the
comparison is made with room rather than exactly.
"""

SILENT_RULE_VOICE = 1e-10
"""Below this a rule is treated as having nothing to say.

The bar is that a rule beats guessing, which for two classes means a weighted
error below one half and therefore a voice above zero -- and testing the voice
rather than the error is the lesson
:mod:`oop_ml.numpy.classification.ensembles.adaboost_classifier` records, since
a rule that is wrong on exactly half the weight can compute to a voice of
2.2e-16 rather than to zero.
"""

PERFECT_RULE_VOICE = 1.0
"""The voice given to a rule that got every weighted row right.

``log((1 - error) / error)`` is infinite there, and an infinite voice would
drown out every other rule in the stage. A perfect rule also ends the round,
since nothing would be reweighted, so what its exact voice is only matters when
it is alone -- and then any positive number gives the same decision.
"""


class Rectangle:
    """A box inside a picture, addressed by its top-left pixel and its size.

    Parameters
    ----------
    top:
        The first row inside the box.
    left:
        The first column inside the box.
    height:
        How many rows.
    width:
        How many columns.

    Raises
    ------
    InvalidValuesError
        If the origin is negative or either side is below one.
    """

    __slots__ = ("_height", "_left", "_top", "_width")

    def __init__(self, top: int, left: int, height: int, width: int) -> None:
        if top < 0 or left < 0:
            raise InvalidValuesError(
                f"a rectangle starts inside the picture, got row {top} and "
                f"column {left}"
            )
        if height < 1 or width < 1:
            raise InvalidValuesError(
                f"a rectangle has at least one pixel on each side, got {height} "
                f"by {width}"
            )
        self._top = int(top)
        self._left = int(left)
        self._height = int(height)
        self._width = int(width)

    @property
    def top(self) -> int:
        """The first row inside the box."""
        return self._top

    @property
    def left(self) -> int:
        """The first column inside the box."""
        return self._left

    @property
    def height(self) -> int:
        """How many rows."""
        return self._height

    @property
    def width(self) -> int:
        """How many columns."""
        return self._width

    @property
    def bottom(self) -> int:
        """One past the last row, so ``top`` and ``bottom`` bracket the rows."""
        return self._top + self._height

    @property
    def right(self) -> int:
        """One past the last column."""
        return self._left + self._width

    @property
    def n_pixels(self) -> int:
        """How many pixels the box covers.

        The number a direct sum would have to read, and the one to hold beside
        :data:`LOOKUPS_PER_RECTANGLE`.
        """
        return self._height * self._width

    def shifted_by(self, rows: int, columns: int) -> Rectangle:
        """The same box moved down and to the right.

        How a feature written against a window's own coordinates becomes a box
        in the picture that window was cut from.
        """
        return Rectangle(
            self._top + rows, self._left + columns, self._height, self._width
        )

    def __eq__(self, other: object) -> bool:
        # Both halves of the guard, the house idiom the foundation uses.
        # The exact-type check keeps a subclass carrying one more thing from
        # comparing equal to a plain one; the isinstance is what narrows
        # ``object`` so the attribute reads below are checkable.
        if not isinstance(other, Rectangle) or type(self) is not type(other):
            return NotImplemented
        return (
            self._top == other._top
            and self._left == other._left
            and self._height == other._height
            and self._width == other._width
        )

    def __hash__(self) -> int:
        return hash((self._top, self._left, self._height, self._width))

    def __repr__(self) -> str:
        return (
            f"Rectangle(top={self._top}, left={self._left}, "
            f"height={self._height}, width={self._width})"
        )


class CornerLookup:
    """One entry of an integral table, and the sign it enters a sum with.

    A class rather than a triple because the three carry different meanings and
    a caller reading ``lookup[2]`` would have to remember which was which.
    """

    __slots__ = ("_column", "_row", "_sign")

    def __init__(self, row: int, column: int, sign: int) -> None:
        self._row = int(row)
        self._column = int(column)
        self._sign = int(sign)

    @property
    def row(self) -> int:
        """Which row of the table is read."""
        return self._row

    @property
    def column(self) -> int:
        """Which column of the table is read."""
        return self._column

    @property
    def sign(self) -> int:
        """``+1`` if the entry is added, ``-1`` if it is subtracted."""
        return self._sign

    def __eq__(self, other: object) -> bool:
        # Both halves of the guard, the house idiom the foundation uses.
        # The exact-type check keeps a subclass carrying one more thing from
        # comparing equal to a plain one; the isinstance is what narrows
        # ``object`` so the attribute reads below are checkable.
        if not isinstance(other, CornerLookup) or type(self) is not type(other):
            return NotImplemented
        return (
            self._row == other._row
            and self._column == other._column
            and self._sign == other._sign
        )

    def __hash__(self) -> int:
        return hash((self._row, self._column, self._sign))

    def __repr__(self) -> str:
        return (
            f"CornerLookup(row={self._row}, column={self._column}, "
            f"sign={self._sign:+d})"
        )


class RectangleCorners:
    """The four table entries one rectangle sum reads.

    The reason this exists rather than the sum being computed and the count
    asserted in prose: a caller can count these, and counting them for a
    one-pixel box and for a nine-hundred-pixel-square box gives the same
    answer. That is the integral image's entire claim, made checkable.

    Raises
    ------
    InvalidValuesError
        If handed anything other than :data:`LOOKUPS_PER_RECTANGLE` lookups.
    """

    __slots__ = ("_lookups",)

    def __init__(self, lookups: Sequence[CornerLookup]) -> None:
        held = tuple(lookups)
        if len(held) != LOOKUPS_PER_RECTANGLE:
            raise InvalidValuesError(
                f"a rectangle sum reads exactly {LOOKUPS_PER_RECTANGLE} table "
                f"entries whatever its size, got {len(held)}"
            )
        self._lookups = held

    def __iter__(self) -> Iterator[CornerLookup]:
        return iter(self._lookups)

    def __len__(self) -> int:
        return len(self._lookups)

    def __getitem__(self, position: int) -> CornerLookup:
        return self._lookups[position]

    def __repr__(self) -> str:
        return f"RectangleCorners({len(self._lookups)} lookups)"


class IntegralImage:
    """Every prefix sum of a picture, so any rectangle sum is four lookups.

    The table is one row and one column larger than the picture, and its extra
    first row and column hold zero. That is not padding for its own sake: it is
    what lets a rectangle touching the top or left edge be read by the same
    four-lookup expression as one in the middle, instead of by three branches
    for the boxes whose ``top`` or ``left`` is zero.

    Parameters
    ----------
    picture:
        What to accumulate.
    """

    __slots__ = ("_shape", "_table")

    def __init__(self, picture: Picture) -> None:
        table = np.zeros((picture.height + 1, picture.width + 1), dtype=np.float64)
        table[1:, 1:] = picture.values.cumsum(axis=0).cumsum(axis=1)
        table.setflags(write=False)
        self._table = table
        self._shape = picture.shape

    @classmethod
    def of(cls, picture: Picture) -> IntegralImage:
        """Accumulate ``picture``. Named so the call reads as what it does."""
        return cls(picture)

    @property
    def table(self) -> FloatArray:
        """The prefix sums, frozen. ``(height + 1, width + 1)``."""
        return self._table

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the picture this was built from."""
        return self._shape

    def check_contains(self, rectangle: Rectangle) -> None:
        """Refuse a box that runs off the picture.

        Raises
        ------
        ShapeMismatchError
            If the box does not fit.
        """
        height, width = self._shape
        if rectangle.bottom > height or rectangle.right > width:
            raise ShapeMismatchError(
                f"{rectangle!r} runs past a {height} by {width} picture"
            )

    def corners_for(self, rectangle: Rectangle) -> RectangleCorners:
        """The four table entries whose signed total is this box's sum.

        Naming them rather than reading them, so that the count is a thing a
        caller can measure. See :class:`RectangleCorners`.

        Raises
        ------
        ShapeMismatchError
            If the box does not fit inside the picture.
        """
        self.check_contains(rectangle)
        return RectangleCorners(
            (
                CornerLookup(rectangle.bottom, rectangle.right, +1),
                CornerLookup(rectangle.top, rectangle.right, -1),
                CornerLookup(rectangle.bottom, rectangle.left, -1),
                CornerLookup(rectangle.top, rectangle.left, +1),
            )
        )

    def sum_over(self, rectangle: Rectangle) -> float:
        """The total brightness inside ``rectangle``.

        Four lookups and three additions, whatever the box's area. The
        expression is written out rather than looped over
        :meth:`corners_for`, because this is the hot path and the two routes
        are pinned against each other in the spec instead.

        Raises
        ------
        ShapeMismatchError
            If the box does not fit inside the picture.
        """
        self.check_contains(rectangle)
        table = self._table
        return float(
            table[rectangle.bottom, rectangle.right]
            - table[rectangle.top, rectangle.right]
            - table[rectangle.bottom, rectangle.left]
            + table[rectangle.top, rectangle.left]
        )

    def __repr__(self) -> str:
        height, width = self._shape
        return f"IntegralImage(height={height}, width={width})"


class FeatureArrangement(StrEnum):
    """How a feature's cells are laid out, and which of them are subtracted.

    A closed enum for the reason
    :class:`~oop_ml.core.distance.metric.DistanceMetric` gives: none of the
    five takes a parameter of its own, so there is nothing for an object to
    hold, and a wrong value should be a type error rather than a plausible
    answer computed from an arrangement nobody chose.
    """

    TWO_HORIZONTAL = "two_horizontal"
    """Two cells side by side. Answers "is the left brighter than the right"."""

    TWO_VERTICAL = "two_vertical"
    """Two cells stacked. Answers "is the top brighter than the bottom"."""

    THREE_HORIZONTAL = "three_horizontal"
    """Three cells in a row, the middle one against the two outside."""

    THREE_VERTICAL = "three_vertical"
    """Three cells in a column, the middle one against the two outside."""

    FOUR_CHEQUER = "four_chequer"
    """Four cells in a square, one diagonal pair against the other."""


CELL_SIGNS: dict[FeatureArrangement, tuple[tuple[int, ...], ...]] = {
    FeatureArrangement.TWO_HORIZONTAL: ((+1, -1),),
    FeatureArrangement.TWO_VERTICAL: ((+1,), (-1,)),
    FeatureArrangement.THREE_HORIZONTAL: ((-1, +1, -1),),
    FeatureArrangement.THREE_VERTICAL: ((-1,), (+1,), (-1,)),
    FeatureArrangement.FOUR_CHEQUER: ((+1, -1), (-1, +1)),
}
"""Each arrangement's cells, laid out as they sit, with the sign each carries.

The three-cell arrangements put the middle cell against the two outside ones
rather than alternating, which is the original paper's reading: the value is
the centre's sum less the two flanks'. That makes the feature a detector for a
*line* -- a bright strip between two dark ones -- where the two-cell
arrangements detect an *edge*, and the four-cell one detects a diagonal
structure that neither of the others can see.

**Two of the five are not zero-mean, and that is the paper's arrangement rather
than an oversight here.** Four ``+1`` and four ``-1`` cancel, so the two-cell
and four-cell features answer exactly zero on a picture of uniform brightness
whatever that brightness is. The three-cell ones carry one ``+1`` against two
``-1``, so on a uniform picture of brightness ``b`` with cells of ``n`` pixels
they answer ``-n b`` -- they read the window's overall *level* as well as its
structure, and turning the lamp up moves them. Viola and Jones answer that by
normalising each window's variance before any threshold is applied, which this
module does not do; so the three-cell features here are the ones that would
break first under a lighting change, and the spec measures the offset rather
than leaving the claim as prose.
"""


class SignedRectangle:
    """One cell of a feature: a box, and whether it is added or subtracted.

    The pairing modelled rather than handed back as a two-tuple, for the reason
    this library gives everywhere: a caller should not have to remember that
    position zero was the box.
    """

    __slots__ = ("_rectangle", "_sign")

    def __init__(self, rectangle: Rectangle, sign: int) -> None:
        if sign not in (-1, +1):
            raise InvalidValuesError(
                f"a feature's cell is added or subtracted, so its sign is +1 or "
                f"-1; got {sign}"
            )
        self._rectangle = rectangle
        self._sign = int(sign)

    @property
    def rectangle(self) -> Rectangle:
        """Where the cell is."""
        return self._rectangle

    @property
    def sign(self) -> int:
        """``+1`` for a bright cell, ``-1`` for a dark one."""
        return self._sign

    def __eq__(self, other: object) -> bool:
        # Both halves of the guard, the house idiom the foundation uses.
        # The exact-type check keeps a subclass carrying one more thing from
        # comparing equal to a plain one; the isinstance is what narrows
        # ``object`` so the attribute reads below are checkable.
        if not isinstance(other, SignedRectangle) or type(self) is not type(other):
            return NotImplemented
        return self._rectangle == other._rectangle and self._sign == other._sign

    def __hash__(self) -> int:
        return hash((self._rectangle, self._sign))

    def __repr__(self) -> str:
        return f"SignedRectangle({self._rectangle!r}, sign={self._sign:+d})"


class RectangleFeature:
    """Bright cells minus dark ones, in one of five fixed arrangements.

    A feature is fixed by four numbers and an arrangement: where its top-left
    corner sits inside the window, and how tall and wide each of its cells is.
    Its own extent follows -- an arrangement of two cells side by side, at cell
    size ``h`` by ``w``, is ``h`` tall and ``2w`` wide.

    Parameters
    ----------
    arrangement:
        Which layout of cells, and therefore which signs.
    top, left:
        The feature's origin, in the *window's* coordinates rather than the
        picture's. A feature says "eleven rows down from wherever this window
        starts", which is what lets one feature be evaluated at every window
        position of a scene from a single integral image.
    cell_height, cell_width:
        The size of one cell. Every cell of an arrangement is the same size,
        which is what makes the count in
        :meth:`FeatureBank.count_available` come out the way it does.

    Raises
    ------
    InvalidValuesError
        If the origin is negative or either cell side is below one.
    """

    __slots__ = ("_arrangement", "_cell_height", "_cell_width", "_left", "_top")

    def __init__(
        self,
        arrangement: FeatureArrangement,
        top: int,
        left: int,
        cell_height: int,
        cell_width: int,
    ) -> None:
        if top < 0 or left < 0:
            raise InvalidValuesError(
                f"a feature starts inside its window, got row {top} and column {left}"
            )
        if cell_height < 1 or cell_width < 1:
            raise InvalidValuesError(
                f"a feature's cell has at least one pixel on each side, got "
                f"{cell_height} by {cell_width}"
            )
        self._arrangement = arrangement
        self._top = int(top)
        self._left = int(left)
        self._cell_height = int(cell_height)
        self._cell_width = int(cell_width)

    @property
    def arrangement(self) -> FeatureArrangement:
        """Which layout of cells this is."""
        return self._arrangement

    @property
    def top(self) -> int:
        """The feature's first row, counted from the window's own top."""
        return self._top

    @property
    def left(self) -> int:
        """The feature's first column, counted from the window's own left."""
        return self._left

    @property
    def cell_height(self) -> int:
        """How many rows one cell covers."""
        return self._cell_height

    @property
    def cell_width(self) -> int:
        """How many columns one cell covers."""
        return self._cell_width

    @property
    def rows_of_cells(self) -> int:
        """How many cells the arrangement stacks."""
        return len(CELL_SIGNS[self._arrangement])

    @property
    def columns_of_cells(self) -> int:
        """How many cells the arrangement puts side by side."""
        return len(CELL_SIGNS[self._arrangement][0])

    @property
    def height(self) -> int:
        """How many rows the whole feature covers."""
        return self._cell_height * self.rows_of_cells

    @property
    def width(self) -> int:
        """How many columns the whole feature covers."""
        return self._cell_width * self.columns_of_cells

    @property
    def n_cells(self) -> int:
        """How many boxes are summed."""
        return self.rows_of_cells * self.columns_of_cells

    def fits_in(self, window_height: int, window_width: int) -> bool:
        """Whether the whole feature lies inside a window that size."""
        return (
            self._top + self.height <= window_height
            and self._left + self.width <= window_width
        )

    def cells_in(self, window: Rectangle) -> tuple[SignedRectangle, ...]:
        """The feature's boxes, in the coordinates of the picture.

        Raises
        ------
        ShapeMismatchError
            If the feature does not fit inside the window.
        """
        if not self.fits_in(window.height, window.width):
            raise ShapeMismatchError(
                f"a {self.height} by {self.width} feature at row {self._top}, "
                f"column {self._left} does not fit inside a {window.height} by "
                f"{window.width} window"
            )
        cells = []
        for cell_row, signs in enumerate(CELL_SIGNS[self._arrangement]):
            for cell_column, sign in enumerate(signs):
                box = Rectangle(
                    window.top + self._top + cell_row * self._cell_height,
                    window.left + self._left + cell_column * self._cell_width,
                    self._cell_height,
                    self._cell_width,
                )
                cells.append(SignedRectangle(box, sign))
        return tuple(cells)

    def value_in(self, integral_image: IntegralImage, window: Rectangle) -> float:
        """The signed total, read through ``integral_image``.

        Costs :data:`LOOKUPS_PER_RECTANGLE` lookups per cell and nothing that
        depends on how large the cells are, which is the point of the whole
        module.

        The cells are totalled in a plain loop rather than with the builtin
        ``sum``, and that is load-bearing rather than fussy. Since Python 3.12
        ``sum`` compensates its rounding over a run of floats, and numpy does
        not, so the obvious spelling of this four-term total disagreed with
        :meth:`FeatureBank.values_over` on 1 of 200 measured features by
        2.2e-16 -- and a stage's threshold is calibrated to sit exactly on one
        of these values, which is precisely where a difference in the last bit
        decides an inequality. The two routes have to round identically, so the
        accurate one is the wrong one.

        Raises
        ------
        ShapeMismatchError
            If the feature does not fit inside the window, or the window does
            not fit inside the picture.
        """
        total = 0.0
        for cell in self.cells_in(window):
            total = total + cell.sign * integral_image.sum_over(cell.rectangle)
        return total

    def __eq__(self, other: object) -> bool:
        # Both halves of the guard, the house idiom the foundation uses.
        # The exact-type check keeps a subclass carrying one more thing from
        # comparing equal to a plain one; the isinstance is what narrows
        # ``object`` so the attribute reads below are checkable.
        if not isinstance(other, RectangleFeature) or type(self) is not type(other):
            return NotImplemented
        return (
            self._arrangement is other._arrangement
            and self._top == other._top
            and self._left == other._left
            and self._cell_height == other._cell_height
            and self._cell_width == other._cell_width
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._arrangement,
                self._top,
                self._left,
                self._cell_height,
                self._cell_width,
            )
        )

    def __repr__(self) -> str:
        return (
            f"RectangleFeature({self._arrangement.value}, top={self._top}, "
            f"left={self._left}, cell={self._cell_height}x{self._cell_width})"
        )


class FeatureBank:
    """Every feature a window admits, or a drawn subset of them.

    Parameters
    ----------
    features:
        The features, all of which must fit inside the stated window.
    window_height, window_width:
        The window the features are written against.

    Raises
    ------
    EmptyValuesError
        If there are no features.
    InvalidValuesError
        If the window is not positive.
    ShapeMismatchError
        If any feature runs off the window.
    """

    __slots__ = ("_features", "_window_height", "_window_width")

    def __init__(
        self,
        features: Sequence[RectangleFeature],
        window_height: int,
        window_width: int,
    ) -> None:
        if window_height < 1 or window_width < 1:
            raise InvalidValuesError(
                f"a window has at least one pixel on each side, got "
                f"{window_height} by {window_width}"
            )
        held = tuple(features)
        if not held:
            raise EmptyValuesError("a feature bank holds at least one feature")
        for feature in held:
            if not feature.fits_in(window_height, window_width):
                raise ShapeMismatchError(
                    f"{feature!r} does not fit inside a {window_height} by "
                    f"{window_width} window"
                )
        self._features = held
        self._window_height = int(window_height)
        self._window_width = int(window_width)

    @staticmethod
    def count_available(window_height: int, window_width: int) -> int:
        """How many features a window of this size admits, without building them.

        The arithmetic rather than the enumeration, because the enumeration at
        a realistic window size is six figures of objects and the number is
        wanted more often than the objects are. For an arrangement of ``r`` by
        ``c`` cells, a cell size of ``h`` by ``w`` gives a feature ``rh`` tall
        and ``cw`` wide, which fits at ``(H - rh + 1)(W - cw + 1)`` positions;
        sum that over every cell size that fits at all.

        The two routes are pinned against each other in the spec, on windows
        small enough to enumerate.

        Raises
        ------
        InvalidValuesError
            If the window is not positive.
        """
        if window_height < 1 or window_width < 1:
            raise InvalidValuesError(
                f"a window has at least one pixel on each side, got "
                f"{window_height} by {window_width}"
            )
        total = 0
        for signs in CELL_SIGNS.values():
            rows_of_cells = len(signs)
            columns_of_cells = len(signs[0])
            down = sum(
                window_height - cell_height * rows_of_cells + 1
                for cell_height in range(1, window_height // rows_of_cells + 1)
            )
            across = sum(
                window_width - cell_width * columns_of_cells + 1
                for cell_width in range(1, window_width // columns_of_cells + 1)
            )
            total += down * across
        return total

    @classmethod
    def exhaustive(cls, window_height: int, window_width: int) -> FeatureBank:
        """Every arrangement, at every cell size, at every position.

        Raises
        ------
        InvalidValuesError
            If the window is not positive.
        EmptyValuesError
            If the window is too small to hold any feature at all, which needs
            at least two pixels on one side.
        """
        if window_height < 1 or window_width < 1:
            raise InvalidValuesError(
                f"a window has at least one pixel on each side, got "
                f"{window_height} by {window_width}"
            )
        features: list[RectangleFeature] = []
        for arrangement, signs in CELL_SIGNS.items():
            rows_of_cells = len(signs)
            columns_of_cells = len(signs[0])
            for cell_height in range(1, window_height // rows_of_cells + 1):
                for cell_width in range(1, window_width // columns_of_cells + 1):
                    height = cell_height * rows_of_cells
                    width = cell_width * columns_of_cells
                    for top in range(window_height - height + 1):
                        for left in range(window_width - width + 1):
                            features.append(
                                RectangleFeature(
                                    arrangement, top, left, cell_height, cell_width
                                )
                            )
        if not features:
            raise EmptyValuesError(
                f"a {window_height} by {window_width} window is too small to hold "
                "any rectangle feature at all"
            )
        return cls(features, window_height, window_width)

    def sample(self, n_features: int, generator: np.random.Generator) -> FeatureBank:
        """A drawn subset, in the same order the full bank held them.

        What a real implementation does for the same reason this one does: the
        exhaustive bank is far larger than any boosting run will look at, and
        a search over a random subset of it finds rules that are very nearly as
        good for a fraction of the arithmetic. Drawn without replacement and
        then re-sorted, so a seeded fit is reproducible and the bank's order
        still means what it meant.

        Raises
        ------
        InvalidValuesError
            If fewer than one feature is asked for, or more than there are.
        """
        if n_features < 1 or n_features > len(self._features):
            raise InvalidValuesError(
                f"a sample of a bank of {len(self._features)} features holds "
                f"between 1 and {len(self._features)} of them, got {n_features}"
            )
        drawn = np.sort(
            generator.choice(len(self._features), size=n_features, replace=False)
        )
        return FeatureBank(
            [self._features[int(position)] for position in drawn],
            self._window_height,
            self._window_width,
        )

    @property
    def window_height(self) -> int:
        """How tall the window these features are written against is."""
        return self._window_height

    @property
    def window_width(self) -> int:
        """How wide that window is."""
        return self._window_width

    def values_over(
        self, integral_images: Sequence[IntegralImage], window: Rectangle
    ) -> FloatArray:
        """Every feature's value on every one of these pictures.

        ``(n_pictures, n_features)``. The vectorised route, so that fitting can
        search thresholds without a Python loop over rows; it is written to
        make bit-identical arithmetic to
        :meth:`RectangleFeature.value_in`, and the spec pins the two together
        rather than trusting that.

        Raises
        ------
        EmptyValuesError
            If no pictures are supplied.
        ShapeMismatchError
            If the pictures differ in shape, or a feature runs off the window.
        """
        images = tuple(integral_images)
        if not images:
            raise EmptyValuesError("there are no pictures to read features from")
        shapes = {image.shape for image in images}
        if len(shapes) != 1:
            raise ShapeMismatchError(
                f"every picture read at once has the same shape, got {sorted(shapes)}"
            )

        tables = np.stack([image.table for image in images])
        values = np.empty((len(images), len(self._features)), dtype=np.float64)
        for position, feature in enumerate(self._features):
            total = np.zeros(len(images), dtype=np.float64)
            for cell in feature.cells_in(window):
                box = cell.rectangle
                total = total + cell.sign * (
                    tables[:, box.bottom, box.right]
                    - tables[:, box.top, box.right]
                    - tables[:, box.bottom, box.left]
                    + tables[:, box.top, box.left]
                )
            values[:, position] = total
        return values

    def __iter__(self) -> Iterator[RectangleFeature]:
        return iter(self._features)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, position: int) -> RectangleFeature:
        return self._features[position]

    def __repr__(self) -> str:
        return (
            f"FeatureBank({len(self._features)} features, window "
            f"{self._window_height}x{self._window_width})"
        )


class RuleDirection(StrEnum):
    """Which side of its threshold a rule votes yes on.

    A closed enum rather than a boolean called something like ``above``,
    because a boolean parameter at a call site reads as ``True`` and says
    nothing about what is true.
    """

    ABOVE = "above"
    """Vote yes when the feature's value is above the threshold."""

    BELOW = "below"
    """Vote yes when the feature's value is at or below the threshold."""


class ThresholdRule:
    """One feature, one threshold, one direction. The whole classifier.

    The simplest thing that can be called a classifier at all, and
    deliberately so: boosting's power comes from combining many learners each
    barely better than a coin, and a single rectangle feature compared against
    a number is the canonical one. It is what a decision stump is, written in
    the vocabulary of a window rather than of a column.

    Parameters
    ----------
    feature:
        What is measured.
    threshold:
        What the measurement is compared against.
    direction:
        Which side counts as a yes.

    Raises
    ------
    InvalidValuesError
        If the threshold is not finite.
    """

    __slots__ = ("_direction", "_feature", "_threshold")

    def __init__(
        self,
        feature: RectangleFeature,
        threshold: float,
        direction: RuleDirection,
    ) -> None:
        if not math.isfinite(threshold):
            raise InvalidValuesError(
                f"a rule's threshold is a finite number, got {threshold}"
            )
        self._feature = feature
        self._threshold = float(threshold)
        self._direction = direction

    @property
    def feature(self) -> RectangleFeature:
        """What the rule measures."""
        return self._feature

    @property
    def threshold(self) -> float:
        """What the measurement is compared against."""
        return self._threshold

    @property
    def direction(self) -> RuleDirection:
        """Which side of the threshold counts as a yes."""
        return self._direction

    def votes_for_value(self, value: float) -> bool:
        """Whether this measurement is a yes."""
        if self._direction is RuleDirection.ABOVE:
            return value > self._threshold
        return value <= self._threshold

    def votes_for_values(self, values: FloatArray) -> MaskArray:
        """The same decision over many measurements at once."""
        if self._direction is RuleDirection.ABOVE:
            return values > self._threshold
        return values <= self._threshold

    def votes_for(self, integral_image: IntegralImage, window: Rectangle) -> bool:
        """Whether this window is a yes, read through the integral image."""
        return self.votes_for_value(self._feature.value_in(integral_image, window))

    def __eq__(self, other: object) -> bool:
        # Both halves of the guard, the house idiom the foundation uses.
        # The exact-type check keeps a subclass carrying one more thing from
        # comparing equal to a plain one; the isinstance is what narrows
        # ``object`` so the attribute reads below are checkable.
        if not isinstance(other, ThresholdRule) or type(self) is not type(other):
            return NotImplemented
        return (
            self._feature == other._feature
            and self._threshold == other._threshold
            and self._direction is other._direction
        )

    def __hash__(self) -> int:
        return hash((self._feature, self._threshold, self._direction))

    def __repr__(self) -> str:
        comparison = ">" if self._direction is RuleDirection.ABOVE else "<="
        return f"ThresholdRule({self._feature!r} {comparison} {self._threshold:g})"


class WeightedRule:
    """A rule and how loudly it votes.

    Parameters
    ----------
    rule:
        The rule.
    voice:
        Its weight in the stage's vote, which boosting sets from how much
        better than guessing the rule was on the rows that were hard at the
        time. Always positive: a rule that would have earned a voice of zero or
        less ends the round instead of joining it.

    Raises
    ------
    InvalidValuesError
        If the voice is not a positive finite number.
    """

    __slots__ = ("_rule", "_voice")

    def __init__(self, rule: ThresholdRule, voice: float) -> None:
        if not math.isfinite(voice) or voice <= 0.0:
            raise InvalidValuesError(
                f"a rule that joins a stage votes, so its voice is positive and "
                f"finite; got {voice}"
            )
        self._rule = rule
        self._voice = float(voice)

    @property
    def rule(self) -> ThresholdRule:
        """The rule."""
        return self._rule

    @property
    def voice(self) -> float:
        """How loudly it votes."""
        return self._voice

    def __eq__(self, other: object) -> bool:
        # Both halves of the guard, the house idiom the foundation uses.
        # The exact-type check keeps a subclass carrying one more thing from
        # comparing equal to a plain one; the isinstance is what narrows
        # ``object`` so the attribute reads below are checkable.
        if not isinstance(other, WeightedRule) or type(self) is not type(other):
            return NotImplemented
        return self._rule == other._rule and self._voice == other._voice

    def __hash__(self) -> int:
        return hash((self._rule, self._voice))

    def __repr__(self) -> str:
        return f"WeightedRule({self._rule!r}, voice={self._voice:.4f})"


class CascadeStage:
    """A weighted vote of rules, and how much of the voice it demands.

    Parameters
    ----------
    weighted_rules:
        The rules and their voices, in the order boosting found them.
    acceptance_share:
        The share of the total voice a window has to attract to survive.
        :data:`DEFAULT_ACCEPTANCE_SHARE` is what a plain boosted vote demands;
        a cascade stage lowers it until it keeps every positive, which is what
        makes a stage a *filter* rather than a classifier.

    Raises
    ------
    EmptyValuesError
        If there are no rules.
    InvalidValuesError
        If the share does not lie in ``[0, 1]``.
    """

    __slots__ = ("_acceptance_share", "_weighted_rules")

    def __init__(
        self,
        weighted_rules: Sequence[WeightedRule],
        acceptance_share: float = DEFAULT_ACCEPTANCE_SHARE,
    ) -> None:
        held = tuple(weighted_rules)
        if not held:
            raise EmptyValuesError("a stage votes, so it holds at least one rule")
        if not math.isfinite(acceptance_share) or not 0.0 <= acceptance_share <= 1.0:
            raise InvalidValuesError(
                f"a stage demands a share of the voice, which lies between zero "
                f"and one; got {acceptance_share}"
            )
        self._weighted_rules = held
        self._acceptance_share = float(acceptance_share)

    @property
    def weighted_rules(self) -> tuple[WeightedRule, ...]:
        """The rules and their voices, in the order boosting found them."""
        return self._weighted_rules

    @property
    def n_rules(self) -> int:
        """How many rules a window costs to pass through this stage.

        Every rule of a stage is evaluated, so this is the stage's whole cost
        and the number :meth:`HaarCascade.scan` accumulates.
        """
        return len(self._weighted_rules)

    @property
    def total_voice(self) -> float:
        """The sum of the rules' voices."""
        return float(sum(one.voice for one in self._weighted_rules))

    @property
    def acceptance_share(self) -> float:
        """The share of the total voice a window has to attract."""
        return self._acceptance_share

    def with_acceptance_share(self, acceptance_share: float) -> CascadeStage:
        """The same rules, demanding a different share.

        Calibration is a change to the threshold and not to the rules, and
        making it a new object rather than a setter keeps a stage immutable
        like everything else here.
        """
        return CascadeStage(self._weighted_rules, acceptance_share)

    def confidence_in(self, integral_image: IntegralImage, window: Rectangle) -> float:
        """The share of the voice this window attracts, in ``[0, 1]``.

        Raises
        ------
        ShapeMismatchError
            If the window does not fit inside the picture.
        """
        agreeing = sum(
            one.voice
            for one in self._weighted_rules
            if one.rule.votes_for(integral_image, window)
        )
        return float(agreeing / self.total_voice)

    def accepts(self, integral_image: IntegralImage, window: Rectangle) -> bool:
        """Whether this window survives the stage.

        Compared with :data:`ACCEPTANCE_TOLERANCE` of room, because a
        calibrated threshold sits exactly on the confidence of the hardest
        positive and the two routes to that confidence can part in the last
        bits.

        Raises
        ------
        ShapeMismatchError
            If the window does not fit inside the picture.
        """
        return (
            self.confidence_in(integral_image, window)
            >= self._acceptance_share - ACCEPTANCE_TOLERANCE
        )

    def __repr__(self) -> str:
        return (
            f"CascadeStage({len(self._weighted_rules)} rules, "
            f"share={self._acceptance_share:.4f})"
        )


class StageOutcome:
    """What one stage did to the training rows, recorded as it was fitted.

    Parameters
    ----------
    stage_number:
        Which stage, counting from one.
    negatives_reaching:
        How many training negatives were still alive when this stage was fitted.
    negatives_rejected:
        How many of them it threw away.
    positives_kept:
        How many of the training positives survived it. Calibration is aimed at
        making this the whole of them, and reading it is how a caller sees
        whether the aim was met.
    """

    __slots__ = (
        "_negatives_reaching",
        "_negatives_rejected",
        "_positives_kept",
        "_stage_number",
    )

    def __init__(
        self,
        stage_number: int,
        negatives_reaching: int,
        negatives_rejected: int,
        positives_kept: int,
    ) -> None:
        self._stage_number = int(stage_number)
        self._negatives_reaching = int(negatives_reaching)
        self._negatives_rejected = int(negatives_rejected)
        self._positives_kept = int(positives_kept)

    @property
    def stage_number(self) -> int:
        """Which stage, counting from one."""
        return self._stage_number

    @property
    def negatives_reaching(self) -> int:
        """How many training negatives were still alive at this stage."""
        return self._negatives_reaching

    @property
    def negatives_rejected(self) -> int:
        """How many of them the stage threw away."""
        return self._negatives_rejected

    @property
    def negatives_surviving(self) -> int:
        """How many were passed on to the next stage."""
        return self._negatives_reaching - self._negatives_rejected

    @property
    def positives_kept(self) -> int:
        """How many training positives survived the stage."""
        return self._positives_kept

    @property
    def rejection_share(self) -> float:
        """The share of what reached it that the stage threw away."""
        return self._negatives_rejected / self._negatives_reaching

    def __repr__(self) -> str:
        return (
            f"StageOutcome(stage={self._stage_number}, "
            f"rejected {self._negatives_rejected} of {self._negatives_reaching})"
        )


class CascadeScan:
    """What a sweep over a picture found, and what it cost to find it.

    This is the object the whole method is argued from. A detector is only
    worth the name if the effort it spends is far below what the same rules
    would cost applied everywhere, and the way to know that is to count rather
    than to reason about it.

    Parameters
    ----------
    accepted:
        The windows that survived every stage.
    windows_examined:
        How many window positions were looked at at all.
    windows_reaching:
        How many windows entered each stage, in stage order. The first entry is
        every window; the last is whatever was still alive by then, and the
        shape of this sequence is the cascade's entire contribution.
    rule_evaluations:
        How many rules were actually evaluated.
    rules_in_the_cascade:
        How many rules the cascade holds in all, so that the cost of the same
        rules with no cascade can be stated beside the cost of the cascade.
    """

    __slots__ = (
        "_accepted",
        "_rule_evaluations",
        "_rules_in_the_cascade",
        "_windows_examined",
        "_windows_reaching",
    )

    def __init__(
        self,
        accepted: Sequence[Rectangle],
        windows_examined: int,
        windows_reaching: Sequence[int],
        rule_evaluations: int,
        rules_in_the_cascade: int,
    ) -> None:
        self._accepted = tuple(accepted)
        self._windows_examined = int(windows_examined)
        self._windows_reaching = tuple(int(one) for one in windows_reaching)
        self._rule_evaluations = int(rule_evaluations)
        self._rules_in_the_cascade = int(rules_in_the_cascade)

    @property
    def accepted(self) -> tuple[Rectangle, ...]:
        """The windows that survived every stage."""
        return self._accepted

    @property
    def n_accepted(self) -> int:
        """How many windows survived."""
        return len(self._accepted)

    @property
    def windows_examined(self) -> int:
        """How many window positions were looked at."""
        return self._windows_examined

    @property
    def windows_reaching(self) -> tuple[int, ...]:
        """How many windows entered each stage, in stage order."""
        return self._windows_reaching

    @property
    def rule_evaluations(self) -> int:
        """How many rules were evaluated in all."""
        return self._rule_evaluations

    @property
    def rule_evaluations_without_the_cascade(self) -> int:
        """What the same rules would have cost run on every window.

        The honest comparison for the *ordering*, holding the rules fixed. The
        other comparison, against evaluating a whole feature bank everywhere,
        is larger by orders of magnitude and is what
        :meth:`FeatureBank.count_available` is for.
        """
        return self._windows_examined * self._rules_in_the_cascade

    @property
    def saving(self) -> float:
        """How many times cheaper the cascade was than running every stage."""
        return self.rule_evaluations_without_the_cascade / self._rule_evaluations

    def __repr__(self) -> str:
        return (
            f"CascadeScan({self.n_accepted} accepted of "
            f"{self._windows_examined} windows, "
            f"{self._rule_evaluations} rule evaluations)"
        )


class HaarCascade(Classifier[Sequence[Picture], Feature]):
    """A sequence of boosted stages, each rejecting most of what reaches it.

    Parameters
    ----------
    n_stages:
        How many stages to fit at most. Fewer are kept when the fit runs out of
        negatives to reject or out of rules worth adding, which
        ``n_stages_fitted`` reports.
    rules_per_stage:
        How many boosting rounds each stage runs. Small, because a stage is a
        filter rather than a classifier: its job is to be cheap and to throw
        away half of what it sees, and the accuracy comes from there being
        several of them.
    minimum_detection_rate:
        The share of training positives every stage has to keep. One by
        default, which is the setting that makes a cascade a cascade -- a stage
        that drops a positive has lost the target for good, since no later
        stage ever sees it again, whereas a negative it lets through only costs
        the next stage a little work.
    n_candidate_features:
        How many features to draw from the exhaustive bank before searching.
        ``None`` searches all of them, which is right for a small window and
        impossible for a realistic one.
    random_seed:
        Fixes the draw, so a fit is reproducible. The boosting itself is
        deterministic -- what changes between rounds is the row weights, not a
        resample -- so this matters only when the bank is sampled.

    Raises
    ------
    NotFittedError
        From any learned property, or from ``predict``, before ``fit``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_stages: int = Field(default=4, ge=1)
    rules_per_stage: int = Field(default=4, ge=1)
    minimum_detection_rate: float = Field(default=1.0, gt=0.0, le=1.0)
    n_candidate_features: int | None = Field(default=None, ge=1)
    random_seed: int | None = None

    _stages: tuple[CascadeStage, ...] | None = PrivateAttr(default=None)
    _stage_outcomes: tuple[StageOutcome, ...] | None = PrivateAttr(default=None)
    _candidate_features: FeatureBank | None = PrivateAttr(default=None)
    _window_shape: tuple[int, int] | None = PrivateAttr(default=None)

    @property
    def stages(self) -> tuple[CascadeStage, ...]:
        """The fitted stages, in the order a window meets them."""
        self._check_fitted()
        assert self._stages is not None
        return self._stages

    @property
    def n_stages_fitted(self) -> int:
        """How many stages the fit actually kept.

        Fewer than ``n_stages`` means the fit ran out of work: either no
        training negative was left to reject, or a stage rejected none of the
        ones that were, at which point every later stage would be a copy of it.
        """
        return len(self.stages)

    @property
    def n_rules(self) -> int:
        """How many rules the whole cascade holds."""
        return sum(stage.n_rules for stage in self.stages)

    @property
    def stage_outcomes(self) -> tuple[StageOutcome, ...]:
        """What each stage did to the training rows as it was fitted.

        Worth reading as a sequence rather than as a summary. The rejection
        shares should stay well away from zero all the way down: a stage that
        rejects nothing is one the boosting could not improve on, and a run of
        them means the cascade has stopped being a cascade.
        """
        self._check_fitted()
        assert self._stage_outcomes is not None
        return self._stage_outcomes

    @property
    def candidate_features(self) -> FeatureBank:
        """The features the fit searched over.

        Kept because the cost of *not* having a cascade is stated against it:
        the exhaustive alternative is this many features at every window
        position, and that number is the argument for the whole method.
        """
        self._check_fitted()
        assert self._candidate_features is not None
        return self._candidate_features

    @property
    def window_shape(self) -> tuple[int, int]:
        """``(height, width)`` of the window this cascade reads."""
        self._check_fitted()
        assert self._window_shape is not None
        return self._window_shape

    def fit(self, input_values: Sequence[Picture], target_values: Feature) -> Self:
        """Fit stage after stage, each on what the ones before it let through.

        Every stage is boosted on the *surviving* negatives rather than on all
        of them, which is the part that makes the later stages hard and the
        earlier ones cheap. A negative the first stage already rejects is not a
        negative the second one has to learn anything about, so each stage in
        turn faces a harder and smaller problem, and the rules it finds are
        correspondingly more particular.

        Parameters
        ----------
        input_values:
            The training windows, all the same size. That size becomes the
            cascade's window.
        target_values:
            One label per picture, ``1.0`` for a target and ``0.0`` for
            anything else.

        Returns
        -------
        Self
            This model, so calls can chain.

        Raises
        ------
        EmptyValuesError
            If no pictures are supplied.
        ShapeMismatchError
            If the pictures are not all the same size.
        NonEqualArrayLengthError
            If there is not exactly one label per picture.
        NonBinaryLabelsError
            If the labels are not all zero or one.
        SingleClassError
            If either class is missing.
        DivergenceError
            If not one stage could be fitted, which means no rule in the bank
            beat guessing on the very first round. An empty cascade that
            reported itself fitted would accept everything it was ever shown.
        """
        pictures = tuple(input_values)
        if not pictures:
            raise EmptyValuesError("a cascade is fitted on at least one window")

        shapes = {picture.shape for picture in pictures}
        if len(shapes) != 1:
            raise ShapeMismatchError(
                f"a cascade reads one window size, so every training picture is "
                f"the same shape; got {sorted(shapes)}"
            )
        height, width = pictures[0].shape

        labels = target_values.column
        if len(labels) != len(pictures):
            raise NonEqualArrayLengthError(
                f"there are {len(pictures)} training windows and {len(labels)} labels"
            )
        labels.check_is_binary()
        labels.check_has_both_classes()

        window = Rectangle(0, 0, height, width)
        bank = FeatureBank.exhaustive(height, width)
        if self.n_candidate_features is not None and self.n_candidate_features < len(
            bank
        ):
            bank = bank.sample(
                self.n_candidate_features, np.random.default_rng(self.random_seed)
            )

        integral_images = tuple(IntegralImage.of(picture) for picture in pictures)
        values = bank.values_over(integral_images, window)
        is_positive = np.asarray(labels.values) == 1.0
        positives = np.flatnonzero(is_positive)
        surviving = np.flatnonzero(~is_positive)

        stages: list[CascadeStage] = []
        outcomes: list[StageOutcome] = []

        for _ in range(self.n_stages):
            if surviving.size == 0:
                break

            rows = np.concatenate([positives, surviving])
            weighted_rules = self._boosted_rules(values[rows], is_positive[rows], bank)
            if not weighted_rules:
                break

            stage = self._calibrated(
                CascadeStage(weighted_rules), integral_images, window, positives
            )
            accepted = np.array(
                [stage.accepts(integral_images[int(row)], window) for row in surviving],
                dtype=bool,
            )
            kept = sum(
                stage.accepts(integral_images[int(row)], window) for row in positives
            )

            outcomes.append(
                StageOutcome(
                    len(stages) + 1,
                    int(surviving.size),
                    int((~accepted).sum()),
                    int(kept),
                )
            )
            stages.append(stage)
            surviving = surviving[accepted]

            if not (~accepted).any():
                # A stage that rejects nothing has been handed the same problem
                # its predecessor was, and every later stage would be a copy of
                # it. Stopping is honest where filling the cascade would not be.
                break

        if not stages:
            raise DivergenceError(
                "no rule in the bank beat guessing on the first round, so not "
                "one stage could be fitted and the cascade would accept "
                "everything. Draw more candidate features, or use windows the "
                "arrangements can actually distinguish"
            )

        self._stages = tuple(stages)
        self._stage_outcomes = tuple(outcomes)
        self._candidate_features = bank
        self._window_shape = (height, width)
        self._mark_fitted()
        return self

    def _boosted_rules(
        self,
        values: FloatArray,
        is_positive: MaskArray,
        bank: FeatureBank,
    ) -> tuple[WeightedRule, ...]:
        """Run the boosting rounds for one stage.

        Discrete AdaBoost over threshold rules, in the shape
        :class:`~oop_ml.numpy.classification.ensembles.adaboost_classifier.AdaBoostClassifier`
        already uses: fit the best rule under the current row weights, give it
        a voice from its weighted error, multiply the weight of every row it
        got wrong, and go again.

        The starting weights split half the total between the positives and
        half between the negatives rather than spreading it evenly over the
        rows. That matters here in a way it does not for a plain boosted
        classifier: by the third stage the surviving negatives are a handful
        against every positive, and even weights would let the first rule score
        well by answering "yes" to everything.
        """
        n_rows, n_features = values.shape
        n_positive = int(is_positive.sum())
        n_negative = n_rows - n_positive

        weights = np.where(is_positive, 0.5 / n_positive, 0.5 / n_negative).astype(
            np.float64
        )

        # Sorted once for the stage rather than once per round. What changes
        # between rounds is the weights, not the order, so the sweep that finds
        # the best threshold reuses this and costs one pass instead of a sort.
        order = np.argsort(values, axis=0, kind="stable")
        sorted_values = np.take_along_axis(values, order, axis=0)
        positive_sorted = is_positive[order]

        # A threshold only means something between two *different* values, and
        # the midpoint of two equal ones is that value, which would split
        # nothing.
        distinct = sorted_values[1:] > sorted_values[:-1]
        midpoints = (sorted_values[1:] + sorted_values[:-1]) / 2.0

        weighted_rules: list[WeightedRule] = []
        if not distinct.any():
            return ()

        for _ in range(self.rules_per_stage):
            weights = weights / weights.sum()
            total_weight = float(weights.sum())
            weight_sorted = weights[order]

            positive_running = np.cumsum(weight_sorted * positive_sorted, axis=0)
            negative_running = np.cumsum(weight_sorted * ~positive_sorted, axis=0)
            total_positive = positive_running[-1]

            # The error of "yes below the split": every positive above it is
            # missed, every negative at or below it is a false alarm. The other
            # direction is the complement, since between them the two
            # directions get every row exactly once.
            error_below = (total_positive - positive_running[:-1]) + negative_running[
                :-1
            ]
            error_above = total_weight - error_below

            errors = np.where(distinct, np.minimum(error_below, error_above), np.inf)
            flattened = int(np.argmin(errors))
            error = float(errors.flat[flattened])
            if not math.isfinite(error):
                break

            split, feature_position = np.unravel_index(flattened, errors.shape)
            direction = (
                RuleDirection.BELOW
                if error_below[split, feature_position]
                <= error_above[split, feature_position]
                else RuleDirection.ABOVE
            )
            rule = ThresholdRule(
                bank[int(feature_position)],
                float(midpoints[split, feature_position]),
                direction,
            )

            voice = self._voice_for(error, total_weight)
            if voice <= SILENT_RULE_VOICE:
                # No better than guessing, so it would drag the vote rather
                # than sharpen it. Tested on the voice rather than on the error
                # for the reason AdaBoost's own module records.
                break

            weighted_rules.append(WeightedRule(rule, voice))
            if error <= 0.0:
                # Nothing was got wrong, so nothing would be reweighted and
                # every later round would find this same rule again.
                break

            wrong = (
                rule.votes_for_values(values[:, int(feature_position)]) != is_positive
            )
            weights = weights * np.exp(voice * np.where(wrong, 1.0, -1.0))

        return tuple(weighted_rules)

    def _voice_for(self, error: float, total_weight: float) -> float:
        """``0.5 * log((1 - error) / error)``, on the weighted error.

        Positive exactly when the rule is wrong on less than half the weight,
        which for two classes is exactly when it beats guessing.
        """
        share = error / total_weight
        if share <= 0.0:
            return PERFECT_RULE_VOICE
        if share >= 1.0:
            return 0.0
        return float(0.5 * np.log((1.0 - share) / share))

    def _calibrated(
        self,
        stage: CascadeStage,
        integral_images: Sequence[IntegralImage],
        window: Rectangle,
        positives: IndexArray,
    ) -> CascadeStage:
        """Lower the stage's threshold until it keeps enough positives.

        Only ever lower. A stage demanding more than
        :data:`DEFAULT_ACCEPTANCE_SHARE` would be a *stricter* classifier than
        the boosting that built it, which is not what a filter is for, and the
        cascade's accuracy is supposed to come from the number of stages rather
        than from any one of them being severe.
        """
        confidences = np.sort(
            np.array(
                [
                    stage.confidence_in(integral_images[int(row)], window)
                    for row in positives
                ],
                dtype=np.float64,
            )
        )
        needed = max(1, math.ceil(self.minimum_detection_rate * confidences.size))
        required = float(confidences[confidences.size - needed])
        return stage.with_acceptance_share(min(DEFAULT_ACCEPTANCE_SHARE, required))

    def _window_for(self, picture: Picture) -> Rectangle:
        """The whole of ``picture`` as a window, checked against the fit's size.

        Raises
        ------
        ShapeMismatchError
            If the picture is not the size the cascade was fitted on.
        """
        height, width = self.window_shape
        if picture.shape != (height, width):
            raise ShapeMismatchError(
                f"this cascade reads a {height} by {width} window, got a "
                f"{picture.height} by {picture.width} picture. To look for the "
                "target inside a larger picture, call scan"
            )
        return Rectangle(0, 0, height, width)

    def _stages_survived(self, picture: Picture) -> int:
        """How many stages in a row this window got through.

        Raises
        ------
        ShapeMismatchError
            If the picture is not the size the cascade was fitted on.
        """
        window = self._window_for(picture)
        integral_image = IntegralImage.of(picture)
        survived = 0
        for stage in self.stages:
            if not stage.accepts(integral_image, window):
                break
            survived += 1
        return survived

    def predict(self, input_values: Sequence[Picture]) -> Predictions:
        """One label per window: ``1.0`` only if it survived every stage.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        ShapeMismatchError
            If a picture is not the size the cascade was fitted on.
        """
        self._check_fitted()
        n_stages = self.n_stages_fitted
        return Predictions.already_checked(
            np.array(
                [
                    1.0 if self._stages_survived(picture) == n_stages else 0.0
                    for picture in input_values
                ],
                dtype=np.float64,
            )
        )

    def predict_probability(self, input_values: Sequence[Picture]) -> Probabilities:
        """How far through the cascade each window got, as a share of the stages.

        **This is not a probability**, and the method is only called that
        because the frame calls it that -- the same position
        ``SupportVectorClassifier.predict_probability`` takes, and for a
        related reason. A cascade is a sequence of hard accept-or-reject
        decisions and no likelihood is computed anywhere in it. What comes back
        is the share of stages a window survived, which is a genuine ordering
        of how nearly a window was a target and is not calibrated to anything.
        A window rejected by the first of four stages scores 0.0 and one
        rejected by the last scores 0.75.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        ShapeMismatchError
            If a picture is not the size the cascade was fitted on.
        """
        self._check_fitted()
        n_stages = self.n_stages_fitted
        return Probabilities(
            np.array(
                [self._stages_survived(picture) / n_stages for picture in input_values],
                dtype=np.float64,
            )
        )

    def scan(self, picture: Picture, stride: int = 1) -> CascadeScan:
        """Slide the window over ``picture`` and report what it cost.

        One integral image is built for the whole scene and every window reads
        from it, which is the arrangement the whole module exists to make
        possible: a feature's value at a window position is four lookups per
        cell into a table that was accumulated once, however many positions
        there are and however large the feature's cells happen to be.

        Parameters
        ----------
        picture:
            The scene to look in, at least as large as the fitted window.
        stride:
            How far the window moves between positions. One looks everywhere;
            larger values trade positions for time and are what a real detector
            does at coarse scales.

        Returns
        -------
        CascadeScan
            The surviving windows, and the counts that say what the cascade
            saved.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the stride is below one.
        ShapeMismatchError
            If the picture is smaller than the fitted window.
        """
        self._check_fitted()
        if stride < 1:
            raise InvalidValuesError(
                f"a scan moves the window at least one pixel at a time, got "
                f"stride {stride}"
            )
        height, width = self.window_shape
        if picture.height < height or picture.width < width:
            raise ShapeMismatchError(
                f"a {height} by {width} window does not fit inside a "
                f"{picture.height} by {picture.width} picture"
            )

        integral_image = IntegralImage.of(picture)
        stages = self.stages
        reaching = [0] * len(stages)
        rule_evaluations = 0
        windows_examined = 0
        accepted: list[Rectangle] = []

        for top in range(0, picture.height - height + 1, stride):
            for left in range(0, picture.width - width + 1, stride):
                window = Rectangle(top, left, height, width)
                windows_examined += 1
                survived = True
                for position, stage in enumerate(stages):
                    reaching[position] += 1
                    rule_evaluations += stage.n_rules
                    if not stage.accepts(integral_image, window):
                        survived = False
                        break
                if survived:
                    accepted.append(window)

        return CascadeScan(
            accepted, windows_examined, reaching, rule_evaluations, self.n_rules
        )

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"HaarCascade(n_stages={self.n_stages}, unfitted)"
        height, width = self.window_shape
        return (
            f"HaarCascade({self.n_stages_fitted} stages, {self.n_rules} rules, "
            f"window {height}x{width})"
        )
