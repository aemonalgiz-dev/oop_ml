"""A patch described by which way its edges point, not by what its pixels are.

Why a count of directions
-------------------------
:mod:`oop_ml.core.computer_vision.edges` makes the case for gradients over
brightness: turn the lamp up and every pixel of a scene changes, while the
directions of its edges do not move at all. That argument produces a gradient
per pixel, which is two numbers per pixel where there was one, so nothing has
been described yet -- a 64 by 128 patch still needs 16,384 numbers, and two
photographs of the same person a pixel apart still disagree in every one of
them.

The oriented histogram is the step that turns a gradient field into a
*description*. Divide the patch into small square cells; inside a cell, throw
away where each edge was and keep only which way it pointed, weighted by how
sharp it was. A cell becomes a handful of numbers saying "mostly vertical
edges here, and strongly", and a person who moves two pixels within the cell
changes it by nothing at all. Dalal and Triggs (2005) built pedestrian
detection on exactly this, and the shape of their descriptor -- 8 by 8 pixel
cells, 2 by 2 cell blocks, 9 unsigned buckets -- is what the defaults here
are.

The three stages, and what each one buys
----------------------------------------
1. **Vote.** Every pixel votes for a direction bucket with a weight equal to
   its gradient magnitude. This buys tolerance to *position*: a shape may
   wander inside its cell and the histogram does not move. What it costs is
   that the histogram no longer knows where anything was, which is why the
   cells are small.
2. **Group into blocks.** Neighbouring cells are collected into a block and
   the block is normalised as one vector. This buys tolerance to *lighting*.
   Multiplying every pixel by a constant multiplies every magnitude by the
   same constant, and dividing a block by its own length cancels it exactly.

   The two halves of a lighting change are bought at different stages, which
   is worth separating because only one of them is what normalisation is for.
   *Adding* a constant is already free before this stage: the gradient
   operator's weights sum to zero, so an unnormalised description of the
   specs' fixture moves by 8e-14 when 12 is added to every pixel.
   *Multiplying* is not: the same unnormalised description moves by 521.0,
   against its own length of 193.0, when every pixel is multiplied by 3.7.
   Under any of the four normalisations both moves come back at around
   1e-16.
3. **Overlap the blocks.** The blocks step one cell at a time, so a cell in
   the middle of a patch is normalised four times over, once in each block it
   belongs to, and appears four times in the answer under four different
   denominators. That looks wasteful and is the part that works: a cell beside
   a strong edge is dimmed by its bright neighbour in one block and left alone
   in another, and both readings survive into the description. Dalal and
   Triggs measured the overlap as worth several points of detection rate on
   its own. The cost is countable: a 3 by 3 grid of cells holds 9 cells and
   its description holds 16 cells' worth of numbers, and the 64 by 128 patch
   the pedestrian detector uses holds 128 cells and describes them in 420.

Normalisation is a *choice* rather than a fixed step, so
:class:`BlockNormalisation` includes ``NONE``. Without it the layout of the
answer is identical and only the division is gone, which makes the lighting
claim measurable rather than assertable: the specs compute the same
description both ways and report both numbers.

Signed or unsigned
------------------
A dark-to-light edge and a light-to-dark edge are geometrically the same edge,
and their gradients point exactly opposite ways. Folding the circle in half,
so that a direction and its reverse land in one bucket, is
:attr:`AngleRange.UNSIGNED`, and it is what the pedestrian work used: a person
in a dark coat against a bright wall and the same person against a dark wall
produce reversed gradients along the same silhouette, and an unsigned
histogram cannot tell them apart, which is the point. Keeping the whole circle
is :attr:`AngleRange.SIGNED`, and it is the better choice when the direction
of the contrast is itself information -- a light-on-dark bar is not a
dark-on-light bar for a reader of printed text.

The two genuinely differ, and the fixture that shows it is a bright bar on a
dark ground. Its two sides are one edge and its reverse, so unsigned puts all
the weight in a single bucket and signed splits it into two, half a circle
apart. Measured on the specs' bar, unsigned occupies 1 bucket of 9 and signed
2 of 18 -- centred at 10 and 190 degrees -- and the fraction in the fullest
bucket falls from 1.0 to 0.5. The sharper measurement is the one the choice is
actually about: a bright bar and the same bar with its brightness inverted
have descriptions exactly 0.0 apart under unsigned and 1.4142 apart under
signed, which is the largest two unit blocks can be.

Splitting a vote between two buckets
------------------------------------
A vote dropped whole into whichever bucket its direction falls in makes the
description *jump*. Two patches whose edges differ by a fifth of a degree, one
either side of a bucket boundary, get descriptions with no weight in common
for that cell. Sharing the vote between the two nearest bucket centres in
proportion to how near it is to each removes the jump: the weight slides from
one bucket to the next as the direction turns.

Both rules are implemented, because the difference is worth pinning rather
than describing. Measured on a linear ramp, whose gradient direction is the
same at every interior pixel and exactly what it was drawn to be, turned from
19 degrees to 21 degrees across the boundary between the first two buckets: an
interior cell's histogram moves 724.08 under
:attr:`VoteSharing.WHOLE_TO_NEAREST` and 72.41 under
:attr:`VoteSharing.SPLIT_BETWEEN_NEIGHBOURS`, a ratio of exactly 10 to within
4e-14. The whole of the cell's weight changes bucket in the first case and a
tenth of it in the second, and a tenth is what two degrees out of twenty ought
to move.

The interpolation is not free, and what it costs surprises people, so the
specs pin it. A direction sitting exactly on a bucket *centre* is unaffected,
but one sitting exactly between two centres is split half and half. So the
horizontal edge, whose gradient points straight down at 90 degrees, lands
entirely in bucket 4 of 9 either way -- 90 degrees is that bucket's centre --
while the vertical edge, whose gradient points along 0 degrees, reads as one
full bucket under the whole-vote rule and as two half-full buckets, 0 and 8,
under interpolation. Both are right. Zero is exactly as far from the centre of
the first bucket as from the centre of the last, and under an unsigned range
the last bucket really is the first one's neighbour.

Only the direction is interpolated here, not the position. Dalal and Triggs
also share each vote between neighbouring *cells*, which removes the same jump
in space that this removes in angle. It is absent rather than hidden, and it
is the obvious next thing.

Where it stops working
----------------------
Rotation. Every stage above is built on the direction being measured from a
fixed axis, so turning the patch turns every direction with it and the
histogram permutes -- and the cells move too, which no bucket arithmetic can
undo. Measured on the specs' fixture at the defaults, a quarter turn moves the
description by 1.4832, in a vector whose own length is 2.0, where a change of
lighting on the same fixture moves it by 9.2e-16. Fifteen orders of magnitude
separate the thing the method is invariant to from the thing it is not, and
that pair is the honest summary of it. It is also why the family that followed
(SIFT and its descendants) begins by measuring a patch's dominant direction
and turning the patch to face it before doing any of this.

Scale, for the same reason: a cell is a fixed number of pixels, so the same
object twice as far away fills a quarter as many cells and produces a
different-length answer. The usual repair is outside this module -- run the
whole thing over a pyramid of resized pictures and keep the best answer.

And the patch has to be an exact multiple of the cell size, which is enforced
rather than rounded away. A patch that does not divide leaves a strip of
pixels that either vote in a cell of a different size or do not vote at all,
and both are silent changes to the meaning of a number in the answer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

import numpy as np
from numpy.typing import DTypeLike
from pydantic import BaseModel, ConfigDict, Field, model_validator

from oop_ml.core.computer_vision.edges import GradientField, GradientOperator
from oop_ml.core.computer_vision.filtering import EdgeRule
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.types import FloatArray, array_for_protocol

CLIP_LIMIT = 0.2
"""How large one entry of a unit block may be before it is cut back.

A module constant rather than a field, because it is read by exactly one
member of :class:`BlockNormalisation` and a field whose meaning depends on the
value of another field is the bag of keyword arguments this library refuses
elsewhere. The value is Dalal and Triggs's, who report the detection rate to
be insensitive across roughly 0.1 to 0.3.

It is a limit on the way in and not on the way out, which reads as a bug and
is not one. The block is made a unit vector *again* after the cut, so an entry
that was clipped to 0.2 is then divided by a length that the clipping made
smaller, and comes back above 0.2. Measured on a patch holding one single
bright pixel, the largest entry of the description goes from 0.4851 under
plain ``L2`` to 0.3052 under ``L2_CLIPPED``: cut back, and still the largest.
What the step buys is the *ratio* between a block's loudest direction and its
quietest, not a ceiling.
"""


class AngleRange(StrEnum):
    """Whether a direction and its reverse are one direction or two."""

    UNSIGNED = "unsigned"
    """Half a circle. A dark-to-light edge and its reverse share a bucket."""

    SIGNED = "signed"
    """The whole circle. Which side is bright is kept."""

    @property
    def span(self) -> float:
        """How much angle, in radians, the buckets divide between them."""
        return np.pi if self is AngleRange.UNSIGNED else 2.0 * np.pi


class VoteSharing(StrEnum):
    """Whether a pixel's vote goes to one bucket or is split between two."""

    WHOLE_TO_NEAREST = "whole_to_nearest"
    """All of it into the bucket the direction falls in. Jumps at a boundary."""

    SPLIT_BETWEEN_NEIGHBOURS = "split_between_neighbours"
    """Shared between the two nearest bucket centres, in proportion."""


class BlockNormalisation(StrEnum):
    """How a block of neighbouring cells is rescaled, if at all.

    A closed enum for :class:`~oop_ml.core.distance.metric.DistanceMetric`'s
    reason: none of them takes a parameter, so there is nothing for an object
    to hold, and a misspelled string should be a type error rather than a
    silently unnormalised answer.
    """

    NONE = "none"
    """Left alone. The same layout, so the difference is measurable."""

    L1 = "l1"
    """Divided by the sum of its entries, so the block sums to one."""

    L1_SQUARE_ROOT = "l1_square_root"
    """``L1`` and then square-rooted, which pulls the large entries down."""

    L2 = "l2"
    """Divided by its own length, so the block is a unit vector."""

    L2_CLIPPED = "l2_clipped"
    """``L2``, cut back at :data:`CLIP_LIMIT`, then made a unit vector again.

    Dalal and Triggs's ``L2-Hys``, and the default. One extremely sharp edge
    inside a block otherwise dominates its cell's whole entry; cutting the
    large entries back and renormalising lets the weaker directions in the
    block keep some say.
    """


def votes_by_bucket(
    directions: FloatArray,
    magnitudes: FloatArray,
    n_buckets: int,
    angle_range: AngleRange = AngleRange.UNSIGNED,
    vote_sharing: VoteSharing = VoteSharing.SPLIT_BETWEEN_NEIGHBOURS,
) -> FloatArray:
    """How much weight each position puts into each direction bucket.

    Answers the arguments' shape with one more axis of length ``n_buckets``.
    Bucket ``i`` covers ``[i * width, (i + 1) * width)`` of the range and is
    centred at ``(i + 0.5) * width``; sharing is between *centres*, and wraps
    around, which for an unsigned range means the last bucket is a neighbour
    of the first because half a circle later is the same direction.

    Parameters
    ----------
    directions:
        Which way the brightness rises, in radians. Any range; it is folded.
    magnitudes:
        How sharply, which is the weight of the vote. Same shape.
    n_buckets:
        How many buckets the range is divided into. At least two.
    angle_range:
        Whether a direction and its reverse are one direction.
    vote_sharing:
        Whether the vote is split between the two nearest centres.

    Raises
    ------
    InvalidValuesError
        If ``n_buckets`` is below two, which leaves one bucket that every
        direction falls into and so describes nothing.
    ShapeMismatchError
        If the two arrays do not describe the same positions.
    """
    if n_buckets < 2:
        raise InvalidValuesError(
            f"a histogram of directions needs at least two buckets to say "
            f"anything, got {n_buckets}"
        )
    if directions.shape != magnitudes.shape:
        raise ShapeMismatchError(
            f"a direction and the weight of its vote belong to one position, "
            f"got {directions.shape} directions and {magnitudes.shape} weights"
        )

    bucket_width = angle_range.span / n_buckets
    position = np.mod(directions, angle_range.span) / bucket_width

    votes = np.zeros(directions.shape + (n_buckets,), dtype=np.float64)
    if vote_sharing is VoteSharing.WHOLE_TO_NEAREST:
        fallen_in = np.mod(np.floor(position).astype(np.intp), n_buckets)
        np.put_along_axis(
            votes, fallen_in[..., np.newaxis], magnitudes[..., np.newaxis], axis=-1
        )
        return votes

    from_lower_centre = position - 0.5
    lower = np.floor(from_lower_centre)
    share_of_upper = from_lower_centre - lower
    lower_bucket = np.mod(lower.astype(np.intp), n_buckets)
    upper_bucket = np.mod(lower_bucket + 1, n_buckets)

    np.put_along_axis(
        votes,
        lower_bucket[..., np.newaxis],
        (magnitudes * (1.0 - share_of_upper))[..., np.newaxis],
        axis=-1,
    )
    np.put_along_axis(
        votes,
        upper_bucket[..., np.newaxis],
        (magnitudes * share_of_upper)[..., np.newaxis],
        axis=-1,
    )
    return votes


def normalised_block(block: FloatArray, rule: BlockNormalisation) -> FloatArray:
    """One block's entries, rescaled by ``rule``.

    A block with no weight anywhere -- a patch of flat brightness produces
    them -- comes back as zeros rather than as a division by nothing, which is
    the rule :meth:`~oop_ml.core.computer_vision.picture.Picture.rescaled_to_unit`
    already takes for a picture with no variation.
    """
    if rule is BlockNormalisation.NONE:
        return block

    if rule in (BlockNormalisation.L1, BlockNormalisation.L1_SQUARE_ROOT):
        total = float(np.abs(block).sum())
        if total == 0.0:
            return np.zeros_like(block)
        shares = block / total
        if rule is BlockNormalisation.L1:
            return shares
        return np.sqrt(shares)

    length = float(np.linalg.norm(block))
    if length == 0.0:
        return np.zeros_like(block)
    unit = block / length
    if rule is BlockNormalisation.L2:
        return unit

    clipped = np.minimum(unit, CLIP_LIMIT)
    clipped_length = float(np.linalg.norm(clipped))
    if clipped_length == 0.0:
        return np.zeros_like(block)
    return clipped / clipped_length


class CellHistograms:
    """One direction histogram per cell, before any block has normalised it.

    Exposed rather than kept private because it is where the method is
    readable: a cell's histogram can be checked against the geometry of a
    hand-drawn edge, and the finished description, which has been normalised
    four times over and concatenated, cannot.

    Parameters
    ----------
    counts:
        ``(n_cell_rows, n_cell_columns, n_buckets)`` of vote weight. Copied
        and frozen.
    angle_range:
        Which range the buckets divide, so a bucket can name its own angle.

    Raises
    ------
    InvalidValuesError
        If the counts are not a finite three-dimensional array, or hold fewer
        than two buckets, or hold a negative weight.
    """

    __slots__ = ("_angle_range", "_counts")

    def __init__(self, counts: FloatArray, angle_range: AngleRange) -> None:
        try:
            as_array = np.asarray(counts, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError("cell histograms hold numbers") from error

        if as_array.ndim != 3:
            raise InvalidValuesError(
                f"cell histograms are (rows, columns, buckets), got shape "
                f"{as_array.shape}"
            )
        if as_array.shape[2] < 2:
            raise InvalidValuesError(
                f"a histogram of directions needs at least two buckets, got "
                f"{as_array.shape[2]}"
            )
        if not np.all(np.isfinite(as_array)):
            raise InvalidValuesError("cell histograms hold finite weights")
        if bool(np.any(as_array < 0.0)):
            raise InvalidValuesError(
                "a vote is weighted by a gradient magnitude, which is never "
                "negative, so no bucket can hold a negative weight"
            )

        frozen = as_array.copy()
        frozen.setflags(write=False)

        self._counts = frozen
        self._angle_range = angle_range

    @property
    def counts(self) -> FloatArray:
        """The weights, frozen. Copy it before writing to it."""
        return self._counts

    @property
    def angle_range(self) -> AngleRange:
        """Which range the buckets divide between them."""
        return self._angle_range

    @property
    def n_cell_rows(self) -> int:
        """How many cells down."""
        return int(self._counts.shape[0])

    @property
    def n_cell_columns(self) -> int:
        """How many cells across."""
        return int(self._counts.shape[1])

    @property
    def n_buckets(self) -> int:
        """How many direction buckets each cell holds."""
        return int(self._counts.shape[2])

    @property
    def bucket_width(self) -> float:
        """How much angle, in radians, one bucket covers."""
        return self._angle_range.span / self.n_buckets

    @property
    def total_weight(self) -> float:
        """Every vote in every cell, added up."""
        return float(self._counts.sum())

    def bucket_centre(self, bucket: int) -> float:
        """The direction, in radians, that bucket ``bucket`` is centred on.

        Raises
        ------
        InvalidValuesError
            If there is no such bucket.
        """
        if bucket < 0 or bucket >= self.n_buckets:
            raise InvalidValuesError(
                f"this histogram has buckets 0 to {self.n_buckets - 1}, "
                f"asked for {bucket}"
            )
        return (bucket + 0.5) * self.bucket_width

    def histogram_at(self, row: int, column: int) -> FloatArray:
        """One cell's histogram, frozen.

        Raises
        ------
        InvalidValuesError
            If there is no such cell.
        """
        if not 0 <= row < self.n_cell_rows or not 0 <= column < self.n_cell_columns:
            raise InvalidValuesError(
                f"this grid is {self.n_cell_rows} cells by {self.n_cell_columns}, "
                f"asked for row {row}, column {column}"
            )
        return self._counts[row, column]

    def fullest_bucket_at(self, row: int, column: int) -> int:
        """Which bucket one cell put the most weight in.

        Ties go to the lower bucket, which is ``argmax``'s own rule and is
        stated because a cell with no weight at all is entirely tied and so
        answers zero rather than refusing.
        """
        return int(np.argmax(self.histogram_at(row, column)))

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._counts, dtype, copy)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CellHistograms) or type(self) is not type(other):
            return NotImplemented
        return self._angle_range is other._angle_range and bool(
            np.array_equal(self._counts, other._counts)
        )

    def __hash__(self) -> int:
        return hash((self._angle_range, self._counts.tobytes()))

    def __repr__(self) -> str:
        return (
            f"CellHistograms(n_cell_rows={self.n_cell_rows}, "
            f"n_cell_columns={self.n_cell_columns}, n_buckets={self.n_buckets})"
        )


class OrientedGradientDescription:
    """A patch's finished description: every normalised block, end to end.

    A value object rather than a bare array for the reason
    :class:`~oop_ml.core.data.column.Column` gives. What a caller does with a
    description is compare it with another one, and two descriptions built
    under different settings have different lengths and different meanings per
    position; a type that refuses to measure the distance between two of those
    turns a silent nonsense into a refusal that names the problem.

    Parameters
    ----------
    values:
        The description, one number per bucket per cell per block. Copied and
        frozen.

    Raises
    ------
    InvalidValuesError
        If the values are not a finite one-dimensional array.
    EmptyValuesError
        If there are none.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Any) -> None:
        try:
            as_array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError("a description is a vector of numbers") from error

        if as_array.ndim != 1:
            raise InvalidValuesError(
                f"a description is one vector, so one dimension; got shape "
                f"{as_array.shape}"
            )
        if as_array.size == 0:
            raise InvalidValuesError("a description holds at least one number")
        if not np.all(np.isfinite(as_array)):
            raise InvalidValuesError("a description holds finite numbers")

        frozen = as_array.copy()
        frozen.setflags(write=False)
        self._values = frozen

    @property
    def values(self) -> FloatArray:
        """The description, frozen. Copy it before writing to it."""
        return self._values

    @property
    def n_values(self) -> int:
        """How long the description is."""
        return int(self._values.size)

    @property
    def length(self) -> float:
        """The description's own Euclidean length, for scaling a distance."""
        return float(np.linalg.norm(self._values))

    def distance_to(self, other: OrientedGradientDescription) -> float:
        """How far this description is from ``other``, Euclidean.

        Raises
        ------
        ShapeMismatchError
            If the two are not the same length, which means they were built
            under different settings and their positions do not correspond.
        """
        if self.n_values != other.n_values:
            raise ShapeMismatchError(
                f"two descriptions can only be compared position by position, "
                f"and these are {self.n_values} and {other.n_values} long, so "
                f"they were built under different settings"
            )
        return float(np.linalg.norm(self._values - other._values))

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._values, dtype, copy)

    def __len__(self) -> int:
        return self.n_values

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrientedGradientDescription) or type(self) is not type(
            other
        ):
            return NotImplemented
        return bool(np.array_equal(self._values, other._values))

    def __hash__(self) -> int:
        return hash(self._values.tobytes())

    def __repr__(self) -> str:
        return f"OrientedGradientDescription(n_values={self.n_values})"


class HistogramOfOrientedGradients(BaseModel):
    """Dalal and Triggs's description of a patch, as a configured object.

    Construction configures and :meth:`describe` reads a picture; nothing is
    learned, so this is not a
    :class:`~oop_ml.core.base.estimator.Fittable`. Every default is the
    pedestrian detector's: 8 by 8 pixel cells, 2 by 2 cell blocks stepping one
    cell at a time, 9 unsigned buckets, and ``L2-Hys`` normalisation.

    Parameters
    ----------
    cell_side:
        How many pixels on each side of a cell. The patch must divide by it.
    cells_per_block_side:
        How many cells on each side of a block.
    block_stride_in_cells:
        How far a block steps. Below ``cells_per_block_side`` the blocks
        overlap, which is the point; equal to it they tile; above it they
        would skip cells altogether and it is refused.
    n_buckets:
        How many direction buckets a cell holds. At least two.
    angle_range:
        Whether a direction and its reverse share a bucket.
    vote_sharing:
        Whether a vote is split between the two nearest bucket centres.
    block_normalisation:
        How a block is rescaled. ``NONE`` keeps the layout and drops the
        division, so the lighting claim can be measured both ways.
    operator:
        Which estimate of the gradient to sweep.
    edge_rule:
        What the sweep reads outside the picture. ``KEEP_VALID`` is refused,
        because it answers a smaller field than the picture and every cell
        here is a fixed block of the picture's own pixels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cell_side: int = Field(default=8, ge=1)
    cells_per_block_side: int = Field(default=2, ge=1)
    block_stride_in_cells: int = Field(default=1, ge=1)
    n_buckets: int = Field(default=9, ge=2)
    angle_range: AngleRange = AngleRange.UNSIGNED
    vote_sharing: VoteSharing = VoteSharing.SPLIT_BETWEEN_NEIGHBOURS
    block_normalisation: BlockNormalisation = BlockNormalisation.L2_CLIPPED
    operator: GradientOperator = GradientOperator.SOBEL
    edge_rule: EdgeRule = EdgeRule.EXTEND

    @model_validator(mode="after")
    def _check_the_blocks_cover_every_cell(self) -> Self:
        if self.block_stride_in_cells > self.cells_per_block_side:
            raise ValueError(
                f"blocks {self.cells_per_block_side} cells across stepping "
                f"{self.block_stride_in_cells} cells at a time leave cells that "
                f"no block covers, so those cells would not reach the answer at "
                f"all; the stride is at most the block's own side"
            )
        return self

    @model_validator(mode="after")
    def _check_the_gradient_field_covers_the_patch(self) -> Self:
        if self.edge_rule is EdgeRule.KEEP_VALID:
            raise ValueError(
                "KEEP_VALID answers a gradient field smaller than the picture, "
                "and a cell here is a fixed block of the picture's own pixels, "
                "so the two would not line up; pass EXTEND, WRAP or "
                "PAD_WITH_ZERO, or hand in a patch already trimmed"
            )
        return self

    def n_cells_along(self, extent: int) -> int:
        """How many whole cells fit along a side ``extent`` pixels long."""
        return extent // self.cell_side

    def n_blocks_along(self, n_cells: int) -> int:
        """How many block positions fit along a side of ``n_cells`` cells."""
        return (n_cells - self.cells_per_block_side) // self.block_stride_in_cells + 1

    def n_values_for(self, height: int, width: int) -> int:
        """How long this configuration's description of that patch will be.

        The arithmetic, written once: the cells across and down are the patch's
        sides divided by the cell side; the block positions along each side
        follow from the block's own side and its stride; and each block
        contributes one number per bucket per cell it holds.

        Raises
        ------
        ShapeMismatchError
            If the patch does not divide into whole cells, or does not hold
            one whole block.
        """
        self._check_the_patch_fits(height, width)
        n_cell_rows = self.n_cells_along(height)
        n_cell_columns = self.n_cells_along(width)
        n_blocks = self.n_blocks_along(n_cell_rows) * self.n_blocks_along(
            n_cell_columns
        )
        return n_blocks * self.cells_per_block_side**2 * self.n_buckets

    def cell_histograms(self, picture: Picture) -> CellHistograms:
        """Vote every pixel of ``picture`` into its own cell's histogram.

        Raises
        ------
        ShapeMismatchError
            If the patch does not divide into whole cells, or does not hold
            one whole block.
        """
        self._check_the_patch_fits(picture.height, picture.width)

        field = GradientField.of(picture, self.operator, self.edge_rule)
        votes = votes_by_bucket(
            field.direction.values,
            field.magnitude.values,
            self.n_buckets,
            self.angle_range,
            self.vote_sharing,
        )

        n_cell_rows = self.n_cells_along(picture.height)
        n_cell_columns = self.n_cells_along(picture.width)
        by_cell = votes.reshape(
            n_cell_rows,
            self.cell_side,
            n_cell_columns,
            self.cell_side,
            self.n_buckets,
        )
        return CellHistograms(by_cell.sum(axis=(1, 3)), self.angle_range)

    def describe(self, picture: Picture) -> OrientedGradientDescription:
        """Normalise each overlapping block of cells and lay them end to end.

        Blocks are read row by row from the top; inside a block the cells are
        read the same way, and inside a cell the buckets run in order. That
        order is a contract rather than an implementation detail, since two
        descriptions are compared position by position.

        Raises
        ------
        ShapeMismatchError
            If the patch does not divide into whole cells, or does not hold
            one whole block.
        """
        histograms = self.cell_histograms(picture)
        side = self.cells_per_block_side
        stride = self.block_stride_in_cells

        blocks: list[FloatArray] = []
        for top in range(0, histograms.n_cell_rows - side + 1, stride):
            for left in range(0, histograms.n_cell_columns - side + 1, stride):
                block = histograms.counts[top : top + side, left : left + side]
                blocks.append(
                    normalised_block(block.reshape(-1).copy(), self.block_normalisation)
                )

        return OrientedGradientDescription(np.concatenate(blocks))

    def _check_the_patch_fits(self, height: int, width: int) -> None:
        if height % self.cell_side != 0 or width % self.cell_side != 0:
            raise ShapeMismatchError(
                f"a {height} by {width} patch does not divide into whole "
                f"{self.cell_side} by {self.cell_side} cells, and the leftover "
                f"strip would either vote in a cell of a different size or not "
                f"vote at all"
            )
        n_cell_rows = self.n_cells_along(height)
        n_cell_columns = self.n_cells_along(width)
        if (
            n_cell_rows < self.cells_per_block_side
            or n_cell_columns < self.cells_per_block_side
        ):
            raise ShapeMismatchError(
                f"a {height} by {width} patch holds {n_cell_rows} by "
                f"{n_cell_columns} cells, which is smaller than the "
                f"{self.cells_per_block_side} by {self.cells_per_block_side} "
                f"block that has to be normalised as one"
            )
