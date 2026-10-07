"""Lining two pictures of one scene up, by what they have in common.

The question the keypoints module left open
--------------------------------------------
The keypoints module ends on a measured admission: after a quarter turn of
the picture its descriptors are **1.443376** apart on a scale whose maximum is
2.0, which is the distance between two *different* corners of one square. The
keypoints survive the turn and the descriptions do not, because a patch of
brightness is read in the frame of the picture, and turning the picture turns
the patch. So two photographs of one scene, taken with the camera tilted, could
not be matched, and matching them is the whole reason for finding keypoints in
the first place.

This module answers that, and then answers the question matching was for: not
"which corner is which" but "how was the camera moved". Four steps, each one
object.

Give each keypoint a direction of its own
-----------------------------------------
A description read in the keypoint's *own* frame rather than the picture's
survives a turn, because turning the picture turns the frame with it. That
needs a direction that belongs to the keypoint, and there are two usual ways to
find one. ORB reads the *intensity centroid*: the brightness around a corner is
heavier on one side, and the direction from the corner to that weighted centre
turns with the picture. SIFT reads the *dominant gradient*: a histogram of
which way the brightness rises in the patch, weighted by how sharply, and the
fullest bin is the direction. :class:`IntensityCentroidRule` and
:class:`DominantGradientRule` are the two, behind one
:class:`OrientationRule`, and either turns a :class:`Keypoint` into an
:class:`OrientedKeypoint`.

Measured, on the six corners of a bent bar, a quarter turn of the picture
moves every intensity-centroid direction by **exactly** a quarter circle,
because a quarter turn permutes whole pixels and the moments are sums of
them. The gradient rule's directions move by a quarter circle to within
4.4e-16 radians, for the same reason one step later: the turn permutes the
gradients, the histogram is a sum of them, and the parabola through three
bins is the same parabola. One thing the gradient rule does that the centroid
rule cannot: at a convex corner of the bar two edges meet at equal strength,
the histogram has two equal peaks, and SIFT's rule keeps both as two
directions, since a tie-break would pick one in the first picture and the
other in the turned one.

Describe the patch in that frame, two ways
------------------------------------------
:class:`BinaryDescriptor` is ORB's rotated BRIEF. A fixed pattern of pixel
pairs is laid over the patch, turned to the keypoint's direction, and each
pair answers one bit: is the first pixel darker than the second. The
description is a string of bits and the distance between two of them is how
many bits differ, which is a count rather than a length. A fixed pattern is
what makes two pictures comparable at all: the pairs are drawn once from a
seeded generator, so every descriptor in a process reads the same pairs, and
:class:`SamplingPattern` is that draw made explicit rather than hidden in a
module global.

:class:`GradientHistogramDescriptor` is SIFT's descriptor, in its plain form.
The patch is cut into a grid of cells and each cell counts which way the
brightness rises, in eight directions, weighted by how sharply and by a bell
centred on the keypoint, with every direction measured *relative to the
keypoint's own*. The counts are normalised to unit length, the largest clipped
at 0.2 so that one sharp edge cannot dominate, and normalised again. Two of
them are compared by ordinary distance.

What neither does, said plainly as the keypoints module says it: no sub-pixel
refinement of the keypoint's position, no scale space finer than the
pyramid's levels, and no learned pair pattern, which is what separates this
from the reference implementations. The pairs are drawn from a Gaussian,
which is the second of the five patterns BRIEF's authors measured and the one
they recommend. The histogram does keep SIFT's two refinements that cost
nothing and matter, which are sharing each sample between neighbouring cells
and bins rather than rounding it into one, and refining the orientation peak
with a parabola rather than taking a bin's centre.

Match, then ask what move explains the matches
----------------------------------------------
Matching is the ratio test the keypoints module already argues for, applied to
either distance. The result is a :class:`Correspondences`, pairs of oriented
keypoints with the distance each won by.

A match is a claim that one position in the first picture is one position in
the second. If the camera moved rigidly, every true match is explained by the
same move: a turn, a change of scale and a shift, which is a
:class:`SimilarityTransform` with four numbers in it. Two matches are enough to
fix those four numbers exactly, and every further match either agrees with
them or does not. That is the shape RANSAC exploits. :class:`AlignmentEstimate`
takes two matches, fits the transform they imply, counts how many of the other
matches land within a pixel or so of where that transform sends them, and keeps
the pair that wins the most agreement; then it refits on everyone who agreed.
The matches that disagree are the outliers, and they are reported rather than
discarded, because which matches were wrong is half of what the method tells
you.

With few matches every pair is tried, in a fixed order, so the answer is the
same on every run; with many, a seeded sample of pairs is tried, which is what
the name RANSAC describes. The two routes share one fit and one count.

The transform estimate itself is the closed form: centre both sets of points,
read the scale off the ratio of their spreads, read the turn off the
two-by-two cross-covariance by its singular value decomposition, and read the
shift off what is left. Measured on three points under a planted turn of 0.6
radians, a scale of 1.3 and a shift of (4, -2), the recovered four numbers
agree with the planted ones to **1.3e-15**, which is the arithmetic and not an
approximation.

What the whole method measures on the teaching scene
-----------------------------------------------------
The classical vision pages share one drawn scene, a square, a disc, a bar and
three copies of a motif under a brightness ramp. Placed in a 64-pixel frame
and turned thirty degrees with a small shift, which lands no pixel on a pixel,
the ORB pair of rules finds 40 keypoints in each picture, keeps 15 matches of
which 10 are right, and the alignment names exactly those 10 as its inliers
and the 5 wrong ones as its outliers. The recovered turn is 0.56 degrees from
the planted one and the mean pixel difference between the two pictures falls
from 0.1755 to 0.0187 once the second is carried back. The SIFT pair keeps 34
matches, since a symmetric corner is two keypoints there, of which 17 are
right, and again the inliers are exactly the right ones.

The pyramid is the one piece that measured worse than expected. Six levels at
ORB's factor of 1.2 find 126 keypoints where one level finds 40, because every
corner is found again at each neighbouring size, and those copies are each
other's runners-up: the ratio test then refuses the true match, and 6 of 32
matches are right where 10 of 15 were. The move is still recovered, from the
6. So the pyramid is kept for the scale changes a single level cannot reach,
and is not the default, and the honest statement is that on a scene this size
the descriptors tolerate a change of scale of a quarter either way on their
own.

Where this stops working
-------------------------
On a scene whose two views differ by more than a similarity, which is to say
on anything photographed from two places rather than one place twice: a
perspective change is eight numbers and this fits four. On repeated structure,
for the keypoints module's reason, since the ratio test correctly refuses
every match on a brick wall. And on a scale change larger than the pyramid
covers: :class:`Pyramid` halves the picture a few times and describes
keypoints at whichever size they were found, which is ORB's answer with a
coarser step, and a picture taken from four times as far away as the pyramid
reaches is not found.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from oop_ml.core.computer_vision.edges import GradientField
from oop_ml.core.computer_vision.filtering import EdgeRule, swept
from oop_ml.core.computer_vision.keypoints import (
    CornerMeasure,
    CornerResponse,
    Keypoint,
    Keypoints,
)
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.types import FloatArray, array_for_protocol

FULL_CIRCLE = 2.0 * np.pi

DEFAULT_ORIENTATION_RADIUS = 4
"""How far from the keypoint the direction-finding rules look.

A disc of this radius holds the patch the descriptors read, so the direction
is found from the same brightness that is then described in it.
"""

ORIENTATION_BINS = 36
"""SIFT's own count: ten degrees a bin."""

DEFAULT_PAIR_COUNT = 256
"""How many pixel pairs the binary descriptor reads. ORB's own figure.

Measured on the workbench scene turned thirty degrees, 256 pairs over a
fifteen-wide patch got 10 of 15 matches right where 128 pairs over a
nine-wide patch got 1 of 8, so the defaults are the reference's rather than
smaller ones that would print more neatly.
"""

DEFAULT_PATCH_SIDE = 15
"""The square the binary descriptor's pairs are drawn inside.

A turned pattern reaches ``7 * sqrt(2)`` from the keypoint, so keypoints
need a border of ten pixels to be described.
"""

DEFAULT_CELLS = 4
"""Cells along each side of the gradient histogram descriptor. SIFT's figure."""

DEFAULT_CELL_SIDE = 4
"""Pixels along each side of one cell. SIFT's own figure.

Four cells of four is a sixteen-wide square, which turned reaches
``8 * sqrt(2)`` from the keypoint, so keypoints need a border of twelve.
Measured on the workbench turned thirty degrees, sixteen-wide got 19 of 24
matches right where eight-wide got 1 of 3.
"""

DIRECTION_BINS = 8
"""Directions each cell of the gradient histogram distinguishes. SIFT's figure."""

CLIP_SHARE = 0.2
"""The largest share of the gradient histogram's length one entry may hold."""

DEFAULT_INLIER_DISTANCE = 1.5
"""How far, in pixels, a match may land from where the transform sends it."""

DEFAULT_TRIALS = 500
"""How many pairs of matches the alignment tries before settling."""


def _folded(angle: float) -> float:
    """``angle`` brought into ``(-pi, pi]``, the gradient field's own range."""
    folded = float((angle + np.pi) % FULL_CIRCLE - np.pi)
    return folded + FULL_CIRCLE if folded <= -np.pi else folded


def _disc(radius: int) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Row offsets, column offsets and a mask of the pixels within ``radius``."""
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    rows, columns = np.meshgrid(offsets, offsets, indexing="ij")
    inside = (rows * rows + columns * columns) <= radius * radius
    return rows, columns, inside


def _patch_around(picture: Picture, row: int, column: int, radius: int) -> FloatArray:
    """The ``(2 radius + 1)`` square around a position, refused at the edge."""
    height, width = picture.shape
    if (
        row - radius < 0
        or column - radius < 0
        or row + radius >= height
        or column + radius >= width
    ):
        raise InvalidValuesError(
            f"a patch of radius {radius} around ({row}, {column}) runs off a "
            f"{height} by {width} picture; detect keypoints with a border of at "
            f"least {radius}, or describe with a smaller patch"
        )
    return picture.values[
        row - radius : row + radius + 1, column - radius : column + radius + 1
    ]


class OrientedKeypoint:
    """A keypoint with a direction of its own, and the size it was found at.

    Parameters
    ----------
    keypoint:
        Where, and how strongly it was preferred.
    angle:
        Its direction in radians, measured as the gradient field measures one:
        from pointing right, turning towards pointing down, in ``(-pi, pi]``.
    scale:
        How many picture pixels one pixel of the level it was found at
        covers. One for the picture itself, two for the first halving.

    Raises
    ------
    InvalidValuesError
        If the angle is not finite or the scale is not positive.
    """

    __slots__ = ("_angle", "_keypoint", "_scale")

    def __init__(self, keypoint: Keypoint, angle: float, scale: float = 1.0) -> None:
        if not np.isfinite(angle):
            raise InvalidValuesError(f"a direction is a finite angle, got {angle}")
        if not np.isfinite(scale) or scale <= 0.0:
            raise InvalidValuesError(
                f"a scale is a positive number of pixels per pixel, got {scale}"
            )
        self._keypoint = keypoint
        self._angle = _folded(float(angle))
        self._scale = float(scale)

    @property
    def keypoint(self) -> Keypoint:
        """The position and strength this direction was found for."""
        return self._keypoint

    @property
    def row(self) -> int:
        """Which row of the picture, from the top."""
        return self._keypoint.row

    @property
    def column(self) -> int:
        """Which column, from the left."""
        return self._keypoint.column

    @property
    def strength(self) -> float:
        """The corner measure's score at this position."""
        return self._keypoint.strength

    @property
    def angle(self) -> float:
        """The direction, in radians, in ``(-pi, pi]``."""
        return self._angle

    @property
    def scale(self) -> float:
        """Picture pixels per pixel of the level this was found at."""
        return self._scale

    @property
    def position(self) -> FloatArray:
        """``(row, column)`` in the picture's own pixels, as floats."""
        return np.asarray([self._keypoint.row, self._keypoint.column], dtype=np.float64)

    def turned_by(self, angle: float) -> OrientedKeypoint:
        """The same keypoint with its direction moved by ``angle``.

        Exists so a spec can state what a turn of the picture should do to a
        direction, and check it.
        """
        return OrientedKeypoint(self._keypoint, self._angle + angle, self._scale)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrientedKeypoint) or type(self) is not type(other):
            return NotImplemented
        return (
            self._keypoint == other._keypoint
            and self._angle == other._angle
            and self._scale == other._scale
        )

    def __hash__(self) -> int:
        return hash((self._keypoint, self._angle, self._scale))

    def __repr__(self) -> str:
        return (
            f"OrientedKeypoint(row={self.row}, column={self.column}, "
            f"angle={self._angle:.4f}, scale={self._scale:g})"
        )


class OrientedKeypoints:
    """The oriented keypoints of one picture, strongest first as found."""

    __slots__ = ("_keypoints",)

    def __init__(self, keypoints: Iterable[OrientedKeypoint]) -> None:
        self._keypoints = tuple(keypoints)

    def __iter__(self) -> Iterator[OrientedKeypoint]:
        return iter(self._keypoints)

    def __len__(self) -> int:
        return len(self._keypoints)

    def __getitem__(self, index: int) -> OrientedKeypoint:
        return self._keypoints[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrientedKeypoints):
            return NotImplemented
        return self._keypoints == other._keypoints

    def __hash__(self) -> int:
        return hash(self._keypoints)

    def __repr__(self) -> str:
        return f"OrientedKeypoints({len(self._keypoints)} keypoints)"


class OrientationRule(BaseModel, ABC):
    """How a keypoint is given a direction that turns with the picture.

    ``radius`` is how far around the keypoint the rule looks. A subclass
    supplies :meth:`_angles_of`, reading a square patch and answering one or
    more directions in radians, strongest first. One keypoint can come back
    as several oriented keypoints, one per direction, which is how a corner
    where two edges meet equally is handled: rather than pick one of the two
    directions by a rule that would not survive a turn, both are kept and
    both travel.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    radius: int = Field(
        default=DEFAULT_ORIENTATION_RADIUS,
        ge=1,
        description="How far from the keypoint the rule looks, in pixels.",
    )

    def angle_at(self, picture: Picture, keypoint: Keypoint) -> float:
        """The strongest direction at ``keypoint``, in ``(-pi, pi]``.

        Raises
        ------
        InvalidValuesError
            If the disc around the keypoint runs off the picture.
        """
        return self.angles_at(picture, keypoint)[0]

    def angles_at(self, picture: Picture, keypoint: Keypoint) -> list[float]:
        """Every direction the rule keeps at ``keypoint``, strongest first."""
        patch = _patch_around(picture, keypoint.row, keypoint.column, self.radius)
        return [_folded(angle) for angle in self._angles_of(patch)]

    def orient(
        self, picture: Picture, keypoints: Keypoints, scale: float = 1.0
    ) -> OrientedKeypoints:
        """Every keypoint given its directions. ``scale`` is carried through."""
        return OrientedKeypoints(
            OrientedKeypoint(keypoint, angle, scale)
            for keypoint in keypoints
            for angle in self.angles_at(picture, keypoint)
        )

    @abstractmethod
    def _angles_of(self, patch: FloatArray) -> list[float]:
        """Radians, strongest first, from a square patch centred on the keypoint."""

    @property
    @abstractmethod
    def description(self) -> str:
        """What the rule reads, for a page or a log."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(radius={self.radius})"


class IntensityCentroidRule(OrientationRule):
    """ORB's direction: from the keypoint towards the patch's centre of brightness.

    The moments ``m10 = sum(column * brightness)`` and
    ``m01 = sum(row * brightness)`` over the disc, with the keypoint at the
    origin, and the direction ``atan2(m01, m10)``. A corner has brightness on
    one side and not the other, so the centroid sits off-centre and the
    direction is well defined; a flat disc has its centroid at the origin and
    answers zero, which is a convention, as the gradient field's own is. One
    direction always, since a centre of brightness is one place.
    """

    def _angles_of(self, patch: FloatArray) -> list[float]:
        rows, columns, inside = _disc(self.radius)
        weights = np.where(inside, patch, 0.0)
        down = float(np.sum(rows * weights))
        right = float(np.sum(columns * weights))
        return [float(math.atan2(down, right))]

    @property
    def description(self) -> str:
        return "the direction from the keypoint to the centre of brightness around it"


PEAK_SHARE = 0.8
"""A histogram peak at least this share of the highest is kept as a direction.

SIFT's own figure. A corner where two edges meet at equal strength has two
peaks of equal height, and keeping both is what lets its description survive
a turn: whichever of the two a tie-break would have picked, the turned
picture would have picked the other.
"""


class DominantGradientRule(OrientationRule):
    """SIFT's directions: the ways the brightness most often rises around it.

    Every pixel of the disc votes for the bin its gradient direction falls
    in, with a weight of its gradient magnitude times a bell centred on the
    keypoint. Every peak of the histogram within :data:`PEAK_SHARE` of the
    highest is a direction, each refined by fitting a parabola through its
    bin and the two neighbours so the answer is not quantised to a bin.
    Thirty-six bins, ten degrees each, as SIFT has.
    """

    bins: int = Field(default=ORIENTATION_BINS, ge=3)

    def _angles_of(self, patch: FloatArray) -> list[float]:
        field = GradientField.of(Picture(patch))
        rows, columns, inside = _disc(self.radius)
        bell = np.exp(
            -(rows * rows + columns * columns) / (2.0 * (self.radius / 2.0) ** 2)
        )
        weights = np.where(inside, field.magnitude.values * bell, 0.0)
        directions = field.direction.values
        # (-pi, pi] onto [0, bins), with the seam at pointing left.
        which = (
            np.floor((directions + np.pi) / FULL_CIRCLE * self.bins).astype(int)
            % self.bins
        )
        votes = np.bincount(which.ravel(), weights=weights.ravel(), minlength=self.bins)
        highest = float(votes.max())
        if highest == 0.0:
            return [float(0.5 / self.bins * FULL_CIRCLE - np.pi)]
        before = np.roll(votes, 1)
        after = np.roll(votes, -1)
        peaks = [
            index
            for index in range(self.bins)
            if votes[index] >= PEAK_SHARE * highest
            and votes[index] >= before[index]
            and votes[index] >= after[index]
        ]
        # Strongest first, ties by bin, so two pictures of one corner agree
        # about the order as well as the set.
        peaks.sort(key=lambda index: (-votes[index], index))
        angles = []
        for index in peaks:
            # The vertex of the parabola through the three bins around the
            # peak, as an offset in bins; zero when the two neighbours tie.
            curvature = before[index] - 2.0 * votes[index] + after[index]
            offset = (
                0.0
                if curvature == 0.0
                else 0.5 * (before[index] - after[index]) / curvature
            )
            angles.append(
                float((index + 0.5 + offset) / self.bins * FULL_CIRCLE - np.pi)
            )
        return angles

    @property
    def description(self) -> str:
        return "the directions the brightness most often rises in around the keypoint"


class Pyramid:
    """A picture at several sizes, each smaller than the one before by one factor.

    A keypoint found on a shrunken picture covers more ground than one found
    on the original, so describing it there describes a wider patch in picture
    pixels, which is how a corner seen from further away comes to have the
    same description. ORB shrinks by 1.2 a level over eight levels; a factor of
    two halves exactly, by averaging each two by two block, and is the cheap
    default. Any other factor resamples bilinearly.

    Parameters
    ----------
    picture:
        The full-size picture, which is level zero.
    levels:
        How many sizes, counting the picture itself.
    factor:
        How much smaller each level is than the one before. Above one.

    Raises
    ------
    InvalidValuesError
        If fewer than one level is asked for, the factor is not above one, or
        the picture is too small to shrink that many times.
    """

    __slots__ = ("_factor", "_levels")

    def __init__(self, picture: Picture, levels: int = 1, factor: float = 2.0) -> None:
        if levels < 1:
            raise InvalidValuesError(f"a pyramid has at least one level, got {levels}")
        if not np.isfinite(factor) or factor <= 1.0:
            raise InvalidValuesError(
                f"each level is smaller than the last by a factor above one, "
                f"got {factor}"
            )
        found = [picture]
        for level in range(1, levels):
            height, width = picture.shape
            shrunk_height = int(round(height / factor**level))
            shrunk_width = int(round(width / factor**level))
            if shrunk_height < 2 or shrunk_width < 2:
                raise InvalidValuesError(
                    f"a {height} by {width} picture shrinks by {factor} only "
                    f"{level} time(s) before a side is under two pixels; "
                    f"{levels} levels were asked for"
                )
            if factor == 2.0:
                previous = found[-1].values
                rows, columns = previous.shape
                values = previous[: rows - rows % 2, : columns - columns % 2]
                found.append(
                    Picture(
                        values.reshape(rows // 2, 2, columns // 2, 2).mean(axis=(1, 3))
                    )
                )
            else:
                shrink = SimilarityTransform(1.0 / factor**level)
                found.append(shrink.warp(picture, (shrunk_height, shrunk_width)))
        self._levels = tuple(found)
        self._factor = float(factor)

    @classmethod
    def of(cls, source: Pyramid | Picture) -> Pyramid:
        """``source`` as a pyramid, a bare picture being one of one level."""
        return source if isinstance(source, Pyramid) else cls(source)

    @property
    def levels(self) -> tuple[Picture, ...]:
        """Largest first."""
        return self._levels

    @property
    def factor(self) -> float:
        return self._factor

    @property
    def picture(self) -> Picture:
        """Level zero."""
        return self._levels[0]

    def scale_of(self, level: int) -> float:
        """Picture pixels per pixel of ``level``."""
        return float(self._factor**level)

    def level_of(self, scale: float) -> int:
        """Which level has ``scale`` picture pixels per pixel.

        Raises
        ------
        InvalidValuesError
            If no level of this pyramid has that scale.
        """
        for level in range(len(self._levels)):
            if math.isclose(self.scale_of(level), scale, rel_tol=1e-9):
                return level
        raise InvalidValuesError(
            f"no level of a {len(self._levels)}-level pyramid with factor "
            f"{self._factor} has a scale of {scale}"
        )

    def __len__(self) -> int:
        return len(self._levels)

    def __getitem__(self, level: int) -> Picture:
        return self._levels[level]

    def __repr__(self) -> str:
        return (
            f"Pyramid({len(self._levels)} levels from {self._levels[0].shape}, "
            f"factor {self._factor:g})"
        )


def oriented_keypoints_of(
    source: Pyramid | Picture,
    rule: OrientationRule | None = None,
    measure: CornerMeasure | None = None,
    minimum_strength: float = 0.0,
    separation: int = 3,
    limit: int | None = None,
    border: int | None = None,
) -> OrientedKeypoints:
    """Detect at every level of a pyramid, and orient.

    The corner response is the keypoints module's, run on each level; each
    keypoint's position is carried back into the picture's own pixels and its
    scale recorded. ``limit`` applies per level and ``border`` defaults to the
    rule's radius, which is the least that keeps every disc inside its level.
    """
    rule = rule or IntensityCentroidRule()
    margin = rule.radius if border is None else border
    pyramid = Pyramid.of(source)
    found: list[OrientedKeypoint] = []
    for level, sized in enumerate(pyramid.levels):
        peaks = CornerResponse.of(sized, measure).peaks(
            minimum_strength=minimum_strength,
            separation=separation,
            limit=limit,
            border=margin,
        )
        scale = pyramid.scale_of(level)
        for oriented in rule.orient(sized, peaks, scale):
            # The position is the level's; carry it back into picture pixels,
            # keeping the strength the level's measure gave it.
            placed = Keypoint(
                int(round(oriented.row * scale)),
                int(round(oriented.column * scale)),
                oriented.strength,
            )
            found.append(OrientedKeypoint(placed, oriented.angle, scale))
    found.sort(key=lambda entry: (-entry.strength, entry.row, entry.column))
    return OrientedKeypoints(found)


class SamplingPattern:
    """The pixel pairs a binary descriptor reads, drawn once.

    Each pair is two offsets from the keypoint, drawn from a Gaussian whose
    spread is a fifth of the patch side and clipped to the patch, which is
    the second of the patterns BRIEF's authors compared and the one they
    recommend. Drawn once from a seed so that every descriptor in a run reads
    the same pairs, and carried as an object so that a reader can see which.

    Parameters
    ----------
    pair_count:
        How many pairs, and so how many bits.
    patch_side:
        The square the pairs lie inside. Odd, so the keypoint is its centre.
    seed:
        What the draw is seeded with.

    Raises
    ------
    InvalidValuesError
        If there is not at least one pair, or the patch has no centre pixel.
    """

    __slots__ = ("_first", "_pair_count", "_patch_side", "_second", "_seed")

    def __init__(
        self,
        pair_count: int = DEFAULT_PAIR_COUNT,
        patch_side: int = DEFAULT_PATCH_SIDE,
        seed: int = 0,
    ) -> None:
        if pair_count < 1:
            raise InvalidValuesError(
                f"a pattern reads at least one pair, got {pair_count}"
            )
        if patch_side < 3 or patch_side % 2 == 0:
            raise InvalidValuesError(
                f"a patch is an odd square of at least three, so it has a "
                f"centre pixel to read around; got {patch_side}"
            )
        self._pair_count = pair_count
        self._patch_side = patch_side
        self._seed = seed
        reach = patch_side // 2
        generator = np.random.default_rng(seed)
        offsets = generator.normal(0.0, patch_side / 5.0, size=(pair_count, 2, 2))
        offsets = np.clip(np.rint(offsets), -reach, reach)
        self._first = offsets[:, 0, :]
        self._second = offsets[:, 1, :]
        self._first.setflags(write=False)
        self._second.setflags(write=False)

    @property
    def pair_count(self) -> int:
        return self._pair_count

    @property
    def patch_side(self) -> int:
        return self._patch_side

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def reach(self) -> int:
        """How far from the keypoint a pair may read, in level pixels."""
        return self._patch_side // 2

    def turned_by(self, angle: float) -> tuple[FloatArray, FloatArray]:
        """Both sets of ``(row, column)`` offsets turned by ``angle``.

        Turned as the gradient field measures a direction, so that a pattern
        turned to a keypoint's own angle reads the patch in the keypoint's
        frame. Rounded to whole pixels, which is what ORB does too.
        """
        cosine, sine = math.cos(angle), math.sin(angle)
        turn = np.asarray([[cosine, sine], [-sine, cosine]])
        first = np.rint(self._first @ turn.T)
        second = np.rint(self._second @ turn.T)
        return first, second

    def __repr__(self) -> str:
        return (
            f"SamplingPattern(pair_count={self._pair_count}, "
            f"patch_side={self._patch_side}, seed={self._seed})"
        )


SMOOTHING = [[1.0 / 9.0] * 3 for _ in range(3)]
"""The three by three mean the binary descriptor reads through.

A single pixel is noisy and a bit flips when it is. BRIEF reads through a
smoothed picture for that reason; the mean is the sweep this package is
built from.
"""


class Descriptor(ABC):
    """What travels between pictures: a keypoint and the reading around it."""

    __slots__ = ("_keypoint",)

    def __init__(self, keypoint: OrientedKeypoint) -> None:
        self._keypoint = keypoint

    @property
    def keypoint(self) -> OrientedKeypoint:
        return self._keypoint

    @abstractmethod
    def distance_to(self, other: Any) -> float:
        """How unlike two readings are, in the descriptor's own units."""

    @property
    @abstractmethod
    def n_values(self) -> int:
        """How many numbers the reading holds."""


class BinaryDescriptor(Descriptor):
    """ORB's reading: one bit per pair, is the first pixel darker than the second.

    Parameters
    ----------
    keypoint:
        Where, and which way it faces.
    bits:
        The answers, one per pair of the pattern.
    pattern:
        Which pairs were read, so that two descriptors from different
        patterns cannot be compared by accident.
    """

    __slots__ = ("_bits", "_pattern")

    def __init__(
        self, keypoint: OrientedKeypoint, bits: Any, pattern: SamplingPattern
    ) -> None:
        super().__init__(keypoint)
        flags = np.asarray(bits, dtype=bool).ravel()
        if flags.size != pattern.pair_count:
            raise ShapeMismatchError(
                f"the pattern reads {pattern.pair_count} pairs, got {flags.size} bits"
            )
        self._bits = flags.copy()
        self._bits.setflags(write=False)
        self._pattern = pattern

    @classmethod
    def of(
        cls,
        picture: Picture,
        keypoint: OrientedKeypoint,
        pattern: SamplingPattern | None = None,
        smoothed: Picture | None = None,
    ) -> BinaryDescriptor:
        """Read the pattern around ``keypoint``, turned to its direction.

        ``smoothed`` lets a caller sweep the picture once and describe many
        keypoints from it; left out, the sweep is done here.

        Raises
        ------
        InvalidValuesError
            If a turned pair would read outside the picture.
        """
        pattern = pattern or SamplingPattern()
        source = (
            smoothed
            if smoothed is not None
            else swept(picture, SMOOTHING, EdgeRule.EXTEND)
        )
        first, second = pattern.turned_by(keypoint.angle)
        centre = np.asarray([keypoint.row, keypoint.column], dtype=np.float64)
        at_first = (first + centre).astype(int)
        at_second = (second + centre).astype(int)
        height, width = source.shape
        reads = np.concatenate([at_first, at_second])
        if reads.min() < 0 or reads[:, 0].max() >= height or reads[:, 1].max() >= width:
            raise InvalidValuesError(
                f"the pattern turned to {keypoint.angle:.3f} radians around "
                f"({keypoint.row}, {keypoint.column}) reads outside a {height} by "
                f"{width} picture; detect with a border of at least "
                f"{math.ceil(pattern.reach * math.sqrt(2))}"
            )
        values = source.values
        bits = (
            values[at_first[:, 0], at_first[:, 1]]
            < values[at_second[:, 0], at_second[:, 1]]
        )
        return cls(keypoint, bits, pattern)

    @property
    def bits(self) -> np.ndarray:
        """The reading, frozen."""
        return self._bits

    @property
    def pattern(self) -> SamplingPattern:
        return self._pattern

    @property
    def n_values(self) -> int:
        return int(self._bits.size)

    def distance_to(self, other: Any) -> float:
        """How many bits differ: the Hamming distance.

        Raises
        ------
        ShapeMismatchError
            If ``other`` is not a binary descriptor from the same pattern.
        """
        if not isinstance(other, BinaryDescriptor):
            raise ShapeMismatchError(
                f"a binary descriptor is compared with another, "
                f"got {type(other).__name__}"
            )
        if other.pattern is not self._pattern and (
            other.pattern.seed != self._pattern.seed
            or other.pattern.pair_count != self._pattern.pair_count
            or other.pattern.patch_side != self._pattern.patch_side
        ):
            raise ShapeMismatchError(
                "two binary descriptors are comparable only when they read the "
                "same pattern of pairs"
            )
        return float(np.count_nonzero(self._bits != other.bits))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        return array_for_protocol(self._bits, dtype, copy)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryDescriptor) or type(self) is not type(other):
            return NotImplemented
        return self._keypoint == other._keypoint and bool(
            np.array_equal(self._bits, other._bits)
        )

    def __hash__(self) -> int:
        return hash((self._keypoint, self._bits.tobytes()))

    def __repr__(self) -> str:
        shown = "".join("1" if bit else "0" for bit in self._bits[:16])
        return f"BinaryDescriptor({self._keypoint!r}, bits={shown}...)"


def _bilinear(values: FloatArray, rows: FloatArray, columns: FloatArray) -> FloatArray:
    """``values`` read at fractional positions, zero outside the picture."""
    height, width = values.shape
    row_floor = np.floor(rows).astype(int)
    column_floor = np.floor(columns).astype(int)
    row_part = rows - row_floor
    column_part = columns - column_floor

    def read(row_index: np.ndarray, column_index: np.ndarray) -> FloatArray:
        inside = (
            (row_index >= 0)
            & (row_index < height)
            & (column_index >= 0)
            & (column_index < width)
        )
        safe_rows = np.clip(row_index, 0, height - 1)
        safe_columns = np.clip(column_index, 0, width - 1)
        return np.where(inside, values[safe_rows, safe_columns], 0.0)

    return (
        read(row_floor, column_floor) * (1.0 - row_part) * (1.0 - column_part)
        + read(row_floor, column_floor + 1) * (1.0 - row_part) * column_part
        + read(row_floor + 1, column_floor) * row_part * (1.0 - column_part)
        + read(row_floor + 1, column_floor + 1) * row_part * column_part
    )


class GradientHistogramDescriptor(Descriptor):
    """SIFT's reading: per cell, how sharply the brightness rises each way.

    Parameters
    ----------
    keypoint:
        Where, and which way it faces.
    values:
        ``cells * cells * DIRECTION_BINS`` numbers, normalised as the module
        docstring says.
    cells:
        Cells along each side of the grid, so a caller can tell two layouts
        apart.
    """

    __slots__ = ("_cells", "_values")

    def __init__(self, keypoint: OrientedKeypoint, values: Any, cells: int) -> None:
        super().__init__(keypoint)
        numbers = np.asarray(values, dtype=np.float64).ravel()
        if numbers.size != cells * cells * DIRECTION_BINS:
            raise ShapeMismatchError(
                f"a {cells} by {cells} grid of {DIRECTION_BINS} directions holds "
                f"{cells * cells * DIRECTION_BINS} numbers, got {numbers.size}"
            )
        if not np.all(np.isfinite(numbers)):
            raise InvalidValuesError("a descriptor holds finite numbers")
        self._values = numbers.copy()
        self._values.setflags(write=False)
        self._cells = cells

    @classmethod
    def of(
        cls,
        picture: Picture,
        keypoint: OrientedKeypoint,
        cells: int = DEFAULT_CELLS,
        cell_side: int = DEFAULT_CELL_SIDE,
        field: GradientField | None = None,
    ) -> GradientHistogramDescriptor:
        """Read the grid around ``keypoint``, in its own frame.

        Sample positions are laid out on a ``cells * cell_side`` square in the
        keypoint's frame, turned into the picture's, and the gradient read
        there by bilinear interpolation. Each sample votes for its cell and
        for the direction bin of its gradient *relative to the keypoint's
        direction*, weighted by magnitude and by a bell over the square.

        ``field`` lets a caller compute the gradients once for many keypoints.

        Raises
        ------
        InvalidValuesError
            If the grid is empty, or the turned square runs off the picture.
        """
        if cells < 1 or cell_side < 1:
            raise InvalidValuesError(
                f"a grid has at least one cell of at least one pixel, got "
                f"{cells} cells of {cell_side}"
            )
        gradients = field if field is not None else GradientField.of(picture)
        side = cells * cell_side
        half = side / 2.0
        # Sample at pixel centres of the square, in the keypoint's frame.
        axis = np.arange(side, dtype=np.float64) - half + 0.5
        local_rows, local_columns = np.meshgrid(axis, axis, indexing="ij")
        cosine, sine = math.cos(keypoint.angle), math.sin(keypoint.angle)
        # A point at (down, right) in the keypoint's frame sits in the picture
        # turned by the keypoint's angle, the way the pattern is turned.
        picture_rows = keypoint.row + local_rows * cosine + local_columns * sine
        picture_columns = keypoint.column - local_rows * sine + local_columns * cosine
        height, width = picture.shape
        if (
            picture_rows.min() < 0
            or picture_columns.min() < 0
            or picture_rows.max() > height - 1
            or picture_columns.max() > width - 1
        ):
            raise InvalidValuesError(
                f"a {side}-wide square turned to {keypoint.angle:.3f} radians "
                f"around ({keypoint.row}, {keypoint.column}) runs off a {height} "
                f"by {width} picture; detect with a border of at least "
                f"{math.ceil(half * math.sqrt(2))}"
            )
        horizontal = _bilinear(
            gradients.horizontal.values, picture_rows, picture_columns
        )
        vertical = _bilinear(gradients.vertical.values, picture_rows, picture_columns)
        magnitude = np.hypot(horizontal, vertical)
        relative = np.arctan2(vertical, horizontal) - keypoint.angle
        bell = np.exp(-(local_rows**2 + local_columns**2) / (2.0 * (half) ** 2))
        weights = magnitude * bell

        # Each sample's weight is shared between the two nearest cells along
        # each axis and the two nearest direction bins, in proportion to how
        # near it sits to each, which is SIFT's trilinear interpolation.
        # Without it a sample on a cell boundary moves whole from one cell to
        # the other under a change of a fraction of a pixel, and two readings
        # of one corner can differ by a whole cell's worth.
        bin_position = (relative % FULL_CIRCLE) / FULL_CIRCLE * DIRECTION_BINS - 0.5
        row_position = (local_rows + half) / cell_side - 0.5
        column_position = (local_columns + half) / cell_side - 0.5
        bin_floor = np.floor(bin_position)
        row_floor = np.floor(row_position)
        column_floor = np.floor(column_position)
        histogram = np.zeros(cells * cells * DIRECTION_BINS)
        for row_step in (0, 1):
            for column_step in (0, 1):
                for bin_step in (0, 1):
                    cell_row = row_floor + row_step
                    cell_column = column_floor + column_step
                    direction_bin = bin_floor + bin_step
                    share = (
                        (1.0 - np.abs(row_position - cell_row))
                        * (1.0 - np.abs(column_position - cell_column))
                        * (1.0 - np.abs(bin_position - direction_bin))
                    )
                    inside = (
                        (cell_row >= 0)
                        & (cell_row < cells)
                        & (cell_column >= 0)
                        & (cell_column < cells)
                    )
                    slot = (
                        np.clip(cell_row, 0, cells - 1).astype(int) * cells
                        + np.clip(cell_column, 0, cells - 1).astype(int)
                    ) * DIRECTION_BINS + direction_bin.astype(int) % DIRECTION_BINS
                    np.add.at(histogram, slot[inside], (weights * share)[inside])

        length = float(np.linalg.norm(histogram))
        if length > 0.0:
            histogram = histogram / length
            histogram = np.minimum(histogram, CLIP_SHARE)
            clipped_length = float(np.linalg.norm(histogram))
            if clipped_length > 0.0:
                histogram = histogram / clipped_length
        return cls(keypoint, histogram, cells)

    @property
    def values(self) -> FloatArray:
        """The reading, frozen."""
        return self._values

    @property
    def cells(self) -> int:
        return self._cells

    @property
    def n_values(self) -> int:
        return int(self._values.size)

    @property
    def is_flat(self) -> bool:
        """Whether nothing in the square rose anywhere, so nothing was read."""
        return bool(np.all(self._values == 0.0))

    def distance_to(self, other: Any) -> float:
        """Ordinary Euclidean distance between the two readings.

        Raises
        ------
        ShapeMismatchError
            If ``other`` is not a gradient histogram of the same layout.
        """
        if not isinstance(other, GradientHistogramDescriptor):
            raise ShapeMismatchError(
                f"a gradient histogram is compared with another, "
                f"got {type(other).__name__}"
            )
        if other.cells != self._cells:
            raise ShapeMismatchError(
                f"two histograms are comparable only on the same grid, got "
                f"{self._cells} and {other.cells} cells a side"
            )
        return float(np.linalg.norm(self._values - other.values))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        return array_for_protocol(self._values, dtype, copy)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GradientHistogramDescriptor) or type(self) is not type(
            other
        ):
            return NotImplemented
        return self._keypoint == other._keypoint and bool(
            np.array_equal(self._values, other._values)
        )

    def __hash__(self) -> int:
        return hash((self._keypoint, self._values.tobytes()))

    def __repr__(self) -> str:
        return (
            f"GradientHistogramDescriptor({self._keypoint!r}, "
            f"{self._cells}x{self._cells} cells)"
        )


class Correspondence:
    """One keypoint in the first picture claimed to be one in the second.

    Parameters
    ----------
    first, second:
        The two oriented keypoints.
    distance:
        How far apart their descriptions were.
    runner_up_distance:
        The second-nearest description in the second picture, which is what
        the ratio test compares against.

    Raises
    ------
    InvalidValuesError
        If either distance is negative or not finite, or the runner-up is
        nearer than the winner.
    """

    __slots__ = ("_distance", "_first", "_runner_up", "_second")

    def __init__(
        self,
        first: OrientedKeypoint,
        second: OrientedKeypoint,
        distance: float,
        runner_up_distance: float,
    ) -> None:
        for name, value in (
            ("distance", distance),
            ("runner-up distance", runner_up_distance),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise InvalidValuesError(
                    f"a {name} is finite and not negative, got {value}"
                )
        if runner_up_distance < distance:
            raise InvalidValuesError(
                f"the runner-up is never nearer than the winner, got {distance} "
                f"against {runner_up_distance}"
            )
        self._first = first
        self._second = second
        self._distance = float(distance)
        self._runner_up = float(runner_up_distance)

    @property
    def first(self) -> OrientedKeypoint:
        return self._first

    @property
    def second(self) -> OrientedKeypoint:
        return self._second

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def runner_up_distance(self) -> float:
        return self._runner_up

    @property
    def ratio(self) -> float:
        """Winner over runner-up, which is what the ratio test reads.

        Zero over zero is one, since two equally good answers are the case
        the test exists to refuse.
        """
        if self._runner_up == 0.0:
            return 1.0
        return self._distance / self._runner_up

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Correspondence) or type(self) is not type(other):
            return NotImplemented
        return (
            self._first == other._first
            and self._second == other._second
            and self._distance == other._distance
            and self._runner_up == other._runner_up
        )

    def __hash__(self) -> int:
        return hash((self._first, self._second, self._distance, self._runner_up))

    def __repr__(self) -> str:
        return (
            f"Correspondence(({self._first.row}, {self._first.column}) -> "
            f"({self._second.row}, {self._second.column}), "
            f"distance={self._distance:.4g}, "
            f"ratio={self.ratio:.3f})"
        )


class Correspondences:
    """The matches between two pictures, most distinctive first."""

    __slots__ = ("_matches",)

    def __init__(self, matches: Iterable[Correspondence]) -> None:
        self._matches = tuple(
            sorted(matches, key=lambda match: (match.ratio, match.distance))
        )

    @property
    def first_positions(self) -> FloatArray:
        """``(n, 2)`` rows and columns in the first picture."""
        return np.asarray([match.first.position for match in self._matches]).reshape(
            -1, 2
        )

    @property
    def second_positions(self) -> FloatArray:
        """``(n, 2)`` rows and columns in the second picture."""
        return np.asarray([match.second.position for match in self._matches]).reshape(
            -1, 2
        )

    def __iter__(self) -> Iterator[Correspondence]:
        return iter(self._matches)

    def __len__(self) -> int:
        return len(self._matches)

    def __getitem__(self, index: int) -> Correspondence:
        return self._matches[index]

    def __repr__(self) -> str:
        return f"Correspondences({len(self._matches)} matches)"


class DescriptorSet:
    """Every description of one picture, and the matching between two sets.

    Parameters
    ----------
    descriptors:
        All of one kind. Mixing a binary reading with a histogram is refused
        at construction, since nothing could compare them.

    Raises
    ------
    EmptyValuesError
        If there are none.
    ShapeMismatchError
        If they are not all of one kind and size.
    """

    __slots__ = ("_descriptors",)

    def __init__(self, descriptors: Iterable[Descriptor]) -> None:
        entries = tuple(descriptors)
        if not entries:
            raise EmptyValuesError(
                "a set of descriptors describes at least one keypoint"
            )
        kinds = {(type(entry), entry.n_values) for entry in entries}
        if len(kinds) != 1:
            raise ShapeMismatchError(
                f"a set holds descriptors of one kind and size, "
                f"got {sorted(str(kind) for kind in kinds)}"
            )
        self._descriptors = entries

    @classmethod
    def binary(
        cls,
        source: Pyramid | Picture,
        keypoints: OrientedKeypoints,
        pattern: SamplingPattern | None = None,
    ) -> DescriptorSet:
        """Every keypoint described by the binary reading, at its own scale."""
        pattern = pattern or SamplingPattern()
        pyramid = Pyramid.of(source)
        smoothed = [
            swept(level, SMOOTHING, EdgeRule.EXTEND) for level in pyramid.levels
        ]
        return cls(
            BinaryDescriptor.of(
                pyramid[pyramid.level_of(keypoint.scale)],
                _at_level(keypoint),
                pattern,
                smoothed[pyramid.level_of(keypoint.scale)],
            )
            for keypoint in keypoints
        )

    @classmethod
    def gradient_histograms(
        cls,
        source: Pyramid | Picture,
        keypoints: OrientedKeypoints,
        cells: int = DEFAULT_CELLS,
        cell_side: int = DEFAULT_CELL_SIDE,
    ) -> DescriptorSet:
        """Every keypoint described by the gradient histogram, at its own scale."""
        pyramid = Pyramid.of(source)
        fields = [GradientField.of(level) for level in pyramid.levels]
        return cls(
            GradientHistogramDescriptor.of(
                pyramid[pyramid.level_of(keypoint.scale)],
                _at_level(keypoint),
                cells,
                cell_side,
                fields[pyramid.level_of(keypoint.scale)],
            )
            for keypoint in keypoints
        )

    @property
    def keypoints(self) -> OrientedKeypoints:
        return OrientedKeypoints(entry.keypoint for entry in self._descriptors)

    def matched_to(
        self, other: DescriptorSet, maximum_ratio: float = 0.8
    ) -> Correspondences:
        """Each of these matched to its nearest in ``other``, under the ratio test.

        Raises
        ------
        InvalidValuesError
            If the ratio is not above zero and at most one.
        TooFewValuesError
            If ``other`` holds fewer than two, since there is no runner-up.
        ShapeMismatchError
            If the two sets are of different kinds.
        """
        if not 0.0 < maximum_ratio <= 1.0:
            raise InvalidValuesError(
                f"a ratio test compares a winner against a runner-up, so its "
                f"threshold lies in (0, 1]; got {maximum_ratio}"
            )
        if len(other) < 2:
            raise TooFewValuesError(
                f"the ratio test needs a runner-up, so it needs at least two "
                f"descriptors to search in; got {len(other)}"
            )
        kept: list[Correspondence] = []
        for mine in self._descriptors:
            distances = sorted(
                (mine.distance_to(theirs), index)
                for index, theirs in enumerate(other._descriptors)
            )
            (nearest, which), (runner_up, _) = distances[0], distances[1]
            candidate = Correspondence(
                mine.keypoint, other._descriptors[which].keypoint, nearest, runner_up
            )
            if candidate.ratio <= maximum_ratio:
                kept.append(candidate)
        return Correspondences(kept)

    def __iter__(self) -> Iterator[Descriptor]:
        return iter(self._descriptors)

    def __len__(self) -> int:
        return len(self._descriptors)

    def __getitem__(self, index: int) -> Descriptor:
        return self._descriptors[index]

    def __repr__(self) -> str:
        return (
            f"DescriptorSet({len(self._descriptors)} "
            f"{type(self._descriptors[0]).__name__})"
        )


def _at_level(keypoint: OrientedKeypoint) -> OrientedKeypoint:
    """The keypoint in the pixels of the level it was found at."""
    return OrientedKeypoint(
        Keypoint(
            int(round(keypoint.row / keypoint.scale)),
            int(round(keypoint.column / keypoint.scale)),
            keypoint.strength,
        ),
        keypoint.angle,
        1.0,
    )


class SimilarityTransform:
    """A turn, a change of scale and a shift: how one picture maps onto another.

    A point at ``(row, column)`` in the first picture lands at
    ``scale * R(angle) (row, column) + (row_shift, column_shift)`` in the
    second, where ``R`` turns as the gradient field measures a direction, from
    pointing right towards pointing down.

    Parameters
    ----------
    scale:
        Positive. Two means the second picture shows everything twice as big.
    angle:
        Radians.
    row_shift, column_shift:
        Where the first picture's origin lands.

    Raises
    ------
    InvalidValuesError
        If the scale is not positive or anything is not finite.
    """

    __slots__ = ("_angle", "_column_shift", "_row_shift", "_scale")

    def __init__(
        self,
        scale: float = 1.0,
        angle: float = 0.0,
        row_shift: float = 0.0,
        column_shift: float = 0.0,
    ) -> None:
        for name, value in (
            ("scale", scale),
            ("angle", angle),
            ("row shift", row_shift),
            ("column shift", column_shift),
        ):
            if not np.isfinite(value):
                raise InvalidValuesError(f"a transform's {name} is finite, got {value}")
        if scale <= 0.0:
            raise InvalidValuesError(f"a scale is positive, got {scale}")
        self._scale = float(scale)
        self._angle = _folded(float(angle))
        self._row_shift = float(row_shift)
        self._column_shift = float(column_shift)

    @classmethod
    def from_pairs(cls, first: Any, second: Any) -> SimilarityTransform:
        """The least-squares transform carrying ``first`` points onto ``second``.

        Centre both sets, read the scale off the ratio of spreads, the turn
        off the singular value decomposition of the cross-covariance, and the
        shift off what remains. Exact for two points, least squares for more.

        Raises
        ------
        TooFewValuesError
            With fewer than two pairs, which leave the four numbers unfixed.
        ShapeMismatchError
            If the two sets are not the same number of ``(row, column)`` pairs.
        InvalidValuesError
            If the first set has no spread, so no scale can be read off it.
        """
        source = np.asarray(first, dtype=np.float64).reshape(-1, 2)
        target = np.asarray(second, dtype=np.float64).reshape(-1, 2)
        if source.shape != target.shape:
            raise ShapeMismatchError(
                f"pairs are matched one to one, got {source.shape[0]} "
                f"and {target.shape[0]}"
            )
        if source.shape[0] < 2:
            raise TooFewValuesError(
                f"a similarity has four numbers, which two pairs fix and one cannot; "
                f"got {source.shape[0]}"
            )
        source_centre = source.mean(axis=0)
        target_centre = target.mean(axis=0)
        source_spread = source - source_centre
        target_spread = target - target_centre
        source_variance = float(np.sum(source_spread**2))
        if source_variance == 0.0:
            raise InvalidValuesError(
                "every first point sits in one place, so no turn or scale can be read"
            )
        covariance = target_spread.T @ source_spread
        left, singular, right_transposed = np.linalg.svd(covariance)
        reflection = np.eye(2)
        if np.linalg.det(left @ right_transposed) < 0.0:
            reflection[1, 1] = -1.0
        rotation = left @ reflection @ right_transposed
        scale = float(np.sum(singular * np.diag(reflection))) / source_variance
        angle = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        shift = target_centre - scale * (rotation @ source_centre)
        return cls(scale, angle, float(shift[0]), float(shift[1]))

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def row_shift(self) -> float:
        return self._row_shift

    @property
    def column_shift(self) -> float:
        return self._column_shift

    @property
    def matrix(self) -> FloatArray:
        """The three by three matrix acting on ``(row, column, 1)``."""
        cosine, sine = math.cos(self._angle), math.sin(self._angle)
        return np.asarray(
            [
                [self._scale * cosine, -self._scale * sine, self._row_shift],
                [self._scale * sine, self._scale * cosine, self._column_shift],
                [0.0, 0.0, 1.0],
            ]
        )

    def applied_to(self, points: Any) -> FloatArray:
        """``(n, 2)`` points of the first picture carried into the second."""
        source = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        linear = self.matrix[:2, :2]
        return source @ linear.T + np.asarray([self._row_shift, self._column_shift])

    @property
    def inverse(self) -> SimilarityTransform:
        """The transform carrying the second picture back onto the first."""
        inverse_scale = 1.0 / self._scale
        cosine, sine = math.cos(-self._angle), math.sin(-self._angle)
        shift = -inverse_scale * np.asarray(
            [
                cosine * self._row_shift - sine * self._column_shift,
                sine * self._row_shift + cosine * self._column_shift,
            ]
        )
        return SimilarityTransform(
            inverse_scale, -self._angle, float(shift[0]), float(shift[1])
        )

    def warp(self, picture: Picture, shape: tuple[int, int] | None = None) -> Picture:
        """``picture`` as the second picture would show it.

        Every output pixel looks back through the inverse to where it came
        from and reads the first picture there by bilinear interpolation, with
        zero where it looks outside. ``shape`` defaults to the picture's own.
        """
        height, width = shape or picture.shape
        rows, columns = np.meshgrid(
            np.arange(height, dtype=np.float64),
            np.arange(width, dtype=np.float64),
            indexing="ij",
        )
        back = self.inverse.applied_to(
            np.stack([rows.ravel(), columns.ravel()], axis=1)
        )
        values = _bilinear(picture.values, back[:, 0], back[:, 1]).reshape(
            height, width
        )
        return Picture(values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimilarityTransform) or type(self) is not type(other):
            return NotImplemented
        return (
            self._scale == other._scale
            and self._angle == other._angle
            and self._row_shift == other._row_shift
            and self._column_shift == other._column_shift
        )

    def __hash__(self) -> int:
        return hash((self._scale, self._angle, self._row_shift, self._column_shift))

    def __repr__(self) -> str:
        return (
            f"SimilarityTransform(scale={self._scale:.4g}, angle={self._angle:.4f}, "
            f"row_shift={self._row_shift:.4g}, column_shift={self._column_shift:.4g})"
        )


class AlignmentEstimate:
    """The move that explains the most matches, and which matches it explains.

    Parameters
    ----------
    transform:
        The similarity refitted on every inlier.
    inliers, outliers:
        The matches the transform explains to within the tolerance, and the
        rest.
    inlier_distance:
        The tolerance, in pixels.

    Raises
    ------
    InvalidValuesError
        If the tolerance is negative or not finite.
    """

    __slots__ = ("_inlier_distance", "_inliers", "_outliers", "_transform")

    def __init__(
        self,
        transform: SimilarityTransform,
        inliers: Correspondences,
        outliers: Correspondences,
        inlier_distance: float,
    ) -> None:
        if not np.isfinite(inlier_distance) or inlier_distance < 0.0:
            raise InvalidValuesError(
                f"an inlier distance is finite and not negative, got {inlier_distance}"
            )
        self._transform = transform
        self._inliers = inliers
        self._outliers = outliers
        self._inlier_distance = float(inlier_distance)

    @classmethod
    def of(
        cls,
        matches: Correspondences,
        inlier_distance: float = DEFAULT_INLIER_DISTANCE,
        trials: int = DEFAULT_TRIALS,
        seed: int = 0,
    ) -> AlignmentEstimate:
        """RANSAC over pairs of matches.

        Every pair of matches proposes a transform. With no more pairs than
        ``trials`` every pair is tried in a fixed order; otherwise ``trials``
        pairs are drawn with ``seed``. The proposal explaining the most matches
        wins, ties going to the smaller total error, and the transform is then
        refitted on everything it explained.

        Raises
        ------
        TooFewValuesError
            With fewer than two matches.
        InvalidValuesError
            If the tolerance is not finite or there are no trials, or every
            pair of matches is degenerate.
        """
        if len(matches) < 2:
            raise TooFewValuesError(
                f"two matches fix a similarity and one cannot; got {len(matches)}"
            )
        if trials < 1:
            raise InvalidValuesError(f"at least one pair is tried, got {trials}")
        if not np.isfinite(inlier_distance) or inlier_distance < 0.0:
            raise InvalidValuesError(
                f"an inlier distance is finite and not negative, got {inlier_distance}"
            )
        first = matches.first_positions
        second = matches.second_positions
        every_pair = list(combinations(range(len(matches)), 2))
        if len(every_pair) <= trials:
            tried = every_pair
        else:
            generator = np.random.default_rng(seed)
            drawn = generator.choice(len(every_pair), size=trials, replace=False)
            tried = [every_pair[index] for index in sorted(drawn)]

        best: tuple[int, float, np.ndarray] | None = None
        for one, two in tried:
            if np.array_equal(first[one], first[two]) or np.array_equal(
                second[one], second[two]
            ):
                continue
            proposal = SimilarityTransform.from_pairs(
                first[[one, two]], second[[one, two]]
            )
            errors = np.linalg.norm(proposal.applied_to(first) - second, axis=1)
            agreeing = errors <= inlier_distance
            count = int(np.count_nonzero(agreeing))
            total = float(np.sum(errors[agreeing]))
            if (
                best is None
                or count > best[0]
                or (count == best[0] and total < best[1])
            ):
                best = (count, total, agreeing)
        if best is None:
            raise InvalidValuesError(
                "every pair of matches shares a position on one side, so no pair "
                "can propose a transform"
            )
        agreeing = best[2]
        transform = SimilarityTransform.from_pairs(first[agreeing], second[agreeing])
        errors = np.linalg.norm(transform.applied_to(first) - second, axis=1)
        agreeing = errors <= inlier_distance
        if np.count_nonzero(agreeing) >= 2:
            transform = SimilarityTransform.from_pairs(
                first[agreeing], second[agreeing]
            )
            errors = np.linalg.norm(transform.applied_to(first) - second, axis=1)
            agreeing = errors <= inlier_distance
        inliers = Correspondences(
            match for match, kept in zip(matches, agreeing, strict=True) if kept
        )
        outliers = Correspondences(
            match for match, kept in zip(matches, agreeing, strict=True) if not kept
        )
        return cls(transform, inliers, outliers, inlier_distance)

    @property
    def transform(self) -> SimilarityTransform:
        return self._transform

    @property
    def inliers(self) -> Correspondences:
        return self._inliers

    @property
    def outliers(self) -> Correspondences:
        return self._outliers

    @property
    def inlier_distance(self) -> float:
        return self._inlier_distance

    @property
    def n_inliers(self) -> int:
        return len(self._inliers)

    @property
    def inlier_share(self) -> float:
        """What fraction of the matches the transform explains."""
        total = len(self._inliers) + len(self._outliers)
        return len(self._inliers) / total if total else 0.0

    @property
    def mean_residual(self) -> float:
        """How far, on average, an inlier lands from where the transform sends it."""
        if len(self._inliers) == 0:
            return 0.0
        errors = np.linalg.norm(
            self._transform.applied_to(self._inliers.first_positions)
            - self._inliers.second_positions,
            axis=1,
        )
        return float(np.mean(errors))

    def aligned(self, second: Picture, shape: tuple[int, int] | None = None) -> Picture:
        """``second`` carried back into the first picture's frame."""
        return self._transform.inverse.warp(second, shape)

    def __repr__(self) -> str:
        return (
            f"AlignmentEstimate({self._transform!r}, {len(self._inliers)} inliers, "
            f"{len(self._outliers)} outliers)"
        )


def mean_absolute_difference(first: Picture, second: Picture) -> float:
    """How unlike two pictures are, pixel by pixel, on average.

    Raises
    ------
    ShapeMismatchError
        If they are not the same size.
    """
    if first.shape != second.shape:
        raise ShapeMismatchError(
            f"two pictures are compared pixel by pixel only at one size, got "
            f"{first.shape} and {second.shape}"
        )
    return float(np.mean(np.abs(first.values - second.values)))
