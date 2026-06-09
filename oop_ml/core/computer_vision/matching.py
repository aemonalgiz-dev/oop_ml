"""Carrying a known picture across an unknown one, and scoring every fit.

Why this was the first thing anyone tried
-----------------------------------------
The question "where is this in that" has an answer a reader can write down
without any theory at all: try the small picture at every position of the large
one, score how well it fits, and keep the best position. Nothing is learned,
nothing is estimated, and the answer is exact in the only sense available --
it really is the best-fitting position, because every position was tried. That
is the whole method, and it is why it predates everything else in this package
by decades.

It is also a sweep, which is why this module owns no loop of its own.
:func:`~oop_ml.core.computer_vision.filtering.windows_of` already offers every
patch of a picture as one array, and a scoring rule is a number computed from
one patch and the template. So the search is the sweep asked a fourth question,
after the gradient operators, and the only thing this module supplies is the
question.

The three rules, and the one that is really being argued for
-------------------------------------------------------------
:attr:`MatchRule.SUM_OF_SQUARED_DIFFERENCES` subtracts the template from the
patch and adds up the squares. Zero is a perfect fit and nothing else can reach
it, so *lower is better* here and in nothing else in this module. It is the
literal reading of "how far apart are these two pictures", and its weakness is
that it takes brightness at face value: the same object photographed a stop
brighter is a long way from the template in this measure, although it is
plainly the same object.

:attr:`MatchRule.CORRELATION` multiplies the patch by the template and adds up
the products, which is the sweep of :mod:`filtering` with the template as its
weights. It is here because it is what a reader tries second, and because what
it gets wrong is the most instructive thing in the module. A large product can
be had two ways -- by agreeing with the template, or by simply being bright --
and the rule cannot tell those apart. Measured on the fixture the spec pins: a
flat patch of brightness 2.0 scores 10.0 against a template whose exact copy
scores 5.0, so the rule prefers a featureless bright square to the very thing
it was given to look for. It also prefers a *brightened copy* of the target,
``2 * template + 0.4``, at 12.0, to the target itself at 5.0, which is the same
failure without even the excuse that the decoy was featureless. Neither is a
corner case. Both are what happens wherever a picture has a highlight in it.

:attr:`MatchRule.NORMALISED_CROSS_CORRELATION` repairs that by removing the two
things brightness can do. Each patch has its own mean subtracted, which kills
an added offset, and the result is divided by the length of the remaining
deviation vector, which kills a multiplying gain. What is left is the cosine
between two deviation patterns, lying in ``[-1, 1]``, one exactly when the
patch is ``gain * template + offset`` for any positive gain and any offset at
all. So a patch twice as bright as the template, and a patch with 0.4 added to
every pixel, both score exactly 1.0 -- which the spec pins, because it is the
property the whole rule exists for. On the fixture above it scores the true
target 1.0 and the bright square 0.0.

The centring is the part that is worth being precise about, because the phrase
"normalised cross-correlation" is used in the literature for both this and for
the version that only divides by the norms. That version survives a gain and
not an offset, so a scene photographed against a lighter background defeats it,
and it is not what is implemented here.

A flat template has no deviation pattern at all, so the cosine is undefined at
every position rather than merely awkward, and :func:`checked_template` refuses
it. A flat *patch* inside the picture is different -- the template is fine and
this one position cannot be scored -- and scores 0.0, which is a stated
convention meaning "no agreement", not a measurement.

That convention holds exactly only when the patch is flat exactly. A patch of
25 pixels all reading 0.1 has a mean that comes back exactly 0.1 and deviations
that are exactly zero, so the division is skipped and the answer is 0.0; the
same patch at 9 pixels leaves deviations of about 1e-17 and scores
-7.850462293418875e-17. Both are zero for any purpose, and the reason they
differ is worth knowing: the near-flat case is scored by dividing rounding by
rounding. What saves it is that the leftover deviation of a flat patch is a
*constant* vector, and a constant is perpendicular to a centred template, so
the quotient stays tiny rather than landing anywhere in ``[-1, 1]``. A patch
that genuinely varies by 1e-17 has no such protection and will be scored ±1 in
full confidence, which is a real property of the rule rather than of this
implementation.

Where the method stops working, in numbers
------------------------------------------
It compares pixels at fixed offsets, so it assumes the thing it is looking for
appears at the same size and the same angle as the template. Both assumptions
fail immediately, and "it fails" is worth much less than how much. Measured
with a five by five glyph on a plain background, where an unturned copy scores
1.0 at its own corner:

* Turned by one right angle, the position the glyph actually occupies scores
  **0.0385**, and the best position is four pixels away scoring 0.5204. Turned
  by two, the true position scores **-0.1218** -- worse than nothing, since an
  upside-down glyph is somewhat the negative of itself -- and the winner is
  3.1623 pixels away at 0.6591. The method does not degrade towards the right
  answer. It confidently reports a different one.
* Doubled in size, the glyph's own corner scores **0.165** and the winner is
  5.099 pixels away at 0.7806.

The surprise was in a shape I had picked for the rotation test and then reused.
An L doubled in size *contains an exact copy of itself*, unscaled, at the join
of its two arms -- rows 2 to 4 and columns 1 to 3 of the doubled L are the
original L, pixel for pixel. So the matcher scores a flawless 1.0 at a position
two rows and one column from where the object begins, and a caller reading only
the score
would have no way to know anything had gone wrong. A method that failed loudly
on a change of size would be much easier to live with than one that sometimes
succeeds perfectly at the wrong place.

Both failures are why the next fifty years of the subject went into
descriptions that do not change when the thing does: gradients rather than
brightness, histograms rather than positions, and eventually features detected
at their own scale.

The best position is always returned, which is the sharpest failure
-------------------------------------------------------------------
There is no position at which this method answers "it is not here". Something
always scores highest, and a caller who reads only :attr:`MatchScores.best`
gets a confident answer from a picture that does not contain the target at all.
:class:`MatchBelievability` is what distinguishes the two cases, by the usual
mechanism: compare the best score against the best score anywhere that does not
overlap it. A real match stands clear of everything else; an absent one is one
of many mediocre positions and its ratio sits near 1.

Measured over six seeded textures, with the same glyph either drawn into the
texture or not: present, the ratio runs 1.8722 to 2.5797; absent, 1.0179 to
1.3654. The two ranges do not touch, and
:data:`DEFAULT_BELIEVABLE_RATIO` sits between them. The *scores* alone come
nowhere near separating the two cases -- an absent picture's best score reaches
0.5341, which is high enough to look like something to anyone who was hoping.

Three things about that measurement cost something to learn.

**The ratio test needs a template big enough that chance cannot match it.** The
first fixtures used a three by three template, and the ratio failed completely:
present ran 1.0652 to 1.4655 and absent 1.0403 to 1.2630, overlapping, with the
present case falling below the threshold on five seeds of six. The cause is not
the ratio but the template. Nine pixels are too few to be distinctive, so the
best chance agreement in a fifteen by fifteen texture averages 0.7752 and
reaches 0.8844, against 0.5047 and 0.6322 at five by five and 0.3326 and 0.4285
at seven by seven. A template's distinctiveness is a fact about how many pixels
it has, and no scoring rule repairs having too few.

**A ratio is a statement about separation and not about quality.** A template
that matches nothing at all, in a picture where everything else matches it even
less, gets an excellent ratio from two poor scores. Both numbers are exposed
for that reason and neither substitutes for the other.

**The test says "not believable" exactly when the target is there twice.** A
picture holding two copies of the glyph scores 1.0 at both, so the ratio is
exactly 1.0 and :meth:`MatchBelievability.is_believable` answers False on a
picture containing the target perfectly, twice over. That is the honest reading
of the question asked -- "is this position the only good one" -- and it is a
different question from "is the target here". A caller who wants every
occurrence should read the whole surface rather than its winner.

What it costs
-------------
Every position is tried and every position reads the whole template, so a
``h`` by ``w`` template on an ``H`` by ``W`` picture searches
``(H - h + 1) * (W - w + 1)`` positions and reads ``h * w`` pixels at each,
which :attr:`MatchScores.n_positions` and :attr:`MatchScores.n_pixels_read`
report rather than leaving to be worked out. It is quadratic in the picture's
side and quadratic in the template's, and the number gets large faster than a
reader expects: five by five on fifteen by fifteen is 121 positions and 3,025
pixel reads, while 64 by 64 on 512 by 512 is 201,601 positions and
**825,757,696** pixel reads for one object at one scale at one angle.
Searching several angles and several scales multiplies that by the product of
the two counts, which is the practical reason the method was abandoned for
anything but the case it is still perfect for -- a rigid, unrotated, unscaled
thing, such as a button on a screen or a die on a printed circuit board.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from oop_ml.core.computer_vision.filtering import windows_of
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import (
    AllSameValuesError,
    InvalidValuesError,
    UndefinedMetricError,
)
from oop_ml.core.types import FloatArray, MaskArray


class MatchRule(StrEnum):
    """How well one patch fits the template, scored.

    A closed enum rather than a class per rule, for the reason
    :class:`~oop_ml.core.computer_vision.edges.GradientOperator` gives: none of
    the three takes a parameter, so an enum member names the whole choice and
    there is nothing for an object to hold.
    """

    SUM_OF_SQUARED_DIFFERENCES = "sum_of_squared_differences"
    """Add up the squared gaps, pixel by pixel. Zero is perfect; lower is better."""

    CORRELATION = "correlation"
    """Add up the products. Higher is better, and brightness alone raises it."""

    NORMALISED_CROSS_CORRELATION = "normalised_cross_correlation"
    """The cosine between the two deviation patterns, in ``[-1, 1]``."""


HIGHER_IS_BETTER: dict[MatchRule, bool] = {
    MatchRule.SUM_OF_SQUARED_DIFFERENCES: False,
    MatchRule.CORRELATION: True,
    MatchRule.NORMALISED_CROSS_CORRELATION: True,
}
"""Which way each rule runs, written down once.

Two of the three agree and the first does not, which is exactly the kind of
fact that gets remembered wrongly at the one call site that matters.
"""

CENTRES_EACH_PATCH: dict[MatchRule, bool] = {
    MatchRule.SUM_OF_SQUARED_DIFFERENCES: False,
    MatchRule.CORRELATION: False,
    MatchRule.NORMALISED_CROSS_CORRELATION: True,
}
"""Which rules subtract a mean before scoring, and so need a varying template."""

DEFAULT_BELIEVABLE_RATIO: float = 1.5
"""How far clear of the runner-up a match has to stand to be believed.

Chosen from the measurement in the module docstring rather than from Lowe's
0.8, which is 1.25 read this way up: over six seeded textures a five by five
glyph present scored 1.8722 to 2.5797 and absent 1.0179 to 1.3654, and 1.5 is
the round number in the gap. It is therefore a threshold fitted to one family
of fixtures, which is exactly as much as it claims to be -- a default a caller
replaces, not a rule inside the method. It does not separate the two cases at
all for a three by three template, and no threshold does; see the module
docstring for why that is the template's fault and not the threshold's.
"""


def checked_template(template: Picture, rule: MatchRule) -> Picture:
    """The template of a search, validated against the rule that will score it.

    Raises
    ------
    AllSameValuesError
        If the rule centres each patch and the template has no variation at
        all. Its deviation pattern is then the zero vector, whose direction
        does not exist, so every position is undefined rather than merely
        badly scored.
    """
    if CENTRES_EACH_PATCH[rule] and template.brightest == template.darkest:
        raise AllSameValuesError(
            f"{rule.value} scores the agreement between two deviation patterns, "
            f"and a template of one repeated brightness ({template.brightest}) "
            f"has no deviation pattern to agree with"
        )
    return template


def score_of_an_exact_copy(rule: MatchRule, template: Picture) -> float:
    """What ``rule`` scores where the patch is the template itself.

    Two of the three answer a constant -- zero for the squared differences and
    one for the normalised correlation -- and the third does not, because plain
    correlation has no scale of its own: an exact copy scores the template's
    own sum of squares, and something brighter scores more. That the number
    depends on the template *is* the flaw, so it is stated here rather than
    hidden behind a constant that only two rules could honour.
    """
    if rule is MatchRule.SUM_OF_SQUARED_DIFFERENCES:
        return 0.0
    if rule is MatchRule.NORMALISED_CROSS_CORRELATION:
        return 1.0
    return float(np.sum(template.values**2))


def score_surface(picture: Picture, template: Picture, rule: MatchRule) -> Picture:
    """Score every position the template fits at, as a picture of scores.

    The answer's ``[row, column]`` is the score of the patch whose top-left
    pixel is ``(row, column)``, so it is smaller than the picture by one less
    than each side of the template. That is
    :attr:`~oop_ml.core.computer_vision.filtering.EdgeRule.KEEP_VALID`'s
    convention, and it is the only honest one here: a template hanging half off
    the picture would have to be scored against invented pixels, and the
    position would win or lose on what was invented.

    Raises
    ------
    AllSameValuesError
        If the template cannot be scored by this rule; see
        :func:`checked_template`.
    ShapeMismatchError
        If the template is larger than the picture, so there is no position to
        try it at.
    """
    checked_template(template, rule)
    height, width = template.shape
    patches = windows_of(picture, height, width)
    values = template.values

    if rule is MatchRule.CORRELATION:
        return Picture(np.einsum("ijkl,kl->ij", patches, values))

    if rule is MatchRule.SUM_OF_SQUARED_DIFFERENCES:
        gaps = patches - values
        return Picture(np.einsum("ijkl,ijkl->ij", gaps, gaps))

    centred_patches = patches - patches.mean(axis=(2, 3), keepdims=True)
    centred_template = values - values.mean()
    agreement = np.einsum("ijkl,kl->ij", centred_patches, centred_template)
    patch_lengths = np.sqrt(
        np.einsum("ijkl,ijkl->ij", centred_patches, centred_patches)
    )
    template_length = float(np.sqrt(np.sum(centred_template**2)))
    lengths = patch_lengths * template_length
    return Picture(
        np.divide(
            agreement,
            lengths,
            out=np.zeros_like(agreement),
            where=lengths > 0.0,
        )
    )


class ScoredPosition:
    """One position the template was tried at, and how well it fitted there.

    Parameters
    ----------
    row:
        The top row of the patch, counted from the top of the picture.
    column:
        The left column of the patch.
    score:
        What the rule answered there. Which direction is better is a property
        of the rule and deliberately not carried here, because a position does
        not know what it was scored by and a number that claimed to would be
        two facts wearing one name.

    Raises
    ------
    InvalidValuesError
        If either coordinate is negative, or the score is not finite.
    """

    __slots__ = ("_column", "_row", "_score")

    def __init__(self, row: int, column: int, score: float) -> None:
        if row < 0 or column < 0:
            raise InvalidValuesError(
                f"a position inside a picture is not negative, got row {row} "
                f"and column {column}"
            )
        if not math.isfinite(score):
            raise InvalidValuesError(f"a match score is finite, got {score}")

        self._row = int(row)
        self._column = int(column)
        self._score = float(score)

    @property
    def row(self) -> int:
        """The top row of the patch."""
        return self._row

    @property
    def column(self) -> int:
        """The left column of the patch."""
        return self._column

    @property
    def score(self) -> float:
        """What the rule answered at this position."""
        return self._score

    def pixels_away_from(self, other: ScoredPosition) -> float:
        """How far this position's corner is from another's, in pixels.

        Straight-line distance, which is what a reader means by "the match
        landed four pixels out".
        """
        return float(math.hypot(self._row - other.row, self._column - other.column))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoredPosition) or type(self) is not type(other):
            return NotImplemented
        return (
            self._row == other._row
            and self._column == other._column
            and self._score == other._score
        )

    def __hash__(self) -> int:
        return hash((self._row, self._column, self._score))

    def __repr__(self) -> str:
        return (
            f"ScoredPosition(row={self._row}, column={self._column}, "
            f"score={self._score!r})"
        )


class MatchBelievability:
    """The best position set against the best position that does not overlap it.

    Parameters
    ----------
    best:
        The winning position.
    runner_up:
        The best position whose patch does not overlap the winner's. Adjacent
        positions are excluded rather than compared, because a template shifted
        by one pixel still overlaps itself almost entirely and scores almost as
        well; the second-best position of any genuine match is its own
        neighbour, and a ratio against that would be near 1 for every match
        ever made.
    ratio:
        How far the winner stands clear, on a scale where 1.0 is "no clearer
        than the runner-up" and larger is more believable, whichever direction
        the rule runs in.

    Raises
    ------
    InvalidValuesError
        If the ratio is below one, which would mean the runner-up won.
    """

    __slots__ = ("_best", "_ratio", "_runner_up")

    def __init__(
        self, best: ScoredPosition, runner_up: ScoredPosition, ratio: float
    ) -> None:
        if ratio < 1.0 or math.isnan(ratio):
            raise InvalidValuesError(
                f"the winner is at least as good as the runner-up, so the ratio "
                f"is at least one; got {ratio}"
            )

        self._best = best
        self._runner_up = runner_up
        self._ratio = float(ratio)

    @property
    def best(self) -> ScoredPosition:
        """The winning position."""
        return self._best

    @property
    def runner_up(self) -> ScoredPosition:
        """The best position that does not overlap the winner."""
        return self._runner_up

    @property
    def ratio(self) -> float:
        """How far the winner stands clear, one being not at all.

        Infinite where the runner-up has nothing to divide by: a perfect fit
        under the squared differences, or a runner-up that is not positively
        correlated under either correlation. Both mean the same thing, which is
        that no second candidate exists rather than that one exists and is far
        behind.
        """
        return self._ratio

    def is_believable(self, minimum_ratio: float = DEFAULT_BELIEVABLE_RATIO) -> bool:
        """Whether the winner stands clear enough of the runner-up to be trusted.

        This says the winner is *distinctive*, not that it is *good*. Two
        equally poor scores can produce a large ratio, so a caller who needs to
        know that anything was found at all reads
        :attr:`ScoredPosition.score` on :attr:`best` as well.

        Raises
        ------
        InvalidValuesError
            If the threshold is below one, which every ratio clears.
        """
        if minimum_ratio < 1.0:
            raise InvalidValuesError(
                f"every ratio is at least one, so a threshold below one believes "
                f"everything; got {minimum_ratio}"
            )
        return self._ratio >= minimum_ratio

    def __repr__(self) -> str:
        return (
            f"MatchBelievability(best={self._best!r}, "
            f"runner_up={self._runner_up!r}, ratio={self._ratio!r})"
        )


class MatchScores:
    """Every position the template was tried at, with its score.

    Parameters
    ----------
    surface:
        The scores, one per position, indexed by the patch's top-left pixel.
    rule:
        What produced them, kept because the collection cannot say which
        position won without knowing which direction the rule runs in.
    template_shape:
        The template's ``(height, width)``. Needed to say what the search cost
        and to decide which positions overlap which.

    Raises
    ------
    InvalidValuesError
        If either side of the template is below one, or the surface is larger
        than a template of that size could have produced.
    """

    __slots__ = ("_rule", "_surface", "_template_height", "_template_width")

    def __init__(
        self, surface: Picture, rule: MatchRule, template_shape: tuple[int, int]
    ) -> None:
        template_height, template_width = template_shape
        if template_height < 1 or template_width < 1:
            raise InvalidValuesError(
                f"a template has at least one pixel on each side, got "
                f"{template_height} by {template_width}"
            )

        self._surface = surface
        self._rule = rule
        self._template_height = int(template_height)
        self._template_width = int(template_width)

    @classmethod
    def of(cls, picture: Picture, template: Picture, rule: MatchRule) -> MatchScores:
        """Try ``template`` at every position of ``picture`` and score each."""
        return cls(score_surface(picture, template, rule), rule, template.shape)

    @property
    def surface(self) -> Picture:
        """The scores as a picture, which is frozen like any other picture.

        Handed out because it is what a reader wants to look at -- the peak of
        a real match is visible in it, and so is the absence of one -- and
        because a :class:`~oop_ml.core.computer_vision.picture.Picture` cannot
        be written to, so this is not the collection's own container escaping.
        """
        return self._surface

    @property
    def rule(self) -> MatchRule:
        """What scored these positions."""
        return self._rule

    @property
    def template_shape(self) -> tuple[int, int]:
        """The ``(height, width)`` of what was carried across."""
        return (self._template_height, self._template_width)

    @property
    def shape(self) -> tuple[int, int]:
        """How many positions there are, down and across."""
        return self._surface.shape

    @property
    def n_positions(self) -> int:
        """How many positions were tried, which is every one that fits.

        ``(picture_height - template_height + 1) * (picture_width -
        template_width + 1)``.
        """
        return self._surface.n_pixels

    @property
    def n_pixels_read(self) -> int:
        """How many pixel comparisons the search cost.

        Every position reads the whole template, so this is
        :attr:`n_positions` times the template's own pixel count. It is the
        number that grows faster than a reader expects.
        """
        return self.n_positions * self._template_height * self._template_width

    @property
    def best(self) -> ScoredPosition:
        """The winning position: lowest under the squared differences, highest
        otherwise.

        Ties go to the position nearest the top-left, because the scores are
        read in row-major order and something has to win. Two positions tying
        exactly is what happens when a picture genuinely contains the target
        twice, and the ratio in :meth:`believability` is the thing that says
        so rather than this.
        """
        values = self._surface.values
        if HIGHER_IS_BETTER[self._rule]:
            flat_position = int(np.argmax(values))
        else:
            flat_position = int(np.argmin(values))
        row, column = divmod(flat_position, self._surface.width)
        return ScoredPosition(row, column, float(values[row, column]))

    def at(self, row: int, column: int) -> ScoredPosition:
        """The score of the patch whose top-left pixel is ``(row, column)``.

        Raises
        ------
        InvalidValuesError
            If no patch starts there.
        """
        height, width = self._surface.shape
        if not (0 <= row < height and 0 <= column < width):
            raise InvalidValuesError(
                f"the template fits at {height} by {width} positions, so there "
                f"is none at row {row}, column {column}"
            )
        return ScoredPosition(row, column, float(self._surface.values[row, column]))

    def believability(
        self, minimum_separation: int | None = None
    ) -> MatchBelievability:
        """Set the winner against the best position that does not overlap it.

        Parameters
        ----------
        minimum_separation:
            How far a competing position has to be from the winner, in rows or
            in columns, to count as a different candidate rather than the same
            one shifted. The default is the template's own size, which is
            exactly the separation at which two patches stop overlapping at
            all.

        Raises
        ------
        InvalidValuesError
            If the separation asked for is below one.
        UndefinedMetricError
            If no position is that far from the winner, so there is no second
            candidate to compare against. A picture barely larger than the
            template holds only one thing, and how believable that one thing
            is cannot be asked of it.
        """
        if minimum_separation is not None and minimum_separation < 1:
            raise InvalidValuesError(
                f"a competing position is at least one pixel away, got "
                f"{minimum_separation}"
            )

        winner = self.best
        distant = self._positions_clear_of(winner, minimum_separation)
        values = self._surface.values
        if not distant.any():
            raise UndefinedMetricError(
                f"every position within a {self._surface.height} by "
                f"{self._surface.width} search overlaps the winner at row "
                f"{winner.row}, column {winner.column}, so there is no second "
                f"candidate and believability is undefined"
            )

        scores = np.where(
            distant, values, -np.inf if HIGHER_IS_BETTER[self._rule] else np.inf
        )
        if HIGHER_IS_BETTER[self._rule]:
            flat_position = int(np.argmax(scores))
        else:
            flat_position = int(np.argmin(scores))
        row, column = divmod(flat_position, self._surface.width)
        runner_up = ScoredPosition(row, column, float(values[row, column]))
        return MatchBelievability(
            winner, runner_up, separation_ratio(self._rule, winner, runner_up)
        )

    def _positions_clear_of(
        self, winner: ScoredPosition, minimum_separation: int | None
    ) -> MaskArray:
        """Which positions are far enough from the winner to compete with it."""
        height, width = self._surface.shape
        row_reach = self._template_height
        column_reach = self._template_width
        if minimum_separation is not None:
            row_reach = minimum_separation
            column_reach = minimum_separation
        row_gaps = np.abs(np.arange(height) - winner.row)[:, None]
        column_gaps = np.abs(np.arange(width) - winner.column)[None, :]
        return (row_gaps >= row_reach) | (column_gaps >= column_reach)

    def __iter__(self) -> Iterator[ScoredPosition]:
        """Every position in row-major order, as objects rather than numbers."""
        values = self._surface.values
        height, width = self._surface.shape
        for row in range(height):
            for column in range(width):
                yield ScoredPosition(row, column, float(values[row, column]))

    def __len__(self) -> int:
        return self.n_positions

    def __repr__(self) -> str:
        height, width = self._surface.shape
        return (
            f"MatchScores(rule={self._rule.value!r}, positions={height}x{width}, "
            f"template={self._template_height}x{self._template_width})"
        )


def separation_ratio(
    rule: MatchRule, best: ScoredPosition, runner_up: ScoredPosition
) -> float:
    """How far the winner stands clear of the runner-up, at least one.

    Defined so that larger always means more believable, whichever way the rule
    runs, which takes three clauses rather than one division:

    * Under a lower-is-better rule the ratio is the runner-up's score over the
      winner's. A perfect fit scores zero, and there is nothing to divide by,
      so the answer is infinite -- which is the truth rather than a fudge: no
      finite score is a fraction of a perfect one.
    * Under a higher-is-better rule it is the winner's score over the
      runner-up's. A runner-up at or below zero is evidence of nothing at all,
      so again the winner stands infinitely clear, and dividing by it would
      answer a negative number that reads as "not believable" when the truth is
      the opposite.
    * A winner that is itself at or below zero has found nothing, and nothing
      is not clearer than nothing, so the answer is one.
    """
    if HIGHER_IS_BETTER[rule]:
        if best.score <= 0.0:
            return 1.0
        if runner_up.score <= 0.0:
            return math.inf
        return best.score / runner_up.score

    if best.score == 0.0:
        return math.inf if runner_up.score > 0.0 else 1.0
    return runner_up.score / best.score


class TemplateMatcher(BaseModel):
    """A known picture, and the rule that will score it against unknown ones.

    Construction configures and the search reads: the template and the rule are
    fixed when the matcher is built, so a refusal that depends on the pair of
    them -- a flat template under a rule that centres -- happens there rather
    than at the first picture that happens to be searched.

    Parameters
    ----------
    template:
        What is being looked for.
    rule:
        How a patch is scored against it. Defaults to the normalised
        correlation, because it is the one whose answer survives the lighting
        changing between the template and the picture, which is the ordinary
        case rather than the careful one.

    Raises
    ------
    AllSameValuesError
        If the template has no variation and the rule needs it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    template: Picture
    rule: MatchRule = MatchRule.NORMALISED_CROSS_CORRELATION

    @model_validator(mode="after")
    def _check_the_template_suits_the_rule(self) -> Self:
        checked_template(self.template, self.rule)
        return self

    @property
    def template_shape(self) -> tuple[int, int]:
        """The ``(height, width)`` of what is being looked for."""
        return self.template.shape

    @property
    def score_of_an_exact_copy(self) -> float:
        """What this rule answers where the patch is the template itself."""
        return score_of_an_exact_copy(self.rule, self.template)

    def scores_on(self, picture: Picture) -> MatchScores:
        """Try the template at every position of ``picture`` and score each.

        Raises
        ------
        ShapeMismatchError
            If the template is larger than the picture.
        """
        return MatchScores.of(picture, self.template, self.rule)

    def best_on(self, picture: Picture) -> ScoredPosition:
        """Where the template fits best, which is not the same as where it is.

        A convenience over :meth:`scores_on`, and the one every caller reaches
        for first. It always answers, including on a picture that does not
        contain the template at all, which is why
        :meth:`MatchScores.believability` exists.
        """
        return self.scores_on(picture).best

    def patch_at(self, picture: Picture, position: ScoredPosition) -> Picture:
        """The piece of ``picture`` that scored at ``position``.

        For a caller who wants to look at what was actually found rather than
        take the score's word for it.
        """
        height, width = self.template_shape
        return picture.patch_at(position.row, position.column, height, width)

    def __repr__(self) -> str:
        height, width = self.template_shape
        return f"TemplateMatcher(template={height}x{width}, rule={self.rule.value!r})"


def scaled_up(picture: Picture, factor: int) -> Picture:
    """The same picture with every pixel repeated ``factor`` times each way.

    Nearest-neighbour enlargement, which is the crudest there is and is
    deliberately so: a smoother one would blur the pattern and confound "the
    template matcher cannot cope with a change of size" with "the template
    matcher cannot cope with blurring". Here the enlarged picture holds exactly
    the same brightnesses as the original, arranged over more pixels, so a
    failure to find it is a failure about size and nothing else.

    Raises
    ------
    InvalidValuesError
        If the factor is below one.
    """
    if factor < 1:
        raise InvalidValuesError(
            f"an enlargement repeats each pixel at least once, got {factor}"
        )
    return Picture(np.kron(picture.values, np.ones((factor, factor))))


def turned_by_a_right_angle(picture: Picture, quarter_turns: int = 1) -> Picture:
    """The picture rotated anticlockwise by whole quarter turns.

    Only right angles, because they are the only rotations that move every
    pixel onto another pixel exactly. Any other angle has to interpolate, and
    then a matcher's failure is partly the interpolation's, which is not the
    thing being demonstrated.
    """
    return Picture(np.rot90(picture.values, quarter_turns))


def placed(background: Picture, thing: Picture, row: int, column: int) -> Picture:
    """``background`` with ``thing`` written into it at ``(row, column)``.

    Building the fixtures is half of what this module is for, so the drawing is
    here rather than in the spec: a reader checking a pinned number should be
    able to see that the target really was put where the test says it was.

    Raises
    ------
    InvalidValuesError
        If the thing does not fit there, by the same rule and the same message
        :meth:`~oop_ml.core.computer_vision.picture.Picture.patch_at` uses for
        reading one out.
    """
    height, width = thing.shape
    if row < 0 or column < 0:
        raise InvalidValuesError(
            f"a thing is drawn inside the picture, got row {row} and column {column}"
        )
    if row + height > background.height or column + width > background.width:
        raise InvalidValuesError(
            f"a {height} by {width} thing at row {row}, column {column} runs past "
            f"a {background.height} by {background.width} picture"
        )
    drawn: FloatArray = background.values.copy()
    drawn[row : row + height, column : column + width] = thing.values
    return Picture(drawn)
