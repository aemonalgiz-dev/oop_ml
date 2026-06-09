"""Spec for the integral image, the rectangle features and the cascade.

The fixture, stated rather than left to be inferred
---------------------------------------------------
Twelve by twelve windows. A *target* is a four-row bright bar across the
middle, at 0.85 against a ground of 0.25, jittered one row either way, mixed
three parts to two with a blocky random field and finished with a little
speckle. A *background* is that random field alone. The blocky field is what
makes the problem hard enough to be worth measuring: it has structure at the
scale the features read, so some backgrounds genuinely do look like a bright
band, where per-pixel noise would have averaged away inside the first rectangle
sum and left the two classes trivially apart.

Fifty targets and two thousand backgrounds, which is the ratio the method is
built for -- a detector's negatives are every window of every picture that is
not the thing, and the cascade exists because there are so many of them.

Every number below was measured on that fixture at the seeds written into this
file, not derived. What it comes to:

* A twelve by twelve window admits **10,344** features and a twenty-four by
  twenty-four one **162,336**, which is 281.8 times its own pixel count.
* The fitted cascade has **three stages and five rules**, and the stages reject
  **1895 of 2000**, then **102 of 105**, then **3 of 3** of the training
  backgrounds, keeping all fifty targets at every stage.
* Sweeping a 66 by 66 scene: **3025** windows enter stage one, **235** reach
  stage two, **137** reach stage three, and **111** are accepted. That costs
  **6,657** rule evaluations against **15,125** for the same rules with no
  cascade, and against **31,290,600** for the exhaustive feature bank at every
  window -- a factor of **4,700**.
* The same targets lit from the other side, or turned a quarter, are found
  **0 of 50** times each, where fresh upright ones are found 43 of 50.

The one thing this is not
-------------------------
A face detector. Nothing here was fitted on faces, and the spec is written so
that every claim can be checked on something small.
"""

import timeit

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.computer_vision.cascade import (
    LOOKUPS_PER_RECTANGLE,
    CascadeStage,
    FeatureArrangement,
    FeatureBank,
    HaarCascade,
    IntegralImage,
    Rectangle,
    RectangleFeature,
    RuleDirection,
    ThresholdRule,
    WeightedRule,
)
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    NotFittedError,
    ShapeMismatchError,
)

WINDOW_SIDE = 12
CLUTTER_BLOCK = 3
WHOLE_WINDOW = Rectangle(0, 0, WINDOW_SIDE, WINDOW_SIDE)

N_TARGETS = 50
N_BACKGROUNDS = 2000

# Hand-drawn, so that every feature value below is checkable on paper. Bright
# top half, dark bottom half, four by four.
BRIGHT_OVER_DARK = [[1.0] * 4, [1.0] * 4, [0.0] * 4, [0.0] * 4]

FLAT = [[0.25] * 4] * 4

# A picture whose pixel at (row, column) is 10 * row + column, so that any
# rectangle's sum can be worked out by hand and no two rectangles collide.
COUNTED = [[10.0 * row + column for column in range(7)] for row in range(6)]


def clutter_field(generator: np.random.Generator) -> np.ndarray:
    """A blocky random ground, structured at the scale the features read."""
    coarse = generator.uniform(
        0.15, 0.85, (WINDOW_SIDE // CLUTTER_BLOCK, WINDOW_SIDE // CLUTTER_BLOCK)
    )
    return np.kron(coarse, np.ones((CLUTTER_BLOCK, CLUTTER_BLOCK)))


def speckled(generator: np.random.Generator, values: np.ndarray) -> Picture:
    return Picture(
        np.clip(values + generator.normal(0.0, 0.05, values.shape), 0.0, 1.0)
    )


def target(
    generator: np.random.Generator,
    lit_from_the_other_side: bool = False,
    turned_a_quarter: bool = False,
) -> Picture:
    """A bright bar across the middle of a cluttered ground."""
    values = np.full((WINDOW_SIDE, WINDOW_SIDE), 0.25)
    top = 4 + int(generator.integers(-1, 2))
    bar = slice(top, top + 4)
    if turned_a_quarter:
        values[:, bar] = 0.85
    else:
        values[bar, :] = 0.85
    if lit_from_the_other_side:
        values = 1.10 - values
    return speckled(generator, 0.55 * values + 0.45 * clutter_field(generator))


def background(generator: np.random.Generator) -> Picture:
    """The cluttered ground with no bar in it."""
    return speckled(generator, clutter_field(generator))


def training_set(seed: int) -> tuple[list[Picture], Feature]:
    generator = np.random.default_rng(seed)
    pictures = [target(generator) for _ in range(N_TARGETS)]
    pictures.extend(background(generator) for _ in range(N_BACKGROUNDS))
    return pictures, Feature("target", [1.0] * N_TARGETS + [0.0] * N_BACKGROUNDS)


PLANTED = ((4, 5), (30, 40), (44, 12))
"""Where the three targets sit in the scanned scene."""


def scene_with_planted_targets() -> Picture:
    generator = np.random.default_rng(3)
    values = np.kron(generator.uniform(0.15, 0.85, (22, 22)), np.ones((3, 3)))
    for top, left in PLANTED:
        values[top : top + WINDOW_SIDE, left : left + WINDOW_SIDE] = np.asarray(
            target(generator).values
        )
    return Picture(values)


@pytest.fixture(scope="module")
def fitted_cascade() -> HaarCascade:
    """One fit for the whole module. Measured at 0.62 seconds."""
    pictures, labels = training_set(seed=7)
    return HaarCascade(
        n_stages=5, rules_per_stage=2, n_candidate_features=600, random_seed=0
    ).fit(pictures, labels)


@pytest.fixture(scope="module")
def scan_of_the_scene(fitted_cascade: HaarCascade):
    return fitted_cascade.scan(scene_with_planted_targets())


def every_rectangle_in(height: int, width: int) -> list[Rectangle]:
    return [
        Rectangle(top, left, box_height, box_width)
        for top in range(height)
        for left in range(width)
        for box_height in range(1, height - top + 1)
        for box_width in range(1, width - left + 1)
    ]


class TestTheIntegralImage:
    def test_the_table_is_one_larger_on_each_side_and_starts_at_zero(self):
        """The extra row and column are not padding for its own sake. They are
        what lets a box touching the top edge be read by the same four-lookup
        expression as one in the middle."""
        integral_image = IntegralImage.of(Picture(COUNTED))

        assert integral_image.table.shape == (7, 8)
        assert np.array_equal(integral_image.table[0, :], np.zeros(8))
        assert np.array_equal(integral_image.table[:, 0], np.zeros(7))

    def test_the_last_entry_is_the_whole_picture(self):
        picture = Picture(COUNTED)
        integral_image = IntegralImage.of(picture)

        assert integral_image.table[-1, -1] == pytest.approx(picture.values.sum())

    def test_the_sum_over_every_rectangle_is_the_direct_sum(self):
        """Every box of a six by seven picture, which is 588 of them."""
        picture = Picture(COUNTED)
        integral_image = IntegralImage.of(picture)
        rectangles = every_rectangle_in(picture.height, picture.width)

        assert len(rectangles) == 588
        for rectangle in rectangles:
            direct = picture.values[
                rectangle.top : rectangle.bottom, rectangle.left : rectangle.right
            ].sum()
            assert integral_image.sum_over(rectangle) == pytest.approx(direct)

    def test_a_box_in_the_top_left_corner_needs_no_special_case(self):
        """The case the extra row and column exist for: two of the four
        lookups land in the zero border and the expression is unchanged."""
        picture = Picture(COUNTED)
        integral_image = IntegralImage.of(picture)

        corner = Rectangle(0, 0, 2, 3)

        assert integral_image.sum_over(corner) == pytest.approx(
            picture.values[0:2, 0:3].sum()
        )

    @pytest.mark.parametrize(
        "height, width, n_pixels",
        [(1, 1, 1), (1, 600, 600), (17, 23, 391), (600, 600, 360000)],
    )
    def test_a_rectangle_sum_reads_four_entries_whatever_its_area(
        self, height, width, n_pixels
    ):
        """The whole claim of the integral image, counted rather than asserted.

        A box of one pixel and a box of 360,000 pixels name the same number of
        table entries, which is the four-hundred-thousand-fold difference the
        method is built on.
        """
        integral_image = IntegralImage.of(Picture(np.zeros((600, 600))))
        rectangle = Rectangle(0, 0, height, width)

        corners = integral_image.corners_for(rectangle)

        assert rectangle.n_pixels == n_pixels
        assert len(corners) == LOOKUPS_PER_RECTANGLE == 4

    def test_the_four_named_lookups_add_up_to_the_sum(self):
        """The two routes agree, so counting the corners really is counting
        what ``sum_over`` does rather than a description of it."""
        picture = Picture(COUNTED)
        integral_image = IntegralImage.of(picture)

        for rectangle in every_rectangle_in(picture.height, picture.width):
            from_corners = sum(
                lookup.sign * integral_image.table[lookup.row, lookup.column]
                for lookup in integral_image.corners_for(rectangle)
            )
            assert from_corners == pytest.approx(integral_image.sum_over(rectangle))

    def test_two_of_the_four_lookups_are_subtracted(self):
        corners = IntegralImage.of(Picture(COUNTED)).corners_for(Rectangle(1, 1, 2, 2))

        assert sorted(lookup.sign for lookup in corners) == [-1, -1, 1, 1]

    def test_the_cost_of_a_rectangle_sum_does_not_grow_with_its_area(self):
        """Measured, because "constant time" is the claim and not a flavour.

        A 600 by 600 box holds 360,000 times the pixels of a one-pixel box.
        Measured over 2000 repetitions, the integral image answers the large
        one in 1.06 times what the small one costs, and summing the pixels
        directly costs 136 times as much. The assertions leave a wide margin,
        since a timing test on a shared machine that is tight is a timing test
        that fails for reasons nothing to do with the code.
        """
        picture = Picture(np.random.default_rng(0).random((600, 600)))
        integral_image = IntegralImage.of(picture)
        values = picture.values
        one_pixel = Rectangle(0, 0, 1, 1)
        whole_picture = Rectangle(0, 0, 600, 600)
        repetitions = 2000

        integral_small = timeit.timeit(
            lambda: integral_image.sum_over(one_pixel), number=repetitions
        )
        integral_large = timeit.timeit(
            lambda: integral_image.sum_over(whole_picture), number=repetitions
        )
        direct_small = timeit.timeit(lambda: values[0:1, 0:1].sum(), number=repetitions)
        direct_large = timeit.timeit(
            lambda: values[0:600, 0:600].sum(), number=repetitions
        )

        assert integral_large < integral_small * 3.0
        assert direct_large > direct_small * 20.0

    def test_a_box_running_off_the_picture_is_refused(self):
        integral_image = IntegralImage.of(Picture(COUNTED))

        with pytest.raises(ShapeMismatchError, match="runs past"):
            integral_image.sum_over(Rectangle(4, 0, 3, 3))

    def test_the_table_is_frozen(self):
        integral_image = IntegralImage.of(Picture(COUNTED))

        with pytest.raises(ValueError):
            integral_image.table[0, 0] = 1.0

    @pytest.mark.parametrize(
        "top, left, height, width",
        [(-1, 0, 1, 1), (0, -1, 1, 1)],
    )
    def test_a_rectangle_starting_outside_is_refused(self, top, left, height, width):
        with pytest.raises(InvalidValuesError, match="starts inside"):
            Rectangle(top, left, height, width)

    @pytest.mark.parametrize("height, width", [(0, 3), (3, 0), (-1, 2)])
    def test_a_rectangle_without_pixels_is_refused(self, height, width):
        with pytest.raises(InvalidValuesError, match="at least one pixel"):
            Rectangle(0, 0, height, width)

    def test_a_shifted_rectangle_keeps_its_size(self):
        moved = Rectangle(1, 2, 3, 4).shifted_by(10, 20)

        assert (moved.top, moved.left) == (11, 22)
        assert (moved.height, moved.width) == (3, 4)


class TestTheRectangleFeatures:
    def test_two_stacked_cells_measure_the_contrast_between_them(self):
        """Bright top, dark bottom, so top minus bottom is the whole of it:
        eight ones above and eight zeros below give 8."""
        integral_image = IntegralImage.of(Picture(BRIGHT_OVER_DARK))
        feature = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 0, 0, 2, 4)

        assert feature.value_in(integral_image, Rectangle(0, 0, 4, 4)) == 8.0

    def test_the_same_arrangement_turned_sees_nothing(self):
        """Two cells side by side on a picture whose halves differ top from
        bottom: each cell holds two bright rows and two dark ones, so they
        cancel. This is the whole of what the module's failure section says,
        in four pixels."""
        integral_image = IntegralImage.of(Picture(BRIGHT_OVER_DARK))
        feature = RectangleFeature(FeatureArrangement.TWO_HORIZONTAL, 0, 0, 4, 2)

        assert feature.value_in(integral_image, Rectangle(0, 0, 4, 4)) == 0.0

    def test_three_cells_set_the_middle_against_the_flanks(self):
        """One bright row between two dark ones, read by three one-row cells.

        Rows 0, 1 and 2 are flank, middle, flank, holding 0, 2 and 0, so the
        value is ``-0 + 2 - 0``. That is a line detector, where the two-cell
        arrangements are edge detectors.
        """
        line = [[0.0] * 2, [1.0] * 2, [0.0] * 2, [0.0] * 2]
        integral_image = IntegralImage.of(Picture(line))
        feature = RectangleFeature(FeatureArrangement.THREE_VERTICAL, 0, 0, 1, 2)

        assert feature.value_in(integral_image, Rectangle(0, 0, 4, 2)) == 2.0

    def test_and_answers_zero_when_the_line_is_as_wide_as_its_flanks(self):
        """The same feature on two bright rows out of three: the middle cell
        holds 2 and one flank holds 2, so they cancel. A line detector is
        genuinely blind to an edge, which is the other half of the claim
        above."""
        edge = [[0.0] * 2, [1.0] * 2, [1.0] * 2, [0.0] * 2]
        integral_image = IntegralImage.of(Picture(edge))
        feature = RectangleFeature(FeatureArrangement.THREE_VERTICAL, 0, 0, 1, 2)

        assert feature.value_in(integral_image, Rectangle(0, 0, 4, 2)) == 0.0

    def test_four_cells_set_one_diagonal_against_the_other(self):
        chequer = [[1.0, 0.0], [0.0, 1.0]]
        integral_image = IntegralImage.of(Picture(chequer))
        feature = RectangleFeature(FeatureArrangement.FOUR_CHEQUER, 0, 0, 1, 1)

        assert feature.value_in(integral_image, Rectangle(0, 0, 2, 2)) == 2.0

    @pytest.mark.parametrize(
        "arrangement",
        [
            FeatureArrangement.TWO_HORIZONTAL,
            FeatureArrangement.TWO_VERTICAL,
            FeatureArrangement.FOUR_CHEQUER,
        ],
    )
    def test_the_balanced_arrangements_answer_zero_on_a_flat_picture(self, arrangement):
        """Equal numbers of bright and dark cells cancel exactly, so these
        three see nothing at all in a picture with no contrast in it, however
        bright that picture is."""
        integral_image = IntegralImage.of(Picture(FLAT))
        feature = RectangleFeature(arrangement, 0, 0, 1, 1)

        assert feature.value_in(integral_image, Rectangle(0, 0, 4, 4)) == 0.0
        assert (
            RectangleFeature(arrangement, 0, 0, 1, 1).value_in(
                IntegralImage.of(Picture(np.full((4, 4), 0.9))), Rectangle(0, 0, 4, 4)
            )
            == 0.0
        )

    @pytest.mark.parametrize(
        "arrangement",
        [FeatureArrangement.THREE_HORIZONTAL, FeatureArrangement.THREE_VERTICAL],
    )
    def test_the_three_cell_arrangements_read_the_brightness_as_well(self, arrangement):
        """Measured, because it is the module's own caveat and not a bug.

        One ``+1`` cell against two ``-1`` cells does not cancel, so a flat
        picture of brightness ``b`` and one-pixel cells answers ``-b`` rather
        than zero -- these features move when the lamp moves. That is what the
        original detector's per-window variance normalisation exists for, and
        this module does not do it.
        """
        feature = RectangleFeature(arrangement, 0, 0, 1, 1)
        window = Rectangle(0, 0, 4, 4)

        dim = feature.value_in(IntegralImage.of(Picture(FLAT)), window)
        bright = feature.value_in(
            IntegralImage.of(Picture(np.full((4, 4), 0.9))), window
        )

        assert dim == pytest.approx(-0.25)
        assert bright == pytest.approx(-0.9)

    def test_a_feature_reads_four_lookups_per_cell_and_nothing_else(self):
        integral_image = IntegralImage.of(Picture(np.zeros((40, 40))))
        window = Rectangle(0, 0, 40, 40)
        small = RectangleFeature(FeatureArrangement.FOUR_CHEQUER, 0, 0, 1, 1)
        large = RectangleFeature(FeatureArrangement.FOUR_CHEQUER, 0, 0, 20, 20)

        assert small.n_cells == large.n_cells == 4
        for feature in (small, large):
            lookups = sum(
                len(integral_image.corners_for(cell.rectangle))
                for cell in feature.cells_in(window)
            )
            assert lookups == 4 * LOOKUPS_PER_RECTANGLE

    @pytest.mark.parametrize(
        "side, expected",
        [(4, 136), (6, 669), (8, 2056), (12, 10344)],
    )
    def test_the_formula_and_the_enumeration_agree_on_how_many_there_are(
        self, side, expected
    ):
        """Two routes to one number, as everywhere else here: the arithmetic
        that counts positions per cell size, and actually building them all."""
        assert FeatureBank.count_available(side, side) == expected
        assert len(FeatureBank.exhaustive(side, side)) == expected

    def test_how_many_features_a_window_admits_is_why_a_cascade_exists(self):
        """The count that makes the method necessary.

        A twenty-four by twenty-four window -- the one Viola and Jones used --
        holds 576 pixels and admits 162,336 features, which is 281.8 times its
        own pixel count. Evaluating all of them at every window position of
        even a small picture is not slow, it is out of the question, and every
        design decision after this one follows from that number.
        """
        pixels = 24 * 24
        available = FeatureBank.count_available(24, 24)

        assert available == 162336
        assert available / pixels == pytest.approx(281.83, abs=0.01)
        assert FeatureBank.count_available(12, 12) == 10344

    def test_the_growth_is_faster_than_the_area(self):
        """Doubling the window's side multiplies its pixels by four and its
        features by nearly sixteen."""
        small = FeatureBank.count_available(12, 12)
        large = FeatureBank.count_available(24, 24)

        assert large / small == pytest.approx(15.69, abs=0.01)

    def test_every_arrangement_is_enumerated(self):
        bank = FeatureBank.exhaustive(8, 8)

        assert {feature.arrangement for feature in bank} == set(FeatureArrangement)

    def test_a_feature_that_runs_off_its_window_is_refused(self):
        integral_image = IntegralImage.of(Picture(FLAT))
        feature = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 3, 0, 2, 2)

        assert not feature.fits_in(4, 4)
        with pytest.raises(ShapeMismatchError, match="does not fit"):
            feature.value_in(integral_image, Rectangle(0, 0, 4, 4))

    def test_a_bank_refuses_a_feature_its_window_cannot_hold(self):
        feature = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 0, 0, 9, 1)

        with pytest.raises(ShapeMismatchError, match="does not fit"):
            FeatureBank([feature], 8, 8)

    def test_the_vectorised_route_agrees_with_the_single_window_route(self):
        """Bit for bit, not merely closely. Fitting searches thresholds over
        the matrix and predicting reads one window at a time, so a rule chosen
        on one route and applied on the other would sit on the wrong side of
        its own threshold if the two ever parted.

        This caught the difference it exists to catch. Totalling a feature's
        cells with the builtin ``sum`` disagreed with the vectorised route on 1
        of these 200 values by 2.2e-16, because since Python 3.12 ``sum``
        compensates its rounding over floats and numpy does not. The accurate
        route was the wrong one; both are naive now.
        """
        generator = np.random.default_rng(11)
        pictures = [Picture(generator.random((8, 8))) for _ in range(5)]
        integral_images = [IntegralImage.of(picture) for picture in pictures]
        window = Rectangle(0, 0, 8, 8)
        bank = FeatureBank.exhaustive(8, 8).sample(40, np.random.default_rng(2))

        values = bank.values_over(integral_images, window)

        one_at_a_time = np.array(
            [
                [feature.value_in(integral_image, window) for feature in bank]
                for integral_image in integral_images
            ]
        )
        assert np.array_equal(values, one_at_a_time)

    def test_a_sampled_bank_is_a_subset_and_the_seed_settles_which(self):
        bank = FeatureBank.exhaustive(8, 8)

        first = bank.sample(50, np.random.default_rng(4))
        again = bank.sample(50, np.random.default_rng(4))
        other = bank.sample(50, np.random.default_rng(5))

        assert len(first) == 50
        assert list(first) == list(again)
        assert list(first) != list(other)
        assert set(first).issubset(set(bank))

    @pytest.mark.parametrize("n_features", [0, 3000])
    def test_a_sample_larger_than_the_bank_or_empty_is_refused(self, n_features):
        with pytest.raises(InvalidValuesError, match="between 1 and"):
            FeatureBank.exhaustive(8, 8).sample(n_features, np.random.default_rng(0))

    def test_a_feature_is_a_value_object(self):
        first = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 1, 2, 3, 4)
        same = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 1, 2, 3, 4)
        other = RectangleFeature(FeatureArrangement.TWO_HORIZONTAL, 1, 2, 3, 4)

        assert first == same
        assert hash(first) == hash(same)
        assert first != other
        assert first.__eq__("a feature") is NotImplemented


class TestTheRules:
    def test_a_rule_votes_on_one_side_of_its_threshold(self):
        feature = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 0, 0, 2, 4)
        above = ThresholdRule(feature, 4.0, RuleDirection.ABOVE)
        below = ThresholdRule(feature, 4.0, RuleDirection.BELOW)

        assert above.votes_for_value(8.0)
        assert not above.votes_for_value(4.0)
        assert below.votes_for_value(4.0)
        assert not below.votes_for_value(8.0)

    def test_the_two_directions_partition_every_measurement(self):
        feature = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 0, 0, 2, 4)
        values = np.linspace(-5.0, 5.0, 21)

        above = ThresholdRule(feature, 0.5, RuleDirection.ABOVE).votes_for_values(
            values
        )
        below = ThresholdRule(feature, 0.5, RuleDirection.BELOW).votes_for_values(
            values
        )

        assert np.array_equal(above, ~below)

    def test_a_rule_reads_the_window_through_the_integral_image(self):
        integral_image = IntegralImage.of(Picture(BRIGHT_OVER_DARK))
        feature = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 0, 0, 2, 4)
        rule = ThresholdRule(feature, 4.0, RuleDirection.ABOVE)

        assert rule.votes_for(integral_image, Rectangle(0, 0, 4, 4))

    def test_a_silent_rule_cannot_join_a_stage(self):
        feature = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 0, 0, 2, 4)
        rule = ThresholdRule(feature, 4.0, RuleDirection.ABOVE)

        with pytest.raises(InvalidValuesError, match="voice is positive"):
            WeightedRule(rule, 0.0)

    def test_a_stage_needs_a_rule_and_a_share_that_is_a_share(self):
        feature = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 0, 0, 2, 4)
        rule = WeightedRule(ThresholdRule(feature, 4.0, RuleDirection.ABOVE), 1.0)

        with pytest.raises(EmptyValuesError, match="at least one rule"):
            CascadeStage([])
        with pytest.raises(InvalidValuesError, match="between zero and one"):
            CascadeStage([rule], 1.5)

    def test_a_stage_reports_the_share_of_the_voice_a_window_attracts(self):
        integral_image = IntegralImage.of(Picture(BRIGHT_OVER_DARK))
        window = Rectangle(0, 0, 4, 4)
        feature = RectangleFeature(FeatureArrangement.TWO_VERTICAL, 0, 0, 2, 4)
        agreeing = WeightedRule(ThresholdRule(feature, 4.0, RuleDirection.ABOVE), 3.0)
        dissenting = WeightedRule(ThresholdRule(feature, 4.0, RuleDirection.BELOW), 1.0)

        stage = CascadeStage([agreeing, dissenting])

        assert stage.total_voice == 4.0
        assert stage.confidence_in(integral_image, window) == pytest.approx(0.75)
        assert stage.accepts(integral_image, window)
        assert not stage.with_acceptance_share(0.9).accepts(integral_image, window)


class TestTheCascade:
    def test_it_separates_the_fixture_it_was_fitted_on(self, fitted_cascade):
        pictures, labels = training_set(seed=7)

        evaluation = fitted_cascade.evaluate(pictures, labels)

        assert evaluation.accuracy == 1.0
        assert evaluation.recall == 1.0
        assert evaluation.precision == 1.0

    def test_and_holds_up_on_a_fresh_draw_of_the_same_fixture(self, fitted_cascade):
        """Measured: accuracy 0.9980, recall 0.94, precision 0.9792. The
        recall is what a detection rate of one on the training positives buys
        and does not promise -- the calibration keeps every positive it *saw*."""
        pictures, labels = training_set(seed=99)

        evaluation = fitted_cascade.evaluate(pictures, labels)

        assert evaluation.accuracy == pytest.approx(0.9980, abs=0.0005)
        assert evaluation.recall == pytest.approx(0.94, abs=0.01)
        assert evaluation.precision == pytest.approx(0.9792, abs=0.0005)

    def test_the_shape_of_the_cascade_it_found(self, fitted_cascade):
        assert fitted_cascade.n_stages_fitted == 3
        assert fitted_cascade.n_rules == 5
        assert fitted_cascade.window_shape == (12, 12)
        assert len(fitted_cascade.candidate_features) == 600

    def test_what_each_stage_rejected(self, fitted_cascade):
        """The shape the method is named for: each stage is handed what the
        one before it let through, so the counts fall away sharply and the
        later stages face a small, hard problem."""
        reaching = [one.negatives_reaching for one in fitted_cascade.stage_outcomes]
        rejected = [one.negatives_rejected for one in fitted_cascade.stage_outcomes]

        assert reaching == [2000, 105, 3]
        assert rejected == [1895, 102, 3]

    def test_every_stage_keeps_every_positive_it_was_shown(self, fitted_cascade):
        """The trade the whole design rests on. A negative let through costs
        the next stage a little work; a positive dropped is the target lost for
        good, because no later stage ever sees it again."""
        assert [one.positives_kept for one in fitted_cascade.stage_outcomes] == [
            50,
            50,
            50,
        ]

    def test_the_first_stage_alone_would_be_a_poor_classifier(self, fitted_cascade):
        """No stage is any good by itself, and that is the design rather than
        a shortfall. The first stage lets 117 of 2000 held-out backgrounds
        through where the cascade lets 1, because the accuracy comes from the
        stages multiplying rather than from any of them being severe."""
        pictures, labels = training_set(seed=99)
        backgrounds = [
            picture
            for picture, label in zip(pictures, labels.values, strict=True)
            if label == 0.0
        ]
        first = fitted_cascade.stages[0]

        through_the_first = sum(
            first.accepts(IntegralImage.of(picture), WHOLE_WINDOW)
            for picture in backgrounds
        )
        through_them_all = int(np.asarray(fitted_cascade.predict(backgrounds)).sum())

        assert through_the_first == 117
        assert through_them_all == 1

    def test_the_share_of_stages_survived_is_not_a_probability(self, fitted_cascade):
        """It is exposed under the frame's name and says so. A window rejected
        by the first of three stages scores 0.0 and one accepted scores 1.0,
        and the values in between are thirds rather than anything calibrated."""
        generator = np.random.default_rng(21)
        pictures = [target(generator) for _ in range(20)]
        pictures.extend(background(generator) for _ in range(20))

        shares = np.asarray(fitted_cascade.predict_probability(pictures))

        assert set(np.round(shares * 3).astype(int)) <= {0, 1, 2, 3}
        assert np.array_equal(
            np.asarray(fitted_cascade.predict(pictures)), (shares == 1.0).astype(float)
        )

    def test_reading_anything_learned_before_fitting_raises(self):
        cascade = HaarCascade()

        for reading in (
            lambda: cascade.stages,
            lambda: cascade.stage_outcomes,
            lambda: cascade.candidate_features,
            lambda: cascade.window_shape,
            lambda: cascade.predict([Picture(FLAT)]),
            lambda: cascade.scan(Picture(FLAT)),
        ):
            with pytest.raises(NotFittedError):
                reading()

    def test_a_window_of_the_wrong_size_is_refused(self, fitted_cascade):
        with pytest.raises(ShapeMismatchError, match="call scan"):
            fitted_cascade.predict([Picture(np.zeros((8, 8)))])

    def test_training_windows_of_different_sizes_are_refused(self):
        pictures = [Picture(np.zeros((4, 4))), Picture(np.zeros((5, 5)))]

        with pytest.raises(ShapeMismatchError, match="one window size"):
            HaarCascade().fit(pictures, Feature("target", [1.0, 0.0]))

    def test_a_label_that_is_not_a_label_is_refused(self):
        generator = np.random.default_rng(1)
        pictures = [target(generator), background(generator)]

        with pytest.raises(NonBinaryLabelsError):
            HaarCascade().fit(pictures, Feature("target", [1.0, 2.0]))

    def test_a_label_per_window_is_required(self):
        generator = np.random.default_rng(1)
        pictures = [target(generator), background(generator)]

        with pytest.raises(NonEqualArrayLengthError, match="2 training windows"):
            HaarCascade().fit(pictures, Feature("target", [1.0, 0.0, 1.0]))

    def test_fitting_on_nothing_is_refused(self):
        with pytest.raises(EmptyValuesError, match="at least one window"):
            HaarCascade().fit([], Feature("target", [1.0]))

    def test_a_hyperparameter_it_does_not_have_is_refused(self):
        """The guard every configurable object in this library inherits: a
        misspelling has to raise rather than leave a plausible default."""
        with pytest.raises(ValidationError):
            HaarCascade(stages=3)  # pyright: ignore[reportCallIssue]


class TestWhatTheCascadeSaves:
    def test_most_windows_die_at_the_first_stage(self, scan_of_the_scene):
        """The load-bearing measurement. Sweeping a 66 by 66 scene at a stride
        of one: 3025 windows enter the first stage, 235 survive to the second
        and 137 to the third. The cheap stage runs on everything and the later
        ones on 7.8% and 4.5% of it."""
        reaching = scan_of_the_scene.windows_reaching

        assert reaching == (3025, 235, 137)
        assert reaching[1] / reaching[0] == pytest.approx(0.0777, abs=0.001)
        assert reaching[2] / reaching[0] == pytest.approx(0.0453, abs=0.001)

    def test_the_ordering_is_what_makes_it_cheap(self, scan_of_the_scene):
        """Against the same five rules with no cascade, which is the honest
        comparison for the ordering alone: 6,657 rule evaluations against
        15,125, a factor of 2.27. Modest, and it is modest for a reason worth
        stating -- the saving is bounded by how many rules there are to skip,
        and this cascade has five. A twenty-stage cascade of six thousand rules
        skips almost all of them almost always, which is where the famous
        numbers come from."""
        assert scan_of_the_scene.rule_evaluations == 6657
        assert scan_of_the_scene.rule_evaluations_without_the_cascade == 15125
        assert scan_of_the_scene.saving == pytest.approx(2.272, abs=0.001)

    def test_and_against_evaluating_every_feature_everywhere(
        self, fitted_cascade, scan_of_the_scene
    ):
        """The comparison the method actually exists to win. A twelve by twelve
        window admits 10,344 features, so answering "is the target here" by
        measuring all of them at all 3025 positions is 31,290,600 feature
        evaluations. The cascade spent 6,657, which is 4,700 times fewer, and
        the ratio grows with the window because the feature count does.
        """
        windows = scan_of_the_scene.windows_examined
        exhaustive = windows * FeatureBank.count_available(12, 12)
        searched = windows * len(fitted_cascade.candidate_features)

        assert exhaustive == 31_290_600
        assert searched == 1_815_000
        assert exhaustive / scan_of_the_scene.rule_evaluations == pytest.approx(
            4700.0, rel=0.01
        )
        assert searched / scan_of_the_scene.rule_evaluations == pytest.approx(
            272.6, rel=0.01
        )

    def test_it_finds_the_planted_targets_exactly(self, scan_of_the_scene):
        found = {(window.top, window.left) for window in scan_of_the_scene.accepted}

        assert set(PLANTED) <= found

    def test_three_stages_still_accept_far_too_much(self, scan_of_the_scene):
        """Honest rather than flattering. 111 of 3025 windows survive, and
        three of them are the planted targets, so 108 are false. That is what a
        three-stage cascade is: each stage keeps every positive and therefore
        keeps a good share of the negatives too, and 0.052 x 0.029 is small but
        is not zero. The answer is more stages, which is why the published
        cascade has thirty-eight."""
        assert scan_of_the_scene.n_accepted == 111
        assert scan_of_the_scene.n_accepted / scan_of_the_scene.windows_examined == (
            pytest.approx(0.0367, abs=0.001)
        )

    def test_a_longer_stride_looks_at_fewer_windows(self, fitted_cascade):
        coarse = fitted_cascade.scan(scene_with_planted_targets(), stride=3)

        assert coarse.windows_examined == 19 * 19
        assert coarse.rule_evaluations < 6657

    def test_a_stride_below_one_is_refused(self, fitted_cascade):
        with pytest.raises(InvalidValuesError, match="at least one pixel at a time"):
            fitted_cascade.scan(scene_with_planted_targets(), stride=0)

    def test_a_scene_smaller_than_the_window_is_refused(self, fitted_cascade):
        with pytest.raises(ShapeMismatchError, match="does not fit"):
            fitted_cascade.scan(Picture(np.zeros((8, 8))))


class TestWhereItFails:
    def test_the_same_target_lit_from_the_other_side_is_never_found(
        self, fitted_cascade
    ):
        """Not "found less often". Found zero times out of fifty.

        The arrangements are signed, so a feature that says "bright here,
        darker there" answers the inverted pattern with the opposite sign, and
        every rule thresholding it in one direction rejects the inversion
        outright. There is nothing gradual about it.
        """
        generator = np.random.default_rng(5)
        inverted = [target(generator, lit_from_the_other_side=True) for _ in range(50)]

        assert float(np.asarray(fitted_cascade.predict(inverted)).sum()) == 0.0

    def test_the_same_target_turned_a_quarter_is_never_found(self, fitted_cascade):
        """The arrangements are axis-aligned, and the vertical three-cell
        feature and the horizontal one are different features. A cascade
        fitted on one has never seen the other, which is why a real detector
        fits one cascade per pose."""
        generator = np.random.default_rng(5)
        turned = [target(generator, turned_a_quarter=True) for _ in range(50)]

        assert float(np.asarray(fitted_cascade.predict(turned)).sum()) == 0.0

    def test_while_a_fresh_upright_target_still_is(self, fitted_cascade):
        """The control, without which the two tests above would pass for a
        cascade that had learned nothing at all. 43 of 50."""
        generator = np.random.default_rng(5)
        upright = [target(generator) for _ in range(50)]

        assert float(np.asarray(fitted_cascade.predict(upright)).sum()) == 43.0

    def test_the_feature_the_first_rule_reads_changes_sign_under_inversion(
        self, fitted_cascade
    ):
        """The mechanism behind the two failures above, measured on one number
        rather than inferred from the outcome. The same feature, on the same
        target lit the other way, answers with the opposite sign."""
        generator = np.random.default_rng(31)
        upright = IntegralImage.of(target(generator))
        generator = np.random.default_rng(31)
        inverted = IntegralImage.of(target(generator, lit_from_the_other_side=True))
        feature = fitted_cascade.stages[0].weighted_rules[0].rule.feature

        upright_value = feature.value_in(upright, WHOLE_WINDOW)
        inverted_value = feature.value_in(inverted, WHOLE_WINDOW)

        assert upright_value * inverted_value < 0.0
