"""Spec for template matching.

The claims here are of two kinds and they are kept apart deliberately.

The first kind is arithmetic a reader can check on paper, and every fixture
that carries one is drawn by hand for that reason. A three by three cross has
five bright pixels, so an exact copy correlates with it at 5.0, a flat patch of
brightness 2.0 correlates at 10.0, and the squared differences against that same
flat patch come to ``5 * 1 + 4 * 4 = 21``. Nothing in those numbers has to be
taken on trust.

The second kind cannot be checked on paper and is not pretending to be: how far
wrong a rotated match lands, and whether a believability ratio separates a
picture containing the target from one that does not. Those are measurements
over fixtures big enough to have somewhere to be wrong, and the absent case
uses a seeded texture rather than a drawing, because a hand-drawn picture that
contains nothing in particular is a hard thing to draw and an easy thing to
draw favourably. The seeds are fixed, so the numbers are pinned exactly even
though no one worked them out.

The single load-bearing test in the module is
``TestBrightnessIsWhatSeparatesTheRules``. Everything else here would pass for
an implementation that had only plain correlation in it.
"""

import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.computer_vision.matching import (
    CENTRES_EACH_PATCH,
    DEFAULT_BELIEVABLE_RATIO,
    HIGHER_IS_BETTER,
    MatchBelievability,
    MatchRule,
    MatchScores,
    ScoredPosition,
    TemplateMatcher,
    checked_template,
    placed,
    scaled_up,
    score_of_an_exact_copy,
    score_surface,
    separation_ratio,
    turned_by_a_right_angle,
)
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import (
    AllSameValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    UndefinedMetricError,
)

SQUARED_DIFFERENCES = MatchRule.SUM_OF_SQUARED_DIFFERENCES
CORRELATION = MatchRule.CORRELATION
NORMALISED = MatchRule.NORMALISED_CROSS_CORRELATION

#: A dim, featureless background. 0.1 rather than 0.0 so that a decoy which
#: scores well by being bright cannot be dismissed as scoring well against
#: nothing.
DIM = 0.1

#: Five bright pixels arranged so that no shift of it overlaps itself, and so
#: that its sums are small enough to work out in a sentence.
CROSS = Picture([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

#: A capital F. Asymmetric under every rotation and every reflection, which is
#: what the rotation claims need, and 25 pixels, which is what the
#: believability claims need.
GLYPH = Picture(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

#: An L. Chosen for the rotation tests and kept for the one about size, because
#: doubling it turned out to produce an exact copy of it.
CORNER = Picture([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

#: A flat square brighter than anything in the template. The decoy.
BRIGHT_SQUARE = Picture(np.full((3, 3), 2.0))

PLAIN_9 = Picture(np.full((9, 9), DIM))
PLAIN_12 = Picture(np.full((12, 12), DIM))
PLAIN_15 = Picture(np.full((15, 15), DIM))
PLAIN_20 = Picture(np.full((20, 20), DIM))


def texture(seed: int, side: int = 15) -> Picture:
    """A picture with no particular thing in it, drawn the same way every time."""
    return Picture(np.random.default_rng(seed).random((side, side)))


class TestAnExactCopyIsFoundWherePutAndScoresPerfectly:
    @pytest.mark.parametrize(
        "rule, perfect",
        [(SQUARED_DIFFERENCES, 0.0), (CORRELATION, 5.0), (NORMALISED, 1.0)],
    )
    def test_the_copy_wins_at_its_own_position_with_the_rule_s_perfect_score(
        self, rule, perfect
    ):
        """Correlation's perfect score is the template's own sum of squares,
        which for five bright pixels is 5.0; the other two are constants."""
        scene = placed(PLAIN_9, CROSS, 1, 1)

        scores = MatchScores.of(scene, CROSS, rule)

        assert (scores.best.row, scores.best.column) == (1, 1)
        assert scores.best.score == pytest.approx(perfect)
        assert score_of_an_exact_copy(rule, CROSS) == pytest.approx(perfect)

    @pytest.mark.parametrize("rule", list(MatchRule))
    def test_the_score_at_the_true_position_is_the_perfect_one(self, rule):
        scores = MatchScores.of(placed(PLAIN_9, CROSS, 1, 1), CROSS, rule)

        assert scores.at(1, 1).score == pytest.approx(
            score_of_an_exact_copy(rule, CROSS)
        )

    def test_the_surface_is_smaller_than_the_picture_by_the_template(self):
        """Only positions where the template fits entirely are scored, because
        a position hanging off the edge would win or lose on invented pixels."""
        surface = score_surface(PLAIN_9, CROSS, NORMALISED)

        assert surface.shape == (7, 7)


class TestBrightnessIsWhatSeparatesTheRules:
    """The load-bearing tests. A flat bright square outscores the real target
    under plain correlation and does not under the normalised rule."""

    @staticmethod
    def fooled_scene() -> Picture:
        """The cross at (1, 1) and a flat square of brightness 2.0 at (5, 5)."""
        return placed(placed(PLAIN_9, CROSS, 1, 1), BRIGHT_SQUARE, 5, 5)

    def test_plain_correlation_prefers_the_bright_square_to_the_target(self):
        """2.0 against each of five bright template pixels is 10.0, where the
        template against itself is 5.0. The rule cannot tell agreement from
        brightness, and here brightness wins by a factor of two."""
        scores = MatchScores.of(self.fooled_scene(), CROSS, CORRELATION)

        assert scores.at(5, 5).score == pytest.approx(10.0)
        assert scores.at(1, 1).score == pytest.approx(5.0)
        assert (scores.best.row, scores.best.column) == (5, 5)

    def test_the_normalised_rule_prefers_the_target_to_the_bright_square(self):
        """Same picture, same template, opposite answer: 1.0 at the target and
        0.0 at the square, which has no deviation pattern to agree with."""
        scores = MatchScores.of(self.fooled_scene(), CROSS, NORMALISED)

        assert scores.at(1, 1).score == pytest.approx(1.0)
        assert scores.at(5, 5).score == pytest.approx(0.0)
        assert (scores.best.row, scores.best.column) == (1, 1)

    def test_the_squared_differences_resist_this_decoy_for_a_different_reason(self):
        """5 * (2 - 1)^2 + 4 * (2 - 0)^2 = 21, against 0 at the copy. This rule
        is not brightness-blind; it simply happens to punish this decoy."""
        scores = MatchScores.of(self.fooled_scene(), CROSS, SQUARED_DIFFERENCES)

        assert scores.at(5, 5).score == pytest.approx(21.0)
        assert scores.at(1, 1).score == pytest.approx(0.0)

    def test_and_the_squared_differences_are_fooled_by_a_change_of_lighting(self):
        """Which is the other half of the argument. The same target brightened
        to 2 * template + 0.4 is 10.44 away from the template, so a matcher run
        on a photograph taken under different light finds nothing at all."""
        brightened = Picture(CROSS.values * 2.0 + 0.4)
        scene = placed(placed(PLAIN_12, CROSS, 1, 1), brightened, 7, 7)

        scores = MatchScores.of(scene, CROSS, SQUARED_DIFFERENCES)

        assert scores.at(7, 7).score == pytest.approx(10.44)

    def test_the_normalised_rule_scores_a_brightened_copy_perfectly(self):
        """Gain and offset are exactly what the centring and the division
        remove, so 2 * template + 0.4 is a perfect match and not a near one."""
        brightened = Picture(CROSS.values * 2.0 + 0.4)
        scene = placed(PLAIN_12, brightened, 7, 7)

        scores = MatchScores.of(scene, CROSS, NORMALISED)

        assert scores.at(7, 7).score == pytest.approx(1.0)

    def test_while_plain_correlation_prefers_the_brightened_copy_to_the_original(
        self,
    ):
        """The failure without even the excuse of a featureless decoy: the
        rule ranks a brightened copy of the target above the target."""
        brightened = Picture(CROSS.values * 2.0 + 0.4)
        scene = placed(placed(PLAIN_12, CROSS, 1, 1), brightened, 7, 7)

        scores = MatchScores.of(scene, CROSS, CORRELATION)

        assert scores.at(7, 7).score == pytest.approx(12.0)
        assert scores.at(1, 1).score == pytest.approx(5.0)
        assert (scores.best.row, scores.best.column) == (7, 7)


class TestItFailsOnRotationAndOnScale:
    def test_an_unturned_copy_is_the_control(self):
        scores = MatchScores.of(placed(PLAIN_15, GLYPH, 4, 4), GLYPH, NORMALISED)

        assert (scores.best.row, scores.best.column) == (4, 4)
        assert scores.best.score == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "quarter_turns, at_the_truth, best_position, best_score, pixels_out",
        [
            (1, 0.038461538461538, (8, 4), 0.520416499866533, 4.0),
            (2, -0.121794871794872, (1, 3), 0.659082043657308, 3.162277660168379),
            (3, 0.038461538461538, (4, 8), 0.520416499866533, 4.0),
        ],
    )
    def test_a_turned_glyph_is_not_found_and_the_answer_given_instead_is_wrong(
        self, quarter_turns, at_the_truth, best_score, best_position, pixels_out
    ):
        """The position the glyph actually occupies scores near nothing, and at
        half a turn it scores *below* nothing, because an upside-down F is
        partly the negative of an F. The winner is a different position several
        pixels away, reported with no sign that anything went wrong."""
        turned = turned_by_a_right_angle(GLYPH, quarter_turns)
        scores = MatchScores.of(placed(PLAIN_15, turned, 4, 4), GLYPH, NORMALISED)

        truth = scores.at(4, 4)
        assert truth.score == pytest.approx(at_the_truth, abs=1e-12)
        assert (scores.best.row, scores.best.column) == best_position
        assert scores.best.score == pytest.approx(best_score, abs=1e-12)
        assert scores.best.pixels_away_from(truth) == pytest.approx(pixels_out)

    def test_a_doubled_glyph_is_not_found_either(self):
        """0.165 where the glyph begins, and a winner 5.099 pixels away at
        0.7806 -- high enough to be reported confidently and nowhere near the
        1.0 an unscaled copy earns."""
        scene = placed(PLAIN_20, scaled_up(GLYPH, 2), 4, 4)

        scores = MatchScores.of(scene, GLYPH, NORMALISED)

        truth = scores.at(4, 4)
        assert truth.score == pytest.approx(0.164971132577477, abs=1e-12)
        assert (scores.best.row, scores.best.column) == (9, 5)
        assert scores.best.score == pytest.approx(0.780624749799800, abs=1e-12)
        assert scores.best.pixels_away_from(truth) == pytest.approx(5.09901951359278)

    def test_a_doubled_L_contains_an_unscaled_L_and_scores_perfectly_elsewhere(self):
        """The surprise of the module. Doubling an L puts an exact copy of the
        original L at the join of its arms, so the matcher answers a flawless
        1.0 at row 5, column 4, where the object starts at row 3, column 3. A
        score is
        no evidence at all that the right thing was found."""
        scene = placed(PLAIN_12, scaled_up(CORNER, 2), 3, 3)

        scores = MatchScores.of(scene, CORNER, NORMALISED)

        assert scores.best.score == pytest.approx(1.0)
        assert (scores.best.row, scores.best.column) == (5, 4)
        assert np.array_equal(np.array(scene.patch_at(5, 4, 3, 3)), np.array(CORNER))


class TestWhatTheSearchCosts:
    @pytest.mark.parametrize(
        "picture_side, template_side, positions",
        [(15, 5, 121), (9, 3, 49), (12, 12, 1), (512, 64, 201601)],
    )
    def test_the_count_of_positions_is_the_number_of_places_it_fits(
        self, picture_side, template_side, positions
    ):
        assert (picture_side - template_side + 1) ** 2 == positions

    def test_the_scores_report_that_count_rather_than_leaving_it_to_be_worked_out(
        self,
    ):
        scores = MatchScores.of(PLAIN_15, GLYPH, NORMALISED)

        assert scores.shape == (11, 11)
        assert scores.n_positions == 121
        assert len(scores) == 121

    def test_every_position_reads_the_whole_template(self):
        """121 positions times 25 pixels. The number a reader underestimates:
        the same sum for a 64 by 64 template on a 512 by 512 picture is
        825,757,696, for one object at one angle at one size."""
        scores = MatchScores.of(PLAIN_15, GLYPH, NORMALISED)

        assert scores.n_pixels_read == 3025
        assert 201601 * 64 * 64 == 825757696

    def test_the_positions_are_iterated_as_objects_rather_than_numbers(self):
        scores = MatchScores.of(PLAIN_15, GLYPH, NORMALISED)

        positions = list(scores)

        assert len(positions) == 121
        assert all(isinstance(position, ScoredPosition) for position in positions)
        assert (positions[0].row, positions[0].column) == (0, 0)
        assert (positions[-1].row, positions[-1].column) == (10, 10)

    def test_the_surface_it_hands_out_cannot_be_written_to(self):
        scores = MatchScores.of(placed(PLAIN_15, GLYPH, 4, 4), GLYPH, NORMALISED)

        with pytest.raises(ValueError):
            scores.surface.values[0, 0] = 99.0


class TestAbsenceIsWhatBelievabilityIsFor:
    @pytest.mark.parametrize("seed", range(6))
    def test_the_best_position_is_returned_whether_or_not_the_target_is_there(
        self, seed
    ):
        """There is no position at which this method answers "not here"."""
        matcher = TemplateMatcher(template=GLYPH)

        best = matcher.best_on(texture(seed))

        assert isinstance(best, ScoredPosition)
        assert best.score > 0.38

    @pytest.mark.parametrize("seed", range(6))
    def test_and_the_ratio_is_what_tells_the_two_cases_apart(self, seed):
        """Present, the ratio clears the default threshold on all six seeds;
        absent, it fails on all six. The best *scores* do not separate them
        nearly so well -- an absent picture reaches 0.5341."""
        matcher = TemplateMatcher(template=GLYPH)
        background = texture(seed)

        absent = matcher.scores_on(background).believability()
        present = matcher.scores_on(placed(background, GLYPH, 4, 4)).believability()

        assert not absent.is_believable()
        assert present.is_believable()
        assert absent.ratio < DEFAULT_BELIEVABLE_RATIO < present.ratio

    @pytest.mark.parametrize(
        "seed, absent_ratio, present_ratio",
        [
            (0, 1.017853668350968, 2.579681203997264),
            (1, 1.183090668884033, 2.306662660280843),
            (2, 1.200169759358281, 1.979140096037214),
            (3, 1.365410710184639, 1.872176405921173),
            (4, 1.036632482720396, 2.124480605883145),
            (5, 1.052701723507502, 2.221646405804611),
        ],
    )
    def test_the_two_ranges_do_not_touch(self, seed, absent_ratio, present_ratio):
        """Pinned exactly, because the threshold was chosen from these numbers
        and a page will quote them. Absent runs 1.0179 to 1.3654 and present
        1.8722 to 2.5797."""
        matcher = TemplateMatcher(template=GLYPH)
        background = texture(seed)

        assert matcher.scores_on(background).believability().ratio == pytest.approx(
            absent_ratio, abs=1e-12
        )
        assert matcher.scores_on(
            placed(background, GLYPH, 4, 4)
        ).believability().ratio == pytest.approx(present_ratio, abs=1e-12)

    def test_a_template_too_small_to_be_distinctive_defeats_the_ratio_entirely(self):
        """Nine pixels match something by chance almost anywhere: the best
        agreement in a texture that contains no cross at all averages 0.7752
        over twenty seeds and reaches 0.8844. The remedy is a bigger template,
        not a better threshold."""
        matcher = TemplateMatcher(
            template=Picture(np.random.default_rng(99).random((3, 3)))
        )

        bests = [matcher.scores_on(texture(seed)).best.score for seed in range(20)]

        assert float(np.mean(bests)) == pytest.approx(0.775220, abs=1e-5)
        assert max(bests) == pytest.approx(0.884399, abs=1e-5)

    @pytest.mark.parametrize(
        "side, mean_chance_agreement",
        [(3, 0.7752203), (5, 0.5046505), (7, 0.3326493)],
    )
    def test_and_a_bigger_template_is_matched_by_chance_much_less_often(
        self, side, mean_chance_agreement
    ):
        matcher = TemplateMatcher(
            template=Picture(np.random.default_rng(99).random((side, side)))
        )

        bests = [matcher.scores_on(texture(seed)).best.score for seed in range(20)]

        assert float(np.mean(bests)) == pytest.approx(mean_chance_agreement, abs=1e-5)

    def test_a_target_present_twice_is_reported_as_not_believable(self):
        """Honest rather than wrong: the ratio answers "is this position the
        only good one", and with two perfect copies the answer is no. The two
        scores are equal, so the ratio is exactly 1.0."""
        twice = placed(placed(PLAIN_15, GLYPH, 1, 1), GLYPH, 9, 9)

        believability = MatchScores.of(twice, GLYPH, NORMALISED).believability()

        assert believability.best.score == pytest.approx(1.0)
        assert believability.runner_up.score == pytest.approx(1.0)
        assert believability.ratio == 1.0
        assert not believability.is_believable()

    def test_the_runner_up_is_never_the_winner_s_own_neighbour(self):
        """A template shifted one pixel still overlaps itself almost entirely,
        so the second-best position of any real match is next door to the best.
        Excluding the overlap is what makes the ratio mean anything."""
        scene = placed(PLAIN_15, GLYPH, 4, 4)

        believability = MatchScores.of(scene, GLYPH, NORMALISED).believability()

        assert (believability.best.row, believability.best.column) == (4, 4)
        assert believability.runner_up.pixels_away_from(believability.best) >= 5.0

    def test_a_search_with_no_second_candidate_refuses_rather_than_inventing_one(self):
        """A picture barely bigger than the template holds one thing, and how
        believable that one thing is cannot be asked of it."""
        scores = MatchScores.of(PLAIN_9, GLYPH, NORMALISED)

        with pytest.raises(UndefinedMetricError, match="no second candidate"):
            scores.believability()

    def test_a_separation_below_one_pixel_is_refused(self):
        scores = MatchScores.of(PLAIN_15, GLYPH, NORMALISED)

        with pytest.raises(InvalidValuesError, match="at least one pixel away"):
            scores.believability(minimum_separation=0)

    def test_a_wider_separation_leaves_fewer_competitors(self):
        scene = placed(texture(0), GLYPH, 4, 4)
        scores = MatchScores.of(scene, GLYPH, NORMALISED)

        close = scores.believability(minimum_separation=1)
        far = scores.believability(minimum_separation=6)

        assert far.ratio >= close.ratio


class TestTheSeparationRatio:
    """Its three clauses, exercised directly, because two of them are reachable
    only from scores no fixture here happens to produce."""

    def test_a_higher_is_better_rule_divides_the_winner_by_the_runner_up(self):
        ratio = separation_ratio(
            NORMALISED, ScoredPosition(0, 0, 1.0), ScoredPosition(9, 9, 0.5)
        )

        assert ratio == pytest.approx(2.0)

    def test_a_lower_is_better_rule_divides_the_other_way_round(self):
        ratio = separation_ratio(
            SQUARED_DIFFERENCES, ScoredPosition(0, 0, 2.0), ScoredPosition(9, 9, 8.0)
        )

        assert ratio == pytest.approx(4.0)

    def test_a_perfect_fit_stands_infinitely_clear(self):
        """No finite score is a fraction of a perfect one."""
        ratio = separation_ratio(
            SQUARED_DIFFERENCES, ScoredPosition(0, 0, 0.0), ScoredPosition(9, 9, 8.0)
        )

        assert math.isinf(ratio)

    @pytest.mark.parametrize("runner_up_score", [0.0, -0.4])
    def test_a_runner_up_that_is_not_positively_correlated_is_no_competitor(
        self, runner_up_score
    ):
        """Dividing by it would answer a negative number, which reads as "not
        believable" when the truth is the opposite."""
        ratio = separation_ratio(
            NORMALISED,
            ScoredPosition(0, 0, 0.9),
            ScoredPosition(9, 9, runner_up_score),
        )

        assert math.isinf(ratio)

    def test_a_winner_that_found_nothing_is_not_clearer_than_nothing(self):
        ratio = separation_ratio(
            NORMALISED, ScoredPosition(0, 0, -0.2), ScoredPosition(9, 9, -0.9)
        )

        assert ratio == 1.0

    def test_two_perfect_fits_are_indistinguishable(self):
        ratio = separation_ratio(
            SQUARED_DIFFERENCES, ScoredPosition(0, 0, 0.0), ScoredPosition(9, 9, 0.0)
        )

        assert ratio == 1.0

    def test_a_believability_whose_runner_up_won_is_refused(self):
        with pytest.raises(InvalidValuesError, match="at least one"):
            MatchBelievability(
                ScoredPosition(0, 0, 1.0), ScoredPosition(9, 9, 2.0), 0.5
            )

    def test_a_threshold_below_one_believes_everything_and_is_refused(self):
        believability = MatchBelievability(
            ScoredPosition(0, 0, 1.0), ScoredPosition(9, 9, 0.5), 2.0
        )

        with pytest.raises(InvalidValuesError, match="believes everything"):
            believability.is_believable(0.5)


class TestTheRulesAndTheirDirections:
    @pytest.mark.parametrize("rule", list(MatchRule))
    def test_every_rule_states_its_direction_and_whether_it_centres(self, rule):
        assert rule in HIGHER_IS_BETTER
        assert rule in CENTRES_EACH_PATCH

    def test_only_the_squared_differences_run_downward(self):
        assert not HIGHER_IS_BETTER[SQUARED_DIFFERENCES]
        assert HIGHER_IS_BETTER[CORRELATION]
        assert HIGHER_IS_BETTER[NORMALISED]

    def test_the_winner_is_the_lowest_score_under_the_squared_differences(self):
        scene = placed(PLAIN_9, CROSS, 1, 1)

        scores = MatchScores.of(scene, CROSS, SQUARED_DIFFERENCES)

        assert scores.best.score == min(position.score for position in scores)

    @pytest.mark.parametrize("rule", [CORRELATION, NORMALISED])
    def test_and_the_highest_under_the_others(self, rule):
        scene = placed(PLAIN_9, CROSS, 1, 1)

        scores = MatchScores.of(scene, CROSS, rule)

        assert scores.best.score == max(position.score for position in scores)

    def test_a_flat_template_is_refused_by_the_rule_that_centres(self):
        """Its deviation pattern is the zero vector, whose direction does not
        exist, so every position is undefined rather than badly scored."""
        with pytest.raises(AllSameValuesError, match="no deviation pattern"):
            checked_template(Picture(np.full((3, 3), 0.5)), NORMALISED)

    @pytest.mark.parametrize("rule", [SQUARED_DIFFERENCES, CORRELATION])
    def test_and_accepted_by_the_rules_that_do_not(self, rule):
        flat = Picture(np.full((3, 3), 0.5))

        assert checked_template(flat, rule) is flat

    def test_a_flat_patch_inside_the_picture_scores_zero_by_convention(self):
        """The template is fine and this one position cannot be scored, which
        is a different situation from a flat template and gets a stated answer
        rather than a refusal. Exactly zero here, because 25 copies of 0.1
        average to 0.1 exactly and leave no deviation at all."""
        surface = score_surface(PLAIN_15, GLYPH, NORMALISED)

        assert surface.values[0, 0] == 0.0

    def test_but_only_exactly_zero_when_the_flat_patch_is_flat_exactly(self):
        """Nine copies of 0.1 do not average to 0.1 exactly, so the leftover
        rounding is divided by rounding. It stays tiny only because the
        leftover is a constant vector and a constant is perpendicular to a
        centred template."""
        surface = score_surface(PLAIN_9, CROSS, NORMALISED)

        assert surface.values[0, 0] != 0.0
        assert abs(surface.values[0, 0]) < 1e-15


class TestScoredPosition:
    def test_it_carries_the_three_things_and_nothing_else(self):
        position = ScoredPosition(3, 4, 0.75)

        assert position.row == 3
        assert position.column == 4
        assert position.score == 0.75

    def test_the_distance_between_two_positions_is_a_straight_line(self):
        assert ScoredPosition(0, 0, 1.0).pixels_away_from(
            ScoredPosition(3, 4, 1.0)
        ) == pytest.approx(5.0)

    @pytest.mark.parametrize("row, column", [(-1, 0), (0, -1)])
    def test_a_position_outside_the_picture_is_refused(self, row, column):
        with pytest.raises(InvalidValuesError, match="not negative"):
            ScoredPosition(row, column, 0.5)

    @pytest.mark.parametrize("score", [np.nan, np.inf])
    def test_a_score_that_is_not_a_number_is_refused(self, score):
        with pytest.raises(InvalidValuesError, match="finite"):
            ScoredPosition(0, 0, score)

    def test_equality_is_by_value_and_defers_to_anything_else(self):
        assert ScoredPosition(1, 2, 0.5) == ScoredPosition(1, 2, 0.5)
        assert ScoredPosition(1, 2, 0.5) != ScoredPosition(1, 2, 0.6)
        assert ScoredPosition(1, 2, 0.5).__eq__("a position") is NotImplemented

    def test_positions_are_hashable_so_they_can_be_collected(self):
        assert len({ScoredPosition(1, 2, 0.5), ScoredPosition(1, 2, 0.5)}) == 1


class TestMatchScores:
    def test_a_position_outside_the_search_is_refused(self):
        scores = MatchScores.of(PLAIN_15, GLYPH, NORMALISED)

        with pytest.raises(InvalidValuesError, match="there is none at row"):
            scores.at(11, 0)

    def test_it_remembers_what_was_carried_across_and_what_scored_it(self):
        scores = MatchScores.of(PLAIN_15, GLYPH, NORMALISED)

        assert scores.template_shape == (5, 5)
        assert scores.rule is NORMALISED

    def test_a_template_with_no_pixels_on_a_side_is_refused(self):
        with pytest.raises(InvalidValuesError, match="at least one pixel"):
            MatchScores(PLAIN_9, NORMALISED, (0, 3))


class TestTheMatcher:
    def test_it_finds_the_best_position_in_one_call(self):
        matcher = TemplateMatcher(template=GLYPH)

        best = matcher.best_on(placed(PLAIN_15, GLYPH, 4, 4))

        assert (best.row, best.column) == (4, 4)

    def test_the_normalised_rule_is_the_default(self):
        assert TemplateMatcher(template=GLYPH).rule is NORMALISED

    def test_it_states_what_a_perfect_fit_would_score(self):
        assert TemplateMatcher(template=CROSS).score_of_an_exact_copy == 1.0
        assert (
            TemplateMatcher(template=CROSS, rule=CORRELATION).score_of_an_exact_copy
            == 5.0
        )

    def test_it_can_hand_back_the_piece_of_the_picture_that_was_found(self):
        scene = placed(PLAIN_15, GLYPH, 4, 4)
        matcher = TemplateMatcher(template=GLYPH)

        found = matcher.patch_at(scene, matcher.best_on(scene))

        assert found == GLYPH

    def test_a_flat_template_is_refused_at_construction_not_at_the_first_search(self):
        """Construction configures, so a refusal that depends on the template
        and the rule together belongs there."""
        with pytest.raises(AllSameValuesError, match="no deviation pattern"):
            TemplateMatcher(template=Picture(np.full((3, 3), 0.5)))

    def test_the_same_flat_template_is_accepted_by_a_rule_that_does_not_centre(self):
        matcher = TemplateMatcher(
            template=Picture(np.full((3, 3), 0.5)), rule=CORRELATION
        )

        assert matcher.template_shape == (3, 3)

    def test_a_misspelled_keyword_is_refused_rather_than_ignored(self):
        """``rule`` written as ``scoring``. Without ``extra="forbid"`` this
        constructs a matcher scoring by the default and raises nothing, which
        is the failure ``test/core/base/test_construction.py`` exists for."""
        constructor: Any = TemplateMatcher  # the indirection is for the checker,
        # which refuses to write the mistake this test is about.

        with pytest.raises(ValidationError):
            constructor(template=GLYPH, scoring=NORMALISED)

    def test_a_template_larger_than_the_picture_has_nowhere_to_be_tried(self):
        matcher = TemplateMatcher(template=GLYPH)

        with pytest.raises(ShapeMismatchError, match="does not fit"):
            matcher.scores_on(Picture(np.zeros((3, 3))))


class TestTheFixtureDrawing:
    """The helpers the claims above are built from, checked so that a failing
    claim cannot be blamed on the drawing."""

    def test_enlarging_repeats_each_pixel_and_invents_no_brightness(self):
        bigger = scaled_up(CORNER, 2)

        assert bigger.shape == (6, 6)
        assert set(np.array(bigger).ravel()) == set(np.array(CORNER).ravel())

    def test_an_enlargement_below_one_is_refused(self):
        with pytest.raises(InvalidValuesError, match="at least once"):
            scaled_up(CORNER, 0)

    def test_four_quarter_turns_come_back_to_the_start(self):
        assert turned_by_a_right_angle(GLYPH, 4) == GLYPH

    def test_a_quarter_turn_is_anticlockwise(self):
        """The top-right pixel of an F is bright and goes to the top left."""
        turned = turned_by_a_right_angle(GLYPH)

        assert turned.values[0, 0] == 1.0
        assert turned.values[0, 4] == 0.0

    def test_a_drawn_thing_really_is_where_the_tests_say_it_is(self):
        scene = placed(PLAIN_15, GLYPH, 4, 4)

        assert scene.patch_at(4, 4, 5, 5) == GLYPH
        assert scene.values[0, 0] == DIM

    def test_a_thing_drawn_off_the_edge_is_refused(self):
        with pytest.raises(InvalidValuesError, match="runs past"):
            placed(PLAIN_9, GLYPH, 7, 7)

    def test_the_background_is_not_written_to(self):
        placed(PLAIN_15, GLYPH, 4, 4)

        assert PLAIN_15.brightest == DIM
