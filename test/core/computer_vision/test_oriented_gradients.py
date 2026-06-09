"""Spec for the histogram of oriented gradients.

Every fixture here is drawn rather than photographed, for the foundation
spec's reason: each number below is meant to be checkable against the geometry
of the thing that produced it. A patch whose bottom half is bright has one
edge, running left to right, whose gradient points straight down at ninety
degrees, and a histogram that puts the weight anywhere but the bucket
containing ninety degrees is wrong rather than merely different.

The linear ramps are the sharpest fixture of the set and are worth explaining
once. ``ramp(30)`` is a picture whose brightness is
``cos(30) * column + sin(30) * row``, so its gradient is the constant vector
``(cos 30, sin 30)`` at every interior pixel -- and exactly that, since every
operator here is exact on a straight ramp. So a cell that touches no border
holds one direction, known to the degree, and can be asked which bucket it
went into and how much of it went there.

The numbers pinned below are measured, not asserted from the definitions, and
several of them are quoted in the module docstring.
"""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.computer_vision.filtering import EdgeRule
from oop_ml.core.computer_vision.oriented_gradients import (
    CLIP_LIMIT,
    AngleRange,
    BlockNormalisation,
    CellHistograms,
    HistogramOfOrientedGradients,
    OrientedGradientDescription,
    VoteSharing,
    normalised_block,
    votes_by_bucket,
)
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError


def horizontal_edge(side: int = 16) -> Picture:
    """Dark top, bright bottom. One edge; its gradient points down, at 90."""
    values = np.zeros((side, side))
    values[side // 2 :, :] = 1.0
    return Picture(values)


def vertical_edge(side: int = 16) -> Picture:
    """Dark left, bright right. One edge; its gradient points right, at 0."""
    values = np.zeros((side, side))
    values[:, side // 2 :] = 1.0
    return Picture(values)


def bright_bar(side: int = 16) -> Picture:
    """A bright vertical bar on a dark ground: one edge and its reverse."""
    values = np.zeros((side, side))
    values[:, 5:11] = 1.0
    return Picture(values)


def ramp(degrees: float, side: int = 24) -> Picture:
    """A picture whose gradient is the same known direction everywhere."""
    rows, columns = np.mgrid[0:side, 0:side]
    angle = np.deg2rad(degrees)
    return Picture(np.cos(angle) * columns + np.sin(angle) * rows)


def shapes(side: int = 24) -> Picture:
    """Three bright rectangles, deliberately not symmetric under a turn."""
    values = np.zeros((side, side))
    values[3:11, 4:19] = 1.0
    values[14:21, 2:9] = 0.6
    values[16:19, 13:22] = 0.35
    return Picture(values)


FLAT = Picture(np.full((16, 16), 0.5))


class TestVoting:
    def test_a_straight_edge_puts_every_vote_in_the_bucket_its_geometry_names(self):
        """The gradient of a dark-over-bright edge points straight down, and
        ninety degrees is the centre of bucket 4 of 9 over half a circle."""
        histograms = HistogramOfOrientedGradients().cell_histograms(horizontal_edge())

        per_bucket = histograms.counts.sum(axis=(0, 1))

        assert histograms.total_weight == pytest.approx(128.0)
        assert per_bucket[4] / histograms.total_weight == pytest.approx(1.0)
        assert histograms.bucket_centre(4) == pytest.approx(np.pi / 2)
        assert np.rad2deg(histograms.bucket_centre(4)) == pytest.approx(90.0)

    def test_and_every_cell_the_edge_crosses_says_the_same(self):
        histograms = HistogramOfOrientedGradients().cell_histograms(horizontal_edge())

        assert histograms.n_cell_rows == 2
        assert histograms.n_cell_columns == 2
        assert all(
            histograms.fullest_bucket_at(row, column) == 4
            for row in range(2)
            for column in range(2)
        )

    @pytest.mark.parametrize(
        "degrees, expected_bucket",
        [(0.0, 0), (30.0, 1), (90.0, 4), (135.0, 6)],
    )
    def test_a_ramp_lands_wholly_in_the_bucket_containing_its_own_angle(
        self, degrees, expected_bucket
    ):
        """An interior cell of a ramp sees one direction and nothing else, so
        under the whole-vote rule every one of its votes goes to one bucket."""
        histograms = HistogramOfOrientedGradients(
            vote_sharing=VoteSharing.WHOLE_TO_NEAREST
        ).cell_histograms(ramp(degrees))

        cell = histograms.histogram_at(1, 1)

        assert int(np.argmax(cell)) == expected_bucket
        assert cell.max() / cell.sum() == pytest.approx(1.0)

    def test_a_flat_patch_votes_nothing_anywhere(self):
        histograms = HistogramOfOrientedGradients().cell_histograms(FLAT)

        assert histograms.total_weight == pytest.approx(0.0)

    def test_the_weight_of_a_vote_is_how_sharp_the_edge_is(self):
        """Twice the contrast, twice the weight, in the same bucket."""
        faint = HistogramOfOrientedGradients().cell_histograms(
            Picture(np.asarray(horizontal_edge()) * 0.5)
        )
        sharp = HistogramOfOrientedGradients().cell_histograms(horizontal_edge())

        assert sharp.total_weight == pytest.approx(2.0 * faint.total_weight)
        assert sharp.fullest_bucket_at(0, 0) == faint.fullest_bucket_at(0, 0)

    def test_one_bucket_would_describe_nothing_and_is_refused(self):
        with pytest.raises(InvalidValuesError, match="at least two buckets"):
            votes_by_bucket(np.zeros((2, 2)), np.zeros((2, 2)), n_buckets=1)

    def test_a_direction_without_its_weight_is_refused(self):
        with pytest.raises(ShapeMismatchError, match="one position"):
            votes_by_bucket(np.zeros((2, 2)), np.zeros((3, 3)), n_buckets=9)

    def test_the_whole_of_every_vote_is_cast_and_no_more(self):
        """Whichever rule, a pixel's magnitude is what it has to spend."""
        directions = np.linspace(-np.pi, np.pi, 37).reshape(1, 37)
        magnitudes = np.full((1, 37), 3.0)

        for sharing in VoteSharing:
            votes = votes_by_bucket(
                directions, magnitudes, 9, AngleRange.UNSIGNED, sharing
            )

            assert votes.shape == (1, 37, 9)
            assert np.allclose(votes.sum(axis=-1), 3.0)


class TestSharingAVoteBetweenBuckets:
    def test_the_whole_vote_rule_jumps_at_a_bucket_boundary(self):
        """Two degrees of turn either side of the boundary between the first
        two buckets moves the entire cell from one bucket to the other."""
        whole = HistogramOfOrientedGradients(vote_sharing=VoteSharing.WHOLE_TO_NEAREST)

        before = whole.cell_histograms(ramp(19.0)).histogram_at(1, 1)
        after = whole.cell_histograms(ramp(21.0)).histogram_at(1, 1)

        assert before[0] / before.sum() == pytest.approx(1.0)
        assert after[1] / after.sum() == pytest.approx(1.0)

    def test_and_sharing_it_slides_instead(self):
        """The same two degrees move a tenth of the weight, which is the
        share of a twenty-degree bucket that two degrees ought to be."""
        split = HistogramOfOrientedGradients()

        before = split.cell_histograms(ramp(19.0)).histogram_at(1, 1)
        after = split.cell_histograms(ramp(21.0)).histogram_at(1, 1)

        assert before[0] / before.sum() == pytest.approx(0.55)
        assert before[1] / before.sum() == pytest.approx(0.45)
        assert after[0] / after.sum() == pytest.approx(0.45)
        assert after[1] / after.sum() == pytest.approx(0.55)

    def test_the_jump_is_ten_times_the_slide_and_the_ratio_is_exact(self):
        """Measured: 724.0773 against 72.4077, a ratio of 10 to within 4e-14."""
        moved = {}
        for sharing in VoteSharing:
            configured = HistogramOfOrientedGradients(vote_sharing=sharing)
            before = configured.cell_histograms(ramp(19.0)).histogram_at(1, 1)
            after = configured.cell_histograms(ramp(21.0)).histogram_at(1, 1)
            moved[sharing] = float(np.linalg.norm(after - before))

        assert moved[VoteSharing.WHOLE_TO_NEAREST] == pytest.approx(724.0773, abs=1e-3)
        assert moved[VoteSharing.SPLIT_BETWEEN_NEIGHBOURS] == pytest.approx(
            72.4077, abs=1e-3
        )
        assert moved[VoteSharing.WHOLE_TO_NEAREST] / moved[
            VoteSharing.SPLIT_BETWEEN_NEIGHBOURS
        ] == pytest.approx(10.0, abs=1e-9)

    def test_a_direction_on_a_bucket_centre_is_untouched_by_the_sharing(self):
        """Ninety degrees is the centre of bucket 4, so both rules agree."""
        centred = horizontal_edge()

        whole = HistogramOfOrientedGradients(
            vote_sharing=VoteSharing.WHOLE_TO_NEAREST
        ).cell_histograms(centred)
        split = HistogramOfOrientedGradients().cell_histograms(centred)

        assert np.allclose(whole.counts, split.counts)

    def test_but_a_direction_between_two_centres_is_split_in_half(self):
        """Zero degrees is as far from the first bucket's centre as from the
        last one's, and under an unsigned range those two are neighbours. So
        a single clean vertical edge reads as two half-full buckets, which is
        correct and is the price of not jumping."""
        split = HistogramOfOrientedGradients().cell_histograms(vertical_edge())

        shares = split.counts.sum(axis=(0, 1)) / split.total_weight

        assert shares[0] == pytest.approx(0.5)
        assert shares[8] == pytest.approx(0.5)
        assert shares[1:8] == pytest.approx(np.zeros(7))

    def test_the_sharing_wraps_around_the_last_bucket(self):
        """The direct claim behind the test above, on one pixel's vote."""
        votes = votes_by_bucket(
            np.zeros((1, 1)), np.ones((1, 1)), 9, AngleRange.UNSIGNED
        )

        assert votes[0, 0, 0] == pytest.approx(0.5)
        assert votes[0, 0, 8] == pytest.approx(0.5)


class TestSignedAgainstUnsigned:
    def test_unsigned_folds_an_edge_and_its_reverse_into_one_bucket(self):
        """A bright bar's two sides are the same edge seen from either side."""
        histograms = HistogramOfOrientedGradients(
            angle_range=AngleRange.UNSIGNED,
            n_buckets=9,
            vote_sharing=VoteSharing.WHOLE_TO_NEAREST,
        ).cell_histograms(bright_bar())

        shares = histograms.counts.sum(axis=(0, 1)) / histograms.total_weight

        assert int(np.count_nonzero(shares > 1e-12)) == 1
        assert shares.max() == pytest.approx(1.0)
        assert np.rad2deg(histograms.bucket_centre(0)) == pytest.approx(10.0)

    def test_and_signed_keeps_them_half_a_circle_apart(self):
        histograms = HistogramOfOrientedGradients(
            angle_range=AngleRange.SIGNED,
            n_buckets=18,
            vote_sharing=VoteSharing.WHOLE_TO_NEAREST,
        ).cell_histograms(bright_bar())

        shares = histograms.counts.sum(axis=(0, 1)) / histograms.total_weight
        occupied = list(np.flatnonzero(shares > 1e-12))

        assert occupied == [0, 9]
        assert shares[0] == pytest.approx(0.5)
        assert shares[9] == pytest.approx(0.5)
        assert np.rad2deg(histograms.bucket_centre(0)) == pytest.approx(10.0)
        assert np.rad2deg(histograms.bucket_centre(9)) == pytest.approx(190.0)

    def test_the_choice_decides_whether_a_bar_and_its_negative_differ(self):
        """The whole argument for the pedestrian detector's unsigned range,
        as one number each. Inverting the brightness reverses every gradient:
        unsigned cannot see it at all, and signed sees nothing else."""
        bar = bright_bar()
        inverted = Picture(1.0 - np.asarray(bar))

        unsigned = HistogramOfOrientedGradients(
            angle_range=AngleRange.UNSIGNED, n_buckets=9
        )
        signed = HistogramOfOrientedGradients(
            angle_range=AngleRange.SIGNED, n_buckets=18
        )

        assert unsigned.describe(bar).distance_to(
            unsigned.describe(inverted)
        ) == pytest.approx(0.0, abs=1e-15)
        assert signed.describe(bar).distance_to(
            signed.describe(inverted)
        ) == pytest.approx(np.sqrt(2.0), abs=1e-12)

    def test_the_two_ranges_divide_different_amounts_of_angle(self):
        assert AngleRange.UNSIGNED.span == pytest.approx(np.pi)
        assert AngleRange.SIGNED.span == pytest.approx(2.0 * np.pi)


class TestBlockNormalisation:
    def test_each_rule_puts_a_block_on_the_scale_it_names(self):
        values = {
            rule: np.asarray(
                HistogramOfOrientedGradients(block_normalisation=rule).describe(
                    shapes()
                )
            ).reshape(-1, 36)
            for rule in BlockNormalisation
        }

        assert np.allclose(values[BlockNormalisation.L1].sum(axis=1), 1.0)
        assert np.allclose(np.linalg.norm(values[BlockNormalisation.L2], axis=1), 1.0)
        assert np.allclose(
            np.linalg.norm(values[BlockNormalisation.L1_SQUARE_ROOT], axis=1), 1.0
        )
        assert np.allclose(
            np.linalg.norm(values[BlockNormalisation.L2_CLIPPED], axis=1), 1.0
        )
        assert not np.allclose(
            np.linalg.norm(values[BlockNormalisation.NONE], axis=1), 1.0
        )

    def test_a_block_with_no_weight_divides_by_nothing_and_answers_zeros(self):
        for rule in BlockNormalisation:
            assert normalised_block(np.zeros(8), rule) == pytest.approx(np.zeros(8))

    def test_a_flat_patch_describes_as_zeros_rather_than_raising(self):
        described = HistogramOfOrientedGradients().describe(FLAT)

        assert described.length == pytest.approx(0.0)

    def test_clipping_cuts_the_loudest_direction_back_but_not_to_the_limit(self):
        """It is a limit on the way in, and the block is made a unit vector
        again afterwards, so the clipped entry comes back above it. Measured
        on a patch holding one bright pixel: 0.4851 becomes 0.3052."""
        one_bright_pixel = np.zeros((16, 16))
        one_bright_pixel[7, 7] = 1.0
        picture = Picture(one_bright_pixel)

        plain = np.asarray(
            HistogramOfOrientedGradients(
                block_normalisation=BlockNormalisation.L2
            ).describe(picture)
        )
        clipped = np.asarray(
            HistogramOfOrientedGradients(
                block_normalisation=BlockNormalisation.L2_CLIPPED
            ).describe(picture)
        )

        assert plain.max() == pytest.approx(0.4851, abs=1e-4)
        assert clipped.max() == pytest.approx(0.3052, abs=1e-4)
        assert clipped.max() < plain.max()
        assert clipped.max() > CLIP_LIMIT

    def test_leaving_the_normalisation_out_changes_nothing_but_the_division(self):
        """Which is what makes the lighting claim below a comparison rather
        than two different measurements."""
        normalised = HistogramOfOrientedGradients().describe(shapes())
        raw = HistogramOfOrientedGradients(
            block_normalisation=BlockNormalisation.NONE
        ).describe(shapes())

        assert normalised.n_values == raw.n_values


class TestLighting:
    def test_a_change_of_lighting_does_not_move_the_description(self):
        """Multiply every pixel by 3.7 and add 12. Measured at 9.2e-16."""
        model = HistogramOfOrientedGradients()
        picture = shapes()
        relit = Picture(np.asarray(picture) * 3.7 + 12.0)

        moved = model.describe(picture).distance_to(model.describe(relit))

        assert moved == pytest.approx(0.0, abs=1e-14)

    @pytest.mark.parametrize(
        "rule",
        [
            BlockNormalisation.L1,
            BlockNormalisation.L1_SQUARE_ROOT,
            BlockNormalisation.L2,
            BlockNormalisation.L2_CLIPPED,
        ],
    )
    def test_and_every_normalisation_buys_it(self, rule):
        model = HistogramOfOrientedGradients(block_normalisation=rule)
        picture = shapes()
        relit = Picture(np.asarray(picture) * 3.7 + 12.0)

        assert model.describe(picture).distance_to(
            model.describe(relit)
        ) == pytest.approx(0.0, abs=1e-14)

    def test_without_normalisation_it_moves_a_long_way(self):
        """Measured: 521.0, against a description whose own length is 193.0.
        The same comparison as above with one stage removed, which is the
        argument that the stage is what buys the invariance."""
        model = HistogramOfOrientedGradients(
            block_normalisation=BlockNormalisation.NONE
        )
        picture = shapes()
        relit = Picture(np.asarray(picture) * 3.7 + 12.0)

        here = model.describe(picture)
        moved = here.distance_to(model.describe(relit))

        assert here.length == pytest.approx(192.978, abs=1e-2)
        assert moved == pytest.approx(521.041, abs=1e-2)
        assert moved / here.length == pytest.approx(2.700, abs=1e-3)

    def test_the_two_halves_of_a_lighting_change_are_bought_at_different_stages(self):
        """Adding a constant is already free before any normalisation, since
        the gradient operator's weights sum to zero; multiplying is not, and
        is what the block normalisation is actually for. Measured
        unnormalised: 7.6e-14 for the shift and 521.0 for the scale."""
        model = HistogramOfOrientedGradients(
            block_normalisation=BlockNormalisation.NONE
        )
        picture = shapes()
        here = model.describe(picture)

        shifted = model.describe(Picture(np.asarray(picture) + 12.0))
        scaled = model.describe(Picture(np.asarray(picture) * 3.7))

        assert here.distance_to(shifted) == pytest.approx(0.0, abs=1e-12)
        assert here.distance_to(scaled) == pytest.approx(521.041, abs=1e-2)


class TestRotation:
    def test_a_quarter_turn_moves_the_description_a_long_way(self):
        """The method's real limit, measured beside the thing it is good at.
        A quarter turn moves the description by 1.4832 out of its own length
        of 2.0, where relighting the same patch moves it by 9.2e-16."""
        model = HistogramOfOrientedGradients()
        picture = shapes()
        turned = Picture(np.rot90(np.asarray(picture)))
        relit = Picture(np.asarray(picture) * 3.7 + 12.0)

        here = model.describe(picture)

        assert here.length == pytest.approx(2.0)
        assert here.distance_to(model.describe(turned)) == pytest.approx(
            1.4832, abs=1e-3
        )
        assert here.distance_to(model.describe(relit)) < 1e-14

    def test_and_no_choice_of_normalisation_rescues_it(self):
        """It is not an artefact of one rule. Every one of the five moves by
        something of the order of the description's own length."""
        picture = shapes()
        turned = Picture(np.rot90(np.asarray(picture)))

        for rule in BlockNormalisation:
            model = HistogramOfOrientedGradients(block_normalisation=rule)
            here = model.describe(picture)

            assert here.distance_to(model.describe(turned)) / here.length > 0.7

    def test_the_buckets_alone_cannot_be_blamed(self):
        """A quarter turn is not a whole number of unsigned buckets at nine of
        them -- ninety degrees is four and a half -- but it is at six, where a
        bucket is thirty degrees. The description still moves, because the
        cells have moved as well as the directions."""
        picture = shapes()
        turned = Picture(np.rot90(np.asarray(picture)))
        model = HistogramOfOrientedGradients(n_buckets=6)

        here = model.describe(picture)

        assert here.distance_to(model.describe(turned)) / here.length > 0.5


class TestTheSizeOfTheDescription:
    def test_the_pedestrian_detector_s_own_number(self):
        """64 wide by 128 tall at the defaults: 8 by 16 cells, so 7 by 15
        block positions, each holding 2 by 2 cells of 9 buckets.
        7 * 15 * 4 * 9 = 3780, which is Dalal and Triggs's figure."""
        model = HistogramOfOrientedGradients()

        assert model.n_cells_along(64) == 8
        assert model.n_cells_along(128) == 16
        assert model.n_blocks_along(8) == 7
        assert model.n_blocks_along(16) == 15
        assert model.n_values_for(height=128, width=64) == 3780

    @pytest.mark.parametrize(
        "height, width, cell_side, block_side, stride, n_buckets, expected",
        [
            (16, 16, 8, 2, 1, 9, 36),
            (24, 24, 8, 2, 1, 9, 144),
            (24, 24, 8, 2, 2, 9, 36),
            (24, 24, 8, 1, 1, 9, 81),
            (128, 64, 8, 2, 1, 18, 7560),
            (32, 32, 4, 2, 1, 9, 1764),
        ],
    )
    def test_the_arithmetic_holds_wherever_the_pieces_are_moved(
        self, height, width, cell_side, block_side, stride, n_buckets, expected
    ):
        model = HistogramOfOrientedGradients(
            cell_side=cell_side,
            cells_per_block_side=block_side,
            block_stride_in_cells=stride,
            n_buckets=n_buckets,
        )

        assert model.n_values_for(height, width) == expected

    def test_and_the_answer_really_is_that_long(self):
        """The arithmetic and the array, which are two routes to one number."""
        model = HistogramOfOrientedGradients()

        described = model.describe(shapes())

        assert len(described) == model.n_values_for(24, 24) == 144

    def test_overlapping_blocks_describe_a_cell_more_than_once(self):
        """Nine cells, described in sixteen cells' worth of numbers, because
        the middle cell belongs to all four blocks and the corners to one
        each. That repetition is the third stage, counted."""
        model = HistogramOfOrientedGradients()
        histograms = model.cell_histograms(shapes())

        n_cells = histograms.n_cell_rows * histograms.n_cell_columns
        cells_in_the_answer = model.n_values_for(24, 24) / model.n_buckets

        assert n_cells == 9
        assert cells_in_the_answer == 16

    def test_tiling_the_blocks_instead_describes_each_cell_exactly_once(self):
        model = HistogramOfOrientedGradients(block_stride_in_cells=2)

        assert model.n_values_for(16, 16) / model.n_buckets == 4


class TestRefusals:
    def test_a_patch_that_does_not_divide_into_cells_is_refused(self):
        with pytest.raises(ShapeMismatchError, match="does not divide"):
            HistogramOfOrientedGradients().describe(Picture(np.zeros((16, 20))))

    def test_a_patch_too_small_to_hold_one_block_is_refused(self):
        with pytest.raises(ShapeMismatchError, match="smaller than the"):
            HistogramOfOrientedGradients().describe(Picture(np.zeros((8, 8))))

    def test_a_stride_wider_than_the_block_would_skip_cells_and_is_refused(self):
        with pytest.raises(ValidationError, match="no block covers"):
            HistogramOfOrientedGradients(
                cells_per_block_side=2, block_stride_in_cells=3
            )

    def test_a_stride_equal_to_the_block_is_allowed_because_it_tiles(self):
        model = HistogramOfOrientedGradients(
            cells_per_block_side=2, block_stride_in_cells=2
        )

        assert model.block_stride_in_cells == 2

    def test_the_edge_rule_that_answers_a_smaller_field_is_refused(self):
        """A cell is a fixed block of the picture's own pixels, so a gradient
        field two pixels smaller than the picture would not line up with it."""
        with pytest.raises(ValidationError, match="smaller than the picture"):
            HistogramOfOrientedGradients(edge_rule=EdgeRule.KEEP_VALID)

    @pytest.mark.parametrize(
        "settings",
        [
            {"cell_side": 0},
            {"cells_per_block_side": 0},
            {"block_stride_in_cells": 0},
            {"n_buckets": 1},
        ],
    )
    def test_a_setting_below_its_own_floor_is_refused(
        self, settings: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            HistogramOfOrientedGradients(**settings)

    def test_an_unknown_setting_is_refused_rather_than_dropped(self):
        """``extra="forbid"``, for the reason the project notes record: a
        misspelling pydantic drops leaves the field at its default, and a
        default is by construction a plausible number."""
        misspelled: dict[str, Any] = {"cellside": 8}

        with pytest.raises(ValidationError):
            HistogramOfOrientedGradients(**misspelled)


class TestCellHistogramsTheObject:
    def test_it_carries_its_own_arrangement(self):
        histograms = HistogramOfOrientedGradients().cell_histograms(shapes())

        assert histograms.n_cell_rows == 3
        assert histograms.n_cell_columns == 3
        assert histograms.n_buckets == 9
        assert histograms.angle_range is AngleRange.UNSIGNED
        assert histograms.bucket_width == pytest.approx(np.pi / 9)

    def test_the_buffer_is_frozen_and_is_not_the_caller_s(self):
        counts = np.ones((2, 2, 9))
        histograms = CellHistograms(counts, AngleRange.UNSIGNED)
        counts[0, 0, 0] = 99.0

        assert histograms.counts[0, 0, 0] == 1.0
        with pytest.raises(ValueError):
            histograms.counts[0, 0, 0] = 2.0

    def test_the_array_protocol_hands_out_a_copy(self):
        histograms = CellHistograms(np.ones((2, 2, 9)), AngleRange.UNSIGNED)

        assert not np.shares_memory(np.array(histograms), histograms.counts)

    def test_a_negative_weight_is_refused_because_no_magnitude_is_negative(self):
        counts = np.ones((2, 2, 9))
        counts[0, 0, 0] = -1.0

        with pytest.raises(InvalidValuesError, match="never negative"):
            CellHistograms(counts, AngleRange.UNSIGNED)

    @pytest.mark.parametrize("counts", [np.ones((2, 2)), np.ones((2, 2, 9, 1))])
    def test_counts_of_the_wrong_arrangement_are_refused(self, counts):
        with pytest.raises(InvalidValuesError, match="rows, columns, buckets"):
            CellHistograms(counts, AngleRange.UNSIGNED)

    def test_a_cell_outside_the_grid_is_refused(self):
        histograms = CellHistograms(np.ones((2, 2, 9)), AngleRange.UNSIGNED)

        with pytest.raises(InvalidValuesError, match="asked for row"):
            histograms.histogram_at(2, 0)

    def test_a_bucket_outside_the_range_is_refused(self):
        histograms = CellHistograms(np.ones((2, 2, 9)), AngleRange.UNSIGNED)

        with pytest.raises(InvalidValuesError, match="asked for 9"):
            histograms.bucket_centre(9)

    def test_equality_is_by_value_and_by_range(self):
        counts = np.ones((2, 2, 9))
        unsigned = CellHistograms(counts, AngleRange.UNSIGNED)

        assert unsigned == CellHistograms(counts, AngleRange.UNSIGNED)
        assert unsigned != CellHistograms(counts, AngleRange.SIGNED)
        assert unsigned != CellHistograms(counts * 2, AngleRange.UNSIGNED)
        assert unsigned.__eq__("histograms") is NotImplemented


class TestTheDescriptionTheObject:
    def test_it_is_a_vector_and_says_how_long(self):
        described = HistogramOfOrientedGradients().describe(shapes())

        assert described.n_values == 144
        assert len(described) == 144
        assert np.asarray(described).shape == (144,)

    def test_the_buffer_is_frozen_and_is_not_the_caller_s(self):
        values = np.arange(4.0)
        described = OrientedGradientDescription(values)
        values[0] = 99.0

        assert described.values[0] == 0.0
        with pytest.raises(ValueError):
            described.values[0] = 1.0

    def test_the_array_protocol_hands_out_a_copy(self):
        described = OrientedGradientDescription(np.arange(4.0))

        assert not np.shares_memory(np.array(described), described.values)

    def test_a_description_is_the_same_distance_from_itself_as_zero(self):
        described = HistogramOfOrientedGradients().describe(shapes())

        assert described.distance_to(described) == pytest.approx(0.0)

    def test_two_descriptions_of_different_lengths_cannot_be_compared(self):
        """Different settings put different things at position 40, so a
        distance between them would be a number with no meaning."""
        nine = HistogramOfOrientedGradients().describe(shapes())
        eighteen = HistogramOfOrientedGradients(
            angle_range=AngleRange.SIGNED, n_buckets=18
        ).describe(shapes())

        with pytest.raises(ShapeMismatchError, match="different settings"):
            nine.distance_to(eighteen)

    @pytest.mark.parametrize(
        "values", [np.zeros((2, 2)), np.array([]), np.array([1.0, np.nan])]
    )
    def test_something_that_is_not_a_finite_vector_is_refused(self, values):
        with pytest.raises(InvalidValuesError):
            OrientedGradientDescription(values)

    def test_equality_is_by_value_and_defers_to_anything_else(self):
        first = OrientedGradientDescription([1.0, 2.0])

        assert first == OrientedGradientDescription([1.0, 2.0])
        assert first != OrientedGradientDescription([1.0, 3.0])
        assert first.__eq__("a description") is NotImplemented

    def test_the_same_picture_described_twice_gives_the_same_answer(self):
        """Nothing here is random, so this is equality rather than closeness."""
        model = HistogramOfOrientedGradients()

        assert model.describe(shapes()) == model.describe(shapes())
