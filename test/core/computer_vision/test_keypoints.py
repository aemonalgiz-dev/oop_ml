"""Spec for corner detection, patch descriptors and descriptor matching.

Every fixture is a hand-drawn shape on a plain background, for the reason the
foundation spec gives: a bright square in a known place has corners in four
known positions, edges of a known length between them, and flat ground
everywhere else, so every number below is checkable rather than merely
reproducible. Two of them come out exactly round -- the smaller eigenvalue at a
corner of the unit square is exactly ``4.0``, and along its edge exactly
``0.0`` -- and both are asserted as equalities rather than approximations,
because they are consequences of the arithmetic and not accidents of it.

The claims here are the ones the family exists to make: that a corner can be
located and an edge cannot, that the positions belong to the scene rather than
to the frame, and that a match is worth believing only when its runner-up is
far behind.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.computer_vision.keypoints import (
    HARRIS_SENSITIVITY_CEILING,
    CornerResponse,
    DescriptorMatch,
    Descriptors,
    HarrisMeasure,
    Keypoint,
    Keypoints,
    PatchDescriptor,
    SmallestEigenvalueMeasure,
    StructureTensor,
)
from oop_ml.core.computer_vision.picture import Picture
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    TooFewValuesError,
)

SQUARE_SIZE = 21
SQUARE_TOP = 6
SQUARE_LEFT = 6
SQUARE_SIDE = 9

# The four corners of the square below, in reading order.
SQUARE_CORNERS = [(6, 6), (6, 14), (14, 6), (14, 14)]

# The middle of its left edge and of its top edge, four pixels clear of any
# corner, which is more than the five-pixel reach of a Sobel sweep followed by
# a three-wide averaging window.
LEFT_EDGE_MIDDLE = (10, 6)
TOP_EDGE_MIDDLE = (6, 10)

# Flat ground outside the square, and flat ground inside it.
OUTSIDE = (2, 2)
INSIDE = (10, 10)


def square_picture(
    size: int = SQUARE_SIZE,
    top: int = SQUARE_TOP,
    left: int = SQUARE_LEFT,
    side: int = SQUARE_SIDE,
) -> Picture:
    """A bright square of ``side`` pixels on dark ground."""
    values = np.zeros((size, size))
    values[top : top + side, left : left + side] = 1.0
    return Picture(values)


def bent_bar_picture(size: int = 21) -> Picture:
    """A shape with no symmetry a quarter turn preserves.

    A square would pass the rotation spec for the wrong reason: it looks the
    same turned, so a descriptor that merely found *a* corner would seem to
    have found *the* corner. This one is an L, whose six corners are all
    distinguishable from each other.
    """
    values = np.zeros((size, size))
    values[5:16, 5:11] = 1.0
    values[5:9, 5:16] = 1.0
    return Picture(values)


def twin_squares_picture(size: int = 31) -> Picture:
    """Two identical squares, which is the repeated structure that defeats this."""
    values = np.zeros((size, size))
    values[5:12, 3:10] = 1.0
    values[5:12, 18:25] = 1.0
    return Picture(values)


def faint_texture(shape: tuple[int, int], amplitude: float = 0.05) -> np.ndarray:
    """A deterministic ripple, small beside the square's own contrast.

    Its only job is to break an exact tie between two identical patches, so
    that a coincidental match reports a ratio it computed rather than the
    convention the zero-over-zero case falls back on.
    """
    rows = np.arange(shape[0])[:, None]
    columns = np.arange(shape[1])[None, :]
    return amplitude * np.sin(rows * 1.7) * np.cos(columns * 2.3)


def turned_position(row: int, column: int, size: int) -> tuple[int, int]:
    """Where ``(row, column)`` lands after a counter-clockwise quarter turn."""
    return (size - 1 - column, row)


class TestTheStructureTensor:
    def test_it_describes_the_same_picture_the_gradients_did(self):
        tensor = StructureTensor.of(square_picture())

        assert tensor.shape == (SQUARE_SIZE, SQUARE_SIZE)

    def test_flat_ground_has_no_structure_at_all(self):
        tensor = StructureTensor.of(square_picture())

        assert tensor.trace.values[OUTSIDE] == 0.0
        assert tensor.determinant.values[OUTSIDE] == 0.0

    def test_an_edge_carries_energy_but_only_in_one_direction(self):
        """The rank-one case, stated in the two numbers that show it.

        The trace is the total gradient energy and it is large. The determinant
        is the product of the two eigenvalues and it is *exactly* zero, because
        every gradient in a window straddling a straight edge points the same
        way. That is the aperture problem written as arithmetic.
        """
        tensor = StructureTensor.of(square_picture())

        assert tensor.trace.values[LEFT_EDGE_MIDDLE] == pytest.approx(10.666667)
        assert tensor.determinant.values[LEFT_EDGE_MIDDLE] == 0.0

    def test_the_energy_cannot_tell_a_corner_from_an_edge_and_the_shape_can(self):
        """The measured reason the corner measures read the determinant.

        The trace at a corner is 8% above the trace at an edge, which is
        nothing. The determinant at a corner is 30.22 against exactly 0.0.
        """
        tensor = StructureTensor.of(square_picture())

        corner_trace = tensor.trace.values[SQUARE_CORNERS[0]]
        edge_trace = tensor.trace.values[LEFT_EDGE_MIDDLE]

        assert corner_trace == pytest.approx(11.555556)
        assert edge_trace == pytest.approx(10.666667)
        assert corner_trace / edge_trace == pytest.approx(1.083333)
        assert tensor.determinant.values[SQUARE_CORNERS[0]] == pytest.approx(30.222222)

    def test_without_a_window_every_matrix_is_rank_one_and_no_corner_exists(self):
        """Why the averaging is the whole of it, rather than a smoothing step.

        A one-pixel window makes each matrix the outer product of one gradient
        with itself, whose determinant is zero everywhere by construction, so
        the strongest corner in the picture scores nothing.
        """
        unaveraged = StructureTensor.of(square_picture(), window_side=1)

        assert unaveraged.response().scores.brightest == pytest.approx(0.0, abs=1e-12)
        assert StructureTensor.of(square_picture()).response().scores.brightest > 3.9

    def test_the_three_parts_must_describe_one_picture(self):
        small = Picture(np.zeros((3, 3)))
        large = Picture(np.zeros((5, 5)))

        with pytest.raises(ShapeMismatchError, match="the same picture"):
            StructureTensor(small, large, small)

    @pytest.mark.parametrize("window_side", [0, -3])
    def test_a_window_with_no_pixels_is_refused(self, window_side):
        with pytest.raises(InvalidValuesError, match="at least one pixel"):
            StructureTensor.of(square_picture(), window_side=window_side)

    def test_an_even_window_is_refused_because_it_has_no_centre(self):
        with pytest.raises(InvalidValuesError, match="centre pixel"):
            StructureTensor.of(square_picture(), window_side=4)

    def test_equality_is_by_value_and_defers_to_anything_else(self):
        one = StructureTensor.of(square_picture())
        again = StructureTensor.of(square_picture())

        assert one == again
        assert one.__eq__("a tensor") is NotImplemented


class TestTheCornerMeasures:
    def test_a_corner_scores_far_above_an_edge_which_scores_no_better_than_flat(self):
        """The headline claim, and it is not the one usually stated.

        Under the smaller eigenvalue a corner reads exactly 4.0 while an edge
        and flat ground both read exactly 0.0. An edge is not partway between;
        it is indistinguishable from empty sky, because it is exactly as
        unlocatable along its own length.
        """
        response = CornerResponse.of(square_picture())

        assert response.at(*SQUARE_CORNERS[0]) == 4.0
        assert response.at(*LEFT_EDGE_MIDDLE) == 0.0
        assert response.at(*TOP_EDGE_MIDDLE) == 0.0
        assert response.at(*OUTSIDE) == 0.0
        assert response.at(*INSIDE) == 0.0

    def test_harris_puts_an_edge_below_flat_ground_rather_than_above_it(self):
        """The same fact in the other measure's words, and the numbers to quote.

        Harris' subtracted trace term has nothing to bite on where the trace is
        zero, so flat ground reads 0.0, and a great deal to bite on along an
        edge, which is why an edge reads negative. Three orders of magnitude,
        in the order corner, flat, edge.
        """
        response = CornerResponse.of(square_picture(), HarrisMeasure())

        assert response.at(*SQUARE_CORNERS[0]) == pytest.approx(24.880988)
        assert response.at(*OUTSIDE) == 0.0
        assert response.at(*LEFT_EDGE_MIDDLE) == pytest.approx(-4.551111)

    def test_the_two_measures_agree_about_which_pixels_are_corners(self):
        """They differ in their numbers and not in their answer."""
        picture = square_picture()

        by_eigenvalue = CornerResponse.of(picture).peaks(minimum_strength=0.5)
        by_harris = CornerResponse.of(picture, HarrisMeasure()).peaks(
            minimum_strength=0.5
        )

        assert [(point.row, point.column) for point in by_eigenvalue] == SQUARE_CORNERS
        assert [(point.row, point.column) for point in by_harris] == SQUARE_CORNERS

    def test_the_smaller_eigenvalue_is_never_negative(self):
        response = CornerResponse.of(square_picture())

        assert response.scores.darkest >= 0.0

    def test_a_sensitivity_at_the_ceiling_finds_nothing_and_is_refused(self):
        """Why the ceiling is a guard rather than a matter of taste.

        The determinant of a real symmetric matrix is at most a quarter of its
        squared trace, so at a sensitivity of a quarter the response cannot be
        positive anywhere. Just below the ceiling it is already useless: at
        0.24 the square's own corners read -1.825 and nothing is found.
        """
        just_below = CornerResponse.of(
            square_picture(), HarrisMeasure(sensitivity=0.24)
        )

        assert just_below.at(*SQUARE_CORNERS[0]) == pytest.approx(-1.825185)
        assert len(just_below.peaks()) == 0
        with pytest.raises(ValidationError):
            HarrisMeasure(sensitivity=HARRIS_SENSITIVITY_CEILING)

    @pytest.mark.parametrize("sensitivity", [0.0, -0.1])
    def test_a_sensitivity_at_or_below_zero_is_refused(self, sensitivity):
        with pytest.raises(ValidationError):
            HarrisMeasure(sensitivity=sensitivity)

    def test_an_unknown_keyword_is_refused_rather_than_dropped(self):
        with pytest.raises(ValidationError):
            HarrisMeasure(sensitivty=0.05)  # pyright: ignore[reportCallIssue]

    def test_the_measure_travels_with_the_scores(self):
        """A reading of -4.55 is an edge under one measure and impossible under
        the other, so the numbers are meaningless without it."""
        response = CornerResponse.of(square_picture(), HarrisMeasure())

        assert isinstance(response.measure, HarrisMeasure)
        assert isinstance(
            CornerResponse.of(square_picture()).measure, SmallestEigenvalueMeasure
        )

    def test_reading_a_position_outside_the_picture_is_refused(self):
        response = CornerResponse.of(square_picture())

        with pytest.raises(InvalidValuesError, match="outside"):
            response.at(SQUARE_SIZE, 0)


class TestKeepingOnePeakPerCorner:
    def test_one_corner_answers_one_keypoint_rather_than_a_blob(self):
        """Measured: the raw response puts 32 pixels above the threshold, which
        is four corners reported eight times each. Every separation from two
        upward answers four."""
        response = CornerResponse.of(square_picture())

        assert len(response.peaks(minimum_strength=0.5, separation=1)) == 32
        assert len(response.peaks(minimum_strength=0.5, separation=2)) == 4
        assert len(response.peaks(minimum_strength=0.5, separation=7)) == 4

    def test_the_peaks_are_the_four_corners_strongest_first(self):
        peaks = CornerResponse.of(square_picture()).peaks(minimum_strength=0.5)

        assert [(point.row, point.column) for point in peaks] == SQUARE_CORNERS
        assert all(point.strength == 4.0 for point in peaks)

    def test_ties_are_broken_by_position_so_two_runs_agree(self):
        """All four corners of a square score identically, so without a stated
        rule the order would be whatever the sort happened to do."""
        picture = square_picture()

        first = CornerResponse.of(picture).peaks(minimum_strength=0.5)
        again = CornerResponse.of(picture).peaks(minimum_strength=0.5)

        assert first == again

    def test_a_limit_stops_early_and_keeps_the_strongest(self):
        peaks = CornerResponse.of(square_picture()).peaks(minimum_strength=0.5, limit=2)

        assert [(point.row, point.column) for point in peaks] == SQUARE_CORNERS[:2]

    def test_a_border_leaves_out_the_keypoints_a_patch_will_not_fit_around(self):
        picture = square_picture(size=17, top=2, left=2, side=9)

        without_border = CornerResponse.of(picture).peaks(minimum_strength=0.5)
        with_border = CornerResponse.of(picture).peaks(minimum_strength=0.5, border=3)

        assert len(without_border) == 4
        assert len(with_border) == 1
        assert (with_border[0].row, with_border[0].column) == (10, 10)

    def test_an_excluded_corner_does_not_promote_the_pixel_beside_it(self):
        """The border drops what is reported, not what suppresses.

        Three of this square's corners sit two pixels from the frame. Dropping
        the *candidates* first, which is the obvious implementation and was the
        first one written here, answers a shoulder pixel at (3, 3) scoring
        3.5556 in place of the corner at (2, 2) scoring 4.0 -- one keypoint per
        corner, at the wrong position, with nothing raised.
        """
        picture = square_picture(size=17, top=2, left=2, side=9)

        peaks = CornerResponse.of(picture).peaks(minimum_strength=0.5, border=3)

        assert [(point.row, point.column) for point in peaks] == [(10, 10)]
        assert CornerResponse.of(picture).at(3, 3) == pytest.approx(3.5556, abs=1e-4)

    def test_a_picture_with_no_corners_answers_nothing(self):
        """A horizon has one long edge and no keypoints, which is correct."""
        horizon = np.zeros((21, 21))
        horizon[10:, :] = 1.0

        peaks = CornerResponse.of(Picture(horizon)).peaks(minimum_strength=1e-9)

        assert len(peaks) == 0
        with pytest.raises(EmptyValuesError, match="no keypoints"):
            _ = peaks.strongest

    @pytest.mark.parametrize(
        "arguments, message",
        [
            ({"separation": 0}, "at least one pixel apart"),
            ({"limit": 0}, "no keypoints at all"),
            ({"border": -1}, "count of pixels"),
        ],
    )
    def test_an_impossible_request_is_refused(self, arguments, message):
        response = CornerResponse.of(square_picture())

        with pytest.raises(InvalidValuesError, match=message):
            response.peaks(**arguments)


class TestAKeypointAndItsCollection:
    def test_it_carries_where_it_is_and_how_strong_it_was(self):
        keypoint = Keypoint(6, 6, 4.0)

        assert (keypoint.row, keypoint.column, keypoint.strength) == (6, 6, 4.0)

    @pytest.mark.parametrize("row, column", [(-1, 0), (0, -1)])
    def test_a_position_outside_a_picture_is_refused(self, row, column):
        with pytest.raises(InvalidValuesError, match="inside a picture"):
            Keypoint(row, column, 1.0)

    def test_a_missing_strength_is_refused(self):
        with pytest.raises(InvalidValuesError, match="finite"):
            Keypoint(0, 0, np.nan)

    def test_separation_is_measured_along_the_wider_axis(self):
        """The response around a corner is a square blob, so the suppression
        radius is a square rather than a circle."""
        keypoint = Keypoint(10, 10, 1.0)

        assert keypoint.is_further_than(3, Keypoint(13, 10, 1.0))
        assert not keypoint.is_further_than(3, Keypoint(12, 12, 1.0))

    def test_the_collection_is_iterated_rather_than_unpacked(self):
        peaks = CornerResponse.of(square_picture()).peaks(minimum_strength=0.5)

        assert len(peaks) == 4
        assert len(list(peaks)) == 4
        assert peaks[0] == Keypoint(6, 6, 4.0)
        assert not hasattr(peaks, "keypoints")

    def test_the_strongest_is_the_highest_scoring_one(self):
        peaks = Keypoints([Keypoint(0, 0, 1.0), Keypoint(5, 5, 9.0)])

        assert peaks.strongest == Keypoint(5, 5, 9.0)

    def test_equality_is_by_value_and_defers_to_anything_else(self):
        assert Keypoints([Keypoint(1, 1, 2.0)]) == Keypoints([Keypoint(1, 1, 2.0)])
        assert Keypoints([]).__eq__("keypoints") is NotImplemented
        assert Keypoint(1, 1, 2.0).__eq__((1, 1, 2.0)) is NotImplemented


class TestTheApertureProblem:
    """The claim the family is built on, measured rather than asserted.

    A patch cut from the middle of an edge is *identical* to a patch cut from
    several other positions along that edge, so nothing about it says where
    along the edge it came from. The corner's patch matches one position, its
    own.
    """

    def test_an_edge_patch_matches_many_positions_along_that_edge(self):
        picture = square_picture()
        reference = PatchDescriptor.of(picture, Keypoint(*LEFT_EDGE_MIDDLE, 0.0), 5)

        along_the_edge = [
            reference.distance_to(PatchDescriptor.of(picture, Keypoint(row, 6, 0.0), 5))
            for row in range(3, 18)
        ]

        assert sum(distance == 0.0 for distance in along_the_edge) == 5

    def test_a_corner_patch_matches_exactly_one_position(self):
        picture = square_picture()
        reference = PatchDescriptor.of(picture, Keypoint(*SQUARE_CORNERS[0], 0.0), 5)

        along_the_edge = [
            reference.distance_to(PatchDescriptor.of(picture, Keypoint(row, 6, 0.0), 5))
            for row in range(3, 18)
        ]

        assert sum(distance == 0.0 for distance in along_the_edge) == 1

    def test_and_the_corner_measure_agrees_with_what_the_patches_showed(self):
        """The two halves of the argument have to meet: the positions the
        descriptor can distinguish are exactly the positions the response
        scores above zero."""
        response = CornerResponse.of(square_picture())

        assert response.at(*SQUARE_CORNERS[0]) > 0.0
        assert all(response.at(row, 6) == 0.0 for row in range(8, 13))


class TestTheDescriptor:
    def test_it_is_the_patch_around_the_keypoint_as_one_run_of_numbers(self):
        descriptor = PatchDescriptor.of(square_picture(), Keypoint(6, 6, 4.0), 5)

        assert descriptor.n_values == 25
        assert descriptor.keypoint == Keypoint(6, 6, 4.0)

    def test_turning_the_lamp_up_does_not_change_the_description(self):
        """Both invariances, and both are rounding rather than exact.

        Centring removes an added constant and dividing by the length removes a
        multiplied one, so a change of exposure moves the descriptor by 1.2e-15
        and a change of contrast by 2.0e-16.
        """
        picture = square_picture()
        keypoint = Keypoint(*SQUARE_CORNERS[0], 0.0)
        reference = PatchDescriptor.of(picture, keypoint, 5)

        brighter = PatchDescriptor.of(Picture(picture.values + 10.0), keypoint, 5)
        harder = PatchDescriptor.of(Picture(picture.values * 3.0), keypoint, 5)

        assert reference.distance_to(brighter) == pytest.approx(0.0, abs=1e-14)
        assert reference.distance_to(harder) == pytest.approx(0.0, abs=1e-14)

    def test_the_scale_runs_to_two_and_a_negative_reaches_it(self):
        picture = square_picture()
        keypoint = Keypoint(*SQUARE_CORNERS[0], 0.0)

        negated = PatchDescriptor.of(Picture(-picture.values), keypoint, 5)

        assert PatchDescriptor.of(picture, keypoint, 5).distance_to(
            negated
        ) == pytest.approx(2.0)

    def test_a_flat_patch_has_nothing_to_normalise_and_says_so(self):
        """It comes back as zeros rather than dividing by nothing, and every
        such descriptor is zero from every other, which is exactly why
        keypoints are hunted at corners."""
        picture = square_picture()

        outside = PatchDescriptor.of(picture, Keypoint(*OUTSIDE, 0.0), 5)
        inside = PatchDescriptor.of(picture, Keypoint(*INSIDE, 0.0), 5)

        assert outside.is_flat
        assert inside.is_flat
        assert outside.distance_to(inside) == 0.0

    def test_two_different_corners_of_one_square_are_far_apart(self):
        picture = square_picture()

        top_left = PatchDescriptor.of(picture, Keypoint(6, 6, 0.0), 5)
        top_right = PatchDescriptor.of(picture, Keypoint(6, 14, 0.0), 5)

        assert top_left.distance_to(top_right) == pytest.approx(1.443376)

    def test_a_patch_running_off_the_picture_is_refused_rather_than_padded(self):
        """Padding would describe pixels the scene does not contain, and two
        pictures' paddings would then match each other."""
        with pytest.raises(ShapeMismatchError, match="runs off"):
            PatchDescriptor.of(square_picture(), Keypoint(0, 0, 1.0), 5)

    @pytest.mark.parametrize("side", [0, 4, -1])
    def test_a_patch_without_a_centre_pixel_is_refused(self, side):
        with pytest.raises(InvalidValuesError, match="odd"):
            PatchDescriptor.of(square_picture(), Keypoint(10, 10, 1.0), side)

    def test_the_buffer_is_frozen_and_the_array_protocol_copies(self):
        descriptor = PatchDescriptor.of(square_picture(), Keypoint(6, 6, 4.0), 5)

        with pytest.raises(ValueError):
            descriptor.values[0] = 1.0
        assert not np.shares_memory(np.array(descriptor), descriptor.values)

    def test_descriptors_of_different_sizes_cannot_be_compared(self):
        picture = square_picture()
        small = PatchDescriptor.of(picture, Keypoint(10, 10, 0.0), 3)
        large = PatchDescriptor.of(picture, Keypoint(10, 10, 0.0), 5)

        with pytest.raises(ShapeMismatchError, match="one size"):
            small.distance_to(large)

    def test_a_set_of_mixed_sizes_is_refused_at_construction(self):
        picture = square_picture()

        with pytest.raises(ShapeMismatchError, match="one size"):
            Descriptors(
                [
                    PatchDescriptor.of(picture, Keypoint(10, 10, 0.0), 3),
                    PatchDescriptor.of(picture, Keypoint(10, 10, 0.0), 5),
                ]
            )

    def test_the_collection_is_iterated_and_remembers_its_positions(self):
        picture = square_picture()
        peaks = CornerResponse.of(picture).peaks(minimum_strength=0.5)

        descriptors = Descriptors.of(picture, peaks, 5)

        assert len(descriptors) == 4
        assert descriptors.keypoints == peaks
        assert not hasattr(descriptors, "descriptors")


class TestKeypointsBelongToTheSceneAndNotTheFrame:
    def test_shifting_the_picture_shifts_every_keypoint_by_exactly_that_much(self):
        before = CornerResponse.of(square_picture()).peaks(minimum_strength=0.5)
        after = CornerResponse.of(
            square_picture(top=SQUARE_TOP + 2, left=SQUARE_LEFT + 3)
        ).peaks(minimum_strength=0.5)

        assert [point.shifted_by(2, 3) for point in before] == list(after)

    def test_and_leaves_their_strengths_bit_identical(self):
        before = CornerResponse.of(square_picture()).peaks(minimum_strength=0.5)
        after = CornerResponse.of(
            square_picture(top=SQUARE_TOP + 2, left=SQUARE_LEFT + 3)
        ).peaks(minimum_strength=0.5)

        assert [point.strength for point in before] == [
            point.strength for point in after
        ]

    def test_a_quarter_turn_moves_every_keypoint_to_exactly_where_it_should_be(self):
        """The one rotation that needs no interpolation, so any disagreement
        would be a real one rather than a resampling artefact."""
        picture = bent_bar_picture()
        turned = Picture(np.rot90(picture.values))

        found = CornerResponse.of(picture).peaks(minimum_strength=0.5, border=3)
        found_turned = CornerResponse.of(turned).peaks(minimum_strength=0.5, border=3)

        assert len(found) == 6
        assert sorted(
            turned_position(point.row, point.column, picture.height) for point in found
        ) == sorted((point.row, point.column) for point in found_turned)

    def test_and_the_strengths_survive_it_to_the_last_bit(self):
        """The structure tensor's determinant and trace are unchanged by a
        rotation of the axes, and a square averaging window is unchanged by a
        quarter turn, so there is nothing left to differ."""
        picture = bent_bar_picture()
        turned = Picture(np.rot90(picture.values))

        strengths = {
            turned_position(point.row, point.column, picture.height): point.strength
            for point in CornerResponse.of(picture).peaks(
                minimum_strength=0.5, border=3
            )
        }
        turned_strengths = {
            (point.row, point.column): point.strength
            for point in CornerResponse.of(turned).peaks(minimum_strength=0.5, border=3)
        }

        assert strengths == turned_strengths

    def test_but_the_descriptions_do_not_survive_it_at_all(self):
        """The honest half, and the number is the interesting part.

        Every corresponding pair sits 1.443376 apart on a scale to 2.0, which
        is the same distance a square's top-left corner sits from its own
        top-right corner. After a quarter turn the descriptor does not
        recognise a corner as itself; it recognises it as a different corner.
        There is no orientation assignment here and the docstring says so.
        """
        picture = bent_bar_picture()
        turned = Picture(np.rot90(picture.values))

        gaps = [
            PatchDescriptor.of(picture, point, 5).distance_to(
                PatchDescriptor.of(
                    turned,
                    Keypoint(
                        *turned_position(point.row, point.column, picture.height),
                        point.strength,
                    ),
                    5,
                )
            )
            for point in CornerResponse.of(picture).peaks(
                minimum_strength=0.5, border=3
            )
        ]

        assert all(gap == pytest.approx(1.443376) for gap in gaps)


class TestTheRatioTest:
    def test_a_genuine_match_wins_by_a_mile(self):
        """One square photographed twice: the winner is at distance 0.0 and the
        runner-up at 1.443376, so the ratio is 0.0."""
        first = square_picture(size=31, top=6, left=6, side=9)
        second = square_picture(size=31, top=10, left=11, side=9)

        matches = Descriptors.of(
            first, CornerResponse.of(first).peaks(minimum_strength=0.5, border=3), 5
        ).matched_to(
            Descriptors.of(
                second,
                CornerResponse.of(second).peaks(minimum_strength=0.5, border=3),
                5,
            )
        )

        assert len(matches) == 4
        best = matches.most_distinctive
        assert best.distance == 0.0
        assert best.runner_up_distance == pytest.approx(1.443376)
        assert best.ratio == 0.0

    def test_and_it_lands_on_the_corner_that_moved_by_the_known_shift(self):
        """A ratio near zero is worth nothing if the winner is the wrong corner."""
        first = square_picture(size=31, top=6, left=6, side=9)
        second = square_picture(size=31, top=10, left=11, side=9)

        matches = Descriptors.of(
            first, CornerResponse.of(first).peaks(minimum_strength=0.5, border=3), 5
        ).matched_to(
            Descriptors.of(
                second,
                CornerResponse.of(second).peaks(minimum_strength=0.5, border=3),
                5,
            )
        )

        assert all(
            match.second.keypoint == match.first.keypoint.shifted_by(4, 5)
            for match in matches
        )

    def test_a_coincidence_ties_with_its_runner_up_and_is_refused(self):
        """Two identical squares in one scene. Both corners are exactly as good
        an answer, so both distances are 0.0 and the ratio is 1.0, which is
        what the zero-over-zero convention is for."""
        single = square_picture(size=31, top=5, left=3, side=7)
        twins = twin_squares_picture()

        asked = Descriptors.of(
            single, CornerResponse.of(single).peaks(minimum_strength=0.5, border=3), 5
        )
        searched = Descriptors.of(
            twins, CornerResponse.of(twins).peaks(minimum_strength=0.5, border=3), 5
        )

        everything = asked.matched_to(searched, maximum_ratio=1.0)

        assert len(everything) == 4
        assert all(match.distance == 0.0 for match in everything)
        assert all(match.runner_up_distance == 0.0 for match in everything)
        assert all(match.ratio == 1.0 for match in everything)
        assert len(asked.matched_to(searched)) == 0

    def test_the_same_coincidence_with_the_tie_broken_reports_0_9807(self):
        """A faint texture over the twins makes the two candidates differ, so
        the ratio is computed rather than conventional: 0.050116 against
        0.051100. Still refused at Lowe's 0.8, which is the point.
        """
        single = square_picture(size=31, top=5, left=3, side=7)
        twins = twin_squares_picture()
        textured = Picture(twins.values + faint_texture(twins.shape))

        asked = Descriptors.of(
            single, CornerResponse.of(single).peaks(minimum_strength=0.5, border=3), 5
        )
        searched = Descriptors.of(
            textured,
            CornerResponse.of(textured).peaks(minimum_strength=0.5, border=3),
            5,
        )

        everything = asked.matched_to(searched, maximum_ratio=1.0)

        top_left = everything[0]
        assert top_left.distance == pytest.approx(0.050116, abs=1e-6)
        assert top_left.runner_up_distance == pytest.approx(0.051100, abs=1e-6)
        assert top_left.ratio == pytest.approx(0.980729, abs=1e-6)
        assert len(asked.matched_to(searched)) == 0

    def test_searching_a_set_with_no_runner_up_is_refused(self):
        """With one candidate the test cannot be asked, and answering the
        nearest anyway is the thing the ratio test exists to stop."""
        picture = square_picture()
        peaks = CornerResponse.of(picture).peaks(minimum_strength=0.5)
        descriptors = Descriptors.of(picture, peaks, 5)

        with pytest.raises(TooFewValuesError, match="runner-up"):
            descriptors.matched_to(Descriptors([descriptors[0]]))

    def test_asking_from_an_empty_set_answers_nothing(self):
        picture = square_picture()
        descriptors = Descriptors.of(
            picture, CornerResponse.of(picture).peaks(minimum_strength=0.5), 5
        )

        assert len(Descriptors([]).matched_to(descriptors)) == 0

    @pytest.mark.parametrize("maximum_ratio", [0.0, -0.5, 1.5])
    def test_a_threshold_outside_the_unit_interval_is_refused(self, maximum_ratio):
        picture = square_picture()
        descriptors = Descriptors.of(
            picture, CornerResponse.of(picture).peaks(minimum_strength=0.5), 5
        )

        with pytest.raises(InvalidValuesError, match=r"\(0, 1\]"):
            descriptors.matched_to(descriptors, maximum_ratio=maximum_ratio)

    def test_sets_describing_different_patch_sizes_cannot_be_matched(self):
        picture = square_picture()
        peaks = CornerResponse.of(picture).peaks(minimum_strength=0.5)

        with pytest.raises(ShapeMismatchError, match="one size"):
            Descriptors.of(picture, peaks, 3).matched_to(
                Descriptors.of(picture, peaks, 5)
            )

    def test_a_runner_up_nearer_than_the_winner_cannot_be_built(self):
        """An ordering cannot produce it, so it is refused rather than trusted."""
        descriptor = PatchDescriptor.of(square_picture(), Keypoint(6, 6, 4.0), 5)

        with pytest.raises(InvalidValuesError, match="not nearer"):
            DescriptorMatch(descriptor, descriptor, 1.0, 0.5)

    @pytest.mark.parametrize(
        "distance, runner_up, message",
        [(-1.0, 1.0, "never negative"), (np.nan, 1.0, "finite")],
    )
    def test_an_impossible_distance_is_refused(self, distance, runner_up, message):
        descriptor = PatchDescriptor.of(square_picture(), Keypoint(6, 6, 4.0), 5)

        with pytest.raises(InvalidValuesError, match=message):
            DescriptorMatch(descriptor, descriptor, distance, runner_up)

    def test_the_matches_are_iterated_rather_than_unpacked(self):
        picture = square_picture()
        descriptors = Descriptors.of(
            picture, CornerResponse.of(picture).peaks(minimum_strength=0.5), 5
        )

        matches = descriptors.matched_to(descriptors)

        assert len(matches) == 4
        assert len(list(matches)) == 4
        assert not hasattr(matches, "matches")

    def test_nothing_matched_has_no_best_match(self):
        picture = square_picture()
        twins = twin_squares_picture()
        asked = Descriptors.of(
            picture, CornerResponse.of(picture).peaks(minimum_strength=0.5), 5
        )
        searched = Descriptors.of(
            twins, CornerResponse.of(twins).peaks(minimum_strength=0.5, border=3), 5
        )

        with pytest.raises(EmptyValuesError, match="no descriptors matched"):
            _ = asked.matched_to(searched).most_distinctive
