"""Spec for the picture, the sweep and the gradient field.

The three things every other method in the package is built from, so the
claims here are the ones a family spec is allowed to assume rather than
re-establish.

Fixtures are hand-drawn rather than photographic, because every number below
is meant to be checkable on paper. A four-wide picture whose left half is dark
and right half bright has exactly one edge, in a known place, at a known
angle, and any operator that disagrees is wrong rather than merely different.
"""

import numpy as np
import pytest

from oop_ml.core.computer_vision.edges import (
    GradientField,
    GradientOperator,
    weights_of,
)
from oop_ml.core.computer_vision.filtering import (
    EdgeRule,
    checked_weights,
    swept,
    windows_of,
)
from oop_ml.core.computer_vision.picture import BRIGHTNESS_WEIGHTS, Picture
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)

# Dark left half, bright right half. One edge, between columns 1 and 2.
VERTICAL_EDGE = [[0.0, 0.0, 1.0, 1.0]] * 5

# Dark top half, bright bottom half. One edge, between rows 1 and 2.
HORIZONTAL_EDGE = [[0.0] * 4, [0.0] * 4, [1.0] * 4, [1.0] * 4]

FLAT = [[0.5] * 4] * 4


class TestPicture:
    def test_carries_its_own_shape(self):
        picture = Picture(VERTICAL_EDGE)

        assert picture.shape == (5, 4)
        assert picture.height == 5
        assert picture.width == 4
        assert picture.n_pixels == 20

    def test_the_buffer_is_frozen_and_is_not_the_caller_s(self):
        values = np.zeros((3, 3))
        picture = Picture(values)
        values[0, 0] = 99.0

        assert picture.values[0, 0] == 0.0
        with pytest.raises(ValueError):
            picture.values[0, 0] = 1.0

    def test_the_array_protocol_hands_out_a_copy(self):
        picture = Picture(FLAT)

        assert not np.shares_memory(np.array(picture), picture.values)

    @pytest.mark.parametrize(
        "values",
        [[1.0, 2.0, 3.0], [[[1.0]]], 5.0],
    )
    def test_a_picture_that_is_not_two_dimensional_is_refused(self, values):
        with pytest.raises(InvalidValuesError, match="two-dimensional"):
            Picture(values)

    def test_a_picture_with_no_pixels_is_refused(self):
        with pytest.raises(EmptyValuesError):
            Picture(np.zeros((0, 4)))

    def test_a_missing_brightness_is_refused(self):
        with pytest.raises(InvalidValuesError, match="finite"):
            Picture([[0.0, np.nan], [1.0, 1.0]])

    def test_no_range_is_imposed(self):
        """A gradient is signed and a photograph runs to 255; both are pictures."""
        assert Picture([[-4.0, 900.0], [0.0, 1.0]]).n_pixels == 4

    def test_colour_collapses_by_weight_rather_than_by_average(self):
        red = Picture.from_channels([[[1.0, 0.0, 0.0]]])
        green = Picture.from_channels([[[0.0, 1.0, 0.0]]])
        blue = Picture.from_channels([[[0.0, 0.0, 1.0]]])

        assert red.values[0, 0] == pytest.approx(BRIGHTNESS_WEIGHTS[0])
        assert green.values[0, 0] == pytest.approx(BRIGHTNESS_WEIGHTS[1])
        assert blue.values[0, 0] == pytest.approx(BRIGHTNESS_WEIGHTS[2])
        assert green.values[0, 0] > blue.values[0, 0] * 9

    def test_a_colour_array_of_the_wrong_shape_is_refused(self):
        with pytest.raises(InvalidValuesError, match="height, width, 3"):
            Picture.from_channels([[[1.0, 0.0]]])

    def test_a_patch_is_the_rectangle_asked_for(self):
        patch = Picture(VERTICAL_EDGE).patch_at(1, 2, height=2, width=2)

        assert patch.shape == (2, 2)
        assert np.array_equal(np.array(patch), np.ones((2, 2)))

    def test_a_patch_running_off_the_edge_is_refused(self):
        with pytest.raises(InvalidValuesError, match="runs past"):
            Picture(VERTICAL_EDGE).patch_at(4, 0, height=3, width=3)

    def test_rescaling_puts_the_darkest_at_zero_and_the_brightest_at_one(self):
        rescaled = Picture([[10.0, 20.0], [30.0, 50.0]]).rescaled_to_unit()

        assert rescaled.darkest == 0.0
        assert rescaled.brightest == 1.0

    def test_rescaling_a_flat_picture_answers_zeros_rather_than_dividing(self):
        assert Picture(FLAT).rescaled_to_unit().brightest == 0.0

    def test_equality_is_by_value_and_defers_to_anything_else(self):
        assert Picture(FLAT) == Picture(FLAT)
        assert Picture(FLAT) != Picture(VERTICAL_EDGE)
        assert Picture(FLAT).__eq__("a picture") is NotImplemented


class TestTheSweep:
    def test_a_flat_grid_of_one_leaves_the_picture_alone(self):
        picture = Picture(VERTICAL_EDGE)

        assert swept(picture, [[1.0]]) == picture

    def test_correlation_rather_than_convolution(self):
        """The weights multiply the pixels in the order written.

        Convolution flips them, which reverses the sign here, and the whole
        package's sign convention rests on this not happening.
        """
        rising = Picture([[0.0, 0.0, 1.0, 1.0]])

        answer = swept(rising, [[-1.0, 0.0, 1.0]], EdgeRule.KEEP_VALID)

        assert answer.values[0, 0] == pytest.approx(1.0)

    def test_keeping_only_valid_positions_answers_a_smaller_picture(self):
        answer = swept(Picture(VERTICAL_EDGE), [[1.0] * 3] * 3, EdgeRule.KEEP_VALID)

        assert answer.shape == (3, 2)

    @pytest.mark.parametrize(
        "edge_rule", [EdgeRule.EXTEND, EdgeRule.WRAP, EdgeRule.PAD_WITH_ZERO]
    )
    def test_every_other_rule_answers_the_same_size(self, edge_rule):
        answer = swept(Picture(VERTICAL_EDGE), [[1.0] * 3] * 3, edge_rule)

        assert answer.shape == (5, 4)

    def test_the_edge_rules_disagree_where_they_are_supposed_to(self):
        """Each invents something different outside, and it shows at the border.

        Extending repeats the dark edge, so the first column stays dark.
        Padding with zero also reads dark there, so those two agree on this
        picture. Wrapping reads the *bright* right-hand side, so it invents an
        edge at the left border that the scene does not contain, which is the
        whole argument against it for a photograph.
        """
        picture = Picture(VERTICAL_EDGE)
        average = [[1 / 9] * 3] * 3

        extended = swept(picture, average, EdgeRule.EXTEND).values[2, 0]
        padded = swept(picture, average, EdgeRule.PAD_WITH_ZERO).values[2, 0]
        wrapped = swept(picture, average, EdgeRule.WRAP).values[2, 0]

        assert extended == pytest.approx(0.0)
        assert padded == pytest.approx(0.0)
        assert wrapped == pytest.approx(1 / 3)

    @pytest.mark.parametrize("weights", [[[1.0, 1.0]], [[1.0], [1.0]]])
    def test_an_even_side_is_refused_because_it_has_no_centre(self, weights):
        with pytest.raises(InvalidValuesError, match="centre pixel"):
            checked_weights(weights)

    def test_weights_that_are_not_a_grid_are_refused(self):
        with pytest.raises(InvalidValuesError, match="two-dimensional"):
            checked_weights([1.0, 0.0, 1.0])

    def test_a_grid_larger_than_the_picture_is_refused(self):
        with pytest.raises(ShapeMismatchError, match="does not fit"):
            swept(Picture([[1.0, 2.0], [3.0, 4.0]]), [[1.0] * 5] * 5)

    def test_every_patch_is_offered_at_every_position(self):
        windows = windows_of(Picture(VERTICAL_EDGE), 3, 3)

        assert windows.shape == (3, 2, 3, 3)
        assert np.array_equal(windows[0, 0], np.asarray(VERTICAL_EDGE)[0:3, 0:3])
        assert np.array_equal(windows[2, 1], np.asarray(VERTICAL_EDGE)[2:5, 1:4])

    def test_the_patches_are_a_view_rather_than_a_copy(self):
        """Asking for every patch of a large picture must not allocate them."""
        picture = Picture(VERTICAL_EDGE)

        assert np.shares_memory(windows_of(picture, 3, 3), picture.values)

    def test_a_patch_larger_than_the_picture_is_refused(self):
        with pytest.raises(ShapeMismatchError, match="does not fit"):
            windows_of(Picture(FLAT), 9, 9)


class TestGradients:
    def test_a_vertical_edge_has_a_gradient_pointing_right(self):
        field = GradientField.of(Picture(VERTICAL_EDGE))

        assert field.direction.values[2, 1] == pytest.approx(0.0)

    def test_and_the_edge_itself_runs_at_a_right_angle_to_that(self):
        field = GradientField.of(Picture(VERTICAL_EDGE))

        assert field.edge_direction.values[2, 1] == pytest.approx(np.pi / 2)

    def test_a_horizontal_edge_has_a_gradient_pointing_down(self):
        field = GradientField.of(Picture(HORIZONTAL_EDGE))

        assert field.direction.values[1, 2] == pytest.approx(np.pi / 2)

    def test_a_flat_picture_has_no_gradient_anywhere(self):
        field = GradientField.of(Picture(FLAT))

        assert field.magnitude.brightest == pytest.approx(0.0)

    def test_the_magnitude_is_never_negative(self):
        field = GradientField.of(Picture(VERTICAL_EDGE))

        assert field.magnitude.darkest >= 0.0

    @pytest.mark.parametrize(
        "operator, expected",
        [
            (GradientOperator.CENTRAL_DIFFERENCE, 0.5),
            (GradientOperator.PREWITT, 3.0),
            (GradientOperator.SOBEL, 4.0),
            (GradientOperator.SCHARR, 16.0),
        ],
    )
    def test_each_operator_answers_its_own_total_weight(self, operator, expected):
        """Worked by hand: on a clean step the answer is the sum of the
        positive weights, since each of them meets a one and each negative
        weight meets a zero."""
        field = GradientField.of(Picture(VERTICAL_EDGE), operator)

        assert field.magnitude.values[2, 1] == pytest.approx(expected)

    def test_every_operator_agrees_about_the_direction(self):
        """They differ in how much they smooth and not in what they mean."""
        directions = {
            operator: GradientField.of(
                Picture(VERTICAL_EDGE), operator
            ).direction.values[2, 1]
            for operator in GradientOperator
        }

        assert all(angle == pytest.approx(0.0) for angle in directions.values())

    def test_the_vertical_grid_is_the_horizontal_one_turned(self):
        across = np.asarray(weights_of(GradientOperator.SOBEL, vertical=False))
        down = np.asarray(weights_of(GradientOperator.SOBEL, vertical=True))

        assert np.array_equal(down, across.T)

    def test_two_directions_of_different_pictures_cannot_be_paired(self):
        with pytest.raises(ShapeMismatchError, match="the same picture"):
            GradientField(Picture(FLAT), Picture(VERTICAL_EDGE))

    def test_a_threshold_keeps_the_sharp_pixels_and_drops_the_flat_ones(self):
        field = GradientField.of(Picture(VERTICAL_EDGE))

        kept = field.strongest_at_least(1.0)

        assert kept.values[2, 1] == 1.0
        assert kept.values[2, 0] == 0.0

    def test_a_negative_threshold_is_refused(self):
        field = GradientField.of(Picture(FLAT))

        with pytest.raises(InvalidValuesError, match="never negative"):
            field.strongest_at_least(-0.1)

    def test_the_field_can_be_read_as_one_vector_per_pixel(self):
        field = GradientField.of(Picture(VERTICAL_EDGE))

        vectors = field.as_vectors()

        assert vectors.shape == (5, 4, 2)
        assert vectors[2, 1, 0] == pytest.approx(4.0)
        assert vectors[2, 1, 1] == pytest.approx(0.0)
