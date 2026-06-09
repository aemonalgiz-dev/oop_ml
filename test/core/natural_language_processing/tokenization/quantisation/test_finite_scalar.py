"""Spec for FiniteScalarQuantizer -- a vector to an id with no codebook to learn.

Pinned against the paper's ``levels = (8, 5, 5, 5)`` on the input
``(1.0, -0.5, 0.2, 3.0)``, worked by hand in the module docstring: digits
``(6, 1, 2, 4)``, id ``894``, reconstruction ``(5/7, -1/2, 0, 1)``. The
distortion constant below was computed independently with ``math.tanh`` and
Python's ``round``; the module lands one ulp away, which is the tolerance.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.tokenization.quantisation.codebook import (
    CodeAssignment,
    Codebook,
    CodebookQuantizer,
)
from oop_ml.core.natural_language_processing.tokenization.quantisation.finite_scalar import (
    FiniteScalarQuantizer,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer

PAPER_LEVELS = (8, 5, 5, 5)
WORKED_INPUT = np.array([[1.0, -0.5, 0.2, 3.0]])
WORKED_DIGITS = (6, 1, 2, 4)
WORKED_ID = 6 + 8 * (1 + 5 * (2 + 5 * 4))
WORKED_RECONSTRUCTION = np.array([[5 / 7, -0.5, 0.0, 1.0]])
# sum of (tanh(z_d) - coordinate_d)^2 over the four coordinates, by hand.
WORKED_DISTORTION = 0.04265467092229371

SATURATING = 1000.0


def make_paper_quantizer() -> FiniteScalarQuantizer:
    return FiniteScalarQuantizer(levels=PAPER_LEVELS)


def random_inputs(dimension: int, n_rows: int = 200) -> np.ndarray:
    return np.random.default_rng(7).normal(0.0, 1.5, size=(n_rows, dimension))


class TestConstruction:
    def test_the_papers_example_has_a_thousand_codes(self):
        quantizer = make_paper_quantizer()

        assert WORKED_ID == 894
        assert quantizer.n_codes == 1000
        assert quantizer.dimension == 4

    def test_a_list_of_levels_becomes_a_tuple(self):
        assert FiniteScalarQuantizer(levels=[8, 5]).levels == (8, 5)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "levels",
        [(1, 5), (0,), (-3,), (), (2.5, 5), ("eight",)],
        ids=["one-level", "zero", "negative", "no-dimensions", "fractional", "text"],
    )
    def test_out_of_range_levels_are_refused(self, levels):
        with pytest.raises(ValidationError):
            FiniteScalarQuantizer(levels=levels)

    def test_levels_are_required(self):
        with pytest.raises(ValidationError):
            FiniteScalarQuantizer()  # type: ignore[call-arg]

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            FiniteScalarQuantizer(levels=(8, 5), codebook_size=40)  # type: ignore[call-arg]

    def test_is_neither_a_tokenizer_nor_fittable(self):
        """Its input is an array, not a text, and its grid is fixed by construction."""
        quantizer = make_paper_quantizer()

        assert not isinstance(quantizer, Tokenizer)
        assert not isinstance(quantizer, Fittable)
        assert not hasattr(quantizer, "fit")


class TestQuantize:
    def test_the_worked_input_lands_on_id_894(self):
        assignment = make_paper_quantizer().quantize(WORKED_INPUT)

        assert isinstance(assignment, CodeAssignment)
        assert assignment.ids == (WORKED_ID,)

    def test_the_worked_reconstruction_is_the_grid_point_exactly(self):
        reconstruction = make_paper_quantizer().quantize(WORKED_INPUT).reconstruction

        assert np.array_equal(reconstruction, WORKED_RECONSTRUCTION)

    def test_the_worked_distortion_is_against_the_bounded_input(self):
        assignment = make_paper_quantizer().quantize(WORKED_INPUT)
        by_definition = float(
            np.sum((np.tanh(WORKED_INPUT) - WORKED_RECONSTRUCTION) ** 2)
        )

        assert assignment.distortion == pytest.approx(WORKED_DISTORTION, abs=1e-15)
        assert assignment.distortion == pytest.approx(by_definition, abs=1e-15)

    def test_a_huge_input_saturates_onto_the_last_level(self):
        assignment = make_paper_quantizer().quantize(np.full((1, 4), SATURATING))

        assert assignment.ids == (999,)
        assert assignment.reconstruction.tolist() == [[1.0, 1.0, 1.0, 1.0]]

    def test_a_huge_negative_input_saturates_onto_the_first_level(self):
        assignment = make_paper_quantizer().quantize(np.full((1, 4), -SATURATING))

        assert assignment.ids == (0,)
        assert assignment.reconstruction.tolist() == [[-1.0, -1.0, -1.0, -1.0]]

    def test_a_saturating_input_has_zero_distortion_because_it_is_measured_bounded(
        self,
    ):
        """Against the raw input the gap would be about 999 per coordinate."""
        assert (
            make_paper_quantizer().quantize(np.full((1, 4), SATURATING)).distortion
            == 0.0
        )

    def test_the_zero_vector_under_an_even_level_rounds_half_to_even(self):
        """tanh(0) * 3.5 - 0.5 = -0.5 rounds to position 0, not -1."""
        assignment = make_paper_quantizer().quantize(np.zeros((1, 4)))

        assert assignment.ids == (500,)
        assert make_paper_quantizer().digits_of(500) == (4, 2, 2, 2)
        assert assignment.reconstruction.tolist() == [[1 / 7, 0.0, 0.0, 0.0]]

    def test_the_zero_vector_under_odd_levels_is_a_grid_point(self):
        assignment = FiniteScalarQuantizer(levels=(5, 5)).quantize(np.zeros((1, 2)))

        assert assignment.ids == (2 + 5 * 2,)
        assert assignment.reconstruction.tolist() == [[0.0, 0.0]]
        assert assignment.distortion == 0.0

    @pytest.mark.parametrize(
        ("inputs", "expected_id", "expected_point"),
        [
            ((0.0, -SATURATING), 1, [0.0, -1.0]),
            ((-SATURATING, SATURATING), 9, [-1.0, 1.0]),
            ((SATURATING, SATURATING), 11, [1.0, 1.0]),
            ((-SATURATING, -SATURATING), 0, [-1.0, -1.0]),
        ],
    )
    def test_dimension_zero_is_the_least_significant_digit(
        self, inputs, expected_id, expected_point
    ):
        """At levels (3, 4): id = digit_0 + 3 * digit_1."""
        assignment = FiniteScalarQuantizer(levels=(3, 4)).quantize(np.array([inputs]))

        assert assignment.ids == (expected_id,)
        assert assignment.reconstruction.tolist() == [expected_point]

    @pytest.mark.parametrize("level", range(2, 10))
    def test_a_sweep_over_one_dimension_produces_exactly_its_level_count_of_ids(
        self, level
    ):
        """The even-level offset: without it eight levels give seven interior positions."""
        sweep = np.linspace(-6.0, 6.0, 20001)[:, None]

        ids = set(FiniteScalarQuantizer(levels=(level,)).quantize(sweep).ids)

        assert ids == set(range(level))

    def test_quantizes_a_batch_row_by_row(self):
        quantizer = make_paper_quantizer()
        batch = np.vstack([WORKED_INPUT, np.zeros((1, 4)), np.full((1, 4), SATURATING)])

        assignment = quantizer.quantize(batch)

        assert assignment.ids == (WORKED_ID, 500, 999)
        assert assignment.n_rows == 3

    def test_the_callers_input_is_not_touched(self):
        inputs = WORKED_INPUT.copy()

        make_paper_quantizer().quantize(inputs)

        assert np.array_equal(inputs, WORKED_INPUT)

    def test_a_width_that_is_not_the_level_count_is_refused(self):
        with pytest.raises(ShapeMismatchError):
            make_paper_quantizer().quantize(np.zeros((1, 3)))

    def test_a_one_dimensional_input_is_refused(self):
        with pytest.raises(InvalidValuesError):
            make_paper_quantizer().quantize(np.zeros(4))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_input_is_refused(self, bad):
        with pytest.raises(InvalidValuesError):
            make_paper_quantizer().quantize(np.array([[bad, 0.0, 0.0, 0.0]]))

    def test_no_rows_is_refused(self):
        with pytest.raises(EmptyValuesError):
            make_paper_quantizer().quantize(np.empty((0, 4)))


class TestDequantize:
    def test_the_worked_id_dequantizes_to_the_worked_point(self):
        assert np.array_equal(
            make_paper_quantizer().dequantize([WORKED_ID]), WORKED_RECONSTRUCTION
        )

    def test_dequantizing_an_assignments_ids_recovers_its_reconstruction_exactly(
        self,
    ):
        quantizer = make_paper_quantizer()
        assignment = quantizer.quantize(random_inputs(4))

        assert np.array_equal(
            quantizer.dequantize(assignment.ids), assignment.reconstruction
        )

    def test_returns_a_fresh_writable_array(self):
        vectors = make_paper_quantizer().dequantize([0, 999])

        assert vectors.shape == (2, 4)

        vectors[0, 0] = 9.0

        assert vectors[0, 0] == 9.0

    def test_an_id_past_the_grid_raises_the_vocabularys_error(self):
        with pytest.raises(UnknownTokenError):
            make_paper_quantizer().dequantize([1000])

    def test_a_negative_id_raises_the_vocabularys_error(self):
        with pytest.raises(UnknownTokenError):
            make_paper_quantizer().dequantize([-1])

    def test_a_non_integer_id_is_refused(self):
        with pytest.raises(InvalidValuesError):
            make_paper_quantizer().dequantize([1.5])  # type: ignore[list-item]

    def test_no_ids_is_refused(self):
        with pytest.raises(EmptyValuesError):
            make_paper_quantizer().dequantize([])


class TestDigitsOf:
    def test_the_worked_id_decomposes_into_the_worked_digits(self):
        assert make_paper_quantizer().digits_of(WORKED_ID) == WORKED_DIGITS

    def test_the_first_and_last_ids_are_the_corners(self):
        quantizer = make_paper_quantizer()

        assert quantizer.digits_of(0) == (0, 0, 0, 0)
        assert quantizer.digits_of(999) == (7, 4, 4, 4)

    def test_every_digit_lies_below_its_level(self):
        quantizer = FiniteScalarQuantizer(levels=(3, 4, 2))

        for code_id in range(quantizer.n_codes):
            digits = quantizer.digits_of(code_id)
            assert all(
                0 <= digit < level
                for digit, level in zip(digits, quantizer.levels, strict=True)
            )

    def test_the_digits_are_distinct_across_the_whole_id_range(self):
        quantizer = FiniteScalarQuantizer(levels=(3, 4, 2))

        assert len({quantizer.digits_of(code_id) for code_id in range(24)}) == 24

    @pytest.mark.parametrize("code_id", [-1, 1000])
    def test_an_id_outside_the_grid_is_refused(self, code_id):
        with pytest.raises(UnknownTokenError):
            make_paper_quantizer().digits_of(code_id)


class TestCodebook:
    def test_materialises_every_grid_point_in_id_order(self):
        assert FiniteScalarQuantizer(levels=(2, 3)).codebook.vectors.tolist() == [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ]

    def test_three_levels_are_minus_one_zero_and_one(self):
        assert FiniteScalarQuantizer(levels=(3,)).codebook.vectors.tolist() == [
            [-1.0],
            [0.0],
            [1.0],
        ]

    def test_eight_levels_are_evenly_spaced_and_symmetric(self):
        grid = FiniteScalarQuantizer(levels=(8,)).codebook.vectors[:, 0]

        assert np.allclose(grid, [-1, -5 / 7, -3 / 7, -1 / 7, 1 / 7, 3 / 7, 5 / 7, 1])
        assert np.array_equal(grid, -grid[::-1])

    def test_is_a_real_codebook_of_the_right_size(self):
        codebook = make_paper_quantizer().codebook

        assert isinstance(codebook, Codebook)
        assert codebook.n_codes == 1000
        assert codebook.dimension == 4

    def test_the_corners_are_the_first_and_last_codes(self):
        codebook = make_paper_quantizer().codebook

        assert codebook.vector_of(0).tolist() == [-1.0, -1.0, -1.0, -1.0]
        assert codebook.vector_of(999).tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_each_row_is_what_dequantize_says_for_that_id(self):
        quantizer = make_paper_quantizer()
        codebook = quantizer.codebook

        assert np.array_equal(
            codebook.vectors, quantizer.dequantize(list(range(quantizer.n_codes)))
        )

    def test_every_grid_point_is_its_own_nearest_code(self):
        codebook = FiniteScalarQuantizer(levels=(3, 4, 2)).codebook

        assignment = CodebookQuantizer(codebook=codebook).quantize(codebook.vectors)

        assert assignment.ids == tuple(range(24))
        assert assignment.distortion == 0.0


class TestAgreementWithTheCodebookLookup:
    """Rounding each coordinate is nearest-neighbour search over the grid."""

    @pytest.mark.parametrize("levels", [PAPER_LEVELS, (3, 4), (2, 2, 2)])
    def test_the_two_routes_assign_identical_ids(self, levels):
        quantizer = FiniteScalarQuantizer(levels=levels)
        inputs = random_inputs(len(levels))

        direct = quantizer.quantize(inputs)
        via_lookup = CodebookQuantizer(codebook=quantizer.codebook).quantize(
            np.tanh(inputs)
        )

        assert direct.ids == via_lookup.ids

    @pytest.mark.parametrize("levels", [PAPER_LEVELS, (3, 4), (2, 2, 2)])
    def test_the_two_routes_agree_on_reconstruction_and_distortion_to_the_bit(
        self, levels
    ):
        quantizer = FiniteScalarQuantizer(levels=levels)
        inputs = random_inputs(len(levels))

        direct = quantizer.quantize(inputs)
        via_lookup = CodebookQuantizer(codebook=quantizer.codebook).quantize(
            np.tanh(inputs)
        )

        assert direct == via_lookup

    def test_random_inputs_use_a_real_spread_of_the_grid(self):
        """A guard on the tests above: they would pass vacuously on one id."""
        ids = set(make_paper_quantizer().quantize(random_inputs(4, n_rows=500)).ids)

        assert len(ids) > 100
