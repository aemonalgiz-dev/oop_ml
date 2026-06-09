"""Spec for the codebook lookup -- vector quantisation's id half.

Pinned against three codes in the plane, worked by hand in the module docstring:
codes (0, 0), (4, 0), (0, 3); inputs (1, 0), (3, 1), (1, 2), (2, 0); ids
0, 1, 2, 0 with squared gaps 1, 2, 2, 4, so a distortion of 9 / 4. The last
input is a deliberate tie between codes 0 and 1.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.clustering.centroids import Centroid, Centroids
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueTokensError,
    ShapeMismatchError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.tokenization.quantisation.codebook import (
    CodeAssignment,
    Codebook,
    CodebookQuantizer,
    checked_code_id,
    checked_vectors,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import Tokenizer

THREE_CODES = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
FOUR_INPUTS = np.array([[1.0, 0.0], [3.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
EXPECTED_IDS = (0, 1, 2, 0)
EXPECTED_SQUARED_GAPS = (1.0, 2.0, 2.0, 4.0)
EXPECTED_DISTORTION = 9 / 4


def make_codebook() -> Codebook:
    return Codebook(THREE_CODES)


def make_quantizer() -> CodebookQuantizer:
    return CodebookQuantizer(codebook=make_codebook())


def make_assignment() -> CodeAssignment:
    return CodeAssignment(
        np.array([0, 1, 2, 0]), THREE_CODES[[0, 1, 2, 0]], EXPECTED_DISTORTION
    )


class TestCodebookConstruction:
    def test_counts_its_codes_and_coordinates(self):
        codebook = make_codebook()

        assert codebook.n_codes == 3
        assert codebook.dimension == 2
        assert codebook.shape == (3, 2)
        assert len(codebook) == 3

    def test_iterates_the_codes_in_id_order(self):
        assert [code.tolist() for code in make_codebook()] == THREE_CODES.tolist()

    def test_coerces_a_nested_list_and_integers_to_float64(self):
        codebook = Codebook([[0, 0], [4, 0], [0, 3]])  # type: ignore[arg-type]

        assert codebook.vectors.dtype == np.float64
        assert codebook == make_codebook()

    def test_a_repeated_code_is_refused(self):
        with pytest.raises(NonUniqueTokensError):
            Codebook(np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 0.0]]))

    def test_zero_and_negative_zero_are_one_point(self):
        """Equality is ==, so two ids for the origin are refused either way."""
        with pytest.raises(NonUniqueTokensError):
            Codebook(np.array([[0.0], [-0.0]]))

    def test_no_codes_is_refused(self):
        with pytest.raises(EmptyValuesError):
            Codebook(np.empty((0, 2)))

    def test_codes_of_width_zero_are_refused(self):
        with pytest.raises(InvalidValuesError):
            Codebook(np.empty((1, 0)))

    @pytest.mark.parametrize(
        "vectors",
        [np.array([1.0, 2.0]), np.zeros((2, 2, 2)), np.float64(3.0)],
        ids=["one-dimensional", "three-dimensional", "scalar"],
    )
    def test_anything_but_a_two_dimensional_block_is_refused(self, vectors):
        with pytest.raises(InvalidValuesError):
            Codebook(vectors)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_coordinate_is_refused(self, bad):
        with pytest.raises(InvalidValuesError):
            Codebook(np.array([[0.0, 0.0], [1.0, bad]]))

    def test_a_non_numeric_block_is_refused(self):
        with pytest.raises(InvalidValuesError):
            Codebook([["a", "b"]])  # type: ignore[list-item]

    def test_from_centroids_takes_cluster_k_as_code_k(self):
        centroids = Centroids(
            [
                Centroid("cluster_0", np.array([0.0, 0.0]), ["x", "y"]),
                Centroid("cluster_1", np.array([4.0, 0.0]), ["x", "y"]),
                Centroid("cluster_2", np.array([0.0, 3.0]), ["x", "y"]),
            ]
        )

        assert Codebook.from_centroids(centroids) == make_codebook()


class TestCodebookLookup:
    def test_vector_of_reads_the_code_at_that_position(self):
        assert make_codebook().vector_of(1).tolist() == [4.0, 0.0]
        assert make_codebook()[2].tolist() == [0.0, 3.0]

    @pytest.mark.parametrize("code_id", [-1, 3, 100])
    def test_an_id_no_code_owns_raises_the_vocabularys_error(self, code_id):
        with pytest.raises(UnknownTokenError):
            make_codebook().vector_of(code_id)

    def test_a_numpy_integer_is_a_whole_number(self):
        assert make_codebook().vector_of(np.int64(2)).tolist() == [0.0, 3.0]  # type: ignore[arg-type]

    @pytest.mark.parametrize("code_id", [1.5, "1", None])
    def test_a_non_integer_id_is_refused(self, code_id):
        with pytest.raises(InvalidValuesError):
            make_codebook().vector_of(code_id)  # type: ignore[arg-type]

    def test_checked_code_id_refuses_before_it_ranges(self):
        assert checked_code_id(np.int64(1), 3) == 1  # type: ignore[arg-type]
        with pytest.raises(UnknownTokenError):
            checked_code_id(3, 3)


class TestCodebookBuffers:
    def test_the_codes_cannot_be_written(self):
        codebook = make_codebook()

        with pytest.raises(ValueError, match="read-only"):
            codebook.vectors[0, 0] = 9.0

    def test_a_looked_up_code_cannot_be_written(self):
        codebook = make_codebook()

        with pytest.raises(ValueError, match="read-only"):
            codebook.vector_of(0)[0] = 9.0

    def test_the_callers_array_stays_theirs(self):
        source = THREE_CODES.copy()
        codebook = Codebook(source)

        source[0, 0] = 9.0

        assert codebook.vectors[0, 0] == 0.0

    def test_np_array_is_a_genuine_writable_copy(self):
        codebook = make_codebook()
        converted = np.array(codebook)

        assert not np.shares_memory(converted, codebook.vectors)

        converted[0, 0] = 9.0

        assert codebook.vectors[0, 0] == 0.0

    def test_asarray_may_still_share(self):
        codebook = make_codebook()

        assert np.shares_memory(np.asarray(codebook), codebook.vectors)

    def test_copy_false_with_a_forced_conversion_raises(self):
        with pytest.raises(ValueError, match="copy"):
            make_codebook().__array__(dtype=np.float32, copy=False)


class TestCodebookEquality:
    def test_equal_when_every_code_matches(self):
        assert make_codebook() == Codebook(THREE_CODES.copy())

    def test_unequal_when_the_order_differs(self):
        """Order is the id assignment, so a reordering is a different book."""
        assert make_codebook() != Codebook(THREE_CODES[[1, 0, 2]])

    def test_unequal_when_the_shape_differs(self):
        assert make_codebook() != Codebook(THREE_CODES[:2])

    def test_compares_unequal_to_a_bare_list(self):
        assert make_codebook() != THREE_CODES.tolist()

    def test_is_not_hashable(self):
        with pytest.raises(TypeError):
            hash(make_codebook())


class TestCodeAssignment:
    def test_reports_ids_as_python_ints(self):
        assignment = make_assignment()

        assert assignment.ids == EXPECTED_IDS
        assert all(type(code_id) is int for code_id in assignment.ids)

    def test_counts_rows_and_coordinates(self):
        assignment = make_assignment()

        assert assignment.n_rows == 4
        assert len(assignment) == 4
        assert assignment.dimension == 2

    def test_keeps_the_distortion_as_a_float(self):
        assert make_assignment().distortion == EXPECTED_DISTORTION
        assert type(make_assignment().distortion) is float

    def test_code_ids_are_intp(self):
        assert make_assignment().code_ids.dtype == np.intp

    def test_a_length_mismatch_is_refused(self):
        with pytest.raises(NonEqualArrayLengthError):
            CodeAssignment(np.array([0, 1]), THREE_CODES, 0.0)

    def test_no_rows_is_refused(self):
        with pytest.raises(EmptyValuesError):
            CodeAssignment(np.array([], dtype=np.intp), np.empty((0, 2)), 0.0)

    def test_a_negative_id_is_refused(self):
        with pytest.raises(InvalidValuesError):
            CodeAssignment(np.array([0, -1]), THREE_CODES[:2], 0.0)

    def test_float_ids_are_refused(self):
        with pytest.raises(InvalidValuesError):
            CodeAssignment(np.array([0.0, 1.0]), THREE_CODES[:2], 0.0)  # type: ignore[arg-type]

    def test_two_dimensional_ids_are_refused(self):
        with pytest.raises(InvalidValuesError):
            CodeAssignment(np.array([[0], [1]]), THREE_CODES[:2], 0.0)

    def test_a_one_dimensional_reconstruction_is_refused(self):
        with pytest.raises(InvalidValuesError):
            CodeAssignment(np.array([0]), np.array([0.0, 0.0]), 0.0)

    def test_a_non_finite_reconstruction_is_refused(self):
        with pytest.raises(InvalidValuesError):
            CodeAssignment(np.array([0]), np.array([[0.0, np.nan]]), 0.0)

    @pytest.mark.parametrize("distortion", [-0.5, np.nan, np.inf, "big"])
    def test_a_distortion_that_is_not_a_finite_non_negative_number_is_refused(
        self, distortion
    ):
        with pytest.raises(InvalidValuesError):
            CodeAssignment(np.array([0]), THREE_CODES[:1], distortion)

    def test_both_buffers_are_frozen(self):
        assignment = make_assignment()

        with pytest.raises(ValueError, match="read-only"):
            assignment.code_ids[0] = 2
        with pytest.raises(ValueError, match="read-only"):
            assignment.reconstruction[0, 0] = 9.0

    def test_the_callers_arrays_stay_theirs(self):
        ids = np.array([0, 1, 2, 0])
        reconstruction = THREE_CODES[[0, 1, 2, 0]]
        assignment = CodeAssignment(ids, reconstruction, EXPECTED_DISTORTION)

        ids[0] = 2
        reconstruction[0, 0] = 9.0

        assert assignment.ids == EXPECTED_IDS
        assert assignment.reconstruction[0, 0] == 0.0

    def test_np_array_of_the_reconstruction_is_a_real_copy(self):
        assignment = make_assignment()
        converted = np.array(assignment.reconstruction)

        assert not np.shares_memory(converted, assignment.reconstruction)

        converted[0, 0] = 9.0

        assert assignment.reconstruction[0, 0] == 0.0

    def test_equal_when_ids_reconstruction_and_distortion_match(self):
        assert make_assignment() == make_assignment()

    def test_unequal_when_only_the_distortion_differs(self):
        other = CodeAssignment(
            np.array([0, 1, 2, 0]), THREE_CODES[[0, 1, 2, 0]], EXPECTED_DISTORTION + 1
        )

        assert make_assignment() != other

    def test_compares_unequal_to_a_tuple_of_ids(self):
        assert make_assignment() != EXPECTED_IDS


class TestQuantize:
    def test_assigns_each_row_its_nearest_code(self):
        assert make_quantizer().quantize(FOUR_INPUTS).ids == EXPECTED_IDS

    def test_reconstructs_each_row_as_its_codes_vector(self):
        reconstruction = make_quantizer().quantize(FOUR_INPUTS).reconstruction

        assert reconstruction.tolist() == THREE_CODES[list(EXPECTED_IDS)].tolist()

    def test_the_distortion_is_the_mean_squared_gap_worked_by_hand(self):
        assignment = make_quantizer().quantize(FOUR_INPUTS)

        assert assignment.distortion == EXPECTED_DISTORTION
        assert assignment.distortion == sum(EXPECTED_SQUARED_GAPS) / 4

    def test_a_tie_goes_to_the_lower_id(self):
        """(2, 0) is exactly 2 from both (0, 0) and (4, 0)."""
        assert make_quantizer().quantize(FOUR_INPUTS[3:4]).ids == (0,)

    def test_the_tie_rule_is_about_the_id_and_not_the_vector(self):
        """With (4, 0) moved to position 0, the tied row follows the position."""
        reordered = CodebookQuantizer(codebook=Codebook(THREE_CODES[[1, 0, 2]]))

        assignment = reordered.quantize(FOUR_INPUTS)

        assert assignment.ids == (1, 0, 2, 0)
        assert assignment.reconstruction[3].tolist() == [4.0, 0.0]

    def test_the_codes_themselves_quantize_to_their_own_ids_at_no_cost(self):
        assignment = make_quantizer().quantize(THREE_CODES)

        assert assignment.ids == (0, 1, 2)
        assert assignment.distortion == 0.0
        assert np.array_equal(assignment.reconstruction, THREE_CODES)

    def test_a_single_vector_is_a_block_of_one_row(self):
        assignment = make_quantizer().quantize(np.array([[3.0, 1.0]]))

        assert assignment.ids == (1,)
        assert assignment.n_rows == 1

    def test_accepts_an_integer_block(self):
        assert make_quantizer().quantize(np.array([[3, 1]])).ids == (1,)

    def test_the_callers_input_is_not_touched(self):
        inputs = FOUR_INPUTS.copy()

        make_quantizer().quantize(inputs)

        assert np.array_equal(inputs, FOUR_INPUTS)

    def test_a_width_that_is_not_the_codes_is_refused(self):
        with pytest.raises(ShapeMismatchError):
            make_quantizer().quantize(np.array([[1.0, 0.0, 0.0]]))

    def test_a_one_dimensional_input_is_refused_even_at_the_right_length(self):
        with pytest.raises(InvalidValuesError):
            make_quantizer().quantize(np.array([1.0, 0.0]))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_a_non_finite_input_is_refused(self, bad):
        with pytest.raises(InvalidValuesError):
            make_quantizer().quantize(np.array([[1.0, bad]]))

    def test_no_rows_is_refused(self):
        with pytest.raises(EmptyValuesError):
            make_quantizer().quantize(np.empty((0, 2)))

    def test_checked_vectors_orders_its_guards_shape_before_finiteness(self):
        with pytest.raises(ShapeMismatchError):
            checked_vectors(np.array([[np.nan, 0.0, 0.0]]), 2)


class TestDequantize:
    def test_returns_the_code_vectors_in_id_order(self):
        assert make_quantizer().dequantize([2, 0, 1]).tolist() == [
            [0.0, 3.0],
            [0.0, 0.0],
            [4.0, 0.0],
        ]

    def test_returns_a_fresh_writable_two_dimensional_array(self):
        vectors = make_quantizer().dequantize([1])

        assert vectors.shape == (1, 2)
        assert not np.shares_memory(vectors, make_codebook().vectors)

        vectors[0, 0] = 9.0

        assert vectors[0, 0] == 9.0

    def test_round_trips_the_codes_exactly(self):
        quantizer = make_quantizer()

        assert np.array_equal(
            quantizer.dequantize(quantizer.quantize(THREE_CODES).ids), THREE_CODES
        )

    def test_round_trips_an_assignments_reconstruction(self):
        quantizer = make_quantizer()
        assignment = quantizer.quantize(FOUR_INPUTS)

        assert np.array_equal(
            quantizer.dequantize(assignment.ids), assignment.reconstruction
        )

    def test_an_unknown_id_raises_the_vocabularys_error(self):
        with pytest.raises(UnknownTokenError):
            make_quantizer().dequantize([0, 3])

    def test_a_non_integer_id_is_refused(self):
        with pytest.raises(InvalidValuesError):
            make_quantizer().dequantize([0.5])  # type: ignore[list-item]

    def test_no_ids_is_refused(self):
        with pytest.raises(EmptyValuesError):
            make_quantizer().dequantize([])


class TestQuantizerConstruction:
    def test_delegates_its_counts_to_the_codebook(self):
        quantizer = make_quantizer()

        assert quantizer.n_codes == 3
        assert quantizer.dimension == 2

    def test_the_codebook_is_required(self):
        with pytest.raises(ValidationError):
            CodebookQuantizer()  # type: ignore[call-arg]

    def test_a_bare_array_is_not_a_codebook(self):
        with pytest.raises(ValidationError):
            CodebookQuantizer(codebook=THREE_CODES)  # type: ignore[arg-type]

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            CodebookQuantizer(codebook=make_codebook(), metric="cosine")  # type: ignore[call-arg]

    def test_two_quantizers_compare_by_their_books(self):
        assert make_quantizer() == make_quantizer()
        assert make_quantizer() != CodebookQuantizer(
            codebook=Codebook(THREE_CODES[[1, 0, 2]])
        )

    def test_is_neither_a_tokenizer_nor_fittable(self):
        """Its input is an array, not a text, and its codebook is learned elsewhere."""
        quantizer = make_quantizer()

        assert not isinstance(quantizer, Tokenizer)
        assert not isinstance(quantizer, Fittable)
        assert not hasattr(quantizer, "fit")
