"""Spec for PointwiseMutualInformationEmbeddings and its PMI matrix.

Pinned against the three sentences worked by hand in the module docstring,
and against the designed two-topic corpus in ``fixtures.py``.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    ShapeMismatchError,
    TooFewValuesError,
    UndefinedMetricError,
)
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.counts.pointwise_mutual_information import (
    PointwiseMutualInformationEmbeddings,
    PointwiseMutualInformationMatrix,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from test.core.natural_language_processing.embeddings.counts.fixtures import (
    COOKING_WORDS,
    SAILING_WORDS,
    TINY_PMI_CORPUS,
    TWO_TOPIC_CORPUS,
)

LOG_TWO = float(np.log(2.0))
LOG_THREE = float(np.log(3.0))
LOG_THREE_HALVES = float(np.log(1.5))
LOG_FIVE = float(np.log(5.0))
# The two smoothed entries worked by hand in the module docstring.
SMOOTHED_CAT_RAN = 0.861995
SMOOTHED_THE_CAT = 0.803104
# Twenty-three words, so the most components the square matrix has.
TWO_TOPIC_FULL_RANK = 23


def tiny(
    context_distribution_smoothing: float = 1.0, shift: float = 0.0
) -> PointwiseMutualInformationEmbeddings:
    return PointwiseMutualInformationEmbeddings(
        dimension=2,
        window=1,
        context_distribution_smoothing=context_distribution_smoothing,
        shift=shift,
    ).fit(TINY_PMI_CORPUS)


def two_topics(**keywords: object) -> PointwiseMutualInformationEmbeddings:
    settings: dict[str, object] = {"dimension": 4, "window": 5}
    settings.update(keywords)
    return PointwiseMutualInformationEmbeddings(**settings).fit(TWO_TOPIC_CORPUS)  # type: ignore[arg-type]


def mean_similarity(
    model: PointwiseMutualInformationEmbeddings, first: list[str], second: list[str]
) -> float:
    pairs = [
        model.similarity(one, other)
        for one in first
        for other in second
        if one != other
    ]
    return float(np.mean(pairs))


class TestTheThreeSentenceExample:
    def test_the_vocabulary_is_commonest_first_then_alphabetical(self):
        assert list(tiny().vocabulary) == ["the", "cat", "sat", "dog", "ran"]

    def test_the_counts_are_as_worked(self):
        cooccurrence = tiny().cooccurrence

        assert cooccurrence.count_between("the", "cat") == 2.0
        assert cooccurrence.count_between("the", "dog") == 1.0
        assert cooccurrence.count_between("cat", "sat") == 1.0
        assert cooccurrence.count_between("cat", "ran") == 1.0
        assert np.array_equal(cooccurrence.word_totals, [3.0, 4.0, 2.0, 2.0, 1.0])
        assert cooccurrence.total == 12.0

    def test_pmi_of_the_and_cat_is_log_two(self):
        assert tiny().pointwise_mutual_information.value_between("the", "cat") == (
            pytest.approx(LOG_TWO, abs=1e-9)
        )

    def test_pmi_of_cat_and_ran_is_log_three(self):
        assert tiny().pointwise_mutual_information.value_between("cat", "ran") == (
            pytest.approx(LOG_THREE, abs=1e-9)
        )

    def test_pmi_of_cat_and_sat_is_log_three_halves(self):
        assert tiny().pointwise_mutual_information.value_between("cat", "sat") == (
            pytest.approx(LOG_THREE_HALVES, abs=1e-9)
        )

    def test_unsmoothed_pmi_is_symmetric_because_the_counts_are(self):
        values = tiny().pointwise_mutual_information.values

        assert np.allclose(values, values.T)

    def test_a_pair_never_seen_together_is_zero_not_minus_infinity(self):
        matrix = tiny().pointwise_mutual_information

        assert matrix.value_between("the", "sat") == 0.0
        assert matrix.value_between("the", "the") == 0.0
        assert np.all(np.isfinite(matrix.values))
        assert matrix.n_positive == 10

    def test_smoothing_lowers_a_rare_contexts_pmi_and_raises_a_frequent_ones(self):
        smoothed = tiny(
            context_distribution_smoothing=0.75
        ).pointwise_mutual_information

        assert smoothed.value_between("cat", "ran") == pytest.approx(
            SMOOTHED_CAT_RAN, abs=1e-6
        )
        assert smoothed.value_between("cat", "ran") < LOG_THREE
        assert smoothed.value_between("the", "cat") == pytest.approx(
            SMOOTHED_THE_CAT, abs=1e-6
        )
        assert smoothed.value_between("the", "cat") > LOG_TWO

    def test_smoothing_breaks_the_symmetry(self):
        """Only the context marginal is smoothed, so ``PMI(w, c) != PMI(c, w)``."""
        values = tiny(
            context_distribution_smoothing=0.75
        ).pointwise_mutual_information.values

        assert not np.allclose(values, values.T)

    def test_a_shift_of_log_five_zeroes_the_whole_matrix_here(self):
        model = tiny(shift=LOG_FIVE)

        assert model.pointwise_mutual_information.n_positive == 0
        assert not model.embeddings.table.any()
        with pytest.raises(UndefinedMetricError):
            model.similarity("the", "cat")

    def test_the_shift_is_subtracted_before_clipping_at_zero(self):
        matrix = tiny(shift=0.5).pointwise_mutual_information

        assert matrix.value_between("the", "cat") == pytest.approx(
            LOG_TWO - 0.5, abs=1e-9
        )
        assert matrix.value_between("cat", "sat") == 0.0

    def test_harmonic_weighting_counts_a_neighbour_two_away_as_a_half(self):
        model = PointwiseMutualInformationEmbeddings(
            dimension=2, window=2, weighting=ContextWeighting.HARMONIC
        ).fit(TINY_PMI_CORPUS)

        assert model.cooccurrence.count_between("the", "sat") == 1.0  # two sentences
        assert model.cooccurrence.count_between("the", "ran") == 0.5


class TestTwoTopics:
    def test_two_cooking_words_are_nearer_than_a_cooking_and_a_sailing_word(self):
        model = two_topics()

        assert model.similarity("flour", "sugar") == pytest.approx(0.9010, abs=1e-3)
        assert model.similarity("flour", "anchor") == pytest.approx(0.0081, abs=1e-3)

    def test_within_topic_similarity_exceeds_across_on_average(self):
        model = two_topics()
        within = mean_similarity(model, COOKING_WORDS, COOKING_WORDS)
        across = mean_similarity(model, COOKING_WORDS, SAILING_WORDS)

        assert within == pytest.approx(0.6732, abs=1e-3)
        assert across == pytest.approx(0.0086, abs=1e-3)

    def test_the_nearest_words_to_flour_are_cooking_words(self):
        nearest = two_topics().most_similar("flour", n_results=3)

        assert set(nearest.words) <= set(COOKING_WORDS)

    def test_a_shift_of_log_five_zeroes_more_entries_than_no_shift(self):
        unshifted = two_topics(window=2).pointwise_mutual_information
        shifted = two_topics(window=2, shift=LOG_FIVE).pointwise_mutual_information

        assert unshifted.n_positive == 166
        assert shifted.n_positive == 40
        assert shifted.n_positive < unshifted.n_positive

    def test_singular_values_descend(self):
        singular_values = two_topics(dimension=8).singular_values

        assert np.all(np.diff(singular_values) <= 0.0)
        assert singular_values[:2] == pytest.approx([6.644, 6.1406], abs=1e-3)

    def test_exponent_one_at_full_rank_reconstructs_the_matrix(self):
        model = two_topics(dimension=TWO_TOPIC_FULL_RANK, singular_value_exponent=1.0)

        assert np.allclose(
            model.embeddings.table @ model.decomposition.right_vectors.T,
            model.pointwise_mutual_information.values,
            atol=1e-10,
        )
        assert np.allclose(
            model.decomposition.reconstruction(),
            model.pointwise_mutual_information.values,
            atol=1e-10,
        )

    def test_exponent_zero_leaves_the_orthonormal_left_vectors(self):
        table = two_topics(singular_value_exponent=0.0).embeddings.table

        assert np.linalg.norm(table, axis=0) == pytest.approx([1.0] * 4)

    def test_exponent_half_squares_to_the_product_of_the_two_extremes(self):
        """``(u sqrt(s))^2 = u * (u s)``, entry by entry."""
        bare = two_topics(singular_value_exponent=0.0).embeddings.table
        half = two_topics(singular_value_exponent=0.5).embeddings.table
        full = two_topics(singular_value_exponent=1.0).embeddings.table

        assert np.allclose(half**2, bare * full)

    def test_a_word_with_no_association_gets_the_zero_vector(self):
        model = PointwiseMutualInformationEmbeddings(dimension=2, window=2).fit(
            ["alone", *TINY_PMI_CORPUS]
        )

        assert not model.vector_of("alone").values.any()
        with pytest.raises(UndefinedMetricError):
            model.similarity("alone", "cat")

    def test_the_matrix_is_over_the_whole_vocabulary(self):
        model = two_topics()

        assert model.pointwise_mutual_information.n_words == TWO_TOPIC_FULL_RANK
        assert model.pointwise_mutual_information.values.shape == (
            TWO_TOPIC_FULL_RANK,
            TWO_TOPIC_FULL_RANK,
        )
        assert model.embeddings.n_words == TWO_TOPIC_FULL_RANK
        assert model.embeddings.dimension == 4


class TestRefusals:
    def test_a_dimension_above_the_vocabulary_is_refused_at_fit(self):
        with pytest.raises(TooFewValuesError):
            PointwiseMutualInformationEmbeddings(dimension=6, window=1).fit(
                TINY_PMI_CORPUS
            )

    def test_exactly_the_vocabulary_size_is_accepted(self):
        assert (
            PointwiseMutualInformationEmbeddings(dimension=5, window=1)
            .fit(TINY_PMI_CORPUS)
            .is_fitted
        )

    @pytest.mark.parametrize(
        "keywords",
        [
            {"dimension": 0},
            {"window": 0},
            {"context_distribution_smoothing": 0.0},
            {"context_distribution_smoothing": 1.5},
            {"shift": -0.1},
            {"singular_value_exponent": 1.5},
            {"singular_value_exponent": -0.1},
            {"minimum_count": 0},
            {"alpha": 0.75},
        ],
    )
    def test_bad_construction_is_refused_by_pydantic(self, keywords: dict[str, object]):
        with pytest.raises(ValidationError):
            PointwiseMutualInformationEmbeddings(**keywords)  # type: ignore[arg-type]

    def test_the_boundaries_of_the_ranges_are_accepted(self):
        model = PointwiseMutualInformationEmbeddings(
            context_distribution_smoothing=1.0, shift=0.0, singular_value_exponent=0.0
        )

        assert model.context_distribution_smoothing == 1.0

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            PointwiseMutualInformationEmbeddings(dimension=1).fit("the cat sat")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            PointwiseMutualInformationEmbeddings(dimension=1).fit(["  ", ""])

    def test_no_word_reaching_the_minimum_count_is_refused(self):
        with pytest.raises(TooFewValuesError):
            PointwiseMutualInformationEmbeddings(dimension=1, minimum_count=10).fit(
                TINY_PMI_CORPUS
            )

    def test_minimum_count_drops_the_rare_words_before_windowing(self):
        model = PointwiseMutualInformationEmbeddings(
            dimension=2, window=1, minimum_count=2
        ).fit(TINY_PMI_CORPUS)

        assert list(model.vocabulary) == ["the", "cat", "sat"]
        # ``the dog sat`` became ``the sat`` once ``dog`` was dropped.
        assert model.cooccurrence.count_between("the", "sat") == 1.0

    def test_before_fit_everything_learned_raises_not_fitted(self):
        model = PointwiseMutualInformationEmbeddings(dimension=2)

        for attribute in (
            "embeddings",
            "cooccurrence",
            "pointwise_mutual_information",
            "singular_values",
            "decomposition",
            "vocabulary",
        ):
            with pytest.raises(NotFittedError):
                getattr(model, attribute)

    def test_fit_returns_self(self):
        model = PointwiseMutualInformationEmbeddings(dimension=2, window=1)

        assert model.fit(TINY_PMI_CORPUS) is model

    def test_a_failed_fit_leaves_the_model_unfitted(self):
        model = PointwiseMutualInformationEmbeddings(dimension=6, window=1)

        with pytest.raises(TooFewValuesError):
            model.fit(TINY_PMI_CORPUS)
        assert not model.is_fitted


def two_word_cooccurrence(counts: list[list[float]]) -> CooccurrenceMatrix:
    return CooccurrenceMatrix(
        Vocabulary(["a", "b"]), np.array(counts), 1, ContextWeighting.UNIFORM
    )


class TestPointwiseMutualInformationMatrix:
    def test_no_cooccurrences_at_all_gives_all_zeros(self):
        matrix = PointwiseMutualInformationMatrix.from_cooccurrence(
            two_word_cooccurrence([[0.0, 0.0], [0.0, 0.0]])
        )

        assert not matrix.values.any()
        assert matrix.n_positive == 0

    def test_two_words_only_ever_together_have_pmi_log_of_the_total_over_one(self):
        """``a b`` alone: counts 1 each way, total 2, marginals 1 each, so
        ``PMI = log(1 * 2 / (1 * 1)) = log 2`` at ``alpha = 1``."""
        matrix = PointwiseMutualInformationMatrix.from_cooccurrence(
            two_word_cooccurrence([[0.0, 1.0], [1.0, 0.0]]),
            context_distribution_smoothing=1.0,
        )

        assert matrix.value_between("a", "b") == pytest.approx(LOG_TWO)
        assert matrix.value_between("a", "a") == 0.0

    def test_the_rule_is_recorded(self):
        matrix = PointwiseMutualInformationMatrix.from_cooccurrence(
            two_word_cooccurrence([[0.0, 1.0], [1.0, 0.0]]), 0.75, 0.25
        )

        assert matrix.context_distribution_smoothing == 0.75
        assert matrix.shift == 0.25
        assert matrix.n_words == 2

    def test_negative_values_are_refused(self):
        with pytest.raises(InvalidValuesError):
            PointwiseMutualInformationMatrix(
                Vocabulary(["a"]), np.array([[-1.0]]), 0.0, 0.75
            )

    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(InvalidValuesError):
            PointwiseMutualInformationMatrix(
                Vocabulary(["a", "b"]), np.zeros((2, 3)), 0.0, 0.75
            )

    def test_a_matrix_of_the_wrong_size_for_the_vocabulary_is_refused(self):
        with pytest.raises(ShapeMismatchError):
            PointwiseMutualInformationMatrix(
                Vocabulary(["a", "b"]), np.zeros((3, 3)), 0.0, 0.75
            )

    def test_a_non_finite_value_is_refused(self):
        with pytest.raises(InvalidValuesError):
            PointwiseMutualInformationMatrix(
                Vocabulary(["a"]), np.array([[np.inf]]), 0.0, 0.75
            )

    def test_non_numeric_values_are_refused(self):
        with pytest.raises(InvalidValuesError):
            PointwiseMutualInformationMatrix(
                Vocabulary(["a"]),
                np.array([["text"]]),  # type: ignore[arg-type]
                0.0,
                0.75,
            )

    @pytest.mark.parametrize("shift", [-0.1, float("nan")])
    def test_a_bad_shift_is_refused_by_both_routes(self, shift: float):
        with pytest.raises(InvalidValuesError):
            PointwiseMutualInformationMatrix(
                Vocabulary(["a"]), np.zeros((1, 1)), shift, 0.75
            )
        with pytest.raises(InvalidValuesError):
            PointwiseMutualInformationMatrix.from_cooccurrence(
                two_word_cooccurrence([[0.0, 1.0], [1.0, 0.0]]), shift=shift
            )

    @pytest.mark.parametrize("smoothing", [0.0, 1.1])
    def test_smoothing_outside_the_unit_interval_is_refused_by_both_routes(
        self, smoothing: float
    ):
        with pytest.raises(InvalidValuesError):
            PointwiseMutualInformationMatrix(
                Vocabulary(["a"]), np.zeros((1, 1)), 0.0, smoothing
            )
        with pytest.raises(InvalidValuesError):
            PointwiseMutualInformationMatrix.from_cooccurrence(
                two_word_cooccurrence([[0.0, 1.0], [1.0, 0.0]]), smoothing
            )

    def test_values_are_copied_and_frozen(self):
        source = np.array([[0.0, 1.0], [1.0, 0.0]])
        matrix = PointwiseMutualInformationMatrix(
            Vocabulary(["a", "b"]), source, 0.0, 1.0
        )
        source[0, 1] = 5.0

        assert matrix.value_between("a", "b") == 1.0
        assert not matrix.values.flags.writeable

    def test_the_array_protocol_copies_when_asked_and_shares_when_allowed(self):
        matrix = tiny().pointwise_mutual_information

        assert not np.shares_memory(np.array(matrix), matrix.values)
        assert np.shares_memory(np.asarray(matrix), matrix.values)
        with pytest.raises(ValueError, match="copy"):
            matrix.__array__(dtype=np.float32, copy=False)

    def test_equality_is_a_verdict_over_words_rule_and_numbers(self):
        matrix = tiny().pointwise_mutual_information

        assert matrix == tiny().pointwise_mutual_information
        assert matrix != tiny(shift=0.5).pointwise_mutual_information
        assert (
            matrix
            != tiny(context_distribution_smoothing=0.75).pointwise_mutual_information
        )
        assert (matrix == 3) is False
        assert hash(matrix) == hash(tiny().pointwise_mutual_information)

    def test_repr_names_the_size_and_the_rule(self):
        assert repr(tiny().pointwise_mutual_information) == (
            "PointwiseMutualInformationMatrix(n_words=5, shift=0.0, "
            "context_distribution_smoothing=1.0)"
        )
