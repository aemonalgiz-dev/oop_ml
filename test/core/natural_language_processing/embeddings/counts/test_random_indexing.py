"""Spec for RandomIndexing and its IndexVectors.

The load-bearing claim is the identity in the module docstring: the context
vectors are the co-occurrence matrix times the index vectors, and equal a
direct sum over every position of every sentence.
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
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
)
from oop_ml.core.natural_language_processing.embeddings.counts.random_indexing import (
    IndexVectors,
    RandomIndexing,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import TokenisedCorpus
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from test.core.natural_language_processing.embeddings.counts.fixtures import (
    COOKING_WORDS,
    SAILING_WORDS,
    TINY_PMI_CORPUS,
    TWO_TOPIC_CORPUS,
)

BOTH_WEIGHTINGS = list(ContextWeighting)


def summed_over_positions(
    corpus: list[str],
    index_vectors: IndexVectors,
    window: int,
    weighting: ContextWeighting,
) -> np.ndarray:
    """The definition: walk every sentence and add each neighbour's index vector."""
    sentences = TokenisedCorpus.from_texts(corpus, WhitespacePreTokenizer()).sentences
    table = np.zeros((index_vectors.n_words, index_vectors.dimension))
    for sentence in sentences:
        for position, word in enumerate(sentence):
            for other_position, neighbour in enumerate(sentence):
                distance = abs(other_position - position)
                if distance == 0 or distance > window:
                    continue
                weight = (
                    1.0 / distance if weighting is ContextWeighting.HARMONIC else 1.0
                )
                table[index_vectors.vocabulary.id_of(word)] += (
                    weight * index_vectors.vector_of(neighbour).values
                )
    return table


def mean_similarity(
    model: RandomIndexing, first: list[str], second: list[str]
) -> float:
    pairs = [
        model.similarity(one, other)
        for one in first
        for other in second
        if one != other
    ]
    return float(np.mean(pairs))


class TestTheIdentity:
    @pytest.mark.parametrize("weighting", BOTH_WEIGHTINGS)
    @pytest.mark.parametrize("window", [1, 2, 5])
    def test_the_context_vectors_equal_a_direct_sum_over_positions(
        self, window: int, weighting: ContextWeighting
    ):
        model = RandomIndexing(
            dimension=8, n_nonzero=4, window=window, weighting=weighting, random_seed=0
        ).fit(TINY_PMI_CORPUS)

        assert np.allclose(
            model.embeddings.table,
            summed_over_positions(
                TINY_PMI_CORPUS, model.index_vectors, window, weighting
            ),
        )

    def test_the_identity_holds_on_the_two_topic_corpus_too(self):
        model = RandomIndexing(dimension=16, random_seed=0).fit(TWO_TOPIC_CORPUS)

        assert np.allclose(
            model.embeddings.table,
            summed_over_positions(
                TWO_TOPIC_CORPUS, model.index_vectors, 5, ContextWeighting.UNIFORM
            ),
        )

    def test_the_context_vectors_are_the_cooccurrence_rows_projected(self):
        model = RandomIndexing(dimension=8, window=2, random_seed=0).fit(
            TINY_PMI_CORPUS
        )

        assert np.allclose(
            model.embeddings.table,
            model.cooccurrence.counts @ model.index_vectors.table,
        )

    def test_a_word_alone_in_its_sentence_gets_the_zero_vector(self):
        model = RandomIndexing(dimension=8, random_seed=0).fit(
            ["alone", *TINY_PMI_CORPUS]
        )

        assert not model.vector_of("alone").values.any()
        with pytest.raises(UndefinedMetricError):
            model.similarity("alone", "cat")

    def test_two_words_with_identical_contexts_get_identical_context_vectors(self):
        model = RandomIndexing(dimension=8, random_seed=3).fit(
            ["the cat sat", "the dog sat"]
        )

        assert np.array_equal(
            model.vector_of("cat").values, model.vector_of("dog").values
        )
        assert not np.array_equal(
            model.index_vectors.vector_of("cat").values,
            model.index_vectors.vector_of("dog").values,
        )


class TestIndexVectors:
    def test_each_index_vector_has_n_nonzero_entries_half_of_each_sign(self):
        table = (
            RandomIndexing(dimension=8, n_nonzero=4, random_seed=0)
            .fit(TINY_PMI_CORPUS)
            .index_vectors.table
        )

        assert table.shape == (5, 8)
        assert np.all(np.count_nonzero(table == 1.0, axis=1) == 2)
        assert np.all(np.count_nonzero(table == -1.0, axis=1) == 2)
        assert np.all(np.count_nonzero(table == 0.0, axis=1) == 4)

    def test_the_index_vectors_are_not_the_embedding(self):
        model = RandomIndexing(dimension=8, random_seed=0).fit(TINY_PMI_CORPUS)

        assert not np.array_equal(model.index_vectors.table, model.embeddings.table)
        assert model.index_vectors.n_nonzero == 4
        assert model.index_vectors.dimension == 8
        assert model.index_vectors.n_words == 5

    def test_near_orthogonality_improves_with_dimension(self):
        """Two vectors of four non-zeros share about ``16 / d`` positions, so the
        mean absolute cosine is about ``4 / d``: measured 0.0744 at 50, 0.0077
        at 500, with 71.8% and 96.9% of pairs exactly orthogonal."""
        vocabulary = Vocabulary([f"word{position}" for position in range(200)])
        measured: dict[int, tuple[float, float]] = {}
        for dimension in (50, 500):
            vectors = IndexVectors.drawn(
                vocabulary, dimension, 4, np.random.default_rng(0)
            )
            normalised = vectors.table / np.linalg.norm(vectors.table, axis=1)[:, None]
            cosines = (normalised @ normalised.T)[np.triu_indices(200, k=1)]
            measured[dimension] = (
                float(np.abs(cosines).mean()),
                float((cosines == 0.0).mean()),
            )

        assert measured[50][0] == pytest.approx(0.0744, abs=1e-3)
        assert measured[500][0] == pytest.approx(0.0077, abs=1e-3)
        assert measured[500][0] < measured[50][0]
        assert measured[50][1] == pytest.approx(0.718, abs=1e-2)
        assert measured[500][1] == pytest.approx(0.969, abs=1e-2)

    @pytest.mark.parametrize("n_nonzero", [3, 1, 0, 10])
    def test_drawn_refuses_odd_too_few_or_too_many_nonzeros(self, n_nonzero: int):
        with pytest.raises(InvalidValuesError):
            IndexVectors.drawn(
                Vocabulary(["a"]), 8, n_nonzero, np.random.default_rng(0)
            )

    def test_drawn_fills_every_position_when_n_nonzero_is_the_dimension(self):
        vectors = IndexVectors.drawn(
            Vocabulary(["a", "b"]), 4, 4, np.random.default_rng(0)
        )

        assert np.all(vectors.table != 0.0)

    def test_the_constructor_refuses_an_entry_outside_the_ternary_set(self):
        with pytest.raises(InvalidValuesError):
            IndexVectors(Vocabulary(["a"]), np.array([[1.0, -1.0, 0.5, 0.0]]), 2)

    def test_the_constructor_refuses_an_unbalanced_row(self):
        with pytest.raises(InvalidValuesError):
            IndexVectors(Vocabulary(["a"]), np.array([[1.0, 1.0, 0.0, 0.0]]), 2)

    def test_the_constructor_refuses_a_row_with_the_wrong_count(self):
        with pytest.raises(InvalidValuesError):
            IndexVectors(Vocabulary(["a"]), np.array([[1.0, -1.0, 1.0, -1.0]]), 2)

    def test_the_constructor_refuses_rows_not_matching_the_vocabulary(self):
        with pytest.raises(ShapeMismatchError):
            IndexVectors(Vocabulary(["a", "b"]), np.array([[1.0, -1.0, 0.0, 0.0]]), 2)

    def test_the_constructor_refuses_an_odd_n_nonzero(self):
        with pytest.raises(InvalidValuesError):
            IndexVectors(Vocabulary(["a"]), np.array([[1.0, -1.0, 0.0, 0.0]]), 3)

    def test_the_constructor_refuses_a_one_dimensional_table(self):
        with pytest.raises(InvalidValuesError):
            IndexVectors(Vocabulary(["a"]), np.array([1.0, -1.0]), 2)

    def test_non_numeric_values_are_refused(self):
        with pytest.raises(InvalidValuesError):
            IndexVectors(Vocabulary(["a"]), np.array([["text"]]), 2)  # type: ignore[arg-type]

    def test_vector_of_binds_the_row_to_the_word(self):
        vectors = IndexVectors(
            Vocabulary(["a", "b"]), np.array([[1.0, -1.0], [-1.0, 1.0]]), 2
        )

        assert vectors.vector_of("b").word == "b"
        assert np.array_equal(vectors.vector_of("b").values, [-1.0, 1.0])
        with pytest.raises(UnknownTokenError):
            vectors.vector_of("c")

    def test_iterates_in_id_order(self):
        vectors = IndexVectors(
            Vocabulary(["a", "b"]), np.array([[1.0, -1.0], [-1.0, 1.0]]), 2
        )

        assert [vector.word for vector in vectors] == ["a", "b"]
        assert len(vectors) == 2

    def test_the_table_is_copied_and_frozen(self):
        source = np.array([[1.0, -1.0]])
        vectors = IndexVectors(Vocabulary(["a"]), source, 2)
        source[0, 0] = -1.0

        assert vectors.table[0, 0] == 1.0
        assert not vectors.table.flags.writeable

    def test_the_array_protocol_copies_when_asked_and_shares_when_allowed(self):
        vectors = IndexVectors(Vocabulary(["a"]), np.array([[1.0, -1.0]]), 2)

        assert not np.shares_memory(np.array(vectors), vectors.table)
        assert np.shares_memory(np.asarray(vectors), vectors.table)
        with pytest.raises(ValueError, match="copy"):
            vectors.__array__(dtype=np.float32, copy=False)

    def test_equality_is_a_verdict(self):
        vectors = IndexVectors(Vocabulary(["a"]), np.array([[1.0, -1.0]]), 2)

        assert vectors == IndexVectors(Vocabulary(["a"]), np.array([[1.0, -1.0]]), 2)
        assert vectors != IndexVectors(Vocabulary(["a"]), np.array([[-1.0, 1.0]]), 2)
        assert vectors != IndexVectors(Vocabulary(["b"]), np.array([[1.0, -1.0]]), 2)
        assert (vectors == "not index vectors") is False
        assert hash(vectors) == hash(
            IndexVectors(Vocabulary(["a"]), np.array([[1.0, -1.0]]), 2)
        )

    def test_repr_names_the_shape_and_sparsity(self):
        assert repr(
            IndexVectors(Vocabulary(["a"]), np.array([[1.0, -1.0, 0.0]]), 2)
        ) == ("IndexVectors(n_words=1, dimension=3, n_nonzero=2)")


class TestTwoTopics:
    def test_two_cooking_words_are_nearer_than_a_cooking_and_a_sailing_word(self):
        model = RandomIndexing(dimension=16, random_seed=0).fit(TWO_TOPIC_CORPUS)

        assert model.similarity("flour", "sugar") == pytest.approx(0.627, abs=1e-3)
        assert model.similarity("flour", "anchor") == pytest.approx(-0.2437, abs=1e-3)

    @pytest.mark.parametrize(("dimension", "random_seed"), [(16, 0), (16, 1), (50, 0)])
    def test_within_topic_similarity_exceeds_across_on_average(
        self, dimension: int, random_seed: int
    ):
        model = RandomIndexing(dimension=dimension, random_seed=random_seed).fit(
            TWO_TOPIC_CORPUS
        )
        within = mean_similarity(model, COOKING_WORDS, COOKING_WORDS)
        across = mean_similarity(model, COOKING_WORDS, SAILING_WORDS)

        assert within > across + 0.3

    def test_the_measured_separation_at_sixteen_dimensions_seed_zero(self):
        """The two topics differ here where they did not for the decompositions,
        because the random index vectors break the fixture's symmetry."""
        model = RandomIndexing(dimension=16, random_seed=0).fit(TWO_TOPIC_CORPUS)

        assert mean_similarity(model, COOKING_WORDS, COOKING_WORDS) == pytest.approx(
            0.6543, abs=1e-3
        )
        assert mean_similarity(model, SAILING_WORDS, SAILING_WORDS) == pytest.approx(
            0.6254, abs=1e-3
        )
        assert mean_similarity(model, COOKING_WORDS, SAILING_WORDS) == pytest.approx(
            0.1247, abs=1e-3
        )

    def test_the_vectors_are_one_per_word_of_the_asked_dimension(self):
        embeddings = (
            RandomIndexing(dimension=16, random_seed=0).fit(TWO_TOPIC_CORPUS).embeddings
        )

        assert embeddings.n_words == 23
        assert embeddings.dimension == 16


class TestDeterminism:
    def test_the_same_seed_gives_the_same_fit(self):
        first = RandomIndexing(dimension=8, random_seed=7).fit(TINY_PMI_CORPUS)
        second = RandomIndexing(dimension=8, random_seed=7).fit(TINY_PMI_CORPUS)

        assert first.embeddings == second.embeddings
        assert first.index_vectors == second.index_vectors

    def test_different_seeds_give_different_index_vectors(self):
        first = RandomIndexing(dimension=8, random_seed=7).fit(TINY_PMI_CORPUS)
        second = RandomIndexing(dimension=8, random_seed=8).fit(TINY_PMI_CORPUS)

        assert first.index_vectors != second.index_vectors

    def test_an_unseeded_fit_still_fits(self):
        assert RandomIndexing(dimension=8).fit(TINY_PMI_CORPUS).is_fitted

    def test_the_cooccurrence_does_not_depend_on_the_seed(self):
        first = RandomIndexing(dimension=8, random_seed=7).fit(TINY_PMI_CORPUS)
        second = RandomIndexing(dimension=8, random_seed=8).fit(TINY_PMI_CORPUS)

        assert first.cooccurrence == second.cooccurrence


class TestRefusals:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"dimension": 4, "n_nonzero": 6},
            {"n_nonzero": 3},
            {"n_nonzero": 1},
            {"n_nonzero": 0},
            {"dimension": 0},
            {"window": 0},
            {"minimum_count": 0},
            {"seed": 1},
        ],
    )
    def test_bad_construction_is_refused_by_pydantic(self, keywords: dict[str, object]):
        with pytest.raises(ValidationError):
            RandomIndexing(**keywords)  # type: ignore[arg-type]

    def test_n_nonzero_equal_to_the_dimension_is_accepted(self):
        model = RandomIndexing(dimension=4, n_nonzero=4, random_seed=0).fit(
            TINY_PMI_CORPUS
        )

        assert np.all(model.index_vectors.table != 0.0)

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            RandomIndexing(dimension=4).fit("the cat sat")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            RandomIndexing(dimension=4).fit(["  ", ""])

    def test_no_word_reaching_the_minimum_count_is_refused(self):
        with pytest.raises(TooFewValuesError):
            RandomIndexing(dimension=4, minimum_count=10).fit(TINY_PMI_CORPUS)

    def test_minimum_count_drops_the_rare_words(self):
        model = RandomIndexing(dimension=4, minimum_count=2, random_seed=0).fit(
            TINY_PMI_CORPUS
        )

        assert list(model.vocabulary) == ["the", "cat", "sat"]

    def test_before_fit_everything_learned_raises_not_fitted(self):
        model = RandomIndexing(dimension=4)

        for attribute in ("embeddings", "index_vectors", "cooccurrence", "vocabulary"):
            with pytest.raises(NotFittedError):
                getattr(model, attribute)

    def test_fit_returns_self(self):
        model = RandomIndexing(dimension=4)

        assert model.fit(TINY_PMI_CORPUS) is model

    def test_a_failed_fit_leaves_the_model_unfitted(self):
        model = RandomIndexing(dimension=4, minimum_count=10)

        with pytest.raises(TooFewValuesError):
            model.fit(TINY_PMI_CORPUS)
        assert not model.is_fitted
