"""Spec for GloVe -- weighted least squares on the logarithm of co-occurrence counts.

Three oracles carry this module. The gradient of one pair's term is pinned
against a finite difference of that term, which is the definition of a
derivative and not any formula. The objective is pinned on ``["a b a"]``,
whose co-occurrence matrix is ``[[0, 2], [2, 0]]`` by hand and whose value at
zero parameters is ``2 f(2) (log 2)^2``. And the recorded loss is pinned
against a recomputation of ``J`` from the fitted parameters on a walk that
stands still, since the two are the same number only when nothing moves
between one pair and the next.

The two-topic corpus is twelve cooking sentences and twelve astronomy
sentences with disjoint content vocabularies and seven shared function words.
It is fitted at ``window=2`` rather than the paper's default of five, and the
reason is measured rather than chosen: on sentences of six to nine words a
window of five reaches every function word from every content word at a
weight of 0.2 to 0.5, so half the nonzero counts sit below one, their
logarithms are negative, and the biases do not absorb the negative mean in 25
epochs -- the negative residual lands in the dot products and the within-topic
cosine comes out *below* the across-topic one on some seeds. At a window of two
the counts are 0.5 or 1 per occurrence and mostly above one once repeated, and
the within-topic cosine beats the across-topic one on every seed tried.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    DivergenceError,
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import (
    TokenisedCorpus,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.glove import (
    GloVe,
    GloVeParameters,
    PairGradient,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.word2vec import (
    TrainingHistory,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (  # noqa: E501
    WhitespacePreTokenizer,
)

COOKING = [
    "the chef chops onion and garlic",
    "the chef stirs soup with butter and salt",
    "the chef stirs onion and garlic in the pan",
    "the chef bakes bread with flour in the oven",
    "the chef stirs salt and pepper into the broth",
    "the chef bakes bread and stirs the broth",
    "flour and butter and salt in the pan",
    "the chef chops onion for the soup and the broth",
    "garlic and pepper in the soup with bread",
    "the chef bakes flour and butter into bread in the oven",
    "onion and garlic and salt in the broth",
    "the chef stirs the soup in the pan with pepper",
]
ASTRONOMY = [
    "the astronomer watches the star and the planet",
    "the astronomer charts the orbit of the moon",
    "the comet and the meteor orbit the star",
    "the telescope watches the galaxy and the nebula",
    "the moon orbits the planet with gravity",
    "the astronomer watches the eclipse of the moon with the telescope",
    "the comet orbits the star and the astronomer charts the orbit",
    "the telescope watches the nebula and the galaxy",
    "the astronomer charts the meteor and the comet with the telescope",
    "the planet orbits the star and the moon orbits the planet",
    "the eclipse of the star and the eclipse of the moon",
    "gravity and the orbit of the galaxy and the nebula",
]
FUNCTION_WORDS = frozenset({"the", "and", "with", "in", "of", "into", "for"})
TWO_TOPICS = COOKING + ASTRONOMY

# ``a b a`` with a window of one: ``a`` sees ``b`` from both ends, ``b`` sees
# ``a`` on both sides, so X_ab = X_ba = 2 and the diagonal is empty.
THREE_WORDS = ["a b a"]
THREE_WORD_COUNTS = [[0.0, 2.0], [2.0, 0.0]]

FIXTURE_DIMENSION = 8
FIXTURE_WINDOW = 2


def fixture_model(random_seed: int | None = 0, **overrides: object) -> GloVe:
    """The two-topic fit at the configuration the module docstring explains."""
    settings: dict[str, object] = {
        "dimension": FIXTURE_DIMENSION,
        "window": FIXTURE_WINDOW,
        "random_seed": random_seed,
    }
    settings.update(overrides)
    return GloVe(**settings).fit(TWO_TOPICS)  # type: ignore[arg-type]


def zero_parameters(n_words: int, dimension: int) -> GloVeParameters:
    return GloVeParameters(
        np.zeros((n_words, dimension)),
        np.zeros((n_words, dimension)),
        np.zeros(n_words),
        np.zeros(n_words),
    )


def content_words(sentences: list[str], model: GloVe) -> list[str]:
    words = {word for sentence in sentences for word in sentence.split()}
    return sorted(
        word
        for word in words
        if word not in FUNCTION_WORDS and word in model.vocabulary
    )


def mean_cosine(model: GloVe, first: list[str], second: list[str]) -> float:
    return float(
        np.mean(
            [
                model.similarity(one, two)
                for one, two in itertools.product(first, second)
                if one != two
            ]
        )
    )


@pytest.fixture(scope="module")
def model() -> GloVe:
    return fixture_model()


@pytest.fixture(scope="module")
def standstill() -> GloVe:
    """One epoch at a learning rate so small that nothing moves."""
    return fixture_model(learning_rate=1e-12, epochs=1)


class TestConstruction:
    def test_the_defaults_are_the_papers(self) -> None:
        model = GloVe()

        assert model.dimension == 50
        assert model.window == 5
        assert model.weighting is ContextWeighting.HARMONIC
        assert model.maximum_count == 100.0
        assert model.weighting_exponent == 0.75
        assert model.epochs == 25
        assert model.learning_rate == 0.05
        assert model.random_seed is None
        assert model.minimum_count == 1

    @pytest.mark.parametrize(
        "keyword",
        [
            {"dimension": 0},
            {"window": 0},
            {"maximum_count": 0.0},
            {"maximum_count": -1.0},
            {"weighting_exponent": 0.0},
            {"weighting_exponent": 1.5},
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"learning_rate": -0.1},
            {"minimum_count": 0},
        ],
    )
    def test_an_out_of_range_field_is_refused(self, keyword: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            GloVe(**keyword)  # type: ignore[arg-type]

    def test_an_exponent_of_one_is_allowed(self) -> None:
        assert GloVe(weighting_exponent=1.0).weighting_exponent == 1.0

    def test_an_unknown_keyword_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            GloVe(x_max=10.0)  # type: ignore[call-arg]


class TestWeightOfCount:
    """The paper's ``f``, at the points where its value is known by hand."""

    def test_at_the_maximum_it_is_exactly_one(self) -> None:
        assert GloVe(maximum_count=100.0).weight_of_count(100.0) == 1.0

    def test_at_half_the_maximum_it_is_a_half_to_the_exponent(self) -> None:
        model = GloVe(maximum_count=100.0, weighting_exponent=0.75)

        assert model.weight_of_count(50.0) == pytest.approx(0.5**0.75)
        assert model.weight_of_count(50.0) == pytest.approx(0.5946035575013605)

    def test_above_the_maximum_it_stays_one(self) -> None:
        model = GloVe(maximum_count=100.0)

        assert model.weight_of_count(150.0) == 1.0
        assert model.weight_of_count(1e9) == 1.0

    def test_a_pair_seen_once_at_the_defaults_weighs_ten_to_the_minus_three_halves(
        self,
    ) -> None:
        assert GloVe().weight_of_count(1.0) == pytest.approx(10.0**-1.5)

    def test_an_exponent_of_a_half_is_a_square_root(self) -> None:
        assert GloVe(maximum_count=100.0, weighting_exponent=0.5).weight_of_count(
            25.0
        ) == pytest.approx(0.5)

    def test_an_exponent_of_one_is_proportional(self) -> None:
        assert GloVe(maximum_count=100.0, weighting_exponent=1.0).weight_of_count(
            30.0
        ) == pytest.approx(0.3)

    def test_a_count_of_zero_weighs_nothing(self) -> None:
        assert GloVe().weight_of_count(0.0) == 0.0

    def test_it_never_decreases_with_the_count(self) -> None:
        model = GloVe(maximum_count=10.0)
        weights = [model.weight_of_count(count / 2.0) for count in range(0, 40)]

        assert all(
            later >= earlier
            for earlier, later in zip(weights, weights[1:], strict=False)
        )

    @pytest.mark.parametrize("count", [-1.0, float("nan"), float("inf")])
    def test_a_count_that_is_not_a_count_is_refused(self, count: float) -> None:
        with pytest.raises(InvalidValuesError):
            GloVe().weight_of_count(count)

    def test_it_answers_a_python_float(self) -> None:
        assert type(GloVe().weight_of_count(3.0)) is float


class TestPairGradient:
    """The hand-written gradient of ``f(X_ij) e^2``, and the finite difference."""

    def test_a_hand_worked_pair(self) -> None:
        """``e = 0.5 - 1.0 + 0.1 + 0.2 - 1 = -1.2`` at a count of ``e`` and weight 1."""
        gradient = GloVe(maximum_count=1.0).pair_gradient(
            math.e, np.array([0.5, -0.5]), np.array([1.0, 2.0]), 0.1, 0.2
        )

        assert gradient.loss == pytest.approx(1.44)
        assert np.allclose(gradient.word_gradient, [-2.4, -4.8])
        assert np.allclose(gradient.context_gradient, [-1.2, 1.2])
        assert gradient.bias_gradient == pytest.approx(-2.4)
        assert gradient.context_bias_gradient == pytest.approx(-2.4)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_every_slope_matches_a_finite_difference(self, seed: int) -> None:
        """The oracle is the definition of a derivative, not any formula."""
        generator = np.random.default_rng(seed)
        model = GloVe(maximum_count=10.0)
        count = float(generator.uniform(0.2, 20.0))
        word = generator.normal(size=3)
        context = generator.normal(size=3)
        bias = float(generator.normal())
        context_bias = float(generator.normal())
        claimed = model.pair_gradient(count, word, context, bias, context_bias)
        step = 1e-6

        def loss(
            word_values: np.ndarray,
            context_values: np.ndarray,
            bias_value: float,
            context_bias_value: float,
        ) -> float:
            return model.pair_gradient(
                count, word_values, context_values, bias_value, context_bias_value
            ).loss

        for position in range(3):
            up, down = word.copy(), word.copy()
            up[position] += step
            down[position] -= step
            measured = (
                loss(up, context, bias, context_bias)
                - loss(down, context, bias, context_bias)
            ) / (2.0 * step)
            assert claimed.word_gradient[position] == pytest.approx(measured, abs=1e-7)

            up, down = context.copy(), context.copy()
            up[position] += step
            down[position] -= step
            measured = (
                loss(word, up, bias, context_bias)
                - loss(word, down, bias, context_bias)
            ) / (2.0 * step)
            assert claimed.context_gradient[position] == pytest.approx(
                measured, abs=1e-7
            )

        measured = (
            loss(word, context, bias + step, context_bias)
            - loss(word, context, bias - step, context_bias)
        ) / (2.0 * step)
        assert claimed.bias_gradient == pytest.approx(measured, abs=1e-7)

        measured = (
            loss(word, context, bias, context_bias + step)
            - loss(word, context, bias, context_bias - step)
        ) / (2.0 * step)
        assert claimed.context_bias_gradient == pytest.approx(measured, abs=1e-7)

    def test_the_weight_scales_the_whole_term(self) -> None:
        unweighted = GloVe(maximum_count=1.0)
        weighted = GloVe(maximum_count=100.0)
        word, context = np.array([0.3, -0.2]), np.array([0.1, 0.4])

        plain = unweighted.pair_gradient(4.0, word, context, 0.05, -0.05)
        scaled = weighted.pair_gradient(4.0, word, context, 0.05, -0.05)
        weight = weighted.weight_of_count(4.0)

        assert scaled.loss == pytest.approx(weight * plain.loss)
        assert np.allclose(scaled.word_gradient, weight * plain.word_gradient)
        assert scaled.bias_gradient == pytest.approx(weight * plain.bias_gradient)

    def test_a_perfect_fit_has_no_gradient(self) -> None:
        """``w . c + b + b~ = log 4`` exactly, so the error is zero."""
        gradient = GloVe().pair_gradient(
            4.0, np.array([1.0, 0.0]), np.array([0.0, 1.0]), math.log(4.0), 0.0
        )

        assert gradient.loss == 0.0
        assert np.array_equal(gradient.word_gradient, [0.0, 0.0])
        assert gradient.bias_gradient == 0.0

    def test_the_gradients_are_frozen(self) -> None:
        gradient = GloVe().pair_gradient(
            2.0, np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0.0, 0.0
        )

        assert not gradient.word_gradient.flags.writeable
        assert not gradient.context_gradient.flags.writeable

    def test_vectors_of_different_widths_are_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            GloVe().pair_gradient(2.0, np.zeros(3), np.zeros(2), 0.0, 0.0)

    @pytest.mark.parametrize("count", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_pair_that_did_not_co_occur_is_refused(self, count: float) -> None:
        with pytest.raises(InvalidValuesError):
            GloVe().pair_gradient(count, np.zeros(2), np.zeros(2), 0.0, 0.0)

    def test_a_non_finite_vector_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            GloVe().pair_gradient(
                2.0, np.array([0.0, float("nan")]), np.zeros(2), 0.0, 0.0
            )

    def test_a_non_finite_bias_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            GloVe().pair_gradient(2.0, np.zeros(2), np.zeros(2), float("inf"), 0.0)

    def test_an_overflowing_error_is_a_divergence(self) -> None:
        """Finite inputs whose product is not: the fit has run away."""
        with pytest.raises(DivergenceError):
            GloVe().pair_gradient(2.0, np.array([1e200]), np.array([1e200]), 0.0, 0.0)


class TestPairGradientValueObject:
    @pytest.mark.parametrize("loss", [-0.1, float("nan"), float("inf")])
    def test_a_loss_that_is_not_a_loss_is_refused(self, loss: float) -> None:
        with pytest.raises(InvalidValuesError):
            PairGradient(loss, np.zeros(2), np.zeros(2), 0.0, 0.0)

    def test_mismatched_widths_are_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            PairGradient(1.0, np.zeros(2), np.zeros(3), 0.0, 0.0)

    def test_a_non_finite_bias_gradient_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            PairGradient(1.0, np.zeros(2), np.zeros(2), float("nan"), 0.0)

    def test_it_copies_what_it_is_handed(self) -> None:
        source = np.array([1.0, 2.0])
        gradient = PairGradient(1.0, source, source, 0.0, 0.0)
        source[0] = 99.0

        assert gradient.word_gradient[0] == 1.0

    def test_repr_names_the_loss(self) -> None:
        assert "loss=1.5000" in repr(PairGradient(1.5, np.zeros(2), np.zeros(2), 0, 0))


class TestGloVeParameters:
    def test_it_knows_its_shape(self) -> None:
        parameters = zero_parameters(5, 3)

        assert parameters.n_words == 5
        assert parameters.dimension == 3
        assert parameters.word_vectors.shape == (5, 3)
        assert parameters.biases.shape == (5,)

    def test_the_combined_vectors_are_the_sum(self) -> None:
        word = np.array([[1.0, 2.0], [3.0, 4.0]])
        context = np.array([[10.0, 20.0], [30.0, 40.0]])
        parameters = GloVeParameters(word, context, np.zeros(2), np.zeros(2))

        assert np.array_equal(parameters.combined_vectors, [[11.0, 22.0], [33.0, 44.0]])

    def test_every_table_is_frozen(self) -> None:
        parameters = zero_parameters(2, 2)

        for table in (
            parameters.word_vectors,
            parameters.context_vectors,
            parameters.biases,
            parameters.context_biases,
            parameters.combined_vectors,
        ):
            assert not table.flags.writeable

    def test_it_copies_what_it_is_handed(self) -> None:
        word = np.zeros((2, 2))
        parameters = GloVeParameters(word, np.zeros((2, 2)), np.zeros(2), np.zeros(2))
        word[0, 0] = 5.0

        assert parameters.word_vectors[0, 0] == 0.0

    def test_vector_tables_of_different_shapes_are_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            GloVeParameters(
                np.zeros((2, 3)), np.zeros((2, 2)), np.zeros(2), np.zeros(2)
            )

    def test_biases_of_the_wrong_length_are_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            GloVeParameters(
                np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(3), np.zeros(2)
            )

    def test_a_one_dimensional_table_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            GloVeParameters(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))

    def test_a_non_finite_entry_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            GloVeParameters(
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.array([0.0, float("nan")]),
                np.zeros(2),
            )

    def test_a_non_numeric_table_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            GloVeParameters(
                [["a", "b"]],  # type: ignore[arg-type]
                np.zeros((1, 2)),
                np.zeros(1),
                np.zeros(1),
            )

    def test_equality_is_by_value(self) -> None:
        assert zero_parameters(2, 2) == zero_parameters(2, 2)
        assert hash(zero_parameters(2, 2)) == hash(zero_parameters(2, 2))
        assert zero_parameters(2, 2) != GloVeParameters(
            np.ones((2, 2)), np.zeros((2, 2)), np.zeros(2), np.zeros(2)
        )

    def test_comparison_with_another_type_is_not_implemented(self) -> None:
        assert zero_parameters(2, 2).__eq__(3) is NotImplemented

    def test_repr_names_the_shape(self) -> None:
        assert repr(zero_parameters(4, 2)) == "GloVeParameters(n_words=4, dimension=2)"


class TestFit:
    def test_fit_returns_self(self) -> None:
        model = GloVe(dimension=2, epochs=1)

        assert model.fit(THREE_WORDS) is model

    def test_the_vocabulary_is_commonest_first(self, model: GloVe) -> None:
        assert list(model.vocabulary)[0] == "the"
        assert "chef" in model.vocabulary

    def test_the_matrix_is_the_one_the_foundation_builds(self, model: GloVe) -> None:
        tokenised = TokenisedCorpus.from_texts(TWO_TOPICS, WhitespacePreTokenizer())
        expected = CooccurrenceMatrix.from_id_sequences(
            model.vocabulary,
            tokenised.id_sequences(model.vocabulary),
            FIXTURE_WINDOW,
            ContextWeighting.HARMONIC,
        )

        assert model.cooccurrence == expected

    def test_the_embeddings_are_the_sum_of_the_two_tables(self, model: GloVe) -> None:
        assert np.array_equal(
            model.embeddings.table, model.word_vectors + model.context_vectors
        )
        assert np.array_equal(model.embeddings.table, model.parameters.combined_vectors)

    def test_the_embeddings_are_a_word_embeddings(self, model: GloVe) -> None:
        assert isinstance(model.embeddings, WordEmbeddings)
        assert model.embeddings.vocabulary == model.vocabulary
        assert model.embeddings.dimension == FIXTURE_DIMENSION
        assert model.embeddings.n_words == model.cooccurrence.n_words

    def test_every_learned_table_is_shaped_and_frozen(self, model: GloVe) -> None:
        n_words = model.vocabulary.n_tokens

        assert model.word_vectors.shape == (n_words, FIXTURE_DIMENSION)
        assert model.context_vectors.shape == (n_words, FIXTURE_DIMENSION)
        assert model.biases.shape == (n_words,)
        assert model.context_biases.shape == (n_words,)
        for table in (
            model.word_vectors,
            model.context_vectors,
            model.biases,
            model.context_biases,
        ):
            assert not table.flags.writeable

    def test_the_two_tables_are_not_the_same_table(self, model: GloVe) -> None:
        assert not np.array_equal(model.word_vectors, model.context_vectors)

    def test_every_parameter_starts_within_a_half_over_the_dimension(
        self, standstill: GloVe
    ) -> None:
        """At a rate of 1e-12 the fit is the initialisation, to 1e-12."""
        bound = 0.5 / FIXTURE_DIMENSION + 1e-9

        for table in (
            standstill.word_vectors,
            standstill.context_vectors,
            standstill.biases,
            standstill.context_biases,
        ):
            assert float(np.abs(table).max()) < bound

    def test_the_weighting_reaches_the_matrix(self) -> None:
        model = fixture_model(weighting=ContextWeighting.UNIFORM, epochs=1)

        assert model.cooccurrence.weighting is ContextWeighting.UNIFORM
        assert np.array_equal(
            model.cooccurrence.counts, np.round(model.cooccurrence.counts)
        )

    def test_the_window_reaches_the_matrix(self, model: GloVe) -> None:
        assert model.cooccurrence.window == FIXTURE_WINDOW

    def test_the_minimum_count_drops_rare_words(self) -> None:
        model = fixture_model(minimum_count=3, epochs=1)

        assert "oven" not in model.vocabulary
        assert "chef" in model.vocabulary

    def test_the_conveniences_read_the_embeddings(self, model: GloVe) -> None:
        assert model.similarity("chef", "soup") == model.embeddings.similarity(
            "chef", "soup"
        )
        assert model.vector_of("chef").dimension == FIXTURE_DIMENSION
        assert model.most_similar("chef", 3).n_words == 3

    def test_a_vocabulary_of_one_word_is_refused(self) -> None:
        with pytest.raises(TooFewValuesError):
            GloVe().fit(["a a a"])

    def test_two_words_that_never_share_a_window_are_refused(self) -> None:
        with pytest.raises(TooFewValuesError):
            GloVe().fit(["a", "b"])

    def test_a_minimum_count_nothing_reaches_is_refused(self) -> None:
        with pytest.raises(TooFewValuesError):
            GloVe(minimum_count=100).fit(TWO_TOPICS)

    def test_a_single_string_corpus_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            GloVe().fit("the chef stirs the soup")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            GloVe().fit(["   ", ""])

    def test_a_learning_rate_that_overflows_is_a_divergence(self) -> None:
        with pytest.raises(DivergenceError):
            fixture_model(learning_rate=1e200, dimension=4, epochs=2)


class TestHistory:
    def test_one_record_per_epoch_in_order(self, model: GloVe) -> None:
        assert isinstance(model.history, TrainingHistory)
        assert model.history.n_epochs == model.epochs
        assert [record.epoch for record in model.history] == list(
            range(1, model.epochs + 1)
        )

    def test_the_loss_fell(self, model: GloVe) -> None:
        assert model.history.fell

    def test_every_loss_is_finite_and_non_negative(self, model: GloVe) -> None:
        for mean_loss in model.history.mean_losses:
            assert math.isfinite(mean_loss)
            assert mean_loss >= 0.0

    def test_every_epoch_saw_every_nonzero_pair(self, model: GloVe) -> None:
        n_nonzero = int(np.count_nonzero(model.cooccurrence.counts))

        assert all(record.n_pairs == n_nonzero for record in model.history)

    def test_the_recorded_rate_is_the_base_rate(self, model: GloVe) -> None:
        assert all(
            record.learning_rate == model.learning_rate for record in model.history
        )


class TestObjective:
    """``J`` by hand, and the two routes to it."""

    def test_the_three_word_matrix_is_as_worked_by_hand(self) -> None:
        matrix = CooccurrenceMatrix.from_id_sequences(
            Vocabulary(["a", "b"]), [(0, 1, 0)], window=1
        )

        assert np.array_equal(matrix.counts, THREE_WORD_COUNTS)

    def test_at_zero_parameters_and_weight_one_it_is_twice_log_two_squared(
        self,
    ) -> None:
        matrix = CooccurrenceMatrix(
            Vocabulary(["a", "b"]),
            np.array(THREE_WORD_COUNTS),
            window=1,
            weighting=ContextWeighting.HARMONIC,
        )

        value = GloVe(maximum_count=1.0).objective_of(matrix, zero_parameters(2, 3))

        assert value == pytest.approx(2.0 * math.log(2.0) ** 2)
        assert value == pytest.approx(0.9609060278364028)

    def test_at_the_defaults_the_weight_is_a_fiftieth_to_the_three_quarters(
        self,
    ) -> None:
        matrix = CooccurrenceMatrix(
            Vocabulary(["a", "b"]),
            np.array(THREE_WORD_COUNTS),
            window=1,
            weighting=ContextWeighting.HARMONIC,
        )

        value = GloVe().objective_of(matrix, zero_parameters(2, 3))

        assert value == pytest.approx(2.0 * 0.02**0.75 * math.log(2.0) ** 2)
        assert value == pytest.approx(0.05110382585192048)

    def test_at_zero_parameters_on_the_fitted_matrix_it_is_the_weighted_log_squares(
        self, model: GloVe
    ) -> None:
        """A from-definition sum over the nonzero counts, in plain Python."""
        counts = model.cooccurrence.counts
        expected = sum(
            model.weight_of_count(float(count)) * math.log(float(count)) ** 2
            for count in counts[counts > 0]
        )
        parameters = zero_parameters(model.cooccurrence.n_words, 3)

        assert model.objective_of(model.cooccurrence, parameters) == pytest.approx(
            expected
        )

    def test_parameters_for_the_wrong_word_count_are_refused(
        self, model: GloVe
    ) -> None:
        with pytest.raises(ShapeMismatchError):
            model.objective_of(model.cooccurrence, zero_parameters(3, 2))

    def test_the_record_and_the_recomputation_agree_when_nothing_moves(
        self, standstill: GloVe
    ) -> None:
        """Two routes to one number: accumulated as the epoch ran, and at rest."""
        recorded_total = standstill.history[0].mean_loss * standstill.history[0].n_pairs

        assert standstill.objective_value() == pytest.approx(recorded_total, rel=1e-9)

    def test_after_a_fit_the_objective_is_below_the_first_record(
        self, model: GloVe
    ) -> None:
        """The last record is not comparable, since it was accumulated while the
        parameters moved; the first record is an upper bound the fit must beat."""
        first_total = model.history[0].mean_loss * model.history[0].n_pairs

        assert model.objective_value() < first_total

    def test_it_is_the_long_form_on_the_fitted_state(self, model: GloVe) -> None:
        assert model.objective_value() == model.objective_of(
            model.cooccurrence, model.parameters
        )

    def test_it_answers_a_non_negative_python_float(self, model: GloVe) -> None:
        assert type(model.objective_value()) is float
        assert model.objective_value() >= 0.0


class TestTwoTopics:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_within_topic_words_are_nearer_than_across(self, seed: int) -> None:
        model = fixture_model(random_seed=seed)
        cooking = content_words(COOKING, model)
        astronomy = content_words(ASTRONOMY, model)

        within = 0.5 * (
            mean_cosine(model, cooking, cooking)
            + mean_cosine(model, astronomy, astronomy)
        )
        across = mean_cosine(model, cooking, astronomy)

        assert within > across

    def test_the_content_vocabularies_really_are_disjoint(self) -> None:
        cooking = {word for sentence in COOKING for word in sentence.split()}
        astronomy = {word for sentence in ASTRONOMY for word in sentence.split()}

        assert cooking & astronomy == set(FUNCTION_WORDS) & cooking & astronomy
        assert (cooking & astronomy) <= FUNCTION_WORDS


class TestDeterminism:
    def test_one_seed_gives_one_answer(self, model: GloVe) -> None:
        again = fixture_model()

        assert again.embeddings == model.embeddings
        assert again.parameters == model.parameters
        assert again.history == model.history

    def test_two_seeds_give_two_answers(self, model: GloVe) -> None:
        other = fixture_model(random_seed=1)

        assert other.embeddings != model.embeddings
        assert other.embeddings.vocabulary == model.embeddings.vocabulary


class TestNotFitted:
    @pytest.mark.parametrize(
        "attribute",
        [
            "embeddings",
            "parameters",
            "word_vectors",
            "context_vectors",
            "biases",
            "context_biases",
            "history",
            "cooccurrence",
            "vocabulary",
        ],
    )
    def test_a_learned_attribute_raises_before_fit(self, attribute: str) -> None:
        with pytest.raises(NotFittedError):
            getattr(GloVe(), attribute)

    def test_the_objective_raises_before_fit(self) -> None:
        with pytest.raises(NotFittedError):
            GloVe().objective_value()

    def test_a_word_lookup_raises_before_fit(self) -> None:
        with pytest.raises(NotFittedError):
            GloVe().vector_of("chef")

    def test_the_hyperparameter_functions_need_no_fit(self) -> None:
        assert GloVe().weight_of_count(1.0) > 0.0
        assert (
            GloVe().pair_gradient(1.0, np.zeros(2), np.zeros(2), 0.0, 0.0).loss == 0.0
        )


class TestTheCap:
    def test_a_huge_maximum_never_binds(self) -> None:
        """Every weight is ``(x / x_max) ** alpha`` and strictly below one."""
        model = fixture_model(epochs=1, maximum_count=1e6, dimension=4)
        counts = model.cooccurrence.counts

        for count in counts[counts > 0]:
            weight = model.weight_of_count(float(count))
            assert weight == pytest.approx((float(count) / 1e6) ** 0.75)
            assert weight < 1.0

    def test_a_maximum_of_one_binds_on_every_repeated_pair(self, model: GloVe) -> None:
        capped = GloVe(maximum_count=1.0)
        counts = model.cooccurrence.counts

        assert all(
            capped.weight_of_count(float(count)) == 1.0
            for count in counts[counts >= 1.0]
        )
        assert any(count < 1.0 for count in counts[counts > 0])
