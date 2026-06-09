"""Spec for ParagraphVectors -- a vector per document, learned as a word always in the window.

The corpus is thirty eight-word documents drawn by a seeded generator from two
disjoint ten-word lists, fifteen about cooking and fifteen about sailing, so
the one thing a document vector can learn is which list its words came from.
Every gradient the module writes by hand is pinned against a central
difference of the loss it reports, and every figure quoted in the module
docstring is asserted here, as an inequality where the claim is one.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

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
from oop_ml.core.natural_language_processing.embeddings.documents.embedder import (
    DocumentVectors,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.objectives import (
    negative_sampling_gradients,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.paragraph_vectors import (
    ParagraphArchitecture,
    ParagraphVectors,
    averaged_hidden,
    distributed_bag_of_words_gradients,
    distributed_memory_gradients,
    shared_input_step,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.sampling import (
    UnigramSampler,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    cosine_similarity,
)

#: The step a central difference takes on either side.
NUDGE = 1e-6

DIMENSION = 8
WINDOW = 3
EPOCHS = 30
LEARNING_RATE = 0.1
INFERENCE_EPOCHS = 200
SEED = 0
N_PER_TOPIC = 15
WORDS_PER_DOCUMENT = 8
#: The bag of words with ``train_words`` sees 5.5x the pairs; ten epochs suffice.
TRAIN_WORDS_EPOCHS = 10

COOKING_WORDS = (
    "simmer",
    "onion",
    "garlic",
    "butter",
    "saucepan",
    "whisk",
    "flour",
    "season",
    "ladle",
    "broth",
)
SAILING_WORDS = (
    "mainsail",
    "rudder",
    "harbour",
    "anchor",
    "keel",
    "tide",
    "starboard",
    "rigging",
    "mooring",
    "gust",
)


def topic_documents(words: Sequence[str], generator: np.random.Generator) -> list[str]:
    """Fifteen eight-word documents drawn with replacement from one list."""
    return [
        " ".join(str(word) for word in generator.choice(words, WORDS_PER_DOCUMENT))
        for _ in range(N_PER_TOPIC)
    ]


_corpus_generator = np.random.default_rng(7)
COOKING_DOCUMENTS = topic_documents(COOKING_WORDS, _corpus_generator)
SAILING_DOCUMENTS = topic_documents(SAILING_WORDS, _corpus_generator)
TWO_TOPICS = [*COOKING_DOCUMENTS, *SAILING_DOCUMENTS]
COOKING_POSITIONS = tuple(range(N_PER_TOPIC))
SAILING_POSITIONS = tuple(range(N_PER_TOPIC, 2 * N_PER_TOPIC))
N_WORDS = len(COOKING_WORDS) + len(SAILING_WORDS)

HELD_OUT_COOKING = "butter garlic onion simmer broth whisk flour ladle"
HELD_OUT_SAILING = "anchor tide rudder harbour keel gust mooring mainsail"

BOTH_ARCHITECTURES = (
    ParagraphArchitecture.DISTRIBUTED_MEMORY,
    ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS,
)


def model(
    architecture: ParagraphArchitecture,
    train_words: bool = False,
    epochs: int = EPOCHS,
    inference_epochs: int = INFERENCE_EPOCHS,
    minimum_count: int = 1,
) -> ParagraphVectors:
    return ParagraphVectors(
        dimension=DIMENSION,
        window=WINDOW,
        epochs=epochs,
        learning_rate=LEARNING_RATE,
        inference_epochs=inference_epochs,
        random_seed=SEED,
        architecture=architecture,
        train_words=train_words,
        minimum_count=minimum_count,
    )


@functools.cache
def fitted_under(
    architecture: ParagraphArchitecture,
    train_words: bool = False,
    epochs: int = EPOCHS,
    inference_epochs: int = INFERENCE_EPOCHS,
) -> ParagraphVectors:
    """One fit per configuration, shared across the module."""
    return model(
        architecture,
        train_words=train_words,
        epochs=epochs,
        inference_epochs=inference_epochs,
    ).fit(TWO_TOPICS)


def fitted_with_trained_words() -> ParagraphVectors:
    return fitted_under(
        ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS,
        train_words=True,
        epochs=TRAIN_WORDS_EPOCHS,
    )


def initial_table(n_rows: int, generator: np.random.Generator) -> np.ndarray:
    """The documented initialisation: uniform in ``(-0.5, 0.5) / dimension``."""
    return (generator.random((n_rows, DIMENSION)) - 0.5) / DIMENSION


def mean_cosine_within(table: DocumentVectors, positions: Sequence[int]) -> float:
    pairs = [
        table.similarity(first, second)
        for first in positions
        for second in positions
        if first < second
    ]
    return float(np.mean(pairs))


def mean_cosine_across(
    table: DocumentVectors,
    first_positions: Sequence[int],
    second_positions: Sequence[int],
) -> float:
    pairs = [
        table.similarity(first, second)
        for first in first_positions
        for second in second_positions
    ]
    return float(np.mean(pairs))


def mean_cosine_to(
    vector: np.ndarray, table: DocumentVectors, positions: Sequence[int]
) -> float:
    return float(
        np.mean(
            [
                cosine_similarity(vector, table.vector_of(position))
                for position in positions
            ]
        )
    )


class TestConstruction:
    def test_defaults_are_the_papers_and_gensims(self) -> None:
        paragraph_vectors = ParagraphVectors()

        assert paragraph_vectors.dimension == 50
        assert paragraph_vectors.window == 5
        assert (
            paragraph_vectors.architecture is ParagraphArchitecture.DISTRIBUTED_MEMORY
        )
        assert paragraph_vectors.n_negative_samples == 5
        assert paragraph_vectors.negative_sampling_exponent == 0.75
        assert paragraph_vectors.epochs == 10
        assert paragraph_vectors.learning_rate == 0.025
        assert paragraph_vectors.minimum_learning_rate == 0.0001
        assert paragraph_vectors.inference_epochs == 20
        assert paragraph_vectors.train_words is False
        assert paragraph_vectors.random_seed is None
        assert paragraph_vectors.minimum_count == 1

    def test_an_unknown_keyword_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ParagraphVectors(dimensions=8)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "field",
        ["dimension", "window", "n_negative_samples", "epochs", "inference_epochs"],
    )
    def test_a_count_below_one_is_refused(self, field: str) -> None:
        with pytest.raises(ValidationError):
            ParagraphVectors(**{field: 0})  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["learning_rate", "minimum_learning_rate"])
    def test_a_rate_at_zero_is_refused(self, field: str) -> None:
        with pytest.raises(ValidationError):
            ParagraphVectors(**{field: 0.0})  # type: ignore[arg-type]

    def test_a_floor_above_the_start_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ParagraphVectors(learning_rate=0.01, minimum_learning_rate=0.02)

    def test_a_floor_equal_to_the_start_is_allowed(self) -> None:
        assert ParagraphVectors(learning_rate=0.01, minimum_learning_rate=0.01)

    def test_train_words_under_distributed_memory_is_refused(self) -> None:
        """Under PV-DM the word vectors are always trained; the switch would be
        accepted and read nowhere."""
        with pytest.raises(ValidationError):
            ParagraphVectors(
                architecture=ParagraphArchitecture.DISTRIBUTED_MEMORY, train_words=True
            )

    def test_train_words_under_the_bag_of_words_is_allowed(self) -> None:
        paragraph_vectors = ParagraphVectors(
            architecture=ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS,
            train_words=True,
        )

        assert paragraph_vectors.train_words is True

    def test_the_architecture_can_be_given_by_name(self) -> None:
        paragraph_vectors = ParagraphVectors(architecture="distributed_bag_of_words")  # type: ignore[arg-type]

        assert (
            paragraph_vectors.architecture
            is ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS
        )


class TestFit:
    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_fit_returns_self(self, architecture: ParagraphArchitecture) -> None:
        paragraph_vectors = model(architecture, epochs=1)

        assert paragraph_vectors.fit(TWO_TOPICS[:4]) is paragraph_vectors

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_every_text_gets_one_document_vector(
        self, architecture: ParagraphArchitecture
    ) -> None:
        fitted = fitted_under(architecture)

        assert fitted.n_documents == len(TWO_TOPICS)
        assert fitted.document_vectors.n_documents == len(TWO_TOPICS)
        assert fitted.document_vectors.dimension == DIMENSION

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_every_word_of_both_lists_gets_a_vector(
        self, architecture: ParagraphArchitecture
    ) -> None:
        embeddings = fitted_under(architecture).embeddings

        assert embeddings.n_words == N_WORDS
        assert embeddings.dimension == DIMENSION
        assert set(embeddings.vocabulary) == set(COOKING_WORDS) | set(SAILING_WORDS)

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_the_output_table_is_one_row_per_word_and_frozen(
        self, architecture: ParagraphArchitecture
    ) -> None:
        output_vectors = fitted_under(architecture).output_vectors

        assert output_vectors.shape == (N_WORDS, DIMENSION)
        with pytest.raises(ValueError):
            output_vectors[0, 0] = 1.0

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_a_seeded_fit_reproduces_every_table_exactly(
        self, architecture: ParagraphArchitecture
    ) -> None:
        first = model(architecture, epochs=2).fit(TWO_TOPICS)
        second = model(architecture, epochs=2).fit(TWO_TOPICS)

        assert first.document_vectors == second.document_vectors
        assert first.embeddings == second.embeddings
        assert np.array_equal(first.output_vectors, second.output_vectors)
        assert first.history == second.history

    def test_a_vocabulary_of_one_word_is_refused(self) -> None:
        with pytest.raises(TooFewValuesError):
            ParagraphVectors().fit(["simmer simmer", "simmer"])

    def test_a_minimum_count_that_leaves_one_word_is_refused(self) -> None:
        with pytest.raises(TooFewValuesError):
            ParagraphVectors(minimum_count=2).fit(["simmer onion", "simmer garlic"])

    def test_a_single_string_corpus_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            ParagraphVectors().fit("simmer the onion")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            ParagraphVectors().fit(["  ", ""])

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_a_document_of_only_rare_words_keeps_its_random_start(
        self, architecture: ParagraphArchitecture
    ) -> None:
        """Every word of the last text appears once, so at a minimum count of
        two it has no ids at all. The fit must not crash, and the document is
        handed the vector it was initialised with, which is re-derived here
        from the documented draw order: the word table first, then the
        document table."""
        corpus = [*TWO_TOPICS, "zzz qqq"]
        fitted = model(architecture, minimum_count=2).fit(corpus)

        generator = np.random.default_rng(SEED)
        initial_table(fitted.embeddings.n_words, generator)
        expected_start = initial_table(len(corpus), generator)[-1]

        assert fitted.n_documents == len(corpus)
        assert "zzz" not in fitted.vocabulary
        assert np.array_equal(
            fitted.document_vectors.vector_of(len(corpus) - 1), expected_start
        )


class TestDistributedMemoryGradients:
    """PV-DM's step, pinned against a central difference of the loss it reports.

    The negatives are drawn from a generator re-seeded identically for every
    evaluation, so the loss is a fixed function of the vectors and the
    difference measures the gradient of that function and nothing else.
    """

    SAMPLER = UnigramSampler([10, 5, 3, 2, 1])
    OUTPUT_VECTORS = (np.random.default_rng(3).random((5, 3)) - 0.5) * 2.0
    DOCUMENT_VECTOR = np.array([0.2, -0.1, 0.4])
    CONTEXT_VECTORS = np.array([[0.0, 0.3, -0.2], [0.1, 0.2, 0.05]])
    TARGET_ID = 1
    N_NEGATIVES = 2
    DRAW_SEED = 11

    def gradients_at(self, document_vector: np.ndarray, context_vectors: np.ndarray):
        return distributed_memory_gradients(
            document_vector,
            context_vectors,
            self.TARGET_ID,
            self.OUTPUT_VECTORS,
            self.SAMPLER,
            self.N_NEGATIVES,
            np.random.default_rng(self.DRAW_SEED),
        )

    def loss_at(
        self, document_vector: np.ndarray, context_vectors: np.ndarray
    ) -> float:
        return self.gradients_at(document_vector, context_vectors).loss

    def test_the_hidden_vector_is_the_mean_of_document_and_context(self) -> None:
        hidden = averaged_hidden(
            np.array([0.2, -0.1, 0.4]), np.array([[0.0, 0.3, -0.2]])
        )

        assert hidden == pytest.approx([0.1, 0.1, 0.1])

    def test_with_no_context_the_hidden_vector_is_the_document_vector(self) -> None:
        """A one-word document has no context and still trains."""
        document_vector = np.array([0.2, -0.1, 0.4])

        assert np.array_equal(
            averaged_hidden(document_vector, np.zeros((0, 3))), document_vector
        )

    def test_context_rows_of_the_wrong_width_are_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            averaged_hidden(np.array([0.2, -0.1, 0.4]), np.zeros((2, 4)))

    def test_it_is_the_foundations_objective_on_the_averaged_hidden(self) -> None:
        """Two routes to one answer: the composed function, and the foundation
        called by hand on the mean, under the same draw."""
        composed = self.gradients_at(self.DOCUMENT_VECTOR, self.CONTEXT_VECTORS)
        by_hand = negative_sampling_gradients(
            np.mean(np.vstack([self.DOCUMENT_VECTOR, self.CONTEXT_VECTORS]), axis=0),
            self.TARGET_ID,
            self.OUTPUT_VECTORS,
            self.SAMPLER,
            self.N_NEGATIVES,
            np.random.default_rng(self.DRAW_SEED),
        )

        assert np.array_equal(composed.output_ids, by_hand.output_ids)
        assert np.allclose(composed.hidden_gradient, by_hand.hidden_gradient)
        assert np.allclose(composed.output_gradients, by_hand.output_gradients)
        assert composed.loss == pytest.approx(by_hand.loss)

    def test_the_document_row_sees_the_hidden_gradient_over_the_averaged_count(
        self,
    ) -> None:
        gradients = self.gradients_at(self.DOCUMENT_VECTOR, self.CONTEXT_VECTORS)
        n_averaged = len(self.CONTEXT_VECTORS) + 1

        measured = np.zeros(3)
        for column in range(3):
            raised = self.DOCUMENT_VECTOR.copy()
            lowered = self.DOCUMENT_VECTOR.copy()
            raised[column] += NUDGE
            lowered[column] -= NUDGE
            measured[column] = (
                self.loss_at(raised, self.CONTEXT_VECTORS)
                - self.loss_at(lowered, self.CONTEXT_VECTORS)
            ) / (2.0 * NUDGE)

        assert np.allclose(measured, gradients.hidden_gradient / n_averaged, atol=1e-8)

    def test_each_context_row_sees_the_same_share(self) -> None:
        gradients = self.gradients_at(self.DOCUMENT_VECTOR, self.CONTEXT_VECTORS)
        n_averaged = len(self.CONTEXT_VECTORS) + 1

        for row in range(len(self.CONTEXT_VECTORS)):
            measured = np.zeros(3)
            for column in range(3):
                raised = self.CONTEXT_VECTORS.copy()
                lowered = self.CONTEXT_VECTORS.copy()
                raised[row, column] += NUDGE
                lowered[row, column] -= NUDGE
                measured[column] = (
                    self.loss_at(self.DOCUMENT_VECTOR, raised)
                    - self.loss_at(self.DOCUMENT_VECTOR, lowered)
                ) / (2.0 * NUDGE)

            assert np.allclose(
                measured, gradients.hidden_gradient / n_averaged, atol=1e-8
            )

    def test_the_output_gradients_agree_with_a_finite_difference_too(self) -> None:
        """Under this seed both negatives are row 0, so the scored rows are
        ``[1, 0, 0]`` and row 0's slope is the sum of two slots. That is the
        argument for applying the output gradients with ``np.add.at``: a plain
        fancy-index assignment would keep one slot and lose the other."""
        gradients = self.gradients_at(self.DOCUMENT_VECTOR, self.CONTEXT_VECTORS)

        assert list(gradients.output_ids) == [1, 0, 0]
        for output_id in set(gradients.output_ids.tolist()):
            expected = gradients.output_gradients[
                gradients.output_ids == output_id
            ].sum(axis=0)
            measured = np.zeros(3)
            for column in range(3):
                raised = self.OUTPUT_VECTORS.copy()
                lowered = self.OUTPUT_VECTORS.copy()
                raised[output_id, column] += NUDGE
                lowered[output_id, column] -= NUDGE
                measured[column] = (
                    distributed_memory_gradients(
                        self.DOCUMENT_VECTOR,
                        self.CONTEXT_VECTORS,
                        self.TARGET_ID,
                        raised,
                        self.SAMPLER,
                        self.N_NEGATIVES,
                        np.random.default_rng(self.DRAW_SEED),
                    ).loss
                    - distributed_memory_gradients(
                        self.DOCUMENT_VECTOR,
                        self.CONTEXT_VECTORS,
                        self.TARGET_ID,
                        lowered,
                        self.SAMPLER,
                        self.N_NEGATIVES,
                        np.random.default_rng(self.DRAW_SEED),
                    ).loss
                ) / (2.0 * NUDGE)

            assert np.allclose(measured, expected, atol=1e-8)

    def test_the_shared_step_is_minus_the_rate_times_the_share(self) -> None:
        step = shared_input_step(np.array([3.0, -6.0, 0.0]), 0.5, 3)

        assert step == pytest.approx([-0.5, 1.0, 0.0])

    def test_the_shared_step_refuses_an_average_of_nothing(self) -> None:
        with pytest.raises(InvalidValuesError):
            shared_input_step(np.array([1.0]), 0.1, 0)

    def test_the_worked_example_in_the_module_docstring(self) -> None:
        """Document ``(0.2, -0.1, 0.4)``, one context word ``(0.0, 0.3, -0.2)``,
        target row ``(1, 0, 0)`` and the only possible negative's row
        ``(0, 1, 0)``. Both scores are ``0.1``; ``sigma(0.1) = 0.524979``."""
        output_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        gradients = distributed_memory_gradients(
            np.array([0.2, -0.1, 0.4]),
            np.array([[0.0, 0.3, -0.2]]),
            0,
            output_vectors,
            UnigramSampler([1, 1]),
            1,
            np.random.default_rng(0),
        )
        sigma = 1.0 / (1.0 + np.exp(-0.1))

        assert list(gradients.output_ids) == [0, 1]
        assert gradients.hidden_gradient == pytest.approx([sigma - 1.0, sigma, 0.0])
        assert gradients.hidden_gradient == pytest.approx(
            [-0.475021, 0.524979, 0.0], abs=1e-6
        )
        assert gradients.loss == pytest.approx(-np.log(sigma) - np.log(1.0 - sigma))
        assert shared_input_step(gradients.hidden_gradient, 0.025, 2) == pytest.approx(
            [0.005938, -0.006562, 0.0], abs=1e-6
        )


class TestDistributedBagOfWordsGradients:
    SAMPLER = UnigramSampler([10, 5, 3, 2, 1])
    OUTPUT_VECTORS = (np.random.default_rng(5).random((5, 3)) - 0.5) * 2.0
    DOCUMENT_VECTOR = np.array([0.3, -0.2, 0.1])
    TARGET_ID = 2
    DRAW_SEED = 13

    def loss_at(self, document_vector: np.ndarray) -> float:
        return distributed_bag_of_words_gradients(
            document_vector,
            self.TARGET_ID,
            self.OUTPUT_VECTORS,
            self.SAMPLER,
            3,
            np.random.default_rng(self.DRAW_SEED),
        ).loss

    def test_it_is_the_foundations_objective_with_the_document_as_hidden(self) -> None:
        composed = distributed_bag_of_words_gradients(
            self.DOCUMENT_VECTOR,
            self.TARGET_ID,
            self.OUTPUT_VECTORS,
            self.SAMPLER,
            3,
            np.random.default_rng(self.DRAW_SEED),
        )
        by_hand = negative_sampling_gradients(
            self.DOCUMENT_VECTOR,
            self.TARGET_ID,
            self.OUTPUT_VECTORS,
            self.SAMPLER,
            3,
            np.random.default_rng(self.DRAW_SEED),
        )

        assert np.array_equal(composed.output_ids, by_hand.output_ids)
        assert np.array_equal(composed.hidden_gradient, by_hand.hidden_gradient)
        assert composed.loss == by_hand.loss

    def test_the_document_vector_sees_the_whole_hidden_gradient(self) -> None:
        gradients = distributed_bag_of_words_gradients(
            self.DOCUMENT_VECTOR,
            self.TARGET_ID,
            self.OUTPUT_VECTORS,
            self.SAMPLER,
            3,
            np.random.default_rng(self.DRAW_SEED),
        )

        measured = np.zeros(3)
        for column in range(3):
            raised = self.DOCUMENT_VECTOR.copy()
            lowered = self.DOCUMENT_VECTOR.copy()
            raised[column] += NUDGE
            lowered[column] -= NUDGE
            measured[column] = (self.loss_at(raised) - self.loss_at(lowered)) / (
                2.0 * NUDGE
            )

        assert np.allclose(measured, gradients.hidden_gradient, atol=1e-8)


class TestDocumentVectorsClusterByTopic:
    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_documents_of_one_topic_are_nearer_each_other_than_the_other_topic(
        self, architecture: ParagraphArchitecture
    ) -> None:
        table = fitted_under(architecture).document_vectors

        within_cooking = mean_cosine_within(table, COOKING_POSITIONS)
        within_sailing = mean_cosine_within(table, SAILING_POSITIONS)
        across = mean_cosine_across(table, COOKING_POSITIONS, SAILING_POSITIONS)

        # Measured: 0.8974 / 0.8696 within and 0.2811 across under PV-DM;
        # 0.7763 / 0.8078 and 0.1445 under PV-DBOW. The across figure is
        # positive, since every document trains against the same output rows,
        # so the claim is a margin and not a sign.
        assert within_cooking > 0.7
        assert within_sailing > 0.7
        assert across < 0.3
        assert min(within_cooking, within_sailing) > across + 0.4

    def test_word2vecs_defaults_are_too_few_updates_for_a_corpus_this_small(
        self,
    ) -> None:
        """Ten epochs from 0.025 over 240 positions leave the document vectors
        at their random starts: measured within-topic cosine 0.0516 and across
        0.0437, indistinguishable, with the mean loss at 4.1155 from 4.1586.
        Recorded so the spec's settings read as a measurement, not a tuning."""
        fitted = ParagraphVectors(
            dimension=DIMENSION, window=WINDOW, random_seed=SEED
        ).fit(TWO_TOPICS)
        table = fitted.document_vectors

        within = mean_cosine_within(table, COOKING_POSITIONS)
        across = mean_cosine_across(table, COOKING_POSITIONS, SAILING_POSITIONS)

        assert abs(within - across) < 0.05
        assert fitted.history.mean_losses[-1] > 4.0

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_the_nearest_documents_to_a_cooking_text_are_all_cooking(
        self, architecture: ParagraphArchitecture
    ) -> None:
        nearest = fitted_under(architecture).most_similar_documents(
            COOKING_POSITIONS[0], N_PER_TOPIC - 1
        )

        assert len(nearest) == N_PER_TOPIC - 1
        assert set(nearest) == set(COOKING_POSITIONS) - {COOKING_POSITIONS[0]}

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_document_similarity_reads_the_table(
        self, architecture: ParagraphArchitecture
    ) -> None:
        fitted = fitted_under(architecture)

        # Measured: 0.9212 against 0.0177 under PV-DM, 0.7556 against 0.1901
        # under PV-DBOW.
        assert fitted.document_similarity(0, 1) == fitted.document_vectors.similarity(
            0, 1
        )
        assert fitted.document_similarity(
            0, SAILING_POSITIONS[0]
        ) < fitted.document_similarity(0, 1)


class TestHistory:
    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_the_loss_fell(self, architecture: ParagraphArchitecture) -> None:
        history = fitted_under(architecture).history

        assert history.fell
        assert history.n_epochs == EPOCHS

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_one_pair_per_position_per_epoch(
        self, architecture: ParagraphArchitecture
    ) -> None:
        history = fitted_under(architecture).history

        assert all(
            record.n_pairs == 2 * N_PER_TOPIC * WORDS_PER_DOCUMENT for record in history
        )

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_the_learning_rate_decays_towards_the_floor(
        self, architecture: ParagraphArchitecture
    ) -> None:
        history = fitted_under(architecture).history
        rates = [record.learning_rate for record in history]

        assert rates == sorted(rates, reverse=True)
        assert rates[-1] >= fitted_under(architecture).minimum_learning_rate

    def test_the_first_epochs_mean_loss_sits_just_below_the_uninformed_value(
        self,
    ) -> None:
        """Six rows scored against a zero output table each cost ``log 2``, so
        an untrained pair costs ``6 log 2 = 4.1589``, and the first epoch's
        mean sits just below it because the rows begin to move within the
        epoch. Measured: 4.1529 under PV-DM."""
        first = fitted_under(ParagraphArchitecture.DISTRIBUTED_MEMORY).history[0]

        assert first.mean_loss < 6.0 * np.log(2.0)
        assert first.mean_loss == pytest.approx(4.1529, abs=1e-3)


class TestInference:
    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_a_held_out_cooking_text_lands_among_the_cooking_documents(
        self, architecture: ParagraphArchitecture
    ) -> None:
        fitted = fitted_under(architecture)
        inferred = fitted.infer_vector(HELD_OUT_COOKING)

        to_cooking = mean_cosine_to(
            inferred, fitted.document_vectors, COOKING_POSITIONS
        )
        to_sailing = mean_cosine_to(
            inferred, fitted.document_vectors, SAILING_POSITIONS
        )

        # Measured at two hundred passes: 0.9081 / 0.1405 under PV-DM and
        # 0.7665 / 0.0924 under PV-DBOW.
        assert to_cooking > 0.7
        assert to_cooking > to_sailing + 0.4

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_a_held_out_sailing_text_lands_among_the_sailing_documents(
        self, architecture: ParagraphArchitecture
    ) -> None:
        fitted = fitted_under(architecture)
        inferred = fitted.infer_vector(HELD_OUT_SAILING)

        to_cooking = mean_cosine_to(
            inferred, fitted.document_vectors, COOKING_POSITIONS
        )
        to_sailing = mean_cosine_to(
            inferred, fitted.document_vectors, SAILING_POSITIONS
        )

        # Measured: 0.2827 / 0.9025 under PV-DM and 0.0753 / 0.7843 under PV-DBOW.
        assert to_sailing > 0.7
        assert to_sailing > to_cooking + 0.4

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_the_nearest_fitted_document_to_an_inferred_vector_shares_its_topic(
        self, architecture: ParagraphArchitecture
    ) -> None:
        fitted = fitted_under(architecture)

        nearest = fitted.document_vectors.similar_to_vector(
            fitted.infer_vector(HELD_OUT_SAILING), n_results=5
        )

        assert set(nearest) <= set(SAILING_POSITIONS)

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_the_same_text_under_the_same_seed_gives_the_same_vector(
        self, architecture: ParagraphArchitecture
    ) -> None:
        fitted = fitted_under(architecture)

        assert np.array_equal(
            fitted.infer_vector(HELD_OUT_COOKING, random_seed=4),
            fitted.infer_vector(HELD_OUT_COOKING, random_seed=4),
        )

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_without_a_seed_the_models_own_seed_is_used(
        self, architecture: ParagraphArchitecture
    ) -> None:
        fitted = fitted_under(architecture)

        assert np.array_equal(
            fitted.infer_vector(HELD_OUT_COOKING),
            fitted.infer_vector(HELD_OUT_COOKING, random_seed=SEED),
        )

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_a_different_seed_gives_a_different_vector_in_the_same_place(
        self, architecture: ParagraphArchitecture
    ) -> None:
        fitted = fitted_under(architecture)
        first = fitted.infer_vector(HELD_OUT_COOKING, random_seed=1)
        second = fitted.infer_vector(HELD_OUT_COOKING, random_seed=2)

        assert not np.array_equal(first, second)
        assert cosine_similarity(first, second) > 0.9

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_inference_leaves_every_fitted_table_untouched(
        self, architecture: ParagraphArchitecture
    ) -> None:
        fitted = fitted_under(architecture)
        documents_before = DocumentVectors(np.array(fitted.document_vectors))
        words_before = np.array(fitted.embeddings.table)
        outputs_before = np.array(fitted.output_vectors)

        fitted.infer_vector(HELD_OUT_COOKING)
        fitted.infer_vector(HELD_OUT_SAILING, random_seed=9)

        assert fitted.document_vectors == documents_before
        assert np.array_equal(fitted.embeddings.table, words_before)
        assert np.array_equal(fitted.output_vectors, outputs_before)

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_the_inferred_vector_is_frozen_and_of_the_models_dimension(
        self, architecture: ParagraphArchitecture
    ) -> None:
        inferred = fitted_under(architecture).infer_vector(HELD_OUT_COOKING)

        assert inferred.shape == (DIMENSION,)
        with pytest.raises(ValueError):
            inferred[0] = 1.0

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    @pytest.mark.parametrize("text", ["zzz qqq", "   ", ""])
    def test_a_text_with_no_known_word_is_handed_its_random_start(
        self, architecture: ParagraphArchitecture, text: str
    ) -> None:
        inferred = fitted_under(architecture).infer_vector(text, random_seed=21)

        assert np.array_equal(inferred, initial_table(1, np.random.default_rng(21))[0])

    @pytest.mark.parametrize("architecture", BOTH_ARCHITECTURES)
    def test_a_text_that_is_not_a_string_is_refused(
        self, architecture: ParagraphArchitecture
    ) -> None:
        with pytest.raises(InvalidValuesError):
            fitted_under(architecture).infer_vector(["butter", "garlic"])  # type: ignore[arg-type]

    def test_the_pass_count_does_not_touch_the_fit(self) -> None:
        one_pass = fitted_under(
            ParagraphArchitecture.DISTRIBUTED_MEMORY, inference_epochs=1
        )

        assert (
            one_pass.document_vectors
            == fitted_under(ParagraphArchitecture.DISTRIBUTED_MEMORY).document_vectors
        )

    def test_distributed_memory_needs_many_passes_because_the_document_is_a_share(
        self,
    ) -> None:
        """Under the averaging variant the new vector is one of ``n_context + 1``
        vectors in the mean and takes that share of the gradient, while the
        fitted word rows have norms 1.74 to 2.68 against document rows of 0.29
        to 0.53. Measured on the held-out cooking text, cosine to the cooking
        documents against the sailing ones: one pass -0.3490 / -0.0298, twenty
        passes 0.4786 / 0.7608 (the wrong topic), two hundred 0.9081 / 0.1405.
        The twenty-pass figure is the default's, pinned as it is."""
        table = fitted_under(ParagraphArchitecture.DISTRIBUTED_MEMORY).document_vectors
        by_passes = {
            passes: fitted_under(
                ParagraphArchitecture.DISTRIBUTED_MEMORY, inference_epochs=passes
            ).infer_vector(HELD_OUT_COOKING)
            for passes in (1, 20, INFERENCE_EPOCHS)
        }

        assert mean_cosine_to(by_passes[1], table, COOKING_POSITIONS) < 0.0
        assert mean_cosine_to(by_passes[20], table, COOKING_POSITIONS) < mean_cosine_to(
            by_passes[20], table, SAILING_POSITIONS
        )
        assert (
            mean_cosine_to(by_passes[INFERENCE_EPOCHS], table, COOKING_POSITIONS) > 0.9
        )

    def test_the_bag_of_words_lands_in_one_pass_and_specialises_with_more(
        self,
    ) -> None:
        """The document vector is the whole hidden vector, so one pass already
        places it: 0.8330 to cooking against 0.4047 to sailing. Further passes
        fit the vector to its own eight words rather than to the topic, and
        the cosine to the topic's documents falls: 0.8565 at twenty, 0.7665 at
        two hundred, while its norm grows from 0.846 to 2.921."""
        table = fitted_under(
            ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS
        ).document_vectors
        one_pass = fitted_under(
            ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS, inference_epochs=1
        ).infer_vector(HELD_OUT_COOKING)
        many_passes = fitted_under(
            ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS
        ).infer_vector(HELD_OUT_COOKING)

        assert mean_cosine_to(one_pass, table, COOKING_POSITIONS) > 0.8
        assert (
            mean_cosine_to(one_pass, table, COOKING_POSITIONS)
            > mean_cosine_to(one_pass, table, SAILING_POSITIONS) + 0.4
        )
        assert mean_cosine_to(many_passes, table, COOKING_POSITIONS) < mean_cosine_to(
            one_pass, table, COOKING_POSITIONS
        )
        assert np.linalg.norm(many_passes) > 3.0 * np.linalg.norm(one_pass)


class TestWordVectorsUnderTheBagOfWords:
    def test_plain_dbow_leaves_the_word_vectors_at_their_random_start(self) -> None:
        """The honest note in the module docstring, pinned: the embeddings a
        plain PV-DBOW fit answers are the initial draw, untouched."""
        fitted = fitted_under(ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS)

        assert fitted.word_vectors_were_trained is False
        assert np.array_equal(
            fitted.embeddings.table, initial_table(N_WORDS, np.random.default_rng(SEED))
        )

    def test_train_words_moves_them(self) -> None:
        fitted = fitted_with_trained_words()

        assert fitted.word_vectors_were_trained is True
        assert not np.array_equal(
            fitted.embeddings.table, initial_table(N_WORDS, np.random.default_rng(SEED))
        )

    def test_train_words_adds_a_skip_gram_pair_per_context_word(self) -> None:
        """Eight words at window three: the two end positions see three context
        words, the next two see four, five and six, and the middle two see six
        each, so ``2 * (3 + 4 + 5) + 2 * 6 = 36`` skip-gram pairs per document
        beside its eight document pairs."""
        fitted = fitted_with_trained_words()

        assert all(
            record.n_pairs == 2 * N_PER_TOPIC * (WORDS_PER_DOCUMENT + 36)
            for record in fitted.history
        )
        assert fitted.history.n_epochs == TRAIN_WORDS_EPOCHS
        assert fitted.history.fell

    def test_train_words_still_clusters_the_documents(self) -> None:
        """Measured: 0.9808 within against 0.1998 across, at ten epochs."""
        table = fitted_with_trained_words().document_vectors

        assert (
            mean_cosine_within(table, COOKING_POSITIONS)
            > mean_cosine_across(table, COOKING_POSITIONS, SAILING_POSITIONS) + 0.4
        )

    def test_train_words_learns_the_topics_for_the_words_too(self) -> None:
        """Measured: ``butter`` to ``garlic`` 0.9666, to ``anchor`` 0.0965."""
        embeddings = fitted_with_trained_words().embeddings

        assert embeddings.similarity("butter", "garlic") > 0.9
        assert embeddings.similarity("butter", "anchor") < 0.3

    def test_distributed_memory_always_trains_the_words(self) -> None:
        fitted = fitted_under(ParagraphArchitecture.DISTRIBUTED_MEMORY)

        assert fitted.word_vectors_were_trained is True
        assert fitted.similarity("butter", "garlic") > fitted.similarity(
            "butter", "anchor"
        )


class TestNotFitted:
    @pytest.mark.parametrize(
        "attribute",
        [
            "embeddings",
            "document_vectors",
            "output_vectors",
            "history",
            "n_documents",
            "word_vectors_were_trained",
            "vocabulary",
        ],
    )
    def test_every_fitted_property_raises(self, attribute: str) -> None:
        with pytest.raises(NotFittedError):
            getattr(ParagraphVectors(), attribute)

    def test_infer_vector_raises(self) -> None:
        with pytest.raises(NotFittedError):
            ParagraphVectors().infer_vector(HELD_OUT_COOKING)

    def test_document_similarity_raises(self) -> None:
        with pytest.raises(NotFittedError):
            ParagraphVectors().document_similarity(0, 1)

    def test_most_similar_documents_raises(self) -> None:
        with pytest.raises(NotFittedError):
            ParagraphVectors().most_similar_documents(0)

    def test_is_fitted_reads_false_then_true(self) -> None:
        paragraph_vectors = model(ParagraphArchitecture.DISTRIBUTED_MEMORY, epochs=1)

        assert paragraph_vectors.is_fitted is False
        assert paragraph_vectors.fit(TWO_TOPICS[:4]).is_fitted is True


class TestDocumentVectors:
    FOUR = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [3.0, 0.0]])

    def test_it_copies_and_freezes(self) -> None:
        source = self.FOUR.copy()
        table = DocumentVectors(source)
        source[0, 0] = 99.0

        assert table.vector_of(0)[0] == 1.0
        with pytest.raises(ValueError):
            table.vectors[0, 0] = 5.0

    def test_it_knows_its_counts(self) -> None:
        table = DocumentVectors(self.FOUR)

        assert table.n_documents == 4
        assert table.dimension == 2
        assert len(table) == 4

    def test_it_iterates_the_vectors_in_order(self) -> None:
        rows = list(DocumentVectors(self.FOUR))

        assert len(rows) == 4
        assert all(
            np.array_equal(row, expected)
            for row, expected in zip(rows, self.FOUR, strict=True)
        )

    def test_a_vector_is_read_by_position_and_frozen(self) -> None:
        vector = DocumentVectors(self.FOUR).vector_of(2)

        assert np.array_equal(vector, [0.0, 1.0])
        with pytest.raises(ValueError):
            vector[0] = 1.0

    @pytest.mark.parametrize("position", [-1, 4, 100])
    def test_a_position_no_document_has_is_refused(self, position: int) -> None:
        with pytest.raises(InvalidValuesError):
            DocumentVectors(self.FOUR).vector_of(position)

    def test_similarity_is_the_cosine(self) -> None:
        table = DocumentVectors(np.array([[1.0, 0.0], [1.0, 1.0]]))

        assert table.similarity(0, 1) == pytest.approx(1.0 / np.sqrt(2.0))
        assert table.similarity(1, 0) == table.similarity(0, 1)

    def test_similarity_to_a_zero_vector_is_undefined(self) -> None:
        table = DocumentVectors(np.array([[1.0, 0.0], [0.0, 0.0]]))

        with pytest.raises(UndefinedMetricError):
            table.similarity(0, 1)

    def test_most_similar_excludes_itself_and_breaks_ties_by_lower_position(
        self,
    ) -> None:
        """Rows 1 and 3 both point the way row 0 does; row 2 is orthogonal."""
        table = DocumentVectors(self.FOUR)

        assert table.most_similar(0) == (1, 3, 2)
        assert table.most_similar(0, n_results=2) == (1, 3)

    def test_most_similar_skips_a_document_with_a_zero_vector(self) -> None:
        table = DocumentVectors(np.vstack([self.FOUR, [[0.0, 0.0]]]))

        assert table.most_similar(0) == (1, 3, 2)

    def test_most_similar_of_a_zero_vector_is_undefined(self) -> None:
        table = DocumentVectors(np.vstack([self.FOUR, [[0.0, 0.0]]]))

        with pytest.raises(UndefinedMetricError):
            table.most_similar(4)

    def test_most_similar_refuses_no_results(self) -> None:
        with pytest.raises(InvalidValuesError):
            DocumentVectors(self.FOUR).most_similar(0, n_results=0)

    def test_a_lone_document_has_no_neighbours(self) -> None:
        assert DocumentVectors(np.array([[1.0, 2.0]])).most_similar(0) == ()

    def test_most_similar_is_similar_to_its_own_vector_with_itself_excluded(
        self,
    ) -> None:
        """Two routes: the convenience and the vector query it delegates to.
        Row 2 is the one row pointing its own way, so unexcluded it is its own
        nearest neighbour; rows 0, 1 and 3 tie at zero behind it and the tie
        rule orders them by position."""
        table = DocumentVectors(self.FOUR)

        assert table.most_similar(2) == table.similar_to_vector(
            table.vector_of(2), excluding=(2,)
        )
        assert table.most_similar(2) == (0, 1, 3)
        assert table.similar_to_vector(table.vector_of(2)) == (2, 0, 1, 3)

    def test_similar_to_vector_refuses_the_wrong_width(self) -> None:
        with pytest.raises(ShapeMismatchError):
            DocumentVectors(self.FOUR).similar_to_vector(np.array([1.0, 0.0, 0.0]))

    def test_similar_to_vector_refuses_the_zero_vector(self) -> None:
        with pytest.raises(UndefinedMetricError):
            DocumentVectors(self.FOUR).similar_to_vector(np.array([0.0, 0.0]))

    def test_similar_to_vector_refuses_a_non_finite_query(self) -> None:
        with pytest.raises(InvalidValuesError):
            DocumentVectors(self.FOUR).similar_to_vector(np.array([1.0, np.nan]))

    def test_a_one_dimensional_table_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.array([1.0, 2.0]))

    def test_a_table_with_no_documents_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            DocumentVectors(np.zeros((0, 3)))

    def test_a_table_with_no_dimensions_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.zeros((3, 0)))

    def test_a_non_finite_entry_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.array([[1.0, np.inf]]))

    def test_a_non_numeric_table_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.array([["a", "b"]]))  # type: ignore[arg-type]

    def test_equality_is_a_verdict_on_the_vectors(self) -> None:
        first = DocumentVectors(self.FOUR)
        second = DocumentVectors(self.FOUR.copy())
        third = DocumentVectors(self.FOUR * 2.0)

        assert first == second
        assert first != third
        assert hash(first) == hash(second)

    def test_comparing_with_something_else_defers(self) -> None:
        table = DocumentVectors(self.FOUR)

        assert table.__eq__(3) is NotImplemented
        assert (table == 3) is False
        assert (table != 3) is True

    def test_the_array_protocol_honours_copy(self) -> None:
        table = DocumentVectors(self.FOUR)

        assert not np.shares_memory(np.array(table), table.vectors)
        assert np.shares_memory(np.asarray(table), table.vectors)
        assert np.array(table, dtype=np.float32).dtype == np.float32
        assert np.array_equal(np.array(table), self.FOUR)

    def test_repr_names_the_counts(self) -> None:
        assert repr(DocumentVectors(self.FOUR)) == (
            "DocumentVectors(n_documents=4, dimension=2)"
        )
