"""Spec for Word2Vec -- the exemplar word embedder, under all four of its shapes.

The corpus is two word lists that never share a sentence, and the claim is that
the fit finds that: the mean cosine between two words of one list is above the
mean cosine between a word of each. Measured at dimension 12, window 3, five
epochs, seed 0, learning rate 0.05, the four within/across pairs are
0.996/0.170 (skip-gram, negative sampling), 0.978/-0.557 (skip-gram, tree),
0.993/0.712 (bag of words, negative sampling) and 0.983/-0.962 (bag of words,
tree). The rate is 0.05 rather than the default 0.025 because at the default,
after five epochs, the bag of words under negative sampling has every vector
sharing one direction, 0.991 within against 0.987 across, which is the
familiar early-training collinearity and not a separation anyone should
assert; doubling the rate or the epochs moves it apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    TooFewValuesError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.word2vec import (
    EpochRecord,
    TrainingHistory,
    Word2Vec,
    Word2VecArchitecture,
    Word2VecObjective,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from test.core.natural_language_processing.embeddings.prediction.corpora import (
    FINANCE_WORDS,
    TWO_TOPIC_CORPUS,
    VERB_WORDS,
)

COMBINATIONS = [
    (architecture, objective)
    for architecture in Word2VecArchitecture
    for objective in Word2VecObjective
]
COMBINATION_IDS = [
    f"{architecture.value}-{objective.value}"
    for architecture, objective in COMBINATIONS
]


def two_topic_model(
    architecture: Word2VecArchitecture = Word2VecArchitecture.SKIP_GRAM,
    objective: Word2VecObjective = Word2VecObjective.NEGATIVE_SAMPLING,
    random_seed: int = 0,
) -> Word2Vec:
    return Word2Vec(
        dimension=12,
        window=3,
        architecture=architecture,
        objective=objective,
        epochs=5,
        learning_rate=0.05,
        random_seed=random_seed,
    )


@pytest.fixture(scope="module")
def fitted_by_combination() -> dict[
    tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
]:
    return {
        (architecture, objective): two_topic_model(architecture, objective).fit(
            TWO_TOPIC_CORPUS
        )
        for architecture, objective in COMBINATIONS
    }


def mean_within_topic_similarity(embeddings: WordEmbeddings) -> float:
    similarities: list[float] = []
    for words in (VERB_WORDS, FINANCE_WORDS):
        for position, first in enumerate(words):
            for second in words[position + 1 :]:
                similarities.append(embeddings.similarity(first, second))
    return float(np.mean(similarities))


def mean_across_topic_similarity(embeddings: WordEmbeddings) -> float:
    return float(
        np.mean(
            [
                embeddings.similarity(first, second)
                for first in VERB_WORDS
                for second in FINANCE_WORDS
            ]
        )
    )


class TestSeparation:
    @pytest.mark.parametrize("combination", COMBINATIONS, ids=COMBINATION_IDS)
    def test_words_of_one_topic_are_nearer_each_other_than_the_other_topic(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
        combination: tuple[Word2VecArchitecture, Word2VecObjective],
    ) -> None:
        embeddings = fitted_by_combination[combination].embeddings

        assert mean_within_topic_similarity(embeddings) > mean_across_topic_similarity(
            embeddings
        )

    @pytest.mark.parametrize("combination", COMBINATIONS, ids=COMBINATION_IDS)
    def test_the_loss_fell(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
        combination: tuple[Word2VecArchitecture, Word2VecObjective],
    ) -> None:
        history = fitted_by_combination[combination].history

        assert history.fell
        assert history.n_epochs == 5
        assert [record.epoch for record in history] == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("combination", COMBINATIONS, ids=COMBINATION_IDS)
    def test_the_table_is_one_row_per_word_of_the_stated_dimension(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
        combination: tuple[Word2VecArchitecture, Word2VecObjective],
    ) -> None:
        model = fitted_by_combination[combination]

        assert model.embeddings.table.shape == (31, 12)
        assert model.embeddings.dimension == 12
        assert set(model.vocabulary) == set(VERB_WORDS) | set(FINANCE_WORDS)

    def test_fit_returns_self(self) -> None:
        model = Word2Vec(dimension=4, epochs=1)

        assert model.fit(["a b c", "b c a"]) is model


class TestDeterminism:
    def test_the_same_seed_reproduces_the_fit(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
    ) -> None:
        first = fitted_by_combination[
            (Word2VecArchitecture.SKIP_GRAM, Word2VecObjective.NEGATIVE_SAMPLING)
        ]
        second = two_topic_model().fit(TWO_TOPIC_CORPUS)

        assert second.embeddings == first.embeddings
        assert second.history == first.history
        assert np.array_equal(second.output_vectors, first.output_vectors)


class TestVocabulary:
    def test_ordered_commonest_first_with_alphabetical_ties(self) -> None:
        """``a`` and ``c`` both appear three times, ``b`` once."""
        model = Word2Vec(dimension=4, epochs=1).fit(["b a a c c c", "a"])

        assert list(model.vocabulary) == ["a", "c", "b"]

    def test_minimum_count_drops_a_word_and_the_dropped_word_is_unknown(
        self,
    ) -> None:
        model = Word2Vec(dimension=4, epochs=1, minimum_count=2).fit(["a a a b b c"])

        assert list(model.vocabulary) == ["a", "b"]
        assert "c" not in model.embeddings
        with pytest.raises(UnknownTokenError):
            model.vector_of("c")

    def test_the_vocabulary_has_no_unknown_token(self) -> None:
        model = Word2Vec(dimension=4, epochs=1).fit(["a b c"])

        assert model.vocabulary.unknown_token is None


class TestOutputVectors:
    def test_one_row_per_word_under_negative_sampling(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
    ) -> None:
        model = fitted_by_combination[
            (Word2VecArchitecture.SKIP_GRAM, Word2VecObjective.NEGATIVE_SAMPLING)
        ]

        assert model.output_vectors.shape == (31, 12)

    def test_one_row_per_internal_node_under_the_tree(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
    ) -> None:
        model = fitted_by_combination[
            (Word2VecArchitecture.SKIP_GRAM, Word2VecObjective.HIERARCHICAL_SOFTMAX)
        ]

        assert model.output_vectors.shape == (30, 12)

    def test_the_output_table_is_frozen(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
    ) -> None:
        for model in fitted_by_combination.values():
            assert not model.output_vectors.flags.writeable

    def test_the_output_table_moved_from_zero(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
    ) -> None:
        for model in fitted_by_combination.values():
            assert np.any(model.output_vectors != 0.0)


class TestLearningRate:
    def test_the_floor_is_honoured_on_the_last_record(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
    ) -> None:
        """Over 6555 positions the linear decay ends at ``0.05 / 6555``, below
        the floor, so the floor is what the last record reports."""
        for model in fitted_by_combination.values():
            assert model.history[-1].learning_rate == model.minimum_learning_rate

    def test_a_high_floor_is_what_the_last_record_reports(self) -> None:
        model = Word2Vec(
            dimension=4, epochs=2, learning_rate=0.025, minimum_learning_rate=0.02
        ).fit(["a b c d", "d c b a"])

        assert model.history[-1].learning_rate == 0.02

    def test_after_one_epoch_of_five_the_rate_has_fallen_by_about_a_fifth(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
    ) -> None:
        model = fitted_by_combination[
            (Word2VecArchitecture.SKIP_GRAM, Word2VecObjective.NEGATIVE_SAMPLING)
        ]

        assert 0.039 < model.history[0].learning_rate < 0.041

    def test_the_rate_falls_from_epoch_to_epoch(
        self,
        fitted_by_combination: dict[
            tuple[Word2VecArchitecture, Word2VecObjective], Word2Vec
        ],
    ) -> None:
        rates = [
            record.learning_rate
            for record in fitted_by_combination[
                (Word2VecArchitecture.SKIP_GRAM, Word2VecObjective.NEGATIVE_SAMPLING)
            ].history
        ]

        assert rates == sorted(rates, reverse=True)


class TestSubsampling:
    def test_keep_probabilities_follow_word2vecs_rule(self) -> None:
        """Counts ``990, 10`` at threshold ``1e-3``: frequencies 0.99 and 0.01,
        so ``(sqrt(f / t) + 1) t / f`` is 0.032792 and 0.416228."""
        keep = Word2Vec(dimension=4, subsampling_threshold=1e-3)._keep_probabilities(
            [990, 10]
        )

        assert keep is not None
        assert np.allclose(keep, [0.032792, 0.416228], atol=1e-6)

    def test_thinning_begins_at_two_point_six_times_the_threshold(self) -> None:
        """``sqrt(t / f) + t / f`` drops below one only once ``t / f < 0.382``,
        so a word is thinned only when its frequency exceeds ``2.618 t``, the
        golden ratio squared. At threshold 0.5 a word at frequency 0.999 is
        kept in full; at 0.3 it is kept with probability
        ``(sqrt(0.999 / 0.3) + 1) * 0.3 / 0.999 = 0.8483``."""
        at_half = Word2Vec(dimension=4, subsampling_threshold=0.5)._keep_probabilities(
            [1, 999]
        )
        at_a_third = Word2Vec(
            dimension=4, subsampling_threshold=0.3
        )._keep_probabilities([1, 999])

        assert at_half is not None and at_a_third is not None
        assert at_half[0] == 1.0 and at_half[1] == 1.0
        assert at_a_third[0] == 1.0
        assert at_a_third[1] == pytest.approx(0.8483, abs=1e-4)

    def test_no_threshold_keeps_every_word(self) -> None:
        sentence = (0, 1, 0, 2, 0)

        assert Word2Vec(dimension=4)._keep_probabilities([3, 1, 1]) is None
        assert Word2Vec._subsampled(sentence, None, np.random.default_rng(0)) == (
            sentence
        )

    def test_a_very_frequent_word_is_thinned(self) -> None:
        """A hundred occurrences kept with probability 0.033 each: about three."""
        keep = Word2Vec(dimension=4, subsampling_threshold=1e-3)._keep_probabilities(
            [990, 10]
        )
        kept = Word2Vec._subsampled((0,) * 100, keep, np.random.default_rng(0))

        assert len(kept) < 20

    def test_a_fit_with_subsampling_sees_fewer_pairs(self) -> None:
        corpus = ["the a the b the c the d the e the f the g"] * 5
        plain = Word2Vec(dimension=4, epochs=1, random_seed=0).fit(corpus)
        thinned = Word2Vec(
            dimension=4, epochs=1, random_seed=0, subsampling_threshold=1e-2
        ).fit(corpus)

        assert thinned.history[0].n_pairs < plain.history[0].n_pairs


class TestConstruction:
    def test_a_floor_above_the_start_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Word2Vec(learning_rate=0.01, minimum_learning_rate=0.02)

    @pytest.mark.parametrize(
        "keywords",
        [
            {"dimension": 0},
            {"window": 0},
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"minimum_learning_rate": 0.0},
            {"n_negative_samples": 0},
            {"negative_sampling_exponent": 0.0},
            {"negative_sampling_exponent": 1.5},
            {"subsampling_threshold": 0.0},
            {"minimum_count": 0},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(
        self, keywords: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            Word2Vec(**keywords)

    def test_an_unknown_keyword_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Word2Vec(size=10)  # type: ignore[call-arg]

    def test_the_defaults_are_the_reference_ones(self) -> None:
        model = Word2Vec()

        assert model.dimension == 100
        assert model.window == 5
        assert model.architecture is Word2VecArchitecture.SKIP_GRAM
        assert model.objective is Word2VecObjective.NEGATIVE_SAMPLING
        assert model.n_negative_samples == 5
        assert model.negative_sampling_exponent == 0.75
        assert model.learning_rate == 0.025
        assert model.shrink_windows is True


class TestRefusals:
    def test_a_one_word_vocabulary_is_refused(self) -> None:
        with pytest.raises(TooFewValuesError):
            Word2Vec(dimension=4).fit(["a a a a"])

    def test_a_minimum_count_leaving_one_word_is_refused(self) -> None:
        with pytest.raises(TooFewValuesError):
            Word2Vec(dimension=4, minimum_count=2).fit(["a a a b"])

    def test_a_single_string_corpus_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            Word2Vec(dimension=4).fit("a b c")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            Word2Vec(dimension=4).fit(["  ", ""])

    def test_before_fit_everything_learned_raises_not_fitted(self) -> None:
        model = Word2Vec(dimension=4)

        with pytest.raises(NotFittedError):
            _ = model.embeddings
        with pytest.raises(NotFittedError):
            _ = model.history
        with pytest.raises(NotFittedError):
            _ = model.output_vectors
        with pytest.raises(NotFittedError):
            _ = model.vocabulary
        with pytest.raises(NotFittedError):
            model.vector_of("a")
        assert not model.is_fitted


class TestEpochRecord:
    def test_carries_its_four_numbers(self) -> None:
        record = EpochRecord(3, 1.25, 40, 0.01)

        assert record.epoch == 3
        assert record.mean_loss == 1.25
        assert record.n_pairs == 40
        assert record.learning_rate == 0.01

    def test_compares_by_value(self) -> None:
        assert EpochRecord(1, 1.0, 4, 0.1) == EpochRecord(1, 1.0, 4, 0.1)
        assert EpochRecord(1, 1.0, 4, 0.1) != EpochRecord(1, 2.0, 4, 0.1)
        assert EpochRecord(1, 1.0, 4, 0.1) != "epoch"
        assert hash(EpochRecord(1, 1.0, 4, 0.1)) == hash(EpochRecord(1, 1.0, 4, 0.1))

    def test_an_epoch_below_one_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            EpochRecord(0, 1.0, 4, 0.1)

    def test_a_negative_pair_count_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            EpochRecord(1, 1.0, -1, 0.1)

    @pytest.mark.parametrize("mean_loss", [-0.1, float("nan"), float("inf")])
    def test_a_negative_or_non_finite_loss_is_refused(self, mean_loss: float) -> None:
        with pytest.raises(InvalidValuesError):
            EpochRecord(1, mean_loss, 4, 0.1)

    @pytest.mark.parametrize("learning_rate", [0.0, -0.1, float("nan")])
    def test_a_non_positive_or_non_finite_rate_is_refused(
        self, learning_rate: float
    ) -> None:
        with pytest.raises(InvalidValuesError):
            EpochRecord(1, 1.0, 4, learning_rate)


class TestTrainingHistory:
    def test_fell_when_the_last_loss_is_below_the_first(self) -> None:
        history = TrainingHistory(
            [
                EpochRecord(1, 3.0, 4, 0.1),
                EpochRecord(2, 3.5, 4, 0.1),
                EpochRecord(3, 2.0, 4, 0.1),
            ]
        )

        assert history.fell
        assert history.mean_losses == (3.0, 3.5, 2.0)
        assert history.n_epochs == 3

    def test_did_not_fall_when_the_last_loss_is_not_below_the_first(self) -> None:
        assert not TrainingHistory(
            [EpochRecord(1, 2.0, 4, 0.1), EpochRecord(2, 2.0, 4, 0.1)]
        ).fell

    def test_a_single_epoch_did_not_fall(self) -> None:
        assert not TrainingHistory([EpochRecord(1, 2.0, 4, 0.1)]).fell

    def test_iterates_and_indexes_its_records(self) -> None:
        records = [EpochRecord(1, 3.0, 4, 0.1), EpochRecord(2, 2.0, 4, 0.05)]
        history = TrainingHistory(records)

        assert list(history) == records
        assert history[1] == records[1]
        assert len(history) == 2

    def test_misnumbered_epochs_are_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            TrainingHistory([EpochRecord(1, 3.0, 4, 0.1), EpochRecord(3, 2.0, 4, 0.1)])
        with pytest.raises(InvalidValuesError):
            TrainingHistory([EpochRecord(2, 3.0, 4, 0.1)])

    def test_an_empty_history_is_allowed_and_did_not_fall(self) -> None:
        history = TrainingHistory([])

        assert history.n_epochs == 0
        assert not history.fell

    def test_compares_by_value(self) -> None:
        records = [EpochRecord(1, 3.0, 4, 0.1)]

        assert TrainingHistory(records) == TrainingHistory(records)
        assert TrainingHistory(records) != TrainingHistory([])
        assert TrainingHistory(records) != records
