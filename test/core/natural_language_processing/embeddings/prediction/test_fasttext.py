"""Spec for FastText -- word2vec whose words are sums over hashed character n-grams.

Three things carry this module. The pieces and the hash are pinned against
published values: the paper's own five n-grams of ``<where>`` at length 3, and
the FNV-1a test vectors ``0x811c9dc5`` for the empty string, ``0xe40c292c`` for
``a`` and ``0xbf9cf968`` for ``foobar``. The reduction to word2vec is pinned
exactly: with a minimum length no wrapped word reaches, the fit is the parent's
to the last bit, table, output rows and every epoch's loss, so everything this
subclass adds is in its pieces and nothing has leaked into the loop. And the
point of the model is pinned on a word the corpus never holds: ``playing`` is
absent from every sentence, its pieces are all present in other verb forms,
and its vector lands at a mean cosine of 0.997 to the twenty verb forms and
0.100 to the eleven finance nouns.

No finite-difference check appears here because the objective is word2vec's,
already pinned in ``test_objectives.py``; the composition's gradient is the
sum's, which is the identity on each part.
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
    UndefinedMetricError,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.fasttext import (
    FOWLER_NOLL_VO_OFFSET_BASIS,
    FastText,
    fowler_noll_vo_hash,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.word2vec import (
    Word2Vec,
    Word2VecArchitecture,
    Word2VecObjective,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    WordEmbeddings,
    WordVector,
    cosine_similarity,
)
from test.core.natural_language_processing.embeddings.prediction.corpora import (
    FINANCE_WORDS,
    TWO_TOPIC_CORPUS,
    UNSEEN_VERB,
    VERB_WORDS,
)

PLAY_FORMS = {"play", "plays", "player", "played"}

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
    learning_rate: float = 0.025,
    epochs: int = 5,
) -> FastText:
    return FastText(
        dimension=12,
        window=3,
        architecture=architecture,
        objective=objective,
        epochs=epochs,
        learning_rate=learning_rate,
        random_seed=0,
    )


def plain_model(
    architecture: Word2VecArchitecture, objective: Word2VecObjective, epochs: int
) -> Word2Vec:
    return Word2Vec(
        dimension=12,
        window=3,
        architecture=architecture,
        objective=objective,
        epochs=epochs,
        random_seed=0,
    )


@pytest.fixture(scope="module")
def fitted() -> FastText:
    return two_topic_model().fit(TWO_TOPIC_CORPUS)


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


def mean_similarity_to(
    vector: WordVector, model: FastText, words: tuple[str, ...]
) -> float:
    return float(
        np.mean(
            [
                cosine_similarity(vector.values, model.vector_of(word).values)
                for word in words
            ]
        )
    )


class TestHash:
    def test_the_hash_of_nothing_is_the_offset_basis(self) -> None:
        assert fowler_noll_vo_hash("") == 0x811C9DC5 == FOWLER_NOLL_VO_OFFSET_BASIS

    def test_the_hash_of_a(self) -> None:
        assert fowler_noll_vo_hash("a") == 0xE40C292C

    def test_the_hash_of_foobar(self) -> None:
        """The third published FNV-1a 32-bit test vector."""
        assert fowler_noll_vo_hash("foobar") == 0xBF9CF968

    def test_the_hash_reads_utf8_bytes(self) -> None:
        """A three-byte character hashes as its three bytes: computed once by an
        independent reduce over ``"零".encode()``, and pinned."""
        assert fowler_noll_vo_hash("零") == 0x6450FF61
        assert fowler_noll_vo_hash("é") != fowler_noll_vo_hash("e")

    @pytest.mark.parametrize("text", ["", "a", "<wh", "where>", "零", "a" * 100])
    def test_the_hash_is_thirty_two_bit(self, text: str) -> None:
        assert 0 <= fowler_noll_vo_hash(text) < 2**32


class TestPieces:
    def test_the_papers_example_where_at_length_three(self) -> None:
        model = FastText(minimum_n_gram_length=3, maximum_n_gram_length=3)

        assert model.n_grams_of("where") == ("<wh", "whe", "her", "ere", "re>")

    def test_where_has_fourteen_pieces_at_the_default_lengths(self) -> None:
        """Five of length 3, four of 4, three of 5, two of 6; the seven-character
        ``<where>`` itself is not one."""
        pieces = FastText().n_grams_of("where")

        assert len(pieces) == 14
        assert "<where>" not in pieces
        assert pieces[:5] == ("<wh", "whe", "her", "ere", "re>")
        assert pieces[-2:] == ("<where", "where>")

    def test_pieces_come_shortest_first_and_left_to_right(self) -> None:
        model = FastText(minimum_n_gram_length=3, maximum_n_gram_length=4)

        assert model.n_grams_of("where") == (
            "<wh",
            "whe",
            "her",
            "ere",
            "re>",
            "<whe",
            "wher",
            "here",
            "ere>",
        )

    def test_the_wrapped_word_itself_is_never_a_piece(self) -> None:
        """``<at>`` has four characters, inside the default lengths, and is still
        not a piece: the word row already stands for the whole word."""
        assert FastText().n_grams_of("at") == ("<at", "at>")

    def test_a_one_character_word_has_no_piece_at_the_defaults(self) -> None:
        assert FastText().n_grams_of("a") == ()
        assert FastText().n_grams_of("零") == ()

    def test_a_repeated_piece_counts_once(self) -> None:
        model = FastText(minimum_n_gram_length=3, maximum_n_gram_length=3)

        assert model.n_grams_of("aaaa") == ("<aa", "aaa", "aa>")

    def test_the_pieces_are_a_function_of_the_configuration_alone(self) -> None:
        unfitted = FastText()

        assert unfitted.n_grams_of("playing") == unfitted.n_grams_of("playing")
        assert len(unfitted.n_grams_of("playing")) == 22

    def test_ids_are_the_sorted_buckets_of_the_pieces(self) -> None:
        """Each of the five pieces of ``<where>`` hashed and taken modulo 2000,
        computed once independently and pinned."""
        model = FastText(minimum_n_gram_length=3, maximum_n_gram_length=3)

        assert model.n_gram_ids_of("where") == (941, 1033, 1420, 1498, 1652)

    def test_a_collision_puts_the_bucket_in_once_per_piece(self) -> None:
        model = FastText(minimum_n_gram_length=3, maximum_n_gram_length=3, n_buckets=1)

        assert model.n_gram_ids_of("where") == (0, 0, 0, 0, 0)

    @pytest.mark.parametrize("n_buckets", [1, 7, 2000])
    def test_ids_lie_inside_the_buckets(self, n_buckets: int) -> None:
        model = FastText(n_buckets=n_buckets)

        for word in (*VERB_WORDS, *FINANCE_WORDS, UNSEEN_VERB):
            assert all(0 <= bucket < n_buckets for bucket in model.n_gram_ids_of(word))
            assert len(model.n_gram_ids_of(word)) == len(model.n_grams_of(word))

    def test_an_empty_word_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            FastText().n_grams_of("")
        with pytest.raises(EmptyValuesError):
            FastText().n_gram_ids_of("")

    def test_a_non_string_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            FastText().n_grams_of(3)  # type: ignore[arg-type]


class TestFit:
    @pytest.mark.parametrize("combination", COMBINATIONS, ids=COMBINATION_IDS)
    def test_separates_the_two_topic_corpus(
        self, combination: tuple[Word2VecArchitecture, Word2VecObjective]
    ) -> None:
        """Inherited from word2vec, and at the rate its spec explains."""
        architecture, objective = combination
        model = two_topic_model(architecture, objective, learning_rate=0.05).fit(
            TWO_TOPIC_CORPUS
        )

        assert mean_within_topic_similarity(
            model.embeddings
        ) > mean_across_topic_similarity(model.embeddings)
        assert model.history.fell

    def test_the_loss_fell(self, fitted: FastText) -> None:
        assert fitted.history.fell
        assert fitted.history.n_epochs == 5

    def test_the_table_is_one_row_per_word_of_the_stated_dimension(
        self, fitted: FastText
    ) -> None:
        assert fitted.embeddings.table.shape == (31, 12)
        assert set(fitted.vocabulary) == set(VERB_WORDS) | set(FINANCE_WORDS)

    def test_the_same_seed_reproduces_the_fit(self, fitted: FastText) -> None:
        again = two_topic_model().fit(TWO_TOPIC_CORPUS)

        assert again.embeddings == fitted.embeddings
        assert again.history == fitted.history
        assert np.array_equal(again.bucket_vectors, fitted.bucket_vectors)

    def test_fit_returns_self(self) -> None:
        model = FastText(dimension=4, epochs=1, n_buckets=10)

        assert model.fit(["abc bcd", "bcd abc"]) is model


class TestComposition:
    def test_every_word_vector_is_its_row_plus_its_bucket_rows(
        self, fitted: FastText
    ) -> None:
        """Two routes to one table: the fitted embeddings, and the two frozen
        parts recombined by hand through the public bucket ids. Measured gap 0.0."""
        for word_id, word in enumerate(fitted.vocabulary):
            buckets = list(fitted.n_gram_ids_of(word))
            composed = fitted.word_row_vectors[word_id] + fitted.bucket_vectors[
                buckets
            ].sum(axis=0)

            assert np.allclose(
                fitted.embeddings.vector_of(word).values, composed, atol=1e-12
            )

    def test_vector_of_unseen_on_a_seen_word_omits_its_row(
        self, fitted: FastText
    ) -> None:
        word_id = fitted.vocabulary.id_of("played")

        assert np.allclose(
            fitted.vector_of("played").values
            - fitted.vector_of_unseen("played").values,
            fitted.word_row_vectors[word_id],
            atol=1e-12,
        )
        assert fitted.vector_of("played") != fitted.vector_of_unseen("played")

    def test_the_bucket_table_is_frozen_and_shaped_by_the_configuration(
        self, fitted: FastText
    ) -> None:
        assert fitted.bucket_vectors.shape == (2000, 12)
        assert not fitted.bucket_vectors.flags.writeable

    def test_the_word_rows_are_frozen_and_one_per_word(self, fitted: FastText) -> None:
        assert fitted.word_row_vectors.shape == (31, 12)
        assert not fitted.word_row_vectors.flags.writeable

    def test_a_bucket_no_word_owns_stays_at_zero_and_an_owned_one_moved(
        self, fitted: FastText
    ) -> None:
        owned = {
            bucket
            for word in fitted.vocabulary
            for bucket in fitted.n_gram_ids_of(word)
        }
        unowned = sorted(set(range(2000)) - owned)

        assert 0 < len(owned) < 2000
        assert not np.any(fitted.bucket_vectors[unowned])
        assert np.all(np.any(fitted.bucket_vectors[sorted(owned)] != 0.0, axis=1))

    def test_the_words_share_pieces_and_buckets_as_measured(
        self, fitted: FastText
    ) -> None:
        """Thirty-one words, 279 distinct pieces, 260 distinct buckets: nineteen
        pieces already share a row at two thousand buckets."""
        pieces = {
            piece for word in fitted.vocabulary for piece in fitted.n_grams_of(word)
        }
        buckets = {
            bucket
            for word in fitted.vocabulary
            for bucket in fitted.n_gram_ids_of(word)
        }

        assert len(pieces) == 279
        assert len(buckets) == 260


class TestUnseenWords:
    def test_vector_of_an_unseen_word_does_not_raise_and_is_its_pieces(
        self, fitted: FastText
    ) -> None:
        assert UNSEEN_VERB not in fitted.vocabulary
        assert fitted.vector_of(UNSEEN_VERB) == fitted.vector_of_unseen(UNSEEN_VERB)
        assert fitted.vector_of(UNSEEN_VERB).word == UNSEEN_VERB
        assert fitted.vector_of(UNSEEN_VERB).dimension == 12

    def test_vector_of_unseen_is_the_sum_of_its_bucket_rows(
        self, fitted: FastText
    ) -> None:
        buckets = list(fitted.n_gram_ids_of(UNSEEN_VERB))

        assert np.allclose(
            fitted.vector_of_unseen(UNSEEN_VERB).values,
            fitted.bucket_vectors[buckets].sum(axis=0),
            atol=1e-12,
        )

    def test_playing_is_nearer_the_verb_forms_than_the_finance_nouns(
        self, fitted: FastText
    ) -> None:
        """Measured 0.997 against 0.100 at these settings."""
        unseen = fitted.vector_of(UNSEEN_VERB)

        to_verbs = mean_similarity_to(unseen, fitted, VERB_WORDS)
        to_finance = mean_similarity_to(unseen, fitted, FINANCE_WORDS)

        assert to_verbs > 0.9
        assert to_finance < 0.3
        assert to_verbs > to_finance

    def test_the_nearest_words_to_playing_are_the_play_forms(
        self, fitted: FastText
    ) -> None:
        assert set(fitted.most_similar(UNSEEN_VERB, 4).words) == PLAY_FORMS

    def test_most_similar_on_an_unseen_word_is_similar_to_vector_of_its_pieces(
        self, fitted: FastText
    ) -> None:
        assert fitted.most_similar(
            UNSEEN_VERB, 5
        ) == fitted.embeddings.similar_to_vector(
            fitted.vector_of_unseen(UNSEEN_VERB).values, 5
        )

    def test_similarity_accepts_an_unseen_word(self, fitted: FastText) -> None:
        assert fitted.similarity(UNSEEN_VERB, "played") > 0.9
        assert fitted.similarity(UNSEEN_VERB, "played") > fitted.similarity(
            UNSEEN_VERB, "stock"
        )
        assert fitted.similarity(UNSEEN_VERB, UNSEEN_VERB) == pytest.approx(1.0)

    def test_similarity_between_seen_words_is_the_embeddings_own(
        self, fitted: FastText
    ) -> None:
        assert fitted.similarity("play", "stock") == fitted.embeddings.similarity(
            "play", "stock"
        )

    def test_a_seen_word_is_excluded_from_its_own_neighbours(
        self, fitted: FastText
    ) -> None:
        assert "play" not in fitted.most_similar("play", 30).words
        assert fitted.most_similar("play", 3) == fitted.embeddings.most_similar(
            "play", 3
        )

    def test_a_word_sharing_no_bucket_has_the_zero_vector_and_no_neighbours(
        self, fitted: FastText
    ) -> None:
        """``zap``'s five pieces land in buckets no word of the corpus owns, so
        nothing ever moved them from zero, and a zero vector has no direction."""
        owned = {
            bucket
            for word in fitted.vocabulary
            for bucket in fitted.n_gram_ids_of(word)
        }

        assert fitted.n_gram_ids_of("zap") == (438, 614, 898, 1260, 1422)
        assert not owned & set(fitted.n_gram_ids_of("zap"))
        assert not np.any(fitted.vector_of("zap").values)
        with pytest.raises(UndefinedMetricError):
            fitted.most_similar("zap")
        with pytest.raises(UndefinedMetricError):
            fitted.similarity("zap", "play")

    def test_an_empty_word_is_refused(self, fitted: FastText) -> None:
        with pytest.raises(EmptyValuesError):
            fitted.vector_of("")
        with pytest.raises(EmptyValuesError):
            fitted.vector_of_unseen("")


class TestReductionToWord2Vec:
    @pytest.mark.parametrize(
        "combination",
        [
            (Word2VecArchitecture.SKIP_GRAM, Word2VecObjective.NEGATIVE_SAMPLING),
            (
                Word2VecArchitecture.CONTINUOUS_BAG_OF_WORDS,
                Word2VecObjective.HIERARCHICAL_SOFTMAX,
            ),
        ],
        ids=[
            "skip_gram-negative_sampling",
            "continuous_bag_of_words-hierarchical_softmax",
        ],
    )
    def test_with_no_pieces_the_fit_is_word2vecs_exactly(
        self, combination: tuple[Word2VecArchitecture, Word2VecObjective]
    ) -> None:
        """A minimum length of 50 exceeds every wrapped word, so no word has a
        piece, the buckets never move, and nothing this subclass adds may show.
        Measured gap 0.0 under all four combinations; pinned at 1e-12 on two."""
        architecture, objective = combination
        reduced = FastText(
            dimension=12,
            window=3,
            architecture=architecture,
            objective=objective,
            epochs=2,
            random_seed=0,
            minimum_n_gram_length=50,
            maximum_n_gram_length=50,
        ).fit(TWO_TOPIC_CORPUS)
        plain = plain_model(architecture, objective, epochs=2).fit(TWO_TOPIC_CORPUS)

        assert all(reduced.n_grams_of(word) == () for word in reduced.vocabulary)
        assert np.allclose(reduced.embeddings.table, plain.embeddings.table, atol=1e-12)
        assert np.array_equal(reduced.output_vectors, plain.output_vectors)
        assert reduced.history == plain.history
        assert reduced.vocabulary == plain.vocabulary
        assert not np.any(reduced.bucket_vectors)


class TestConstruction:
    def test_a_maximum_below_the_minimum_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FastText(minimum_n_gram_length=4, maximum_n_gram_length=3)

    def test_equal_lengths_are_allowed(self) -> None:
        model = FastText(minimum_n_gram_length=4, maximum_n_gram_length=4)

        assert model.n_grams_of("where") == ("<whe", "wher", "here", "ere>")

    @pytest.mark.parametrize(
        "keywords",
        [
            {"minimum_n_gram_length": 0},
            {"maximum_n_gram_length": 0},
            {"n_buckets": 0},
            {"dimension": 0},
            {"learning_rate": 0.01, "minimum_learning_rate": 0.02},
        ],
    )
    def test_out_of_range_hyperparameters_are_refused(
        self, keywords: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            FastText(**keywords)

    def test_an_unknown_keyword_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FastText(bucket=2000)  # type: ignore[call-arg]

    def test_the_defaults_are_the_papers_lengths_and_a_small_table(self) -> None:
        model = FastText()

        assert model.minimum_n_gram_length == 3
        assert model.maximum_n_gram_length == 6
        assert model.n_buckets == 2000
        assert isinstance(model, Word2Vec)


class TestNotFitted:
    def test_everything_learned_raises_not_fitted(self) -> None:
        model = FastText(dimension=4)

        with pytest.raises(NotFittedError):
            _ = model.bucket_vectors
        with pytest.raises(NotFittedError):
            _ = model.word_row_vectors
        with pytest.raises(NotFittedError):
            model.vector_of_unseen("playing")
        with pytest.raises(NotFittedError):
            model.vector_of("playing")
        with pytest.raises(NotFittedError):
            model.similarity("playing", "played")
        with pytest.raises(NotFittedError):
            model.most_similar("playing")
        with pytest.raises(NotFittedError):
            _ = model.embeddings
