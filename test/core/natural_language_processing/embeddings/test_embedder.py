"""Spec for TokenisedCorpus and the WordEmbedder frame."""

from collections.abc import Sequence
from typing import Self

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
from oop_ml.core.natural_language_processing.embeddings.embedder import (
    TokenisedCorpus,
    WordEmbedder,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (  # noqa: E501
    WhitespacePreTokenizer,
)

CORPUS = ["the cat sat on the mat", "the dog sat", "a cat"]


def tokenised() -> TokenisedCorpus:
    return TokenisedCorpus.from_texts(CORPUS, WhitespacePreTokenizer())


class OneHotEmbedder(WordEmbedder):
    """The smallest concrete embedder: each word is its own axis."""

    def fit(self, corpus: Sequence[str]) -> Self:
        vocabulary = self._tokenised(corpus).vocabulary(self.minimum_count)
        self._embeddings = WordEmbeddings(vocabulary, np.eye(vocabulary.n_tokens))
        self._mark_fitted()
        return self

    @property
    def embeddings(self) -> WordEmbeddings:
        self._check_fitted()
        return self._embeddings


class TestTokenisedCorpus:
    def test_splits_every_text_in_order(self):
        assert tokenised().sentences == (
            ("the", "cat", "sat", "on", "the", "mat"),
            ("the", "dog", "sat"),
            ("a", "cat"),
        )

    def test_counts_words_across_texts(self):
        corpus = tokenised()

        assert corpus.word_counts["the"] == 3
        assert corpus.word_counts["cat"] == 2
        assert corpus.n_words == 11
        assert corpus.n_sentences == 3
        assert len(corpus) == 3

    def test_a_blank_text_is_kept_as_an_empty_sentence(self):
        corpus = TokenisedCorpus.from_texts(["a b", "   "], WhitespacePreTokenizer())

        assert corpus.sentences == (("a", "b"), ())

    def test_a_single_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            TokenisedCorpus("the cat")  # type: ignore[arg-type]

    def test_a_sentence_that_is_one_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            TokenisedCorpus(["the cat", ["a", "b"]])

    def test_an_empty_word_is_refused(self):
        with pytest.raises(InvalidValuesError):
            TokenisedCorpus([["a", ""]])

    def test_no_sentences_raise(self):
        with pytest.raises(EmptyValuesError):
            TokenisedCorpus([])

    def test_no_words_anywhere_raise(self):
        with pytest.raises(EmptyValuesError):
            TokenisedCorpus([[], []])


class TestVocabularyOrder:
    def test_commonest_first_then_alphabetical(self):
        assert list(tokenised().vocabulary()) == [
            "the",
            "cat",
            "sat",
            "a",
            "dog",
            "mat",
            "on",
        ]

    def test_minimum_count_drops_rare_words(self):
        assert list(tokenised().vocabulary(minimum_count=2)) == ["the", "cat", "sat"]

    def test_has_no_unknown_token(self):
        assert tokenised().vocabulary().unknown_token is None

    def test_a_minimum_nothing_reaches_raises(self):
        with pytest.raises(TooFewValuesError):
            tokenised().vocabulary(minimum_count=4)


class TestIdSequences:
    def test_maps_words_to_positions(self):
        corpus = tokenised()
        vocabulary = corpus.vocabulary()

        assert corpus.id_sequences(vocabulary)[1] == tuple(
            vocabulary.id_of(word) for word in ("the", "dog", "sat")
        )

    def test_rare_words_are_dropped_so_their_neighbours_touch(self):
        corpus = tokenised()
        vocabulary = corpus.vocabulary(minimum_count=2)

        assert corpus.id_sequences(vocabulary)[0] == tuple(
            vocabulary.id_of(word) for word in ("the", "cat", "sat", "the")
        )

    def test_a_sentence_of_only_rare_words_becomes_empty(self):
        corpus = tokenised()

        assert corpus.id_sequences(Vocabulary(["the"]))[2] == ()


class TestWordEmbedderFrame:
    def test_fit_then_ask(self):
        embedder = OneHotEmbedder().fit(CORPUS)

        assert embedder.vocabulary.n_tokens == 7
        assert embedder.vector_of("the").values[0] == 1.0
        assert embedder.similarity("the", "cat") == 0.0
        assert embedder.most_similar("the", n_results=2).n_words == 2

    def test_minimum_count_reaches_the_vocabulary(self):
        assert OneHotEmbedder(minimum_count=2).fit(CORPUS).vocabulary.n_tokens == 3

    def test_a_dropped_word_has_no_vector(self):
        with pytest.raises(UnknownTokenError):
            OneHotEmbedder(minimum_count=2).fit(CORPUS).vector_of("dog")

    def test_before_fit_raises_not_fitted(self):
        with pytest.raises(NotFittedError):
            _ = OneHotEmbedder().embeddings
        with pytest.raises(NotFittedError):
            OneHotEmbedder().vector_of("the")

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            OneHotEmbedder().fit("the cat")  # type: ignore[arg-type]

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            OneHotEmbedder(min_count=2)  # type: ignore[call-arg]

    def test_the_base_cannot_be_constructed(self):
        with pytest.raises(TypeError):
            WordEmbedder()  # type: ignore[abstract]
