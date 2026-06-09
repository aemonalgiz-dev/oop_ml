"""Spec for Corpus / WordCounts -- the coercion boundary for text."""

import pytest

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.corpus import (
    Corpus,
    WordCount,
    WordCounts,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.words import Words
from test.core.natural_language_processing.fixtures import (
    SENNRICH_CORPUS,
    SENNRICH_WORD_COUNTS,
)


class FindsNothing(PreTokenizer):
    def _words_of(self, text: str) -> Words:
        return Words([])


class TestCorpus:
    def test_holds_its_texts_in_order(self):
        corpus = Corpus(["first", "second"])

        assert list(corpus) == ["first", "second"]
        assert corpus.n_texts == 2
        assert len(corpus) == 2
        assert corpus[1] == "second"

    def test_of_is_idempotent(self):
        corpus = Corpus(["first"])

        assert Corpus.of(corpus) is corpus

    def test_of_wraps_a_plain_sequence(self):
        assert isinstance(Corpus.of(["first"]), Corpus)

    def test_a_single_string_is_refused(self):
        """It would iterate as characters and fit a vocabulary of letters."""
        with pytest.raises(InvalidValuesError):
            Corpus("one string")  # type: ignore[arg-type]

    def test_empty_raises(self):
        with pytest.raises(EmptyValuesError):
            Corpus([])

    def test_all_blank_raises(self):
        with pytest.raises(EmptyValuesError):
            Corpus(["", "   ", "\n"])

    def test_a_non_string_entry_raises(self):
        with pytest.raises(InvalidValuesError):
            Corpus(["fine", 3])  # type: ignore[list-item]

    def test_one_blank_among_real_texts_is_allowed(self):
        assert Corpus(["", "real"]).n_texts == 2


class TestWordCounts:
    def test_counts_every_distinct_word_across_the_texts(self):
        counts = Corpus(SENNRICH_CORPUS).word_counts(WhitespacePreTokenizer())

        assert {word_count.word: word_count.count for word_count in counts} == (
            SENNRICH_WORD_COUNTS
        )

    def test_total_is_the_number_of_occurrences(self):
        counts = Corpus(SENNRICH_CORPUS).word_counts(WhitespacePreTokenizer())

        assert counts.total == 16
        assert counts.n_words == 4
        assert len(counts) == 4

    def test_words_are_in_first_seen_order(self):
        counts = Corpus(["b a b", "c"]).word_counts(WhitespacePreTokenizer())

        assert counts.words == ("b", "a", "c")

    def test_an_absent_word_counts_zero(self):
        counts = Corpus(["a"]).word_counts(WhitespacePreTokenizer())

        assert counts.count_of("b") == 0
        assert counts["a"] == 1
        assert "a" in counts
        assert "b" not in counts

    def test_a_pre_tokenizer_finding_no_words_raises(self):
        with pytest.raises(EmptyValuesError):
            Corpus(["some text"]).word_counts(FindsNothing())

    def test_nothing_is_lower_cased(self):
        counts = Corpus(["The the"]).word_counts(WhitespacePreTokenizer())

        assert counts["The"] == 1
        assert counts["the"] == 1

    def test_a_word_counted_twice_raises(self):
        with pytest.raises(InvalidValuesError):
            WordCounts([WordCount("a", 1), WordCount("a", 2)])

    def test_empty_counts_raise(self):
        with pytest.raises(EmptyValuesError):
            WordCounts([])


class TestWordCount:
    def test_carries_word_and_count(self):
        word_count = WordCount("fox", 3)

        assert word_count.word == "fox"
        assert word_count.count == 3

    def test_a_count_below_one_raises(self):
        with pytest.raises(InvalidValuesError):
            WordCount("fox", 0)

    def test_an_empty_word_raises(self):
        with pytest.raises(EmptyValuesError):
            WordCount("", 1)

    def test_equal_when_word_and_count_match(self):
        assert WordCount("fox", 3) == WordCount("fox", 3)
        assert WordCount("fox", 3) != WordCount("fox", 4)
        assert WordCount("fox", 3) != "fox"
