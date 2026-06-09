"""Spec for DictionaryEntry, WordDictionary and SegmentedCorpus -- what a segmenter learns from."""

import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueTokensError,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (
    DictionaryEntry,
    SegmentedCorpus,
    WordDictionary,
    checked_word,
)
from test.core.natural_language_processing.tokenization.segmentation.fixtures import (
    CJK_SEGMENTED_CORPUS,
    LIFE_FAVOURING_FREQUENCIES,
    RESEARCH_WORDS,
)


def life_dictionary() -> WordDictionary:
    return WordDictionary(
        [
            DictionaryEntry(word, frequency)
            for word, frequency in LIFE_FAVOURING_FREQUENCIES.items()
        ]
    )


class TestCheckedWord:
    def test_returns_the_word_unchanged(self):
        assert checked_word("研究", "dictionary word") == "研究"

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            checked_word(3, "dictionary word")

    def test_an_empty_word_is_refused(self):
        with pytest.raises(EmptyValuesError):
            checked_word("", "dictionary word")

    @pytest.mark.parametrize("word", ["a b", "研究 ", "\tx", "a　b"])
    def test_whitespace_inside_a_word_is_refused(self, word):
        with pytest.raises(InvalidValuesError):
            checked_word(word, "dictionary word")


class TestDictionaryEntry:
    def test_holds_word_and_frequency(self):
        entry = DictionaryEntry("研究", 10)

        assert entry.word == "研究"
        assert entry.frequency == 10

    @pytest.mark.parametrize("frequency", [0, -1])
    def test_a_frequency_below_one_is_refused(self, frequency):
        with pytest.raises(InvalidValuesError):
            DictionaryEntry("研究", frequency)

    def test_an_empty_word_is_refused(self):
        with pytest.raises(EmptyValuesError):
            DictionaryEntry("", 1)

    def test_a_word_with_whitespace_is_refused(self):
        with pytest.raises(InvalidValuesError):
            DictionaryEntry("the cat", 1)

    def test_equality_is_by_value(self):
        assert DictionaryEntry("a", 1) == DictionaryEntry("a", 1)
        assert DictionaryEntry("a", 1) != DictionaryEntry("a", 2)
        assert DictionaryEntry("a", 1) != DictionaryEntry("b", 1)
        assert hash(DictionaryEntry("a", 1)) == hash(DictionaryEntry("a", 1))

    def test_comparison_with_another_type_is_not_implemented(self):
        assert DictionaryEntry("a", 1).__eq__("a") is NotImplemented


class TestWordDictionary:
    def test_counts_and_totals(self):
        dictionary = life_dictionary()

        assert dictionary.n_words == 5
        assert len(dictionary) == 5
        assert dictionary.total_frequency == 10 + 8 + 5 + 6 + 4
        assert dictionary.longest_word_length == 3

    def test_frequency_of_an_absent_word_is_zero(self):
        dictionary = life_dictionary()

        assert dictionary.frequency_of("研究") == 10
        assert dictionary.frequency_of("研") == 0

    def test_membership(self):
        dictionary = life_dictionary()

        assert "生命" in dictionary
        assert "生" not in dictionary
        assert 3 not in dictionary

    def test_iterates_entries_in_given_order(self):
        dictionary = life_dictionary()

        assert [entry.word for entry in dictionary] == list(LIFE_FAVOURING_FREQUENCIES)
        assert dictionary.words == tuple(LIFE_FAVOURING_FREQUENCIES)
        assert all(isinstance(entry, DictionaryEntry) for entry in dictionary)

    def test_an_empty_dictionary_is_refused(self):
        with pytest.raises(EmptyValuesError):
            WordDictionary([])

    def test_a_repeated_word_is_refused_as_a_repeated_token(self):
        with pytest.raises(NonUniqueTokensError):
            WordDictionary([DictionaryEntry("a", 1), DictionaryEntry("a", 2)])

    def test_from_words_gives_every_word_frequency_one(self):
        dictionary = WordDictionary.from_words(RESEARCH_WORDS)

        assert dictionary.n_words == 5
        assert dictionary.total_frequency == 5
        assert all(entry.frequency == 1 for entry in dictionary)

    def test_from_words_refuses_a_repeat(self):
        with pytest.raises(NonUniqueTokensError):
            WordDictionary.from_words(["a", "b", "a"])

    def test_from_segmented_corpus_counts_occurrences(self):
        dictionary = WordDictionary.from_segmented_corpus(CJK_SEGMENTED_CORPUS)

        assert dictionary.frequency_of("研究") == 3
        assert dictionary.frequency_of("很") == 2
        assert dictionary.frequency_of("学生") == 2
        assert dictionary.frequency_of("多") == 1
        assert dictionary.frequency_of("研究生") == 0
        assert dictionary.total_frequency == 3 + 3 + 3 + 2 + 3
        assert dictionary.n_words == 7

    def test_from_segmented_corpus_accepts_a_corpus_object(self):
        corpus = SegmentedCorpus(CJK_SEGMENTED_CORPUS)

        assert WordDictionary.from_segmented_corpus(
            corpus
        ) == WordDictionary.from_segmented_corpus(CJK_SEGMENTED_CORPUS)

    def test_from_segmented_corpus_refuses_a_single_string(self):
        with pytest.raises(InvalidValuesError):
            WordDictionary.from_segmented_corpus("研究 生命")  # type: ignore[arg-type]

    def test_equality_is_by_entries_regardless_of_order(self):
        forward = WordDictionary([DictionaryEntry("a", 1), DictionaryEntry("b", 2)])
        reversed_order = WordDictionary(
            [DictionaryEntry("b", 2), DictionaryEntry("a", 1)]
        )

        assert forward == reversed_order
        assert hash(forward) == hash(reversed_order)
        assert forward != WordDictionary([DictionaryEntry("a", 1)])
        assert forward.__eq__("a") is NotImplemented


class TestSegmentedCorpus:
    def test_holds_sentences_as_tuples_of_words(self):
        corpus = SegmentedCorpus([["ab", "c"], ["abc"]])

        assert corpus.n_sentences == 2
        assert len(corpus) == 2
        assert corpus.n_words == 3
        assert list(corpus) == [("ab", "c"), ("abc",)]
        assert corpus[1] == ("abc",)

    def test_of_is_idempotent(self):
        corpus = SegmentedCorpus([["ab"]])

        assert SegmentedCorpus.of(corpus) is corpus
        assert list(SegmentedCorpus.of([["ab"]])) == list(corpus)

    def test_a_single_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            SegmentedCorpus("ab c")  # type: ignore[arg-type]

    def test_a_sentence_that_is_one_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            SegmentedCorpus(["ab c"])  # type: ignore[list-item]

    def test_no_sentences_is_refused(self):
        with pytest.raises(EmptyValuesError):
            SegmentedCorpus([])

    def test_an_empty_sentence_is_refused(self):
        with pytest.raises(EmptyValuesError):
            SegmentedCorpus([["ab"], []])

    def test_an_empty_word_is_refused(self):
        with pytest.raises(EmptyValuesError):
            SegmentedCorpus([["ab", ""]])

    def test_a_non_string_word_is_refused(self):
        with pytest.raises(InvalidValuesError):
            SegmentedCorpus([["ab", 3]])  # type: ignore[list-item]

    def test_a_word_containing_whitespace_is_refused(self):
        with pytest.raises(InvalidValuesError):
            SegmentedCorpus([["ab", "c d"]])
