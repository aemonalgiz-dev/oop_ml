"""Spec for Word / Words -- a word that knows where in the source it was."""

import pytest

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words


class TestWord:
    def test_carries_text_and_span(self):
        word = Word("fox", 4, 7)

        assert word.text == "fox"
        assert word.start == 4
        assert word.end == 7

    def test_of_cuts_the_slice_from_the_source(self):
        assert Word.of("the fox", 4, 7) == Word("fox", 4, 7)

    def test_text_may_differ_in_length_from_the_span(self):
        """A normalising rule may rewrite what the span stood for."""
        assert Word("``", 0, 1).text == "``"

    def test_empty_text_raises(self):
        with pytest.raises(EmptyValuesError):
            Word("", 0, 1)

    @pytest.mark.parametrize(("start", "end"), [(-1, 2), (3, 3), (4, 2)])
    def test_a_span_must_run_forward_from_a_non_negative_offset(self, start, end):
        with pytest.raises(InvalidValuesError):
            Word("fox", start, end)

    def test_equal_when_text_and_span_match(self):
        assert Word("fox", 4, 7) == Word("fox", 4, 7)
        assert hash(Word("fox", 4, 7)) == hash(Word("fox", 4, 7))

    def test_unequal_when_the_span_differs(self):
        assert Word("fox", 4, 7) != Word("fox", 5, 8)

    def test_compares_unequal_to_a_bare_string(self):
        assert Word("fox", 4, 7) != "fox"


class TestWords:
    def test_iterates_word_objects_in_order(self):
        words = Words([Word.of("the fox", 0, 3), Word.of("the fox", 4, 7)])

        assert [word.text for word in words] == ["the", "fox"]

    def test_texts_reads_the_strings_off(self):
        words = Words([Word.of("the fox", 0, 3), Word.of("the fox", 4, 7)])

        assert words.texts == ("the", "fox")

    def test_counts_and_indexes(self):
        words = Words([Word.of("the fox", 0, 3), Word.of("the fox", 4, 7)])

        assert len(words) == 2
        assert words.n_words == 2
        assert words[1] == Word("fox", 4, 7)

    def test_may_be_empty(self):
        assert len(Words([])) == 0
        assert Words([]).texts == ()

    def test_adjacent_words_are_allowed(self):
        """Punctuation split off a word starts exactly where the word ended."""
        Words([Word.of("fox.", 0, 3), Word.of("fox.", 3, 4)])

    def test_overlapping_words_raise(self):
        with pytest.raises(InvalidValuesError):
            Words([Word.of("the fox", 0, 4), Word.of("the fox", 3, 7)])

    def test_words_out_of_source_order_raise(self):
        with pytest.raises(InvalidValuesError):
            Words([Word.of("the fox", 4, 7), Word.of("the fox", 0, 3)])

    def test_equal_when_every_word_matches(self):
        first = Words([Word.of("the fox", 0, 3), Word.of("the fox", 4, 7)])
        second = Words([Word.of("the fox", 0, 3), Word.of("the fox", 4, 7)])

        assert first == second
        assert hash(first) == hash(second)

    def test_compares_unequal_to_a_bare_tuple(self):
        assert Words([Word.of("fox", 0, 3)]) != ("fox",)
