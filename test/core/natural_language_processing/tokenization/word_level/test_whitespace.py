"""Spec for WhitespacePreTokenizer -- the baseline every other rule is measured against."""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word


class TestWhitespacePreTokenizer:
    def test_splits_on_runs_of_spaces(self):
        assert WhitespacePreTokenizer().split("the  quick fox").texts == (
            "the",
            "quick",
            "fox",
        )

    def test_reports_where_each_word_was(self):
        words = WhitespacePreTokenizer().split(" the fox")

        assert list(words) == [Word("the", 1, 4), Word("fox", 5, 8)]

    def test_punctuation_stays_attached(self):
        """The limitation the rest of the family exists to repair."""
        assert WhitespacePreTokenizer().split("Hello, world.").texts == (
            "Hello,",
            "world.",
        )

    @pytest.mark.parametrize("separator", ["\t", "\n", " ", "　", " "])
    def test_any_unicode_whitespace_ends_a_word(self, separator):
        assert WhitespacePreTokenizer().split(f"a{separator}b").texts == ("a", "b")

    def test_a_blank_text_has_no_words(self):
        assert WhitespacePreTokenizer().split("   ").texts == ()

    def test_an_empty_text_has_no_words(self):
        assert WhitespacePreTokenizer().split("").texts == ()

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            WhitespacePreTokenizer().split(b"bytes")  # type: ignore[arg-type]

    def test_takes_no_parameters(self):
        with pytest.raises(ValidationError):
            WhitespacePreTokenizer(pattern=r"\s")  # type: ignore[call-arg]
