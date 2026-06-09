"""Spec for PatternPreTokenizer and the two patterns exported beside it."""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.word_level.pattern import (
    GPT2_PATTERN,
    WORD_OR_PUNCTUATION_PATTERN,
    PatternPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word


def slices_reproduce_the_source(splitter: PatternPreTokenizer, text: str) -> bool:
    """A pattern splitter never rewrites, so every word is its own source slice."""
    return all(
        text[word.start : word.end] == word.text for word in splitter.split(text)
    )


class TestPatternPreTokenizer:
    def test_every_non_empty_match_is_a_word(self):
        assert PatternPreTokenizer(pattern=r"\d+").split("a1b22c").texts == ("1", "22")

    def test_reports_where_each_word_was(self):
        words = PatternPreTokenizer(pattern=r"\d+").split("a1b22c")

        assert list(words) == [Word("1", 1, 2), Word("22", 3, 5)]

    def test_the_gaps_between_matches_are_not_words(self):
        assert PatternPreTokenizer(pattern=r"[a-z]+").split("A dog, a cat.").texts == (
            "dog",
            "a",
            "cat",
        )

    def test_empty_matches_are_skipped_rather_than_refused(self):
        """``\\w*`` matches the empty string between words; a Word cannot hold nothing."""
        assert PatternPreTokenizer(pattern=r"\w*").split("a b").texts == ("a", "b")

    @pytest.mark.parametrize(
        ("pattern", "text"),
        [
            (r"\S+", "the  quick fox"),
            (WORD_OR_PUNCTUATION_PATTERN, "Hello, world! 3.14"),
            (GPT2_PATTERN, "Hello world's fun"),
        ],
    )
    def test_offsets_reproduce_the_source_slice(self, pattern, text):
        assert slices_reproduce_the_source(PatternPreTokenizer(pattern=pattern), text)

    def test_the_pattern_is_kept_as_written(self):
        assert PatternPreTokenizer(pattern=r"\S+").pattern == r"\S+"

    @pytest.mark.parametrize("bad_pattern", ["(", "[a-", r"\p{L}+", "*"])
    def test_an_invalid_regular_expression_is_refused_at_construction(
        self, bad_pattern
    ):
        with pytest.raises(ValidationError):
            PatternPreTokenizer(pattern=bad_pattern)

    def test_an_empty_pattern_is_refused(self):
        with pytest.raises(ValidationError):
            PatternPreTokenizer(pattern="")

    def test_the_pattern_is_required(self):
        with pytest.raises(ValidationError):
            PatternPreTokenizer()  # type: ignore[call-arg]

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            PatternPreTokenizer(pattern=r"\S+", flags=0)  # type: ignore[call-arg]

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            PatternPreTokenizer(pattern=r"\S+").split(b"bytes")  # type: ignore[arg-type]

    def test_an_empty_text_has_no_words(self):
        assert PatternPreTokenizer(pattern=r"\S+").split("").texts == ()

    def test_a_text_nothing_matches_has_no_words(self):
        assert PatternPreTokenizer(pattern=r"\d+").split("no digits").texts == ()


class TestWordOrPunctuationPattern:
    @pytest.fixture
    def splitter(self) -> PatternPreTokenizer:
        return PatternPreTokenizer(pattern=WORD_OR_PUNCTUATION_PATTERN)

    def test_separates_punctuation_from_the_word_it_touches(self, splitter):
        assert splitter.split("Hello, world!").texts == ("Hello", ",", "world", "!")

    def test_a_run_of_punctuation_is_one_word(self, splitter):
        assert splitter.split("wait...").texts == ("wait", "...")

    def test_knows_nothing_about_numbers(self, splitter):
        """The limitation the Treebank and Unicode rules exist to repair."""
        assert splitter.split("3.14").texts == ("3", ".", "14")

    def test_underscore_is_a_word_character(self, splitter):
        assert splitter.split("snake_case").texts == ("snake_case",)


class TestGpt2Pattern:
    @pytest.fixture
    def splitter(self) -> PatternPreTokenizer:
        return PatternPreTokenizer(pattern=GPT2_PATTERN)

    def test_pins_the_worked_example(self, splitter):
        assert splitter.split("Hello world's fun").texts == (
            "Hello",
            " world",
            "'s",
            " fun",
        )

    def test_the_worked_example_offsets_tile_the_source(self, splitter):
        text = "Hello world's fun"
        words = list(splitter.split(text))

        assert words == [
            Word("Hello", 0, 5),
            Word(" world", 5, 11),
            Word("'s", 11, 13),
            Word(" fun", 13, 17),
        ]
        assert "".join(text[word.start : word.end] for word in words) == text

    def test_a_word_carries_its_leading_space(self, splitter):
        assert splitter.split("a b").texts == ("a", " b")

    def test_a_run_of_whitespace_leaves_its_last_space_to_the_word_that_follows(
        self, splitter
    ):
        """``\\s+(?!\\S)`` stops one short of a non-space, which the next branch claims."""
        assert splitter.split("  two  spaces ").texts == (
            " ",
            " two",
            " ",
            " spaces",
            " ",
        )

    def test_trailing_whitespace_is_one_word(self, splitter):
        assert splitter.split("x  ").texts == ("x", "  ")

    def test_the_seven_clitics_are_words_of_their_own(self, splitter):
        assert splitter.split("don't I'll we've they're I'm he'd it's").texts == (
            "don",
            "'t",
            " I",
            "'ll",
            " we",
            "'ve",
            " they",
            "'re",
            " I",
            "'m",
            " he",
            "'d",
            " it",
            "'s",
        )

    def test_letters_and_digits_are_separate_words(self, splitter):
        assert splitter.split("abc123").texts == ("abc", "123")

    def test_a_run_of_punctuation_is_one_word_with_its_leading_space(self, splitter):
        assert splitter.split("Hello, world!?").texts == ("Hello", ",", " world", "!?")

    def test_underscore_is_punctuation_as_in_the_original(self, splitter):
        """``\\p{L}`` excludes it, so the translation must not let ``\\w`` admit it."""
        assert splitter.split("snake_case").texts == ("snake", "_", "case")

    def test_each_newline_is_its_own_word(self, splitter):
        assert splitter.split("a\n\nb").texts == ("a", "\n", "\n", "b")

    def test_a_vulgar_fraction_is_a_letter_here_and_a_number_to_the_original(
        self, splitter
    ):
        """``½`` is General Category No: ``\\p{N}`` to GPT-2, so ``1½`` is one number
        run there, but ``\\d`` is Nd only and ``\\w`` admits No as a letter, so it is
        two words here. The difference is pinned rather than hidden."""
        assert splitter.split("1½").texts == ("1", "½")

    def test_a_roman_numeral_joins_the_letters_before_it(self, splitter):
        """``Ⅻ`` is Nl: a number to the original, which would cut ``a`` from it."""
        assert splitter.split("aⅫ").texts == ("aⅫ",)

    def test_arabic_indic_digits_are_numbers_on_both_readings(self, splitter):
        """Category Nd, so ``\\d`` and ``\\p{N}`` agree."""
        assert splitter.split("a٣٤").texts == ("a", "٣٤")
