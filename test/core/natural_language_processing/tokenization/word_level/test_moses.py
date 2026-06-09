"""Spec for MosesPreTokenizer -- Koehn et al.'s tokenizer.perl rules as a scanner.

Every expected split below was worked from the Perl's substitutions by hand
before being run; where this backend deliberately departs from the Perl (the
backtick, the built-in English prefix list) the test says so.
"""

import re

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.word_level.moses import (
    ESCAPES,
    HYPHEN_PLACEHOLDER,
    NONBREAKING_PREFIXES,
    NUMERIC_ONLY_PREFIXES,
    UNESCAPES,
    MosesPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word
from test.core.natural_language_processing.fixtures import PROSE_CORPUS

# Sentences exercising every rule at once, for the properties that must hold
# of any output: nothing lost, nothing invented, every span a real slice.
MIXED_SENTENCES: list[str] = [
    *PROSE_CORPUS,
    "Mr. Smith went to Washington.",
    "No. 5 wins... doesn't it?",
    'He said "it\'s 5,300 (or so)" -- twice!',
    "well-known students' 1990's l'homme",
]


UNESCAPE_PATTERN = re.compile("|".join(re.escape(escaped) for escaped in UNESCAPES))


def texts_of(
    text: str,
    language: str = "en",
    aggressive_hyphen_splitting: bool = False,
    escape_special_characters: bool = False,
) -> tuple[str, ...]:
    pre_tokenizer = MosesPreTokenizer(
        language=language,
        aggressive_hyphen_splitting=aggressive_hyphen_splitting,
        escape_special_characters=escape_special_characters,
    )
    return pre_tokenizer.split(text).texts


def unescaped(text: str) -> str:
    """Undo the Moses escaping in one pass, so an entity is never re-read."""
    return UNESCAPE_PATTERN.sub(lambda match: UNESCAPES[match.group()], text)


class TestConstruction:
    def test_defaults_are_english_without_hyphen_splitting_or_escaping(self):
        pre_tokenizer = MosesPreTokenizer()

        assert pre_tokenizer.language == "en"
        assert pre_tokenizer.aggressive_hyphen_splitting is False
        assert pre_tokenizer.escape_special_characters is False

    @pytest.mark.parametrize("language", ["en", "fr", "it"])
    def test_accepts_the_three_languages_with_apostrophe_rules(self, language):
        assert MosesPreTokenizer(language=language).language == language

    @pytest.mark.parametrize("language", ["de", "EN", "english", ""])
    def test_refuses_any_other_language(self, language):
        with pytest.raises(ValidationError):
            MosesPreTokenizer(language=language)

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            MosesPreTokenizer(aggressive=True)  # type: ignore[call-arg]

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            MosesPreTokenizer().split(b"bytes")  # type: ignore[arg-type]

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_a_blank_text_has_no_words(self, text):
        assert texts_of(text) == ()


class TestSpecialCharacters:
    def test_punctuation_stands_alone(self):
        assert texts_of("Hello, world!") == ("Hello", ",", "world", "!")

    def test_each_special_character_is_its_own_word_not_a_run(self):
        """The Perl pads every character individually, so ?! is two words."""
        assert texts_of("what?!") == ("what", "?", "!")

    def test_brackets_and_quotes_come_off_both_ends(self):
        assert texts_of('("word")') == ("(", '"', "word", '"', ")")

    def test_a_backtick_stands_alone_here_where_the_perl_kept_it(self):
        """A deliberate departure: the assignment's character set governs."""
        assert texts_of("`quoted`") == ("`", "quoted", "`")

    def test_a_comma_between_digits_stays_inside_the_number(self):
        assert texts_of("5,300") == ("5,300",)
        assert texts_of("1,2,3") == ("1,2,3",)

    def test_a_comma_beside_a_letter_or_at_an_edge_stands_alone(self):
        assert texts_of("a,b") == ("a", ",", "b")
        assert texts_of("5,") == ("5", ",")
        assert texts_of(",5") == (",", "5")

    def test_a_hyphen_stays_inside_by_default(self):
        assert texts_of("well-known -5 -- re-") == ("well-known", "-5", "--", "re-")

    def test_a_period_inside_a_word_stays(self):
        assert texts_of("3.14 U.S.A") == ("3.14", "U.S.A")


class TestFinalPeriods:
    def test_pins_mr_smith(self):
        assert list(MosesPreTokenizer().split("Mr. Smith went to Washington.")) == [
            Word("Mr.", 0, 3),
            Word("Smith", 4, 9),
            Word("went", 10, 14),
            Word("to", 15, 17),
            Word("Washington", 18, 28),
            Word(".", 28, 29),
        ]

    def test_pins_number_five(self):
        assert list(MosesPreTokenizer().split("No. 5 wins.")) == [
            Word("No.", 0, 3),
            Word("5", 4, 5),
            Word("wins", 6, 10),
            Word(".", 10, 11),
        ]

    @pytest.mark.parametrize("prefix", ["No", "Art"])
    def test_a_numeric_only_prefix_protects_the_period_only_before_a_digit(
        self, prefix
    ):
        assert texts_of(f"{prefix}. 5") == (f"{prefix}.", "5")
        assert texts_of(f"{prefix}. Nothing") == (prefix, ".", "Nothing")

    def test_numeric_only_prefixes_are_not_in_the_unconditional_list(self):
        assert sorted(NUMERIC_ONLY_PREFIXES) == ["Art", "No"]
        assert NUMERIC_ONLY_PREFIXES.isdisjoint(NONBREAKING_PREFIXES)

    def test_the_built_in_list_is_twenty_seven_words_and_the_capitals(self):
        assert len(NONBREAKING_PREFIXES) == 53
        assert all(
            letter in NONBREAKING_PREFIXES for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

    @pytest.mark.parametrize(
        "prefix", sorted(NONBREAKING_PREFIXES - set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    )
    def test_every_nonbreaking_prefix_keeps_its_period_before_a_capital(self, prefix):
        assert texts_of(f"{prefix}. Then") == (f"{prefix}.", "Then")

    def test_a_single_capital_keeps_its_period_and_a_lowercase_letter_does_not(self):
        """The list is case-sensitive, as the Perl's is."""
        assert texts_of("A. B.") == ("A.", "B.")
        assert texts_of("a. B") == ("a", ".", "B")

    def test_a_word_with_an_inner_period_and_a_letter_keeps_its_last(self):
        assert texts_of("U.S. law") == ("U.S.", "law")
        assert texts_of("e.g. This") == ("e.g.", "This")

    def test_an_inner_period_without_a_letter_does_not_protect_the_last(self):
        assert texts_of("3.5.") == ("3.5", ".")

    def test_a_period_stays_when_the_next_word_begins_in_lowercase(self):
        """The Perl reads a following lowercase letter as an abbreviation
        mid-sentence, whatever the word before the period was."""
        assert texts_of("went. home") == ("went.", "home")
        assert texts_of("Dr. jones") == ("Dr.", "jones")

    def test_the_next_word_may_be_inside_the_same_chunk(self):
        assert texts_of("(Washington.)") == ("(", "Washington", ".", ")")

    def test_a_multi_period_run_stays_together_and_apart_from_the_word(self):
        assert texts_of("wait... what") == ("wait", "...", "what")
        assert texts_of("wait...") == ("wait", "...")
        assert texts_of("..") == ("..",)

    def test_a_lone_period_is_a_word(self):
        assert texts_of(".") == (".",)

    def test_a_period_after_a_clitic_is_split_like_any_other(self):
        assert texts_of("don't.") == ("don", "'t", ".")


class TestEnglishApostrophes:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("don't", ("don", "'t")),
            ("it's", ("it", "'s")),
            ("students'", ("students", "'")),
            ("'tis", ("'", "tis")),
            ("1990's", ("1990", "'s")),
            ("5'6", ("5", "'", "6")),
            ("rock 'n' roll", ("rock", "'", "n", "'", "roll")),
        ],
    )
    def test_splits_a_clitic_to_the_right_and_isolates_the_rest(self, text, expected):
        assert texts_of(text) == expected

    def test_a_digit_before_any_letter_but_s_is_reached_by_no_rule(self):
        """The Perl's 1990's rule names the letter s; nothing covers 5'a."""
        assert texts_of("5'a") == ("5'a",)

    def test_the_split_clitic_keeps_its_span(self):
        assert list(MosesPreTokenizer().split("don't")) == [
            Word("don", 0, 3),
            Word("'t", 3, 5),
        ]


class TestFrenchAndItalianApostrophes:
    @pytest.mark.parametrize("language", ["fr", "it"])
    def test_an_elision_splits_to_the_left(self, language):
        assert texts_of("l'homme", language=language) == ("l'", "homme")
        assert texts_of("dell'anno", language=language) == ("dell'", "anno")

    def test_the_same_rule_applied_to_english_gives_the_mirror_split(self):
        assert texts_of("don't", language="fr") == ("don'", "t")

    def test_an_apostrophe_not_between_letters_stands_alone(self):
        assert texts_of("amis' 'a 5'6", language="fr") == (
            "amis",
            "'",
            "'",
            "a",
            "5",
            "'",
            "6",
        )

    def test_the_prefix_list_is_english_whatever_the_language(self):
        assert texts_of("Mr. Dupont", language="fr") == ("Mr.", "Dupont")


class TestAggressiveHyphenSplitting:
    def test_a_hyphen_between_alphanumerics_becomes_the_placeholder(self):
        assert texts_of("well-known", aggressive_hyphen_splitting=True) == (
            "well",
            HYPHEN_PLACEHOLDER,
            "known",
        )

    def test_the_placeholder_stands_for_the_one_source_character(self):
        words = list(
            MosesPreTokenizer(aggressive_hyphen_splitting=True).split("well-known")
        )

        assert words == [Word("well", 0, 4), Word("@-@", 4, 5), Word("known", 5, 10)]

    def test_a_hyphen_without_an_alphanumeric_on_both_sides_is_untouched(self):
        assert texts_of("-5 re- a--b", aggressive_hyphen_splitting=True) == (
            "-5",
            "re-",
            "a--b",
        )

    def test_digits_count_as_alphanumeric(self):
        assert texts_of("3-4", aggressive_hyphen_splitting=True) == ("3", "@-@", "4")


class TestEscaping:
    @pytest.mark.parametrize(("plain", "escaped"), sorted(ESCAPES.items()))
    def test_each_special_character_is_rewritten(self, plain, escaped):
        assert texts_of(f"a {plain} b", escape_special_characters=True) == (
            "a",
            escaped,
            "b",
        )

    def test_the_text_changes_and_the_span_does_not(self):
        words = list(MosesPreTokenizer(escape_special_characters=True).split("don't"))

        assert words == [Word("don", 0, 3), Word("&apos;t", 3, 5)]
        assert len(words[1].text) == 7
        assert words[1].end - words[1].start == 2

    def test_escaping_is_off_by_default(self):
        assert texts_of("Tom & Jerry") == ("Tom", "&", "Jerry")

    def test_unescaping_every_word_recovers_its_source_slice(self):
        """The two things a word carries agree once the rewrite is undone."""
        pre_tokenizer = MosesPreTokenizer(escape_special_characters=True)
        for text in MIXED_SENTENCES:
            for word in pre_tokenizer.split(text):
                assert unescaped(word.text) == text[word.start : word.end]


class TestNothingLostNothingInvented:
    @pytest.mark.parametrize("text", MIXED_SENTENCES)
    def test_every_word_is_the_slice_it_claims(self, text):
        for word in MosesPreTokenizer().split(text):
            assert word.text == text[word.start : word.end]

    @pytest.mark.parametrize("text", MIXED_SENTENCES)
    def test_the_words_joined_are_the_text_without_its_whitespace(self, text):
        assert "".join(texts_of(text)) == "".join(text.split())

    @pytest.mark.parametrize("text", MIXED_SENTENCES)
    def test_words_never_overlap_and_never_run_backwards(self, text):
        previous_end = 0
        for word in MosesPreTokenizer().split(text):
            assert word.start >= previous_end
            previous_end = word.end

    def test_the_prose_corpus_word_by_word(self):
        assert texts_of(PROSE_CORPUS[1]) == (
            "The",
            "dog",
            "didn",
            "'t",
            "mind",
            ";",
            "it",
            "was",
            "asleep",
            ".",
        )
