"""Spec for RuleAndExceptionPreTokenizer -- whitespace, a table, then affix rules.

The mechanism is tested with tiny hand-written rule sets, so that a failure
names the step that broke; the English configuration is pinned separately.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.word_level.rule_based import (
    ENGLISH_EXCEPTIONS,
    ENGLISH_INFIXES,
    ENGLISH_PREFIXES,
    ENGLISH_SUFFIXES,
    RuleAndExceptionPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word
from test.core.natural_language_processing.fixtures import PROSE_CORPUS

PINNED_SENTENCE = '"Don\'t stop," she said (quietly).'

ENGLISH_SENTENCES: list[str] = [
    *PROSE_CORPUS,
    PINNED_SENTENCE,
    "U.S. law is well-known, e.g. to Mr. Smith (Ph.D.).",
    "I'm sure they're gonna say \"we can't\"... aren't they?",
    "She’ll arrive at 5 a.m. on Jan. 3; the U.K. won’t.",
]


def english_texts(text: str) -> tuple[str, ...]:
    return RuleAndExceptionPreTokenizer.english().split(text).texts


def with_fragments(
    role: str, fragments: tuple[str, ...]
) -> RuleAndExceptionPreTokenizer:
    """One rule set holding ``fragments`` in the named role and nothing else."""
    if role == "prefixes":
        return RuleAndExceptionPreTokenizer(prefixes=fragments)
    if role == "suffixes":
        return RuleAndExceptionPreTokenizer(suffixes=fragments)
    return RuleAndExceptionPreTokenizer(infixes=fragments)


class TestConstruction:
    def test_no_rules_and_no_exceptions_by_default(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer()

        assert pre_tokenizer.prefixes == ()
        assert pre_tokenizer.suffixes == ()
        assert pre_tokenizer.infixes == ()
        assert pre_tokenizer.exceptions == {}

    def test_english_assembles_the_four_module_constants(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer.english()

        assert pre_tokenizer.prefixes == ENGLISH_PREFIXES
        assert pre_tokenizer.suffixes == ENGLISH_SUFFIXES
        assert pre_tokenizer.infixes == ENGLISH_INFIXES
        assert pre_tokenizer.exceptions == ENGLISH_EXCEPTIONS

    @pytest.mark.parametrize("role", ["prefixes", "suffixes", "infixes"])
    def test_a_malformed_fragment_is_refused(self, role):
        with pytest.raises(ValidationError):
            with_fragments(role, ("(",))

    @pytest.mark.parametrize("role", ["prefixes", "suffixes", "infixes"])
    def test_an_empty_fragment_is_refused(self, role):
        with pytest.raises(ValidationError):
            with_fragments(role, ("",))

    @pytest.mark.parametrize("fragment", ["a*", "b?", "(?:)"])
    def test_a_fragment_that_can_match_nothing_is_refused(self, fragment):
        """It would peel zero characters forever."""
        with pytest.raises(ValidationError):
            RuleAndExceptionPreTokenizer(prefixes=(fragment,))

    def test_a_zero_length_match_the_constructor_could_not_see_is_ignored(self):
        """A bare lookbehind matches nothing on the empty string the constructor
        probes with, so the guard is in the loop as well: a match of no
        characters is no match."""
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            prefixes=("(?<=x)",), suffixes=("(?<=x)",), infixes=("(?<=x)",)
        )

        assert pre_tokenizer.split("axb").texts == ("axb",)

    def test_an_exception_that_splits_is_accepted(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            exceptions={"don't": ("do", "n't")}
        )

        assert pre_tokenizer.exceptions == {"don't": ("do", "n't")}

    def test_an_exception_that_keeps_a_word_whole_is_accepted(self):
        assert RuleAndExceptionPreTokenizer(exceptions={"U.S.": ("U.S.",)}).exceptions

    def test_an_exception_that_rewrites_is_refused(self):
        """The table says where a word is split, not what it becomes."""
        with pytest.raises(ValidationError):
            RuleAndExceptionPreTokenizer(exceptions={"don't": ("do", "not")})

    @pytest.mark.parametrize(
        "table",
        [
            {"a b": ("a", " b")},
            {"": ()},
            {"ab": ()},
            {"ab": ("ab", "")},
        ],
    )
    def test_a_key_with_whitespace_or_an_empty_piece_is_refused(self, table):
        with pytest.raises(ValidationError):
            RuleAndExceptionPreTokenizer(exceptions=table)

    def test_a_wrong_type_is_refused_by_pydantic(self):
        with pytest.raises(ValidationError):
            RuleAndExceptionPreTokenizer(prefixes=5)  # type: ignore[arg-type]

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            RuleAndExceptionPreTokenizer(special_cases={})  # type: ignore[call-arg]

    def test_a_non_string_text_is_refused(self):
        with pytest.raises(InvalidValuesError):
            RuleAndExceptionPreTokenizer().split(["a"])  # type: ignore[arg-type]


class TestWithoutRules:
    def test_splits_on_whitespace_and_nothing_else(self):
        assert RuleAndExceptionPreTokenizer().split("a (b) c.").texts == (
            "a",
            "(b)",
            "c.",
        )

    @pytest.mark.parametrize("text", ["", "  ", "\t\n"])
    def test_a_blank_text_has_no_words(self, text):
        assert RuleAndExceptionPreTokenizer().split(text).texts == ()


class TestExceptionTable:
    def test_a_listed_chunk_is_split_as_listed(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            exceptions={"don't": ("do", "n't")}
        )

        assert list(pre_tokenizer.split("I don't")) == [
            Word("I", 0, 1),
            Word("do", 2, 4),
            Word("n't", 4, 7),
        ]

    def test_the_table_wins_over_every_rule(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            suffixes=(r"\.",), exceptions={"etc.": ("etc.",)}
        )

        assert pre_tokenizer.split("etc. abc.").texts == ("etc.", "abc", ".")

    def test_the_table_is_consulted_after_a_prefix_peel(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            prefixes=(r"\(",), exceptions={"don't": ("do", "n't")}
        )

        assert pre_tokenizer.split("(don't").texts == ("(", "do", "n't")

    def test_the_table_is_consulted_after_a_suffix_peel(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            suffixes=(r"\)",), exceptions={"don't": ("do", "n't")}
        )

        assert pre_tokenizer.split("don't)").texts == ("do", "n't", ")")

    def test_the_table_is_case_sensitive(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            exceptions={"don't": ("do", "n't")}
        )

        assert pre_tokenizer.split("Don't").texts == ("Don't",)


class TestPrefixesAndSuffixes:
    def test_prefixes_peel_repeatedly_from_the_front(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(prefixes=(r"\(", r'"'))

        assert pre_tokenizer.split('("word').texts == ("(", '"', "word")

    def test_suffixes_peel_repeatedly_from_the_back_and_return_in_source_order(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(suffixes=(r"\)", r'"', r"\."))

        assert pre_tokenizer.split('word").').texts == ("word", '"', ")", ".")

    def test_a_chunk_that_is_all_affixes_leaves_no_remainder(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            prefixes=(r"\(",), suffixes=(r"\)",)
        )

        assert pre_tokenizer.split("()").texts == ("(", ")")
        assert pre_tokenizer.split("(").texts == ("(",)

    def test_the_suffix_whose_match_starts_furthest_left_wins(self):
        """No length rule is needed: 's begins one character before '."""
        pre_tokenizer = RuleAndExceptionPreTokenizer(suffixes=("'", "'s"))

        assert pre_tokenizer.split("it's").texts == ("it", "'s")

    def test_a_prefix_can_be_longer_than_one_character(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(prefixes=(r"\.\.+",))

        assert pre_tokenizer.split("...word").texts == ("...", "word")

    def test_a_lookbehind_sees_only_the_remainder(self):
        """As in spaCy, the suffix pattern is matched against what is left,
        so a lookbehind at the remainder's start sees nothing."""
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            prefixes=("a",), suffixes=(r"(?<=a)b",)
        )

        assert pre_tokenizer.split("ab").texts == ("a", "b")
        assert pre_tokenizer.split("acb").texts == ("a", "cb")


class TestInfixes:
    def test_the_remainder_is_cut_at_every_infix_match(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(infixes=("-",))

        assert pre_tokenizer.split("a-b-c").texts == ("a", "-", "b", "-", "c")

    def test_infixes_run_after_the_affixes_are_peeled(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(
            prefixes=(r"\(",), suffixes=(r"\)",), infixes=("-",)
        )

        assert pre_tokenizer.split("(a-b)").texts == ("(", "a", "-", "b", ")")

    def test_an_infix_at_either_edge_leaves_no_empty_piece(self):
        pre_tokenizer = RuleAndExceptionPreTokenizer(infixes=("-",))

        assert pre_tokenizer.split("-a-").texts == ("-", "a", "-")

    def test_fragments_matching_at_one_position_go_to_the_first_listed(self):
        """Python's alternation order, stated as the tie rule."""
        longer_first = RuleAndExceptionPreTokenizer(infixes=("ab", "a"))
        shorter_first = RuleAndExceptionPreTokenizer(infixes=("a", "ab"))

        assert longer_first.split("xaby").texts == ("x", "ab", "y")
        assert shorter_first.split("xaby").texts == ("x", "a", "by")


class TestEnglish:
    def test_pins_the_quoted_sentence(self):
        assert english_texts(PINNED_SENTENCE) == (
            '"',
            "Do",
            "n't",
            "stop",
            ",",
            '"',
            "she",
            "said",
            "(",
            "quietly",
            ")",
            ".",
        )

    def test_the_pinned_sentence_is_twelve_pieces_with_these_spans(self):
        words = list(RuleAndExceptionPreTokenizer.english().split(PINNED_SENTENCE))

        assert len(words) == 12
        assert words[:3] == [Word('"', 0, 1), Word("Do", 1, 3), Word("n't", 3, 6)]
        assert words[-4:] == [
            Word("(", 23, 24),
            Word("quietly", 24, 31),
            Word(")", 31, 32),
            Word(".", 32, 33),
        ]

    def test_pins_us_law(self):
        assert english_texts("U.S. law") == ("U.S.", "law")

    def test_pins_well_known(self):
        assert english_texts("well-known") == ("well", "-", "known")

    def test_the_table_is_consulted_after_a_peel(self):
        assert english_texts("(don't)") == ("(", "do", "n't", ")")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("don't", ("do", "n't")),
            ("can't", ("ca", "n't")),
            ("won't", ("wo", "n't")),
            ("Can't", ("Ca", "n't")),
            ("I'm", ("I", "'m")),
            ("I'd", ("I", "'d")),
            ("cannot", ("can", "not")),
            ("y'all", ("y'", "all")),
            ("ma'am", ("ma'am",)),
        ],
    )
    def test_the_listed_contractions(self, text, expected):
        assert english_texts(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("they're", ("they", "'re")),
            ("we've", ("we", "'ve")),
            ("she'll", ("she", "'ll")),
            ("he'd", ("he", "'d")),
            ("it's", ("it", "'s")),
            ("isn't", ("is", "n't")),
            ("students'", ("students", "'")),
            ("'tis", ("'", "tis")),
        ],
    )
    def test_the_clitic_suffixes_reach_words_the_table_never_listed(
        self, text, expected
    ):
        assert english_texts(text) == expected

    def test_a_curly_apostrophe_is_read_like_a_straight_one(self):
        assert english_texts("don’t she’ll") == ("do", "n’t", "she", "’ll")

    @pytest.mark.parametrize(
        "abbreviation", ["Mr.", "Dr.", "Prof.", "e.g.", "i.e.", "etc."]
    )
    def test_a_listed_abbreviation_keeps_its_period(self, abbreviation):
        assert english_texts(f"{abbreviation} Then") == (abbreviation, "Then")

    def test_an_unlisted_abbreviation_comes_apart(self):
        """The honest half of the mechanism: the table is never complete."""
        assert english_texts("Gov. Smith") == ("Gov", ".", "Smith")

    def test_a_final_period_splits_after_a_lowercase_letter_a_digit_or_a_closer(self):
        assert english_texts("said.") == ("said", ".")
        assert english_texts("5.") == ("5", ".")
        assert english_texts("(so).") == ("(", "so", ")", ".")

    def test_a_final_period_splits_after_two_capitals_but_not_one(self):
        """NASA. is a sentence end; A. is an initial; U.S. is an initialism,
        and the last two are kept by the rule alone, not only by the table."""
        assert english_texts("NASA.") == ("NASA", ".")
        assert english_texts("A. Lincoln") == ("A.", "Lincoln")
        assert RuleAndExceptionPreTokenizer(suffixes=ENGLISH_SUFFIXES).split(
            "U.S."
        ).texts == ("U.S.",)

    def test_a_period_inside_a_number_stays(self):
        assert english_texts("3.14") == ("3.14",)

    def test_a_comma_inside_a_number_stays_and_between_letters_does_not(self):
        assert english_texts("5,300 hello,world") == ("5,300", "hello", ",", "world")

    def test_an_ellipsis_is_one_piece_wherever_it_falls(self):
        assert english_texts("wait... what?") == ("wait", "...", "what", "?")
        assert english_texts("...so") == ("...", "so")

    def test_quotes_and_punctuation_peel_in_the_right_order(self):
        assert english_texts('he said, "Don\'t!"') == (
            "he",
            "said",
            ",",
            '"',
            "Do",
            "n't",
            "!",
            '"',
        )

    def test_a_currency_sign_is_a_prefix(self):
        assert english_texts("$5") == ("$", "5")

    def test_the_prose_corpus_word_by_word(self):
        assert english_texts(PROSE_CORPUS[1]) == (
            "The",
            "dog",
            "did",
            "n't",
            "mind",
            ";",
            "it",
            "was",
            "asleep",
            ".",
        )

    def test_the_table_size_is_as_the_module_docstring_says(self):
        """90 entries: 38 abbreviations, and 17 contractions grown to 52 by
        the capitalised and curly-apostrophe spellings. 46 stand for
        themselves: the 38 abbreviations plus ma'am and o'clock in all four
        spellings each."""
        assert len(ENGLISH_EXCEPTIONS) == 90
        assert sum(len(pieces) == 1 for pieces in ENGLISH_EXCEPTIONS.values()) == 46
        assert sum("’" in key for key in ENGLISH_EXCEPTIONS) == 22


class TestOffsets:
    @pytest.mark.parametrize("text", ENGLISH_SENTENCES)
    def test_every_piece_reproduces_its_source_slice(self, text):
        for word in RuleAndExceptionPreTokenizer.english().split(text):
            assert word.text == text[word.start : word.end]

    @pytest.mark.parametrize("text", ENGLISH_SENTENCES)
    def test_the_pieces_joined_are_the_text_without_its_whitespace(self, text):
        assert "".join(english_texts(text)) == "".join(text.split())

    @pytest.mark.parametrize("text", ENGLISH_SENTENCES)
    def test_pieces_never_overlap_and_never_run_backwards(self, text):
        previous_end = 0
        for word in RuleAndExceptionPreTokenizer.english().split(text):
            assert word.start >= previous_end
            previous_end = word.end
