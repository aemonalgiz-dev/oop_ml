"""Spec for PennTreebankPreTokenizer -- the 1993 rules, as a scanner that keeps offsets.

The oracle at the bottom is MacIntyre's ``tokenizer.sed``, transcribed rule by
rule as the substitution pipeline NLTK's ``TreebankWordTokenizer`` runs. It is
written from the rules, not from the scanner, and it knows nothing about
offsets, which is exactly why the scanner exists; it checks the scanner's texts
on a corpus of sentences and nothing else.
"""

import re

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.word_level.penn_treebank import (
    BRACKET_NAMES,
    CLOSING_QUOTE,
    OPENING_QUOTE,
    PennTreebankPreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word
from test.core.natural_language_processing.fixtures import PROSE_CORPUS

WORKED_EXAMPLE = '"Hello," she said. "It\'s 3.14, isn\'t it?"'

REWRITTEN = {OPENING_QUOTE, CLOSING_QUOTE, *BRACKET_NAMES.values()}
"""The only texts a word may carry that are not a slice of the source."""


def unrewritten_words_are_source_slices(
    splitter: PennTreebankPreTokenizer, text: str
) -> bool:
    return all(
        text[word.start : word.end] == word.text
        for word in splitter.split(text)
        if word.text not in REWRITTEN
    )


class TestTheWorkedExample:
    def test_pins_the_sixteen_words(self):
        assert PennTreebankPreTokenizer().split(WORKED_EXAMPLE).texts == (
            "``",
            "Hello",
            ",",
            "''",
            "she",
            "said.",
            "``",
            "It",
            "'s",
            "3.14",
            ",",
            "is",
            "n't",
            "it",
            "?",
            "''",
        )

    def test_pins_their_spans(self):
        assert list(PennTreebankPreTokenizer().split(WORKED_EXAMPLE)) == [
            Word("``", 0, 1),
            Word("Hello", 1, 6),
            Word(",", 6, 7),
            Word("''", 7, 8),
            Word("she", 9, 12),
            Word("said.", 13, 18),
            Word("``", 19, 20),
            Word("It", 20, 22),
            Word("'s", 22, 24),
            Word("3.14", 25, 29),
            Word(",", 29, 30),
            Word("is", 31, 33),
            Word("n't", 33, 36),
            Word("it", 37, 39),
            Word("?", 39, 40),
            Word("''", 40, 41),
        ]

    def test_unrewritten_words_reproduce_their_source_slice(self):
        assert unrewritten_words_are_source_slices(
            PennTreebankPreTokenizer(), WORKED_EXAMPLE
        )

    def test_a_rewritten_quote_keeps_the_span_of_the_one_character_it_stands_for(self):
        opening = PennTreebankPreTokenizer().split(WORKED_EXAMPLE)[0]

        assert opening == Word("``", 0, 1)
        assert WORKED_EXAMPLE[opening.start : opening.end] == '"'


class TestClitics:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Don't", ("Do", "n't")),
            ("can't", ("ca", "n't")),
            ("won't", ("wo", "n't")),
            ("isn't", ("is", "n't")),
            ("It's", ("It", "'s")),
            ("I'll", ("I", "'ll")),
            ("they're", ("they", "'re")),
            ("we've", ("we", "'ve")),
            ("I'm", ("I", "'m")),
            ("he'd", ("he", "'d")),
        ],
    )
    def test_a_clitic_comes_off_its_host(self, text, expected):
        assert PennTreebankPreTokenizer().split(text).texts == expected

    def test_the_treebank_keeps_ca_from_cannot_and_the_test_says_so(self):
        """``n't`` is the morpheme; what is left is what is left."""
        assert PennTreebankPreTokenizer().split("can't").texts == ("ca", "n't")

    def test_upper_case_clitics_come_off_too(self):
        assert PennTreebankPreTokenizer().split("IT'S DON'T").texts == (
            "IT",
            "'S",
            "DO",
            "N'T",
        )

    def test_a_possessive_apostrophe_after_s_comes_off_alone(self):
        assert PennTreebankPreTokenizer().split("the dogs' bowls").texts == (
            "the",
            "dogs",
            "'",
            "bowls",
        )

    def test_a_clitic_needs_a_host(self):
        assert PennTreebankPreTokenizer().split("'s").texts == ("'s",)

    def test_a_clitic_followed_by_a_mid_text_period_stays_attached(self):
        """The script ran one sentence at a time; a period that is not final is
        an abbreviation's, and ``'s.`` is not a clitic at the end of a word."""
        assert PennTreebankPreTokenizer().split("It's. Then").texts == ("It's.", "Then")

    def test_a_clitic_comes_off_before_a_final_period(self):
        assert PennTreebankPreTokenizer().split("It's.").texts == ("It", "'s", ".")


class TestContractions:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("cannot", ("can", "not")),
            ("Cannot", ("Can", "not")),
            ("gonna", ("gon", "na")),
            ("gotta", ("got", "ta")),
            ("d'ye", ("d", "'ye")),
            ("gimme", ("gim", "me")),
            ("lemme", ("lem", "me")),
            ("wanna", ("wan", "na")),
            ("more'n", ("more", "'n")),
            ("'tis", ("'t", "is")),
            ("'Twas", ("'T", "was")),
        ],
    )
    def test_the_fixed_list_is_cut_between_its_halves(self, text, expected):
        assert PennTreebankPreTokenizer().split(text).texts == expected

    def test_wanna_is_cut_only_at_the_end_of_a_piece(self):
        """The script's ``(?=\\s)`` where the others have ``\\b``."""
        assert PennTreebankPreTokenizer().split("wanna. wanna").texts == (
            "wanna.",
            "wan",
            "na",
        )

    def test_cannot_is_cut_even_with_a_mid_text_period_attached(self):
        """``\\b`` holds between ``t`` and ``.``, so the period rides with ``not``."""
        assert PennTreebankPreTokenizer().split("cannot. So").texts == (
            "can",
            "not.",
            "So",
        )

    def test_a_contraction_inside_a_longer_word_is_left_alone(self):
        assert PennTreebankPreTokenizer().split("scannot cannots").texts == (
            "scannot",
            "cannots",
        )


class TestPunctuation:
    def test_a_period_comes_off_only_at_the_end_of_the_text(self):
        assert PennTreebankPreTokenizer().split("U.S. Navy.").texts == (
            "U.S.",
            "Navy",
            ".",
        )

    def test_the_final_period_comes_off_before_closing_quotes_and_brackets(self):
        assert PennTreebankPreTokenizer().split('he said.")').texts == (
            "he",
            "said",
            ".",
            "''",
            ")",
        )

    def test_trailing_whitespace_does_not_hide_the_final_period(self):
        assert PennTreebankPreTokenizer().split("end.  ").texts == ("end", ".")

    @pytest.mark.parametrize("number", ["1,000", "3.14", "3:30", "1,000,000"])
    def test_punctuation_inside_a_number_stays(self, number):
        assert PennTreebankPreTokenizer().split(number).texts == (number,)

    def test_a_number_before_the_final_period_still_loses_it(self):
        assert PennTreebankPreTokenizer().split("1,000 and 3.14.").texts == (
            "1,000",
            "and",
            "3.14",
            ".",
        )

    def test_a_comma_or_colon_not_followed_by_a_digit_comes_off(self):
        assert PennTreebankPreTokenizer().split("note: yes, no").texts == (
            "note",
            ":",
            "yes",
            ",",
            "no",
        )

    def test_an_ellipsis_is_one_word(self):
        assert PennTreebankPreTokenizer().split("He said: hello; then...").texts == (
            "He",
            "said",
            ":",
            "hello",
            ";",
            "then",
            "...",
        )

    def test_a_fourth_dot_after_an_ellipsis_is_the_final_period(self):
        assert PennTreebankPreTokenizer().split("wow....").texts == ("wow", "...", ".")

    def test_a_double_dash_is_its_own_word(self):
        assert PennTreebankPreTokenizer().split("wait -- no").texts == (
            "wait",
            "--",
            "no",
        )

    def test_an_attached_double_dash_comes_off(self):
        assert PennTreebankPreTokenizer().split("wait--no").texts == (
            "wait",
            "--",
            "no",
        )

    def test_a_hyphenated_word_stays_whole(self):
        assert PennTreebankPreTokenizer().split("well-known").texts == ("well-known",)

    def test_the_always_separate_characters(self):
        assert PennTreebankPreTokenizer().split(
            "AT&T $5 50% #tag a@b.com; yes!"
        ).texts == (
            "AT",
            "&",
            "T",
            "$",
            "5",
            "50",
            "%",
            "#",
            "tag",
            "a",
            "@",
            "b.com",
            ";",
            "yes",
            "!",
        )

    def test_each_question_and_exclamation_mark_is_a_word(self):
        assert PennTreebankPreTokenizer().split("what?!!").texts == (
            "what",
            "?",
            "!",
            "!",
        )


class TestQuotes:
    def test_a_double_quote_opens_at_the_start_and_closes_after_a_word(self):
        assert PennTreebankPreTokenizer().split('"quoted"').texts == (
            "``",
            "quoted",
            "''",
        )

    def test_a_double_quote_after_a_space_opens(self):
        assert PennTreebankPreTokenizer().split('say "hi"').texts == (
            "say",
            "``",
            "hi",
            "''",
        )

    def test_a_double_quote_after_an_opening_bracket_opens(self):
        assert PennTreebankPreTokenizer().split('("nested")').texts == (
            "(",
            "``",
            "nested",
            "''",
            ")",
        )

    def test_a_pair_of_apostrophes_is_a_quote_and_opens_or_closes_like_a_double_quote(
        self,
    ):
        assert PennTreebankPreTokenizer().split("''quoted'' text").texts == (
            "``",
            "quoted",
            "''",
            "text",
        )

    def test_two_backquotes_are_an_opening_quote(self):
        assert PennTreebankPreTokenizer().split("``quoted'' text").texts == (
            "``",
            "quoted",
            "''",
            "text",
        )

    def test_a_backquote_at_word_start_is_its_own_word(self):
        assert PennTreebankPreTokenizer().split("`quoted' text").texts == (
            "`",
            "quoted",
            "'",
            "text",
        )

    def test_a_closing_quote_stands_for_its_own_character(self):
        closing = PennTreebankPreTokenizer().split('"quoted"')[2]

        assert closing == Word("''", 7, 8)

    def test_a_pair_of_apostrophes_stands_for_two_characters(self):
        assert list(PennTreebankPreTokenizer().split("''a''")) == [
            Word("``", 0, 2),
            Word("a", 2, 3),
            Word("''", 3, 5),
        ]


class TestBrackets:
    def test_brackets_come_off_and_are_kept_as_themselves_by_default(self):
        assert PennTreebankPreTokenizer().split("f(x) [y] {z}").texts == (
            "f",
            "(",
            "x",
            ")",
            "[",
            "y",
            "]",
            "{",
            "z",
            "}",
        )

    def test_brackets_are_rewritten_on_request(self):
        assert PennTreebankPreTokenizer(convert_brackets=True).split(
            "f(x) [y] {z}"
        ).texts == (
            "f",
            "-LRB-",
            "x",
            "-RRB-",
            "-LSB-",
            "y",
            "-RSB-",
            "-LCB-",
            "z",
            "-RCB-",
        )

    def test_a_rewritten_bracket_keeps_the_span_of_its_source_character(self):
        assert list(PennTreebankPreTokenizer(convert_brackets=True).split("(a)")) == [
            Word("-LRB-", 0, 1),
            Word("a", 1, 2),
            Word("-RRB-", 2, 3),
        ]

    def test_angle_brackets_come_off_but_are_never_rewritten(self):
        assert PennTreebankPreTokenizer(convert_brackets=True).split("<w>").texts == (
            "<",
            "w",
            ">",
        )

    def test_convert_brackets_defaults_to_false(self):
        assert PennTreebankPreTokenizer().convert_brackets is False


class TestBoundary:
    def test_a_blank_text_has_no_words(self):
        assert PennTreebankPreTokenizer().split("   ").texts == ()

    def test_an_empty_text_has_no_words(self):
        assert PennTreebankPreTokenizer().split("").texts == ()

    def test_runs_of_whitespace_are_skipped(self):
        assert PennTreebankPreTokenizer().split("  spaced   out  ").texts == (
            "spaced",
            "out",
        )

    def test_a_non_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            PennTreebankPreTokenizer().split(["a"])  # type: ignore[arg-type]

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            PennTreebankPreTokenizer(convert_parentheses=True)  # type: ignore[call-arg]

    def test_a_non_boolean_is_refused(self):
        with pytest.raises(ValidationError):
            PennTreebankPreTokenizer(convert_brackets="yes please")  # type: ignore[arg-type]


# --- the oracle: tokenizer.sed as a substitution pipeline, from the rules ---

STARTING_QUOTES = [
    (re.compile(r'^"'), r"``"),
    (re.compile(r"(``)"), r" \1 "),
    (re.compile(r"([ \(\[{<])(\"|'{2})"), r"\1 `` "),
]
PUNCTUATION = [
    (re.compile(r"([:,])([^\d])"), r" \1 \2"),
    (re.compile(r"([:,])$"), r" \1 "),
    (re.compile(r"\.\.\."), r" ... "),
    (re.compile(r"[;@#$%&]"), r" \g<0> "),
    (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r"\1 \2\3 "),
    (re.compile(r"[?!]"), r" \g<0> "),
    (re.compile(r"([^'])' "), r"\1 ' "),
]
PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")
CONVERT_PARENTHESES = [
    (re.compile(r"\("), "-LRB-"),
    (re.compile(r"\)"), "-RRB-"),
    (re.compile(r"\["), "-LSB-"),
    (re.compile(r"\]"), "-RSB-"),
    (re.compile(r"\{"), "-LCB-"),
    (re.compile(r"\}"), "-RCB-"),
]
DOUBLE_DASHES = (re.compile(r"--"), r" -- ")
ENDING_QUOTES = [
    (re.compile(r"''"), " '' "),
    (re.compile(r'"'), " '' "),
    (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
    (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
]
CONTRACTIONS_TWO = [
    re.compile(r"(?i)\b(can)(not)\b"),
    re.compile(r"(?i)\b(d)('ye)\b"),
    re.compile(r"(?i)\b(gim)(me)\b"),
    re.compile(r"(?i)\b(gon)(na)\b"),
    re.compile(r"(?i)\b(got)(ta)\b"),
    re.compile(r"(?i)\b(lem)(me)\b"),
    re.compile(r"(?i)\b(more)('n)\b"),
    re.compile(r"(?i)\b(wan)(na)(?=\s)"),
]
CONTRACTIONS_THREE = [
    re.compile(r"(?i) ('t)(is)\b"),
    re.compile(r"(?i) ('t)(was)\b"),
]


def treebank_by_substitution(text: str, convert_brackets: bool = False) -> list[str]:
    """MacIntyre's sed script, as NLTK transcribes it: pad with spaces, then split."""
    for pattern, substitution in STARTING_QUOTES:
        text = pattern.sub(substitution, text)
    for pattern, substitution in PUNCTUATION:
        text = pattern.sub(substitution, text)
    pattern, substitution = PARENS_BRACKETS
    text = pattern.sub(substitution, text)
    if convert_brackets:
        for pattern, substitution in CONVERT_PARENTHESES:
            text = pattern.sub(substitution, text)
    pattern, substitution = DOUBLE_DASHES
    text = pattern.sub(substitution, text)
    text = " " + text + " "
    for pattern, substitution in ENDING_QUOTES:
        text = pattern.sub(substitution, text)
    for pattern in CONTRACTIONS_TWO:
        text = pattern.sub(r" \1 \2 ", text)
    for pattern in CONTRACTIONS_THREE:
        text = pattern.sub(r" \1 \2 ", text)
    return text.split()


ORACLE_CORPUS = [
    *PROSE_CORPUS,
    WORKED_EXAMPLE,
    "I can't believe it's not butter.",
    "They're gonna wanna see this -- trust me!",
    "The U.S. Navy (est. 1775) bought 1,000 ships at $5,000,000 each.",
    '"Don\'t," he said; "you cannot win."',
    "It costs 50% more'n I'd pay: about 3.14 dollars, give or take...",
    "Gimme a break, lemme think, d'ye mind?",
    "AT&T's shares [NYSE: T] fell 2.5%; investors weren't pleased!",
    "'Tis the season, 'twas the night.",
    "The dogs' bowls were empty (again).",
    "Well-known authors' books sell; unknown ones' don't.",
    "``Quoted'' with typewriter quotes, and \"quoted\" with straight ones.",
    "Is it 3:30 yet?!",
    'He whispered: "wait..." and then "go!"',
    "What?? No!! Really?!",
    "Prices: $1, $2.50 and $3,000.",
    "I'LL DO IT, I'M SURE, THEY'VE SAID SO.",
    "e-mail me at a@b.com #now",
    "She said (quietly) that it's fine.",
]


class TestAgainstTheSubstitutionOracle:
    def test_the_oracle_reproduces_the_worked_example(self):
        """The oracle is checked against the known Treebank output before it
        checks anything else."""
        assert treebank_by_substitution(WORKED_EXAMPLE) == list(
            PennTreebankPreTokenizer().split(WORKED_EXAMPLE).texts
        )

    @pytest.mark.parametrize("text", ORACLE_CORPUS)
    def test_the_scanner_agrees_with_the_substitutions_on_texts(self, text):
        assert list(PennTreebankPreTokenizer().split(text).texts) == (
            treebank_by_substitution(text)
        )

    @pytest.mark.parametrize("text", ORACLE_CORPUS)
    def test_and_with_brackets_converted(self, text):
        assert list(
            PennTreebankPreTokenizer(convert_brackets=True).split(text).texts
        ) == treebank_by_substitution(text, convert_brackets=True)

    @pytest.mark.parametrize("text", ORACLE_CORPUS)
    def test_unrewritten_words_reproduce_their_source_slice(self, text):
        assert unrewritten_words_are_source_slices(PennTreebankPreTokenizer(), text)


class TestWhereTheScannerDeliberatelyDiffersFromTheScript:
    """Each divergence is recorded in the module docstring; here each is pinned
    beside the script's own answer, so a reader can see both."""

    def test_every_comma_is_examined_where_the_script_skips_the_one_after_a_split(
        self,
    ):
        assert PennTreebankPreTokenizer().split("x,,y").texts == ("x", ",", ",", "y")
        assert treebank_by_substitution("x,,y") == ["x", ",", ",y"]

    def test_a_pair_of_apostrophes_at_the_start_opens_where_the_script_closes(self):
        assert PennTreebankPreTokenizer().split("''hi''").texts == ("``", "hi", "''")
        assert treebank_by_substitution("''hi''") == ["''", "hi", "''"]

    def test_any_white_space_opens_a_quote_where_the_script_needs_a_space(self):
        assert PennTreebankPreTokenizer().split('a\t"b"').texts == (
            "a",
            "``",
            "b",
            "''",
        )
        assert treebank_by_substitution('a\t"b"') == ["a", "''", "b", "''"]

    def test_a_run_of_backquotes_is_one_word_where_the_script_takes_two(self):
        assert PennTreebankPreTokenizer().split("```x").texts == ("```", "x")
        assert treebank_by_substitution("```x") == ["``", "`x"]
