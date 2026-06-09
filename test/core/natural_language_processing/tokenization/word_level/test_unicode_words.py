"""Spec for UnicodeWordPreTokenizer -- UAX #29 word boundaries over General Category.

Every expected segmentation below was derived by walking the annex's rules by
hand on the boundaries of the text, as the module docstring does for
``"can't stop"``, and then confirmed by the implementation.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.word_level.unicode_words import (
    UnicodeWordPreTokenizer,
    WordBreakClass,
    word_break_class_of,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word


@pytest.fixture
def words_only() -> UnicodeWordPreTokenizer:
    return UnicodeWordPreTokenizer()


@pytest.fixture
def with_punctuation() -> UnicodeWordPreTokenizer:
    return UnicodeWordPreTokenizer(keep_punctuation=True)


class TestWordBreakClassOf:
    @pytest.mark.parametrize(
        ("character", "expected"),
        [
            ("\r", WordBreakClass.CARRIAGE_RETURN),
            ("\n", WordBreakClass.LINE_FEED),
            ("", WordBreakClass.NEWLINE),
            (" ", WordBreakClass.NEWLINE),
            ("́", WordBreakClass.EXTEND),
            ("\U0001f3fd", WordBreakClass.EXTEND),
            ("‍", WordBreakClass.FORMAT),
            ("‌", WordBreakClass.FORMAT),
            ("カ", WordBreakClass.KATAKANA),
            ("ー", WordBreakClass.KATAKANA),
            ("ｶ", WordBreakClass.KATAKANA),
            ("ㇰ", WordBreakClass.KATAKANA),
            ("・", WordBreakClass.OTHER),
            ("א", WordBreakClass.HEBREW_LETTER),
            ("׳", WordBreakClass.ALPHABETIC_LETTER),
            ("״", WordBreakClass.MID_LETTER),
            ("a", WordBreakClass.ALPHABETIC_LETTER),
            ("é", WordBreakClass.ALPHABETIC_LETTER),
            ("한", WordBreakClass.ALPHABETIC_LETTER),
            ("Ⅻ", WordBreakClass.ALPHABETIC_LETTER),
            ("々", WordBreakClass.ALPHABETIC_LETTER),
            ("日", WordBreakClass.OTHER),
            ("ひ", WordBreakClass.OTHER),
            ("ก", WordBreakClass.OTHER),
            ("'", WordBreakClass.SINGLE_QUOTE),
            ('"', WordBreakClass.DOUBLE_QUOTE),
            (":", WordBreakClass.MID_LETTER),
            ("·", WordBreakClass.MID_LETTER),
            (",", WordBreakClass.MID_NUMBER),
            (";", WordBreakClass.MID_NUMBER),
            ("٬", WordBreakClass.MID_NUMBER),
            (".", WordBreakClass.MID_NUMBER_LETTER),
            ("’", WordBreakClass.MID_NUMBER_LETTER),
            ("7", WordBreakClass.NUMERIC),
            ("٣", WordBreakClass.NUMERIC),
            ("½", WordBreakClass.OTHER),
            ("_", WordBreakClass.EXTEND_NUMBER_LETTER),
            (" ", WordBreakClass.EXTEND_NUMBER_LETTER),
            ("\U0001f1eb", WordBreakClass.REGIONAL_INDICATOR),
            (" ", WordBreakClass.WORD_SEGMENT_SPACE),
            ("　", WordBreakClass.WORD_SEGMENT_SPACE),
            ("\t", WordBreakClass.OTHER),
            ("!", WordBreakClass.OTHER),
            ("👍", WordBreakClass.OTHER),
        ],
    )
    def test_classifies_from_general_category_and_the_lists(self, character, expected):
        assert word_break_class_of(character) is expected


class TestThePinnedExamples:
    def test_an_apostrophe_between_letters_does_not_break(self, words_only):
        assert words_only.split("can't stop").texts == ("can't", "stop")

    def test_the_worked_example_segments(self, words_only):
        assert list(words_only.segments_of("can't stop")) == [
            Word("can't", 0, 5),
            Word(" ", 5, 6),
            Word("stop", 6, 10),
        ]

    def test_a_period_or_comma_between_digits_does_not_break(self, words_only):
        assert words_only.split("3.14 and 1,000").texts == ("3.14", "and", "1,000")

    def test_a_period_between_letters_does_not_break_but_a_final_one_does(
        self, with_punctuation
    ):
        assert with_punctuation.split("e.g. this").texts == ("e.g", ".", "this")

    def test_the_final_period_is_dropped_without_punctuation(self, words_only):
        assert words_only.split("e.g. this").texts == ("e.g", "this")

    def test_underscores_join_letters(self, words_only):
        assert words_only.split("snake_case_name").texts == ("snake_case_name",)

    def test_cjk_ideographs_are_single_character_words(self, words_only):
        """The annex leaves scripts without spaces to dictionary segmentation, so
        every ideograph is Other and WB999 breaks on both sides of it."""
        assert words_only.split("日本語").texts == ("日", "本", "語")

    def test_a_precomposed_accented_word_is_whole(self, words_only):
        assert words_only.split("résumé").texts == ("résumé",)

    def test_a_combining_accented_word_is_whole_too(self, words_only):
        """WB4 hides the marks, so the letters on either side meet under WB5."""
        text = "résumé"

        assert list(words_only.split(text)) == [Word(text, 0, 8)]

    def test_two_flags_are_two_words_with_punctuation_kept(self, with_punctuation):
        assert with_punctuation.split("🇫🇷🇩🇪").texts == ("🇫🇷", "🇩🇪")

    def test_a_flag_holds_no_letter_so_it_is_not_a_word_by_default(self, words_only):
        assert words_only.split("🇫🇷🇩🇪").texts == ()

    def test_each_punctuation_mark_is_its_own_word_when_kept(self, with_punctuation):
        assert with_punctuation.split("Hello, world!").texts == (
            "Hello",
            ",",
            "world",
            "!",
        )

    def test_punctuation_is_dropped_by_default(self, words_only):
        assert words_only.split("Hello, world!").texts == ("Hello", "world")


class TestOffsets:
    @pytest.mark.parametrize(
        "text",
        [
            "can't stop",
            "3.14 and 1,000",
            "e.g. this",
            "日本語 とひらがな",
            "résumé résumé",
            "🇫🇷🇩🇪 Hello, world!",
            "a\r\nb\tc  d",
        ],
    )
    def test_every_word_is_its_own_source_slice(self, text):
        for splitter in (
            UnicodeWordPreTokenizer(),
            UnicodeWordPreTokenizer(keep_punctuation=True),
        ):
            assert all(
                text[word.start : word.end] == word.text
                for word in splitter.split(text)
            )

    @pytest.mark.parametrize(
        "text",
        [
            "can't stop",
            "3.14 and 1,000",
            "日本語 とひらがな",
            "résumé",
            "🇫🇷🇩🇪🇯",
            "a\r\nb\tc  d e",
            "́a",
            "   ",
        ],
    )
    def test_the_segments_tile_the_text(self, text):
        """The observable route: nothing is dropped and nothing overlaps."""
        segments = UnicodeWordPreTokenizer().segments_of(text)

        assert "".join(segments.texts) == text
        assert [segment.start for segment in segments][1:] == [
            segment.end for segment in segments
        ][:-1]

    def test_the_words_are_a_subsequence_of_the_segments(self):
        text = "e.g. this: 3.14, 🇫🇷!"
        splitter = UnicodeWordPreTokenizer(keep_punctuation=True)
        segments = list(splitter.segments_of(text))

        assert all(word in segments for word in splitter.split(text))

    def test_an_empty_text_has_no_segments_and_no_words(self, words_only):
        assert words_only.segments_of("").texts == ()
        assert words_only.split("").texts == ()


class TestNewlinesAndSpaces:
    def test_carriage_return_and_line_feed_stay_together(self, words_only):
        """WB3."""
        assert words_only.segments_of("a\r\nb").texts == ("a", "\r\n", "b")

    def test_breaks_on_both_sides_of_a_newline(self, words_only):
        """WB3a and WB3b, before WB4 can hide a mark after the newline."""
        assert words_only.segments_of("a\ńb").texts == ("a", "\n", "́", "b")

    @pytest.mark.parametrize("newline", ["\n", "\r", "", "", "", " ", " "])
    def test_every_newline_character_ends_a_word(self, words_only, newline):
        assert words_only.split(f"a{newline}b").texts == ("a", "b")

    def test_horizontal_white_space_stays_together(self, words_only):
        """WB3d."""
        assert words_only.segments_of("a  　b").texts == ("a", "  　", "b")

    def test_a_tab_is_not_a_word_segment_space_and_stands_alone(self, words_only):
        """The annex leaves TAB as Other, so two tabs are two segments."""
        assert words_only.segments_of("a\t\tb").texts == ("a", "\t", "\t", "b")

    def test_white_space_segments_are_never_words(self, with_punctuation):
        assert with_punctuation.split("a \t\n　b").texts == ("a", "b")


class TestMarksAndFormatCharacters:
    def test_a_combining_mark_attaches_to_the_letter_before_it(self, words_only):
        """WB4."""
        assert words_only.segments_of("á b").texts == ("á", " ", "b")

    def test_a_mark_at_the_start_has_nothing_to_attach_to(self, with_punctuation):
        """WB4 does not apply after sot, so the mark is a segment of its own."""
        assert with_punctuation.segments_of("́a").texts == ("́", "a")
        assert with_punctuation.split("́a").texts == ("́", "a")

    def test_a_lone_mark_is_not_a_word_by_default(self, words_only):
        assert words_only.split("́a").texts == ("a",)

    def test_a_mark_between_letters_does_not_break_the_word(self, words_only):
        assert words_only.split("ab́cd").texts == ("ab́cd",)

    def test_a_format_character_is_hidden_too(self, words_only):
        assert words_only.split("ab‌cd").texts == ("ab‌cd",)

    def test_a_skin_tone_modifier_attaches_to_its_emoji(self, with_punctuation):
        assert with_punctuation.split("👍🏽").texts == ("👍🏽",)

    def test_the_zero_width_joiner_is_a_format_character_so_emoji_sequences_split(
        self, with_punctuation
    ):
        """The stated simplification: no WB3c, since Python has no
        Extended_Pictographic. The joiner attaches to the man; the woman is
        another segment."""
        assert with_punctuation.split("👨‍👩").texts == ("👨‍", "👩")


class TestLetters:
    @pytest.mark.parametrize("text", ["don't", "o'clock", "can’t", "e.g", "a:b", "a·b"])
    def test_a_single_mid_character_between_letters_does_not_break(
        self, words_only, text
    ):
        """WB6 and WB7."""
        assert words_only.split(text).texts == (text,)

    def test_two_mid_characters_in_a_row_do_break(self, with_punctuation):
        """WB6 reaches across exactly one."""
        assert with_punctuation.split("a..b").texts == ("a", ".", ".", "b")

    def test_a_mid_character_with_no_letter_after_it_breaks(self, with_punctuation):
        assert with_punctuation.split("end.").texts == ("end", ".")

    def test_a_mid_character_with_no_letter_before_it_breaks(self, with_punctuation):
        assert with_punctuation.split("'tis").texts == ("'", "tis")

    def test_hangul_syllables_are_letters(self, words_only):
        """Korean is written with spaces; the annex keeps its syllables in ALetter."""
        assert words_only.split("한국어 텍스트").texts == ("한국어", "텍스트")

    def test_hiragana_is_left_to_dictionary_segmentation(self, words_only):
        assert words_only.split("ひらがな").texts == ("ひ", "ら", "が", "な")

    def test_thai_is_left_to_dictionary_segmentation(self, words_only):
        assert words_only.split("กข").texts == ("ก", "ข")

    def test_a_roman_numeral_is_a_letter(self, words_only):
        assert words_only.split("aⅫb").texts == ("aⅫb",)

    def test_a_double_quote_does_not_join_latin_letters(self, with_punctuation):
        assert with_punctuation.split('"hello"').texts == ('"', "hello", '"')


class TestHebrew:
    def test_a_gershayim_or_double_quote_inside_a_hebrew_word_does_not_break(
        self, words_only
    ):
        """WB7b and WB7c: the abbreviation for doctor stays one word."""
        assert words_only.split('ד"ר כהן').texts == ('ד"ר', "כהן")

    def test_a_trailing_geresh_or_apostrophe_stays_with_its_letter(self, words_only):
        """WB7a."""
        assert words_only.split("א' ב").texts == ("א'", "ב")

    def test_the_geresh_itself_is_a_letter(self, words_only):
        assert words_only.split("א׳").texts == ("א׳",)


class TestNumbers:
    def test_digits_join_digits(self, words_only):
        """WB8."""
        assert words_only.split("12345").texts == ("12345",)

    def test_letters_and_digits_join_in_either_order(self, words_only):
        """WB9 and WB10."""
        assert words_only.split("ab12cd 12ab").texts == ("ab12cd", "12ab")

    @pytest.mark.parametrize(
        "number", ["3.14", "1,000", "1'000", "1’000", "1;2", "1٬000"]
    )
    def test_a_single_mid_character_between_digits_does_not_break(
        self, words_only, number
    ):
        """WB11 and WB12, including the Swiss apostrophe separator."""
        assert words_only.split(number).texts == (number,)

    def test_a_trailing_or_leading_period_breaks_from_a_number(self, with_punctuation):
        assert with_punctuation.split("3. .5").texts == ("3", ".", ".", "5")

    def test_a_space_between_digit_groups_breaks(self, words_only):
        assert words_only.split("1 000").texts == ("1", "000")

    def test_a_narrow_no_break_space_between_digit_groups_does_not(self, words_only):
        """U+202F is ExtendNumLet to the annex, whatever its General Category says."""
        assert words_only.split("1 000").texts == ("1 000",)

    def test_a_vulgar_fraction_is_a_word_on_its_own(self, words_only):
        """Category No: not Numeric to the rules, but a number to the filter."""
        assert words_only.split("1½").texts == ("1", "½")

    def test_a_mid_number_character_does_not_join_letters(self, with_punctuation):
        assert with_punctuation.split("a,b").texts == ("a", ",", "b")

    def test_a_mid_letter_character_does_not_join_digits(self, with_punctuation):
        assert with_punctuation.split("1:2").texts == ("1", ":", "2")


class TestKatakana:
    def test_katakana_joins_katakana(self, words_only):
        """WB13."""
        assert words_only.split("カタカナ").texts == ("カタカナ",)

    def test_halfwidth_katakana_joins_too(self, words_only):
        assert words_only.split("ﾊﾝｶｼ").texts == ("ﾊﾝｶｼ",)

    def test_katakana_does_not_join_latin_letters(self, words_only):
        assert words_only.split("カa").texts == ("カ", "a")

    def test_the_middle_dot_is_not_katakana(self, words_only):
        assert words_only.split("アイ・ウエ").texts == ("アイ", "ウエ")


class TestExtendNumLet:
    @pytest.mark.parametrize("text", ["a_", "_a", "1_", "_1", "カ_", "a_1", "a__b"])
    def test_an_underscore_joins_letters_digits_and_katakana_on_either_side(
        self, words_only, text
    ):
        """WB13a and WB13b."""
        assert words_only.split(text).texts == (text,)

    def test_a_lone_underscore_is_not_a_word_by_default(self, words_only):
        assert words_only.split("_").texts == ()

    def test_a_lone_underscore_is_a_word_with_punctuation_kept(self, with_punctuation):
        assert with_punctuation.split("_").texts == ("_",)

    def test_an_underscore_does_not_join_ideographs(self, with_punctuation):
        assert with_punctuation.split("日_").texts == ("日", "_")


class TestRegionalIndicators:
    def test_pairs_of_regional_indicators_are_flags(self, with_punctuation):
        """WB15 and WB16: a break only after an even run."""
        assert with_punctuation.segments_of("🇫🇷🇩🇪").texts == ("🇫🇷", "🇩🇪")

    def test_an_odd_indicator_at_the_end_stands_alone(self, with_punctuation):
        assert with_punctuation.split("🇫🇷🇩🇪🇯").texts == ("🇫🇷", "🇩🇪", "🇯")

    def test_a_letter_before_the_run_restarts_the_count(self, with_punctuation):
        assert with_punctuation.split("a🇫🇷🇩").texts == ("a", "🇫🇷", "🇩")

    def test_a_hidden_mark_inside_a_pair_does_not_split_it(self, with_punctuation):
        assert with_punctuation.split("🇫‍🇷").texts == ("🇫‍🇷",)


class TestKeepPunctuation:
    def test_defaults_to_false(self):
        assert UnicodeWordPreTokenizer().keep_punctuation is False

    def test_a_segment_needs_a_letter_or_number_to_be_a_word_by_default(
        self, words_only
    ):
        assert words_only.split("!!! ... ---").texts == ()

    def test_every_non_space_segment_is_a_word_when_kept(self, with_punctuation):
        assert with_punctuation.split("!!! ...").texts == ("!", "!", "!", ".", ".", ".")

    def test_an_emoji_is_a_word_only_when_punctuation_is_kept(self):
        assert UnicodeWordPreTokenizer().split("👍").texts == ()
        assert UnicodeWordPreTokenizer(keep_punctuation=True).split("👍").texts == (
            "👍",
        )

    def test_a_non_boolean_is_refused(self):
        with pytest.raises(ValidationError):
            UnicodeWordPreTokenizer(keep_punctuation="sometimes")  # type: ignore[arg-type]

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            UnicodeWordPreTokenizer(locale="en")  # type: ignore[call-arg]

    def test_a_non_string_is_refused(self, words_only):
        with pytest.raises(InvalidValuesError):
            words_only.split(b"bytes")  # type: ignore[arg-type]

    def test_segments_of_refuses_a_non_string_too(self, words_only):
        with pytest.raises(InvalidValuesError):
            words_only.segments_of(None)  # type: ignore[arg-type]
