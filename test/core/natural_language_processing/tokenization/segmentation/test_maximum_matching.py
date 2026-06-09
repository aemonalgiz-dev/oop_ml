"""Spec for MaximumMatchingSegmenter -- greedy longest match, forward, backward, or both.

Pinned against the two classic sentences worked in the module docstring: the
zoo sentence, where the directions disagree on the word count, and the
research sentence, where they tie on count and part on single characters.
"""

import pytest
from pydantic import ValidationError

from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (
    WordDictionary,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.maximum_matching import (
    MatchingDirection,
    MaximumMatchingSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word
from test.core.natural_language_processing.tokenization.segmentation.fixtures import (
    LIFE_READING,
    RESEARCH_SENTENCE,
    RESEARCH_WORDS,
    STUDENT_READING,
    ZOO_BACKWARD,
    ZOO_FORWARD,
    ZOO_SENTENCE,
    ZOO_WORDS,
)

ZOO = WordDictionary.from_words(ZOO_WORDS)
RESEARCH = WordDictionary.from_words(RESEARCH_WORDS)
THE_CAT = WordDictionary.from_words(["the", "cat", "sat"])


def segmenter(
    dictionary: WordDictionary, direction: MatchingDirection
) -> MaximumMatchingSegmenter:
    return MaximumMatchingSegmenter(dictionary=dictionary, direction=direction)


class TestForward:
    def test_takes_the_longest_word_starting_at_each_position(self):
        assert segmenter(ZOO, MatchingDirection.FORWARD).split(ZOO_SENTENCE).texts == (
            ZOO_FORWARD
        )

    def test_the_default_direction_is_forward(self):
        assert (
            MaximumMatchingSegmenter(dictionary=RESEARCH).split(RESEARCH_SENTENCE).texts
            == STUDENT_READING
        )

    def test_a_character_outside_the_dictionary_is_its_own_word(self):
        assert segmenter(THE_CAT, MatchingDirection.FORWARD).split(
            "thecatsatxyz"
        ).texts == ("the", "cat", "sat", "x", "y", "z")

    def test_a_dictionary_of_one_character_words_cuts_every_character(self):
        assert segmenter(
            WordDictionary.from_words(["a", "b"]), MatchingDirection.FORWARD
        ).split("abab").texts == ("a", "b", "a", "b")

    def test_a_run_shorter_than_the_longest_word_is_handled(self):
        assert segmenter(THE_CAT, MatchingDirection.FORWARD).split("th").texts == (
            "t",
            "h",
        )


class TestBackward:
    def test_takes_the_longest_word_ending_at_each_position(self):
        assert segmenter(ZOO, MatchingDirection.BACKWARD).split(ZOO_SENTENCE).texts == (
            ZOO_BACKWARD
        )

    def test_reads_the_research_sentence_as_life(self):
        assert (
            segmenter(RESEARCH, MatchingDirection.BACKWARD)
            .split(RESEARCH_SENTENCE)
            .texts
            == LIFE_READING
        )

    def test_words_come_back_in_source_order(self):
        assert segmenter(THE_CAT, MatchingDirection.BACKWARD).split(
            "xthecat"
        ).texts == ("x", "the", "cat")


class TestBidirectional:
    def test_prefers_fewer_words(self):
        """Forward gives six words, backward five; backward wins."""
        assert len(ZOO_FORWARD) == 6
        assert len(ZOO_BACKWARD) == 5
        assert (
            segmenter(ZOO, MatchingDirection.BIDIRECTIONAL).split(ZOO_SENTENCE).texts
            == ZOO_BACKWARD
        )

    def test_then_prefers_fewer_single_character_words(self):
        """Both readings have three words; forward's 命 is a single, backward has none."""
        assert len(LIFE_READING) == len(STUDENT_READING) == 3
        assert sum(len(word) == 1 for word in STUDENT_READING) == 1
        assert sum(len(word) == 1 for word in LIFE_READING) == 0
        assert (
            segmenter(RESEARCH, MatchingDirection.BIDIRECTIONAL)
            .split(RESEARCH_SENTENCE)
            .texts
            == LIFE_READING
        )

    def test_then_prefers_backward(self):
        """ab | c against a | bc: two words and one single each way. Backward."""
        dictionary = WordDictionary.from_words(["ab", "bc", "a", "c"])

        assert segmenter(dictionary, MatchingDirection.FORWARD).split("abc").texts == (
            "ab",
            "c",
        )
        assert segmenter(dictionary, MatchingDirection.BACKWARD).split("abc").texts == (
            "a",
            "bc",
        )
        assert segmenter(dictionary, MatchingDirection.BIDIRECTIONAL).split(
            "abc"
        ).texts == ("a", "bc")

    def test_prefers_forward_when_it_has_fewer_words(self):
        """abcd with {abc, bc, a, c, cd}: forward takes abc then d, two words;
        backward takes cd, then b alone, then a, three. Forward wins."""
        dictionary = WordDictionary.from_words(["abc", "bc", "a", "c", "cd"])

        assert segmenter(dictionary, MatchingDirection.FORWARD).split("abcd").texts == (
            "abc",
            "d",
        )
        assert segmenter(dictionary, MatchingDirection.BACKWARD).split(
            "abcd"
        ).texts == ("a", "b", "cd")
        assert segmenter(dictionary, MatchingDirection.BIDIRECTIONAL).split(
            "abcd"
        ).texts == ("abc", "d")

    def test_agrees_with_both_when_they_agree(self):
        text = "thecatsat"

        assert segmenter(THE_CAT, MatchingDirection.BIDIRECTIONAL).split(
            text
        ) == segmenter(THE_CAT, MatchingDirection.FORWARD).split(text)


class TestOffsetsAndWhitespace:
    def test_offsets_are_in_the_source_text(self):
        assert list(
            MaximumMatchingSegmenter(dictionary=THE_CAT).split("thecat sat")
        ) == [
            Word("the", 0, 3),
            Word("cat", 3, 6),
            Word("sat", 7, 10),
        ]

    def test_whitespace_ends_a_run_so_no_match_crosses_it(self):
        dictionary = WordDictionary.from_words(["the", "thecat"])

        assert MaximumMatchingSegmenter(dictionary=dictionary).split(
            "the cat"
        ).texts == (
            "the",
            "c",
            "a",
            "t",
        )
        assert MaximumMatchingSegmenter(dictionary=dictionary).split(
            "thecat"
        ).texts == ("thecat",)

    def test_a_blank_text_has_no_words(self):
        assert MaximumMatchingSegmenter(dictionary=THE_CAT).split("  ").n_words == 0

    def test_cjk_offsets(self):
        assert list(
            segmenter(RESEARCH, MatchingDirection.BACKWARD).split("研究生命起源 很好")
        ) == [
            Word("研究", 0, 2),
            Word("生命", 2, 4),
            Word("起源", 4, 6),
            Word("很", 7, 8),
            Word("好", 8, 9),
        ]


class TestConstruction:
    def test_the_dictionary_is_required(self):
        with pytest.raises(ValidationError):
            MaximumMatchingSegmenter()  # type: ignore[call-arg]

    def test_a_dictionary_of_the_wrong_type_is_refused(self):
        with pytest.raises(ValidationError):
            MaximumMatchingSegmenter(dictionary=["the", "cat"])  # type: ignore[arg-type]

    def test_an_unknown_direction_is_refused(self):
        with pytest.raises(ValidationError):
            MaximumMatchingSegmenter(dictionary=THE_CAT, direction="sideways")  # type: ignore[arg-type]

    def test_a_direction_may_be_given_by_value(self):
        assert (
            MaximumMatchingSegmenter(dictionary=THE_CAT, direction="backward").direction  # type: ignore[arg-type]
            is MatchingDirection.BACKWARD
        )

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValidationError):
            MaximumMatchingSegmenter(dictionary=THE_CAT, mode="forward")  # type: ignore[call-arg]
