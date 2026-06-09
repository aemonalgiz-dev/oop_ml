"""Spec for RunSegmenter -- whitespace ends a run, and offsets are in the original text.

Tested through the smallest concrete subclass, one that cuts a run into pairs
of characters, so that a failure here is a failure of the template and not of
any segmenter's rule.
"""

import pytest

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.segmentation.runs import (
    RunSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words


class PairSegmenter(RunSegmenter):
    """Every two characters are a word; an odd character at the end stands alone."""

    def _words_of_run(self, run: str) -> tuple[str, ...]:
        return tuple(run[position : position + 2] for position in range(0, len(run), 2))


class TestRunSegmenter:
    def test_a_single_run_is_cut_by_the_rule(self):
        assert PairSegmenter().split("abcde").texts == ("ab", "cd", "e")

    def test_whitespace_ends_a_run_so_the_rule_never_reaches_across_it(self):
        assert PairSegmenter().split("abc de").texts == ("ab", "c", "de")

    def test_offsets_are_in_the_original_text(self):
        assert list(PairSegmenter().split("abc  de")) == [
            Word("ab", 0, 2),
            Word("c", 2, 3),
            Word("de", 5, 7),
        ]

    def test_unicode_whitespace_ends_a_run(self):
        """An ideographic space, U+3000, is whitespace too."""
        assert list(PairSegmenter().split("研究　生命")) == [
            Word("研究", 0, 2),
            Word("生命", 3, 5),
        ]

    def test_leading_and_trailing_whitespace_produce_no_words(self):
        assert PairSegmenter().split("  ab  ").texts == ("ab",)

    def test_a_blank_text_has_no_words(self):
        assert PairSegmenter().split("   ") == Words([])
        assert PairSegmenter().split("") == Words([])

    def test_a_non_string_is_refused_at_the_boundary(self):
        with pytest.raises(InvalidValuesError):
            PairSegmenter().split(["ab"])  # type: ignore[arg-type]

    def test_the_template_cannot_be_instantiated_without_a_rule(self):
        with pytest.raises(TypeError):
            RunSegmenter()  # type: ignore[abstract]
