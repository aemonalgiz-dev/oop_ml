"""Word boundaries by the Unicode rules, which is what a double-click selects.

The idea
--------
Unicode Standard Annex #29 answers "where does one word stop and the next begin"
for every script at once, with a small table of character classes and a short
list of numbered rules. Each rule looks at a boundary between two characters and
says whether to break there: a letter followed by a letter, no (WB5); a letter,
a single apostrophe and another letter, no (WB6 and WB7, which is what keeps
``can't`` whole); a digit, a comma or a period and another digit, no (WB11 and
WB12, which is what keeps ``1,000`` and ``3.14`` whole); anything not covered by
a rule, yes (WB999). Combining marks and format characters are made invisible to
the rules first (WB4), so ``re\u0301sume\u0301``, written with combining accents,
is one word exactly as ``résumé`` is.

The rules are stated on a boundary, so the algorithm is one pass over the text
asking each of the ``n - 1`` boundaries in turn, and the answer is a list of
segments that tile the text. Worked on ``"can't stop"``, positions 0 to 9::

    c a n ' t   s t o p
    0 1 2 3 4 5 6 7 8 9

Boundaries 1 and 2 are letter-letter (WB5, keep). Boundary 3 is a letter before
an apostrophe with a letter after it (WB6, keep) and boundary 4 is the mirror of
that (WB7, keep). Boundary 5 is a letter before a space: no rule, so WB999
breaks. Boundary 6 is a space before a letter: WB999 again. Boundaries 7 to 9
are WB5. Three segments, ``can't`` at ``[0, 5)``, the space at ``[5, 6)`` and
``stop`` at ``[6, 10)``, and two of them are words.

Segments and words are two routes
---------------------------------
:meth:`UnicodeWordPreTokenizer.segments_of` answers with every segment, spaces
and punctuation included, and the segments always reproduce the text when
joined. :meth:`UnicodeWordPreTokenizer.split` keeps the ones that are words.
With ``keep_punctuation`` false, a segment is a word if it holds at least one
letter or number (General Category ``L*`` or ``N*``) or a Katakana character;
with it true, every segment that does not begin with white space is a word, so
each punctuation mark is a word of its own, since WB999 breaks between any two
of them. A flag emoji is two regional indicators kept together by WB15, neither
of them a letter, so it is a word only with punctuation kept.

Not ICU: which classes are approximated
---------------------------------------
Python exposes ``unicodedata.category`` and nothing else the annex needs, so
the word-break classes here are built from General Category and explicit
character lists, not from the ``Word_Break`` property table ICU ships. The
rules are the annex's default rules WB1 to WB999 for Unicode 15, complete; the
*classes* are approximations, and these are the ones that are:

- ``Extend`` is categories ``Mn``, ``Mc``, ``Me`` and the five emoji skin-tone
  modifiers. The property is ``Grapheme_Extend`` plus spacing marks, which
  agrees with that on every character the tests reach.
- ``Format`` is category ``Cf``, and that includes U+200D ZERO WIDTH JOINER,
  which the annex gives a class of its own so that WB3c can keep an emoji ZWJ
  sequence together. Here the joiner is absorbed into the character before it
  by WB4 and the sequence breaks after it: ``👨\u200d👩`` is two segments.
- ``ALetter`` is category ``L*`` or ``Nl``, minus the Hebrew letters (their own
  class), minus Katakana and Hiragana, minus the ideographic ranges (CJK
  ideographs, Tangut, Nushu), and minus the blocks the annex excludes because
  their scripts need a dictionary to segment: Thai, Lao, Myanmar, Khmer and
  the Tai scripts. A character in any of those is ``Other``, so WB999 breaks
  on both sides of it and every CJK ideograph or Hiragana character becomes a
  single-character word. That is the annex's own position -- it leaves those
  scripts to dictionary segmentation -- and not a bug in the table. Hangul is
  *not* excluded, because Korean is written with spaces and the annex keeps its
  syllables as letters.
- ``Katakana`` is the ranges the property table lists: the main block less the
  middle dot, the phonetic extensions, the circled and squared forms, the
  halfwidth forms, and the archaic letters in the Kana Supplement.
- ``Numeric`` is category ``Nd``; ``ExtendNumLet`` is category ``Pc`` plus the
  narrow no-break space; ``WSegSpace`` is the rest of ``Zs``.
- ``MidLetter``, ``MidNum``, ``MidNumLet``, ``Single_Quote`` and
  ``Double_Quote`` are the property table's own lists, transcribed. U+0027 is
  ``Single_Quote`` and U+2019 is ``MidNumLet``, so both join letters (WB6, WB7)
  and digits (WB11, WB12): ``1'000`` stays whole, as the annex intends for the
  Swiss thousands separator.
- ``Regional_Indicator`` is U+1F1E6 to U+1F1FF; CR, LF and ``Newline`` are the
  annex's own lists.

Cost
----
One decision per boundary, so linear in the text, except that WB4 has each
decision look back over any run of ignored marks, which is quadratic in the
length of that run. A table of each position's nearest non-ignored predecessor,
built once, is the usual repair; a run of marks long enough to matter is not a
word anyone has written.

The tie rule
------------
There is none to state: the rules are tried in the annex's order and the first
that applies decides, and WB999 always applies. Two rules never disagree
because the earlier one wins by construction.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from enum import StrEnum

from pydantic import Field

from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    PreTokenizer,
    checked_text,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words


class WordBreakClass(StrEnum):
    """The annex's ``Word_Break`` property values, spelled out.

    The annex's own names are given beside each so the rules can be read
    against it: CR, LF, Newline, Extend, Format, Katakana, Hebrew_Letter,
    ALetter, Single_Quote, Double_Quote, MidLetter, MidNum, MidNumLet, Numeric,
    ExtendNumLet, Regional_Indicator, WSegSpace, and Other for everything else.
    """

    CARRIAGE_RETURN = "carriage_return"
    LINE_FEED = "line_feed"
    NEWLINE = "newline"
    EXTEND = "extend"
    FORMAT = "format"
    KATAKANA = "katakana"
    HEBREW_LETTER = "hebrew_letter"
    ALPHABETIC_LETTER = "alphabetic_letter"
    SINGLE_QUOTE = "single_quote"
    DOUBLE_QUOTE = "double_quote"
    MID_LETTER = "mid_letter"
    MID_NUMBER = "mid_number"
    MID_NUMBER_LETTER = "mid_number_letter"
    NUMERIC = "numeric"
    EXTEND_NUMBER_LETTER = "extend_number_letter"
    REGIONAL_INDICATOR = "regional_indicator"
    WORD_SEGMENT_SPACE = "word_segment_space"
    OTHER = "other"


NEWLINE_CHARACTERS = frozenset("\u000b\u000c\u0085\u2028\u2029")
"""The annex's ``Newline`` class: vertical tab, form feed, next line, and the
line and paragraph separators. CR and LF have classes of their own."""

MID_LETTER_CHARACTERS = frozenset(
    "\u003a\u00b7\u0387\u055f\u05f4\u2027\ufe13\ufe55\uff1a"
)
"""Colon, the middle dots, the Armenian abbreviation mark, gershayim, the
hyphenation point, and the vertical and fullwidth colons."""
MID_NUMBER_CHARACTERS = frozenset(
    "\u002c\u003b\u037e\u0589\u060c\u060d\u066c\u07f8\u2044"
    "\ufe10\ufe14\ufe50\ufe54\uff0c\uff1b"
)
"""Comma, semicolon, the Greek question mark, the Armenian full stop, the Arabic
comma, date separator and thousands separator, the N'Ko comma, the fraction
slash, and the vertical, small and fullwidth forms."""
MID_NUMBER_LETTER_CHARACTERS = frozenset("\u002e\u2018\u2019\u2024\ufe52\uff07\uff0e")
"""Full stop, the curly single quotes, one dot leader, and the small and
fullwidth full stops and apostrophe."""
SINGLE_QUOTE_CHARACTER = "\u0027"
DOUBLE_QUOTE_CHARACTER = "\u0022"
HEBREW_GERESH = "\u05f3"
"""Listed as ``ALetter`` by the property table, alone among the punctuation."""
NARROW_NO_BREAK_SPACE = "\u202f"
"""Category ``Zs``, but ``ExtendNumLet`` to the annex, so a French thousands
group written with it, ``1\u202f000``, is one word."""

CodePointRange = tuple[int, int]

REGIONAL_INDICATOR_RANGE: CodePointRange = (0x1F1E6, 0x1F1FF)
EMOJI_MODIFIER_RANGE: CodePointRange = (0x1F3FB, 0x1F3FF)
HIRAGANA_RANGE: CodePointRange = (0x3040, 0x309F)

KATAKANA_RANGES: tuple[CodePointRange, ...] = (
    (0x3031, 0x3035),
    (0x309B, 0x309C),
    (0x30A0, 0x30FA),
    (0x30FC, 0x30FF),
    (0x31F0, 0x31FF),
    (0x32D0, 0x32FE),
    (0x3300, 0x3357),
    (0xFF66, 0xFF9D),
    (0x1B000, 0x1B000),
    (0x1B120, 0x1B122),
    (0x1B155, 0x1B155),
    (0x1B164, 0x1B167),
    (0x1F201, 0x1F202),
    (0x1F213, 0x1F213),
)

HEBREW_RANGES: tuple[CodePointRange, ...] = ((0x0591, 0x05FF), (0xFB1D, 0xFB4F))

IDEOGRAPHIC_RANGES: tuple[CodePointRange, ...] = (
    (0x2E80, 0x2FDF),
    (0x3006, 0x3007),
    (0x3021, 0x3029),
    (0x3038, 0x303B),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x17000, 0x18D7F),
    (0x1B170, 0x1B2FF),
    (0x20000, 0x323AF),
)

DICTIONARY_SCRIPT_RANGES: tuple[CodePointRange, ...] = (
    (0x0E00, 0x0E7F),
    (0x0E80, 0x0EFF),
    (0x1000, 0x109F),
    (0x1780, 0x17FF),
    (0x1950, 0x197F),
    (0x1980, 0x19DF),
    (0x1A20, 0x1AAF),
    (0xA9E0, 0xA9FF),
    (0xAA60, 0xAA7F),
    (0xAA80, 0xAADF),
)
"""Thai, Lao, Myanmar, Khmer and the Tai scripts: letters the annex leaves out of
``ALetter`` because their words need a dictionary to find."""


def _in_range(code_point: int, code_point_range: CodePointRange) -> bool:
    low, high = code_point_range
    return low <= code_point <= high


def _in_any(code_point: int, ranges: Sequence[CodePointRange]) -> bool:
    return any(_in_range(code_point, code_point_range) for code_point_range in ranges)


def word_break_class_of(character: str) -> WordBreakClass:
    """The word-break class of one character, from General Category and the lists above.

    Parameters
    ----------
    character:
        Exactly one character.
    """
    if character == "\r":
        return WordBreakClass.CARRIAGE_RETURN
    if character == "\n":
        return WordBreakClass.LINE_FEED
    if character in NEWLINE_CHARACTERS:
        return WordBreakClass.NEWLINE
    if character == SINGLE_QUOTE_CHARACTER:
        return WordBreakClass.SINGLE_QUOTE
    if character == DOUBLE_QUOTE_CHARACTER:
        return WordBreakClass.DOUBLE_QUOTE
    if character in MID_LETTER_CHARACTERS:
        return WordBreakClass.MID_LETTER
    if character in MID_NUMBER_CHARACTERS:
        return WordBreakClass.MID_NUMBER
    if character in MID_NUMBER_LETTER_CHARACTERS:
        return WordBreakClass.MID_NUMBER_LETTER
    if character == HEBREW_GERESH:
        return WordBreakClass.ALPHABETIC_LETTER
    if character == NARROW_NO_BREAK_SPACE:
        return WordBreakClass.EXTEND_NUMBER_LETTER

    code_point = ord(character)
    if _in_range(code_point, REGIONAL_INDICATOR_RANGE):
        return WordBreakClass.REGIONAL_INDICATOR
    if _in_range(code_point, EMOJI_MODIFIER_RANGE):
        return WordBreakClass.EXTEND
    if _in_any(code_point, KATAKANA_RANGES):
        return WordBreakClass.KATAKANA

    category = unicodedata.category(character)
    if category in ("Mn", "Mc", "Me"):
        return WordBreakClass.EXTEND
    if category == "Cf":
        return WordBreakClass.FORMAT
    if category == "Zs":
        return WordBreakClass.WORD_SEGMENT_SPACE
    if category == "Nd":
        return WordBreakClass.NUMERIC
    if category == "Pc":
        return WordBreakClass.EXTEND_NUMBER_LETTER
    if category.startswith("L") or category == "Nl":
        if _in_any(code_point, HEBREW_RANGES):
            return WordBreakClass.HEBREW_LETTER
        if _in_range(code_point, HIRAGANA_RANGE):
            return WordBreakClass.OTHER
        if _in_any(code_point, IDEOGRAPHIC_RANGES):
            return WordBreakClass.OTHER
        if _in_any(code_point, DICTIONARY_SCRIPT_RANGES):
            return WordBreakClass.OTHER
        return WordBreakClass.ALPHABETIC_LETTER
    return WordBreakClass.OTHER


NEWLINE_CLASSES = frozenset(
    {WordBreakClass.CARRIAGE_RETURN, WordBreakClass.LINE_FEED, WordBreakClass.NEWLINE}
)
IGNORED_CLASSES = frozenset({WordBreakClass.EXTEND, WordBreakClass.FORMAT})
"""What WB4 makes invisible to every later rule."""
AH_LETTER_CLASSES = frozenset(
    {WordBreakClass.ALPHABETIC_LETTER, WordBreakClass.HEBREW_LETTER}
)
"""The annex's ``AHLetter``: ALetter or Hebrew_Letter."""
MID_LETTER_OR_MID_NUMBER_LETTER_Q = frozenset(
    {
        WordBreakClass.MID_LETTER,
        WordBreakClass.MID_NUMBER_LETTER,
        WordBreakClass.SINGLE_QUOTE,
    }
)
"""What WB6 and WB7 let a letter reach across: MidLetter or ``MidNumLetQ``."""
MID_NUMBER_OR_MID_NUMBER_LETTER_Q = frozenset(
    {
        WordBreakClass.MID_NUMBER,
        WordBreakClass.MID_NUMBER_LETTER,
        WordBreakClass.SINGLE_QUOTE,
    }
)
"""What WB11 and WB12 let a digit reach across: MidNum or ``MidNumLetQ``."""
JOINS_EXTEND_NUMBER_LETTER = AH_LETTER_CLASSES | {
    WordBreakClass.NUMERIC,
    WordBreakClass.KATAKANA,
}
"""What WB13a and WB13b join to an ExtendNumLet on either side."""


class UnicodeWordPreTokenizer(PreTokenizer):
    """Words by Unicode Standard Annex #29's default word boundary rules.

    Parameters
    ----------
    keep_punctuation:
        False, the default, keeps only the segments holding a letter, a number
        or a Katakana character. True keeps every segment that does not begin
        with white space, so each punctuation mark and each emoji is a word.
    """

    keep_punctuation: bool = Field(default=False)

    def segments_of(self, text: str) -> Words:
        """Every segment the rules find, white space and punctuation included.

        The observable route: the segments tile ``text`` exactly, so joining
        their texts reproduces it, and :meth:`split` is these with the
        non-words dropped.

        Raises
        ------
        InvalidValuesError
            If ``text`` is not a string.
        """
        text = checked_text(text)
        classes = [word_break_class_of(character) for character in text]
        segments: list[Word] = []
        segment_start = 0
        for position in range(1, len(text)):
            if _breaks_before(classes, position):
                segments.append(Word.of(text, segment_start, position))
                segment_start = position
        if text:
            segments.append(Word.of(text, segment_start, len(text)))
        return Words(segments)

    def _words_of(self, text: str) -> Words:
        return Words(
            [segment for segment in self.segments_of(text) if self._is_word(segment)]
        )

    def _is_word(self, segment: Word) -> bool:
        if self.keep_punctuation:
            return not segment.text[0].isspace()
        return any(
            unicodedata.category(character)[0] in ("L", "N")
            or word_break_class_of(character) is WordBreakClass.KATAKANA
            for character in segment.text
        )


def _breaks_before(classes: Sequence[WordBreakClass], position: int) -> bool:
    """Whether the annex breaks before ``classes[position]``.

    The rules in the annex's order; the first that applies decides.
    """
    previous = classes[position - 1]
    current = classes[position]

    # WB3: CR × LF
    if (
        previous is WordBreakClass.CARRIAGE_RETURN
        and current is WordBreakClass.LINE_FEED
    ):
        return False
    # WB3a: (Newline | CR | LF) ÷
    if previous in NEWLINE_CLASSES:
        return True
    # WB3b: ÷ (Newline | CR | LF)
    if current in NEWLINE_CLASSES:
        return True
    # WB3d: WSegSpace × WSegSpace, on the literal neighbours, before WB4 hides anything
    if (
        previous is WordBreakClass.WORD_SEGMENT_SPACE
        and current is WordBreakClass.WORD_SEGMENT_SPACE
    ):
        return False
    # WB4: X (Extend | Format)* → X, so never break before one of them
    if current in IGNORED_CLASSES:
        return False

    nearest, second = _left_context(classes, position)
    following = _right_context(classes, position)

    # WB5: AHLetter × AHLetter
    if nearest in AH_LETTER_CLASSES and current in AH_LETTER_CLASSES:
        return False
    # WB6: AHLetter × (MidLetter | MidNumLetQ) AHLetter
    if (
        nearest in AH_LETTER_CLASSES
        and current in MID_LETTER_OR_MID_NUMBER_LETTER_Q
        and following in AH_LETTER_CLASSES
    ):
        return False
    # WB7: AHLetter (MidLetter | MidNumLetQ) × AHLetter
    if (
        second in AH_LETTER_CLASSES
        and nearest in MID_LETTER_OR_MID_NUMBER_LETTER_Q
        and current in AH_LETTER_CLASSES
    ):
        return False
    # WB7a: Hebrew_Letter × Single_Quote
    if (
        nearest is WordBreakClass.HEBREW_LETTER
        and current is WordBreakClass.SINGLE_QUOTE
    ):
        return False
    # WB7b: Hebrew_Letter × Double_Quote Hebrew_Letter
    if (
        nearest is WordBreakClass.HEBREW_LETTER
        and current is WordBreakClass.DOUBLE_QUOTE
        and following is WordBreakClass.HEBREW_LETTER
    ):
        return False
    # WB7c: Hebrew_Letter Double_Quote × Hebrew_Letter
    if (
        second is WordBreakClass.HEBREW_LETTER
        and nearest is WordBreakClass.DOUBLE_QUOTE
        and current is WordBreakClass.HEBREW_LETTER
    ):
        return False
    # WB8: Numeric × Numeric
    if nearest is WordBreakClass.NUMERIC and current is WordBreakClass.NUMERIC:
        return False
    # WB9: AHLetter × Numeric
    if nearest in AH_LETTER_CLASSES and current is WordBreakClass.NUMERIC:
        return False
    # WB10: Numeric × AHLetter
    if nearest is WordBreakClass.NUMERIC and current in AH_LETTER_CLASSES:
        return False
    # WB11: Numeric (MidNum | MidNumLetQ) × Numeric
    if (
        second is WordBreakClass.NUMERIC
        and nearest in MID_NUMBER_OR_MID_NUMBER_LETTER_Q
        and current is WordBreakClass.NUMERIC
    ):
        return False
    # WB12: Numeric × (MidNum | MidNumLetQ) Numeric
    if (
        nearest is WordBreakClass.NUMERIC
        and current in MID_NUMBER_OR_MID_NUMBER_LETTER_Q
        and following is WordBreakClass.NUMERIC
    ):
        return False
    # WB13: Katakana × Katakana
    if nearest is WordBreakClass.KATAKANA and current is WordBreakClass.KATAKANA:
        return False
    # WB13a: (AHLetter | Numeric | Katakana | ExtendNumLet) × ExtendNumLet
    if (
        nearest in JOINS_EXTEND_NUMBER_LETTER
        or nearest is WordBreakClass.EXTEND_NUMBER_LETTER
    ) and current is WordBreakClass.EXTEND_NUMBER_LETTER:
        return False
    # WB13b: ExtendNumLet × (AHLetter | Numeric | Katakana)
    if (
        nearest is WordBreakClass.EXTEND_NUMBER_LETTER
        and current in JOINS_EXTEND_NUMBER_LETTER
    ):
        return False
    # WB15 and WB16: an odd run of regional indicators × Regional_Indicator
    inside_a_flag = (
        nearest is WordBreakClass.REGIONAL_INDICATOR
        and current is WordBreakClass.REGIONAL_INDICATOR
        and _regional_indicators_ending_before(classes, position) % 2 == 1
    )
    # WB999: Any ÷ Any
    return not inside_a_flag


def _left_context(
    classes: Sequence[WordBreakClass], position: int
) -> tuple[WordBreakClass | None, WordBreakClass | None]:
    """The two nearest classes before ``position`` once WB4 has hidden the ignored ones.

    ``None`` stands for the start of the text. A mark at the very start is not
    absorbed by anything, which the annex states as WB4 not applying after sot,
    and here as the walk finding nothing to attach it to.
    """
    found: list[WordBreakClass] = []
    index = position - 1
    while index >= 0 and len(found) < 2:
        if classes[index] not in IGNORED_CLASSES:
            found.append(classes[index])
        index -= 1
    nearest = found[0] if found else None
    second = found[1] if len(found) > 1 else None
    return nearest, second


def _right_context(
    classes: Sequence[WordBreakClass], position: int
) -> WordBreakClass | None:
    """The nearest class after ``position`` once WB4 has hidden the ignored ones."""
    for index in range(position + 1, len(classes)):
        if classes[index] not in IGNORED_CLASSES:
            return classes[index]
    return None


def _regional_indicators_ending_before(
    classes: Sequence[WordBreakClass], position: int
) -> int:
    """How many regional indicators run up to ``position``, ignoring what WB4 hides."""
    count = 0
    index = position - 1
    while index >= 0:
        if classes[index] in IGNORED_CLASSES:
            index -= 1
            continue
        if classes[index] is not WordBreakClass.REGIONAL_INDICATOR:
            break
        count += 1
        index -= 1
    return count
