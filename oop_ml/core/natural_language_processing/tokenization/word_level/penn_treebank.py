"""The Penn Treebank rules, which are what "tokenized English" has meant since 1993.

The idea
--------
When the Penn Treebank was annotated, its text was first run through a short
``sed`` script by Robert MacIntyre that decided what a word was, and every
parser trained on the treebank since has expected its input cut the same way.
The rules are few and every one of them is a repair to whitespace splitting.
Punctuation comes off the word it touches, so ``world,`` is ``world`` and ``,``.
A comma or colon *inside* a number does not, so ``1,000`` and ``3:30`` stay
whole. A period comes off only at the very end of the text, because the script
ran on one sentence at a time and a period anywhere else is an abbreviation's:
``said.`` in the middle of a text keeps its period, ``it?`` at the end loses its
question mark. Quotes are rewritten into the treebank's typewriter convention,
an opening ``"`` becoming two backquotes and a closing one two apostrophes, so a
parser can tell the two apart. And the English clitics are cut from their hosts
in the way that makes the grammar come out regular: ``It's`` is ``It`` and
``'s``, ``don't`` is ``do`` and ``n't``, and ``can't`` is ``ca`` and ``n't``,
which looks wrong and is right, since ``n't`` is one morpheme and what is left
is what is left.

Worked, on ``"Hello," she said. "It's 3.14, isn't it?"``
---------------------------------------------------------
Sixteen words. Each is shown with the span of source text it stands for::

    ``     [0, 1)     Hello  [1, 6)     ,      [6, 7)     ''     [7, 8)
    she    [9, 12)    said.  [13, 18)   ``     [19, 20)   It     [20, 22)
    's     [22, 24)   3.14   [25, 29)   ,      [29, 30)   is     [31, 33)
    n't    [33, 36)   it     [37, 39)   ?      [39, 40)   ''     [40, 41)

``said.`` keeps its period because it is not at the end of the text; ``3.14``
keeps its comma-free interior because the period is not final either and the
comma after it is followed by a space, not a digit. The two rewritten quotes
each stand for one source character, so their spans are one wide while their
texts are two: that is the case
:class:`~oop_ml.core.natural_language_processing.tokenization.words.Word` allows
``len(text) != end - start`` for.

Why a scanner and not the substitutions
---------------------------------------
The script, and NLTK's ``TreebankWordTokenizer`` which transcribes it, is a
sequence of regular-expression substitutions that insert spaces and then a
split on whitespace. That loses the offsets: once ``"`` has become ``` `` ```
nothing records where in the source it was. So this module is a scanner that
walks each whitespace-delimited chunk once and emits words with spans, and the
substitution pipeline survives only as the independent oracle in the tests,
where it checks the scanner's *texts* on a corpus of sentences.

The rules, as the scanner applies them
--------------------------------------
Some characters are words on their own wherever they stand: ``; @ # $ % &``,
``?`` and ``!``, the brackets ``( ) [ ] { } < >``, a ``"``, a pair of
apostrophes, a run of backquotes, a double dash ``--`` and an ellipsis ``...``.
A comma or colon is a word on its own unless a digit follows it. The final
period is a word on its own when it is the last character of the text apart
from closing quotes, brackets and white space, and is not itself preceded by a
period. Everything between those is a piece, and a piece is cut twice more: a
clitic at its end comes off (``'s 's 'm 'd 'll 're 've n't``, in either case,
and a lone trailing apostrophe), then the fixed contractions are cut wherever
they stand as words -- ``cannot``, ``d'ye``, ``gimme``, ``gonna``, ``gotta``,
``lemme``, ``more'n``, ``wanna`` (only when it ends the piece), and ``'tis``
and ``'twas`` (only when they begin one), each case-insensitively.

A ``"`` or ``''`` is an opening quote when it begins the text or follows white
space or an opening bracket, and a closing quote otherwise. With
``convert_brackets`` true the six brackets are rewritten to the treebank's
``-LRB- -RRB- -LSB- -RSB- -LCB- -RCB-``; ``<`` and ``>`` are separated but
never rewritten, as in the script.

Where this deliberately differs from the script
-----------------------------------------------
Four places, each recorded so the oracle comparison in the tests can avoid
them. The script opens a quote after a literal space and nothing else, so a
quote after a tab closes; here any white space opens it. The script opens only
a ``"`` at the very start of the text and leaves ``''`` there as closing; here
both open, since the two spellings are treated alike everywhere else. The
script's comma rule consumes the character after a split comma, so of ``,,``
only the first is examined and the second stays attached to whatever follows;
here every comma is examined. And the script separates exactly two backquotes
and leaves a third attached; here a run of any length is one word, which is
also what a single backquote at the start of a word becomes.

Cost and tie rules
------------------
Linear in the text: one pass over each chunk, one suffix check per piece, and
the ten contraction patterns tried once each per piece. There is no tie to
settle. The standalone characters are recognised left to right, a piece loses at
most one clitic and it is the one at its end, and the contractions are cut at
every place they match.
"""

from __future__ import annotations

import re
from itertools import pairwise

from pydantic import Field

from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    NON_WHITESPACE_RUN,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words

OPENING_QUOTE = "``"
CLOSING_QUOTE = "''"
BACKQUOTE = "`"
APOSTROPHE = "'"
DOUBLE_QUOTE = '"'
DOUBLE_DASH = "--"
ELLIPSIS = "..."
PERIOD = "."

ALWAYS_SEPARATE = frozenset(";@#$%&?!")
"""Characters that are a word on their own wherever they stand."""
COMMA_OR_COLON = frozenset(",:")
"""Separate unless a digit follows, so ``1,000`` and ``3:30`` stay whole."""
BRACKETS = frozenset("()[]{}<>")
OPENING_BRACKETS = frozenset("([{<")
"""What a quote may follow and still be an opening quote."""
FINAL_CLOSERS = frozenset("])}>\"'")
"""What may stand between the final period and the end of the text."""

BRACKET_NAMES = {
    "(": "-LRB-",
    ")": "-RRB-",
    "[": "-LSB-",
    "]": "-RSB-",
    "{": "-LCB-",
    "}": "-RCB-",
}
"""The treebank's spellings, used only when ``convert_brackets`` is true."""

CLITIC_SUFFIXES = (
    "'s",
    "'S",
    "'m",
    "'M",
    "'d",
    "'D",
    "'ll",
    "'LL",
    "'re",
    "'RE",
    "'ve",
    "'VE",
    "n't",
    "N'T",
    APOSTROPHE,
)
"""What comes off the end of a piece, in the script's two cases, plus a lone
trailing apostrophe. A piece loses at most one, and no two can end one piece."""

CONTRACTIONS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(can)(not)\b",
        r"\b(d)('ye)\b",
        r"\b(gim)(me)\b",
        r"\b(gon)(na)\b",
        r"\b(got)(ta)\b",
        r"\b(lem)(me)\b",
        r"\b(more)('n)\b",
        r"\b(wan)(na)$",
        r"^('t)(is)\b",
        r"^('t)(was)\b",
    )
)
"""MacIntyre's fixed list. Each is cut between its two groups, wherever it
stands as a word; ``wanna`` only at the end of a piece and ``'tis`` and
``'twas`` only at the start, as the script has them."""


class PennTreebankPreTokenizer(PreTokenizer):
    """The Penn Treebank word rules, as a scanner that keeps offsets.

    Parameters
    ----------
    convert_brackets:
        False, the default, leaves a separated bracket as itself. True rewrites
        the six round, square and curly brackets to the treebank's
        ``-LRB-``-style names; the span still covers the one source character.
    """

    convert_brackets: bool = Field(default=False)

    def _words_of(self, text: str) -> Words:
        final_period = _final_period_position(text)
        words: list[Word] = []
        for chunk in NON_WHITESPACE_RUN.finditer(text):
            words.extend(
                self._words_of_chunk(text, chunk.start(), chunk.end(), final_period)
            )
        return Words(words)

    def _words_of_chunk(
        self, text: str, start: int, end: int, final_period: int | None
    ) -> list[Word]:
        """Cut one chunk into standalone words and the pieces between them."""
        words: list[Word] = []
        piece_start: int | None = None
        position = start
        while position < end:
            standalone = self._standalone_word_at(text, position, final_period)
            if standalone is None:
                if piece_start is None:
                    piece_start = position
                position += 1
                continue
            if piece_start is not None:
                words.extend(_pieces_of(text, piece_start, position))
                piece_start = None
            words.append(standalone)
            position = standalone.end
        if piece_start is not None:
            words.extend(_pieces_of(text, piece_start, end))
        return words

    def _standalone_word_at(
        self, text: str, position: int, final_period: int | None
    ) -> Word | None:
        """The standalone word beginning at ``position``, or None.

        A standalone word is one that is a token whatever its neighbours are.
        """
        character = text[position]
        if character == BACKQUOTE:
            run_end = position
            while run_end < len(text) and text[run_end] == BACKQUOTE:
                run_end += 1
            return Word.of(text, position, run_end)
        if text.startswith(CLOSING_QUOTE, position):
            return Word(_quote_text(text, position), position, position + 2)
        if character == DOUBLE_QUOTE:
            return Word(_quote_text(text, position), position, position + 1)
        if text.startswith(DOUBLE_DASH, position):
            return Word.of(text, position, position + 2)
        if text.startswith(ELLIPSIS, position):
            return Word.of(text, position, position + 3)
        if character in ALWAYS_SEPARATE:
            return Word.of(text, position, position + 1)
        if character in BRACKETS:
            return Word(self._bracket_text(character), position, position + 1)
        if character in COMMA_OR_COLON and not _digit_follows(text, position):
            return Word.of(text, position, position + 1)
        if position == final_period:
            return Word.of(text, position, position + 1)
        return None

    def _bracket_text(self, bracket: str) -> str:
        if self.convert_brackets and bracket in BRACKET_NAMES:
            return BRACKET_NAMES[bracket]
        return bracket


def _quote_text(text: str, position: int) -> str:
    """Which quote a ``"`` or ``''`` at ``position`` is.

    Opening after the start of the text, white space or an opening bracket;
    closing otherwise.
    """
    if position == 0:
        return OPENING_QUOTE
    before = text[position - 1]
    if before.isspace() or before in OPENING_BRACKETS:
        return OPENING_QUOTE
    return CLOSING_QUOTE


def _digit_follows(text: str, position: int) -> bool:
    return position + 1 < len(text) and text[position + 1].isdecimal()


def _final_period_position(text: str) -> int | None:
    """Where the final period is, if the text ends in one apart from closers and space.

    The period must be preceded by something that is not a period, so a final
    ellipsis is left to the ellipsis rule.
    """
    position = len(text) - 1
    while position >= 0 and text[position].isspace():
        position -= 1
    while position >= 0 and text[position] in FINAL_CLOSERS:
        position -= 1
    if position >= 1 and text[position] == PERIOD and text[position - 1] != PERIOD:
        return position
    return None


def _pieces_of(text: str, start: int, end: int) -> list[Word]:
    """Cut a run of ordinary characters: its clitic, then its contractions."""
    clitic_start = _clitic_start(text[start:end])
    if clitic_start is None:
        return _contraction_pieces(text, start, end)
    return [
        *_contraction_pieces(text, start, start + clitic_start),
        Word.of(text, start + clitic_start, end),
    ]


def _clitic_start(piece: str) -> int | None:
    """Offset within ``piece`` where its trailing clitic begins, or None.

    The clitic needs a host: at least one character before it, and not an
    apostrophe, which is the script's ``[^' ]``.
    """
    for suffix in CLITIC_SUFFIXES:
        if (
            piece.endswith(suffix)
            and len(piece) > len(suffix)
            and piece[-len(suffix) - 1] != APOSTROPHE
        ):
            return len(piece) - len(suffix)
    return None


def _contraction_pieces(text: str, start: int, end: int) -> list[Word]:
    """Cut ``text[start:end]`` at every contraction boundary the fixed list finds."""
    piece = text[start:end]
    cuts = sorted(
        {match.end(1) for pattern in CONTRACTIONS for match in pattern.finditer(piece)}
    )
    bounds = [0, *cuts, len(piece)]
    return [
        Word.of(text, start + piece_start, start + piece_end)
        for piece_start, piece_end in pairwise(bounds)
        if piece_end > piece_start
    ]
