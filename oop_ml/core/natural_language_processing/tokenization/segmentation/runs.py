"""The rule every segmenter here shares: whitespace ends a run, and offsets are true.

What a segmenter actually decides
---------------------------------
A segmenter for a script written without spaces decides where the words are
inside a run of characters that has no spaces in it. When the text it is handed
*does* contain whitespace -- a Chinese sentence quoting an English name, a
Japanese page with line breaks, two sentences separated by a space -- that
whitespace is already a boundary, and no dictionary lookup or tagging model
should be allowed to reach across it. A maximum-matching segmenter that could
match ``thecat`` across ``the cat`` would be producing a word the text does not
contain.

So every segmenter in this family answers a narrower question,
:meth:`RunSegmenter._words_of_run`: given one run of non-whitespace characters,
which pieces is it cut into. :meth:`RunSegmenter._words_of` is the template
that finds the runs, asks that question of each, and rebuilds the
:class:`~oop_ml.core.natural_language_processing.tokenization.words.Word` objects
with their offsets in the *original* text, so that a word cut from the second
run of ``"研究 生命"`` reports a start of 3, not 0. Written once, because a loop
that shifts offsets is exactly the kind of thing four copies would get wrong
in four different ways.

Whitespace means Unicode whitespace, by the same pattern
:class:`~oop_ml.core.natural_language_processing.tokenization.word_level.whitespace.WhitespacePreTokenizer`
uses, so an ideographic space ends a run as an ASCII one does.

A subclass cuts and never rewrites
----------------------------------
The pieces a subclass answers with are concatenated back into the run they
came from, in order, and the template slices each word out of the source text
at the offsets that concatenation implies. A segmenter therefore cannot
normalise a character on the way through; that is a fact about this family,
whose whole output is a set of boundaries, and it is what lets
:meth:`~oop_ml.core.natural_language_processing.tokenization.words.Word.of` be used
rather than the general constructor.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    NON_WHITESPACE_RUN,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words


class RunSegmenter(PreTokenizer):
    """A pre-tokenizer that segments each whitespace-free run on its own.

    :meth:`_words_of` is the template; a subclass supplies
    :meth:`_words_of_run`, the pieces of one run in order, and nothing about
    whitespace or offsets.
    """

    def _words_of(self, text: str) -> Words:
        words: list[Word] = []
        for match in NON_WHITESPACE_RUN.finditer(text):
            position = match.start()
            for piece in self._words_of_run(match.group()):
                words.append(Word.of(text, position, position + len(piece)))
                position += len(piece)
        return Words(words)

    @abstractmethod
    def _words_of_run(self, run: str) -> Sequence[str]:
        """The pieces one run of non-whitespace characters is cut into, in order.

        Their concatenation must be ``run``; the template relies on it to
        place each piece.
        """
