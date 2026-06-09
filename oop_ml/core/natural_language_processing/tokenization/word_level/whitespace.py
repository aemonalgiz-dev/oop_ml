"""Splitting on whitespace, which is the baseline every other rule is measured against.

What it gets right and what it cannot
-------------------------------------
For a language written with spaces, a run of non-space characters is very
nearly a word, and this splitter says exactly that: every maximal run of
characters that are not whitespace is one word, and nothing else is looked at.
It is the shortest correct answer to "where are the words", it is the
pre-tokenizer the subword trainers here default to, and it is what a caller
gets when they have not yet decided on anything cleverer.

What it cannot do is separate punctuation. ``"Hello,"`` is one word to it, and
``"(world)"`` is another, so a corpus of ordinary prose produces a vocabulary in
which ``world``, ``world,`` and ``world.`` are three unrelated words. Every other
rule in this family exists to repair that, and each pays for the repair with a
list of exceptions -- which is why this one, which has none, stays as the
baseline.

Whitespace means Unicode whitespace
-----------------------------------
The rule is Python's own ``str.isspace``, applied through the regular
expression's ``\\s`` class in Unicode mode, so a no-break space, an ideographic
space and a paragraph separator all end a word. A rule limited to the ASCII
space would treat ``"a\\u00a0b"`` as one word, which no reader would.
"""

from __future__ import annotations

import re

from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words

NON_WHITESPACE_RUN = re.compile(r"\S+")


class WhitespacePreTokenizer(PreTokenizer):
    """Every maximal run of non-whitespace characters is a word.

    Takes no parameters, so every instance behaves alike; it is a class rather
    than a function so that it can sit wherever a
    :class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.PreTokenizer`
    is asked for.
    """

    def _words_of(self, text: str) -> Words:
        return Words(
            [
                Word.of(text, match.start(), match.end())
                for match in NON_WHITESPACE_RUN.finditer(text)
            ]
        )
