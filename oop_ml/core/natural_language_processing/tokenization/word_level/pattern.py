"""Splitting on a regular expression, the notation every other rule can be written in.

Why a pattern is a pre-tokenizer
--------------------------------
Every rule-based splitter answers "where are the words" by naming the runs of
characters that count as one. A regular expression is a notation for exactly
that, and most published pre-tokenizers are one written down: GPT-2's is a
single pattern, and ``\\w+|[^\\w\\s]+`` is the "words and punctuation" rule that
a dozen libraries ship under a dozen names. So one class takes the pattern as
its parameter, and the pattern is the whole rule. Every non-empty match is a
word, the gaps between matches are not, and the offsets are the match spans, so
nothing is ever rewritten and ``source[start:end]`` is always the word itself.

Empty matches are skipped rather than refused. A pattern such as ``\\w*`` matches
the empty string at every position between words, and a
:class:`~oop_ml.core.natural_language_processing.tokenization.words.Word` cannot hold
nothing, so the empty matches are simply not words. That is what the caller
meant, and refusing the pattern at construction would require deciding whether
a pattern *can* match empty, which is undecidable to read off the pattern in
general and cheap to observe match by match.

Why WhitespacePreTokenizer is not a PatternPreTokenizer
-------------------------------------------------------
``\\S+`` expresses whitespace splitting, so the whitespace rule could have been
``PatternPreTokenizer(pattern=r"\\S+")``, or a subclass of this one that fixes
the field. It is neither, because a rule with no parameters should not carry a
parameter it does not use. A subclass would inherit a ``pattern`` field that a
caller could set to something other than ``\\S+`` while still holding an object
whose type says ``WhitespacePreTokenizer``, which is a name that has stopped
meaning anything. The whitespace rule is the family's baseline precisely because
it has nothing to configure, and its type says so by having no fields.

Where the pattern is compiled
-----------------------------
The field validator compiles the pattern once to prove it is a regular
expression, so an invalid one is refused at construction as a pydantic
``ValidationError`` with the rest of the library's hyperparameters, and
``model_post_init`` compiles it again into a private attribute for the splits
to use. The second compile is a hit in ``re``'s own cache, so nothing is paid
for keeping the validator a pure check.

The GPT-2 pattern, translated
-----------------------------
Radford et al. (2019) pre-tokenize with::

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

Three things about it. A word carries its leading space, so ``" world"`` and
``"world"`` are different words and a byte-level vocabulary learns both, which
is how the model knows where the spaces were without an end-of-word marker.
The seven clitics are listed first, so ``"world's"`` is ``" world"`` and then
``"'s"``. And ``\\s+(?!\\S)`` gives a run of whitespace up to but not including
its last character, which is left for the following word to claim, so
``"  two"`` is ``" "`` and then ``" two"``. Worked on ``"Hello world's fun"``::

    Hello   [0, 5)      world  [5, 11)     's  [11, 13)      fun  [13, 17)

four words whose spans tile the source exactly, since every character of the
text is inside some match.

Python's ``re`` has no ``\\p{L}``, so :data:`GPT2_PATTERN` writes a letter as
``[^\\W\\d_]``, a word character that is neither a digit nor the underscore, a
number as ``\\d``, and the leftovers as ``[^\\s\\w]|_``, since the underscore is
a word character to ``\\w`` and punctuation to the original. The two classes
are not the original ones. ``\\d`` is General Category ``Nd`` only, where
``\\p{N}`` also covers ``Nl`` and ``No``; and ``\\w`` is whatever ``str.isalnum``
accepts, which admits ``Nl`` and ``No`` as *letters*. So a Roman numeral ``Ⅻ``
or a vulgar fraction ``½`` is a number to the original and a letter to this
translation: ``"1½"`` is one word ``"1½"`` under the original and two words,
``"1"`` and ``"½"``, here. On ASCII and on alphabetic scripts the two agree
exactly, and the difference is pinned rather than hidden.
"""

from __future__ import annotations

import re

from pydantic import Field, PrivateAttr, field_validator

from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words

WORD_OR_PUNCTUATION_PATTERN = r"\w+|[^\w\s]+"
"""A run of word characters, or a run of anything that is neither word nor space.

``"Hello, world!"`` becomes ``Hello`` ``,`` ``world`` ``!``; ``"3.14"`` becomes
``3`` ``.`` ``14``, because the rule knows nothing about numbers.
"""

GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
)
"""GPT-2's pre-tokenization pattern in ``re``'s dialect. See the module docstring.

A letter is ``[^\\W\\d_]`` and a number is ``\\d``, so characters of General
Category ``Nl`` and ``No`` (Roman numerals, vulgar fractions, superscripts) are
letters here where the original's ``\\p{N}`` makes them numbers.
"""


class PatternPreTokenizer(PreTokenizer):
    """Every non-empty match of a regular expression is a word.

    Parameters
    ----------
    pattern:
        The regular expression, in Python's ``re`` dialect. At least one
        character, and must compile. Two are exported beside this class:
        :data:`WORD_OR_PUNCTUATION_PATTERN` and :data:`GPT2_PATTERN`.

    Raises
    ------
    pydantic.ValidationError
        At construction, if ``pattern`` is empty or does not compile.
    """

    pattern: str = Field(min_length=1)

    _compiled_pattern: re.Pattern[str] = PrivateAttr()

    @field_validator("pattern")
    @classmethod
    def _check_pattern_compiles(cls, pattern: str) -> str:
        """Refuse a pattern ``re`` cannot compile, in pydantic's words."""
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(
                f"pattern {pattern!r} is not a valid regular expression: {error}"
            ) from error
        return pattern

    def model_post_init(self, context: object) -> None:
        """Compile once for every split to share; a cache hit after the validator."""
        self._compiled_pattern = re.compile(self.pattern)

    def _words_of(self, text: str) -> Words:
        return Words(
            [
                Word.of(text, match.start(), match.end())
                for match in self._compiled_pattern.finditer(text)
                if match.end() > match.start()
            ]
        )
