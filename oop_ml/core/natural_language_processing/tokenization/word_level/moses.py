"""Moses' tokenizer: the rule set of the phrase-based machine translation era.

What it was for
---------------
A phrase-based translation system (Koehn, Och and Marcu, 2003; the Moses
toolkit of Koehn et al., 2007) learns a table of phrase pairs from word-aligned
parallel text, and every distinct surface string in that text is a separate
row of the table. ``Washington``, ``Washington.`` and ``Washington,`` would be
three rows, with the evidence for one word split three ways, so the first job
in the pipeline was to make the surface vocabulary as small and as consistent
as a handful of rules could -- and to do it *reversibly*, because the system's
output had to be turned back into readable text by a detokenizer that saw only
the tokens. ``tokenizer.perl`` is that handful of rules, and
:class:`MosesPreTokenizer` reproduces the ones that mattered, as the
``sacremoses`` port reproduces them.

The rules, and what each bought
-------------------------------
- **Every other character stands alone.** A character that is not a letter, a
  digit, a period, an apostrophe or a hyphen becomes a word by itself, so
  punctuation stops varying the spelling of the word it touches. The Perl pads
  each such character individually, so ``?!`` is two words and not one run,
  and this class does the same. One exception, also the Perl's: a comma with a
  digit on both sides stays, so ``5,300`` remains a number.
- **A final period is split off, unless it belongs to an abbreviation.** The
  sentence-final period was a token of its own in the phrase table, and an
  abbreviation's period is not one. A word ending in exactly one period keeps
  it when the part before the period holds another period and a letter
  (``U.S.``, ``e.g.``), or is in the nonbreaking-prefix list (``Mr``, ``Dr``,
  ``etc``, the months, every capital letter), or is a *numeric-only* prefix
  and a digit begins the next word (``No. 5``, ``Art. 3`` -- but ``No.`` before
  ``Nothing`` is ``No`` and a period), or when the next word begins with a
  lowercase letter, because a sentence does not. A run of two or more periods
  is one word, so ``wait...`` is ``wait`` and ``...``. The list is
  :data:`NONBREAKING_PREFIXES`, 53 entries (27 titles and abbreviations plus
  the 26 capitals) with :data:`NUMERIC_ONLY_PREFIXES` beside it; the Perl loads
  one list per language and only the English one is built in here, so it is
  consulted whatever ``language`` says.
- **English clitics split to the right; French and Italian elisions split to
  the left.** ``don't`` is ``don`` and ``'t``, ``it's`` is ``it`` and ``'s``,
  ``students'`` is ``students`` and ``'``; ``l'homme`` is ``l'`` and ``homme``.
  For a translation system this was the whole point of touching apostrophes at
  all: ``'s`` aligns to a possessive marker in the target language and ``n't``
  to a negation, and neither can while glued to the word beside it. The French
  rule is the mirror because the elided article is the clitic there. An
  apostrophe with a letter on one side only stands alone, as does one between
  two digits; a digit followed by ``'s`` splits like a letter (``1990's``); a
  digit followed by any other letter is reached by none of the Perl's rules
  and stays inside the word.
- **Aggressive hyphen splitting**, off by default, turns ``well-known`` into
  ``well``, ``@-@``, ``known``. The placeholder is how the detokenizer told a
  hyphen that joined a compound from a dash between two words, and splitting
  let the compound share its statistics with ``well`` and with ``known``.
- **Escaping**, off by default, rewrites ``& | < > ' " [ ]`` to
  ``&amp; &#124; &lt; &gt; &apos; &quot; &#91; &#93;``. Moses' factored format
  separates the factors of a word with ``|`` and the fields of a phrase table
  with ``|||``; the decoder reads ``<`` and ``>`` as XML markup for forced
  translations; the hierarchical models write non-terminals as ``[X]``; and
  ``&``, ``'`` and ``"`` complete the XML set. A raw ``|`` in a training
  sentence would have been read as a factor boundary and corrupted its row.
  Here the *text* of the word changes and its span does not, so ``&amp;``
  stands for one source character, which is what
  :class:`~oop_ml.core.natural_language_processing.tokenization.words.Word` allows
  ``len(text) != end - start`` for.

Why a subword tokenizer needs none of it
----------------------------------------
Byte pair encoding and the unigram model learn their pieces from a corpus
that has been split only on whitespace and broad character classes, and the
merges do the rest: ``Mr.`` becomes one piece because it is frequent, a rare
abbreviation falls back to letters, and nothing is ever unknown, so there is
no surface vocabulary to keep small and no list of prefixes to maintain per
language. Detokenization is exact by construction, because the pieces
concatenate and a marker records where the spaces were, so a ``@-@``
placeholder has nothing to restore. And a model's input is not a
pipe-separated text format, so there is nothing for escaping to protect. What
the rules cost is what every rule-based splitter costs: the lists are the
language, they are never complete, and each new language needs its own.
GPT-2's pre-tokenizer keeps one idea from here, the rightward English clitic
split, and keeps it as a pattern with no exception list.

Worked
------
``Mr. Smith went to Washington.`` gives six words -- ``Mr.``, ``Smith``,
``went``, ``to``, ``Washington``, ``.`` -- and ``No. 5 wins.`` gives four,
``No.``, ``5``, ``wins``, ``.``. Under escaping ``don't`` is ``don`` and
``&apos;t``, the second a seven-character text standing for the two source
characters at ``[3, 5)``. All of it is pinned in
``test/core/natural_language_processing/tokenization/word_level/test_moses.py``.

Cost and tie rules
------------------
Linear in the text: the scanner visits each character a bounded number of
times, and the period rule is one more pass over the words. The prefix list is
case-sensitive, as the Perl's is, so ``mr.`` before a capital splits.
"""

from __future__ import annotations

import re
from enum import Enum, auto

from pydantic import field_validator

from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words

NON_WHITESPACE_RUN = re.compile(r"\S+")

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en", "fr", "it"})

# Moses' nonbreaking_prefix.en, reduced to the entries that carry no
# annotation: a word here keeps a following period whatever comes next.
NONBREAKING_PREFIXES: frozenset[str] = frozenset(
    {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Sr",
        "Jr",
        "St",
        "Mt",
        "vs",
        "etc",
        "Inc",
        "Ltd",
        "Co",
        "Corp",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Sept",
        "Oct",
        "Nov",
        "Dec",
        *"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    }
)

# The entries the Perl marks #NUMERIC_ONLY#: the period stays only when the
# next word begins with a digit, so that "No. 5" holds and "No. Nothing" splits.
NUMERIC_ONLY_PREFIXES: frozenset[str] = frozenset({"No", "Art"})

HYPHEN_PLACEHOLDER = "@-@"

ESCAPES: dict[str, str] = {
    "&": "&amp;",
    "|": "&#124;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&apos;",
    '"': "&quot;",
    "[": "&#91;",
    "]": "&#93;",
}
ESCAPE_TABLE = str.maketrans(ESCAPES)
UNESCAPES: dict[str, str] = {escaped: plain for plain, escaped in ESCAPES.items()}


class _ApostrophePlacement(Enum):
    """Where an apostrophe goes, decided from the characters on either side."""

    STARTS_NEXT_WORD = auto()
    ENDS_PREVIOUS_WORD = auto()
    STANDS_ALONE = auto()
    STAYS_INSIDE = auto()


class _UnitCollector:
    """Accumulates one chunk's units as the scanner walks it.

    Holds the start of the run of ordinary characters being read, so the
    scanner only ever says "close the run here" or "this character stands
    alone", and the bookkeeping lives in one place.
    """

    __slots__ = ("_run_start", "_text", "_units")

    def __init__(self, text: str, chunk_start: int) -> None:
        self._text = text
        self._run_start = chunk_start
        self._units: list[Word] = []

    def close_run(self, end: int) -> None:
        """End the current run at ``end``, emitting it if it holds anything."""
        if end > self._run_start:
            self._units.append(Word.of(self._text, self._run_start, end))
        self._run_start = end

    def emit_span(self, start: int, end: int) -> None:
        """Close the run before ``start`` and emit ``[start, end)`` as one unit."""
        self.close_run(start)
        self._units.append(Word.of(self._text, start, end))
        self._run_start = end

    def stand_alone(self, position: int, spelled_as: str | None = None) -> None:
        """Close the run before ``position`` and emit that character alone.

        ``spelled_as`` rewrites the unit's text while its span stays the one
        source character; the hyphen placeholder is the one caller.
        """
        self.close_run(position)
        text = self._text[position] if spelled_as is None else spelled_as
        self._units.append(Word(text, position, position + 1))
        self._run_start = position + 1

    @property
    def units(self) -> list[Word]:
        """Everything emitted so far, in source order."""
        return self._units


class MosesPreTokenizer(PreTokenizer):
    """Koehn et al.'s ``tokenizer.perl`` rules, as a scanner over whitespace chunks.

    Parameters
    ----------
    language:
        ``"en"`` splits a clitic to the right (``don`` ``'t``); ``"fr"`` and
        ``"it"`` split an elision to the left (``l'`` ``homme``). Nothing else
        depends on it; the nonbreaking-prefix list is English throughout.
    aggressive_hyphen_splitting:
        When true, a hyphen between two alphanumerics becomes its own word
        spelled ``@-@``, the Perl's placeholder for a detokenizer to restore.
    escape_special_characters:
        When true, ``& | < > ' " [ ]`` are rewritten to their XML-style
        entities in each word's text. The spans are untouched.

    Raises
    ------
    pydantic.ValidationError
        If ``language`` is not one of ``en``, ``fr``, ``it``.
    """

    language: str = "en"
    aggressive_hyphen_splitting: bool = False
    escape_special_characters: bool = False

    @field_validator("language")
    @classmethod
    def _check_language_is_supported(cls, language: str) -> str:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language must be one of {sorted(SUPPORTED_LANGUAGES)}, got "
                f"{language!r}; the Perl has apostrophe rules for these three "
                f"and pads every apostrophe for the rest"
            )
        return language

    def _words_of(self, text: str) -> Words:
        units: list[Word] = []
        for chunk in NON_WHITESPACE_RUN.finditer(text):
            units.extend(self._units_of(text, chunk.start(), chunk.end()))

        words: list[Word] = []
        for position, unit in enumerate(units):
            following_first_character = (
                text[units[position + 1].start] if position + 1 < len(units) else ""
            )
            words.extend(
                self._with_final_period_settled(unit, following_first_character)
            )

        if self.escape_special_characters:
            words = [
                Word(word.text.translate(ESCAPE_TABLE), word.start, word.end)
                for word in words
            ]
        return Words(words)

    def _units_of(self, text: str, chunk_start: int, chunk_end: int) -> list[Word]:
        """One whitespace-free chunk, cut by every rule except the final period."""
        collector = _UnitCollector(text, chunk_start)
        position = chunk_start
        while position < chunk_end:
            character = text[position]
            previous = text[position - 1] if position > chunk_start else ""
            following = text[position + 1] if position + 1 < chunk_end else ""

            if character == "." and following == ".":
                run_end = position
                while run_end < chunk_end and text[run_end] == ".":
                    run_end += 1
                collector.emit_span(position, run_end)
                position = run_end
                continue

            stays_inside_the_word = (
                character.isalnum()
                or character in ".-"
                or (character == "," and previous.isnumeric() and following.isnumeric())
            )
            if character == "'":
                self._place_apostrophe(collector, position, previous, following)
            elif (
                character == "-"
                and self.aggressive_hyphen_splitting
                and previous.isalnum()
                and following.isalnum()
            ):
                collector.stand_alone(position, spelled_as=HYPHEN_PLACEHOLDER)
            elif not stays_inside_the_word:
                collector.stand_alone(position)
            position += 1

        collector.close_run(chunk_end)
        return collector.units

    def _place_apostrophe(
        self, collector: _UnitCollector, position: int, previous: str, following: str
    ) -> None:
        placement = self._apostrophe_placement(previous, following)
        if placement is _ApostrophePlacement.STARTS_NEXT_WORD:
            collector.close_run(position)
        elif placement is _ApostrophePlacement.ENDS_PREVIOUS_WORD:
            collector.close_run(position + 1)
        elif placement is _ApostrophePlacement.STANDS_ALONE:
            collector.stand_alone(position)

    def _apostrophe_placement(
        self, previous: str, following: str
    ) -> _ApostrophePlacement:
        """The Perl's four (English: five) substitutions, read as one decision.

        ``previous`` and ``following`` are the neighbouring characters inside
        the chunk, or ``""`` at either edge, which the Perl saw as a space.
        """
        between_letters = previous.isalpha() and following.isalpha()
        if self.language == "en":
            if between_letters or (previous.isnumeric() and following == "s"):
                return _ApostrophePlacement.STARTS_NEXT_WORD
            if previous.isnumeric() and following.isalpha():
                return _ApostrophePlacement.STAYS_INSIDE
            return _ApostrophePlacement.STANDS_ALONE

        if between_letters:
            return _ApostrophePlacement.ENDS_PREVIOUS_WORD
        return _ApostrophePlacement.STANDS_ALONE

    def _with_final_period_settled(
        self, unit: Word, following_first_character: str
    ) -> list[Word]:
        """Split a single final period off, unless a rule says it belongs."""
        spelling = unit.text
        if len(spelling) < 2 or not spelling.endswith(".") or spelling[-2] == ".":
            return [unit]

        before_period = spelling[:-1]
        if self._period_stays_attached(before_period, following_first_character):
            return [unit]
        return [
            Word(before_period, unit.start, unit.end - 1),
            Word(".", unit.end - 1, unit.end),
        ]

    @staticmethod
    def _period_stays_attached(
        before_period: str, following_first_character: str
    ) -> bool:
        if "." in before_period and any(
            character.isalpha() for character in before_period
        ):
            return True
        if before_period in NONBREAKING_PREFIXES:
            return True
        if following_first_character.islower():
            return True
        return (
            before_period in NUMERIC_ONLY_PREFIXES
            and following_first_character.isnumeric()
        )
