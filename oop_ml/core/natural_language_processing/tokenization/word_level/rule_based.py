"""Rules and an exception table: the shape NLTK's and spaCy's word tokenizers share.

The mechanism
-------------
Split on whitespace first. Then, for each chunk: if the chunk is in the
exception table, emit the pieces the table lists for it and stop. Otherwise
peel characters off the front while a prefix pattern matches them, and off the
back while a suffix pattern matches, consulting the table again after every
peel, because ``(don't)`` is not an exception but the ``don't`` inside its
brackets is. Whatever remains is cut wherever an infix pattern matches, and the
peeled suffixes are put back in source order after it. spaCy's ``Tokenizer``
is exactly this loop; NLTK's Treebank tokenizer is the same loop written as a
fixed cascade of substitutions with the exceptions folded into the patterns.

What the table buys over a pure regular expression
--------------------------------------------------
A regular expression that splits ``don't`` into ``do`` and ``n't`` is a rule
about every word ending in ``n't``, and the day it meets ``can't`` it produces
``ca`` and ``n't`` -- which is what the Treebank convention actually asks for,
and a rule that instead produced ``can`` and ``'t`` would need rewriting to
know about one word. Here that word is one line: ``"can't": ("ca", "n't")``,
checked before any rule runs, and no rule changes. Abbreviations are the same
story from the other side: ``Mr.`` ends in a lowercase letter and a period,
which is precisely the shape of ``said.``, so no pattern over characters can
keep the one and split the other. ``"Mr.": ("Mr.",)`` keeps it, and ``Gov.``,
which nobody listed, still comes apart as ``Gov`` and ``.`` -- pinned, because
it is the honest half of the mechanism.

What it costs
-------------
The table is the language. Every contraction, every abbreviation, every
title, in every capitalisation and with every kind of apostrophe, is a row
someone had to write, and the rows are never complete: spaCy's English table
runs past a thousand entries and still misses words. Every new language needs
a new table and new prefix, suffix and infix lists, which is the maintenance
the subword tokenizers exist to escape, since their pieces are learned from a
corpus rather than listed.

Why the table describes a split and not a rewrite
-------------------------------------------------
An entry's pieces must concatenate to its key. ``"don't": ("do", "n't")`` is
accepted; ``"don't": ("do", "not")`` is refused with
:class:`~oop_ml.core.exceptions.InvalidValuesError`. The reason is the offsets:
every piece here is a slice of the source, so a caller can always recover the
span each piece stands for, and a rewrite would have broken that promise for
whichever words happened to be listed. Normalising is a different job, and
:mod:`~oop_ml.core.natural_language_processing.tokenization.word_level.moses`
does it openly, with spans that say so.

The English configuration
-------------------------
:meth:`RuleAndExceptionPreTokenizer.english` assembles
:data:`ENGLISH_PREFIXES` (opening brackets, opening quotes, currency signs,
leading punctuation), :data:`ENGLISH_SUFFIXES` (closing brackets and quotes,
trailing punctuation, the clitics ``'s n't 'll 're 've 'm 'd`` in both
apostrophes, and a final period only after a lowercase letter, a digit or a
closer, or after two capitals -- so ``said.`` and ``NASA.`` split and the
initial ``A.`` and the initialism ``U.S.`` do not), :data:`ENGLISH_INFIXES`
(an ellipsis, a hyphen or dash between letters, a comma between letters) and
:data:`ENGLISH_EXCEPTIONS`. The letter classes are ASCII plus the Latin-1
accented block, an approximation of spaCy's exhaustive Unicode tables that
covers the Western European languages and nothing further.

Worked
------
``"Don't stop," she said (quietly).`` gives twelve pieces: ``"``, ``Do``,
``n't``, ``stop``, ``,``, ``"``, ``she``, ``said``, ``(``, ``quietly``, ``)``,
``.``. ``U.S. law`` gives two and ``well-known`` three. The English exception
table holds 90 entries: 17 contractions written once, grown to 52 by the
capitalised and curly-apostrophe spellings, and 38 abbreviations that stand
for themselves.
Every piece reproduces its source slice, asserted over every fixture in
``test/core/natural_language_processing/tokenization/word_level/test_rule_based.py``.

Cost and tie rules
------------------
Each peel is one regular-expression match over the remainder, and a chunk of
``L`` characters can peel up to ``L`` times, so a chunk costs O(L^2) in the
worst case (a chunk that is nothing but brackets); the usual repair is a
single anchored match that consumes every prefix at once. Where two suffix
fragments could both match, the one whose match starts furthest left wins,
which is how ``'s`` beats ``'`` on ``it's`` without a length rule; where two
fragments match at the same position, the one listed first wins, which is
Python's alternation order and is stated here so that it is a rule rather
than an accident. A fragment that can match the empty string is refused at
construction, because it would peel nothing forever.
"""

from __future__ import annotations

import re
from typing import Self

from pydantic import Field, PrivateAttr, model_validator

from oop_ml.core.natural_language_processing.tokenization.tokenizer import PreTokenizer
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words

NON_WHITESPACE_RUN = re.compile(r"\S+")

# ASCII plus the Latin-1 accented block, which is what spaCy's exhaustive
# tables come to for the Western European languages.
LOWERCASE_LETTERS = r"a-zà-öø-ÿ"
UPPERCASE_LETTERS = r"A-ZÀ-ÖØ-Þ"
LETTERS = LOWERCASE_LETTERS + UPPERCASE_LETTERS

# A final period is a suffix after any of these: a digit, a lowercase letter,
# a percent sign, a closing bracket or a closing quote...
CLOSERS_BEFORE_A_FINAL_PERIOD = r"0-9" + LOWERCASE_LETTERS + r"%\)\]\}\"'’”»"
FINAL_PERIOD_AFTER_A_CLOSER = rf"(?<=[{CLOSERS_BEFORE_A_FINAL_PERIOD}])\."
# ...or after two capitals, so that NASA. splits and U.S. and the initial A. do
# not, since in both of those the period follows one capital preceded by a
# period or by nothing.
FINAL_PERIOD_AFTER_TWO_CAPITALS = rf"(?<=[{UPPERCASE_LETTERS}][{UPPERCASE_LETTERS}])\."

ENGLISH_PREFIXES: tuple[str, ...] = (
    r"\(",
    r"\[",
    r"\{",
    r"<",
    r'"',
    r"'",
    r"`",
    r"‘",
    r"“",
    r"«",
    r"\$",
    r"£",
    r"€",
    r"#",
    r"\.\.+",
    r"…",
    r",",
    r";",
    r":",
    r"!",
    r"\?",
    r"¡",
    r"¿",
)

ENGLISH_SUFFIXES: tuple[str, ...] = (
    r"\)",
    r"\]",
    r"\}",
    r">",
    r'"',
    r"'",
    r"’",
    r"”",
    r"»",
    r",",
    r";",
    r":",
    r"!",
    r"\?",
    r"\.\.+",
    r"…",
    r"'s",
    r"'S",
    r"’s",
    r"’S",
    r"n't",
    r"N'T",
    r"n’t",
    r"N’T",
    r"'ll",
    r"'re",
    r"'ve",
    r"'m",
    r"'d",
    r"’ll",
    r"’re",
    r"’ve",
    r"’m",
    r"’d",
    FINAL_PERIOD_AFTER_A_CLOSER,
    FINAL_PERIOD_AFTER_TWO_CAPITALS,
)

ENGLISH_INFIXES: tuple[str, ...] = (
    r"\.\.+",
    r"…",
    rf"(?<=[{LETTERS}])[-–—](?=[{LETTERS}])",
    rf"(?<=[{LETTERS}]),(?=[{LETTERS}])",
)

# The contractions whose split the suffix rules alone would not produce, or
# would produce only by accident, each in its lowercase spelling; the
# capitalised and curly-apostrophe variants are generated below.
_ENGLISH_CONTRACTIONS: dict[str, tuple[str, ...]] = {
    "don't": ("do", "n't"),
    "can't": ("ca", "n't"),
    "won't": ("wo", "n't"),
    "shan't": ("sha", "n't"),
    "ain't": ("ai", "n't"),
    "cannot": ("can", "not"),
    "gonna": ("gon", "na"),
    "wanna": ("wan", "na"),
    "gotta": ("got", "ta"),
    "let's": ("let", "'s"),
    "y'all": ("y'", "all"),
    "I'm": ("I", "'m"),
    "I'll": ("I", "'ll"),
    "I've": ("I", "'ve"),
    "I'd": ("I", "'d"),
    "ma'am": ("ma'am",),
    "o'clock": ("o'clock",),
}

# Abbreviations that keep their period. Those ending in a lowercase letter
# (Mr., e.g., etc.) are kept only by being listed; the initialisms (U.S.) are
# also kept by the two-capitals guard on the final-period suffix.
_ENGLISH_ABBREVIATIONS: tuple[str, ...] = (
    "U.S.",
    "U.S.A.",
    "U.K.",
    "U.N.",
    "E.U.",
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "cf.",
    "a.m.",
    "p.m.",
    "Ph.D.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Jr.",
    "Sr.",
    "St.",
    "Mt.",
    "Inc.",
    "Ltd.",
    "Co.",
    "Corp.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Jun.",
    "Jul.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
)


def _capitalised(word: str) -> str:
    return word[0].upper() + word[1:]


def _with_spelling_variants(
    contractions: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Each contraction, plus its capitalised form and both with a curly apostrophe."""
    with_capitals: dict[str, tuple[str, ...]] = {}
    for key, pieces in contractions.items():
        with_capitals[key] = pieces
        if key[0].islower():
            with_capitals[_capitalised(key)] = (_capitalised(pieces[0]), *pieces[1:])

    with_curly_apostrophes = dict(with_capitals)
    for key, pieces in with_capitals.items():
        if "'" in key:
            with_curly_apostrophes[key.replace("'", "’")] = tuple(
                piece.replace("'", "’") for piece in pieces
            )
    return with_curly_apostrophes


ENGLISH_EXCEPTIONS: dict[str, tuple[str, ...]] = {
    **_with_spelling_variants(_ENGLISH_CONTRACTIONS),
    **{abbreviation: (abbreviation,) for abbreviation in _ENGLISH_ABBREVIATIONS},
}


def _compiled_alternation(
    fragments: tuple[str, ...], template: str, role: str
) -> re.Pattern[str] | None:
    """One pattern trying every fragment in order, or ``None`` when there are none.

    Raises
    ------
    InvalidValuesError
        If a fragment is empty, is not a regular expression, or can match the
        empty string.
    """
    if not fragments:
        return None

    for fragment in fragments:
        if not fragment:
            raise ValueError(f"a {role} fragment must not be empty")
        try:
            compiled = re.compile(fragment)
        except re.error as error:
            raise ValueError(
                f"the {role} fragment {fragment!r} is not a regular expression: {error}"
            ) from error
        if compiled.match("") is not None:
            raise ValueError(
                f"the {role} fragment {fragment!r} can match the empty string, so "
                f"it would peel nothing forever"
            )

    return re.compile(
        template.format("|".join(f"(?:{fragment})" for fragment in fragments))
    )


class RuleAndExceptionPreTokenizer(PreTokenizer):
    """Whitespace, then an exception table, then prefix, suffix and infix rules.

    Parameters
    ----------
    prefixes:
        Regular-expression fragments, tried in order at the front of a chunk.
        Empty means nothing is peeled from the front.
    suffixes:
        Fragments tried at the back of a chunk. Empty means nothing is peeled
        from the back.
    infixes:
        Fragments matched inside whatever remains after peeling; each match
        becomes a piece of its own. Empty means the remainder is one piece.
    exceptions:
        Chunks that are split as listed, and never by the rules. Each value's
        pieces must concatenate to its key, and a one-piece value keeps the
        word whole.

    Raises
    ------
    pydantic.ValidationError
        If a fragment is empty, malformed or able to match the empty string;
        if an exception key is empty or contains whitespace; if an entry has
        no pieces or an empty piece; if an entry's pieces do not concatenate
        to its key; or if a field has the wrong type or a keyword is unknown.
        A ``ValueError`` raised inside a validator is what pydantic reports,
        which is the library's rule for every refusal made at construction.
    """

    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    infixes: tuple[str, ...] = ()
    exceptions: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    _prefix_pattern: re.Pattern[str] | None = PrivateAttr(default=None)
    _suffix_pattern: re.Pattern[str] | None = PrivateAttr(default=None)
    _infix_pattern: re.Pattern[str] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _compile_the_rules_and_check_the_table(self) -> Self:
        self._prefix_pattern = _compiled_alternation(self.prefixes, "^(?:{})", "prefix")
        self._suffix_pattern = _compiled_alternation(self.suffixes, "(?:{})$", "suffix")
        self._infix_pattern = _compiled_alternation(self.infixes, "(?:{})", "infix")

        for key, pieces in self.exceptions.items():
            if not key or any(character.isspace() for character in key):
                raise ValueError(
                    f"an exception key must be one whitespace-free word, got {key!r}"
                )
            if not pieces or any(not piece for piece in pieces):
                raise ValueError(
                    f"the exception for {key!r} must list at least one non-empty "
                    f"piece, got {pieces!r}"
                )
            if "".join(pieces) != key:
                raise ValueError(
                    f"the exception for {key!r} lists pieces {pieces!r}, which join "
                    f"to {''.join(pieces)!r}; an exception says where a word is "
                    f"split, not what it is rewritten to"
                )
        return self

    @classmethod
    def english(cls) -> Self:
        """The English configuration: the four ``ENGLISH_*`` module constants."""
        return cls(
            prefixes=ENGLISH_PREFIXES,
            suffixes=ENGLISH_SUFFIXES,
            infixes=ENGLISH_INFIXES,
            exceptions=ENGLISH_EXCEPTIONS,
        )

    def _words_of(self, text: str) -> Words:
        words: list[Word] = []
        for chunk in NON_WHITESPACE_RUN.finditer(text):
            words.extend(self._pieces_of_chunk(text, chunk.start(), chunk.end()))
        return Words(words)

    def _pieces_of_chunk(
        self, text: str, chunk_start: int, chunk_end: int
    ) -> list[Word]:
        peeled_prefixes: list[Word] = []
        peeled_suffixes: list[Word] = []  # in peeling order, so last-in-source first
        start, end = chunk_start, chunk_end

        while start < end:
            remainder = text[start:end]
            if remainder in self.exceptions:
                return [
                    *peeled_prefixes,
                    *self._exception_pieces(remainder, start),
                    *reversed(peeled_suffixes),
                ]

            prefix_length = self._match_length(self._prefix_pattern, remainder)
            if prefix_length:
                peeled_prefixes.append(Word.of(text, start, start + prefix_length))
                start += prefix_length
                continue

            suffix_length = self._match_length(self._suffix_pattern, remainder)
            if suffix_length:
                peeled_suffixes.append(Word.of(text, end - suffix_length, end))
                end -= suffix_length
                continue

            break

        return [
            *peeled_prefixes,
            *self._split_at_infixes(text, start, end),
            *reversed(peeled_suffixes),
        ]

    @staticmethod
    def _match_length(pattern: re.Pattern[str] | None, remainder: str) -> int:
        """How many characters the anchored pattern takes, zero when none."""
        if pattern is None:
            return 0
        match = pattern.search(remainder)
        return 0 if match is None else match.end() - match.start()

    def _exception_pieces(self, key: str, start: int) -> list[Word]:
        pieces: list[Word] = []
        position = start
        for piece in self.exceptions[key]:
            pieces.append(Word(piece, position, position + len(piece)))
            position += len(piece)
        return pieces

    def _split_at_infixes(self, text: str, start: int, end: int) -> list[Word]:
        if start >= end:
            return []
        if self._infix_pattern is None:
            return [Word.of(text, start, end)]

        pieces: list[Word] = []
        piece_start = start
        for match in self._infix_pattern.finditer(text[start:end]):
            infix_start, infix_end = start + match.start(), start + match.end()
            if infix_end == infix_start:
                continue
            if infix_start > piece_start:
                pieces.append(Word.of(text, piece_start, infix_start))
            pieces.append(Word.of(text, infix_start, infix_end))
            piece_start = infix_end
        if piece_start < end:
            pieces.append(Word.of(text, piece_start, end))
        return pieces
