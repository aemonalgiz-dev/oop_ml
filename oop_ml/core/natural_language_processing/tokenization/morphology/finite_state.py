"""A finite-state morphological analyser: the lexicon is the machine.

What a transducer is
--------------------
A finite-state transducer is an automaton that reads one tape and writes
another. Fed the surface string ``walked`` it writes the analysis
``walk+V +PAST``, and because it is a relation rather than a function it can be
run backwards -- fed ``walk+V +PAST`` it writes ``walked`` -- and it can write
several outputs for one input when the language is genuinely ambiguous. The
analysers behind HFST and Xerox's tools, and Buckwalter's analyser for Arabic,
are all this one object: a network whose states are "which kind of morpheme may
come next" and whose arcs each consume a surface morph and emit its analysis.

Why the lexicon *is* the analyser
---------------------------------
Write the morphemes down as lexicons, each entry a surface form, its analysis
and the name of the lexicon that may follow it -- the ``lexc`` format -- and
the machine is already described. Each lexicon is a state, each entry an arc,
the root lexicon the start state and the ``END`` continuation the accepting
one. HFST's ``hfst-lexc`` is a *compiler* for exactly this: it turns the entries
into a minimised transducer so that analysis runs in time linear in the word.
Buckwalter's analyser is one such set of lexicons for Arabic, three tables of
prefixes, stems and suffixes with compatibility tables saying which may join.
This module skips the compilation and walks the lexicons directly, which
answers the same question by depth-first search: from the root, at position 0,
try every entry whose surface begins there, and follow its continuation from
the position after it; a path that reaches ``END`` exactly at the end of the
word is an analysis. Results are memoised on ``(position, lexicon)``, so the
suffix table for a given state at a given offset is computed once however many
prefixes lead to it. The output itself can still be large: a lexicon whose
entries are ``a`` and ``aa``, each continuing to itself or to the end, gives
``aaaa`` five analyses, the number of ways to write 4 as ordered ones and twos,
and that count grows as the Fibonacci numbers. Enumerating them is the cost;
the search that finds them is shared.

How a bare form reaches the end
-------------------------------
Surfaces are non-empty, so a bare ``walk`` cannot be a root entry followed by
an empty suffix. Instead the root lexicon lists the stem twice: once continuing
to the suffix lexicon, once continuing straight to ``END``. That is the lexc
idiom, and it keeps the invariant that every arc consumes at least one
character, which is what makes the search terminate without a cycle check.

The tie rule, stated once
-------------------------
:attr:`Analyses.shortest` is the analysis with the fewest morphemes; among
those, the one whose tag string sorts first; among those, the one whose morph
sequence sorts first. So ``walks`` as ``walk+N +PL`` beats ``walk+V +3SG``
because ``N`` sorts before ``V``, which is arbitrary and written down. The
shortest analysis is what :meth:`FiniteStateAnalyzer.segment` cuts a word by,
and what the pre-tokenizer route uses, so it has to be deterministic.

What rewrite rules are, and where they are not
----------------------------------------------
``bake`` plus ``ed`` is ``bakeed``, and English writes ``baked``. A two-level
rule compiler (Koskenniemi's, or Xerox's ``xfst`` and HFST's ``hfst-twolc``)
handles that with a rule -- delete a stem-final ``e`` before a vowel-initial
suffix -- composed onto the lexicon transducer, and the result is a machine in
which the ``bak`` arc exists although nobody typed it. Here no rule compiler
exists, so the alternation is written as it would come *out* of one: an
allomorph entry ``bak`` whose analysis is still ``bake+V`` and whose
continuation is a lexicon of vowel-initial suffixes. That is honest about what
is and is not built: the analysis is right, the morphs are the surface pieces
actually present in the word, and a language with many alternations needs
many entries where a rule would need one. The spec pins both halves -- the
naive lexicon fails ``baked`` and accepts ``bakeed``; the repaired one does the
reverse.

Serving as a morph splitter
---------------------------
A :class:`FiniteStateAnalyzer` is a
:class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.PreTokenizer`:
its :meth:`split` cuts each whitespace word by its shortest analysis, with
exact offsets, and leaves a word it cannot analyse whole. That is precisely
what
:class:`~oop_ml.core.natural_language_processing.tokenization.morphology.constrained_merges.MorphemeConstrainedBytePairEncoding`
asks of its ``morph_splitter``, so the hand-written morphology can be the
boundary the learned merges respect.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Self

from pydantic import PrivateAttr, model_validator

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    PreTokenizer,
    checked_text,
)
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.words import Word, Words

END = "#"
"""The continuation that accepts: the word must end exactly here. lexc's ``#``."""

_WHITESPACE = WhitespacePreTokenizer()


class LexiconEntry:
    """One arc of the analyser: a surface morph, its analysis, and what may follow.

    Parameters
    ----------
    surface:
        The characters this morph contributes to a word. At least one, so
        that every arc consumes something and the search terminates.
    analysis:
        What the morph means, in whatever tag notation the lexicon uses --
        ``walk+V``, ``+PAST``. May be empty for a morph that adds nothing to
        the tag string.
    continuation:
        The name of the lexicon that may follow, or :data:`END`.

    Raises
    ------
    EmptyValuesError
        If ``surface`` or ``continuation`` is empty.
    InvalidValuesError
        If ``analysis`` is not a string.
    """

    __slots__ = ("_analysis", "_continuation", "_surface")

    def __init__(self, surface: str, analysis: str, continuation: str) -> None:
        if not isinstance(surface, str) or not surface:
            raise EmptyValuesError(
                "a lexicon entry's surface must hold at least one character; a "
                "bare form is written as a second entry continuing to END"
            )

        if not isinstance(analysis, str):
            raise InvalidValuesError(
                f"an analysis must be a str, got {type(analysis).__name__}"
            )

        if not isinstance(continuation, str) or not continuation:
            raise EmptyValuesError(
                f"the entry {surface!r} needs a continuation: a lexicon name or END"
            )

        self._surface = surface
        self._analysis = analysis
        self._continuation = continuation

    @property
    def surface(self) -> str:
        """The characters the morph contributes."""
        return self._surface

    @property
    def analysis(self) -> str:
        """The morph's tag or gloss."""
        return self._analysis

    @property
    def continuation(self) -> str:
        """The lexicon that may follow, or :data:`END`."""
        return self._continuation

    @property
    def leads_to_end(self) -> bool:
        """Whether a word may finish after this morph."""
        return self._continuation == END

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LexiconEntry):
            return NotImplemented
        return (
            self._surface == other._surface
            and self._analysis == other._analysis
            and self._continuation == other._continuation
        )

    def __hash__(self) -> int:
        return hash((self._surface, self._analysis, self._continuation))

    def __repr__(self) -> str:
        return (
            f"LexiconEntry({self._surface!r}, {self._analysis!r}, "
            f"{self._continuation!r})"
        )


class Lexicon:
    """One state of the analyser: a named set of arcs that may be taken from it.

    Parameters
    ----------
    name:
        What continuations refer to it by. Non-empty and not :data:`END`.
    entries:
        The arcs, in the order they are tried. Non-empty; the same entry may
        not be listed twice, since it would produce the same analysis twice.

    Raises
    ------
    EmptyValuesError
        If the name is empty or there are no entries.
    InvalidValuesError
        If the name is :data:`END`, or an entry is repeated.
    """

    __slots__ = ("_entries", "_name")

    def __init__(self, name: str, entries: Sequence[LexiconEntry]) -> None:
        if not isinstance(name, str) or not name:
            raise EmptyValuesError("a lexicon needs a name for continuations to use")

        if name == END:
            raise InvalidValuesError(
                f"{END!r} is the continuation that ends a word and cannot name a "
                f"lexicon"
            )

        if len(entries) == 0:
            raise EmptyValuesError(f"the lexicon {name!r} has no entries")

        seen: set[LexiconEntry] = set()
        for entry in entries:
            if entry in seen:
                raise InvalidValuesError(
                    f"the lexicon {name!r} lists {entry!r} twice, which would "
                    f"analyse every word through it twice"
                )
            seen.add(entry)

        self._name = name
        self._entries = tuple(entries)

    @property
    def name(self) -> str:
        """What continuations call this lexicon."""
        return self._name

    @property
    def n_entries(self) -> int:
        """How many arcs leave this state."""
        return len(self._entries)

    @property
    def continuations(self) -> frozenset[str]:
        """Every lexicon name, or :data:`END`, that an entry here leads to."""
        return frozenset(entry.continuation for entry in self._entries)

    def __iter__(self) -> Iterator[LexiconEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lexicon):
            return NotImplemented
        return self._name == other._name and self._entries == other._entries

    def __hash__(self) -> int:
        return hash((self._name, self._entries))

    def __repr__(self) -> str:
        return f"Lexicon({self._name!r}, n_entries={self.n_entries})"


class Analysis:
    """One path through the lexicons: the entries taken, root to end, in order.

    Parameters
    ----------
    entries:
        The arcs the path took. At least one, since a word is non-empty.

    Raises
    ------
    EmptyValuesError
        If there are no entries.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: Sequence[LexiconEntry]) -> None:
        if len(entries) == 0:
            raise EmptyValuesError("an analysis takes at least one entry")

        self._entries = tuple(entries)

    @property
    def morphs(self) -> tuple[str, ...]:
        """The surface pieces, in order; they concatenate to the word."""
        return tuple(entry.surface for entry in self._entries)

    @property
    def surface(self) -> str:
        """The word this analysis is of: its morphs joined."""
        return "".join(self.morphs)

    @property
    def tags(self) -> str:
        """The analyses joined by single spaces, empty ones skipped: ``walk+V +PAST``.

        A tag string rather than a tuple because that is how every analyser
        in this tradition reports, and because :attr:`Analyses.shortest`
        sorts on it.
        """
        return " ".join(entry.analysis for entry in self._entries if entry.analysis)

    @property
    def n_morphemes(self) -> int:
        """How many arcs the path took."""
        return len(self._entries)

    def __iter__(self) -> Iterator[LexiconEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Analysis):
            return NotImplemented
        return self._entries == other._entries

    def __hash__(self) -> int:
        return hash(self._entries)

    def __repr__(self) -> str:
        return f"Analysis({self.tags!r}, morphs={list(self.morphs)!r})"


class Analyses:
    """Every analysis of one word, in the order the tie rule ranks them.

    Parameters
    ----------
    word:
        The word analysed. Non-empty.
    analyses:
        Every path found, in any order; each must spell exactly ``word``.
        May be empty, which is how an unanalysable word is reported.

    Raises
    ------
    EmptyValuesError
        If ``word`` is empty.
    InvalidValuesError
        If an analysis does not spell the word.
    """

    __slots__ = ("_analyses", "_word")

    def __init__(self, word: str, analyses: Sequence[Analysis]) -> None:
        if not isinstance(word, str) or not word:
            raise EmptyValuesError("analyses are of a word of at least one character")

        for analysis in analyses:
            if analysis.surface != word:
                raise InvalidValuesError(
                    f"{analysis!r} spells {analysis.surface!r}, not the word {word!r}"
                )

        self._word = word
        self._analyses = tuple(
            sorted(
                analyses,
                key=lambda analysis: (
                    analysis.n_morphemes,
                    analysis.tags,
                    analysis.morphs,
                ),
            )
        )

    @property
    def word(self) -> str:
        """The word analysed."""
        return self._word

    @property
    def n_analyses(self) -> int:
        """How many paths reached the end of the word."""
        return len(self._analyses)

    @property
    def is_analysable(self) -> bool:
        """Whether at least one path did."""
        return len(self._analyses) > 0

    @property
    def shortest(self) -> Analysis:
        """Fewest morphemes; ties by tag string, then by morph sequence.

        Raises
        ------
        EmptyValuesError
            If the word has no analysis. Check :attr:`is_analysable` first.
        """
        if not self._analyses:
            raise EmptyValuesError(
                f"{self._word!r} has no analysis, so no shortest one; check "
                f"is_analysable first"
            )
        return self._analyses[0]

    def __iter__(self) -> Iterator[Analysis]:
        return iter(self._analyses)

    def __len__(self) -> int:
        return len(self._analyses)

    def __getitem__(self, position: int) -> Analysis:
        return self._analyses[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Analyses):
            return NotImplemented
        return self._word == other._word and self._analyses == other._analyses

    def __hash__(self) -> int:
        return hash((self._word, self._analyses))

    def __repr__(self) -> str:
        return f"Analyses({self._word!r}, n_analyses={self.n_analyses})"


class FiniteStateAnalyzer(PreTokenizer):
    """Lexicons walked as a transducer, refusing an inconsistent set at construction.

    Parameters
    ----------
    lexicons:
        Every state of the machine. Non-empty, no name twice, and every
        continuation must name one of them or be :data:`END`.
    root:
        The name of the lexicon a word starts in.

    Raises
    ------
    pydantic.ValidationError
        If there are no lexicons, two lexicons share a name, ``root`` names
        no lexicon, or a continuation names neither a lexicon nor
        :data:`END`. A ``ValueError`` raised inside a validator is
        what pydantic reports, which is the library's rule for every
        refusal made at construction.
    """

    lexicons: tuple[Lexicon, ...]
    root: str

    _lexicons_by_name: dict[str, Lexicon] = PrivateAttr()

    @model_validator(mode="after")
    def _check_the_machine_is_closed(self) -> Self:
        """Check the machine is closed before any word is read."""
        if len(self.lexicons) == 0:
            raise ValueError("an analyser needs at least one lexicon")

        lexicons_by_name: dict[str, Lexicon] = {}
        for lexicon in self.lexicons:
            if lexicon.name in lexicons_by_name:
                raise ValueError(
                    f"two lexicons are named {lexicon.name!r}; a continuation "
                    f"could not say which it meant"
                )
            lexicons_by_name[lexicon.name] = lexicon

        if self.root not in lexicons_by_name:
            raise ValueError(
                f"root={self.root!r} names no lexicon; the lexicons are "
                f"{sorted(lexicons_by_name)}"
            )

        for lexicon in self.lexicons:
            for continuation in sorted(lexicon.continuations):
                if continuation != END and continuation not in lexicons_by_name:
                    raise ValueError(
                        f"the lexicon {lexicon.name!r} continues to "
                        f"{continuation!r}, which is neither a lexicon nor END"
                    )

        self._lexicons_by_name = lexicons_by_name
        return self

    @property
    def n_lexicons(self) -> int:
        """How many states the machine has."""
        return len(self.lexicons)

    def analyse(self, word: str) -> Analyses:
        """Every path from the root whose surfaces spell exactly ``word``.

        Raises
        ------
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If ``word`` is empty.
        """
        word = checked_text(word)
        if not word:
            raise EmptyValuesError("cannot analyse an empty word")

        memo: dict[tuple[int, str], tuple[tuple[LexiconEntry, ...], ...]] = {}
        paths = self._paths_from(word, 0, self.root, memo)
        return Analyses(word, [Analysis(path) for path in paths])

    def _paths_from(
        self,
        word: str,
        position: int,
        lexicon_name: str,
        memo: dict[tuple[int, str], tuple[tuple[LexiconEntry, ...], ...]],
    ) -> tuple[tuple[LexiconEntry, ...], ...]:
        """Every entry sequence that consumes ``word[position:]`` from this state."""
        if lexicon_name == END:
            return ((),) if position == len(word) else ()

        if position >= len(word):
            return ()

        key = (position, lexicon_name)
        if key in memo:
            return memo[key]

        paths: list[tuple[LexiconEntry, ...]] = []
        for entry in self._lexicons_by_name[lexicon_name]:
            if not word.startswith(entry.surface, position):
                continue
            for tail in self._paths_from(
                word, position + len(entry.surface), entry.continuation, memo
            ):
                paths.append((entry, *tail))

        memo[key] = tuple(paths)
        return memo[key]

    def segment(self, word: str) -> Words:
        """One word cut by its shortest analysis, with offsets into the word.

        The word whole, as a single
        :class:`~oop_ml.core.natural_language_processing.tokenization.words.Word`,
        if it has no analysis -- the same answer :meth:`split` gives, so that
        this can serve as a morph splitter without a special case.

        Raises
        ------
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If ``word`` is empty.
        """
        return self._segment_at(checked_text(word), 0)

    def _segment_at(self, word: str, offset: int) -> Words:
        """``segment``, with every span shifted by where the word sits in a text."""
        analyses = self.analyse(word)
        if not analyses.is_analysable:
            return Words([Word(word, offset, offset + len(word))])

        words: list[Word] = []
        position = offset
        for morph in analyses.shortest.morphs:
            words.append(Word(morph, position, position + len(morph)))
            position += len(morph)
        return Words(words)

    def _words_of(self, text: str) -> Words:
        words: list[Word] = []
        for word in _WHITESPACE.split(text):
            words.extend(self._segment_at(word.text, word.start))
        return Words(words)
