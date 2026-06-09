"""Lattice segmentation: all the dictionary's readings, and the best path through them.

Why greedy is not enough
------------------------
Maximum matching commits to a word the moment it sees one, so ``研究生命起源``
becomes ``研究生 / 命 / 起源`` forward and ``研究 / 生命 / 起源`` backward, and
the heuristic that chooses between them knows nothing about how likely either
reading is. The dictionary does. If ``研究`` occurs 10 times, ``生命`` 8,
``起源`` 5, ``研究生`` 6 and ``命`` 4, then under the unigram assumption that a
word's probability is its share of the dictionary's total, the two readings
have probability ``10 * 8 * 5 / 33^3`` and ``6 * 4 * 5 / 33^3``, and the first
is ``10 / 3`` times as likely.

Jieba's ``__cut_DAG`` makes that comparison over *every* reading rather than
two. For each position in the run it lists every dictionary word starting
there, which forms a directed acyclic graph -- a lattice -- of candidate
edges; each edge scores ``log(frequency / total)``; and the best path from the
start of the run to its end is found by dynamic programming from the right,
where ``best(position) = max over edges e from position of score(e) +
best(e.end)`` and ``best(end of run) = 0``. That is exactly the Viterbi
recurrence with words as states, and it is the same shape as the hidden Markov
segmenter's, one level up.

Worked, on the two frequency tables pinned in the spec
------------------------------------------------------
With the table above, total 33::

    研究 | 生命 | 起源    log(10/33) + log(8/33) + log(5/33)  = -4.4981
    研究生 | 命 | 起源    log(6/33)  + log(4/33) + log(5/33)  = -5.7020

The first wins by ``1.2040``, which is ``log(400 / 120) = log(10 / 3)`` exactly.
Swap the frequencies so that ``研究生`` is 20 and ``命`` 12, total 55, and the
same lattice answers the other way: ``-6.0305`` against ``-4.9319``. The
dictionary decides, not the segmenter.

The single-character fallback, and what it costs
------------------------------------------------
Every position also gets an edge for the single character standing there,
whether or not the dictionary holds it, so that the lattice always has a path
from start to end. A character the dictionary lacks is scored as if it had a
frequency of one, ``log(1 / total)`` -- one count's worth, the smallest
frequency a real entry can have -- which is Jieba's ``log(FREQ.get(word) or
1)``. That choice means a run of unknown characters is *possible* but
expensive: at total 33 each costs ``-3.4965``, so six of them cost ``-20.979``
and any dictionary reading covering them wins. It is a convention rather than
an estimate, and it is stated here so that it can be argued with.

Unknown runs, and the model that takes them
-------------------------------------------
After the best path is chosen, any stretch of consecutive single characters
that are *not* dictionary words is a stretch the dictionary knew nothing
about: a name, a coinage, a transliteration. If the segmenter was built with
an ``unknown_segmenter`` -- a fitted
:class:`~oop_ml.core.natural_language_processing.tokenization.segmentation.hidden_markov.HiddenMarkovSegmenter`
-- that stretch is handed to it and its words replace the characters, with
offsets shifted back into place. A single character on its own is left alone,
since a tagger can only answer ``S`` for it. A single character that *is* a
dictionary word ends a stretch and stays as it is, because the dictionary
vouched for it. Without an unknown segmenter the characters stay single, which
is what the lattice alone can say.

An unfitted unknown segmenter is refused at the first :meth:`split`, not at
construction and not at the first text that happens to contain an unknown
run: a segmenter configured to hand off and unable to is misconfigured on
every text, and the failure should say so on the first one.

Ties, stated once
-----------------
Two candidates at one position with equal path score keep the *longer* word,
and after that the earlier candidate. :func:`candidate_beats` is the one copy
of that rule and both routes below consult it. A :class:`Lattice` refuses two
candidates over the same span, so the second clause cannot be reached through
one; it is kept because the rule reads as complete with it.

Two routes, and the agreement between them
------------------------------------------
:meth:`DictionaryLatticeSegmenter.split` runs the recurrence directly on the
run with scores and lengths in two lists. :meth:`DictionaryLatticeSegmenter.lattice_of`
builds every candidate as a :class:`LatticeEdge` inside a :class:`Lattice`,
and :meth:`Lattice.best_path` runs the same recurrence over those objects,
answering a :class:`LatticePath` whose total is the sum of its edges. The spec
holds the two to the same words on every fixture, which is what makes the
observed route evidence about the efficient one rather than a second opinion.

Cost is ``O(n * L)`` in the run length and the longest dictionary word, since
each position tries at most ``L`` substrings. Jieba bounds the inner loop with
a prefix dictionary so that a position stops as soon as no longer word can
start there; that is the usual repair and is not done here.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import InvalidValuesError, NotFittedError
from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (  # noqa: E501
    WordDictionary,
    checked_word,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.hidden_markov import (  # noqa: E501
    HiddenMarkovSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.runs import (
    RunSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import checked_text
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    NON_WHITESPACE_RUN,
)


def candidate_beats(
    score: float, length: int, incumbent_score: float, incumbent_length: int
) -> bool:
    """Whether a candidate replaces the incumbent: higher score, then longer word.

    A full tie keeps the incumbent, which is the earlier candidate.
    """
    if score != incumbent_score:
        return score > incumbent_score
    return length > incumbent_length


class LatticeEdge:
    """One candidate word at one position of the text, with its log score.

    Parameters
    ----------
    word:
        The candidate. Non-empty, no whitespace.
    start, end:
        Its span in the text, so that ``end - start == len(word)``.
    log_score:
        Its log probability under the unigram model; finite.

    Raises
    ------
    InvalidValuesError
        If the span does not fit the word, or the score is not finite.
    EmptyValuesError
        If ``word`` is empty.
    """

    __slots__ = ("_end", "_log_score", "_start", "_word")

    def __init__(self, word: str, start: int, end: int, log_score: float) -> None:
        self._word = checked_word(word, "lattice word")

        if start < 0 or end - start != len(self._word):
            raise InvalidValuesError(
                f"the span [{start}, {end}) does not fit the word {word!r} of "
                f"length {len(self._word)}"
            )
        if not math.isfinite(log_score):
            raise InvalidValuesError(
                f"a lattice edge's log score must be finite, got {log_score}"
            )

        self._start = int(start)
        self._end = int(end)
        self._log_score = float(log_score)

    @property
    def word(self) -> str:
        """The candidate word."""
        return self._word

    @property
    def start(self) -> int:
        """Offset of its first character."""
        return self._start

    @property
    def end(self) -> int:
        """Offset one past its last character."""
        return self._end

    @property
    def length(self) -> int:
        """How many characters it covers."""
        return self._end - self._start

    @property
    def log_score(self) -> float:
        """Its log probability under the unigram model."""
        return self._log_score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LatticeEdge):
            return NotImplemented
        return (
            self._word == other._word
            and self._start == other._start
            and self._end == other._end
            and self._log_score == other._log_score
        )

    def __hash__(self) -> int:
        return hash((self._word, self._start, self._end, self._log_score))

    def __repr__(self) -> str:
        return (
            f"LatticeEdge({self._word!r}, {self._start}, {self._end}, "
            f"log_score={self._log_score:.4f})"
        )


class LatticePath:
    """The edges one walk through a lattice chose, in order, and their total.

    Parameters
    ----------
    edges:
        The chosen edges in source order, never overlapping. Adjacent within a
        run; across whitespace, which no edge covers, not adjacent. May be
        empty, for a text of nothing but whitespace.

    Raises
    ------
    InvalidValuesError
        If an edge starts before the previous one ended.
    """

    __slots__ = ("_edges", "_total_log_score")

    def __init__(self, edges: Sequence[LatticeEdge]) -> None:
        previous_end = 0
        for edge in edges:
            if edge.start < previous_end:
                raise InvalidValuesError(
                    f"a path's edges must be in source order and must not overlap; "
                    f"{edge!r} starts before offset {previous_end}"
                )
            previous_end = edge.end

        self._edges = tuple(edges)
        self._total_log_score = float(sum(edge.log_score for edge in edges))

    @property
    def words(self) -> tuple[str, ...]:
        """The chosen words as strings, in order."""
        return tuple(edge.word for edge in self._edges)

    @property
    def total_log_score(self) -> float:
        """The sum of the chosen edges' log scores."""
        return self._total_log_score

    @property
    def n_edges(self) -> int:
        """How many edges the path holds."""
        return len(self._edges)

    def __iter__(self) -> Iterator[LatticeEdge]:
        """Iterate the chosen edges in order."""
        return iter(self._edges)

    def __len__(self) -> int:
        return len(self._edges)

    def __getitem__(self, position: int) -> LatticeEdge:
        return self._edges[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LatticePath):
            return NotImplemented
        return self._edges == other._edges

    def __hash__(self) -> int:
        return hash(self._edges)

    def __repr__(self) -> str:
        return f"LatticePath({list(self.words)!r}, total={self._total_log_score:.4f})"


class Lattice:
    """Every candidate word at every position of a text.

    Parameters
    ----------
    text:
        The text the candidates were drawn from.
    edges:
        Every candidate, each matching ``text`` over its span. Every position
        holding a non-whitespace character must have at least one candidate
        starting there, so that a path always exists; no two candidates may
        cover the same span.

    Raises
    ------
    InvalidValuesError
        If an edge does not match the text over its span, two edges share a
        span, or a non-whitespace position has no candidate.
    """

    __slots__ = ("_edges_by_start", "_text")

    def __init__(self, text: str, edges: Sequence[LatticeEdge]) -> None:
        self._text = checked_text(text)

        edges_by_start: dict[int, list[LatticeEdge]] = {}
        spans: set[tuple[int, int]] = set()
        for edge in edges:
            if edge.end > len(text) or text[edge.start : edge.end] != edge.word:
                raise InvalidValuesError(
                    f"{edge!r} does not match the text over its span, which reads "
                    f"{text[edge.start : edge.end]!r}"
                )
            if (edge.start, edge.end) in spans:
                raise InvalidValuesError(
                    f"two candidates cover the span [{edge.start}, {edge.end})"
                )
            spans.add((edge.start, edge.end))
            edges_by_start.setdefault(edge.start, []).append(edge)

        for position, character in enumerate(text):
            if not character.isspace() and position not in edges_by_start:
                raise InvalidValuesError(
                    f"no candidate starts at position {position} ({character!r}), "
                    f"so no path can cover it"
                )

        self._edges_by_start = {
            start: tuple(edges_by_start[start]) for start in sorted(edges_by_start)
        }

    @property
    def text(self) -> str:
        """The text the lattice is over."""
        return self._text

    @property
    def n_positions(self) -> int:
        """How many characters the text has, whitespace included."""
        return len(self._text)

    @property
    def n_edges(self) -> int:
        """How many candidates there are in all."""
        return sum(len(edges) for edges in self._edges_by_start.values())

    @property
    def positions(self) -> tuple[int, ...]:
        """Every position at which some candidate starts, ascending."""
        return tuple(self._edges_by_start)

    def edges_from(self, position: int) -> tuple[LatticeEdge, ...]:
        """The candidates starting at ``position``, in the order they were listed."""
        return self._edges_by_start.get(position, ())

    def best_path(self) -> LatticePath:
        """The path from the start of the text to its end with the highest total.

        Dynamic programming from the right; a position with no candidate is
        whitespace and is stepped over at no cost. Ties follow
        :func:`candidate_beats`.
        """
        n_positions = len(self._text)
        best_scores = [0.0] * (n_positions + 1)
        choices: list[LatticeEdge | None] = [None] * (n_positions + 1)

        for position in range(n_positions - 1, -1, -1):
            incumbent: LatticeEdge | None = None
            incumbent_score = -math.inf
            for edge in self.edges_from(position):
                score = edge.log_score + best_scores[edge.end]
                if incumbent is None or candidate_beats(
                    score, edge.length, incumbent_score, incumbent.length
                ):
                    incumbent = edge
                    incumbent_score = score
            if incumbent is None:
                best_scores[position] = best_scores[position + 1]
            else:
                best_scores[position] = incumbent_score
                choices[position] = incumbent

        chosen: list[LatticeEdge] = []
        position = 0
        while position < n_positions:
            edge = choices[position]
            if edge is None:
                position += 1
            else:
                chosen.append(edge)
                position = edge.end
        return LatticePath(chosen)

    def __iter__(self) -> Iterator[LatticeEdge]:
        """Iterate every candidate, by start position and then listing order."""
        for edges in self._edges_by_start.values():
            yield from edges

    def __len__(self) -> int:
        return self.n_edges

    def __repr__(self) -> str:
        return f"Lattice(n_positions={self.n_positions}, n_edges={self.n_edges})"


class DictionaryLatticeSegmenter(RunSegmenter):
    """Most-probable-path segmentation over every dictionary reading.

    Parameters
    ----------
    dictionary:
        The words that may be read, with the frequencies that score them.
    unknown_segmenter:
        A fitted hidden Markov segmenter to hand runs of unknown characters
        to, or ``None`` to leave them as single characters. Checked for
        fitted state at the first :meth:`split`.
    """

    dictionary: WordDictionary
    unknown_segmenter: HiddenMarkovSegmenter | None = None

    def lattice_of(self, text: str) -> Lattice:
        """Every candidate at every position, the observed route into :meth:`split`.

        Raises
        ------
        InvalidValuesError
            If ``text`` is not a string.
        """
        text = checked_text(text)
        edges: list[LatticeEdge] = []
        for match in NON_WHITESPACE_RUN.finditer(text):
            edges.extend(self._edges_of_run(match.group(), match.start()))
        return Lattice(text, edges)

    def best_path(self, text: str) -> LatticePath:
        """The chosen edges and their total; its words are what :meth:`split` answers.

        Before any unknown run is handed to the unknown segmenter, so with one
        configured the two can differ inside such a run and nowhere else.

        Raises
        ------
        InvalidValuesError
            If ``text`` is not a string.
        """
        return self.lattice_of(text).best_path()

    def _words_of_run(self, run: str) -> tuple[str, ...]:
        if self.unknown_segmenter is not None and not self.unknown_segmenter.is_fitted:
            raise NotFittedError(
                "the unknown_segmenter must be fit before this segmenter can split; "
                "fit it on segmented sentences first"
            )

        pieces = self._best_pieces(run)
        if self.unknown_segmenter is None:
            return pieces
        return self._with_unknown_runs_resegmented(pieces, self.unknown_segmenter)

    def _best_pieces(self, run: str) -> tuple[str, ...]:
        """The recurrence on the bare run: two lists, no objects."""
        n_positions = len(run)
        total = self.dictionary.total_frequency
        longest = self.dictionary.longest_word_length
        best_scores = [0.0] * (n_positions + 1)
        best_lengths = [0] * (n_positions + 1)

        for position in range(n_positions - 1, -1, -1):
            incumbent_score = -math.inf
            incumbent_length = 0
            for length in range(1, min(longest, n_positions - position) + 1):
                frequency = self.dictionary.frequency_of(
                    run[position : position + length]
                )
                if frequency == 0 and length > 1:
                    continue
                score = (
                    math.log((frequency or 1) / total) + best_scores[position + length]
                )
                if candidate_beats(score, length, incumbent_score, incumbent_length):
                    incumbent_score = score
                    incumbent_length = length
            best_scores[position] = incumbent_score
            best_lengths[position] = incumbent_length

        pieces: list[str] = []
        position = 0
        while position < n_positions:
            length = best_lengths[position]
            pieces.append(run[position : position + length])
            position += length
        return tuple(pieces)

    def _edges_of_run(self, run: str, offset: int) -> list[LatticeEdge]:
        """Every candidate of one run, placed at ``offset`` in the whole text."""
        total = self.dictionary.total_frequency
        longest = self.dictionary.longest_word_length
        edges: list[LatticeEdge] = []
        for position in range(len(run)):
            start = offset + position
            single = run[position]
            if single not in self.dictionary:
                edges.append(LatticeEdge(single, start, start + 1, math.log(1 / total)))
            for length in range(1, min(longest, len(run) - position) + 1):
                candidate = run[position : position + length]
                frequency = self.dictionary.frequency_of(candidate)
                if frequency:
                    edges.append(
                        LatticeEdge(
                            candidate,
                            start,
                            start + length,
                            math.log(frequency / total),
                        )
                    )
        return edges

    def _with_unknown_runs_resegmented(
        self, pieces: Sequence[str], unknown_segmenter: HiddenMarkovSegmenter
    ) -> tuple[str, ...]:
        """Hand every stretch of unknown single characters to the tagger."""
        result: list[str] = []
        unknown: list[str] = []
        for piece in pieces:
            if len(piece) == 1 and piece not in self.dictionary:
                unknown.append(piece)
                continue
            result.extend(self._resegmented(unknown, unknown_segmenter))
            unknown = []
            result.append(piece)
        result.extend(self._resegmented(unknown, unknown_segmenter))
        return tuple(result)

    @staticmethod
    def _resegmented(
        characters: Sequence[str], unknown_segmenter: HiddenMarkovSegmenter
    ) -> Sequence[str]:
        if len(characters) < 2:
            return characters
        return unknown_segmenter.split("".join(characters)).texts
