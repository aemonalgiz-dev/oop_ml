"""Segmentation as tagging: a hidden Markov model of each character's place in its word.

The idea
--------
A dictionary segmenter can only produce words it has been given. Names, new
coinages and transliterations are exactly the words no dictionary has, and
they are exactly the words a reader most needs cut correctly. Xue (2003)
turned the problem around: instead of asking "which word is this", ask of each
*character* "where in its word does it sit". Every character is one of four
things -- the beginning of a multi-character word (``B``), somewhere in its
middle (``M``), its end (``E``), or a word on its own (``S``) -- and a sentence
is segmented by tagging its characters and cutting after every ``E`` and
``S``. ``研究生命起源`` read as ``研究 / 生命 / 起源`` is ``B E B E B E``;
read as ``研究生 / 命 / 起源`` it is ``B M E S B E``.

The tag sequence is a hidden Markov chain and the characters are what it
emits. Three tables describe it, all counted from a segmented corpus: how
often a sentence *starts* with each tag, how often each tag *follows* each
other tag, and how often each tag emits each character. Viterbi then finds the
tag sequence that makes the observed characters most probable, in log space
so that products of small numbers do not underflow.

The structural constraints are not learned
------------------------------------------
Of the sixteen possible transitions, half cannot occur in a tagging of any
segmentation. After ``B`` or ``M`` a word is still open, so only ``M`` or ``E``
may follow; after ``E`` or ``S`` a word has just closed, so only ``B`` or ``S``
may. A sentence starts with ``B`` or ``S`` and ends with ``E`` or ``S``. Those
eight transitions and two initial tags are the only ones counted, the only
ones smoothed, and the only ones Viterbi is allowed to take; the forbidden
ones have log probability ``-inf`` and are never smoothed to anything else.
Smoothing them would let the decoder emit ``B B``, which is a tag sequence that
cannot be cut into words at all -- a wrong answer of the kind no test on the
words would explain. The last tag is enforced by taking the better of the two
admissible end states rather than the best of four.

Smoothing, worked on a two-sentence corpus
------------------------------------------
Additive smoothing with ``smoothing = 1`` adds one imaginary count to every
admissible outcome. On the corpus ``ab / c`` and ``abc``, whose tags are
``B E S`` and ``B M E``::

    initial      B (2 + 1) / (2 + 2) = 3/4      S (0 + 1) / 4 = 1/4
    after B      M (1 + 1) / (2 + 2) = 1/2      E 1/2
    after M      M (0 + 1) / (1 + 2) = 1/3      E 2/3
    after E      B (0 + 1) / (1 + 2) = 1/3      S 2/3
    after S      B 1/2                          S 1/2      (never seen)

Emissions are smoothed over the corpus alphabet *plus one slot* standing for
any character never seen, so a character met for the first time at
segmentation time has a probability rather than a crash. The alphabet here is
``a b c``, four slots in all. Under ``B``, which emitted ``a`` twice::

    P(a | B) = (2 + 1) / (2 + 4) = 1/2      P(b | B) = P(c | B) = P(unseen | B) = 1/6

and under ``S``, which emitted only ``c``, once, ``P(c | S) = 2/5`` and
everything else ``1/5``. A character the corpus contains but this tag never
emitted gets the same floor as one the corpus never contained; the alphabet
slot exists so the floor is defined, not to distinguish the two.

Smoothing decides a contradiction, and the threshold was measured
------------------------------------------------------------------
A corpus in which ``研究`` is a word three times and ``研究生`` a word once
contradicts itself about ``研究生``, and the model has to side with one
reading. At the default smoothing of 1, and at 0.5, it answers ``研究 / 生``
for the sentence that contained ``研究生`` -- it does not reproduce its own
training sentence, and that is the smoothing doing its job rather than a
failure to fit. At 0.1 and below the counts are trusted enough that the
minority sentence comes back as written. A recovery test on a generative
model therefore needs a corpus that agrees with itself, and the spec's does.

Ties, stated once
-----------------
Two paths into one tag with equal score keep the earlier predecessor in the
order ``B M E S``; two admissible end states with equal score keep ``E``. Both
are arbitrary, and both matter only on hand-built fixtures, where they matter
completely.

What it cannot do, which is the point of the pointwise segmenter
----------------------------------------------------------------
Every count above needs *whole* sentences. A transition count needs two
adjacent tags, an initial count needs to know which character is first, and a
tag is known only where every boundary around it is. A corpus annotated at
some gaps and not others gives this model nothing it can count. That is the
gap :mod:`~oop_ml.core.natural_language_processing.tokenization.segmentation.pointwise`
exists to fill.

Cost is ``O(n)`` in the run length with a constant of at most eight
transitions per character, so there is nothing here to optimise.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.natural_language_processing.tokenization.segmentation.dictionary import (  # noqa: E501
    SegmentedCorpus,
)
from oop_ml.core.natural_language_processing.tokenization.segmentation.runs import (
    RunSegmenter,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import checked_text
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    NON_WHITESPACE_RUN,
)


class BoundaryTag(StrEnum):
    """Where a character sits in its word.

    Attributes
    ----------
    BEGIN:
        First character of a word of two or more.
    MIDDLE:
        Neither first nor last of a word of three or more.
    END:
        Last character of a word of two or more.
    SINGLE:
        A word of one character.
    """

    BEGIN = "B"
    MIDDLE = "M"
    END = "E"
    SINGLE = "S"


INITIAL_TAGS: tuple[BoundaryTag, ...] = (BoundaryTag.BEGIN, BoundaryTag.SINGLE)
"""The tags a sentence may start with: a word is either opening or complete."""

FINAL_TAGS: tuple[BoundaryTag, ...] = (BoundaryTag.END, BoundaryTag.SINGLE)
"""The tags a sentence may end with: the last word is closed either way."""

ADMISSIBLE_SUCCESSORS: dict[BoundaryTag, tuple[BoundaryTag, ...]] = {
    BoundaryTag.BEGIN: (BoundaryTag.MIDDLE, BoundaryTag.END),
    BoundaryTag.MIDDLE: (BoundaryTag.MIDDLE, BoundaryTag.END),
    BoundaryTag.END: (BoundaryTag.BEGIN, BoundaryTag.SINGLE),
    BoundaryTag.SINGLE: (BoundaryTag.BEGIN, BoundaryTag.SINGLE),
}
"""Which tag may follow which: an open word continues, a closed one starts anew."""


def tags_of_word(word: str) -> tuple[BoundaryTag, ...]:
    """The tags a word's characters carry: ``S`` alone, else ``B``, ``M``..., ``E``."""
    if len(word) == 1:
        return (BoundaryTag.SINGLE,)
    return (
        BoundaryTag.BEGIN,
        *([BoundaryTag.MIDDLE] * (len(word) - 2)),
        BoundaryTag.END,
    )


class HiddenMarkovSegmenter(RunSegmenter, Fittable):
    """Character tagging by a four-state hidden Markov model, decoded with Viterbi.

    Parameters
    ----------
    smoothing:
        The count added to every admissible outcome before turning counts into
        probabilities. Must be positive, because a zero would let an unseen
        character or transition have probability zero and every path through
        it score ``-inf``.
    """

    smoothing: float = Field(default=1.0, gt=0)

    _initial_log_probabilities: dict[BoundaryTag, float] = PrivateAttr()
    _transition_log_probabilities: dict[BoundaryTag, dict[BoundaryTag, float]] = (
        PrivateAttr()
    )
    _emission_log_probabilities: dict[BoundaryTag, dict[str, float]] = PrivateAttr()
    _unseen_emission_log_probabilities: dict[BoundaryTag, float] = PrivateAttr()
    _alphabet: tuple[str, ...] = PrivateAttr()

    def fit(self, sentences: SegmentedCorpus | Sequence[Sequence[str]]) -> Self:
        """Count tags, transitions and emissions from segmented sentences.

        Raises
        ------
        InvalidValuesError
            If ``sentences`` is one string, a sentence is one string, a word is
            not a string, or a word contains whitespace.
        EmptyValuesError
            If there are no sentences, a sentence is empty, or a word is empty.
        """
        corpus = SegmentedCorpus.of(sentences)

        initial_counts: Counter[BoundaryTag] = Counter()
        transition_counts: dict[BoundaryTag, Counter[BoundaryTag]] = {
            tag: Counter() for tag in BoundaryTag
        }
        emission_counts: dict[BoundaryTag, Counter[str]] = {
            tag: Counter() for tag in BoundaryTag
        }
        for sentence in corpus:
            tags = [tag for word in sentence for tag in tags_of_word(word)]
            characters = "".join(sentence)
            initial_counts[tags[0]] += 1
            for previous, following in zip(tags, tags[1:], strict=False):
                transition_counts[previous][following] += 1
            for character, tag in zip(characters, tags, strict=True):
                emission_counts[tag][character] += 1

        alphabet = tuple(
            sorted(
                {character for sentence in corpus for character in "".join(sentence)}
            )
        )
        smoothing = self.smoothing

        initial_total = sum(initial_counts.values())
        initial_log_probabilities = {
            tag: math.log(
                (initial_counts[tag] + smoothing)
                / (initial_total + smoothing * len(INITIAL_TAGS))
            )
            for tag in INITIAL_TAGS
        }

        transition_log_probabilities: dict[BoundaryTag, dict[BoundaryTag, float]] = {}
        for previous, successors in ADMISSIBLE_SUCCESSORS.items():
            total = sum(transition_counts[previous].values())
            transition_log_probabilities[previous] = {
                following: math.log(
                    (transition_counts[previous][following] + smoothing)
                    / (total + smoothing * len(successors))
                )
                for following in successors
            }

        emission_log_probabilities: dict[BoundaryTag, dict[str, float]] = {}
        unseen_emission_log_probabilities: dict[BoundaryTag, float] = {}
        n_slots = len(alphabet) + 1
        for tag in BoundaryTag:
            denominator = sum(emission_counts[tag].values()) + smoothing * n_slots
            emission_log_probabilities[tag] = {
                character: math.log((count + smoothing) / denominator)
                for character, count in emission_counts[tag].items()
            }
            unseen_emission_log_probabilities[tag] = math.log(smoothing / denominator)

        self._initial_log_probabilities = initial_log_probabilities
        self._transition_log_probabilities = transition_log_probabilities
        self._emission_log_probabilities = emission_log_probabilities
        self._unseen_emission_log_probabilities = unseen_emission_log_probabilities
        self._alphabet = alphabet
        self._mark_fitted()
        return self

    @property
    def alphabet(self) -> tuple[str, ...]:
        """Every character the corpus contained, in codepoint order.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._alphabet

    @property
    def n_characters(self) -> int:
        """How many distinct characters the corpus contained. See :attr:`alphabet`."""
        return len(self.alphabet)

    def initial_log_probability(self, tag: BoundaryTag) -> float:
        """Log probability that a run starts with ``tag``; ``-inf`` for ``M`` and ``E``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()
        return self._initial_log_probabilities.get(tag, -math.inf)

    def transition_log_probability(
        self, previous: BoundaryTag, following: BoundaryTag
    ) -> float:
        """Log probability of ``following`` after ``previous``; ``-inf`` if forbidden.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()
        return self._transition_log_probabilities[previous].get(following, -math.inf)

    def emission_log_probability(self, tag: BoundaryTag, character: str) -> float:
        """Log probability that ``tag`` emits ``character``, smoothed.

        A character never emitted under ``tag`` -- whether or not the corpus
        contained it at all -- gets the smoothed floor.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()
        return self._emission(tag, character)

    def tags_of(self, text: str) -> tuple[BoundaryTag, ...]:
        """The most probable tag of every non-whitespace character, in order.

        Whitespace carries no tag, because it is never inside a word; each
        run is decoded on its own, so the tag after a space is always ``B`` or
        ``S``. The observed route: :meth:`split` is exactly a cut after every
        ``E`` and ``S`` here.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``text`` is not a string.
        """
        self._check_fitted()
        tags: list[BoundaryTag] = []
        for match in NON_WHITESPACE_RUN.finditer(checked_text(text)):
            tags.extend(self._tags_of_run(match.group()))
        return tuple(tags)

    def _words_of_run(self, run: str) -> tuple[str, ...]:
        self._check_fitted()
        pieces: list[str] = []
        start = 0
        for position, tag in enumerate(self._tags_of_run(run)):
            if tag in FINAL_TAGS:
                pieces.append(run[start : position + 1])
                start = position + 1
        return tuple(pieces)

    def _tags_of_run(self, run: str) -> list[BoundaryTag]:
        """Viterbi over the four tags, constrained to admissible transitions."""
        scores: dict[BoundaryTag, float] = {tag: -math.inf for tag in BoundaryTag}
        for tag in INITIAL_TAGS:
            scores[tag] = self._initial_log_probabilities[tag] + self._emission(
                tag, run[0]
            )

        back_pointers: list[dict[BoundaryTag, BoundaryTag]] = []
        for character in run[1:]:
            next_scores: dict[BoundaryTag, float] = {
                tag: -math.inf for tag in BoundaryTag
            }
            pointers: dict[BoundaryTag, BoundaryTag] = {}
            for previous, successors in ADMISSIBLE_SUCCESSORS.items():
                if scores[previous] == -math.inf:
                    continue
                for following in successors:
                    candidate = (
                        scores[previous]
                        + self._transition_log_probabilities[previous][following]
                        + self._emission(following, character)
                    )
                    if candidate > next_scores[following]:
                        next_scores[following] = candidate
                        pointers[following] = previous
            back_pointers.append(pointers)
            scores = next_scores

        last = max(FINAL_TAGS, key=lambda tag: scores[tag])
        path = [last]
        for pointers in reversed(back_pointers):
            path.append(pointers[path[-1]])
        path.reverse()
        return path

    def _emission(self, tag: BoundaryTag, character: str) -> float:
        return self._emission_log_probabilities[tag].get(
            character, self._unseen_emission_log_probabilities[tag]
        )
