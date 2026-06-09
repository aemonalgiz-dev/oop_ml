"""Morfessor Baseline: unsupervised morphology by the shortest two-part description.

The idea
--------
A word list can be written down in two ways. Spell every distinct word out in
full, or keep a list of *pieces* and write each word as a run of references to
them. The first needs no list but spells ``walk`` four times over, once inside
each of ``walk``, ``walks``, ``walked`` and ``walking``. The second spells
``walk`` once and then writes ``walking`` as "piece 1, piece 4", which is
shorter provided the list stays short and the references stay cheap. Creutz and
Lagus (2002, 2005) turned that trade into a segmentation method: choose the
pieces, the *morphs*, so that the list of morphs plus the corpus written in
morphs is as short as it can be. That is the minimum description length
principle, and the two lengths are literally the two parts of one code.

The two parts, in plain words
-----------------------------
**The lexicon cost** is what it takes to write the list of morph *types* down
once. Each type is spelled character by character, each character costing
``-log p(character)`` under the frequencies of characters in the lexicon (not
in the corpus), plus one end-of-morph symbol per type so a reader knows where
one spelling stops and the next begins. Then the counts have to be written too:
given ``N`` morph tokens and ``M`` types, the counts are one of
``C(N - 1, M - 1)`` ways of dealing ``N`` tokens into ``M`` non-empty piles, so
they cost ``log C(N - 1, M - 1)``, computed as
``lgamma(N) - lgamma(M) - lgamma(N - M + 1)``. That is the Baseline's own
log-binomial term.

**The corpus cost** is what it takes to write the corpus as a run of morph
tokens: ``-sum log p(morph)`` over every token, with ``p(morph) = count / N``,
the maximum-likelihood unigram. A morph used often is cheap per use; a morph
used once costs ``log N`` every time. ``corpus_weight`` scales this part, so a
weight above one pays more for a long corpus and splits less, and a weight
below one pays more for a long lexicon and splits more. All logarithms here are
natural, so every cost is in nats.

Worked, on two words
--------------------
The corpus is ``walk walks``, once each. Left whole, the lexicon holds two
types and the corpus two tokens. The corpus cost is ``2 log 2 - 0 = 1.3863``.
The spelling cost counts the lexicon's characters -- ``w a l k`` twice each and
``s`` once, nine characters, plus two end symbols, eleven in all -- and comes to
``11 log 11 - 4 (2 log 2) - 2 log 2 = 19.4454``. With two types for two tokens
there is exactly one way to deal the counts, so the count cost is
``lgamma(2) - lgamma(2) - lgamma(1) = 0``. Total ``20.8317``.

Split into ``walk`` twice and ``s`` once, the corpus cost rises to
``3 log 3 - 2 log 2 = 1.9095`` (three tokens now, one of them rare), but the
lexicon shrinks to five characters plus two ends, ``7 log 7 - 2 log 2 =
12.2351``, and the count cost is ``log C(2, 1) = log 2 = 0.6931``. Total
``14.8378``, which is 5.9939 nats shorter. The shared ``walk`` paid for the
split on a corpus of two words, and every number above is pinned in the spec.

How the search runs
-------------------
Every distinct word starts as one morph. An epoch visits the words in a
seeded shuffled order; for each, its current morphs are taken out of the
accounts and the word is put back whole, then the recursion of Creutz and
Lagus's Baseline runs: compare the total cost with the word left whole against
the cost with it split at each of its ``len - 1`` points, keep the cheapest,
and if a split won, recurse into each half with the other half already in
place. A split has to be *strictly* cheaper than leaving the piece whole, and
among split points the earliest of the cheapest wins; the tie rule is stated
once here. Training stops when an epoch changes the total by less than
``convergence_threshold`` of itself, or at ``max_epochs``.

The accounts are two dictionaries, morph counts and lexicon character counts,
each adjusted as a morph is added or removed, and every cost is computed from
the dictionaries as they stand. That makes each cost evaluation ``O(types)``;
Morfessor keeps running ``sum count log count`` totals to make it ``O(1)``, and
that is the usual repair. It is deliberately not done here, because two
dictionaries that are recomputed from cannot drift, and the spec holds the
dictionaries maintained through the whole search to the morphs recounted from
the final segmentations: exactly equal.

On the corpus ``walk walks walked walking talk talks talked talking play plays
played playing``, each word three times, the search finds six morphs -- the
three stems ``walk``, ``talk``, ``play`` at count 12 each and the three
suffixes ``s``, ``ed``, ``ing`` at count 9 each -- converging on its second
epoch at 185.4634 nats against 301.0812 for the unsplit start, under every
seed tried. A corpus of unrelated words that share no substring is left exactly
as it was, because every split there adds a type and a token and shares
nothing.

Where the greedy search fails, measured
----------------------------------------
The same twelve words *five* times each are left whole, at 367.2274 nats, and
that is a local minimum rather than the answer: the six-morph lexicon costs
262.9413 on that corpus, 104 nats less. The search cannot get there because
every one of the 66 possible first splits from the whole-word start is uphill
-- the cheapest, ``plays`` into ``play`` and ``s``, costs +1.7333 -- whereas at
three repeats that same split is -2.4630 and the walk goes through. The reason
is in the two parts: the lexicon saving of a split is the same whatever the
counts, while the corpus cost of the extra token grows with them, so frequent
words are undersegmented under raw counts. That is exactly why Morfessor 2.0
offers ``--dampening log`` and ``--dampening ones``, and why the weight is a
parameter here: at ``corpus_weight=0.5`` the five-times corpus finds the six
morphs, at 169.4115 nats on its own scale.

Encoding, and the unknown-character rule
----------------------------------------
A word, seen or not, is segmented by Viterbi search over the learned lexicon
for the segmentation of lowest total ``-log p(morph)``. A substring that is not
a morph is allowed only when it is a single character, and it then costs
``len(word) log N + 1``, which is more than any all-morph segmentation of the
word could cost, since such a segmentation has at most ``len(word)`` pieces at
``log N`` each. That is Morfessor's ``badlikelihood`` rule, and its effect is
the one wanted: the unknown token appears exactly where no known morph covers a
character, one unknown token per such character, and nowhere else. Among
equal-cost paths the one whose last piece starts earliest wins, which prefers
the longer final piece; also stated once.

Decoding is byte pair encoding's string operation, not Morfessor's. Every
word-final piece carries ``end_of_word_marker`` in the vocabulary and in an
encoding, so a decoder joins the pieces and turns markers into spaces. The
learned lexicon itself is over plain morphs, as Morfessor's is -- ``s`` at the
end of ``walks`` and ``s`` inside ``walksome`` are one morph -- so the
vocabulary holds each morph twice, plain and marked, because a Viterbi path can
put any morph in either position.

What FlatCat and EM+Prune add
-----------------------------
This is the Baseline. Morfessor FlatCat (Grönroos et al., 2014) puts a hidden
Markov model of four categories -- prefix, stem, suffix, non-morpheme -- over
the same morphs, so that ``s`` as a suffix and ``s`` as a stem are told apart
and a stem cannot follow a suffix. Morfessor EM+Prune (Grönroos et al., 2020)
replaces the search with expectation-maximisation over a unigram lexicon that
is then pruned down to a target size, which is exactly the unigram tokenizer
of :mod:`oop_ml.core.natural_language_processing.tokenization.subword.unigram`.
Neither is built here, and both are named so that they can be declined by
name rather than forgotten.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    LearnedTokenizer,
    PreTokenizer,
    checked_text,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.network.purpose import PassPurpose


class Morph:
    """One morph type of a learned lexicon, with how many tokens of it the corpus has.

    Parameters
    ----------
    text:
        The morph's spelling. At least one character.
    count:
        How many times it occurs across the segmented corpus. At least one,
        since a morph nothing uses is not in the lexicon.

    Raises
    ------
    EmptyValuesError
        If ``text`` is empty.
    InvalidValuesError
        If ``count`` is below one.
    """

    __slots__ = ("_count", "_text")

    def __init__(self, text: str, count: int) -> None:
        if not isinstance(text, str) or not text:
            raise EmptyValuesError("a morph must hold at least one character")

        if count < 1:
            raise InvalidValuesError(
                f"a morph of the lexicon is used at least once, got count {count}"
            )

        self._text = text
        self._count = int(count)

    @property
    def text(self) -> str:
        """The morph's spelling."""
        return self._text

    @property
    def count(self) -> int:
        """How many tokens of it the segmented corpus holds."""
        return self._count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Morph):
            return NotImplemented
        return self._text == other._text and self._count == other._count

    def __hash__(self) -> int:
        return hash((self._text, self._count))

    def __repr__(self) -> str:
        return f"Morph({self._text!r}, {self._count})"


class Morphs:
    """A learned lexicon: every morph type with its count, in codepoint order.

    Parameters
    ----------
    morphs:
        One :class:`Morph` per type, in any order. Non-empty, no spelling
        twice. Stored sorted by spelling so that two lexicons holding the same
        morphs compare equal whatever order they were built in.

    Raises
    ------
    EmptyValuesError
        If there are no morphs.
    InvalidValuesError
        If a spelling appears twice.
    """

    __slots__ = ("_morphs_by_text", "_total")

    def __init__(self, morphs: Sequence[Morph]) -> None:
        if len(morphs) == 0:
            raise EmptyValuesError("a lexicon needs at least one morph")

        morphs_by_text: dict[str, Morph] = {}
        for morph in sorted(morphs, key=lambda each: each.text):
            if morph.text in morphs_by_text:
                raise InvalidValuesError(f"the morph {morph.text!r} is listed twice")
            morphs_by_text[morph.text] = morph

        self._morphs_by_text = morphs_by_text
        self._total = sum(morph.count for morph in morphs)

    @property
    def n_morphs(self) -> int:
        """How many types the lexicon holds."""
        return len(self._morphs_by_text)

    @property
    def total(self) -> int:
        """How many morph tokens the segmented corpus holds, all types together."""
        return self._total

    @property
    def texts(self) -> tuple[str, ...]:
        """The spellings alone, in codepoint order."""
        return tuple(self._morphs_by_text)

    def count_of(self, text: str) -> int:
        """How many tokens of ``text`` there are, or zero if it is not a morph."""
        morph = self._morphs_by_text.get(text)
        return 0 if morph is None else morph.count

    def __getitem__(self, text: str) -> int:
        """The count for ``text``, so that ``morphs["walk"]`` reads well."""
        return self.count_of(text)

    def __contains__(self, text: object) -> bool:
        return text in self._morphs_by_text

    def __iter__(self) -> Iterator[Morph]:
        """Iterate the morphs themselves, in codepoint order."""
        return iter(self._morphs_by_text.values())

    def __len__(self) -> int:
        return self.n_morphs

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Morphs):
            return NotImplemented
        return self._morphs_by_text == other._morphs_by_text

    def __hash__(self) -> int:
        return hash(tuple(self._morphs_by_text.values()))

    def __repr__(self) -> str:
        return f"Morphs(n_morphs={self.n_morphs}, total={self._total})"


class WordSegmentation:
    """How the fit segmented one distinct word of its corpus.

    Parameters
    ----------
    word:
        The word, as the pre-tokenizer spelled it.
    count:
        How many times the corpus used it. At least one.
    morphs:
        The morphs it was cut into, in order. Their concatenation must be the
        word, since a segmentation is a choice of cut points and nothing else.

    Raises
    ------
    EmptyValuesError
        If the word or the morph sequence is empty.
    InvalidValuesError
        If the count is below one or the morphs do not spell the word.
    """

    __slots__ = ("_count", "_morphs", "_word")

    def __init__(self, word: str, count: int, morphs: Sequence[str]) -> None:
        if not isinstance(word, str) or not word:
            raise EmptyValuesError("a segmented word must hold at least one character")

        if count < 1:
            raise InvalidValuesError(
                f"a word of the corpus appeared at least once, got count {count}"
            )

        if len(morphs) == 0:
            raise EmptyValuesError(f"the word {word!r} was cut into no morphs")

        if "".join(morphs) != word:
            raise InvalidValuesError(
                f"the morphs {list(morphs)!r} do not concatenate to the word {word!r}"
            )

        self._word = word
        self._count = int(count)
        self._morphs = tuple(morphs)

    @property
    def word(self) -> str:
        """The word."""
        return self._word

    @property
    def count(self) -> int:
        """How many times the corpus used it."""
        return self._count

    @property
    def morphs(self) -> tuple[str, ...]:
        """The morphs it was cut into, in order."""
        return self._morphs

    @property
    def n_morphs(self) -> int:
        """How many pieces the word became."""
        return len(self._morphs)

    def __iter__(self) -> Iterator[str]:
        return iter(self._morphs)

    def __len__(self) -> int:
        return len(self._morphs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WordSegmentation):
            return NotImplemented
        return (
            self._word == other._word
            and self._count == other._count
            and self._morphs == other._morphs
        )

    def __hash__(self) -> int:
        return hash((self._word, self._count, self._morphs))

    def __repr__(self) -> str:
        return (
            f"WordSegmentation({self._word!r}, count={self._count}, "
            f"morphs={list(self._morphs)!r})"
        )


class Segmentations:
    """The fit's segmentation of every distinct word it saw, addressable by word.

    The second route to the lexicon: :attr:`morphs` recounts the morph tokens
    from these, and the spec holds that recount equal to the accounts the
    search maintained through every add and remove.

    Parameters
    ----------
    segmentations:
        One per distinct word. Non-empty, no word twice.

    Raises
    ------
    EmptyValuesError
        If there are none.
    InvalidValuesError
        If a word is segmented twice.
    """

    __slots__ = ("_segmentations_by_word",)

    def __init__(self, segmentations: Sequence[WordSegmentation]) -> None:
        if len(segmentations) == 0:
            raise EmptyValuesError("segmentations need at least one word")

        segmentations_by_word: dict[str, WordSegmentation] = {}
        for segmentation in segmentations:
            if segmentation.word in segmentations_by_word:
                raise InvalidValuesError(
                    f"the word {segmentation.word!r} is segmented twice"
                )
            segmentations_by_word[segmentation.word] = segmentation

        self._segmentations_by_word = segmentations_by_word

    @property
    def n_words(self) -> int:
        """How many distinct words were segmented."""
        return len(self._segmentations_by_word)

    @property
    def morphs(self) -> Morphs:
        """The lexicon recounted from the segmentations: each morph's tokens
        are its occurrences per word times that word's count."""
        counter: Counter[str] = Counter()
        for segmentation in self._segmentations_by_word.values():
            for morph in segmentation:
                counter[morph] += segmentation.count
        return Morphs([Morph(text, count) for text, count in counter.items()])

    def __getitem__(self, word: str) -> WordSegmentation:
        """The segmentation of ``word``.

        Raises
        ------
        InvalidValuesError
            If the fit never saw the word.
        """
        segmentation = self._segmentations_by_word.get(word)
        if segmentation is None:
            raise InvalidValuesError(
                f"the word {word!r} was not in the corpus; use best_segmentation "
                f"for a word the fit never saw"
            )
        return segmentation

    def __contains__(self, word: object) -> bool:
        return word in self._segmentations_by_word

    def __iter__(self) -> Iterator[WordSegmentation]:
        return iter(self._segmentations_by_word.values())

    def __len__(self) -> int:
        return self.n_words

    def __repr__(self) -> str:
        return f"Segmentations(n_words={self.n_words})"


class DescriptionLength:
    """The three parts of a lexicon-plus-corpus code, and their weighted total.

    Parameters
    ----------
    spelling_cost:
        Nats to spell every morph type, characters and end symbols, under the
        lexicon's own character frequencies.
    count_cost:
        Nats to write the morph counts: ``log C(N - 1, M - 1)``.
    corpus_cost:
        Nats to write the corpus as morph tokens under the unigram model.
    corpus_weight:
        What the corpus cost is multiplied by in the total. Positive.

    Raises
    ------
    InvalidValuesError
        If a cost is negative or not finite, or the weight is not positive.
    """

    __slots__ = ("_corpus_cost", "_corpus_weight", "_count_cost", "_spelling_cost")

    def __init__(
        self,
        spelling_cost: float,
        count_cost: float,
        corpus_cost: float,
        corpus_weight: float,
    ) -> None:
        for name, value in (
            ("spelling_cost", spelling_cost),
            ("count_cost", count_cost),
            ("corpus_cost", corpus_cost),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise InvalidValuesError(
                    f"{name} is a code length and must be finite and non-negative, "
                    f"got {value}"
                )

        if not math.isfinite(corpus_weight) or corpus_weight <= 0.0:
            raise InvalidValuesError(
                f"corpus_weight must be positive and finite, got {corpus_weight}"
            )

        self._spelling_cost = float(spelling_cost)
        self._count_cost = float(count_cost)
        self._corpus_cost = float(corpus_cost)
        self._corpus_weight = float(corpus_weight)

    @property
    def spelling_cost(self) -> float:
        """Nats to spell the lexicon."""
        return self._spelling_cost

    @property
    def count_cost(self) -> float:
        """Nats to write the morph counts."""
        return self._count_cost

    @property
    def lexicon_cost(self) -> float:
        """Spelling plus counts: the whole first part of the code."""
        return self._spelling_cost + self._count_cost

    @property
    def corpus_cost(self) -> float:
        """Nats to write the corpus as morph tokens, before weighting."""
        return self._corpus_cost

    @property
    def corpus_weight(self) -> float:
        """The multiplier on the corpus cost."""
        return self._corpus_weight

    @property
    def total(self) -> float:
        """``lexicon_cost + corpus_weight * corpus_cost``, what the search minimises."""
        return self.lexicon_cost + self._corpus_weight * self._corpus_cost

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DescriptionLength):
            return NotImplemented
        return (
            self._spelling_cost == other._spelling_cost
            and self._count_cost == other._count_cost
            and self._corpus_cost == other._corpus_cost
            and self._corpus_weight == other._corpus_weight
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._spelling_cost,
                self._count_cost,
                self._corpus_cost,
                self._corpus_weight,
            )
        )

    def __repr__(self) -> str:
        return (
            f"DescriptionLength(spelling={self._spelling_cost:.4f}, "
            f"counts={self._count_cost:.4f}, corpus={self._corpus_cost:.4f}, "
            f"weight={self._corpus_weight:g}, total={self.total:.4f})"
        )


def _entropy_term(counts: Mapping[str, int]) -> float:
    """``sum count log count`` over a count table; the part of ``N log N - ...``
    that a maximum-likelihood code length is built from."""
    return sum(count * math.log(count) for count in counts.values())


def _description_length_from_counts(
    morph_counts: Mapping[str, int],
    character_counts: Mapping[str, int],
    corpus_weight: float,
) -> DescriptionLength:
    """The one implementation of the cost, fed by either route.

    ``morph_counts`` is the lexicon with its token counts; ``character_counts``
    is how often each character occurs across the morph *types*, unweighted by
    count. The accounts hand in the tables they maintained incrementally;
    :func:`description_length` hands in tables counted afresh from a
    :class:`Morphs`.
    """
    n_types = len(morph_counts)
    if n_types == 0:
        return DescriptionLength(0.0, 0.0, 0.0, corpus_weight)

    total_tokens = sum(morph_counts.values())
    corpus_cost = total_tokens * math.log(total_tokens) - _entropy_term(morph_counts)

    lexicon_characters = sum(character_counts.values()) + n_types
    spelling_cost = (
        lexicon_characters * math.log(lexicon_characters)
        - _entropy_term(character_counts)
        - n_types * math.log(n_types)
    )

    count_cost = (
        math.lgamma(total_tokens)
        - math.lgamma(n_types)
        - math.lgamma(total_tokens - n_types + 1)
    )

    # Each term is a difference of two quantities that agree in the exact
    # case, so the exact zero can round to a tiny negative; clamp it.
    return DescriptionLength(
        max(spelling_cost, 0.0),
        max(count_cost, 0.0),
        max(corpus_cost, 0.0),
        corpus_weight,
    )


def description_length(morphs: Morphs, corpus_weight: float = 1.0) -> DescriptionLength:
    """The two-part code length of a lexicon, computed from scratch.

    The observable route to the number the search minimises: count the
    characters of every morph type, and evaluate the three costs from the
    module docstring. Pair it with :attr:`MorfessorBaseline.cost`, which the
    search maintained incrementally, and the two agree.

    Raises
    ------
    InvalidValuesError
        If ``corpus_weight`` is not positive.
    """
    if not math.isfinite(corpus_weight) or corpus_weight <= 0.0:
        raise InvalidValuesError(
            f"corpus_weight must be positive and finite, got {corpus_weight}"
        )

    character_counts: Counter[str] = Counter()
    for morph in morphs:
        character_counts.update(morph.text)

    return _description_length_from_counts(
        {morph.text: morph.count for morph in morphs}, character_counts, corpus_weight
    )


class MorphSegmentation:
    """One word cut into morphs by the Viterbi search, with the cost of that cut.

    Parameters
    ----------
    word:
        The word that was segmented.
    morphs:
        The pieces, in order, concatenating to the word. A piece the lexicon
        lacks is a single character, kept as itself here; the encoder is what
        turns it into the unknown token.
    cost:
        The total ``-log p`` of the path, unknown characters at their penalty.

    Raises
    ------
    EmptyValuesError
        If the word or the pieces are empty.
    InvalidValuesError
        If the pieces do not spell the word, or the cost is negative or not
        finite.
    """

    __slots__ = ("_cost", "_morphs", "_word")

    def __init__(self, word: str, morphs: Sequence[str], cost: float) -> None:
        if not isinstance(word, str) or not word:
            raise EmptyValuesError("a segmented word must hold at least one character")

        if len(morphs) == 0:
            raise EmptyValuesError(f"the word {word!r} was cut into no pieces")

        if "".join(morphs) != word:
            raise InvalidValuesError(
                f"the pieces {list(morphs)!r} do not concatenate to the word {word!r}"
            )

        if not math.isfinite(cost) or cost < 0.0:
            raise InvalidValuesError(
                f"a segmentation's cost is a code length and must be finite and "
                f"non-negative, got {cost}"
            )

        self._word = word
        self._morphs = tuple(morphs)
        self._cost = float(cost)

    @property
    def word(self) -> str:
        """The word that was segmented."""
        return self._word

    @property
    def morphs(self) -> tuple[str, ...]:
        """The pieces, in order."""
        return self._morphs

    @property
    def n_morphs(self) -> int:
        """How many pieces the word became."""
        return len(self._morphs)

    @property
    def cost(self) -> float:
        """The path's total code length in nats, as the Viterbi search summed it."""
        return self._cost

    def __iter__(self) -> Iterator[str]:
        return iter(self._morphs)

    def __len__(self) -> int:
        return len(self._morphs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MorphSegmentation):
            return NotImplemented
        return (
            self._word == other._word
            and self._morphs == other._morphs
            and self._cost == other._cost
        )

    def __hash__(self) -> int:
        return hash((self._word, self._morphs, self._cost))

    def __repr__(self) -> str:
        return (
            f"MorphSegmentation({self._word!r}, morphs={list(self._morphs)!r}, "
            f"cost={self._cost:.4f})"
        )


class _MorphAccounts:
    """The mutable books the search keeps: morph counts and lexicon characters.

    Adding a morph raises its count, and if it is a new type, adds its
    characters to the lexicon's character table; removing reverses both, and a
    type whose count reaches zero leaves the lexicon and takes its characters
    with it. Nothing else is stored, so the cost is always computed from
    tables that are exactly what the current segmentation implies.
    """

    __slots__ = ("_character_counts", "_morph_counts")

    def __init__(self) -> None:
        self._morph_counts: dict[str, int] = {}
        self._character_counts: dict[str, int] = {}

    def add(self, morph: str, count: int) -> None:
        """Record ``count`` more tokens of ``morph``."""
        previous = self._morph_counts.get(morph, 0)
        self._morph_counts[morph] = previous + count
        if previous == 0:
            for character in morph:
                self._character_counts[character] = (
                    self._character_counts.get(character, 0) + 1
                )

    def remove(self, morph: str, count: int) -> None:
        """Forget ``count`` tokens of ``morph``, which must all be present."""
        previous = self._morph_counts[morph]
        if count > previous:
            raise InvalidValuesError(
                f"cannot remove {count} tokens of {morph!r}; only {previous} are held"
            )
        remaining = previous - count
        if remaining > 0:
            self._morph_counts[morph] = remaining
            return

        del self._morph_counts[morph]
        for character in morph:
            left = self._character_counts[character] - 1
            if left > 0:
                self._character_counts[character] = left
            else:
                del self._character_counts[character]

    def cost(self, corpus_weight: float) -> float:
        """The total description length of the books as they stand."""
        return _description_length_from_counts(
            self._morph_counts, self._character_counts, corpus_weight
        ).total

    def morphs(self) -> Morphs:
        """The lexicon as a value object."""
        return Morphs(
            [Morph(text, count) for text, count in self._morph_counts.items()]
        )


class MorfessorBaseline(LearnedTokenizer):
    """Unsupervised morphological segmentation by minimum description length.

    Parameters
    ----------
    pre_tokenizer:
        Decides where the words are. Morphs never cross a word boundary.
    corpus_weight:
        Multiplier on the corpus cost in the total. Positive; one is the plain
        two-part code, above one splits less, below one splits more.
    convergence_threshold:
        An epoch that changes the total cost by less than this fraction of the
        cost ends training. Positive.
    max_epochs:
        Epochs to run at most, converged or not. At least one.
    end_of_word_marker:
        Appended to each word's last piece in the vocabulary and in an
        encoding, so decoding knows where the spaces go.
    unknown_token:
        Stands in for a character no morph covers.
    random_seed:
        Seeds the shuffled order the words are visited in each epoch, so the
        same corpus under the same seed learns the same lexicon.
    """

    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    corpus_weight: float = Field(default=1.0, gt=0)
    convergence_threshold: float = Field(default=0.005, gt=0)
    max_epochs: int = Field(default=10, ge=1)
    end_of_word_marker: str = Field(default="</w>", min_length=1)
    unknown_token: str = Field(default="[UNK]", min_length=1)
    random_seed: int | None = None

    _morphs: Morphs = PrivateAttr()
    _segmentations: Segmentations = PrivateAttr()
    _vocabulary: Vocabulary = PrivateAttr()
    _cost: float = PrivateAttr()
    _epochs_run: int = PrivateAttr()
    _converged: bool = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the morph lexicon from ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        NonUniqueTokensError
            If the unknown token, or a marked morph, collides with a morph.
        """
        counts = Corpus.of(corpus).word_counts(self.pre_tokenizer)
        generator = random.Random(self.random_seed)

        accounts = _MorphAccounts()
        segmentations: dict[str, tuple[str, ...]] = {}
        for word_count in counts:
            accounts.add(word_count.word, word_count.count)
            segmentations[word_count.word] = (word_count.word,)

        previous_cost = accounts.cost(self.corpus_weight)
        epochs_run = 0
        converged = False
        while epochs_run < self.max_epochs:
            order = list(counts.words)
            generator.shuffle(order)
            for word in order:
                count = counts.count_of(word)
                for morph in segmentations[word]:
                    accounts.remove(morph, count)
                accounts.add(word, count)
                segmentations[word] = self._resplit(word, count, accounts)

            epochs_run += 1
            current_cost = accounts.cost(self.corpus_weight)
            if abs(previous_cost - current_cost) < self.convergence_threshold * abs(
                current_cost
            ):
                converged = True
                break
            previous_cost = current_cost

        morphs = accounts.morphs()
        tokens = [
            self.unknown_token,
            *morphs.texts,
            *(text + self.end_of_word_marker for text in morphs.texts),
        ]

        self._morphs = morphs
        self._segmentations = Segmentations(
            [
                WordSegmentation(word, counts.count_of(word), pieces)
                for word, pieces in segmentations.items()
            ]
        )
        self._vocabulary = Vocabulary(tokens, unknown_token=self.unknown_token)
        self._cost = accounts.cost(self.corpus_weight)
        self._epochs_run = epochs_run
        self._converged = converged
        self._mark_fitted()
        return self

    def _resplit(
        self, morph: str, count: int, accounts: _MorphAccounts
    ) -> tuple[str, ...]:
        """The Baseline recursion on one piece already present in the accounts.

        Compares the cost with the piece whole against every binary split,
        with everything else in the accounts as it stands. A split must be
        strictly cheaper to win, and the earliest cheapest split point wins.
        Returns the pieces, leaving exactly those in the accounts.
        """
        whole_cost = accounts.cost(self.corpus_weight)
        accounts.remove(morph, count)

        best_cost = whole_cost
        best_position: int | None = None
        for position in range(1, len(morph)):
            left, right = morph[:position], morph[position:]
            accounts.add(left, count)
            accounts.add(right, count)
            split_cost = accounts.cost(self.corpus_weight)
            accounts.remove(left, count)
            accounts.remove(right, count)
            if split_cost < best_cost:
                best_cost = split_cost
                best_position = position

        if best_position is None:
            accounts.add(morph, count)
            return (morph,)

        left, right = morph[:best_position], morph[best_position:]
        accounts.add(left, count)
        accounts.add(right, count)
        return self._resplit(left, count, accounts) + self._resplit(
            right, count, accounts
        )

    @property
    def vocabulary(self) -> Vocabulary:
        """The unknown token, every morph plain, then every morph marked.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def morphs(self) -> Morphs:
        """The learned lexicon with its counts, as the search's accounts hold it.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._morphs

    @property
    def segmentations(self) -> Segmentations:
        """How each distinct training word was cut, the other route to the lexicon.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._segmentations

    @property
    def cost(self) -> float:
        """The final total description length in nats.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._cost

    @property
    def epochs_run(self) -> int:
        """How many epochs the search ran.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._epochs_run

    @property
    def converged(self) -> bool:
        """Whether the last epoch changed the cost by less than the threshold.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._converged

    def best_segmentation(self, word: str) -> MorphSegmentation:
        """The cheapest cut of one word into known morphs, by Viterbi search.

        A character no morph covers stays as a one-character piece at the
        unknown penalty; see the module docstring for the rule and the tie.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If ``word`` is empty.
        """
        self._check_fitted()
        word = checked_text(word)
        if not word:
            raise EmptyValuesError("cannot segment an empty word")

        penalty = self._unknown_character_cost(word)
        log_total = math.log(self._morphs.total)

        best_cost: list[float] = [math.inf] * (len(word) + 1)
        best_cost[0] = 0.0
        cut_before: list[int] = [0] * (len(word) + 1)
        for end in range(1, len(word) + 1):
            for start in range(end):
                piece = word[start:end]
                count = self._morphs.count_of(piece)
                if count > 0:
                    piece_cost = log_total - math.log(count)
                elif end - start == 1:
                    piece_cost = penalty
                else:
                    continue
                path_cost = best_cost[start] + piece_cost
                if path_cost < best_cost[end]:
                    best_cost[end] = path_cost
                    cut_before[end] = start

        pieces: list[str] = []
        end = len(word)
        while end > 0:
            start = cut_before[end]
            pieces.append(word[start:end])
            end = start
        pieces.reverse()
        return MorphSegmentation(word, pieces, best_cost[len(word)])

    def cost_of_segmentation(self, morphs: Sequence[str]) -> float:
        """The code length of one cut, summed piece by piece.

        The observable route beside :meth:`best_segmentation`'s running total:
        a known piece costs ``-log p(morph)``, an unknown single character the
        penalty for the word the pieces spell, and the two routes agree on the
        segmentation the search chose.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        EmptyValuesError
            If there are no pieces or one is empty.
        InvalidValuesError
            If an unknown piece is longer than one character, which no
            segmentation this model produces can hold.
        """
        self._check_fitted()
        if len(morphs) == 0 or any(not piece for piece in morphs):
            raise EmptyValuesError("a segmentation needs at least one non-empty piece")

        word = "".join(morphs)
        penalty = self._unknown_character_cost(word)
        log_total = math.log(self._morphs.total)

        total = 0.0
        for piece in morphs:
            count = self._morphs.count_of(piece)
            if count > 0:
                total += log_total - math.log(count)
            elif len(piece) == 1:
                total += penalty
            else:
                raise InvalidValuesError(
                    f"{piece!r} is not a morph and is longer than one character; "
                    f"an unknown run is spelled one character at a time"
                )
        return total

    def _unknown_character_cost(self, word: str) -> float:
        """Morfessor's ``badlikelihood``: dearer than any all-morph cut of the word."""
        return len(word) * math.log(self._morphs.total) + 1.0

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        pieces: list[str] = []
        for word in self.pre_tokenizer.split(text).texts:
            morphs = self.best_segmentation(word).morphs
            pieces.extend(morphs[:-1])
            pieces.append(morphs[-1] + self.end_of_word_marker)
        return tuple(pieces)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces).replace(self.end_of_word_marker, " ").rstrip(" ")
