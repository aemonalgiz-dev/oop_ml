"""The unigram language model: a vocabulary chosen by what it costs to lose each piece.

The idea, and why it runs backwards
-----------------------------------
Byte pair encoding grows a vocabulary. It starts from characters and adds the
piece a greedy rule likes best, so a piece is in the vocabulary because of the
moment it was added, and nothing later revisits that. Kudo (2018) turned the
procedure round. Start from a vocabulary far too large -- the most frequent few
thousand substrings the corpus contains -- give every piece a probability, and
treat a word as having been produced by drawing pieces independently from that
table. Under that model a word has many spellings, each with a probability that
is the product of its pieces' probabilities, and the corpus has a likelihood.
The vocabulary is then *shrunk*: the pieces whose removal would cost the
likelihood least are dropped, a fraction at a time, until the requested size is
reached. A piece survives because the corpus is hard to spell without it, which
is a statement about the whole vocabulary rather than about one greedy step.

Two calculations, both over one lattice
----------------------------------------
Every spelling of a word is a path through the positions between its symbols,
one piece per step, so the questions the model asks are dynamic programmes over
that lattice.

*Which spelling is most probable* is Viterbi: the best score for the tail of a
word from position ``i`` is the best, over pieces starting there, of the
piece's log probability plus the best score of what follows. It is run from the
end of the word so that the tie rule reads naturally: an equal score goes to the
spelling with fewer pieces, and an equal count to the longer piece at the
earliest position where the two spellings differ. Two spellings that agree up to
a position and have pieces of equal length there have the *same* piece, so no
third rule is ever reached.

*How probable is the word at all* is the forward algorithm: the same recurrence
with log-sum-exp in place of max, so ``alpha[j]`` is the log of the total
probability of every way of reaching position ``j`` and ``alpha[end]`` is the
marginal log-likelihood of the word. It sums over the best path and every other,
so it is never below the Viterbi score, and it equals it exactly when a word has
one spelling only. Its mirror ``beta`` runs from the end, and ``alpha[end]``
equals ``beta[0]`` because both are the same total. Both are agreement tests in
the spec.

Expectation-maximisation, worked
--------------------------------
The probabilities are not known and are estimated by EM. The E step asks how
many times each piece is *expected* to have been used: for a piece spanning
positions ``i`` to ``j`` the posterior probability that the word's spelling
passes through it is ``exp(alpha[i] + log p + beta[j] - alpha[end])``, and the
expected count is that times the word's count, summed over the corpus. The M
step sets each log probability to the log of its share of the expected total.

Worked on the smallest corpus that has a choice: the one word ``ab``, once,
spelled ``a b</w>``, with the three pieces ``a``, ``b</w>`` and ``ab</w>`` at
equal probability ``1/3``, which is exactly what the seed produces from that
corpus. The word has two spellings, ``ab</w>`` at ``1/3`` and ``a b</w>`` at
``1/9``, so the marginal is ``4/9`` and the posterior of the one-piece spelling
is ``(1/3) / (4/9) = 3/4`` -- the ``1 / (1 + p)`` form, at ``p = 1/3``. Expected
counts: ``ab</w>`` 0.75, ``a`` 0.25, ``b</w>`` 0.25, total 1.25, and the M step
gives ``0.6, 0.2, 0.2``. A second round puts the one-piece spelling at
``0.6 / 0.64 = 0.9375`` and the pieces at ``0.88235, 0.05882, 0.05882``: the
model is learning that the word is one piece. Both rounds are pinned.

Pruning, and what it costs
--------------------------
The loss of a piece is the fall in corpus log-likelihood if it were removed,
measured on the Viterbi spelling rather than the full marginal: for every word
whose best spelling uses the piece, the best score with the whole table minus
the best score with the piece excluded, times the word's count. A word whose
best spelling does not use the piece contributes nothing, because its best path
is unchanged. That is Kudo's approximation, and it is what keeps the step
affordable. With ``V`` pieces and ``n`` distinct words the exact loss would run
a forward pass on every word for every piece, ``O(n V)`` lattices per round;
this runs Viterbi only on the words whose best spelling uses the piece, which is
still ``O(n V)`` in the worst case and a small fraction of it on any real
corpus, because most pieces sit in few words' best spellings. Pieces are ranked
by loss, ties lexicographic, the top ``shrinking_factor`` of them are kept --
never fewer than the target, never dropping a single symbol -- and the survivors
are renormalised. A single symbol is never a candidate because the vocabulary
must go on spelling everything the corpus contains, and a word may have no
spelling at all without it.

The loop is Kudo's: EM rounds, then stop if the table is small enough, else
prune. EM always precedes the size check, so the probabilities reported are
always estimated on the pieces reported. A seed already within the target is
estimated and never pruned, and the vocabulary comes out smaller than asked, as
byte pair encoding's does when the pairs run out.

Subword regularisation, sampled exactly
---------------------------------------
Kudo's second contribution is that a model trained on the *best* spelling of
every word only ever sees that spelling, and generalises better if it meets the
others in proportion to their probability. Under ``TRAINING`` this tokenizer
draws a spelling from the posterior over spellings, sharpened by
``sampling_temperature``: spelling ``s`` is drawn with probability proportional
to ``P(s) ** sampling_temperature``. That is Kudo's ``alpha``. The field name is
the one the assignment fixed, and it carries a warning: larger is *peakier*,
which is the inverse of the physical convention, where a higher temperature
flattens. At 1 the draw is the exact posterior; as it grows the draw
concentrates on the Viterbi spelling, and at 50 the two agree on every draw in
the spec.

The draw is forward filtering, backward sampling. Run the forward algorithm with
the scaled scores, then walk back from the end of the word choosing each
preceding piece with probability ``exp(alpha[i] + t log p - alpha[j])``. That is
an exact sample from the sharpened posterior. Kudo's implementation samples from
an n-best list instead, an approximation whose bias depends on the list length;
the lattice is already built for the forward pass, so the exact draw costs
nothing more. Under ``PREDICTING`` the generator is never touched, which the
spec checks by asking for a training draw after ten predictions and finding it
equal to a fresh tokenizer's first.

A symbol the table lacks
------------------------
A word may contain a symbol the corpus never used. It still has to be spelled,
so an absent single symbol scores ``UNKNOWN_SYMBOL_PENALTY`` nats below the
rarest piece in the table: any spelling that avoids it wins, and when none can
the word still has a path. The piece comes out as the symbol itself, and the
encode template turns it into the unknown token. SentencePiece uses the same
device with the same constant.

One arithmetic, two spellings
-----------------------------
Everything above is arithmetic over sequences of symbols with counts, and none
of it knows what a symbol is. :class:`UnigramLanguageModel` spells a word as
characters with an end-of-word marker on the last, ``l o w</w>``, exactly as
:class:`~oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding.BytePairEncoding`
does, so the two vocabularies are comparable and decoding is the same string
operation. The framework,
:class:`~oop_ml.core.natural_language_processing.tokenization.subword.sentence_piece.SentencePiece`,
marks the boundary the other way round, a whitespace marker in front. So the
arithmetic lives on :class:`PieceTable`, which takes already-spelled words
(:class:`~oop_ml.core.natural_language_processing.tokenization.subword.merging.SpelledWord`,
the type the merge arithmetic takes) and knows nothing about markers, and each
tokenizer supplies its own spelling and its own decoding. :func:`learn_piece_table`
is the whole seed-estimate-prune loop over that table, :func:`seed_pieces` is
the seed alone, which the shortest-path tokenizer borrows, and
:class:`PieceScores` is the number-per-piece that the seed, the E step and the
pruning step all answer with.

Determinism
-----------
The words are sorted before any sum is formed, so two corpora holding the same
words in a different order give the same table to the last bit. Every ranking
breaks ties lexicographically. The vocabulary is the unknown token, the alphabet
in codepoint order, then the remaining pieces by falling log probability.

What it costs
-------------
Per word of ``n`` symbols the forward, backward and Viterbi passes each examine
``O(n^2)`` candidate spans, and each span joins ``O(n)`` symbols, so a pass is
``O(n^3)`` in the word length; a trie over the table would make it ``O(n L)``
for a longest piece of ``L`` symbols. Nothing here is optimised, and words are
short.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    UnknownTokenError,
    VocabularyTooSmallError,
)
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
    alphabet_of,
)
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

UNKNOWN_SYMBOL_PENALTY = 10.0
"""How many nats below the rarest piece an absent single symbol scores."""


def log_sum_exp(values: Sequence[float]) -> float:
    """``log(sum(exp(v)))`` without overflow, and ``-inf`` for no values."""
    if len(values) == 0:
        return -math.inf
    largest = max(values)
    if largest == -math.inf:
        return -math.inf
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def checked_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    """``symbols`` as a tuple, refused if empty or holding an empty symbol.

    Raises
    ------
    EmptyValuesError
        If there are no symbols, or one of them is the empty string.
    """
    if len(symbols) == 0 or any(not symbol for symbol in symbols):
        raise EmptyValuesError("a word is spelled in at least one non-empty symbol")
    return tuple(symbols)


def checked_word(word: object) -> str:
    """One word confirmed to be a non-empty string.

    Raises
    ------
    InvalidValuesError
        If ``word`` is not a ``str``.
    EmptyValuesError
        If it is empty.
    """
    text = checked_text(word)
    if not text:
        raise EmptyValuesError("a word holds at least one character")
    return text


class Segmentation:
    """One spelling of a word, and the model's log probability of it.

    Parameters
    ----------
    pieces:
        The pieces in order. At least one, none empty.
    log_probability:
        The sum of the pieces' log probabilities under the table that chose
        them, an absent single symbol contributing its penalised score. At
        most zero.

    Raises
    ------
    EmptyValuesError
        If there are no pieces or one is empty.
    InvalidValuesError
        If the log probability is not finite or is above zero.
    """

    __slots__ = ("_log_probability", "_pieces")

    def __init__(self, pieces: Sequence[str], log_probability: float) -> None:
        if len(pieces) == 0 or any(not piece for piece in pieces):
            raise EmptyValuesError("a segmentation holds at least one non-empty piece")

        if not math.isfinite(log_probability) or log_probability > 0.0:
            raise InvalidValuesError(
                f"a log probability is finite and at most zero, got {log_probability}"
            )

        self._pieces = tuple(pieces)
        self._log_probability = float(log_probability)

    @property
    def pieces(self) -> tuple[str, ...]:
        """The pieces, in order."""
        return self._pieces

    @property
    def log_probability(self) -> float:
        """The log probability of this spelling under the table that chose it."""
        return self._log_probability

    @property
    def n_tokens(self) -> int:
        """How many pieces the word became."""
        return len(self._pieces)

    def __iter__(self) -> Iterator[str]:
        return iter(self._pieces)

    def __len__(self) -> int:
        return len(self._pieces)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segmentation):
            return NotImplemented
        return (
            self._pieces == other._pieces
            and self._log_probability == other._log_probability
        )

    def __hash__(self) -> int:
        return hash((self._pieces, self._log_probability))

    def __repr__(self) -> str:
        return (
            f"Segmentation({list(self._pieces)!r}, "
            f"log_probability={self._log_probability:.6g})"
        )


class PieceScores:
    """One non-negative number per piece: a seed frequency, an expected count,
    or a pruning loss.

    Iterating gives the pieces from the highest score to the lowest, ties
    lexicographic, which is the order every ranking in this module reads them
    in. May be empty, since a table of nothing but single symbols has no piece
    that can be pruned.

    Parameters
    ----------
    scores:
        The number for each piece. Every piece a non-empty string, every score
        finite and at least zero.

    Raises
    ------
    InvalidValuesError
        If a piece is not a non-empty string or a score is negative or not
        finite.
    """

    __slots__ = ("_ranked", "_scores")

    def __init__(self, scores: Mapping[str, float]) -> None:
        checked: dict[str, float] = {}
        for piece, score in scores.items():
            if not isinstance(piece, str) or not piece:
                raise InvalidValuesError(
                    f"every piece must be a non-empty string, got {piece!r}"
                )
            if not math.isfinite(score) or score < 0.0:
                raise InvalidValuesError(
                    f"a piece's score is finite and at least zero, got {score} "
                    f"for {piece!r}"
                )
            checked[piece] = float(score)

        self._scores = checked
        self._ranked = tuple(
            sorted(checked, key=lambda piece: (-checked[piece], piece))
        )

    @property
    def pieces(self) -> tuple[str, ...]:
        """The pieces, highest score first, ties lexicographic."""
        return self._ranked

    @property
    def n_pieces(self) -> int:
        """How many pieces are scored."""
        return len(self._ranked)

    @property
    def total(self) -> float:
        """The sum of every score."""
        return sum(self._scores.values())

    def score_of(self, piece: str) -> float:
        """The number for ``piece``.

        Raises
        ------
        UnknownTokenError
            If the piece is not scored.
        """
        score = self._scores.get(piece)
        if score is None:
            raise UnknownTokenError(f"the piece {piece!r} is not among those scored")
        return score

    def __contains__(self, piece: object) -> bool:
        return piece in self._scores

    def __iter__(self) -> Iterator[str]:
        """Iterate the pieces, highest score first."""
        return iter(self._ranked)

    def __len__(self) -> int:
        return len(self._ranked)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PieceScores):
            return NotImplemented
        return self._scores == other._scores

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._scores.items())))

    def __repr__(self) -> str:
        return f"PieceScores(n_pieces={self.n_pieces}, total={self.total:.6g})"


class PieceTable:
    """Every piece the model knows with its log probability: the unigram model.

    All of the lattice arithmetic is behaviour on this object, because every
    one of those calculations reads exactly this table and nothing else. The
    table is immutable; the E step, the M step and pruning each answer with a
    new one.

    Parameters
    ----------
    log_probabilities:
        The log probability of each piece. At least one piece, every piece a
        non-empty string, every value finite and at most zero. The values need
        not sum to one, so a tie can be built by hand; :meth:`normalised`
        makes them.

    Raises
    ------
    EmptyValuesError
        If there are no pieces.
    InvalidValuesError
        If a piece is not a non-empty string, or a log probability is not
        finite or is above zero.
    """

    __slots__ = ("_log_probabilities", "_unknown_score")

    def __init__(self, log_probabilities: Mapping[str, float]) -> None:
        if len(log_probabilities) == 0:
            raise EmptyValuesError("a piece table needs at least one piece")

        checked: dict[str, float] = {}
        for piece, log_probability in log_probabilities.items():
            if not isinstance(piece, str) or not piece:
                raise InvalidValuesError(
                    f"every piece must be a non-empty string, got {piece!r}"
                )
            if not math.isfinite(log_probability) or log_probability > 0.0:
                raise InvalidValuesError(
                    f"a log probability is finite and at most zero, got "
                    f"{log_probability} for {piece!r}"
                )
            checked[piece] = float(log_probability)

        self._log_probabilities = checked
        self._unknown_score = min(checked.values()) - UNKNOWN_SYMBOL_PENALTY

    @classmethod
    def seeded_from(
        cls, spelled: Sequence[SpelledWord], seed_size: int, max_piece_length: int
    ) -> PieceTable:
        """The seed table: :func:`seed_pieces` with each frequency normalised.

        The initial log probability of a piece is the log of its share of the
        seed's total frequency, so a substring seen twice as often starts twice
        as likely.
        """
        seed = seed_pieces(spelled, seed_size, max_piece_length)
        log_total = math.log(seed.total)
        return cls(
            {piece: math.log(seed.score_of(piece)) - log_total for piece in seed}
        )

    @property
    def pieces(self) -> tuple[str, ...]:
        """Every piece, in codepoint order."""
        return tuple(sorted(self._log_probabilities))

    @property
    def pieces_by_probability(self) -> tuple[str, ...]:
        """Every piece, most probable first, ties lexicographic."""
        return tuple(
            sorted(
                self._log_probabilities,
                key=lambda piece: (-self._log_probabilities[piece], piece),
            )
        )

    @property
    def n_pieces(self) -> int:
        """How many pieces the table holds."""
        return len(self._log_probabilities)

    @property
    def unknown_symbol_score(self) -> float:
        """What an absent single symbol scores: the rarest piece less the penalty."""
        return self._unknown_score

    def log_probability_of(self, piece: str) -> float:
        """The log probability of ``piece``.

        Raises
        ------
        UnknownTokenError
            If the piece is not in the table.
        """
        log_probability = self._log_probabilities.get(piece)
        if log_probability is None:
            raise UnknownTokenError(f"the piece {piece!r} is not in this table")
        return log_probability

    def normalised(self) -> PieceTable:
        """The same pieces with their probabilities summing to one."""
        log_total = log_sum_exp(list(self._log_probabilities.values()))
        return PieceTable(
            {
                piece: log_probability - log_total
                for piece, log_probability in self._log_probabilities.items()
            }
        )

    def without(self, piece: str) -> PieceTable:
        """The table with one piece removed, unnormalised.

        Raises
        ------
        UnknownTokenError
            If the piece is not in the table.
        EmptyValuesError
            If it was the only piece.
        """
        if piece not in self._log_probabilities:
            raise UnknownTokenError(f"the piece {piece!r} is not in this table")
        return PieceTable(
            {
                other: log_probability
                for other, log_probability in self._log_probabilities.items()
                if other != piece
            }
        )

    def best_segmentation(self, symbols: Sequence[str]) -> Segmentation:
        """The most probable spelling of ``symbols``: Viterbi over the lattice.

        Ties go to the spelling with fewer pieces, then to the longer piece at
        the earliest position where the spellings differ. An absent single
        symbol is spelled as itself at :attr:`unknown_symbol_score`.

        Raises
        ------
        EmptyValuesError
            If there are no symbols, or one is empty.
        """
        symbols = checked_symbols(symbols)
        n_symbols = len(symbols)
        best_score = [-math.inf] * (n_symbols + 1)
        best_count = [0] * (n_symbols + 1)
        best_length = [0] * (n_symbols + 1)
        best_score[n_symbols] = 0.0

        for start in range(n_symbols - 1, -1, -1):
            for length in range(1, n_symbols - start + 1):
                score = self._span_score(
                    "".join(symbols[start : start + length]), length
                )
                if score is None:
                    continue
                candidate = (
                    score + best_score[start + length],
                    -(1 + best_count[start + length]),
                    length,
                )
                if candidate > (
                    best_score[start],
                    -best_count[start],
                    best_length[start],
                ):
                    best_score[start] = candidate[0]
                    best_count[start] = -candidate[1]
                    best_length[start] = length

        pieces: list[str] = []
        position = 0
        while position < n_symbols:
            length = best_length[position]
            pieces.append("".join(symbols[position : position + length]))
            position += length
        return Segmentation(pieces, best_score[0])

    def marginal_log_likelihood(self, symbols: Sequence[str]) -> float:
        """The log of the total probability of every spelling of ``symbols``.

        The forward algorithm's final entry. Never below the Viterbi score, and
        equal to it when the word has exactly one spelling.

        Raises
        ------
        EmptyValuesError
            If there are no symbols, or one is empty.
        """
        return self._forward(checked_symbols(symbols), 1.0)[-1]

    def sample_segmentation(
        self, symbols: Sequence[str], temperature: float, generator: random.Random
    ) -> Segmentation:
        """One spelling drawn with probability proportional to
        ``P(spelling) ** temperature``: forward filtering, backward sampling.

        The returned log probability is the spelling's unscaled score under
        the table, so it is comparable with :meth:`best_segmentation`.

        Raises
        ------
        EmptyValuesError
            If there are no symbols, or one is empty.
        InvalidValuesError
            If the temperature is not positive.
        """
        symbols = checked_symbols(symbols)
        if temperature <= 0.0:
            raise InvalidValuesError(
                f"the sampling temperature is positive, got {temperature}"
            )

        alpha = self._forward(symbols, temperature)
        pieces_from_the_end: list[str] = []
        end = len(symbols)
        while end > 0:
            starts: list[int] = []
            weights: list[float] = []
            for start in range(end):
                score = self._span_score("".join(symbols[start:end]), end - start)
                if score is None:
                    continue
                starts.append(start)
                weights.append(
                    math.exp(alpha[start] + temperature * score - alpha[end])
                )

            draw = generator.random() * sum(weights)
            chosen = len(starts) - 1
            running = 0.0
            for position, weight in enumerate(weights):
                running += weight
                if draw < running:
                    chosen = position
                    break

            pieces_from_the_end.append("".join(symbols[starts[chosen] : end]))
            end = starts[chosen]

        pieces = tuple(reversed(pieces_from_the_end))
        return Segmentation(pieces, sum(self._piece_score(piece) for piece in pieces))

    def expected_counts(self, spelled: Sequence[SpelledWord]) -> PieceScores:
        """The E step: how often each piece is expected to have been used.

        For a piece spanning positions ``i`` to ``j`` of a word, the posterior
        probability that the word's spelling passes through it is
        ``exp(alpha[i] + log p + beta[j] - alpha[end])``; the expected count is
        that times the word's count, summed over the words. A piece no word
        can use scores zero.
        """
        expected = dict.fromkeys(self._log_probabilities, 0.0)
        for word in spelled:
            symbols = word.symbols
            n_symbols = len(symbols)
            alpha = self._forward(symbols, 1.0)
            beta = self._backward(symbols, 1.0)
            total = alpha[n_symbols]
            for start in range(n_symbols):
                for end in range(start + 1, n_symbols + 1):
                    piece = "".join(symbols[start:end])
                    log_probability = self._log_probabilities.get(piece)
                    if log_probability is None:
                        continue
                    posterior = math.exp(
                        alpha[start] + log_probability + beta[end] - total
                    )
                    expected[piece] += word.count * posterior
        return PieceScores(expected)

    def expectation_maximisation_round(
        self, spelled: Sequence[SpelledWord]
    ) -> PieceTable:
        """One E step and one M step: the same pieces, re-estimated.

        The M step sets each log probability to the log of its share of the
        expected total. A piece whose expected count is exactly zero has no
        finite log, so it takes the smallest log probability of any piece that
        was used -- SentencePiece's rule for the characters it re-adds at the
        end -- and pruning, which ranks by loss, removes it first if it may.
        """
        expected = self.expected_counts(spelled)
        log_total = math.log(expected.total)
        estimated = {
            piece: math.log(expected.score_of(piece)) - log_total
            for piece in expected
            if expected.score_of(piece) > 0.0
        }
        floor = min(estimated.values())
        return PieceTable(
            {piece: estimated.get(piece, floor) for piece in self._log_probabilities}
        )

    def losses_if_removed(self, spelled: Sequence[SpelledWord]) -> PieceScores:
        """How much corpus log-likelihood each removable piece is worth.

        For every word whose Viterbi spelling uses the piece, the best score
        with the piece minus the best score without it, times the word's
        count. The single symbols of ``spelled`` are not candidates and are
        absent from the answer. ``O(n V)`` Viterbi passes in the worst case,
        far fewer in practice.
        """
        protected = set(alphabet_of(spelled))
        users: dict[str, list[SpelledWord]] = {
            piece: [] for piece in self._log_probabilities if piece not in protected
        }
        best_scores: dict[int, float] = {}
        for position, word in enumerate(spelled):
            best = self.best_segmentation(word.symbols)
            best_scores[position] = best.log_probability
            for piece in set(best.pieces):
                if piece in users:
                    users[piece].append(word)

        losses: dict[str, float] = {}
        for piece, words_using_it in users.items():
            if not words_using_it:
                losses[piece] = 0.0
                continue
            without = self.without(piece)
            loss = 0.0
            for word in words_using_it:
                with_piece = self.best_segmentation(word.symbols).log_probability
                without_piece = without.best_segmentation(word.symbols).log_probability
                loss += word.count * (with_piece - without_piece)
            losses[piece] = loss
        return PieceScores(losses)

    def pruned_to(self, n_pieces: int, spelled: Sequence[SpelledWord]) -> PieceTable:
        """At most ``n_pieces`` pieces: every single symbol of ``spelled``, then
        the removable pieces in falling order of :meth:`losses_if_removed`,
        renormalised.

        Never fewer than the single symbols, so the answer may exceed
        ``n_pieces`` when the alphabet alone does. A table already within the
        size is only renormalised.
        """
        if self.n_pieces <= n_pieces:
            return self.normalised()

        alphabet = set(alphabet_of(spelled))
        protected = [piece for piece in self.pieces if piece in alphabet]
        losses = self.losses_if_removed(spelled)
        n_removable_kept = max(0, n_pieces - len(protected))
        kept = [*protected, *losses.pieces[:n_removable_kept]]
        return PieceTable(
            {piece: self._log_probabilities[piece] for piece in kept}
        ).normalised()

    def _span_score(self, piece: str, length: int) -> float | None:
        """The score of one span: its log probability, the unknown score for an
        absent single symbol, and ``None`` for an absent longer span."""
        log_probability = self._log_probabilities.get(piece)
        if log_probability is not None:
            return log_probability
        if length == 1:
            return self._unknown_score
        return None

    def _piece_score(self, piece: str) -> float:
        """The score of a piece a segmentation holds: absent means an unknown symbol."""
        return self._log_probabilities.get(piece, self._unknown_score)

    def _forward(self, symbols: tuple[str, ...], temperature: float) -> list[float]:
        """``alpha[j]``: the log total score of every way of reaching position ``j``."""
        n_symbols = len(symbols)
        alpha = [-math.inf] * (n_symbols + 1)
        alpha[0] = 0.0
        for end in range(1, n_symbols + 1):
            terms: list[float] = []
            for start in range(end):
                score = self._span_score("".join(symbols[start:end]), end - start)
                if score is not None:
                    terms.append(alpha[start] + temperature * score)
            alpha[end] = log_sum_exp(terms)
        return alpha

    def _backward(self, symbols: tuple[str, ...], temperature: float) -> list[float]:
        """``beta[i]``: the log total score of every way of finishing from ``i``."""
        n_symbols = len(symbols)
        beta = [-math.inf] * (n_symbols + 1)
        beta[n_symbols] = 0.0
        for start in range(n_symbols - 1, -1, -1):
            terms: list[float] = []
            for end in range(start + 1, n_symbols + 1):
                score = self._span_score("".join(symbols[start:end]), end - start)
                if score is not None:
                    terms.append(temperature * score + beta[end])
            beta[start] = log_sum_exp(terms)
        return beta

    def __contains__(self, piece: object) -> bool:
        return piece in self._log_probabilities

    def __iter__(self) -> Iterator[str]:
        """Iterate the pieces in codepoint order."""
        return iter(self.pieces)

    def __len__(self) -> int:
        return len(self._log_probabilities)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PieceTable):
            return NotImplemented
        return self._log_probabilities == other._log_probabilities

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._log_probabilities.items())))

    def __repr__(self) -> str:
        return f"PieceTable(n_pieces={self.n_pieces})"


class LearnedPieceTable:
    """What :func:`learn_piece_table` produced: the table and how it got there.

    Parameters
    ----------
    table:
        The final piece table.
    n_pruning_rounds:
        How many times the table was shrunk. Zero when the seed was already
        within the target.

    Raises
    ------
    InvalidValuesError
        If the round count is negative.
    """

    __slots__ = ("_n_pruning_rounds", "_table")

    def __init__(self, table: PieceTable, n_pruning_rounds: int) -> None:
        if n_pruning_rounds < 0:
            raise InvalidValuesError(
                f"a count of pruning rounds is at least zero, got {n_pruning_rounds}"
            )
        self._table = table
        self._n_pruning_rounds = int(n_pruning_rounds)

    @property
    def table(self) -> PieceTable:
        """The final piece table."""
        return self._table

    @property
    def n_pruning_rounds(self) -> int:
        """How many times the table was shrunk."""
        return self._n_pruning_rounds

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LearnedPieceTable):
            return NotImplemented
        return (
            self._table == other._table
            and self._n_pruning_rounds == other._n_pruning_rounds
        )

    def __hash__(self) -> int:
        return hash((self._table, self._n_pruning_rounds))

    def __repr__(self) -> str:
        return (
            f"LearnedPieceTable(n_pieces={self._table.n_pieces}, "
            f"n_pruning_rounds={self._n_pruning_rounds})"
        )


def seed_pieces(
    spelled: Sequence[SpelledWord], seed_size: int, max_piece_length: int
) -> PieceScores:
    """The seed vocabulary, each piece scored by its frequency.

    Every substring of every word of one to ``max_piece_length`` *symbols* is a
    candidate, with a frequency of the summed counts of the words it occurs in,
    counted once per occurrence. The candidates are ranked by frequency times
    length in symbols, ties lexicographic, the top ``seed_size`` are kept, and
    every single symbol is kept whatever its rank, so the seed can exceed
    ``seed_size`` by the alphabet and everything stays spellable. Length is
    counted in symbols, not characters, because ``w</w>`` is one symbol.

    Raises
    ------
    EmptyValuesError
        If there are no words.
    InvalidValuesError
        If ``seed_size`` or ``max_piece_length`` is below one.
    """
    if len(spelled) == 0:
        raise EmptyValuesError("a seed needs at least one spelled word")
    if seed_size < 1 or max_piece_length < 1:
        raise InvalidValuesError(
            f"seed_size and max_piece_length are at least one, got {seed_size} "
            f"and {max_piece_length}"
        )

    frequencies: dict[str, int] = {}
    lengths: dict[str, int] = {}
    for word in spelled:
        symbols = word.symbols
        for start in range(len(symbols)):
            longest = min(max_piece_length, len(symbols) - start)
            for length in range(1, longest + 1):
                piece = "".join(symbols[start : start + length])
                frequencies[piece] = frequencies.get(piece, 0) + word.count
                lengths[piece] = length

    ranked = sorted(
        frequencies, key=lambda piece: (-frequencies[piece] * lengths[piece], piece)
    )
    kept = set(ranked[:seed_size]) | set(alphabet_of(spelled))
    return PieceScores({piece: float(frequencies[piece]) for piece in kept})


def learn_piece_table(
    spelled: Sequence[SpelledWord],
    n_pieces: int,
    seed_size: int,
    max_piece_length: int,
    shrinking_factor: float,
    n_expectation_maximisation_rounds: int,
) -> LearnedPieceTable:
    """Kudo's loop: seed, then repeat EM rounds and prune until ``n_pieces``.

    The words are sorted before anything is summed, so the answer does not
    depend on the order they arrived in. Each pruning keeps the larger of
    ``n_pieces`` and ``shrinking_factor`` of the current table, so the last one
    lands on ``n_pieces`` exactly; a seed already within the target is
    estimated once and never pruned.

    Parameters
    ----------
    spelled:
        Every distinct word as symbols, with its count.
    n_pieces:
        The target size, not counting any unknown token. At least the number
        of distinct symbols.
    seed_size, max_piece_length:
        See :func:`seed_pieces`.
    shrinking_factor:
        The fraction of the table each pruning keeps, strictly between zero
        and one.
    n_expectation_maximisation_rounds:
        How many EM rounds precede each size check. At least one.

    Raises
    ------
    EmptyValuesError
        If there are no words.
    InvalidValuesError
        If the shrinking factor is not strictly between zero and one, or the
        round count is below one.
    VocabularyTooSmallError
        If ``n_pieces`` is below the number of distinct symbols, which could
        never be reached because single symbols are never pruned.
    """
    if len(spelled) == 0:
        raise EmptyValuesError("a piece table is learned from at least one word")
    if not 0.0 < shrinking_factor < 1.0:
        raise InvalidValuesError(
            f"the shrinking factor is strictly between zero and one, got "
            f"{shrinking_factor}"
        )
    if n_expectation_maximisation_rounds < 1:
        raise InvalidValuesError(
            f"at least one expectation-maximisation round is needed, got "
            f"{n_expectation_maximisation_rounds}"
        )

    ordered = sorted(spelled, key=lambda word: word.symbols)
    alphabet = alphabet_of(ordered)
    if n_pieces < len(alphabet):
        raise VocabularyTooSmallError(
            f"{n_pieces} pieces cannot hold the {len(alphabet)} symbols the words "
            f"are spelled in, and a single symbol is never pruned"
        )

    table = PieceTable.seeded_from(ordered, seed_size, max_piece_length)
    n_pruning_rounds = 0
    while True:
        for _ in range(n_expectation_maximisation_rounds):
            table = table.expectation_maximisation_round(ordered)
        if table.n_pieces <= n_pieces:
            break
        n_kept = max(n_pieces, int(shrinking_factor * table.n_pieces))
        table = table.pruned_to(n_kept, ordered)
        n_pruning_rounds += 1
    return LearnedPieceTable(table, n_pruning_rounds)


class UnigramLanguageModel(LearnedTokenizer):
    """Subword tokenization by a probabilistic model over pieces, Kudo (2018).

    Parameters
    ----------
    vocabulary_size:
        How many tokens to learn, counting the unknown token and every symbol
        of the alphabet. The fit lands on it exactly unless the seed was
        already smaller, and ``vocabulary.n_tokens`` says which.
    pre_tokenizer:
        Decides where the words are. A piece never crosses a word boundary.
    seed_size:
        How many candidate substrings the seed keeps, on top of the alphabet.
    max_piece_length:
        The longest candidate, in symbols.
    shrinking_factor:
        The fraction of the table each pruning round keeps.
    n_expectation_maximisation_rounds:
        How many EM rounds are run before each size check.
    sampling_temperature:
        Kudo's ``alpha``: a spelling is drawn under ``TRAINING`` with
        probability proportional to its probability raised to this. Larger is
        peakier, and 1 is the exact posterior.
    end_of_word_marker:
        Appended to each word's last character, as byte pair encoding does, so
        the two vocabularies compare and decoding is the same operation.
    unknown_token:
        Stands in for any symbol the corpus never used.
    random_seed:
        Seeds the training-time draws.
    """

    vocabulary_size: int = Field(ge=2)
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    seed_size: int = Field(default=1000, ge=1)
    max_piece_length: int = Field(default=16, ge=1)
    shrinking_factor: float = Field(default=0.75, gt=0.0, lt=1.0)
    n_expectation_maximisation_rounds: int = Field(default=2, ge=1)
    sampling_temperature: float = Field(default=1.0, gt=0.0)
    end_of_word_marker: str = Field(default="</w>", min_length=1)
    unknown_token: str = Field(default="[UNK]", min_length=1)
    random_seed: int | None = None

    _vocabulary: Vocabulary = PrivateAttr()
    _piece_table: PieceTable = PrivateAttr()
    _n_pruning_rounds: int = PrivateAttr()
    _generator: random.Random = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the piece table from ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        VocabularyTooSmallError
            If ``vocabulary_size`` is below the alphabet plus the unknown token.
        NonUniqueTokensError
            If the unknown token collides with a piece.
        """
        counts = Corpus.of(corpus).word_counts(self.pre_tokenizer)
        spelled = [
            SpelledWord(self._symbols_of(word_count.word), word_count.count)
            for word_count in counts
        ]
        alphabet = alphabet_of(spelled)

        smallest_possible = len(alphabet) + 1
        if self.vocabulary_size < smallest_possible:
            raise VocabularyTooSmallError(
                f"vocabulary_size={self.vocabulary_size} cannot hold the "
                f"{len(alphabet)} symbols this corpus is spelled in plus the "
                f"unknown token; the smallest workable size is {smallest_possible}"
            )

        learned = learn_piece_table(
            spelled,
            n_pieces=self.vocabulary_size - 1,
            seed_size=self.seed_size,
            max_piece_length=self.max_piece_length,
            shrinking_factor=self.shrinking_factor,
            n_expectation_maximisation_rounds=self.n_expectation_maximisation_rounds,
        )
        alphabet_set = set(alphabet)
        tokens = [
            self.unknown_token,
            *alphabet,
            *(
                piece
                for piece in learned.table.pieces_by_probability
                if piece not in alphabet_set
            ),
        ]

        self._vocabulary = Vocabulary(tokens, unknown_token=self.unknown_token)
        self._piece_table = learned.table
        self._n_pruning_rounds = learned.n_pruning_rounds
        self._generator = random.Random(self.random_seed)
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        """The unknown token, the alphabet in codepoint order, then the other
        pieces by falling probability.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def piece_table(self) -> PieceTable:
        """Every piece with its learned log probability.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._piece_table

    @property
    def n_pruning_rounds(self) -> int:
        """How many times the fit shrank the table. See :attr:`piece_table`."""
        self._check_fitted()
        return self._n_pruning_rounds

    def log_probability_of(self, piece: str) -> float:
        """The learned log probability of one piece.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        UnknownTokenError
            If the piece is not in the vocabulary.
        """
        return self.piece_table.log_probability_of(piece)

    def best_segmentation(self, word: str) -> Segmentation:
        """The most probable spelling of one word, with its log probability.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If it is empty.
        """
        return self.piece_table.best_segmentation(self._symbols_of(checked_word(word)))

    def marginal_log_likelihood(self, word: str) -> float:
        """The log probability of one word summed over every spelling.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If it is empty.
        """
        return self.piece_table.marginal_log_likelihood(
            self._symbols_of(checked_word(word))
        )

    def sample_segmentation(self, word: str) -> Segmentation:
        """One spelling of ``word`` drawn at the configured temperature.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If it is empty.
        """
        return self.piece_table.sample_segmentation(
            self._symbols_of(checked_word(word)),
            self.sampling_temperature,
            self._generator,
        )

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        pieces: list[str] = []
        for word in self.pre_tokenizer.split(text).texts:
            symbols = self._symbols_of(word)
            if purpose is PassPurpose.TRAINING:
                segmentation = self._piece_table.sample_segmentation(
                    symbols, self.sampling_temperature, self._generator
                )
            else:
                segmentation = self._piece_table.best_segmentation(symbols)
            pieces.extend(segmentation.pieces)
        return tuple(pieces)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces).replace(self.end_of_word_marker, " ").rstrip(" ")

    def _symbols_of(self, word: str) -> tuple[str, ...]:
        """A word as characters, the last one carrying the end-of-word marker."""
        return (*word[:-1], word[-1] + self.end_of_word_marker)
