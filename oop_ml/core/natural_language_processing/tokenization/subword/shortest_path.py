"""Shortest-path tokenization: the fewest pieces a vocabulary can spell a word in.

The objective made explicit
---------------------------
Byte pair encoding spells a word by replaying merges in rank order, and the
unigram model spells it by likelihood. In both, how many tokens the word becomes
is a *consequence*: nothing in either rule asks for few. Yet the token count is
the quantity a downstream model pays for, in sequence length and in attention
that grows with its square. Schmidt et al. (2024), whose PathPiece this module
follows, made the count the objective from both sides. Given a vocabulary, the
spelling of a word is the one with the fewest pieces, found by dynamic
programming over the positions between its symbols. And the vocabulary is
chosen to make that count small over the corpus: start from far too many
candidate pieces, and repeatedly drop the ones whose removal would lengthen the
corpus least.

The dynamic programme, and its tie rule
---------------------------------------
``best[i]`` is the fewest pieces the tail of the word from position ``i`` can be
spelled in: one plus the best over pieces starting at ``i`` of the tail that
follows. It runs from the end of the word so that the tie rule reads directly
off the recurrence. Two spellings with the same count are separated by the
*longer piece at the earliest position where they differ*. The assignment named
a third rule, lexicographic order, and it is unreachable: two spellings that
agree up to a position and have pieces of the same length there have the same
piece, because a piece is exactly the symbols it spans. So two rules settle
every tie, and the spec's fixture -- ``abc`` under ``{a, b, c, ab, bc}``, two
pieces either way -- comes out ``ab c``.

A single symbol the vocabulary lacks is still one step, spelled as itself, and
the encode template turns it into the unknown token. The count is exact either
way.

The pruning loop, and what it costs
-----------------------------------
The seed is the unigram model's, borrowed from
:func:`~oop_ml.core.natural_language_processing.tokenization.subword.unigram.seed_pieces`:
the top substrings by frequency times length, plus every single symbol. Then,
until the vocabulary is at the target: spell every word by shortest path and
total the token count over the corpus; for each removable piece, the increase
in that total if it were dropped; rank the pieces by increase, ties
lexicographic; keep the larger of the target and ``shrinking_factor`` of the
current size, never dropping a single symbol.

The increase is computed only for the words whose current shortest path uses
the piece, because a word whose path does not use it has the same shortest
count without it. That makes a round cost one shortest-path pass per piece per
word that uses it, ``O(n V)`` passes over ``n`` distinct words and ``V`` pieces
in the worst case; a word of ``m`` symbols costs ``O(m^2)`` spans per pass, each
span an ``O(m)`` join. Nothing here is optimised, and the usual repair is a trie
over the pieces so that the spans starting at a position are enumerated in
``O(L)`` for a longest piece of ``L`` symbols.

Removing a piece can never shorten the corpus, because every spelling that
avoids the piece was already a candidate before it was removed, and the minimum
over a subset is no smaller. The spec checks that for every removable piece on
Sennrich's corpus. A piece on no word's shortest path has an increase of zero,
and a piece on the *recorded* path whose removal leaves another path of the
same length also has an increase of zero, which is the right answer both times:
the corpus is exactly as long without it.

Worked, on Sennrich's corpus
----------------------------
Eleven symbols, so the smallest vocabulary is twelve, and with one piece per
training word the corpus costs four pieces per occurrence at that size: sixteen
occurrences at ``low``, ``lower``, ``newest``, ``widest`` spelled in three, five,
six and six symbols is ``15 + 10 + 36 + 18 = 79`` tokens. At a vocabulary of
sixteen the four whole words are the pieces worth keeping and the corpus costs
sixteen tokens, one per occurrence, which is the floor. Both figures are
pinned. The unseen ``lowest`` at sixteen has no piece of its own: it is spelled
``l o w e s t</w>`` in six.

Vocabulary order
----------------
The unknown token, the alphabet in codepoint order, then the remaining pieces
in codepoint order. There is no learned rank to order them by: a piece is in
the vocabulary because it survived, and every survivor survived equally.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    VocabularyTooSmallError,
)
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
    alphabet_of,
)
from oop_ml.core.natural_language_processing.tokenization.subword.unigram import (
    PieceScores,
    checked_symbols,
    checked_word,
    seed_pieces,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    LearnedTokenizer,
    PreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.network.purpose import PassPurpose


class ShortestSegmentation:
    """One spelling of a word in the fewest pieces the vocabulary allows.

    Parameters
    ----------
    pieces:
        The pieces in order. At least one, none empty.

    Raises
    ------
    EmptyValuesError
        If there are no pieces or one is empty.
    """

    __slots__ = ("_pieces",)

    def __init__(self, pieces: Sequence[str]) -> None:
        if len(pieces) == 0 or any(not piece for piece in pieces):
            raise EmptyValuesError("a segmentation holds at least one non-empty piece")
        self._pieces = tuple(pieces)

    @property
    def pieces(self) -> tuple[str, ...]:
        """The pieces, in order."""
        return self._pieces

    @property
    def n_tokens(self) -> int:
        """How many pieces the word became, which is what was minimised."""
        return len(self._pieces)

    def __iter__(self) -> Iterator[str]:
        return iter(self._pieces)

    def __len__(self) -> int:
        return len(self._pieces)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShortestSegmentation):
            return NotImplemented
        return self._pieces == other._pieces

    def __hash__(self) -> int:
        return hash(self._pieces)

    def __repr__(self) -> str:
        return f"ShortestSegmentation({list(self._pieces)!r})"


class LearnedPieces:
    """What :func:`learn_shortest_path_pieces` produced.

    Parameters
    ----------
    pieces:
        The surviving pieces, in codepoint order, the alphabet among them.
    n_pruning_rounds:
        How many times the set was shrunk.
    total_token_count:
        How many tokens the training corpus costs under the final pieces.

    Raises
    ------
    EmptyValuesError
        If there are no pieces.
    InvalidValuesError
        If either count is negative.
    """

    __slots__ = ("_n_pruning_rounds", "_pieces", "_total_token_count")

    def __init__(
        self, pieces: Collection[str], n_pruning_rounds: int, total_token_count: int
    ) -> None:
        if len(pieces) == 0:
            raise EmptyValuesError("a learned vocabulary holds at least one piece")
        if n_pruning_rounds < 0 or total_token_count < 0:
            raise InvalidValuesError(
                f"counts are at least zero, got {n_pruning_rounds} pruning rounds "
                f"and {total_token_count} tokens"
            )
        self._pieces = tuple(sorted(pieces))
        self._n_pruning_rounds = int(n_pruning_rounds)
        self._total_token_count = int(total_token_count)

    @property
    def pieces(self) -> tuple[str, ...]:
        """The surviving pieces, in codepoint order."""
        return self._pieces

    @property
    def n_pruning_rounds(self) -> int:
        """How many times the set was shrunk."""
        return self._n_pruning_rounds

    @property
    def total_token_count(self) -> int:
        """The corpus's token count under the final pieces."""
        return self._total_token_count

    def __contains__(self, piece: object) -> bool:
        return piece in self._pieces

    def __iter__(self) -> Iterator[str]:
        return iter(self._pieces)

    def __len__(self) -> int:
        return len(self._pieces)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LearnedPieces):
            return NotImplemented
        return (
            self._pieces == other._pieces
            and self._n_pruning_rounds == other._n_pruning_rounds
            and self._total_token_count == other._total_token_count
        )

    def __hash__(self) -> int:
        return hash((self._pieces, self._n_pruning_rounds, self._total_token_count))

    def __repr__(self) -> str:
        return (
            f"LearnedPieces(n_pieces={len(self._pieces)}, "
            f"n_pruning_rounds={self._n_pruning_rounds}, "
            f"total_token_count={self._total_token_count})"
        )


class ShortestPathLattice:
    """The dynamic programme over one word: every tail's fewest pieces, and the
    piece that starts each tail's best spelling.

    Both public routes read this one table. :attr:`n_tokens` is the count at
    the start of the word and needs no reconstruction; :attr:`segmentation`
    follows the recorded lengths from the start to the end. The spec checks
    that the two agree.

    Parameters
    ----------
    symbols:
        The word's spelling. At least one symbol, none empty.
    pieces:
        The vocabulary. A single symbol absent from it is still one step.

    Raises
    ------
    EmptyValuesError
        If there are no symbols, or one is empty.
    """

    __slots__ = ("_best_count", "_best_length", "_symbols")

    def __init__(self, symbols: Sequence[str], pieces: Collection[str]) -> None:
        checked = checked_symbols(symbols)
        n_symbols = len(checked)
        best_count = [0] * (n_symbols + 1)
        best_length = [0] * (n_symbols + 1)
        for start in range(n_symbols - 1, -1, -1):
            best_count[start] = n_symbols + 1
            for length in range(1, n_symbols - start + 1):
                piece = "".join(checked[start : start + length])
                if length > 1 and piece not in pieces:
                    continue
                count = 1 + best_count[start + length]
                if count < best_count[start] or (
                    count == best_count[start] and length > best_length[start]
                ):
                    best_count[start] = count
                    best_length[start] = length

        self._symbols = checked
        self._best_count = best_count
        self._best_length = best_length

    @property
    def n_tokens(self) -> int:
        """The fewest pieces the whole word can be spelled in."""
        return self._best_count[0]

    @property
    def segmentation(self) -> ShortestSegmentation:
        """The spelling that reaches that count, ties to the longer piece at the
        earliest position where two spellings differ."""
        spelled: list[str] = []
        position = 0
        while position < len(self._symbols):
            length = self._best_length[position]
            spelled.append("".join(self._symbols[position : position + length]))
            position += length
        return ShortestSegmentation(spelled)

    def __repr__(self) -> str:
        return (
            f"ShortestPathLattice(n_symbols={len(self._symbols)}, "
            f"n_tokens={self.n_tokens})"
        )


def token_count_of(symbols: Sequence[str], pieces: Collection[str]) -> int:
    """The fewest pieces ``symbols`` can be spelled in, without the spelling.

    Raises
    ------
    EmptyValuesError
        If there are no symbols, or one is empty.
    """
    return ShortestPathLattice(symbols, pieces).n_tokens


def shortest_path(
    symbols: Sequence[str], pieces: Collection[str]
) -> ShortestSegmentation:
    """The spelling of ``symbols`` in the fewest pieces.

    Ties go to the longer piece at the earliest position where two spellings
    differ. A single symbol absent from ``pieces`` is one step spelled as
    itself.

    Raises
    ------
    EmptyValuesError
        If there are no symbols, or one is empty.
    """
    return ShortestPathLattice(symbols, pieces).segmentation


def corpus_token_count(spelled: Sequence[SpelledWord], pieces: Collection[str]) -> int:
    """How many tokens the corpus costs: each word's fewest pieces times its count."""
    return sum(token_count_of(word.symbols, pieces) * word.count for word in spelled)


def token_count_increases_if_removed(
    spelled: Sequence[SpelledWord], pieces: Collection[str]
) -> PieceScores:
    """How many tokens the corpus would grow by if each removable piece went.

    Computed only over the words whose current shortest path uses the piece;
    every other word's count is unchanged. The single symbols of ``spelled``
    are never removable and are absent from the answer.
    """
    protected = set(alphabet_of(spelled))
    users: dict[str, list[SpelledWord]] = {
        piece: [] for piece in pieces if piece not in protected
    }
    for word in spelled:
        for piece in set(shortest_path(word.symbols, pieces).pieces):
            if piece in users:
                users[piece].append(word)

    increases: dict[str, float] = {}
    for piece, words_using_it in users.items():
        without = {other for other in pieces if other != piece}
        increases[piece] = float(
            sum(
                word.count
                * (
                    token_count_of(word.symbols, without)
                    - token_count_of(word.symbols, pieces)
                )
                for word in words_using_it
            )
        )
    return PieceScores(increases)


def learn_shortest_path_pieces(
    spelled: Sequence[SpelledWord],
    n_pieces: int,
    seed_size: int,
    max_piece_length: int,
    shrinking_factor: float,
) -> LearnedPieces:
    """Seed, then drop the least useful pieces a fraction at a time until
    ``n_pieces`` remain.

    The words are sorted first so that the answer does not depend on the order
    they arrived in. Each round keeps the larger of ``n_pieces`` and
    ``shrinking_factor`` of the current size, so the last round lands on
    ``n_pieces`` exactly; a seed already within the target is never pruned.

    Raises
    ------
    EmptyValuesError
        If there are no words.
    InvalidValuesError
        If the shrinking factor is not strictly between zero and one.
    VocabularyTooSmallError
        If ``n_pieces`` is below the number of distinct symbols, which could
        never be reached because a single symbol is never pruned.
    """
    if len(spelled) == 0:
        raise EmptyValuesError("a vocabulary is learned from at least one word")
    if not 0.0 < shrinking_factor < 1.0:
        raise InvalidValuesError(
            f"the shrinking factor is strictly between zero and one, got "
            f"{shrinking_factor}"
        )

    ordered = sorted(spelled, key=lambda word: word.symbols)
    alphabet = alphabet_of(ordered)
    if n_pieces < len(alphabet):
        raise VocabularyTooSmallError(
            f"{n_pieces} pieces cannot hold the {len(alphabet)} symbols the words "
            f"are spelled in, and a single symbol is never pruned"
        )

    pieces = set(seed_pieces(ordered, seed_size, max_piece_length).pieces)
    n_pruning_rounds = 0
    while len(pieces) > n_pieces:
        increases = token_count_increases_if_removed(ordered, pieces)
        n_kept = max(n_pieces, int(shrinking_factor * len(pieces)))
        n_removable_kept = max(0, n_kept - len(alphabet))
        pieces = set(alphabet) | set(increases.pieces[:n_removable_kept])
        n_pruning_rounds += 1

    return LearnedPieces(pieces, n_pruning_rounds, corpus_token_count(ordered, pieces))


class ShortestPathTokenizer(LearnedTokenizer):
    """Subword tokenization that minimises the token count, after PathPiece.

    Parameters
    ----------
    vocabulary_size:
        How many tokens to learn, counting the unknown token and every symbol
        of the alphabet. The fit lands on it exactly unless the seed was
        already smaller.
    pre_tokenizer:
        Decides where the words are. A piece never crosses a word boundary.
    max_piece_length:
        The longest candidate, in symbols.
    seed_size:
        How many candidate substrings the seed keeps, on top of the alphabet.
    shrinking_factor:
        The fraction of the candidates each pruning round keeps.
    end_of_word_marker:
        Appended to each word's last character, as byte pair encoding does.
    unknown_token:
        Stands in for any symbol the corpus never used.
    """

    vocabulary_size: int = Field(ge=2)
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    max_piece_length: int = Field(default=16, ge=1)
    seed_size: int = Field(default=1000, ge=1)
    shrinking_factor: float = Field(default=0.75, gt=0.0, lt=1.0)
    end_of_word_marker: str = Field(default="</w>", min_length=1)
    unknown_token: str = Field(default="[UNK]", min_length=1)

    _vocabulary: Vocabulary = PrivateAttr()
    _pieces: frozenset[str] = PrivateAttr()
    _n_pruning_rounds: int = PrivateAttr()
    _total_token_count: int = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the pieces from ``corpus``.

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

        learned = learn_shortest_path_pieces(
            spelled,
            n_pieces=self.vocabulary_size - 1,
            seed_size=self.seed_size,
            max_piece_length=self.max_piece_length,
            shrinking_factor=self.shrinking_factor,
        )
        alphabet_set = set(alphabet)
        tokens = [
            self.unknown_token,
            *alphabet,
            *(piece for piece in learned.pieces if piece not in alphabet_set),
        ]

        self._vocabulary = Vocabulary(tokens, unknown_token=self.unknown_token)
        self._pieces = frozenset(learned.pieces)
        self._n_pruning_rounds = learned.n_pruning_rounds
        self._total_token_count = learned.total_token_count
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        """The unknown token, the alphabet, then the other pieces, each in
        codepoint order.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def total_token_count(self) -> int:
        """How many tokens the training corpus costs under the learned pieces.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._total_token_count

    @property
    def n_pruning_rounds(self) -> int:
        """How many times the fit shrank the candidate set.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._n_pruning_rounds

    def best_segmentation(self, word: str) -> ShortestSegmentation:
        """The spelling of one word in the fewest pieces.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If it is empty.
        """
        self._check_fitted()
        return shortest_path(self._symbols_of(checked_word(word)), self._pieces)

    def token_count_of(self, word: str) -> int:
        """How many pieces one word becomes, without spelling it.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If it is empty.
        """
        self._check_fitted()
        return token_count_of(self._symbols_of(checked_word(word)), self._pieces)

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        pieces: list[str] = []
        for word in self.pre_tokenizer.split(text).texts:
            pieces.extend(shortest_path(self._symbols_of(word), self._pieces).pieces)
        return tuple(pieces)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces).replace(self.end_of_word_marker, " ").rstrip(" ")

    def _symbols_of(self, word: str) -> tuple[str, ...]:
        """A word as characters, the last one carrying the end-of-word marker."""
        return (*word[:-1], word[-1] + self.end_of_word_marker)
