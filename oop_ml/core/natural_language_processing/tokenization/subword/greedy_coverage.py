"""Greedy coverage: the vocabulary as a weighted maximum-coverage problem.

The reframing
-------------
Lim et al. (2024), whose GreedTok this module follows, asked what a subword
vocabulary is *for* and answered: to cover the corpus. Every position in every
word should lie under some multi-symbol piece, because a position left to a
single symbol is a token spent on one character. Choosing a vocabulary of a
given size that covers the most positions, each word weighted by how often it
occurs, is the weighted maximum coverage problem. That problem is NP-hard, and
the greedy rule -- repeatedly take the candidate covering the most positions not
yet covered -- is its classic approximation, guaranteed to reach at least
``1 - 1/e`` of the best possible coverage. So the vocabulary is built by that
rule and nothing else. No merges, no probabilities, no likelihood: a piece is in
the vocabulary because, at the moment it was chosen, it covered more of what was
still bare than anything else did.

What a piece covers
-------------------
Applying a piece to a word covers its leftmost non-overlapping occurrences on
positions no higher-ranked piece has already taken: scan from the left, and
wherever the piece matches on bare positions, claim them and skip past. The
pieces chosen so far are applied in rank order first, and the candidate then
covers whatever is still bare. Coverage is counted in *symbols*, the positions
of the spelled word, not in characters: ``t</w>`` is one position. The choice
matters, and it is pinned. Candidates are every substring of two to
``max_piece_length`` symbols; single symbols are in the vocabulary
unconditionally, cover nothing that was not already going to be a token, and
are not candidates.

Worked, on Sennrich's corpus
----------------------------
``low`` five times, ``lower`` twice, ``newest`` six, ``widest`` three, spelled
``l o w</w>``, ``l o w e r</w>``, ``n e w e s t</w>``, ``w i d e s t</w>``.

The first choice is ``newest</w>``: six positions in a word seen six times,
``36``. Its rivals are ``ewest</w>`` at ``5 x 6 = 30``, ``est</w>`` at
``3 x (6 + 3) = 27`` and ``west</w>`` at ``4 x 6 = 24``. Had coverage been
counted in characters with the marker as one, ``est</w>`` would score
``4 x 9 = 36`` and tie ``newest</w>``, and the lexicographic tie rule would
choose it -- which is why the unit is stated rather than assumed.

Then ``widest</w>`` at ``6 x 3 = 18``; then ``low</w>`` at ``3 x 5 = 15``; then
``lower</w>`` at ``5 x 2 = 10``. The unmarked ``low`` never wins, because the
word ``low`` ends in the symbol ``w</w>`` and so does not contain it: it occurs
only inside ``lower``, worth ``3 x 2 = 6``, and the whole word beats it there.
After four choices every one of the ``15 + 10 + 36 + 18 = 79`` weighted
positions is covered, so no candidate can cover anything and the fit stops at
four pieces however many were asked for; ``vocabulary.n_tokens`` says so. The
four choices, their coverages and the stop are pinned.

Encoding replays the rule: ``lower`` becomes ``lower</w>``, and the unseen
``lowest`` becomes six bare symbols, ``l o w e s t</w>``, because no chosen
piece occurs in it -- ``low</w>`` needs the marked ``w</w>`` that ``lowest`` has
in the middle as a bare ``w``. A piece does cover part of an unseen word when
it occurs in it: ``unwidest`` becomes ``u`` ``n`` ``widest</w>``.

Two routes to one number
------------------------
Each choice records the coverage it won with, and those marginal gains
telescope: their sum is the number of weighted positions the whole vocabulary
covers. Encoding every training word and summing the covered positions reaches
the same number by a different route, ``36 + 18 + 15 + 10 = 79`` on Sennrich's
corpus, and the spec asserts the two agree.

Ties and cost
-------------
Ties in coverage go to the lexicographically smaller piece. Each step scores
every remaining candidate against every word, ``O(candidates x corpus)`` per
chosen piece; GreedTok's repair is a priority queue over candidates with lazy
re-scoring, since a candidate's gain can only fall as others are chosen. Nothing
here is optimised.

Vocabulary order
----------------
The unknown token, the alphabet in codepoint order, then the chosen pieces in
the order they were chosen, which is their rank.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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
from oop_ml.core.natural_language_processing.tokenization.subword.unigram import (
    checked_symbols,
    checked_word,
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


class ChosenPiece:
    """One piece the greedy rule chose, and the coverage it won with.

    Parameters
    ----------
    symbols:
        The symbols the piece spans. At least two, none empty, because a single
        symbol is never a candidate.
    coverage:
        How many weighted positions it newly covered when chosen. At least one,
        since a piece covering nothing is never chosen.

    Raises
    ------
    EmptyValuesError
        If a symbol is empty.
    InvalidValuesError
        If there are fewer than two symbols, or the coverage is below one.
    """

    __slots__ = ("_coverage", "_symbols")

    def __init__(self, symbols: Sequence[str], coverage: int) -> None:
        checked = checked_symbols(symbols)
        if len(checked) < 2:
            raise InvalidValuesError(
                f"a chosen piece spans at least two symbols, got {list(checked)!r}"
            )
        if coverage < 1:
            raise InvalidValuesError(
                f"a chosen piece covered at least one position, got {coverage}"
            )
        self._symbols = checked
        self._coverage = int(coverage)

    @property
    def symbols(self) -> tuple[str, ...]:
        """The symbols the piece spans."""
        return self._symbols

    @property
    def piece(self) -> str:
        """The piece as the vocabulary spells it: its symbols joined."""
        return "".join(self._symbols)

    @property
    def n_symbols(self) -> int:
        """How many positions one occurrence covers."""
        return len(self._symbols)

    @property
    def coverage(self) -> int:
        """The weighted count of positions it newly covered when chosen."""
        return self._coverage

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChosenPiece):
            return NotImplemented
        return self._symbols == other._symbols and self._coverage == other._coverage

    def __hash__(self) -> int:
        return hash((self._symbols, self._coverage))

    def __repr__(self) -> str:
        return f"ChosenPiece({self.piece!r}, coverage={self._coverage})"


class ChosenPieces:
    """Every piece the greedy rule chose, in the order it chose them.

    The order is the rank, and the rank is the model: encoding applies the
    pieces in this order. May be empty, since a corpus of one-symbol words has
    nothing to cover.

    Parameters
    ----------
    chosen:
        The pieces in rank order. No piece twice.

    Raises
    ------
    InvalidValuesError
        If a piece appears twice.
    """

    __slots__ = ("_chosen", "_coverages")

    def __init__(self, chosen: Sequence[ChosenPiece]) -> None:
        coverages: dict[str, int] = {}
        for rank, chosen_piece in enumerate(chosen):
            if chosen_piece.piece in coverages:
                raise InvalidValuesError(
                    f"the piece {chosen_piece.piece!r} is chosen twice, the second "
                    f"time at rank {rank}"
                )
            coverages[chosen_piece.piece] = chosen_piece.coverage

        self._chosen = tuple(chosen)
        self._coverages = coverages

    @property
    def pieces(self) -> tuple[str, ...]:
        """The pieces as strings, in rank order."""
        return tuple(chosen_piece.piece for chosen_piece in self._chosen)

    @property
    def n_pieces(self) -> int:
        """How many pieces were chosen."""
        return len(self._chosen)

    @property
    def total_coverage(self) -> int:
        """The sum of every marginal coverage: how many weighted positions the
        whole set covers."""
        return sum(self._coverages.values())

    def coverage_of(self, piece: str) -> int:
        """The coverage ``piece`` won with when it was chosen.

        Raises
        ------
        UnknownTokenError
            If the piece was never chosen.
        """
        coverage = self._coverages.get(piece)
        if coverage is None:
            raise UnknownTokenError(f"the piece {piece!r} was never chosen")
        return coverage

    def __contains__(self, piece: object) -> bool:
        return piece in self._coverages

    def __iter__(self) -> Iterator[ChosenPiece]:
        """Iterate the chosen pieces in rank order."""
        return iter(self._chosen)

    def __len__(self) -> int:
        return len(self._chosen)

    def __getitem__(self, rank: int) -> ChosenPiece:
        return self._chosen[rank]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChosenPieces):
            return NotImplemented
        return self._chosen == other._chosen

    def __hash__(self) -> int:
        return hash(self._chosen)

    def __repr__(self) -> str:
        return f"ChosenPieces(n_pieces={self.n_pieces})"


class CoverageSegmentation:
    """One word spelled by cover: the pieces, and how much of the word they took.

    Parameters
    ----------
    pieces:
        The pieces in order, chosen pieces and bare single symbols alike.
    n_covered_symbols:
        How many of the word's positions lie under a chosen piece. The rest are
        bare and spelled as single symbols.

    Raises
    ------
    EmptyValuesError
        If there are no pieces or one is empty.
    InvalidValuesError
        If the covered count is negative.
    """

    __slots__ = ("_n_covered_symbols", "_pieces")

    def __init__(self, pieces: Sequence[str], n_covered_symbols: int) -> None:
        if len(pieces) == 0 or any(not piece for piece in pieces):
            raise EmptyValuesError("a segmentation holds at least one non-empty piece")
        if n_covered_symbols < 0:
            raise InvalidValuesError(
                f"a covered count is at least zero, got {n_covered_symbols}"
            )
        self._pieces = tuple(pieces)
        self._n_covered_symbols = int(n_covered_symbols)

    @property
    def pieces(self) -> tuple[str, ...]:
        """The pieces, in order."""
        return self._pieces

    @property
    def n_tokens(self) -> int:
        """How many pieces the word became."""
        return len(self._pieces)

    @property
    def n_covered_symbols(self) -> int:
        """How many of the word's positions a chosen piece covers."""
        return self._n_covered_symbols

    def __iter__(self) -> Iterator[str]:
        return iter(self._pieces)

    def __len__(self) -> int:
        return len(self._pieces)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoverageSegmentation):
            return NotImplemented
        return (
            self._pieces == other._pieces
            and self._n_covered_symbols == other._n_covered_symbols
        )

    def __hash__(self) -> int:
        return hash((self._pieces, self._n_covered_symbols))

    def __repr__(self) -> str:
        return (
            f"CoverageSegmentation({list(self._pieces)!r}, "
            f"n_covered_symbols={self._n_covered_symbols})"
        )


def newly_covered_positions(
    symbols: Sequence[str], covered: Sequence[bool], piece_symbols: Sequence[str]
) -> tuple[int, ...]:
    """The positions ``piece_symbols`` would claim on ``symbols``: its leftmost
    non-overlapping occurrences lying wholly on positions not yet covered.

    Raises
    ------
    InvalidValuesError
        If ``covered`` is not one flag per symbol.
    """
    if len(covered) != len(symbols):
        raise InvalidValuesError(
            f"one coverage flag per symbol: {len(symbols)} symbols, "
            f"{len(covered)} flags"
        )

    length = len(piece_symbols)
    claimed: list[int] = []
    position = 0
    while position + length <= len(symbols):
        span = range(position, position + length)
        if tuple(symbols[position : position + length]) == tuple(
            piece_symbols
        ) and not any(covered[index] for index in span):
            claimed.extend(span)
            position += length
        else:
            position += 1
    return tuple(claimed)


def cover(symbols: Sequence[str], chosen: ChosenPieces) -> CoverageSegmentation:
    """Spell one word by applying the chosen pieces in rank order.

    Each piece claims its leftmost non-overlapping bare occurrences; whatever
    stays bare is a single symbol; the pieces are then read off left to right.

    Raises
    ------
    EmptyValuesError
        If there are no symbols, or one is empty.
    """
    checked = checked_symbols(symbols)
    covered = [False] * len(checked)
    piece_lengths_by_start: dict[int, int] = {}
    for chosen_piece in chosen:
        claimed = newly_covered_positions(checked, covered, chosen_piece.symbols)
        for position in claimed:
            covered[position] = True
        for position in claimed[:: chosen_piece.n_symbols]:
            piece_lengths_by_start[position] = chosen_piece.n_symbols

    pieces: list[str] = []
    position = 0
    while position < len(checked):
        length = piece_lengths_by_start.get(position, 1)
        pieces.append("".join(checked[position : position + length]))
        position += length
    return CoverageSegmentation(pieces, sum(covered))


def learn_chosen_pieces(
    spelled: Sequence[SpelledWord], n_pieces: int, max_piece_length: int
) -> ChosenPieces:
    """Up to ``n_pieces`` pieces, each the candidate covering the most bare
    weighted positions at the moment it was chosen.

    Stops early when no candidate covers anything, so the answer may be
    shorter than asked. Ties go to the lexicographically smaller piece. The
    words are sorted first so the answer does not depend on their order, though
    coverage is a sum of integers and would agree regardless.

    Raises
    ------
    EmptyValuesError
        If there are no words.
    InvalidValuesError
        If ``n_pieces`` is negative or ``max_piece_length`` is below one.
    """
    if len(spelled) == 0:
        raise EmptyValuesError("a vocabulary is learned from at least one word")
    if n_pieces < 0 or max_piece_length < 1:
        raise InvalidValuesError(
            f"n_pieces is at least zero and max_piece_length at least one, got "
            f"{n_pieces} and {max_piece_length}"
        )

    ordered = sorted(spelled, key=lambda word: word.symbols)
    candidates: dict[str, tuple[str, ...]] = {}
    for word in ordered:
        symbols = word.symbols
        for start in range(len(symbols)):
            longest = min(max_piece_length, len(symbols) - start)
            for length in range(2, longest + 1):
                span = symbols[start : start + length]
                candidates["".join(span)] = span

    covered = [[False] * len(word.symbols) for word in ordered]
    chosen: list[ChosenPiece] = []
    remaining = sorted(candidates)
    while len(chosen) < n_pieces and remaining:
        best_piece: str | None = None
        best_coverage = 0
        for piece in remaining:
            coverage = sum(
                word.count
                * len(newly_covered_positions(word.symbols, flags, candidates[piece]))
                for word, flags in zip(ordered, covered, strict=True)
            )
            if coverage > best_coverage:
                best_piece = piece
                best_coverage = coverage

        if best_piece is None:
            break

        for word, flags in zip(ordered, covered, strict=True):
            for position in newly_covered_positions(
                word.symbols, flags, candidates[best_piece]
            ):
                flags[position] = True
        chosen.append(ChosenPiece(candidates[best_piece], best_coverage))
        remaining.remove(best_piece)

    return ChosenPieces(chosen)


class GreedyCoverageTokenizer(LearnedTokenizer):
    """Subword tokenization by greedy weighted maximum coverage, after GreedTok.

    Parameters
    ----------
    vocabulary_size:
        How many tokens to learn, counting the unknown token and every symbol
        of the alphabet. The fit stops early once no candidate covers a bare
        position, and ``vocabulary.n_tokens`` says how many it reached.
    pre_tokenizer:
        Decides where the words are. A piece never crosses a word boundary.
    max_piece_length:
        The longest candidate, in symbols.
    end_of_word_marker:
        Appended to each word's last character, as byte pair encoding does.
    unknown_token:
        Stands in for any symbol the corpus never used.
    """

    vocabulary_size: int = Field(ge=2)
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    max_piece_length: int = Field(default=16, ge=1)
    end_of_word_marker: str = Field(default="</w>", min_length=1)
    unknown_token: str = Field(default="[UNK]", min_length=1)

    _vocabulary: Vocabulary = PrivateAttr()
    _chosen_pieces: ChosenPieces = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Choose the pieces from ``corpus``.

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

        chosen = learn_chosen_pieces(
            spelled,
            n_pieces=self.vocabulary_size - smallest_possible,
            max_piece_length=self.max_piece_length,
        )
        tokens = [self.unknown_token, *alphabet, *chosen.pieces]

        self._vocabulary = Vocabulary(tokens, unknown_token=self.unknown_token)
        self._chosen_pieces = chosen
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        """The unknown token, the alphabet in codepoint order, then the chosen
        pieces in rank order.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def chosen_pieces(self) -> ChosenPieces:
        """Every chosen piece with the coverage it won with, in rank order.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._chosen_pieces

    @property
    def n_chosen_pieces(self) -> int:
        """How many pieces the fit chose. See :attr:`chosen_pieces`."""
        return self.chosen_pieces.n_pieces

    def coverage_of(self, piece: str) -> int:
        """The weighted positions ``piece`` newly covered when it was chosen.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        UnknownTokenError
            If the piece was never chosen.
        """
        return self.chosen_pieces.coverage_of(piece)

    def best_segmentation(self, word: str) -> CoverageSegmentation:
        """One word spelled by cover, with how much of it the chosen pieces took.

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
        return cover(self._symbols_of(checked_word(word)), self._chosen_pieces)

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        pieces: list[str] = []
        for word in self.pre_tokenizer.split(text).texts:
            pieces.extend(cover(self._symbols_of(word), self._chosen_pieces).pieces)
        return tuple(pieces)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces).replace(self.end_of_word_marker, " ").rstrip(" ")

    def _symbols_of(self, word: str) -> tuple[str, ...]:
        """A word as characters, the last one carrying the end-of-word marker."""
        return (*word[:-1], word[-1] + self.end_of_word_marker)
