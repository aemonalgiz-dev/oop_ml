"""WordPiece: merges chosen by likelihood, words cut by longest match.

The idea
--------
WordPiece (Schuster and Nakajima, 2012; Wu et al., 2016, which is the form BERT
uses) grows a vocabulary the way byte pair encoding does -- start from
characters, repeatedly join the best adjacent pair -- and differs in two
places. The first is what "best" means. Byte pair encoding takes the most
frequent pair; WordPiece takes the pair whose merge most raises the likelihood
of the corpus under a unigram model of its symbols, which works out to
``count(pair) / (count(left) count(right))``. Frequency rewards pairs of common
symbols whether or not they belong together. Likelihood rewards pairs that
occur together *more often than their parts predict*, so two rare symbols that
are only ever seen side by side merge before two common ones that merely happen
to be adjacent a lot.

The second is how a word is cut once the vocabulary exists. Byte pair encoding
replays its merges. WordPiece forgets them: from the start of the word it takes
the longest vocabulary piece that is a prefix of what remains, advances, and
repeats, with every piece after the first looked up in its continuation form.
If at any point no piece matches, the *whole word* becomes the unknown token,
not the character that failed. Two consequences follow. The merges are how the
vocabulary was learned and not what it is, so this class exposes how many there
were and not the merges themselves. And a character is a different symbol at
the start of a word and inside one, so a word is unknown as a whole if any
character of it was never seen *in that position*: on the corpus below ``o``
opens no word, and ``old`` is unknown although every letter in it is in the
alphabet.

Where the marker goes, and why it moves
---------------------------------------
BERT writes the boundary on the continuation piece: ``low`` is ``l ##o ##w``,
and ``##`` says "this continues a word". The shared merge arithmetic in
:mod:`~oop_ml.core.natural_language_processing.tokenization.subword.merging` makes a
merged symbol by plain concatenation, and under that spelling ``##o`` + ``##w``
is ``##o##w``, which is not a piece. The prefix would have to be stripped from
the right symbol at every merge, and the arithmetic has no hook for that, nor
should it grow one for one caller.

The same bit can be written on the other piece. SentencePiece marks the
*word-initial* symbol, ``▁low``, and leaves continuations plain, and under that
spelling concatenation is always right: ``▁l`` + ``o`` is ``▁lo`` and ``o`` +
``w`` is ``ow``, because a word-initial symbol is only ever the left operand of
a merge, and a marker at the far left of the left operand stays at the far left
of the result. So the merges are learned over SentencePiece's spelling and the
vocabulary a caller sees is translated into BERT's, symbol by symbol: strip the
``▁`` from a word-initial piece, prefix ``##`` to a continuation. Encoding
never meets the internal marker at all, since longest match works on the word's
own characters against the translated vocabulary.

The marker has to be a character no corpus word contains, or a continuation
symbol beginning with it would translate as word-initial, so a word holding
``▁`` is refused at ``fit``. A word holding the continuation prefix itself is
refused too, and at encoding time such a word is unknown: BERT's convention
cannot say whether ``##x`` at the start of a word is the word ``##x`` or the
continuation ``x``, and this library would rather say so than guess.

Worked, on Sennrich's toy corpus
--------------------------------
``low`` five times, ``lower`` twice, ``newest`` six, ``widest`` three. Under
the continuation spelling the symbol counts are not the ones byte pair encoding
sees, because a word-initial ``w`` (``widest``, 3) and a continuation ``##w``
(``low``, ``lower``, ``newest``: 13) are different symbols::

    l 7   n 6   w 3   ##d 3   ##e 17   ##i 3   ##o 7   ##r 2   ##s 9   ##t 9   ##w 13

Frequency's first merge would be ``##e ##s`` at 9. Likelihood divides each
pair's count by its two symbols' counts, and two pairs tie at the top with
``3 / (3 * 3) = 1/3``: ``w ##i`` and ``##i ##d``, each a pair of symbols that
never appear apart. The tie goes to the lexicographically smaller pair *in the
internal spelling*, where ``▁`` (U+2581) sorts after every Latin letter, so
``(i, d)`` beats ``(▁w, i)`` and the first merge is ``##id``. The marker's
codepoint is deciding a tie there, which is arbitrary, and stated. The first
three merges, each with the count it carried and the score that chose it::

    ##i ##d   3    3 / (3 * 3) = 1/3
    w ##id    3    3 / (3 * 3) = 1/3, now unopposed
    l ##o     7    7 / (7 * 7) = 1/7

Ten merges in, ``lowest``, which the corpus never contained, is cut by longest
match into ``low`` and ``##est``. Twelve merges exhaust the corpus: every word
is one piece and no pair is left.

Costs
-----
Longest match tries every end offset from the word's end down to one past the
current start, so a word of ``L`` characters costs ``O(L^2)`` lookups. A trie
over the vocabulary makes it ``O(L)`` and is the usual repair;
``max_word_characters`` is the other, and is BERT's own: a word longer than it
is unknown before any lookup, so no single word can cost more than ``100^2``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import InvalidValuesError, VocabularyTooSmallError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    PairScoring,
    SpelledWord,
    alphabet_of,
    learn_merges,
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

WORD_INITIAL_MARKER = "▁"
"""Prefixed to a word's first symbol while the merges are learned.

``▁``, the character SentencePiece uses for the same fact on the same piece.
Never appears in the vocabulary a caller sees, and refused in any corpus word.
"""


class WordPiece(LearnedTokenizer):
    """Subword tokenization by likelihood-scored merging and longest-match cutting.

    Parameters
    ----------
    vocabulary_size:
        How many tokens to learn, counting the unknown token and every symbol
        of the alphabet in both its word-initial and its continuation form.
        The fit stops early if no pair reaches ``minimum_pair_frequency``
        first, and ``vocabulary.n_tokens`` says how many it actually learned.
    pre_tokenizer:
        Decides where the words are. Merges never cross a word boundary and
        longest match runs one word at a time.
    minimum_pair_frequency:
        A pair seen fewer times than this is never merged, whatever its
        likelihood score: a ratio computed from one occurrence is a fact about
        one word.
    continuation_prefix:
        Marks a piece that continues a word rather than starting one. BERT's
        ``##``.
    unknown_token:
        Stands in for a whole word that longest match cannot spell.
    max_word_characters:
        A word longer than this is unknown outright, before any lookup.
    """

    vocabulary_size: int = Field(ge=2)
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    minimum_pair_frequency: int = Field(default=2, ge=1)
    continuation_prefix: str = Field(default="##", min_length=1)
    unknown_token: str = Field(default="[UNK]", min_length=1)
    max_word_characters: int = Field(default=100, ge=1)

    _vocabulary: Vocabulary = PrivateAttr()
    _n_merges: int = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the vocabulary from ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string, or a word
            of it contains the internal word-initial marker ``▁`` or the
            continuation prefix.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        VocabularyTooSmallError
            If ``vocabulary_size`` is below the alphabet plus the unknown token.
        NonUniqueTokensError
            If the unknown token collides with a piece of the vocabulary.
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
                f"{len(alphabet)} word-initial and continuation symbols this corpus "
                f"is spelled in plus the unknown token; the smallest workable size "
                f"is {smallest_possible}"
            )

        merges = learn_merges(
            spelled,
            n_merges=self.vocabulary_size - smallest_possible,
            minimum_pair_frequency=self.minimum_pair_frequency,
            scoring=PairScoring.LIKELIHOOD,
        )
        word_initial = [
            self._piece_of(symbol)
            for symbol in alphabet
            if symbol.startswith(WORD_INITIAL_MARKER)
        ]
        continuation = [
            self._piece_of(symbol)
            for symbol in alphabet
            if not symbol.startswith(WORD_INITIAL_MARKER)
        ]
        tokens = [
            self.unknown_token,
            *word_initial,
            *continuation,
            *(self._piece_of(merge.merged) for merge in merges),
        ]

        self._vocabulary = Vocabulary(tokens, unknown_token=self.unknown_token)
        self._n_merges = merges.n_merges
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        """The unknown token, the word-initial symbols, the continuation symbols
        (each group in codepoint order), then the merged pieces in learned order.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def n_merges(self) -> int:
        """How many merges the fit learned, which is how many pieces follow the
        alphabet in the vocabulary. The merges themselves are not exposed, because
        encoding never reads them; the vocabulary's order is their record.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._n_merges

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()

        pieces: list[str] = []
        for word in self.pre_tokenizer.split(text).texts:
            pieces.extend(self._word_pieces_of(word))
        return tuple(pieces)

    def _text_from(self, pieces: Sequence[str]) -> str:
        words: list[str] = []
        for piece in pieces:
            if piece.startswith(self.continuation_prefix):
                continuation = piece[len(self.continuation_prefix) :]
                if words:
                    words[-1] += continuation
                else:
                    words.append(continuation)
            else:
                words.append(piece)
        return " ".join(words)

    def _word_pieces_of(self, word: str) -> tuple[str, ...]:
        """Greedy longest-match-first, or the unknown token for the whole word."""
        if len(word) > self.max_word_characters or self.continuation_prefix in word:
            return (self.unknown_token,)

        pieces: list[str] = []
        start = 0
        while start < len(word):
            prefix = "" if start == 0 else self.continuation_prefix
            piece = self._longest_piece(word, start, prefix)
            if piece is None:
                return (self.unknown_token,)
            pieces.append(piece)
            start += len(piece) - len(prefix)
        return tuple(pieces)

    def _longest_piece(self, word: str, start: int, prefix: str) -> str | None:
        """The longest vocabulary piece spelling a prefix of ``word[start:]``."""
        for end in range(len(word), start, -1):
            piece = prefix + word[start:end]
            if piece in self._vocabulary:
                return piece
        return None

    def _symbols_of(self, word: str) -> tuple[str, ...]:
        """A word in the internal spelling: the first character marked, the rest plain.

        Raises
        ------
        InvalidValuesError
            If the word contains the marker or the continuation prefix, either
            of which would make the translation to the ``##`` form ambiguous.
        """
        if WORD_INITIAL_MARKER in word:
            raise InvalidValuesError(
                f"the word {word!r} contains {WORD_INITIAL_MARKER!r}, which WordPiece "
                f"reserves to mark a word-initial symbol while learning"
            )
        if self.continuation_prefix in word:
            raise InvalidValuesError(
                f"the word {word!r} contains the continuation prefix "
                f"{self.continuation_prefix!r}, so its pieces could not be told from "
                f"continuations; choose a pre-tokenizer that splits it off or a "
                f"different prefix"
            )
        return (WORD_INITIAL_MARKER + word[0], *word[1:])

    def _piece_of(self, symbol: str) -> str:
        """An internal symbol as the vocabulary shows it: ``##`` on a continuation."""
        if symbol.startswith(WORD_INITIAL_MARKER):
            return symbol[len(WORD_INITIAL_MARKER) :]
        return self.continuation_prefix + symbol
