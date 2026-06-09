"""SentencePiece: whitespace as a symbol, so that no pre-tokenizer is needed.

What it is, and what it is not
------------------------------
SentencePiece is not a third subword algorithm. Kudo and Richardson (2018)
built a *framework* around two that already existed, byte pair encoding and the
unigram language model, and the contribution is the treatment of the text
before either runs. Every other tokenizer in this package takes a pre-tokenizer
that decides where the words are, and that rule is a fact about a language:
spaces for English, nothing at all for Japanese. SentencePiece refuses the
argument. Whitespace is turned into an ordinary symbol, ``▁`` (U+2581, LOWER
ONE EIGHTH BLOCK, chosen because it looks like a space and never occurs in
text), and from then on the text is one stream of symbols like any other. The
same model handles a language with spaces and one without, and decoding is a
string operation with nothing to guess: join the pieces, turn every ``▁`` back
into a space.

The normalisation, and what it loses
------------------------------------
Every run of whitespace becomes one marker and one marker is prepended, so
``hello  world`` becomes ``▁hello▁world``. Then, because the framework's default
``split_by_whitespace`` is true, the marked text is cut at the markers so that
each unit is ``▁word`` with the marker attached to the front, and pieces never
cross a unit. The two steps together are ``text.split()`` with a marker in
front of each word, and that is how they are implemented; a marker of several
characters is still one symbol.

Three things are lost, all of them by SentencePiece's own default
``remove_extra_whitespaces``, and none hidden: a run of spaces decodes as one,
leading and trailing whitespace disappears, and a literal marker in the source
text decodes as a space. The spec pins the first as a documented loss and the
round trip on ordinary text.

Where the boundary is marked, and why the arithmetic does not care
------------------------------------------------------------------
Both the byte pair tokenizer,
:class:`~oop_ml.core.natural_language_processing.tokenization.subword.byte_pair_encoding.BytePairEncoding`,
and the unigram tokenizer,
:class:`~oop_ml.core.natural_language_processing.tokenization.subword.unigram.UnigramLanguageModel`,
mark the *end* of a word, ``l o w</w>``. This framework marks the *start*,
``▁ l o w``. The merge arithmetic in
:mod:`~oop_ml.core.natural_language_processing.tokenization.subword.merging` and the
lattice arithmetic on
:class:`~oop_ml.core.natural_language_processing.tokenization.subword.unigram.PieceTable`
both take words already spelled in symbols with counts, and neither knows what
a symbol is, so this class spells its units its own way and hands them to the
same two functions the marked-at-the-end tokenizers use. Nothing about
seeding, expectation-maximisation, pruning, Viterbi or sampling is written
twice; the design decision recorded in the unigram module is what lets this
class add nothing but the spelling and the decoding.

Under ``BYTE_PAIR`` there is no end-of-word marker at all. The ``▁`` at the
front of a unit is the boundary, and a merge that reaches it is a merge that
knows it is at a word start, which is the same information the ``</w>`` carries
from the other side. Under ``UNIGRAM`` the unit's symbols go to
:func:`~oop_ml.core.natural_language_processing.tokenization.subword.unigram.learn_piece_table`
unchanged, and encoding is Viterbi under ``PREDICTING`` and a sampled spelling
under ``TRAINING`` exactly as in the unigram tokenizer. Byte pair encoding here
is deterministic under both purposes: merge dropout belongs to the byte pair
class and is not a SentencePiece option.

Worked, on Sennrich's corpus
----------------------------
The alphabet is the ten characters plus the marker, eleven symbols, so the
smallest vocabulary is twelve. Under ``UNIGRAM`` at thirty tokens every training
word is one piece, ``▁low`` ``▁lower`` ``▁newest`` ``▁widest``, and the unseen
``lowest`` is spelled ``▁low`` ``est``: the marker rides on the first piece and
the word end needs no marker at all. Under ``BYTE_PAIR`` the first merges are
``e s`` at 9, ``es t`` at 9, ``l o`` at 7, ``lo w`` at 7 and ``▁ low`` at 7 --
the marker joins a word only once the word has assembled itself -- and
``lowest`` comes out ``▁low`` ``est`` there too. So the two algorithms agree on
that word while holding different vocabularies: asked for thirty, the byte pair
fit stops at twenty-seven when no pair reaches a count of two, and the two share
twenty-three tokens, seven being the unigram model's alone and four the byte
pair model's. All of it is pinned.

Vocabulary order
----------------
The unknown token, the alphabet in codepoint order with the marker among it,
then the merges in learned order or the pieces by falling probability.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import InvalidValuesError, VocabularyTooSmallError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.merges import Merges
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
    alphabet_of,
    apply_merges,
    learn_merges,
)
from oop_ml.core.natural_language_processing.tokenization.subword.unigram import (
    PieceTable,
    Segmentation,
    checked_word,
    learn_piece_table,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    LearnedTokenizer,
    checked_text,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.network.purpose import PassPurpose

WHITESPACE_MARKER = "▁"
"""``▁``, LOWER ONE EIGHTH BLOCK: SentencePiece's stand-in for a space."""

WHITESPACE = WhitespacePreTokenizer()
"""The one splitter the normalisation uses: every whitespace run ends a unit."""


class SubwordAlgorithm(StrEnum):
    """Which of the two algorithms the framework wraps."""

    BYTE_PAIR = "byte_pair"
    """Greedy most-frequent-pair merging, Sennrich et al. (2016)."""

    UNIGRAM = "unigram"
    """A probabilistic model over pieces shrunk by likelihood loss, Kudo (2018)."""


class SentencePiece(LearnedTokenizer):
    """Kudo and Richardson's framework: raw text in, whitespace as a symbol.

    There is no ``pre_tokenizer`` field, on purpose; passing one is refused at
    construction, since the point of the framework is that the text needs no
    language-specific splitting first.

    Parameters
    ----------
    vocabulary_size:
        How many tokens to learn, counting the unknown token and every symbol
        of the alphabet, the marker among them.
    algorithm:
        Which learner runs over the marked units.
    unknown_token:
        Stands in for any symbol the corpus never used.
    whitespace_marker:
        The symbol every whitespace run becomes. One symbol whatever its
        length.
    minimum_pair_frequency:
        Under ``BYTE_PAIR``, a pair seen fewer times than this is never merged.
    seed_size, max_piece_length, shrinking_factor,
    n_expectation_maximisation_rounds, sampling_temperature:
        Under ``UNIGRAM``, passed through to the unigram learner unchanged; see
        :class:`~oop_ml.core.natural_language_processing.tokenization.subword.unigram.UnigramLanguageModel`.
    random_seed:
        Seeds the training-time draws of the unigram algorithm.
    """

    vocabulary_size: int = Field(ge=2)
    algorithm: SubwordAlgorithm = SubwordAlgorithm.UNIGRAM
    unknown_token: str = Field(default="[UNK]", min_length=1)
    whitespace_marker: str = Field(default=WHITESPACE_MARKER, min_length=1)
    minimum_pair_frequency: int = Field(default=2, ge=1)
    seed_size: int = Field(default=1000, ge=1)
    max_piece_length: int = Field(default=16, ge=1)
    shrinking_factor: float = Field(default=0.75, gt=0.0, lt=1.0)
    n_expectation_maximisation_rounds: int = Field(default=2, ge=1)
    sampling_temperature: float = Field(default=1.0, gt=0.0)
    random_seed: int | None = None

    _vocabulary: Vocabulary = PrivateAttr()
    _merges: Merges | None = PrivateAttr()
    _piece_table: PieceTable | None = PrivateAttr()
    _n_pruning_rounds: int = PrivateAttr()
    _generator: random.Random = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn a vocabulary over the marked units of ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty or blank.
        VocabularyTooSmallError
            If ``vocabulary_size`` is below the alphabet plus the unknown token.
        NonUniqueTokensError
            If the unknown token collides with a symbol or a piece.
        """
        counts = Corpus.of(corpus).word_counts(WHITESPACE)
        spelled = [
            SpelledWord(self._symbols_of(word_count.word), word_count.count)
            for word_count in counts
        ]
        alphabet = alphabet_of(spelled)

        smallest_possible = len(alphabet) + 1
        if self.vocabulary_size < smallest_possible:
            raise VocabularyTooSmallError(
                f"vocabulary_size={self.vocabulary_size} cannot hold the "
                f"{len(alphabet)} symbols this corpus is spelled in, the whitespace "
                f"marker among them, plus the unknown token; the smallest workable "
                f"size is {smallest_possible}"
            )

        merges: Merges | None = None
        piece_table: PieceTable | None = None
        n_pruning_rounds = 0
        if self.algorithm is SubwordAlgorithm.BYTE_PAIR:
            merges = learn_merges(
                spelled,
                n_merges=self.vocabulary_size - smallest_possible,
                minimum_pair_frequency=self.minimum_pair_frequency,
            )
            learned_tokens = [merge.merged for merge in merges]
        else:
            learned = learn_piece_table(
                spelled,
                n_pieces=self.vocabulary_size - 1,
                seed_size=self.seed_size,
                max_piece_length=self.max_piece_length,
                shrinking_factor=self.shrinking_factor,
                n_expectation_maximisation_rounds=(
                    self.n_expectation_maximisation_rounds
                ),
            )
            piece_table = learned.table
            n_pruning_rounds = learned.n_pruning_rounds
            alphabet_set = set(alphabet)
            learned_tokens = [
                piece
                for piece in piece_table.pieces_by_probability
                if piece not in alphabet_set
            ]

        self._vocabulary = Vocabulary(
            [self.unknown_token, *alphabet, *learned_tokens],
            unknown_token=self.unknown_token,
        )
        self._merges = merges
        self._piece_table = piece_table
        self._n_pruning_rounds = n_pruning_rounds
        self._generator = random.Random(self.random_seed)
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        """The unknown token, the alphabet with the marker among it, then what
        the algorithm learned.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def merges(self) -> Merges | None:
        """The learned merges under ``BYTE_PAIR``; ``None`` under ``UNIGRAM``,
        which has nothing of the kind.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._merges

    @property
    def piece_table(self) -> PieceTable | None:
        """The learned piece table under ``UNIGRAM``; ``None`` under
        ``BYTE_PAIR``, which has nothing of the kind.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._piece_table

    @property
    def n_pruning_rounds(self) -> int:
        """How many times the unigram table was shrunk; zero under ``BYTE_PAIR``.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._n_pruning_rounds

    def units_of(self, text: str) -> tuple[str, ...]:
        """The marked units ``text`` is cut into: one ``▁word`` per whitespace-
        delimited word, every whitespace run collapsed.

        Raises
        ------
        InvalidValuesError
            If ``text`` is not a string.
        """
        return tuple(
            self.whitespace_marker + word
            for word in WHITESPACE.split(checked_text(text)).texts
        )

    def best_segmentation(self, word: str) -> Segmentation:
        """Under ``UNIGRAM``, the most probable spelling of one marked word.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``word`` is not a string, or the algorithm is ``BYTE_PAIR``,
            which has no probabilities to rank spellings by.
        EmptyValuesError
            If the word is empty.
        """
        self._check_fitted()
        if self._piece_table is None:
            raise InvalidValuesError(
                "best_segmentation needs a piece table, and the byte pair "
                "algorithm learns merges instead"
            )
        return self._piece_table.best_segmentation(self._symbols_of(checked_word(word)))

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        pieces: list[str] = []
        for unit in self.units_of(text):
            symbols = tuple(self._symbols_of_unit(unit))
            if self._merges is not None:
                pieces.extend(apply_merges(symbols, self._merges))
            elif self._piece_table is not None:
                if purpose is PassPurpose.TRAINING:
                    segmentation = self._piece_table.sample_segmentation(
                        symbols, self.sampling_temperature, self._generator
                    )
                else:
                    segmentation = self._piece_table.best_segmentation(symbols)
                pieces.extend(segmentation.pieces)
        return tuple(pieces)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces).replace(self.whitespace_marker, " ").removeprefix(" ")

    def _symbols_of(self, word: str) -> tuple[str, ...]:
        """A word as its characters with the marker in front as one symbol."""
        return (self.whitespace_marker, *word)

    def _symbols_of_unit(self, unit: str) -> tuple[str, ...]:
        """A marked unit back into symbols: the marker, then the characters."""
        return (self.whitespace_marker, *unit[len(self.whitespace_marker) :])
