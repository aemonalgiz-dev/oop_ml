"""Byte pair encoding: a vocabulary grown one most-frequent pair at a time.

The problem it solves
---------------------
A word-level vocabulary has to choose between two bad options. Keep every word
the corpus contains and the table runs to hundreds of thousands of rows, most of
them seen once; cap it at forty thousand and every rarer word -- every name,
every number, every inflection the cap missed -- collapses to one unknown token
and the model reads nothing. A character-level vocabulary has neither problem
and a worse one: a sentence becomes hundreds of tokens, each carrying almost no
meaning on its own.

Sennrich, Haddow and Birch (2016) borrowed a compression trick to sit between
the two. Start from characters, so that every word is spellable. Then repeat:
count how often each pair of adjacent symbols occurs across the corpus, and
replace the most frequent pair with a single new symbol. Every merge adds one
token to the vocabulary, so the loop runs until the vocabulary is the size the
caller asked for. Frequent words end up as one symbol; rare ones stay as a few
pieces that are themselves frequent. Nothing is ever unknown that is spelled in
characters the corpus used.

Worked, on Sennrich's own toy corpus
------------------------------------
``low`` five times, ``lower`` twice, ``newest`` six times, ``widest`` three.
Each word is spelled in characters with an end-of-word marker on the last one,
so ``low`` is ``l o w</w>``, and the pair counts are weighted by how often the
word occurs. The first few merges, with the count that chose each::

    e s        9    from newest (6) and widest (3)
    es t</w>   9
    l o        7    from low (5) and lower (2)
    e w        6
    ew est</w> 6
    n ewest</w> 6
    lo w</w>   5

At the end of ten merges, ``lowest``, which the corpus never contained, encodes
as ``lo``, ``w``, ``est</w>``: three pieces, every one of them a symbol the
corpus taught, and ``est</w>`` carrying the fact that a word ended there.

Why the marker sits on the last character
-----------------------------------------
``w</w>`` and ``w`` are different symbols from the first step, so a merge that
ends a word and a merge that continues one are counted apart -- ``est</w>`` in
``newest`` is a different piece from ``est`` in ``estimate``, which is correct,
since they behave differently. It also makes decoding a string operation: join
the pieces and replace every marker with a space. Without a marker ``lo w est``
could be one word or three.

The tie rule, written down once
-------------------------------
Pair counts tie constantly on a small corpus (``e s`` and ``es t</w>`` both
score 9 above), and which pair wins changes every later merge. The reference
implementation takes whichever pair its dictionary happened to hold first,
which is to say insertion order, which is to say the order words were met in.
Here the tie goes to the lexicographically smaller ``(left, right)`` pair, so
that the same corpus gives the same vocabulary whatever order its texts arrive
in. Arbitrary, and stated, like :meth:`~oop_ml.core.tree.split.Split.beats`.

Merge dropout
-------------
Provilkov, Emelianenko and Voita (2020) noticed that a model only ever sees one
spelling of each word, the best one, and so never learns that ``est</w>`` and
``e s t</w>`` mean the same thing. Under ``TRAINING`` this tokenizer skips each
applicable merge with probability ``merge_dropout`` as it encodes, so the model
meets many spellings of one word; under ``PREDICTING`` it applies every merge
and the spelling is the one the model should expect. The default purpose is
``PREDICTING`` for the reason
:mod:`oop_ml.core.network.purpose` gives: the mistake that costs every answer
is the one the default must prevent.

What decoding cannot recover
----------------------------
A word whose last symbol was never in the corpus encodes to the unknown token
without its end-of-word marker, so decoding runs it into the word that follows.
That is a fact about any vocabulary with an unknown token: the token stands for
"something was here", and nothing about what.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import VocabularyTooSmallError
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.merges import Merges
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
    alphabet_of,
    apply_merges,
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


class BytePairEncoding(LearnedTokenizer):
    """Subword tokenization by greedy most-frequent-pair merging.

    Parameters
    ----------
    vocabulary_size:
        How many tokens to learn, counting the unknown token and every symbol
        of the alphabet. The fit stops early if no pair reaches
        ``minimum_pair_frequency`` first, and ``vocabulary.n_tokens`` says how
        many it actually learned.
    pre_tokenizer:
        Decides where the words are before any merging. Merges never cross a
        word boundary.
    minimum_pair_frequency:
        A pair seen fewer times than this is never merged. Sennrich's
        ``--min-frequency``, at his default of 2: a pair seen once is a fact
        about one word, not about the language.
    end_of_word_marker:
        Appended to each word's last character before merging, so pieces that
        end a word are distinct from pieces that continue one and decoding
        knows where the spaces go.
    unknown_token:
        Stands in for any symbol the corpus never used.
    merge_dropout:
        Probability of skipping each applicable merge while encoding under
        ``TRAINING``. Zero, the default, is ordinary byte pair encoding.
    random_seed:
        Seeds the dropout draws, so a seeded tokenizer produces the same
        sequence of spellings for the same sequence of calls.
    """

    vocabulary_size: int = Field(ge=2)
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    minimum_pair_frequency: int = Field(default=2, ge=1)
    end_of_word_marker: str = Field(default="</w>", min_length=1)
    unknown_token: str = Field(default="[UNK]", min_length=1)
    merge_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    random_seed: int | None = None

    _vocabulary: Vocabulary = PrivateAttr()
    _merges: Merges = PrivateAttr()
    _generator: random.Random = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the merges from ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        VocabularyTooSmallError
            If ``vocabulary_size`` is below the alphabet plus the unknown token.
        NonUniqueTokensError
            If the unknown token collides with a symbol of the alphabet.
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

        merges = learn_merges(
            spelled,
            n_merges=self.vocabulary_size - smallest_possible,
            minimum_pair_frequency=self.minimum_pair_frequency,
        )
        tokens = [self.unknown_token, *alphabet, *(merge.merged for merge in merges)]

        self._vocabulary = Vocabulary(tokens, unknown_token=self.unknown_token)
        self._merges = merges
        self._generator = random.Random(self.random_seed)
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        """The unknown token, the alphabet in codepoint order, then the merges.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def merges(self) -> Merges:
        """Every merge learned, in the order it was learned.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._merges

    @property
    def n_merges(self) -> int:
        """How many merges the fit learned. See :attr:`merges`."""
        return self.merges.n_merges

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        dropout = self.merge_dropout if purpose is PassPurpose.TRAINING else 0.0

        pieces: list[str] = []
        for word in self.pre_tokenizer.split(text).texts:
            pieces.extend(
                apply_merges(
                    self._symbols_of(word), self._merges, dropout, self._generator
                )
            )
        return tuple(pieces)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces).replace(self.end_of_word_marker, " ").rstrip(" ")

    def _symbols_of(self, word: str) -> tuple[str, ...]:
        """A word as characters, the last one carrying the end-of-word marker."""
        return (*word[:-1], word[-1] + self.end_of_word_marker)
