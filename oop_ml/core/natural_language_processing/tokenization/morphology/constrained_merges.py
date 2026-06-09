"""Byte pair encoding whose merges never cross a morpheme boundary.

What plain byte pair encoding gets wrong about morphology
---------------------------------------------------------
Byte pair encoding merges whichever adjacent pair is most frequent, and
frequency has no opinion about where a morpheme ends. On a corpus of ``undo``,
``untie``, ``redo`` and ``retie``, once ``un`` and ``do</w>`` have each become a
symbol, the pair ``un`` + ``do</w>`` is as frequent as any other and merges into
``undo</w>``, a piece spanning the prefix and the root. A model reading that
piece cannot see that ``undo`` shares ``un`` with ``untie`` and ``do`` with
``redo``; the morphology the corpus plainly exhibits has been merged away.
Measured on that corpus with each word ten times and a vocabulary of thirty
asked for, plain byte pair encoding learns nine merges: five at count 20 that
build ``do</w>``, ``ie</w>``, ``re``, ``tie</w>`` and ``un``, then four at
count 10 that join them into ``redo</w>``, ``retie</w>``, ``undo</w>`` and
``untie</w>``, so its nineteen-token vocabulary holds every word whole.

The constraint, and why the shared loop did not change
------------------------------------------------------
Morphologically-informed byte pair encoding -- MorphPiece (Jabbar, 2023) is
the recent name for the idea -- learns the same greedy merges but forbids any
merge that would join two morphs. Here that is not a rule added to the loop; it
is a fact about how the words are spelled before the loop starts. Each word is
cut into morphs by ``morph_splitter``, and each *morph* is handed to
:func:`~oop_ml.core.natural_language_processing.tokenization.subword.merging.learn_merges`
as its own
:class:`~oop_ml.core.natural_language_processing.tokenization.subword.merging.SpelledWord`.
The pair counter only ever looks at symbols adjacent *inside one spelled word*,
so ``n`` at the end of ``un`` and ``d`` at the start of ``do`` are never
adjacent to it, and no merge across the boundary can be proposed, let alone
chosen. The loop is byte pair encoding's own, character for character; only
the spelling changed. On the same corpus and size the constrained fit learns
the same first five merges and then *stops*, at fifteen tokens, because every
morph is by then a single symbol and no pair is left anywhere: its vocabulary
is exactly the first fifteen entries of the plain one, and every piece in it
is a substring of one morph.

Only the word's last morph carries the end-of-word marker, on its last
character, because the marker means "a space follows" and a space follows the
word, not each morph. So ``un`` inside ``undo`` is the same symbol as ``un``
inside ``untie``, and neither is ``un</w>``. Decoding is byte pair encoding's:
join the pieces, turn each marker into a space.

What the splitter is
--------------------
Any
:class:`~oop_ml.core.natural_language_processing.tokenization.tokenizer.PreTokenizer`,
applied to one word at a time. A pattern rule, a lookup table, or a
:class:`~oop_ml.core.natural_language_processing.tokenization.morphology.finite_state.FiniteStateAnalyzer`,
which is a pre-tokenizer precisely so that it can stand here. A learned
segmenter such as
:class:`~oop_ml.core.natural_language_processing.tokenization.morphology.morfessor.MorfessorBaseline`
is a tokenizer rather than a pre-tokenizer, and is adapted in a three-line
subclass that wraps its ``best_segmentation`` into ``Words``. A splitter that
finds no morph in a word -- a pattern that matches nothing in it -- leaves the
word whole, since a word cut into nothing cannot be spelled. And if the
splitter rewrites what it cuts, dropping a hyphen say, decoding returns what
the splitter kept, as it does for every normalising pre-tokenizer in this
package.

With a splitter that leaves every word whole, this tokenizer *is* byte pair
encoding: same merges, same vocabulary, to the last symbol. The spec holds it
to that.
"""

from __future__ import annotations

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
    checked_text,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from oop_ml.core.network.purpose import PassPurpose


class MorphemeConstrainedBytePairEncoding(LearnedTokenizer):
    """Byte pair encoding that merges inside morphs and never across them.

    Parameters
    ----------
    vocabulary_size:
        How many tokens to learn, counting the unknown token and every symbol
        of the alphabet. The fit stops early if no pair reaches
        ``minimum_pair_frequency`` first.
    morph_splitter:
        Cuts one word into its morphs. Merges never cross a cut it makes.
    pre_tokenizer:
        Decides where the words are before any splitting into morphs.
    minimum_pair_frequency:
        A pair seen fewer times than this is never merged.
    end_of_word_marker:
        Appended to the last character of each word's last morph, so decoding
        knows where the spaces go.
    unknown_token:
        Stands in for any symbol the corpus never used.
    """

    vocabulary_size: int = Field(ge=2)
    morph_splitter: PreTokenizer
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    minimum_pair_frequency: int = Field(default=2, ge=1)
    end_of_word_marker: str = Field(default="</w>", min_length=1)
    unknown_token: str = Field(default="[UNK]", min_length=1)

    _vocabulary: Vocabulary = PrivateAttr()
    _merges: Merges = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the merges from ``corpus``, one spelled word per morph.

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
            SpelledWord(symbols, word_count.count)
            for word_count in counts
            for symbols in self._spelled_morphs_of(word_count.word)
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

    def morphs_of(self, word: str) -> tuple[str, ...]:
        """The morphs ``morph_splitter`` cuts one word into, in order.

        The word whole if the splitter finds no morph in it. Needs no fit,
        since the splitter is configuration.

        Raises
        ------
        InvalidValuesError
            If ``word`` is not a string.
        """
        morphs = self.morph_splitter.split(checked_text(word)).texts
        return morphs if morphs else (word,)

    def _spelled_morphs_of(self, word: str) -> tuple[tuple[str, ...], ...]:
        """Each morph as its characters; the last morph's last one is marked."""
        morphs = self.morphs_of(word)
        spelled: list[tuple[str, ...]] = []
        for position, morph in enumerate(morphs):
            if position == len(morphs) - 1:
                spelled.append((*morph[:-1], morph[-1] + self.end_of_word_marker))
            else:
                spelled.append(tuple(morph))
        return tuple(spelled)

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        pieces: list[str] = []
        for word in self.pre_tokenizer.split(text).texts:
            for symbols in self._spelled_morphs_of(word):
                pieces.extend(apply_merges(symbols, self._merges))
        return tuple(pieces)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces).replace(self.end_of_word_marker, " ").rstrip(" ")
