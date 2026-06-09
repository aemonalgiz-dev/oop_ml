"""Byte-level byte pair encoding: GPT-2's alphabet of 256, so nothing is unknown.

The problem with characters
---------------------------
A subword vocabulary built over characters is only as complete as its corpus.
Every character the training texts never used -- an accented letter, a CJK
ideograph, an emoji -- has no symbol, so every word holding one is unknown in
whole or in part, and the model reads nothing where it was. Unicode assigns
about 150,000 codepoints, and no corpus uses them all.

Radford et al. (2019) took the alphabet down a level. Every string is some
sequence of UTF-8 bytes, there are exactly 256 byte values, and so an alphabet
of the 256 bytes spells every string there is, seen or unseen. The merges are
then learned over byte sequences exactly as Sennrich learns them over
characters: a frequent word becomes one token in a few merges, and an unseen
character falls back to its bytes rather than to nothing. ``日`` is three bytes;
on a corpus of English it encodes to three tokens and decodes back to ``日``.
The price is that a model choosing ids one at a time can cut a multi-byte
character *between* its bytes, which is why :func:`text_of_symbols` decodes
invalid UTF-8 to the replacement character rather than raising.

A closed vocabulary
-------------------
The alphabet is all 256 byte symbols whether or not the corpus used them, so
the vocabulary is closed by construction and is built with no unknown token:
there is nothing such a token could stand for. A byte-level tokenizer that ever
reached one would be a tokenizer whose byte mapping had gone wrong, and the
:class:`~oop_ml.core.natural_language_processing.tokenization.vocabulary.Vocabulary`
refuses the lookup rather than hiding that. It also means ``vocabulary_size``
can never be below 256, which ``fit`` refuses with the same error every subword
trainer here raises for a size below its alphabet.

The alphabet is laid out in byte order rather than codepoint order, so that the
id of a byte symbol *is* its byte value: token 0 names byte 0, token 32 names
the space, token 255 names byte 255, and the merged tokens follow from 256.

Where the space goes
--------------------
Sennrich marks the end of a word with ``</w>``. GPT-2 marks the start of one
with the space that preceded it, kept as part of the word, so ``hello world``
is the two byte sequences ``hello`` and ``Ġworld``, ``Ġ`` being the name of
byte 32. A word and its space then merge into one token, and ``Ġworld`` and
``world`` (at the start of a text) are different tokens, which is correct,
because they are different in the text. Decoding is exact with no marker to
remove, since the spaces are in the bytes.

Here the text attached to a word is everything between the end of the previous
word and the start of this one, read off the pre-tokenizer's offsets, which
under the whitespace pre-tokenizer is exactly the whitespace: two spaces or a
newline before a word are kept in it. Whitespace *after* the last word of a
text has no word to attach to and is dropped, so ``decode(encode("hello "))``
is ``"hello"``. GPT-2's regular expression splits a run of several spaces into
tokens of its own; that is a pre-tokenizer's decision, and this class takes
whichever pre-tokenizer it is given.

Worked, on Sennrich's toy corpus as one text
--------------------------------------------
``low low low low low lower lower newest ... widest ...``. The first ``low``
opens the text and has no space; the other four are ``Ġlow``, so where byte
pair encoding sees one word five times this sees ``low`` once and ``Ġlow``
four times. ASCII letters name themselves, so the within-word pair counts are
Sennrich's and the first two merges are his: ``e s`` at 9 and ``es t`` at 9.
The space then behaves as one more symbol: ``Ġ`` followed by ``low`` occurs
6 times, in ``Ġlow`` and ``Ġlower``, and the eighth merge joins them into a
token that begins with a space. Ten merges in, ``low lowest`` encodes to ``low``,
``Ġlow``, ``est``: the second word carries its space and the first does not.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import EmptyValuesError, VocabularyTooSmallError
from oop_ml.core.natural_language_processing.tokenization.characters.byte_symbols import (  # noqa: E501
    BYTE_SYMBOLS,
    symbols_of_text,
    text_of_symbols,
)
from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.subword.merges import Merges
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    SpelledWord,
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

N_BYTE_VALUES = len(BYTE_SYMBOLS)
"""The alphabet's size, 256, and the smallest vocabulary that can exist."""


class ByteLevelBytePairEncoding(LearnedTokenizer):
    """Byte pair encoding over UTF-8 bytes, with the space kept on the word it precedes.

    Parameters
    ----------
    vocabulary_size:
        How many tokens to learn, counting the 256 byte symbols. Must be at
        least 256; the fit stops early if no pair reaches
        ``minimum_pair_frequency`` first.
    pre_tokenizer:
        Decides where the words are. Merges never cross a word boundary, and
        the text between two words is attached to the second.
    minimum_pair_frequency:
        A pair seen fewer times than this is never merged.
    merge_dropout:
        Probability of skipping each applicable merge while encoding under
        ``TRAINING``. Zero, the default, is ordinary merging.
    random_seed:
        Seeds the dropout draws.
    """

    vocabulary_size: int = Field(ge=1)
    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    minimum_pair_frequency: int = Field(default=2, ge=1)
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
            If ``vocabulary_size`` is below 256, the size of the byte alphabet.
        """
        checked = Corpus.of(corpus)
        if self.vocabulary_size < N_BYTE_VALUES:
            raise VocabularyTooSmallError(
                f"vocabulary_size={self.vocabulary_size} cannot hold the "
                f"{N_BYTE_VALUES} byte symbols every text is spelled in; the smallest "
                f"workable size is {N_BYTE_VALUES}"
            )

        counter: Counter[str] = Counter()
        for text in checked:
            counter.update(self._spaced_words_of(text))
        if not counter:
            raise EmptyValuesError(
                f"{type(self.pre_tokenizer).__name__} found no words in any of the "
                f"{checked.n_texts} texts"
            )

        spelled = [
            SpelledWord(symbols_of_text(spaced_word), count)
            for spaced_word, count in counter.items()
        ]
        merges = learn_merges(
            spelled,
            n_merges=self.vocabulary_size - N_BYTE_VALUES,
            minimum_pair_frequency=self.minimum_pair_frequency,
        )
        tokens = [*BYTE_SYMBOLS, *(merge.merged for merge in merges)]

        self._vocabulary = Vocabulary(tokens, unknown_token=None)
        self._merges = merges
        self._generator = random.Random(self.random_seed)
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        """The 256 byte symbols in byte order, then the merges. No unknown token.

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
        for spaced_word in self._spaced_words_of(text):
            pieces.extend(
                apply_merges(
                    symbols_of_text(spaced_word), self._merges, dropout, self._generator
                )
            )
        return tuple(pieces)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return text_of_symbols("".join(pieces))

    def _spaced_words_of(self, text: str) -> tuple[str, ...]:
        """Each word with the text between it and the previous word attached in front.

        Whatever follows the last word is attached to nothing and is dropped.
        """
        spaced: list[str] = []
        previous_end = 0
        for word in self.pre_tokenizer.split(text):
            spaced.append(text[previous_end : word.start] + word.text)
            previous_end = word.end
        return tuple(spaced)
