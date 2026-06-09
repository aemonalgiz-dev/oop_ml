"""Character-level tokenization: every character the corpus used is a token.

The smallest vocabulary that still spells the corpus
----------------------------------------------------
A word-level vocabulary runs to hundreds of thousands of rows and still meets
words it has never seen. A subword vocabulary sits between the two by learning
which pieces to keep. This tokenizer takes the other end of that trade: the
units are the characters themselves, so the vocabulary is as small as it can be
while spelling every seen text exactly, and a text becomes one token per
character. Sutskever, Martens and Hinton (2011) and Graves (2013) generated text
this way; Kim et al. (2016) read words through it. It is the baseline the
subword family is measured against, and the one that cannot fail to spell a
word made of familiar letters, since ``lowest`` is six known tokens whether or
not the corpus ever held it.

What a "character" is here
--------------------------
A Unicode codepoint, which is what iterating a Python string yields. A
combining sequence such as ``e`` followed by ``U+0301`` is two characters, and
an emoji written with a skin-tone modifier is two as well; a grapheme-cluster
rule would need the Unicode segmentation tables, and this library has none.
Every character of every text counts, spaces and punctuation included, so that
``decode(encode(text))`` reproduces a seen text exactly. Dropping whitespace
from the vocabulary would make the tokenizer unable to say where the words
were, which the byte pair encoder avoids with an end-of-word marker and this
one avoids by simply keeping the space.

The vocabulary is the unknown token first, then the characters in codepoint
order. Codepoint order is a fact about Unicode rather than about the corpus,
so the same set of characters gives the same vocabulary whatever order the
texts arrived in. A character the corpus never held encodes to the unknown
token, and decodes to that token's spelling, since nothing about what stood
there survives.

Why there is no codepoint tokenizer
-----------------------------------
The obvious open-vocabulary character tokenizer needs no fitting at all:
``id = ord(character)``, every string ever written already encoded. It is not
here, because a
:class:`~oop_ml.core.natural_language_processing.tokenization.vocabulary.Vocabulary`
is a materialised table whose width becomes the width of an embedding table,
and Unicode has 1,114,112 codepoints. At a width of 768 that is 855,638,016
embedding parameters, roughly 7.8 times the whole of BERT-base, for a table
almost every row of which no text will ever touch. CANINE's answer is to hash
the codepoint into a fixed number of buckets instead, 16,384 of them at the
same width for 12,582,912 parameters, sixty-eight times fewer, which is
:mod:`oop_ml.core.natural_language_processing.tokenization.hashing.hashed_characters`.
So the open-vocabulary character case is the hashing one, and this class is
the closed one: the characters a corpus taught, and a token for everything
else.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from pydantic import Field, PrivateAttr

from oop_ml.core.natural_language_processing.tokenization.corpus import Corpus
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    LearnedTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.network.purpose import PassPurpose


class CharacterTokenizer(LearnedTokenizer):
    """One token per character, over the characters a corpus contained.

    Parameters
    ----------
    unknown_token:
        Stands in for any character the corpus never used.
    """

    unknown_token: str = Field(default="[UNK]", min_length=1)

    _vocabulary: Vocabulary = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the set of characters in ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty or every text is blank.
        NonUniqueTokensError
            If the unknown token is itself a single character the corpus used.
        """
        characters = sorted(
            {character for text in Corpus.of(corpus) for character in text}
        )
        vocabulary = Vocabulary(
            [self.unknown_token, *characters], unknown_token=self.unknown_token
        )

        self._vocabulary = vocabulary
        self._mark_fitted()
        return self

    @property
    def vocabulary(self) -> Vocabulary:
        """The unknown token, then every character seen, in codepoint order.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def n_characters(self) -> int:
        """How many distinct characters the corpus held. See :attr:`vocabulary`."""
        return self.vocabulary.n_tokens - 1

    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        self._check_fitted()
        return tuple(text)

    def _text_from(self, pieces: Sequence[str]) -> str:
        return "".join(pieces)
