"""FastText: a word is the sum of its pieces, so a word never seen still has a vector.

The problem with a row per word
-------------------------------
Word2vec gives every word one row and nothing else, and two facts are lost at
once. ``played`` and ``playing`` share nothing, though they are one verb, and
must learn their nearness from the corpus alone, which a rare inflection of a
rare verb never manages. And a word absent from the corpus has no row at all,
so word2vec's answer to ``playing``, if only ``played`` was seen, is a refusal.
Bojanowski, Grave, Joulin and Mikolov (2017) repaired both with one change: a
word's vector is its own row plus one row for each of its character n-grams.
``played`` and ``playing`` then share the rows for ``<pl``, ``pla``, ``lay`` and
every other piece they have in common, and a word never seen is the sum of the
rows for its pieces, each of which some seen word may have trained.

The pieces
----------
A word ``w`` is wrapped as ``<w>``, so that a prefix and a suffix are told
apart from the same letters inside a word, and every character n-gram of the
wrapped form whose length lies in ``minimum_n_gram_length ..
maximum_n_gram_length`` is a piece. The paper's own example, ``where`` at
length 3, gives exactly five: ``<wh``, ``whe``, ``her``, ``ere``, ``re>``. At
the default lengths 3 to 6 the same word has fourteen. The wrapped word itself
is never a piece here, whatever the lengths: the paper adds it as a special
sequence so that the word has a row of its own, and here the word row already
exists. The reference code does hash a short wrapped word as one more n-gram,
so for a word of fewer than ``maximum_n_gram_length - 1`` characters it has
one row more than this module; recorded as a difference rather than hidden. A
piece that occurs twice in one word, ``aaa`` in ``<aaaa>``, counts once,
because the paper's collection of n-grams is a set.

Hashing rather than a vocabulary of pieces
------------------------------------------
There are far more distinct n-grams than words, so the pieces are not given a
vocabulary. Each is hashed into one of ``n_buckets`` rows with the 32-bit
FNV-1a hash of its UTF-8 bytes, ``hash = (hash xor byte) * 16777619`` from an
offset of 2166136261, and two pieces that land in one bucket share a row. The
hash is the standard one, ``FNV-1a("") = 0x811c9dc5`` and ``FNV-1a("a") =
0xe40c292c``; the reference sign-extends bytes above 127 before the xor, so
its buckets for non-ASCII pieces differ from these and its buckets for ASCII
pieces agree. The paper uses two million buckets. The default here is two
thousand so that a test table stays small, and a vocabulary of a few dozen
words already occupies a fair share of them. Collisions are the price of a
fixed table: a rare piece sharing a row with a common one reads the common
one's direction.

The composition, and its gradient
---------------------------------
The input table has ``n_words + n_buckets`` rows, the word rows first in id
order and the buckets after. The vector read for a word is its row plus the
sum of its buckets' rows, which is the paper's composition. Because the
composition is a sum, the derivative of the loss with respect to each part is
the derivative with respect to the sum, so the whole step is added to the word
row and to every bucket row of the word, and that is the exact gradient rather
than a convention. The reference code averages the parts on the way forward
and still adds the whole step to each on the way back, a gradient scaled by
the number of parts; this module keeps the paper's sum and the exact step.
Cosine similarity ignores a vector's length, so for any one word the choice
between sum and mean is invisible to ``similarity`` and ``most_similar``.

The bucket rows start at zero where the word rows start at word2vec's small
uniform draw, for two reasons. A word's composed vector then begins exactly
where word2vec's would, so a model with no pieces at all -- a minimum length
longer than any wrapped word -- is word2vec to the last bit, and a test says
so. And a random start for a bucket carries nothing about the corpus: two
buckets that only ever appear together receive identical steps and would
differ forever by their initial offset alone.

A word never seen
-----------------
:meth:`FastText.vector_of_unseen` is the sum of the bucket rows for any word's
pieces, seen or not, and :meth:`FastText.vector_of` falls back to it for a
word outside the vocabulary, so this model has no unknown words, only unseen
ones. For a seen word ``vector_of_unseen`` deliberately omits the word row: it
answers what the word's vector would have been had the word never been seen,
which is how much of the vector is spelling. There is no
``most_similar_to_unseen``, because ``embeddings.similar_to_vector(
vector_of_unseen(word).values, n_results)`` is that already;
:meth:`FastText.most_similar` calls it for a word outside the vocabulary. A
word none of whose pieces shares a bucket with any seen word has the zero
vector, and asking what is near it raises
:class:`~oop_ml.core.exceptions.UndefinedMetricError` rather than answering
with a direction the model never learned.

The two states the hooks need
-----------------------------
Word2vec's loop hands its four hooks only word ids, so which bucket rows a
word owns must be known to the model before the loop starts. They are
computed once, from the vocabulary, inside :meth:`_initial_input_vectors`,
which is the hook that receives the vocabulary and the one the loop calls
first, and kept on a private attribute. That is the one place this model
assigns state before the end of ``fit``; every fitted property is still
guarded, so a fit that raises part-way leaves nothing readable.

What was measured
-----------------
On two hundred sentences drawn from two word lists, twenty verb forms and
eleven finance nouns, skip-gram with negative sampling at dimension 12, window
3, five epochs, seed 0: the vector of ``playing``, which the corpus never
holds, has a mean cosine of 0.997 to the verb forms that share its pieces
(the smallest 0.994) and 0.100 to the finance nouns that share none (the
largest 0.111), and its four nearest words are ``play``, ``plays``,
``player`` and ``played``. With the minimum length set to 50, so that no word
has a piece, the fitted table, the output table and every epoch's loss agree
with :class:`Word2Vec` under the same seed exactly, gap 0.0, under all four
architecture and objective pairs. At two thousand buckets the thirty-one
words of that corpus own 279 distinct pieces landing in 260 distinct
buckets, so nineteen pieces already share a row. ``zap`` shares no bucket
with any of them, so its vector is exactly zero and ``most_similar`` refuses
it.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.embeddings.prediction.word2vec import (
    Word2Vec,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    SimilarWords,
    WordVector,
    cosine_similarity,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray, IndexArray

FOWLER_NOLL_VO_OFFSET_BASIS = 0x811C9DC5
"""Where the 32-bit FNV-1a hash starts, which is also the hash of nothing."""

FOWLER_NOLL_VO_PRIME = 0x01000193
"""The 32-bit FNV prime, 16777619."""

WORD_START = "<"
"""Marks the start of a wrapped word, so a prefix piece is told from a middle one."""

WORD_END = ">"
"""Marks the end of a wrapped word, so a suffix piece is told from a middle one."""


def fowler_noll_vo_hash(text: str) -> int:
    """The 32-bit FNV-1a hash of ``text``'s UTF-8 bytes.

    Xor each byte in, then multiply by the prime, modulo ``2 ** 32``. Standard
    over every byte value; the reference fastText sign-extends bytes above
    127 first, which this deliberately does not reproduce.
    """
    hashed = FOWLER_NOLL_VO_OFFSET_BASIS
    for byte in text.encode("utf-8"):
        hashed ^= byte
        hashed = (hashed * FOWLER_NOLL_VO_PRIME) & 0xFFFFFFFF
    return hashed


def checked_word(word: object) -> str:
    """A non-empty string, or a refusal that says which rule it broke.

    Raises
    ------
    InvalidValuesError
        If ``word`` is not a string.
    EmptyValuesError
        If ``word`` is the empty string.
    """
    if not isinstance(word, str):
        raise InvalidValuesError(f"a word is a str, got {type(word).__name__}")
    if not word:
        raise EmptyValuesError("a word has at least one character")
    return word


class FastText(Word2Vec):
    """Word2vec whose word vectors are sums over hashed character n-grams.

    Every field of :class:`Word2Vec` applies unchanged. The three below decide
    what a word's pieces are and how many rows they share.

    Parameters
    ----------
    minimum_n_gram_length:
        The shortest piece taken from the wrapped word ``<w>``.
    maximum_n_gram_length:
        The longest. At least ``minimum_n_gram_length``.
    n_buckets:
        How many rows the pieces are hashed into. The paper's value is two
        million; the default is small so that a table stays small.
    """

    minimum_n_gram_length: int = Field(default=3, ge=1)
    maximum_n_gram_length: int = Field(default=6, ge=1)
    n_buckets: int = Field(default=2000, ge=1)

    _n_words: int = PrivateAttr()
    _piece_rows_by_word_id: tuple[IndexArray, ...] = PrivateAttr()
    _word_row_vectors: FloatArray = PrivateAttr()
    _bucket_vectors: FloatArray = PrivateAttr()

    @model_validator(mode="after")
    def _check_the_lengths_are_ordered(self) -> Self:
        if self.maximum_n_gram_length < self.minimum_n_gram_length:
            raise ValueError(
                f"maximum_n_gram_length={self.maximum_n_gram_length} is below "
                f"minimum_n_gram_length={self.minimum_n_gram_length}"
            )
        return self

    def n_grams_of(self, word: str) -> tuple[str, ...]:
        """The distinct pieces of ``<word>``, shortest first and left to right.

        Every character n-gram of the wrapped word whose length lies in
        ``minimum_n_gram_length .. maximum_n_gram_length``, except the wrapped
        word itself, each once in order of first appearance. A function of the
        configuration alone, so it can be asked before ``fit``.

        Raises
        ------
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If ``word`` is empty.
        """
        wrapped = WORD_START + checked_word(word) + WORD_END
        longest = min(self.maximum_n_gram_length, len(wrapped) - 1)
        pieces: dict[str, None] = {}
        for length in range(self.minimum_n_gram_length, longest + 1):
            for start in range(len(wrapped) - length + 1):
                pieces.setdefault(wrapped[start : start + length], None)
        return tuple(pieces)

    def n_gram_ids_of(self, word: str) -> tuple[int, ...]:
        """The bucket of each distinct piece of ``word``, sorted.

        One entry per distinct piece, so two pieces that hash to one bucket
        put that bucket in twice, which is how often its row enters the
        word's sum. Buckets index :attr:`bucket_vectors`, from zero.

        Raises
        ------
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If ``word`` is empty.
        """
        return tuple(
            sorted(
                fowler_noll_vo_hash(piece) % self.n_buckets
                for piece in self.n_grams_of(word)
            )
        )

    @property
    def word_row_vectors(self) -> FloatArray:
        """The ``(n_words, dimension)`` rows learned for the whole words, frozen.

        What a word contributes to its own vector beyond its pieces;
        ``embeddings.table`` is this plus each word's bucket rows summed.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._word_row_vectors

    @property
    def bucket_vectors(self) -> FloatArray:
        """The ``(n_buckets, dimension)`` rows learned for the pieces, frozen.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._bucket_vectors

    def vector_of_unseen(self, word: str) -> WordVector:
        """The sum of the bucket rows for ``word``'s pieces, seen or not.

        For a word outside the vocabulary this is its vector. For a word
        inside it this omits the word row, so it is what the vector would have
        been had the word never been seen. A word with no piece in a trained
        bucket gets the zero vector.

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
        buckets = np.asarray(self.n_gram_ids_of(word), dtype=np.intp)
        return WordVector(word, self._bucket_vectors[buckets].sum(axis=0))

    def vector_of(self, word: str) -> WordVector:
        """The vector for ``word``: its learned vector if seen, its pieces' if not.

        A FastText model has no unknown words, only unseen ones.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``word`` is not a string.
        EmptyValuesError
            If ``word`` is empty.
        """
        if word in self.embeddings:
            return self.embeddings.vector_of(word)
        return self.vector_of_unseen(word)

    def similarity(self, first_word: str, second_word: str) -> float:
        """Cosine similarity between two words, either or both unseen.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        UndefinedMetricError
            If either vector is zero.
        """
        return cosine_similarity(
            self.vector_of(first_word).values, self.vector_of(second_word).values
        )

    def most_similar(self, word: str, n_results: int = 10) -> SimilarWords:
        """The nearest vocabulary words to ``word``, which need not be seen.

        A seen word is excluded from its own answer, as in
        :meth:`~oop_ml.core.natural_language_processing.embeddings.vectors.WordEmbeddings.most_similar`;
        an unseen word is compared to every vocabulary word through
        :meth:`~oop_ml.core.natural_language_processing.embeddings.vectors.WordEmbeddings.similar_to_vector`.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        UndefinedMetricError
            If the word's vector is zero.
        InvalidValuesError
            If ``n_results`` is below one.
        """
        if word in self.embeddings:
            return self.embeddings.most_similar(word, n_results)
        return self.embeddings.similar_to_vector(
            self.vector_of_unseen(word).values, n_results
        )

    def _initial_input_vectors(
        self, vocabulary: Vocabulary, generator: np.random.Generator
    ) -> FloatArray:
        """Word2vec's word rows, then ``n_buckets`` rows of zeros.

        The word rows are drawn exactly as the parent draws them, and the
        buckets consume nothing from the generator, so a model with no pieces
        walks the parent's walk. This hook is also where each word's bucket
        rows are worked out and kept, because the hooks that follow receive
        only ids; see the module docstring.
        """
        word_rows = super()._initial_input_vectors(vocabulary, generator)
        table = np.zeros((vocabulary.n_tokens + self.n_buckets, self.dimension))
        table[: vocabulary.n_tokens] = word_rows

        self._n_words = vocabulary.n_tokens
        self._piece_rows_by_word_id = tuple(
            np.asarray(
                [vocabulary.n_tokens + bucket for bucket in self.n_gram_ids_of(word)],
                dtype=np.intp,
            )
            for word in vocabulary
        )
        return table

    def _input_vector(self, input_vectors: FloatArray, word_id: int) -> FloatArray:
        """The word's row plus the sum of its bucket rows, as a fresh array."""
        return input_vectors[word_id] + input_vectors[
            self._piece_rows_by_word_id[word_id]
        ].sum(axis=0)

    def _apply_input_gradient(
        self, input_vectors: FloatArray, word_id: int, step: FloatArray
    ) -> None:
        """Add the whole step to the word row and to every one of its bucket rows.

        ``np.add.at`` rather than fancy-index assignment, because a bucket that
        two of the word's pieces share appears twice and must move twice.
        """
        input_vectors[word_id] += step
        np.add.at(input_vectors, self._piece_rows_by_word_id[word_id], step)

    def _word_table(
        self, input_vectors: FloatArray, vocabulary: Vocabulary
    ) -> FloatArray:
        """Every word's composed vector, and the two frozen parts it is made of.

        The last hook the parent's ``fit`` calls, so this is where the word
        rows and the bucket rows are copied out and frozen for the properties.
        """
        word_row_vectors = input_vectors[: vocabulary.n_tokens].copy()
        word_row_vectors.setflags(write=False)
        bucket_vectors = input_vectors[vocabulary.n_tokens :].copy()
        bucket_vectors.setflags(write=False)

        self._word_row_vectors = word_row_vectors
        self._bucket_vectors = bucket_vectors
        return np.stack(
            [
                self._input_vector(input_vectors, word_id)
                for word_id in range(vocabulary.n_tokens)
            ]
        )
