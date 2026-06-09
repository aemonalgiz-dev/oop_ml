"""A document from the vectors of its words: averaged, then averaged with care.

The simplest composition
------------------------
Once every word has a vector, the cheapest vector for a sentence is the mean
of its words' vectors. It discards order entirely -- ``dog bites man`` and
``man bites dog`` are one point -- and it is still a strong baseline, because
what a sentence is about is mostly which words it holds, and the mean of a few
nearby vectors lands near all of them. :class:`MeanPooling` is that mean. Both
classes here take a fitted
:class:`~oop_ml.core.natural_language_processing.embeddings.vectors.WordEmbeddings`
as a *field*, because the word vectors are configuration rather than something
this model learns: any word embedder's table can be handed in, and the words
that have vectors were decided when it was fitted. That is also why
``minimum_count`` is refused at any value but its default -- there is no
vocabulary here for a count to threshold -- while the default passes, so a
constructor written against the frame still constructs.

What the plain mean gets wrong
------------------------------
Every sentence has ``the``, and ``the`` has a vector, so every sentence's mean
is pulled towards the same place, and the more function words a sentence has
the less its mean says about it. Two repairs. :class:`MeanPooling` under
:attr:`PoolingWeighting.INVERSE_DOCUMENT_FREQUENCY` weights each word by the
smoothed inverse document frequency learned from the fit corpus, so a word in
every document counts one and a word in few documents counts more; the mean is
divided by the total weight, so it is a weighted mean and not a weighted sum.
:class:`SmoothInverseFrequency` is Arora, Liang and Ma's (2017) baseline, and
it repairs the same fault twice. Each word is weighted by ``a / (a + p(w))``
for its corpus probability ``p(w)``, which is near one for a rare word and near
zero for a common one; then the first principal direction of the training
documents' vectors, which is where the common words all point, is projected
out of every vector. The paper's ``a`` is the ``smoothing`` field.

Worked, at the paper's ``a = 0.001``
------------------------------------
A word with ``p = 0.5`` -- half of every text -- weighs
``0.001 / 0.501 = 0.001996``; one with ``p = 0.001`` weighs exactly ``0.5``;
one with ``p = 0.000001`` weighs ``0.999001``. The common word is down-weighted
five hundred times against the rare one, before any component is removed.

On a hand-built table of seven words in four dimensions -- three animal words
pointing along the first axis, three finance words along the second, and
``the`` twice as long as any of them along the third -- six two-word
documents each carrying ``the`` twice separate as follows, measuring the
smallest within-topic cosine minus the largest between-topic cosine. The plain
mean: 0.9990 against 0.8349, a margin of 0.164, because ``the`` dominates every
vector. The idf-weighted mean, with ``idf(the) = 1`` and every topic word at
``log(7 / 3) + 1 = 1.8473``: a margin of 0.387. The paper's weights alone, with
``p(the) = 0.5``: 0.789. And with the first component removed: the two topics
land exactly opposite, ``-1.0`` between and at least ``0.9877`` within, a
margin of 1.975. The first component on that fixture is
``(0.638, 0.638, 0.430, 0)``, which is the average topic direction plus the
direction of ``the``; removing it leaves each document with what
distinguishes its topic from the other. Every number is pinned in the spec.

What is a mean of nothing
-------------------------
A document none of whose words has a vector is the zero vector, under every
weighting and both classes. Not an error, because a model asked about text
outside its vocabulary has answered honestly that it knows nothing, and not a
mean of some default, because there is no default word. Its similarity to
anything is then refused by
:meth:`~oop_ml.core.natural_language_processing.embeddings.documents.embedder.DocumentVectors.similarity`,
which is the honest reading of a vector with no direction.

Two decisions written down
--------------------------
The first component is the first right singular vector of the raw matrix of
training document vectors, *uncentred*, which is what the paper and its
reference code do: they fit a truncated singular value decomposition on the
matrix as it stands, and it is the mean direction they mean to remove, which
centring would take away first. A singular vector is determined only up to
sign, and on a single training document numpy returned the negative of the
document's direction, so the sign is fixed here: the entry of largest
magnitude is made positive, ties going to the lowest index. The projection
``u u^T v`` is the same under either sign; the rule exists so that
``first_component`` is a function of the data alone.

Under :attr:`PoolingWeighting.UNIFORM` the transform reads nothing that the fit
learned, and ``fit`` is still required, because the frame's promise is that
``transform`` before ``fit`` raises, and a model that sometimes needed fitting
and sometimes did not would make that promise depend on a field. The inverse
document frequencies are learned under both weightings so that the property
means one thing whichever is set.

Cost: pooling is one pass over the words, and the decomposition is of an
``(n_documents, dimension)`` matrix, ``O(n_documents * dimension^2)``, which
is the cost of one pass over the corpus for the dimensions used here. Nothing
is quadratic in the vocabulary or the corpus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import StrEnum
from functools import partial
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from oop_ml.core.exceptions import InvalidValuesError, TooFewValuesError
from oop_ml.core.natural_language_processing.embeddings.counts.term_document import (
    TermDocumentMatrix,
    TermWeighting,
)
from oop_ml.core.natural_language_processing.embeddings.documents.embedder import (
    DocumentEmbedder,
    DocumentVectors,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import (
    TokenisedCorpus,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.types import FloatArray


class PoolingWeighting(StrEnum):
    """How much each word's vector counts in the mean."""

    UNIFORM = "uniform"
    """Every word alike: the plain mean of the vectors."""

    INVERSE_DOCUMENT_FREQUENCY = "inverse_document_frequency"
    """Each word by its smoothed inverse document frequency from the fit corpus."""


def smooth_inverse_frequency_weights(
    smoothing: float, probabilities: FloatArray
) -> FloatArray:
    """``a / (a + p)`` per word: near one for a rare word, near zero for a common one.

    Raises
    ------
    InvalidValuesError
        If ``smoothing`` is not positive, or a probability is outside
        ``[0, 1]``.
    """
    if not smoothing > 0.0:
        raise InvalidValuesError(f"the smoothing a must be positive, got {smoothing}")
    as_array = np.asarray(probabilities, dtype=np.float64)
    if bool(np.any(as_array < 0.0)) or bool(np.any(as_array > 1.0)):
        raise InvalidValuesError("a word probability lies in [0, 1]")
    return smoothing / (smoothing + as_array)


class WordVectorPooling(DocumentEmbedder, ABC):
    """A document embedder that composes a fitted table of word vectors.

    The base owns the table and the sweep over documents; a subclass supplies
    ``fit`` and :meth:`_table_for`, and reaches the sweep through
    :meth:`_table_of` with the rule for one document.

    Parameters
    ----------
    embeddings:
        The word vectors to pool. Which words have a vector was decided when
        this was fitted, so it is configuration here, not something learned.

    Raises
    ------
    pydantic.ValidationError
        If ``minimum_count`` is anything but its default of one: there is no
        vocabulary here for a count to threshold.
    """

    embeddings: WordEmbeddings

    @model_validator(mode="after")
    def _check_minimum_count_is_the_default(self) -> Self:
        if self.minimum_count != 1:
            raise ValueError(
                f"minimum_count={self.minimum_count} has nothing to threshold: the "
                f"words that have vectors were decided when embeddings was fitted"
            )
        return self

    def transform(self, texts: Sequence[str]) -> DocumentVectors:
        """One pooled vector per text; zero for a text with no known word.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``texts`` is a single string or holds a non-string.
        EmptyValuesError
            If ``texts`` is empty, blank, or yields no words at all.
        """
        self._check_fitted()
        return DocumentVectors(self._table_for(self._tokenised(texts)))

    @abstractmethod
    def _table_for(self, tokenised: TokenisedCorpus) -> FloatArray:
        """``(n_documents, dimension)`` for a tokenised corpus, once fitted."""

    def _table_of(
        self,
        tokenised: TokenisedCorpus,
        document_vector: Callable[[Sequence[int]], FloatArray],
    ) -> FloatArray:
        """Apply ``document_vector`` to each text's in-vocabulary ids.

        A text with no in-vocabulary word is left as the zero row, so the rule
        is only ever asked about a non-empty document.
        """
        table = np.zeros((tokenised.n_sentences, self.embeddings.dimension))
        for row, word_ids in enumerate(
            tokenised.id_sequences(self.embeddings.vocabulary)
        ):
            if word_ids:
                table[row] = document_vector(word_ids)
        return table

    def _vectors_of(self, word_ids: Sequence[int]) -> FloatArray:
        """``(n_words, dimension)``: the table's rows for these ids, in order."""
        return self.embeddings.table[np.asarray(word_ids, dtype=np.intp)]


class MeanPooling(WordVectorPooling):
    """The mean of a document's word vectors, uniform or idf-weighted.

    Parameters
    ----------
    embeddings:
        The word vectors to pool.
    weighting:
        Whether every word counts alike, or by its fitted inverse document
        frequency. Under the weighted form the result is divided by the total
        weight, so it is a weighted mean.
    """

    weighting: PoolingWeighting = PoolingWeighting.UNIFORM

    _inverse_document_frequencies: FloatArray = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn each word's inverse document frequency from ``corpus``.

        Learned under both weightings; read by the transform only under
        :attr:`PoolingWeighting.INVERSE_DOCUMENT_FREQUENCY`.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        """
        tokenised = self._tokenised(corpus)
        inverse_document_frequencies = TermDocumentMatrix.from_tokenised(
            tokenised, self.embeddings.vocabulary, TermWeighting.COUNT
        ).inverse_document_frequencies.copy()
        inverse_document_frequencies.setflags(write=False)

        self._inverse_document_frequencies = inverse_document_frequencies
        self._mark_fitted()
        return self

    @property
    def inverse_document_frequencies(self) -> FloatArray:
        """Each vocabulary word's smoothed weight from the fit corpus, frozen.

        A word in every training document weighs exactly one; a word in none
        weighs ``log(1 + n_documents) + 1``, which is the smoothing doing its
        job.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._inverse_document_frequencies

    def _table_for(self, tokenised: TokenisedCorpus) -> FloatArray:
        return self._table_of(tokenised, self._mean_of)

    def _mean_of(self, word_ids: Sequence[int]) -> FloatArray:
        """The plain or idf-weighted mean of these words' vectors."""
        vectors = self._vectors_of(word_ids)
        if self.weighting is PoolingWeighting.UNIFORM:
            return vectors.mean(axis=0)
        weights = self._inverse_document_frequencies[
            np.asarray(word_ids, dtype=np.intp)
        ]
        # The smoothed idf is at least one, so the total weight cannot vanish.
        return (weights[:, None] * vectors).sum(axis=0) / weights.sum()


class SmoothInverseFrequency(WordVectorPooling):
    """Arora, Liang and Ma's weighted mean with the first component removed.

    Parameters
    ----------
    embeddings:
        The word vectors to pool.
    smoothing:
        The paper's ``a``: each word weighs ``a / (a + p(w))`` for its
        probability ``p(w)`` in the fit corpus.
    remove_first_component:
        Whether the first principal direction of the training documents'
        vectors is projected out of every vector. The direction is learned
        either way and exposed as :attr:`first_component`.
    """

    smoothing: float = Field(default=1e-3, gt=0.0)
    remove_first_component: bool = True

    _word_probabilities: FloatArray = PrivateAttr()
    _word_weights: FloatArray = PrivateAttr()
    _first_component: FloatArray = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the word probabilities and the first component from ``corpus``.

        The probabilities are over occurrences of words that have a vector, so
        they sum to one across the vocabulary; a vocabulary word the corpus
        never used has probability zero and weight one.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string, or every
            training document's vector is zero, so no direction is principal.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        TooFewValuesError
            If no word of the corpus has a vector.
        """
        tokenised = self._tokenised(corpus)
        counts = np.asarray(
            [tokenised.word_counts[word] for word in self.embeddings.vocabulary],
            dtype=np.float64,
        )
        total = float(counts.sum())
        if total == 0.0:
            raise TooFewValuesError(
                f"no word of the corpus has a vector among the "
                f"{self.embeddings.n_words} in the embeddings"
            )

        word_probabilities = counts / total
        word_weights = smooth_inverse_frequency_weights(
            self.smoothing, word_probabilities
        )
        training_vectors = self._table_of(
            tokenised, partial(self._weighted_mean_of, weights=word_weights)
        )
        first_component = self._first_right_singular_vector(training_vectors)

        for values in (word_probabilities, word_weights, first_component):
            values.setflags(write=False)

        self._word_probabilities = word_probabilities
        self._word_weights = word_weights
        self._first_component = first_component
        self._mark_fitted()
        return self

    @property
    def word_probabilities(self) -> FloatArray:
        """``p(w)`` per vocabulary word from the fit corpus, frozen; sums to one.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._word_probabilities

    @property
    def word_weights(self) -> FloatArray:
        """``a / (a + p(w))`` per vocabulary word, frozen.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._word_weights

    @property
    def first_component(self) -> FloatArray:
        """The unit direction removed from every vector, frozen.

        The first right singular vector of the uncentred matrix of training
        document vectors, with its largest-magnitude entry positive.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._first_component

    def weight_of(self, word: str) -> float:
        """``a / (a + p(word))`` for one word of the vocabulary.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        UnknownTokenError
            If the word has no vector.
        """
        self._check_fitted()
        return float(self._word_weights[self.embeddings.vocabulary.id_of(word)])

    def _table_for(self, tokenised: TokenisedCorpus) -> FloatArray:
        table = self._table_of(
            tokenised, partial(self._weighted_mean_of, weights=self._word_weights)
        )
        if self.remove_first_component:
            return self._without(table, self._first_component)
        return table

    def _weighted_mean_of(
        self, word_ids: Sequence[int], weights: FloatArray
    ) -> FloatArray:
        """``mean over words of weights[w] * vector(w)``: divided by the count.

        The paper divides by the sentence length and not by the total weight,
        so a document of common words comes out short rather than rescaled.
        """
        positions = np.asarray(word_ids, dtype=np.intp)
        return (weights[positions][:, None] * self._vectors_of(word_ids)).mean(axis=0)

    @staticmethod
    def _first_right_singular_vector(vectors: FloatArray) -> FloatArray:
        """The uncentred first principal direction, largest entry positive.

        Raises
        ------
        InvalidValuesError
            If every row is zero, since a zero matrix has no direction.
        """
        _, singular_values, right_singular_vectors = np.linalg.svd(
            vectors, full_matrices=False
        )
        if singular_values[0] == 0.0:
            raise InvalidValuesError(
                "every training document's vector is zero, so there is no first "
                "component to remove"
            )
        component = right_singular_vectors[0].copy()
        if component[int(np.argmax(np.abs(component)))] < 0.0:
            component = -component
        return component

    @staticmethod
    def _without(table: FloatArray, component: FloatArray) -> FloatArray:
        """Each row less its projection on ``component``: ``v - u (u . v)``."""
        return table - np.outer(table @ component, component)
