"""A document as the words it uses, counted: the vector space model, per text.

What a bag of words is
----------------------
Throw away the order and keep the counts. A document becomes one number per
vocabulary word, how many times it used it, so the vector's dimension is the
vocabulary's size and two documents are near when they use the same words in
the same proportions. It is Salton's vector space model (1975) read one column
at a time, and it is the baseline every learned document embedding is measured
against, because for a great many tasks -- spam, topic, sentiment on clean text
-- it is very hard to beat.

What is learned, and what is only applied
-----------------------------------------
``fit`` learns two things from the training corpus: which words have a
column, and how much one use of each word should count. The first is the
vocabulary, ordered commonest first with ``minimum_count`` applied. The second
is the smoothed inverse document frequency of every word, computed from the
training
:class:`~oop_ml.core.natural_language_processing.embeddings.counts.term_document.TermDocumentMatrix`
and kept. ``transform`` then counts the vocabulary's words in each new text --
a word outside the vocabulary is ignored, since it has no column -- and, under
:attr:`~oop_ml.core.natural_language_processing.embeddings.counts.term_document.TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY`,
multiplies each count by the *fitted* weight of its word. The document
frequencies are never re-estimated on the texts being transformed, which is
the rule every transformer in this library keeps; here the reason is concrete,
because a word's weight is a statement about how widely it is used in the
collection the model knows, not in the batch it is being asked about.

Worked, on three documents
--------------------------
``the cat sat``, ``the dog sat``, ``a cat``. Counting occurrences, ``cat``,
``sat`` and ``the`` appear twice each and ``a`` and ``dog`` once, so with ties
broken alphabetically the vocabulary is ``cat, sat, the, a, dog``. Each of the
first three is in two of the three documents and each of the last two in one,
so the smoothed weights are ``log(4 / 3) + 1 = 1.2877`` for ``the`` and
``log(4 / 2) + 1 = 1.6931`` for ``dog``. Under counts, ``the dog sat`` is the
vector ``(0, 1, 1, 0, 1)``; under TF-IDF it is
``(0, 1.2877, 1.2877, 0, 1.6931)``, and ``the zebra sat`` is ``(0, 1.2877,
1.2877, 0, 0)`` with the unseen word contributing nothing. Transforming the
training texts themselves reproduces the training matrix, transposed, under
either weighting, and the spec says so.

Why normalise, and what a zero document does
--------------------------------------------
A long document uses more words, so its raw vector is longer, and Euclidean
distance between two documents then mostly measures their lengths. Dividing
each vector by its own Euclidean norm puts every document on the unit sphere,
where distance and cosine agree and length is gone. A document with no known
word is the zero vector, has no norm to divide by, and stays zero; its
similarity to anything is then refused by
:meth:`~oop_ml.core.natural_language_processing.embeddings.documents.embedder.DocumentVectors.similarity`
rather than reported as zero.

The vectors are dense and of vocabulary width. Real collections make that
sparse and wide, and the usual repair is a sparse matrix; here the vocabularies
are small enough that a dense table is the readable choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import PrivateAttr

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
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray


def unit_rows(values: FloatArray) -> FloatArray:
    """Every non-zero row divided by its Euclidean norm; a zero row stays zero."""
    norms = np.linalg.norm(values, axis=1)
    divisors = np.where(norms == 0.0, 1.0, norms)
    return values / divisors[:, None]


class BagOfWords(DocumentEmbedder):
    """One count per vocabulary word per document, weighted as asked.

    Parameters
    ----------
    weighting:
        Whether a use counts one, or one times the word's fitted inverse
        document frequency.
    normalise:
        Whether each document's vector is divided by its Euclidean norm. A
        document with no known word stays the zero vector.
    """

    weighting: TermWeighting = TermWeighting.COUNT
    normalise: bool = False

    _vocabulary: Vocabulary = PrivateAttr()
    _term_document_matrix: TermDocumentMatrix = PrivateAttr()
    _inverse_document_frequencies: FloatArray = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn the vocabulary and every word's weight from ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        TooFewValuesError
            If no word reaches ``minimum_count``.
        """
        tokenised = self._tokenised(corpus)
        vocabulary = tokenised.vocabulary(self.minimum_count)
        term_document_matrix = TermDocumentMatrix.from_tokenised(
            tokenised, vocabulary, self.weighting
        )
        inverse_document_frequencies = (
            term_document_matrix.inverse_document_frequencies.copy()
        )
        inverse_document_frequencies.setflags(write=False)

        self._vocabulary = vocabulary
        self._term_document_matrix = term_document_matrix
        self._inverse_document_frequencies = inverse_document_frequencies
        self._mark_fitted()
        return self

    def transform(self, texts: Sequence[str]) -> DocumentVectors:
        """Count the vocabulary's words in each text, weighted as fitted.

        The dimension is the vocabulary's size. A word outside the vocabulary
        contributes nothing, and a text with no vocabulary word is the zero
        vector, normalised or not.

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
        values = self._counts_of(self._tokenised(texts))
        if self.weighting is TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY:
            values = values * self._inverse_document_frequencies[None, :]
        if self.normalise:
            values = unit_rows(values)
        return DocumentVectors(values)

    @property
    def vocabulary(self) -> Vocabulary:
        """The words that have a column, commonest first.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._vocabulary

    @property
    def inverse_document_frequencies(self) -> FloatArray:
        """Each vocabulary word's smoothed weight from the training corpus, frozen.

        Learned under either weighting, so the property means one thing; the
        transform reads it only under TF-IDF.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._inverse_document_frequencies

    @property
    def term_document_matrix(self) -> TermDocumentMatrix:
        """The training corpus as ``(n_terms, n_documents)``, weighted as fitted.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._term_document_matrix

    def _counts_of(self, tokenised: TokenisedCorpus) -> FloatArray:
        """``(n_documents, n_terms)``: how often each text used each word."""
        counts = np.zeros((tokenised.n_sentences, self._vocabulary.n_tokens))
        for row, sentence in enumerate(tokenised.sentences):
            for word in sentence:
                if word in self._vocabulary:
                    counts[row, self._vocabulary.id_of(word)] += 1.0
        return counts
