"""Texts and a hand-built table shared by the document specs, each with what is known.

Every number here was worked from the definitions before the implementation
ran, then confirmed by it. The helpers are written from the definitions too --
a plain dot product over two norms -- so that no oracle restates the code it
is checking.
"""

from __future__ import annotations

import math

import numpy as np

from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray

# --- The bag of words, on three documents -----------------------------------

THREE_DOCUMENTS: list[str] = ["the cat sat", "the dog sat", "a cat"]

# Occurrence counts: cat 2, sat 2, the 2, dog 1, a 1. Commonest first, ties
# alphabetical.
THREE_DOCUMENT_VOCABULARY: list[str] = ["cat", "sat", "the", "a", "dog"]

# (n_terms, n_documents), one row per vocabulary word in that order.
THREE_DOCUMENT_COUNTS: FloatArray = np.array(
    [
        [1.0, 0.0, 1.0],  # cat
        [1.0, 1.0, 0.0],  # sat
        [1.0, 1.0, 0.0],  # the
        [0.0, 0.0, 1.0],  # a
        [0.0, 1.0, 0.0],  # dog
    ]
)

# Smoothed: log((1 + 3) / (1 + document_frequency)) + 1.
IDF_IN_TWO_OF_THREE: float = math.log(4.0 / 3.0) + 1.0  # 1.2877
IDF_IN_ONE_OF_THREE: float = math.log(4.0 / 2.0) + 1.0  # 1.6931

# --- Two topics, hand-built ---------------------------------------------------

# Animal words point along the first axis, finance words along the second, and
# ``the`` is twice as long as any of them along the third, so a plain mean of a
# document carrying ``the`` twice is dominated by it.
TOPIC_WORDS: list[str] = ["the", "cat", "dog", "pet", "stock", "bond", "market"]

TOPIC_TABLE: FloatArray = np.array(
    [
        [0.0, 0.0, 2.0, 0.0],  # the
        [1.0, 0.0, 0.0, 0.1],  # cat
        [1.0, 0.0, 0.0, -0.1],  # dog
        [0.9, 0.1, 0.0, 0.0],  # pet
        [0.0, 1.0, 0.0, 0.1],  # stock
        [0.0, 1.0, 0.0, -0.1],  # bond
        [0.1, 0.9, 0.0, 0.0],  # market
    ]
)

# Six two-word documents, ``the`` twice in each: 24 words, 12 of them ``the``.
TOPIC_CORPUS: list[str] = [
    "the cat the dog",
    "the pet the cat",
    "the dog the pet",
    "the stock the bond",
    "the market the stock",
    "the bond the market",
]

ANIMAL_DOCUMENTS: tuple[int, ...] = (0, 1, 2)
FINANCE_DOCUMENTS: tuple[int, ...] = (3, 4, 5)

# Word probabilities over the corpus: ``the`` is 12 of 24, every other word 2.
PROBABILITY_OF_THE: float = 0.5
PROBABILITY_OF_A_TOPIC_WORD: float = 2.0 / 24.0

# ``the`` is in all six documents, every topic word in two of six.
TOPIC_IDF_OF_THE: float = math.log(7.0 / 7.0) + 1.0  # exactly 1.0
TOPIC_IDF_OF_A_TOPIC_WORD: float = math.log(7.0 / 3.0) + 1.0  # 1.8473

# Smallest within-topic cosine minus largest between-topic cosine, worked in
# the module docstring of ``pooling.py``.
UNIFORM_MEAN_MARGIN: float = 0.1640
IDF_MEAN_MARGIN: float = 0.3871
SIF_WEIGHTS_ONLY_MARGIN: float = 0.7886
SIF_MARGIN: float = 1.9755


def topic_embeddings() -> WordEmbeddings:
    return WordEmbeddings(Vocabulary(TOPIC_WORDS), TOPIC_TABLE)


def cosine(first: FloatArray, second: FloatArray) -> float:
    """The definition, independent of the library's implementation."""
    return float(
        np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    )


def separation_margin(vectors: FloatArray) -> float:
    """Smallest within-topic cosine minus largest between-topic cosine."""
    within = [
        cosine(vectors[first], vectors[second])
        for group in (ANIMAL_DOCUMENTS, FINANCE_DOCUMENTS)
        for first in group
        for second in group
        if first < second
    ]
    between = [
        cosine(vectors[first], vectors[second])
        for first in ANIMAL_DOCUMENTS
        for second in FINANCE_DOCUMENTS
    ]
    return min(within) - max(between)
