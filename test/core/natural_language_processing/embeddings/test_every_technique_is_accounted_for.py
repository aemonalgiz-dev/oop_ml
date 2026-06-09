"""Every embedding technique the request named is provided as its mechanism or declined.

The request was word2vec "and the other techniques". This table is the list
that answer was taken to mean, one entry per technique, mapped to the classes
that implement its mechanism -- or to nothing, which means the package must say
why in ``NOT_PROVIDED``. The tokenization counterpart is
``test_every_tokenizer_is_accounted_for``, and the rule is the same: a
technique can be provided or it can be declined, and it cannot be quietly
dropped, and it cannot be both.
"""

from __future__ import annotations

import pytest

from oop_ml.core.natural_language_processing import embeddings

REQUESTED: dict[str, tuple[str, ...]] = {
    # Count-based
    "one-hot and bag-of-words document vectors": ("BagOfWords", "DocumentVectors"),
    "term frequency times inverse document frequency": (
        "TermDocumentMatrix",
        "TermWeighting",
        "BagOfWords",
    ),
    "latent semantic analysis": (
        "LatentSemanticAnalysis",
        "TruncatedSingularValueDecomposition",
    ),
    "co-occurrence counting with a window": ("CooccurrenceMatrix", "ContextWeighting"),
    "positive pointwise mutual information with a decomposition": (
        "PointwiseMutualInformationEmbeddings",
        "PointwiseMutualInformationMatrix",
    ),
    "random indexing": ("RandomIndexing", "IndexVectors"),
    # Prediction-based
    "word2vec skip-gram and continuous bag of words": (
        "Word2Vec",
        "Word2VecArchitecture",
    ),
    "negative sampling": ("Word2Vec", "Word2VecObjective", "UnigramSampler"),
    "hierarchical softmax over a Huffman tree": (
        "Word2Vec",
        "Word2VecObjective",
        "HuffmanTree",
    ),
    "GloVe": ("GloVe",),
    "FastText subword vectors": ("FastText",),
    "paragraph vectors (doc2vec)": ("ParagraphVectors", "ParagraphArchitecture"),
    "item2vec, node2vec and DeepWalk": ("Word2Vec",),
    # Documents from word vectors
    "mean pooling of word vectors": ("MeanPooling", "PoolingWeighting"),
    "smooth inverse frequency": ("SmoothInverseFrequency",),
    # Asking the table
    "nearest neighbours and analogies by cosine": (
        "WordEmbeddings",
        "SimilarWords",
        "cosine_similarity",
    ),
    # Declined
    "contextual embeddings (ELMo, BERT, GPT)": (),
    "pre-trained vector files (word2vec binary, GloVe text)": (),
    "cross-lingual alignment (MUSE, orthogonal Procrustes)": (),
    "hyperbolic embeddings (Poincare)": (),
}

PROVIDED = {item: classes for item, classes in REQUESTED.items() if classes}
DECLINED = tuple(item for item, classes in REQUESTED.items() if not classes)


def test_the_request_is_not_empty() -> None:
    """A guard on the guard: an emptied table would pass everything below."""
    assert len(REQUESTED) >= 18
    assert len(PROVIDED) >= 14
    assert len(DECLINED) >= 3


@pytest.mark.parametrize("item", sorted(PROVIDED))
def test_a_provided_item_is_exported_and_not_declined(item: str) -> None:
    for name in PROVIDED[item]:
        assert hasattr(embeddings, name), (
            f"{item!r} is said to be provided by {name}, which the package does "
            f"not export"
        )
        assert name in embeddings.__all__
    assert item not in embeddings.NOT_PROVIDED, (
        f"{item!r} is both provided and declined; resolve the contradiction"
    )


@pytest.mark.parametrize("item", DECLINED)
def test_a_declined_item_names_a_reason(item: str) -> None:
    assert item in embeddings.NOT_PROVIDED, (
        f"{item!r} is neither provided nor declined; add the mechanism or a "
        f"reason to NOT_PROVIDED"
    )
    assert embeddings.NOT_PROVIDED[item].strip()


def test_nothing_is_declined_that_was_not_requested() -> None:
    assert set(embeddings.NOT_PROVIDED) <= set(REQUESTED)


def test_every_reason_names_where_the_mechanism_would_live() -> None:
    for item, reason in embeddings.NOT_PROVIDED.items():
        names_something = any(
            marker in reason
            for marker in (
                "WordEmbeddings",
                "network",
                "persistence",
                "decomposition",
                "optimiser",
            )
        )
        assert names_something, f"the reason for {item!r} names no mechanism"


def test_every_export_is_a_real_attribute() -> None:
    missing = [name for name in embeddings.__all__ if not hasattr(embeddings, name)]

    assert not missing
