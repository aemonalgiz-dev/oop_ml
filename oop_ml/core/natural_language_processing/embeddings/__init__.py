"""Embeddings: a vector per word, or per text, learned from what keeps company.

A tokenizer ends in ids, and an id says which row of a table and nothing about
what that row is near. An embedding fills the table so that the geometry
carries what the ids cannot: words that keep the same company end up in the
same place. Every technique here is a different answer to how the table should
be filled, and every one hands back the same object, a
:class:`~oop_ml.core.natural_language_processing.embeddings.vectors.WordEmbeddings`
(or, for a text, a
:class:`~oop_ml.core.natural_language_processing.embeddings.documents.embedder.DocumentVectors`),
so the questions a caller asks -- how near, which nearest, what completes the
analogy -- are asked of one class however the vectors were learned.

The families sit in subpackages named for how the vectors are found.
``counts`` reads them off a matrix of counts: a term-document matrix and its
decomposition (latent semantic analysis), a co-occurrence matrix made into
pointwise mutual information and decomposed, or projected at random (random
indexing). ``prediction`` learns them by predicting neighbours: word2vec in its
two architectures and two objectives, FastText's subword extension, GloVe's
weighted least squares on the logarithm of the counts, and paragraph vectors,
which give a text a vector by the same trick. ``documents`` turns a text into a
vector: the bag of words with its weighting, and the two poolings of word
vectors, the plain mean and the smooth inverse frequency baseline.

Named systems are provided as their mechanism, as in the tokenization package.
item2vec, node2vec and DeepWalk are word2vec over sequences a caller supplies;
they are provided by :class:`Word2Vec` and their sequences (purchases, random
walks) are data. What has no mechanism here is listed in :data:`NOT_PROVIDED`
with the reason, and the accounting test holds the two lists to the request.
"""

from __future__ import annotations

from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.counts.latent_semantic_analysis import (  # noqa: E501
    LatentSemanticAnalysis,
    TruncatedSingularValueDecomposition,
)
from oop_ml.core.natural_language_processing.embeddings.counts.pointwise_mutual_information import (  # noqa: E501
    PointwiseMutualInformationEmbeddings,
    PointwiseMutualInformationMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.counts.random_indexing import (  # noqa: E501
    IndexVectors,
    RandomIndexing,
)
from oop_ml.core.natural_language_processing.embeddings.counts.term_document import (
    TermDocumentMatrix,
    TermWeighting,
)
from oop_ml.core.natural_language_processing.embeddings.documents.bag_of_words import (  # noqa: E501
    BagOfWords,
)
from oop_ml.core.natural_language_processing.embeddings.documents.embedder import (
    DocumentEmbedder,
    DocumentVectors,
)
from oop_ml.core.natural_language_processing.embeddings.documents.pooling import (
    MeanPooling,
    PoolingWeighting,
    SmoothInverseFrequency,
    WordVectorPooling,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import (
    CorpusEmbedder,
    TokenisedCorpus,
    WordEmbedder,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.fasttext import (
    FastText,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.glove import (
    GloVe,
    GloVeParameters,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.huffman import (
    HuffmanCode,
    HuffmanTree,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.objectives import (
    PairGradients,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.paragraph_vectors import (  # noqa: E501
    ParagraphArchitecture,
    ParagraphVectors,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.sampling import (
    UnigramSampler,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.word2vec import (
    EpochRecord,
    TrainingHistory,
    Word2Vec,
    Word2VecArchitecture,
    Word2VecObjective,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    SimilarWord,
    SimilarWords,
    WordEmbeddings,
    WordVector,
    cosine_similarity,
)

NOT_PROVIDED: dict[str, str] = {
    "contextual embeddings (ELMo, BERT, GPT)": (
        "a vector that depends on the sentence the word sits in is the hidden "
        "state of a network read at that position, so there is no table to "
        "learn and hand back; the table these models start from is an "
        "Embedding layer in oop_ml.core.network, which WordEmbeddings.table "
        "can seed, and the layers above it are a network's, not an embedder's"
    ),
    "pre-trained vector files (word2vec binary, GloVe text)": (
        "a reader for two file formats and the files themselves, which are "
        "data measured in gigabytes; loading one is persistence, and this "
        "library's persistence has a closed registry of its own models and a "
        "LEARNED_STATE audit still open. WordEmbeddings(vocabulary, table) "
        "takes any table a caller has already read"
    ),
    "cross-lingual alignment (MUSE, orthogonal Procrustes)": (
        "maps one language's table onto another's with the orthogonal matrix "
        "that best carries a seed dictionary across, which is a singular value "
        "decomposition of the cross-covariance of two tables; it is a step of "
        "linear algebra over two fitted WordEmbeddings and a dictionary a "
        "caller supplies, and belongs beside the decompositions rather than "
        "among the embedders"
    ),
    "hyperbolic embeddings (Poincare)": (
        "place the vectors in a curved space where distance grows "
        "exponentially towards the boundary, so hierarchies fit in few "
        "dimensions; learning them is Riemannian gradient descent with a "
        "retraction onto the ball, a different optimiser from the flat steps "
        "every embedder here takes, and cosine similarity is the wrong "
        "question to ask of them"
    ),
}
"""Techniques with no mechanism here, each with the reason.

Every key is an item of the request this package was built from, and the
accounting test holds the two lists, provided and declined, to cover that
request exactly once each.
"""

__all__ = [
    # The vocabulary every family speaks
    "WordVector",
    "WordEmbeddings",
    "SimilarWord",
    "SimilarWords",
    "cosine_similarity",
    "TokenisedCorpus",
    "CorpusEmbedder",
    "WordEmbedder",
    "ContextWeighting",
    "CooccurrenceMatrix",
    # Counts: vectors read off a matrix
    "TermWeighting",
    "TermDocumentMatrix",
    "TruncatedSingularValueDecomposition",
    "LatentSemanticAnalysis",
    "PointwiseMutualInformationMatrix",
    "PointwiseMutualInformationEmbeddings",
    "IndexVectors",
    "RandomIndexing",
    # Prediction: vectors learned by predicting neighbours
    "Word2Vec",
    "Word2VecArchitecture",
    "Word2VecObjective",
    "EpochRecord",
    "TrainingHistory",
    "UnigramSampler",
    "HuffmanCode",
    "HuffmanTree",
    "PairGradients",
    "FastText",
    "GloVe",
    "GloVeParameters",
    "ParagraphVectors",
    "ParagraphArchitecture",
    # Documents: a vector per text
    "DocumentVectors",
    "DocumentEmbedder",
    "BagOfWords",
    "PoolingWeighting",
    "WordVectorPooling",
    "MeanPooling",
    "SmoothInverseFrequency",
    # What is declined, by name
    "NOT_PROVIDED",
]
