"""Giving words a geometry, four ways, and asking each the same questions.

A tokenizer ends in ids, and an id is a name: it says which row of a table and
nothing about what the row is near. An embedding fills the table so that the
geometry carries what the ids cannot. This script fits four techniques to one
small corpus of two topics -- cooking and sailing, with a few function words
shared between them -- and asks each which words sit near which.

The four are chosen because they disagree about *how* to fill the table. Word2vec
learns by prediction: slide a window along the corpus and nudge the vectors so a
word predicts its neighbours. GloVe fits the logarithm of the co-occurrence
counts by weighted least squares. Latent semantic analysis decomposes the
term-document matrix. Pointwise mutual information asks how much more often two
words appear together than chance would give, and decomposes that.

Two things to notice. Every technique puts the cooking words together and the
sailing words together, so the within-topic similarity beats the across-topic
one whichever route was taken; that is the distributional hypothesis, measured.
And the function words that appear in both topics land between them under every
technique, because their company is mixed -- which is also why they are the
words a model learns least from.
"""

from __future__ import annotations

import logging

import numpy as np

from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import (
    GloVe,
    LatentSemanticAnalysis,
    MeanPooling,
    PointwiseMutualInformationEmbeddings,
    Word2Vec,
)

logger = logging.getLogger(__name__)

COOKING = ["flour", "sugar", "butter", "oven", "whisk", "batter", "bake", "knead"]
SAILING = ["mast", "sail", "rudder", "anchor", "harbour", "tide", "keel", "moor"]
FUNCTION_WORDS = ["the", "and", "then", "with"]

DIMENSION = 12
RANDOM_SEED = 3


def make_corpus() -> list[str]:
    """Twenty-four short texts per topic, each topic word recurring often.

    Built with a seeded generator so the report is the same every run. Each
    text is five topic words joined by one function word, which is what gives
    the function words mixed company.
    """
    generator = np.random.default_rng(RANDOM_SEED)
    texts: list[str] = []
    for topic in (COOKING, SAILING):
        for _ in range(24):
            words = list(generator.choice(topic, size=5, replace=False))
            words.insert(2, str(generator.choice(FUNCTION_WORDS)))
            texts.append(" ".join(words))
    generator.shuffle(texts)
    return texts


def mean_similarity(model, first_words: list[str], second_words: list[str]) -> float:
    return float(
        np.mean(
            [
                model.similarity(first, second)
                for first in first_words
                for second in second_words
                if first != second
            ]
        )
    )


def main() -> None:
    report = Report(logger)
    corpus = make_corpus()

    report.heading("One corpus, four tables")
    models = {
        "word2vec": Word2Vec(
            dimension=DIMENSION,
            window=3,
            epochs=15,
            learning_rate=0.05,
            random_seed=RANDOM_SEED,
        ),
        "GloVe": GloVe(
            dimension=DIMENSION, window=3, epochs=40, random_seed=RANDOM_SEED
        ),
        "latent semantic analysis": LatentSemanticAnalysis(dimension=DIMENSION),
        "pointwise mutual information": PointwiseMutualInformationEmbeddings(
            dimension=DIMENSION, window=3
        ),
    }
    for model in models.values():
        model.fit(corpus)

    rows = []
    for label, model in models.items():
        rows.append(
            [
                label,
                f"{mean_similarity(model, COOKING, COOKING):.3f}",
                f"{mean_similarity(model, SAILING, SAILING):.3f}",
                f"{mean_similarity(model, COOKING, SAILING):.3f}",
            ]
        )
    report.table(["technique", "cooking-cooking", "sailing-sailing", "across"], rows)
    report.paragraph(
        "Four different objectives, one geometry: within a topic the vectors\n"
        "point the same way and across topics they do not. The numbers differ\n"
        "because the techniques measure company differently -- a window of\n"
        "predicted neighbours, a weighted fit to log counts, a decomposition of\n"
        "which documents a word appears in -- and agree because the corpus has\n"
        "one fact to find."
    )

    report.heading("Which words are nearest")
    rows = []
    for label, model in models.items():
        rows.append(
            [
                label,
                " ".join(model.most_similar("flour", n_results=3).words),
                " ".join(model.most_similar("anchor", n_results=3).words),
                " ".join(model.most_similar("the", n_results=3).words),
            ]
        )
    report.table(
        ["technique", "nearest to flour", "nearest to anchor", "nearest to the"], rows
    )
    report.paragraph(
        "flour's neighbours are cooking words and anchor's are sailing words\n"
        "under every technique. the has no topic, so its neighbours are the\n"
        "other function words or whichever topic it happened to land nearer;\n"
        "mixed company gives a word a vector between the groups, which is the\n"
        "honest answer to a word that means nothing on its own."
    )

    report.heading("What word2vec recorded while it learned")
    word2vec = models["word2vec"]
    history = word2vec.history
    report.line(
        f"mean loss per epoch: "
        f"{', '.join(f'{record.mean_loss:.3f}' for record in history)}"
    )
    report.line(f"loss fell over training: {history.fell}")
    report.paragraph(
        "The loss is not a number anyone reports for word2vec, since it depends\n"
        "on which negatives were drawn, but it is what the optimiser lowers,\n"
        "and a history that does not fall is a learning rate that was wrong."
    )

    report.heading("From words to texts")
    pooling = MeanPooling(embeddings=word2vec.embeddings).fit(corpus)
    documents = pooling.transform(
        ["knead the batter then bake", "moor the keel with the tide"]
    )
    nearest_word = pooling.embeddings.similar_to_vector(
        documents.vector_of(0), n_results=1
    )
    report.line(
        f"cooking text against sailing text, cosine: {documents.similarity(0, 1):.3f}"
    )
    report.line(
        f"nearest word to the cooking text: {nearest_word.words[0]} at "
        f"{nearest_word.similarities[0]:.3f}"
    )
    report.paragraph(
        "A text is the mean of its words' vectors. The two texts still agree\n"
        "somewhat, because both are short and both contain the, whose vector\n"
        "sits between the topics; but the cooking text points almost exactly\n"
        "where its own topic's words do, at bake. The bag of words could count\n"
        "the shared the, and nothing else: it has no idea that knead and bake\n"
        "belong together."
    )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
