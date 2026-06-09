"""Paragraph vectors: a vector per document, learned as a word always in the window.

The idea, in the paper's own framing
------------------------------------
Word2vec learns a word's vector by asking it to help predict the words that
appear beside it. Le and Mikolov (2014) noticed that the same loop can learn a
vector for a whole document by treating the document as one more token: one
that is present at every window inside that document and at no window outside
it. The paragraph vector is, in their phrase, another word that is always in
the window. At every position the loop then trains two things at once -- the
vectors of the words involved and the vector of the document -- and after
enough passes the document vector holds whatever it needed to hold to help
predict that document's words. Since what the words of one document share and
the words of a single context do not is the subject, that is roughly what the
vector ends up carrying. This is the model gensim calls Doc2Vec.

Two architectures
-----------------
:attr:`ParagraphArchitecture.DISTRIBUTED_MEMORY` (PV-DM) is the continuous
bag of words with the document added to the bag: the hidden vector is the mean
of the document vector and the context words' vectors, and it predicts the
centre word. The paper also concatenates the vectors rather than averaging
them; averaging is the variant here because it is what the shared objective
already handles and because a concatenation fixes the window size into the
model. The hidden gradient is spread evenly over everything that was
averaged, so the document row and each context row move by the same vector,
``-learning_rate * hidden_gradient / (n_context + 1)``, which is the claim
:func:`shared_input_step` holds and the spec pins against a finite difference.
A one-word document has no context, and its vector alone predicts the word,
so no position is skipped.

:attr:`ParagraphArchitecture.DISTRIBUTED_BAG_OF_WORDS` (PV-DBOW) is skip-gram
with the document standing where the centre word stood: the hidden vector is
the document vector alone, and it predicts a word of the document. The paper
samples a window and then a word from it at each step; here every word of the
document is predicted once per epoch, which is the same objective with the
sampling replaced by a sweep. Only the document vector and the output rows
move. **The word input vectors are not trained under PV-DBOW** and the
``embeddings`` a plain DBOW fit answers are the untouched random start,
which is what gensim's ``dm=0`` also produces. ``train_words`` (gensim's
``dbow_words``) adds an ordinary skip-gram pair for each context word at each
position, sharing the output table, and then the word vectors are learned.
Under PV-DM the word vectors are always trained, so ``train_words`` is refused
there rather than accepted and read nowhere.

Why inference is a small optimisation and not a lookup
------------------------------------------------------
A word seen at fit time has a row in a table, and so does a document seen at
fit time. A document that arrives afterwards has no row, and there is nothing
to look up. What there is is the same loop: freeze the word vectors and the
output vectors, give the new document a fresh random vector, and run the same
updates for ``inference_epochs`` passes, moving only that one vector.
:meth:`ParagraphVectors.infer_vector` is that loop. It is gradient descent, so
the answer depends on its random start and on how many passes it is given; the
same text under the same seed gives the same vector to the last bit, and the
fitted tables are not touched. A new text whose words are all outside the
vocabulary has nothing to descend on and is handed its random start, which is
also what a training document made entirely of rare words is left with.

What was measured
-----------------
Thirty eight-word documents, fifteen about cooking and fifteen about sailing,
drawn from two ten-word lists with no word in common, at dimension 8, window
3, five negatives, thirty epochs from a learning rate of 0.1, seed 0.
Word2vec's own defaults, ten epochs from 0.025, are too few updates for a
corpus this small: 2,400 positions leave the mean loss at 4.116 from 4.159
and the document vectors at their random starts, with a within-topic cosine
of 0.05. At the settings above, the mean cosine between two documents of one
topic against the mean between one of each:

* PV-DM: 0.8974 and 0.8696 within, 0.2811 across; mean loss 4.1529 to 2.0891.
* PV-DBOW: 0.7763 and 0.8078 within, 0.1445 across; 4.1538 to 1.8720.

Under both, the fourteen documents nearest the first cooking document are the
other fourteen cooking documents. The across figure is positive rather than
negative: every document is trained against the same output rows and so every
document vector shares a component, which is why the claim is a margin and
not a sign. An untrained pair costs ``6 log 2 = 4.1589``, one ``log 2`` per
row scored against a zero output table, and the first epoch's mean sits just
below it because the rows begin to move within the epoch.

Inference is where the two architectures part, and the averaging is the
cause. Under PV-DM a word row takes the same step as the document row at
every position it appears at, and a word appears in many documents where a
document has eight positions, so after the fit the word vectors have norms
1.74 to 2.68 and the document vectors 0.29 to 0.53. A new document's vector
starts near 0.1 as a small share of a mean dominated by words, and needs many
passes: on a held-out cooking text, twenty passes give 0.4786 to the cooking
documents and 0.7608 to the sailing ones, which is the wrong topic; one
hundred give 0.9031 and 0.2116; two hundred give 0.9081 and 0.1405. Under
PV-DBOW the document vector *is* the hidden vector, and a single pass already
gives 0.8330 against 0.4047, twenty give 0.8565 against 0.1420, and further
passes make the vector more particular to its own eight words rather than
nearer the topic, 0.7665 against 0.0924 at two hundred. So
``inference_epochs`` defaults to twenty, which serves PV-DBOW, and a PV-DM
user of a small model should raise it; the spec runs both at two hundred and
pins the twenty-pass PV-DM figure as it is.

The word vectors: under PV-DM ``butter`` and ``garlic`` reach cosine 0.994
against 0.120 for ``butter`` and ``anchor``; under plain PV-DBOW they stay at
their random start, where the same two pairs happen to read 0.4206 and
-0.2104; with ``train_words`` at ten epochs they reach 0.9666 and 0.0965.
Every figure is pinned in the spec, as an inequality where the claim is one.

One position's gradient, worked by hand at dimension three: document vector
``(0.2, -0.1, 0.4)``, one context word ``(0.0, 0.3, -0.2)``, so the hidden
vector is their mean ``(0.1, 0.1, 0.1)``. With the target's output row
``(1, 0, 0)`` and one negative's ``(0, 1, 0)`` both scores are ``0.1``, the
probability is ``sigma(0.1) = 0.524979``, so the errors are ``-0.475021`` for
the target and ``0.524979`` for the negative, and the hidden gradient is
``(-0.475021, 0.524979, 0)``. At a learning rate of ``0.025`` the shared step
is ``-0.025 * gradient / 2 = (0.005938, -0.006562, 0)``, added to the
document row and to the context row alike.

Seeds, ties and order
---------------------
One generator, seeded by ``random_seed``, draws the word table first and the
document table second, both uniform in ``(-0.5, 0.5) / dimension``, and then
the negatives, in corpus order. The output table starts at zero, as in
word2vec, so the input tables are what break the symmetry. Documents are
visited in corpus order and positions left to right; the window is fixed at
``window`` on each side rather than shrunk at random. The learning rate falls
linearly from ``learning_rate`` towards ``minimum_learning_rate`` over every
position of every epoch, and again over every position of every inference
pass. ``most_similar`` on the table ranks by cosine with a stable sort, so a
tie goes to the lower document position, and a document whose vector is zero
is skipped since nothing is similar to it.

Cost
----
Each position costs one hidden vector of ``dimension`` numbers and
``n_negative_samples + 1`` output rows, so a fit is linear in the corpus and
inference linear in the text; the loop is plain Python over positions, which
is the same shape as word2vec here and the same place a vectorised batch of
positions would be the repair.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from oop_ml.core.exceptions import (
    InvalidValuesError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.natural_language_processing.embeddings.documents.embedder import (
    DocumentVectors,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import WordEmbedder
from oop_ml.core.natural_language_processing.embeddings.prediction.objectives import (
    PairGradients,
    negative_sampling_gradients,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.sampling import (
    UnigramSampler,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.word2vec import (
    EpochRecord,
    TrainingHistory,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    WordEmbeddings,
)
from oop_ml.core.types import FloatArray


class ParagraphArchitecture(StrEnum):
    """What the document vector is averaged with, if anything, to predict a word."""

    DISTRIBUTED_MEMORY = "distributed_memory"
    """The document and its context words, averaged, predict the centre word."""

    DISTRIBUTED_BAG_OF_WORDS = "distributed_bag_of_words"
    """The document vector alone predicts each word of the document."""


# NOTE: a small value object of this module's own, because the documents family
# (a DocumentEmbedder base and a DocumentVectors object) is being written in
# parallel and does not exist yet. The coordinator will reconcile the two
# afterwards; until then this is the one table a paragraph-vector fit answers.
def averaged_hidden(
    document_vector: FloatArray, context_vectors: FloatArray
) -> FloatArray:
    """The mean of the document vector and the context words' vectors.

    The hidden vector PV-DM predicts the centre word from. With no context, as
    in a one-word document, it is the document vector alone.

    Parameters
    ----------
    document_vector:
        ``(dimension,)``.
    context_vectors:
        ``(n_context, dimension)``; ``n_context`` may be zero.

    Raises
    ------
    ShapeMismatchError
        If the context rows are not of the document vector's width.
    """
    if (
        context_vectors.ndim != 2
        or context_vectors.shape[1] != document_vector.shape[0]
    ):
        raise ShapeMismatchError(
            f"context vectors of shape {context_vectors.shape} against a document "
            f"vector of width {document_vector.shape[0]}"
        )
    if context_vectors.shape[0] == 0:
        return np.array(document_vector, dtype=np.float64, copy=True)
    return np.mean(np.vstack([document_vector, context_vectors]), axis=0)


def shared_input_step(
    hidden_gradient: FloatArray, learning_rate: float, n_averaged: int
) -> FloatArray:
    """The step every averaged vector takes: ``-rate * gradient / n_averaged``.

    The hidden vector is a mean of ``n_averaged`` rows, so each row's share of
    the hidden gradient is one ``n_averaged``-th of it, and every row moves by
    the same vector.

    Raises
    ------
    InvalidValuesError
        If ``n_averaged`` is below one, since nothing was averaged.
    """
    if n_averaged < 1:
        raise InvalidValuesError(
            f"a mean is over at least one vector, got n_averaged={n_averaged}"
        )
    return -learning_rate * hidden_gradient / n_averaged


def distributed_memory_gradients(
    document_vector: FloatArray,
    context_vectors: FloatArray,
    target_id: int,
    output_vectors: FloatArray,
    sampler: UnigramSampler,
    n_negative_samples: int,
    generator: np.random.Generator,
) -> PairGradients:
    """One PV-DM position: the averaged hidden vector against the centre word.

    The foundation's negative-sampling step on :func:`averaged_hidden`, so the
    hidden gradient it returns is with respect to the mean and has to be shared
    out by :func:`shared_input_step`.
    """
    return negative_sampling_gradients(
        averaged_hidden(document_vector, context_vectors),
        target_id,
        output_vectors,
        sampler,
        n_negative_samples,
        generator,
    )


def distributed_bag_of_words_gradients(
    document_vector: FloatArray,
    target_id: int,
    output_vectors: FloatArray,
    sampler: UnigramSampler,
    n_negative_samples: int,
    generator: np.random.Generator,
) -> PairGradients:
    """One PV-DBOW position: the document vector alone against one of its words.

    The foundation's negative-sampling step with the document vector as the
    hidden vector, so the whole hidden gradient belongs to the document row.
    """
    return negative_sampling_gradients(
        document_vector,
        target_id,
        output_vectors,
        sampler,
        n_negative_samples,
        generator,
    )


class ParagraphVectors(WordEmbedder):
    """Le and Mikolov's paragraph vectors: one vector per document, and per word.

    Each text of the corpus handed to ``fit`` is one document with its own
    vector. A text seen afterwards is given a vector by
    :meth:`infer_vector`, which runs the same descent against the frozen
    tables.

    Parameters
    ----------
    dimension:
        How many numbers each vector holds, word and document alike.
    window:
        How many positions on each side of the centre count as context. Read
        under distributed memory, and under the bag of words only when
        ``train_words`` is set.
    architecture:
        Whether the document is averaged with its context to predict the centre
        word, or predicts each word alone.
    n_negative_samples:
        How many random words each pair is trained against.
    negative_sampling_exponent:
        The flattening power applied to counts before drawing negatives.
    epochs:
        How many passes over the corpus.
    learning_rate:
        The step size at the start of training, and of each inference.
    minimum_learning_rate:
        The floor the rate decays towards, linearly over every position of
        every epoch. At most ``learning_rate``.
    inference_epochs:
        How many passes :meth:`infer_vector` runs over a new text. Twenty
        serves the bag of words; distributed memory moves a new vector by a
        share of the hidden gradient and wants many more on a small model, as
        the module docstring measures.
    train_words:
        Under the bag of words, whether to also train the word vectors with a
        skip-gram pair per context word (gensim's ``dbow_words``). Refused
        under distributed memory, where the word vectors are always trained.
    random_seed:
        Seeds the initial tables and the negatives, so a seeded fit is
        reproducible. Also the default seed of :meth:`infer_vector`.
    """

    dimension: int = Field(default=50, ge=1)
    window: int = Field(default=5, ge=1)
    architecture: ParagraphArchitecture = ParagraphArchitecture.DISTRIBUTED_MEMORY
    n_negative_samples: int = Field(default=5, ge=1)
    negative_sampling_exponent: float = Field(default=0.75, gt=0.0, le=1.0)
    epochs: int = Field(default=10, ge=1)
    learning_rate: float = Field(default=0.025, gt=0.0)
    minimum_learning_rate: float = Field(default=0.0001, gt=0.0)
    inference_epochs: int = Field(default=20, ge=1)
    train_words: bool = False
    random_seed: int | None = None

    _embeddings: WordEmbeddings = PrivateAttr()
    _document_vectors: DocumentVectors = PrivateAttr()
    _output_vectors: FloatArray = PrivateAttr()
    _history: TrainingHistory = PrivateAttr()
    _sampler: UnigramSampler = PrivateAttr()
    _word_vectors_were_trained: bool = PrivateAttr()

    @model_validator(mode="after")
    def _check_the_floor_is_below_the_start(self) -> Self:
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError(
                f"minimum_learning_rate={self.minimum_learning_rate} is above "
                f"learning_rate={self.learning_rate}; the rate decays downward"
            )
        return self

    @model_validator(mode="after")
    def _check_train_words_is_a_bag_of_words_switch(self) -> Self:
        if (
            self.train_words
            and self.architecture is ParagraphArchitecture.DISTRIBUTED_MEMORY
        ):
            raise ValueError(
                "train_words is a distributed bag of words switch; under "
                "distributed memory the word vectors are always trained, so the "
                "field would be accepted and read nowhere"
            )
        return self

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn a vector per word and per text of ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        TooFewValuesError
            If fewer than two words reach ``minimum_count``, since one word
            has no negative to draw.
        """
        tokenised = self._tokenised(corpus)
        vocabulary = tokenised.vocabulary(self.minimum_count)
        if vocabulary.n_tokens < 2:
            raise TooFewValuesError(
                "paragraph vectors need at least two words in the vocabulary: one "
                "word has no negative to draw"
            )

        id_sequences = tokenised.id_sequences(vocabulary)
        counts = [tokenised.word_counts[word] for word in vocabulary]
        generator = np.random.default_rng(self.random_seed)

        word_vectors = self._initial_vectors(vocabulary.n_tokens, generator)
        document_vectors = self._initial_vectors(len(id_sequences), generator)
        output_vectors = np.zeros((vocabulary.n_tokens, self.dimension))
        sampler = UnigramSampler(counts, self.negative_sampling_exponent)

        total_positions = sum(len(sentence) for sentence in id_sequences) * self.epochs
        positions_seen = 0
        learning_rate = self.learning_rate
        records: list[EpochRecord] = []

        for epoch in range(1, self.epochs + 1):
            loss_total = 0.0
            n_pairs = 0
            for document_position, sentence in enumerate(id_sequences):
                for position, centre_id in enumerate(sentence):
                    learning_rate = self._decayed_learning_rate(
                        positions_seen, total_positions
                    )
                    positions_seen += 1
                    context = self._context_of(sentence, position)

                    if self.architecture is ParagraphArchitecture.DISTRIBUTED_MEMORY:
                        gradients = distributed_memory_gradients(
                            document_vectors[document_position],
                            word_vectors[list(context)],
                            centre_id,
                            output_vectors,
                            sampler,
                            self.n_negative_samples,
                            generator,
                        )
                        np.add.at(
                            output_vectors,
                            gradients.output_ids,
                            -learning_rate * gradients.output_gradients,
                        )
                        step = shared_input_step(
                            gradients.hidden_gradient, learning_rate, len(context) + 1
                        )
                        document_vectors[document_position] += step
                        for context_id in context:
                            word_vectors[context_id] += step
                        loss_total += gradients.loss
                        n_pairs += 1
                    else:
                        gradients = distributed_bag_of_words_gradients(
                            document_vectors[document_position],
                            centre_id,
                            output_vectors,
                            sampler,
                            self.n_negative_samples,
                            generator,
                        )
                        np.add.at(
                            output_vectors,
                            gradients.output_ids,
                            -learning_rate * gradients.output_gradients,
                        )
                        document_vectors[document_position] -= (
                            learning_rate * gradients.hidden_gradient
                        )
                        loss_total += gradients.loss
                        n_pairs += 1

                        if self.train_words:
                            for context_id in context:
                                pair = negative_sampling_gradients(
                                    word_vectors[centre_id],
                                    context_id,
                                    output_vectors,
                                    sampler,
                                    self.n_negative_samples,
                                    generator,
                                )
                                np.add.at(
                                    output_vectors,
                                    pair.output_ids,
                                    -learning_rate * pair.output_gradients,
                                )
                                word_vectors[centre_id] -= (
                                    learning_rate * pair.hidden_gradient
                                )
                                loss_total += pair.loss
                                n_pairs += 1

            records.append(
                EpochRecord(
                    epoch,
                    loss_total / n_pairs if n_pairs else 0.0,
                    n_pairs,
                    learning_rate,
                )
            )

        frozen_outputs = output_vectors.copy()
        frozen_outputs.setflags(write=False)

        self._embeddings = WordEmbeddings(vocabulary, word_vectors)
        self._document_vectors = DocumentVectors(document_vectors)
        self._output_vectors = frozen_outputs
        self._history = TrainingHistory(records)
        self._sampler = sampler
        self._word_vectors_were_trained = (
            self.architecture is ParagraphArchitecture.DISTRIBUTED_MEMORY
            or self.train_words
        )
        self._mark_fitted()
        return self

    @property
    def embeddings(self) -> WordEmbeddings:
        """The word input vectors, one per word.

        Under plain distributed bag of words these are the random start,
        untouched by the fit; see :attr:`word_vectors_were_trained`.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._embeddings

    @property
    def document_vectors(self) -> DocumentVectors:
        """One vector per text of the fitted corpus, in corpus order.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._document_vectors

    @property
    def output_vectors(self) -> FloatArray:
        """The output table, frozen: one row per word, shared by both
        architectures and by the word pairs ``train_words`` adds.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._output_vectors

    @property
    def history(self) -> TrainingHistory:
        """Each epoch's mean loss over its pairs and closing learning rate.

        Under the bag of words with ``train_words`` the skip-gram pairs are
        counted and summed alongside the document pairs.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._history

    @property
    def n_documents(self) -> int:
        """How many texts the fit gave a vector to.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._document_vectors.n_documents

    @property
    def word_vectors_were_trained(self) -> bool:
        """Whether the fit moved the word vectors at all.

        True under distributed memory and under the bag of words with
        ``train_words``; false under the plain bag of words, whose
        :attr:`embeddings` are then the random initialisation.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._word_vectors_were_trained

    def document_similarity(self, first_position: int, second_position: int) -> float:
        """Cosine similarity between two fitted documents' vectors.

        See :meth:`DocumentVectors.similarity`.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return self.document_vectors.similarity(first_position, second_position)

    def most_similar_documents(
        self, position: int, n_results: int = 10
    ) -> tuple[int, ...]:
        """The positions of the fitted documents nearest to one, itself excluded.

        See :meth:`DocumentVectors.most_similar`.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return self.document_vectors.most_similar(position, n_results)

    def infer_vector(self, text: str, random_seed: int | None = None) -> FloatArray:
        """A vector for a text the fit never saw, by descent against frozen tables.

        The text is split by this model's pre-tokenizer, words outside the
        vocabulary are dropped, a fresh vector is drawn uniform in
        ``(-0.5, 0.5) / dimension`` from a generator seeded by ``random_seed``
        -- or by the model's own ``random_seed`` when none is given -- and
        ``inference_epochs`` passes of this architecture's update move that
        vector alone. The word and output tables are read and never written,
        so the fitted model is unchanged, and the same text under the same seed
        gives the same vector exactly. A text with no word in the vocabulary is
        handed its random start.

        Returns
        -------
        FloatArray
            ``(dimension,)``, frozen.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``text`` is not a string.
        """
        self._check_fitted()
        if not isinstance(text, str):
            raise InvalidValuesError(
                f"a text to infer a vector for must be a str, got {type(text).__name__}"
            )

        vocabulary = self._embeddings.vocabulary
        word_vectors = self._embeddings.table
        output_vectors = self._output_vectors
        sentence = tuple(
            vocabulary.id_of(word)
            for word in self.pre_tokenizer.split(text).texts
            if word in vocabulary
        )

        seed = self.random_seed if random_seed is None else random_seed
        generator = np.random.default_rng(seed)
        document_vector = self._initial_vectors(1, generator)[0]

        total_positions = len(sentence) * self.inference_epochs
        positions_seen = 0
        for _ in range(self.inference_epochs):
            for position, centre_id in enumerate(sentence):
                learning_rate = self._decayed_learning_rate(
                    positions_seen, total_positions
                )
                positions_seen += 1

                if self.architecture is ParagraphArchitecture.DISTRIBUTED_MEMORY:
                    context = self._context_of(sentence, position)
                    gradients = distributed_memory_gradients(
                        document_vector,
                        word_vectors[list(context)],
                        centre_id,
                        output_vectors,
                        self._sampler,
                        self.n_negative_samples,
                        generator,
                    )
                    document_vector += shared_input_step(
                        gradients.hidden_gradient, learning_rate, len(context) + 1
                    )
                else:
                    gradients = distributed_bag_of_words_gradients(
                        document_vector,
                        centre_id,
                        output_vectors,
                        self._sampler,
                        self.n_negative_samples,
                        generator,
                    )
                    document_vector -= learning_rate * gradients.hidden_gradient

        document_vector.setflags(write=False)
        return document_vector

    def _initial_vectors(
        self, n_rows: int, generator: np.random.Generator
    ) -> FloatArray:
        """``n_rows`` vectors uniform in ``(-0.5, 0.5) / dimension``.

        Word2vec's initialisation, used for the word table, the document table
        and each inferred vector alike: small, so nothing starts with a strong
        opinion, and spread, so rows start distinguishable.
        """
        return (generator.random((n_rows, self.dimension)) - 0.5) / self.dimension

    def _decayed_learning_rate(
        self, positions_seen: int, total_positions: int
    ) -> float:
        """The rate in force at position ``positions_seen`` of ``total_positions``."""
        return max(
            self.minimum_learning_rate,
            self.learning_rate * (1.0 - positions_seen / total_positions),
        )

    def _context_of(self, sentence: Sequence[int], position: int) -> tuple[int, ...]:
        """The ids within ``window`` of ``position``, the position itself excluded."""
        return (
            *sentence[max(0, position - self.window) : position],
            *sentence[position + 1 : position + 1 + self.window],
        )
