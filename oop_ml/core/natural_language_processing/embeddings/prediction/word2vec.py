"""Word2vec: a vector per word, learned by predicting neighbours from a window.

The idea, in one sentence
-------------------------
Give every word a vector, slide a window along the corpus, and nudge the
vectors so that a word's vector predicts the words that appear beside it. Words
that keep the same company get pushed to the same place, and that place is the
meaning the model can see. Mikolov et al. (2013) made this cheap enough to run
over billions of words, and the two decisions that did it are the two enums
below.

Two architectures
-----------------
:attr:`Word2VecArchitecture.SKIP_GRAM` reads the centre word and predicts each context
word in turn: one training pair per neighbour.
:attr:`Word2VecArchitecture.CONTINUOUS_BAG_OF_WORDS`
averages the context words' vectors and predicts the centre from the average:
one training pair per position. Skip-gram sees rare words more often, since a
rare centre still produces a pair for every neighbour, and is the usual choice
for small corpora; the bag of words is several times faster and smooths over
the context, which suits large ones.

The paper describes skip-gram as centre predicting context. The reference code
walks the same pairs the other way round, reading the *context* word's vector
to predict the centre; summed over a corpus the two visit every ordered pair
once each, so the objective is the same and only the order of updates differs.
This module follows the paper.

Two objectives
--------------
Predicting one word out of the whole vocabulary is a softmax whose gradient
touches every output row. :attr:`Word2VecObjective.NEGATIVE_SAMPLING` replaces it with
"is this the true neighbour or one of ``k`` random words", touching ``k + 1``
rows; :attr:`Word2VecObjective.HIERARCHICAL_SOFTMAX` replaces it with the binary
choices down a Huffman tree, touching about ``log2(n)`` rows. Both are the one
logistic step in
:mod:`~oop_ml.core.natural_language_processing.embeddings.prediction.objectives`;
what differs is which rows and which labels. The output table is one row per
word under negative sampling and one per internal node under the tree, and it
is exposed as ``output_vectors`` because under negative sampling those rows
are the "context vectors" that some downstream methods average with the input
ones.

What the loop does at each position
-----------------------------------
Rare words are gone before the loop starts, dropped by ``minimum_count`` in
:class:`~oop_ml.core.natural_language_processing.embeddings.embedder.TokenisedCorpus`.
Frequent words are thinned by ``subsampling_threshold``: a word with corpus
frequency ``f`` is kept with probability ``(sqrt(f / t) + 1) * t / f``, which
is above one, and so keeps every occurrence, until ``f`` passes about
``2.618 t`` (the formula crosses one where ``t / f`` is the golden ratio's
square, ``0.382``), and falls as the word gets commoner, so ``the`` is seen a
fraction of the time and the pairs it would have consumed go to words that
carry more. The window shrinks at random to a reach drawn from
``1 .. window`` at every position when ``shrink_windows`` is set, which is
the reference behaviour and weights near neighbours more without a weighting
formula. The learning rate falls linearly from ``learning_rate`` towards
``minimum_learning_rate`` over every position of every epoch.

What is recorded
----------------
Each epoch's mean loss over its training pairs, as a
:class:`TrainingHistory`. It is not the quantity anyone reports for a word2vec
model, since the loss depends on which negatives were drawn, but it is the
quantity the optimiser lowers, and a fit whose history does not fall is a fit
whose learning rate was wrong.

Two hooks for the models built on this one
------------------------------------------
FastText composes a word's input vector from the word's row and the rows of
its character n-grams, and spreads the gradient back across them. Everything
else about its training is this loop, so the loop reads and writes input
vectors through :meth:`Word2Vec._input_vector` and
:meth:`Word2Vec._apply_input_gradient`, allocates the table through
:meth:`Word2Vec._initial_input_vectors`, and composes the final word table
through :meth:`Word2Vec._word_table`. A subclass overrides those four and
inherits the rest.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from oop_ml.core.exceptions import InvalidValuesError, TooFewValuesError
from oop_ml.core.natural_language_processing.embeddings.embedder import WordEmbedder
from oop_ml.core.natural_language_processing.embeddings.prediction.huffman import (
    HuffmanTree,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.objectives import (
    PairGradients,
    hierarchical_softmax_gradients,
    negative_sampling_gradients,
)
from oop_ml.core.natural_language_processing.embeddings.prediction.sampling import (
    UnigramSampler,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray


class Word2VecArchitecture(StrEnum):
    """Which side of the window predicts which."""

    SKIP_GRAM = "skip_gram"
    """The centre word predicts each context word: one pair per neighbour."""

    CONTINUOUS_BAG_OF_WORDS = "continuous_bag_of_words"
    """The averaged context predicts the centre word: one pair per position."""


class Word2VecObjective(StrEnum):
    """How the full softmax over the vocabulary is avoided."""

    NEGATIVE_SAMPLING = "negative_sampling"
    """The true neighbour against ``k`` words drawn from the unigram table."""

    HIERARCHICAL_SOFTMAX = "hierarchical_softmax"
    """Binary choices down a Huffman tree, one output vector per internal node."""


class EpochRecord:
    """What one pass over the corpus cost, and where the learning rate stood.

    Parameters
    ----------
    epoch:
        Which pass, counting from one.
    mean_loss:
        The mean binary logistic loss over the epoch's training pairs.
    n_pairs:
        How many training pairs the epoch saw.
    learning_rate:
        The rate in force at the epoch's last position.

    Raises
    ------
    InvalidValuesError
        If the epoch is below one, a count is negative, or the loss or rate is
        negative or not finite.
    """

    __slots__ = ("_epoch", "_learning_rate", "_mean_loss", "_n_pairs")

    def __init__(
        self, epoch: int, mean_loss: float, n_pairs: int, learning_rate: float
    ) -> None:
        if epoch < 1:
            raise InvalidValuesError(f"epochs count from one, got {epoch}")
        if n_pairs < 0:
            raise InvalidValuesError(f"a pair count is non-negative, got {n_pairs}")
        if not np.isfinite(mean_loss) or mean_loss < 0.0:
            raise InvalidValuesError(
                f"a mean loss is finite and non-negative, got {mean_loss}"
            )
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise InvalidValuesError(
                f"a learning rate is finite and positive, got {learning_rate}"
            )

        self._epoch = int(epoch)
        self._mean_loss = float(mean_loss)
        self._n_pairs = int(n_pairs)
        self._learning_rate = float(learning_rate)

    @property
    def epoch(self) -> int:
        """Which pass, counting from one."""
        return self._epoch

    @property
    def mean_loss(self) -> float:
        """The mean loss over the epoch's pairs."""
        return self._mean_loss

    @property
    def n_pairs(self) -> int:
        """How many training pairs the epoch saw."""
        return self._n_pairs

    @property
    def learning_rate(self) -> float:
        """The rate in force at the epoch's last position."""
        return self._learning_rate

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EpochRecord):
            return NotImplemented
        return (
            self._epoch == other._epoch
            and self._mean_loss == other._mean_loss
            and self._n_pairs == other._n_pairs
            and self._learning_rate == other._learning_rate
        )

    def __hash__(self) -> int:
        return hash((self._epoch, self._mean_loss, self._n_pairs, self._learning_rate))

    def __repr__(self) -> str:
        return (
            f"EpochRecord(epoch={self._epoch}, mean_loss={self._mean_loss:.4f}, "
            f"n_pairs={self._n_pairs})"
        )


class TrainingHistory:
    """Every epoch's record, in order.

    Parameters
    ----------
    records:
        One per epoch, numbered ``1, 2, ...`` in sequence.

    Raises
    ------
    InvalidValuesError
        If the records are not numbered consecutively from one.
    """

    __slots__ = ("_records",)

    def __init__(self, records: Sequence[EpochRecord]) -> None:
        for position, record in enumerate(records):
            if record.epoch != position + 1:
                raise InvalidValuesError(
                    f"epoch records run 1, 2, ...; found epoch {record.epoch} at "
                    f"position {position}"
                )
        self._records = tuple(records)

    @property
    def n_epochs(self) -> int:
        """How many passes were made."""
        return len(self._records)

    @property
    def mean_losses(self) -> tuple[float, ...]:
        """Each epoch's mean loss, in order."""
        return tuple(record.mean_loss for record in self._records)

    @property
    def fell(self) -> bool:
        """Whether the last epoch's mean loss is below the first's.

        The cheapest sign that the optimiser was doing what it was asked; a
        history that did not fall usually means a learning rate too large.
        """
        return (
            len(self._records) > 1
            and self._records[-1].mean_loss < self._records[0].mean_loss
        )

    def __iter__(self) -> Iterator[EpochRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, position: int) -> EpochRecord:
        return self._records[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrainingHistory):
            return NotImplemented
        return self._records == other._records

    def __hash__(self) -> int:
        return hash(self._records)

    def __repr__(self) -> str:
        return f"TrainingHistory(n_epochs={self.n_epochs})"


class Word2Vec(WordEmbedder):
    """Skip-gram or continuous bag of words, by negative sampling or a Huffman tree.

    Parameters
    ----------
    dimension:
        How many numbers each word's vector holds.
    window:
        How many positions on each side of the centre count as context.
    architecture:
        Whether the centre predicts its context or the context its centre.
    objective:
        How the softmax over the vocabulary is avoided.
    n_negative_samples:
        How many random words each pair is trained against. Read only under
        negative sampling; the tree has no such number.
    negative_sampling_exponent:
        The flattening power applied to counts before drawing negatives.
    epochs:
        How many passes over the corpus.
    learning_rate:
        The step size at the start of training.
    minimum_learning_rate:
        The floor the rate decays towards, linearly over every position of
        every epoch. At most ``learning_rate``.
    subsampling_threshold:
        Word2vec's ``sample``: a word with corpus frequency above this is
        thinned as the module docstring describes. ``None`` keeps every word.
    shrink_windows:
        Whether the reach is redrawn from ``1 .. window`` at every position.
    random_seed:
        Seeds the initial vectors, the window draws, the negatives and the
        subsampling, so a seeded fit is reproducible.
    """

    dimension: int = Field(default=100, ge=1)
    window: int = Field(default=5, ge=1)
    architecture: Word2VecArchitecture = Word2VecArchitecture.SKIP_GRAM
    objective: Word2VecObjective = Word2VecObjective.NEGATIVE_SAMPLING
    n_negative_samples: int = Field(default=5, ge=1)
    negative_sampling_exponent: float = Field(default=0.75, gt=0.0, le=1.0)
    epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=0.025, gt=0.0)
    minimum_learning_rate: float = Field(default=0.0001, gt=0.0)
    subsampling_threshold: float | None = Field(default=None, gt=0.0)
    shrink_windows: bool = True
    random_seed: int | None = None

    _embeddings: WordEmbeddings = PrivateAttr()
    _output_vectors: FloatArray = PrivateAttr()
    _history: TrainingHistory = PrivateAttr()

    @model_validator(mode="after")
    def _check_the_floor_is_below_the_start(self) -> Self:
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError(
                f"minimum_learning_rate={self.minimum_learning_rate} is above "
                f"learning_rate={self.learning_rate}; the rate decays downward"
            )
        return self

    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn a vector per word by sliding a window over ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        TooFewValuesError
            If fewer than two words reach ``minimum_count``, since one word
            has no neighbour to predict and no negative to draw.
        """
        tokenised = self._tokenised(corpus)
        vocabulary = tokenised.vocabulary(self.minimum_count)
        if vocabulary.n_tokens < 2:
            raise TooFewValuesError(
                "word2vec needs at least two words in the vocabulary: one word "
                "has no neighbour to predict and no negative to draw"
            )

        id_sequences = tokenised.id_sequences(vocabulary)
        counts = [tokenised.word_counts[word] for word in vocabulary]
        generator = np.random.default_rng(self.random_seed)

        input_vectors = self._initial_input_vectors(vocabulary, generator)
        sampler: UnigramSampler | None = None
        tree: HuffmanTree | None = None
        if self.objective is Word2VecObjective.NEGATIVE_SAMPLING:
            sampler = UnigramSampler(counts, self.negative_sampling_exponent)
            n_outputs = vocabulary.n_tokens
        else:
            tree = HuffmanTree.from_counts(counts)
            n_outputs = tree.n_internal_nodes
        output_vectors = np.zeros((n_outputs, self.dimension))

        keep_probabilities = self._keep_probabilities(counts)
        total_positions = sum(len(sentence) for sentence in id_sequences) * self.epochs
        positions_seen = 0
        learning_rate = self.learning_rate
        records: list[EpochRecord] = []

        for epoch in range(1, self.epochs + 1):
            loss_total = 0.0
            n_pairs = 0
            for sentence in id_sequences:
                kept = self._subsampled(sentence, keep_probabilities, generator)
                for position, center in enumerate(kept):
                    learning_rate = max(
                        self.minimum_learning_rate,
                        self.learning_rate * (1.0 - positions_seen / total_positions),
                    )
                    positions_seen += 1

                    reach = (
                        int(generator.integers(1, self.window + 1))
                        if self.shrink_windows
                        else self.window
                    )
                    context = (
                        kept[max(0, position - reach) : position]
                        + kept[position + 1 : position + 1 + reach]
                    )
                    if not context:
                        continue

                    if self.architecture is Word2VecArchitecture.SKIP_GRAM:
                        for target in context:
                            hidden = self._input_vector(input_vectors, center)
                            gradients = self._pair_gradients(
                                hidden, target, output_vectors, sampler, tree, generator
                            )
                            np.add.at(
                                output_vectors,
                                gradients.output_ids,
                                -learning_rate * gradients.output_gradients,
                            )
                            self._apply_input_gradient(
                                input_vectors,
                                center,
                                -learning_rate * gradients.hidden_gradient,
                            )
                            loss_total += gradients.loss
                            n_pairs += 1
                    else:
                        hidden = np.mean(
                            [
                                self._input_vector(input_vectors, word)
                                for word in context
                            ],
                            axis=0,
                        )
                        gradients = self._pair_gradients(
                            hidden, center, output_vectors, sampler, tree, generator
                        )
                        np.add.at(
                            output_vectors,
                            gradients.output_ids,
                            -learning_rate * gradients.output_gradients,
                        )
                        shared = (
                            -learning_rate * gradients.hidden_gradient / len(context)
                        )
                        for word in context:
                            self._apply_input_gradient(input_vectors, word, shared)
                        loss_total += gradients.loss
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

        self._embeddings = WordEmbeddings(
            vocabulary, self._word_table(input_vectors, vocabulary)
        )
        self._output_vectors = frozen_outputs
        self._history = TrainingHistory(records)
        self._mark_fitted()
        return self

    @property
    def embeddings(self) -> WordEmbeddings:
        """The input vectors, one per word, which are the word vectors.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._embeddings

    @property
    def output_vectors(self) -> FloatArray:
        """The output table, frozen: one row per word under negative sampling
        (the context vectors), one per internal node under the tree.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._output_vectors

    @property
    def history(self) -> TrainingHistory:
        """Each epoch's mean loss and closing learning rate.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._history

    def _initial_input_vectors(
        self, vocabulary: Vocabulary, generator: np.random.Generator
    ) -> FloatArray:
        """One row per word, uniform in ``[-0.5, 0.5) / dimension``.

        The reference initialisation: small, so no word starts with a strong
        opinion, and spread, so words start distinguishable. The output table
        starts at zero, as there, which is why the input table is the one that
        must break the symmetry. A subclass with more rows to learn, such as a
        table of character n-grams, overrides this to allocate them.
        """
        return (
            generator.random((vocabulary.n_tokens, self.dimension)) - 0.5
        ) / self.dimension

    def _input_vector(self, input_vectors: FloatArray, word_id: int) -> FloatArray:
        """The vector the model reads for ``word_id``: its row, here."""
        return input_vectors[word_id]

    def _apply_input_gradient(
        self, input_vectors: FloatArray, word_id: int, step: FloatArray
    ) -> None:
        """Add ``step`` to what :meth:`_input_vector` reads for ``word_id``."""
        input_vectors[word_id] += step

    def _word_table(
        self, input_vectors: FloatArray, vocabulary: Vocabulary
    ) -> FloatArray:
        """The final ``(n_words, dimension)`` table: the input rows themselves."""
        return input_vectors[: vocabulary.n_tokens]

    def _pair_gradients(
        self,
        hidden: FloatArray,
        target_id: int,
        output_vectors: FloatArray,
        sampler: UnigramSampler | None,
        tree: HuffmanTree | None,
        generator: np.random.Generator,
    ) -> PairGradients:
        if sampler is not None:
            return negative_sampling_gradients(
                hidden,
                target_id,
                output_vectors,
                sampler,
                self.n_negative_samples,
                generator,
            )
        if tree is None:
            raise InvalidValuesError("an objective needs a sampler or a tree")
        return hierarchical_softmax_gradients(hidden, target_id, output_vectors, tree)

    def _keep_probabilities(self, counts: Sequence[int]) -> FloatArray | None:
        """Per word, the probability of keeping an occurrence, or ``None``.

        Word2vec's rule: ``(sqrt(f / t) + 1) * t / f`` for frequency ``f`` and
        threshold ``t``, clipped to one.
        """
        if self.subsampling_threshold is None:
            return None
        frequencies = np.asarray(counts, dtype=np.float64) / float(sum(counts))
        ratio = self.subsampling_threshold / frequencies
        return np.minimum(1.0, (np.sqrt(1.0 / ratio) + 1.0) * ratio)

    @staticmethod
    def _subsampled(
        sentence: Sequence[int],
        keep_probabilities: FloatArray | None,
        generator: np.random.Generator,
    ) -> tuple[int, ...]:
        """The sentence with frequent words thinned at random, or as it was."""
        if keep_probabilities is None:
            return tuple(sentence)
        draws = generator.random(len(sentence))
        return tuple(
            word_id
            for word_id, draw in zip(sentence, draws, strict=True)
            if draw < keep_probabilities[word_id]
        )
