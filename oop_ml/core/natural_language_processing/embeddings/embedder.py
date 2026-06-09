"""What every word embedder shares: a tokenised corpus in, a table of vectors out.

The boundary, again
-------------------
A tokenizer's ``fit`` takes texts and a rule for where the words are. An
embedder's ``fit`` takes the same two things, and what it needs from them is
the *sequence* of words in each text -- byte pair encoding only ever needed the
counts, but a co-occurrence window and a skip-gram both read words in order.
:class:`TokenisedCorpus` is that sequence, validated once: each text split by
the pre-tokenizer, the words counted, and two questions answered for every
embedder at once. Which words are in the vocabulary, given a minimum count, and
in what order. And what each text looks like as ids once the rare words are
dropped.

Why rare words are dropped before windowing
-------------------------------------------
Every embedder here takes a ``minimum_count``, and a word below it is not given
a vector: a word seen once has one context, and a vector learned from one
context is noise wearing a direction. Word2vec removes those words from the
sentence *before* forming windows, so the words on either side of a removed one
become neighbours. That is the behaviour of :meth:`TokenisedCorpus.id_sequences`,
because it is what the reference implementations do and because the
alternative -- leaving a gap -- would make the window's reach depend on how
many rare words happened to fall inside it.

Why the vocabulary is ordered by frequency
------------------------------------------
Ids run from the commonest word down. Nothing about the geometry needs that,
but two of the techniques do: the unigram table negative sampling draws from
and the Huffman tree hierarchical softmax walks are both built from the counts
in id order, and a reader comparing a vocabulary here with one from the
reference implementations will find the same word at the same position. Ties
break alphabetically, so the order is a function of the corpus alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Self

from pydantic import Field

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    TooFewValuesError,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    SimilarWords,
    WordEmbeddings,
    WordVector,
)
from oop_ml.core.natural_language_processing.tokenization.corpus import (
    Corpus,
    WordCount,
    WordCounts,
)
from oop_ml.core.natural_language_processing.tokenization.tokenizer import (
    PreTokenizer,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (  # noqa: E501
    WhitespacePreTokenizer,
)


class TokenisedCorpus:
    """Every text of a corpus as its sequence of words, with the counts.

    Parameters
    ----------
    sentences:
        One sequence of words per text, in order. A text with no words is kept
        as an empty sequence, so positions still line up with the corpus; at
        least one word must appear somewhere.

    Raises
    ------
    InvalidValuesError
        If ``sentences`` is a single string, or a word is not a non-empty
        string.
    EmptyValuesError
        If there are no sentences, or no words in any of them.
    """

    __slots__ = ("_sentences", "_word_counts")

    def __init__(self, sentences: Sequence[Sequence[str]]) -> None:
        if isinstance(sentences, str):
            raise InvalidValuesError(
                "a tokenised corpus is a sequence of word sequences, not one string"
            )

        if len(sentences) == 0:
            raise EmptyValuesError("a tokenised corpus needs at least one sentence")

        checked: list[tuple[str, ...]] = []
        counter: Counter[str] = Counter()
        for position, sentence in enumerate(sentences):
            if isinstance(sentence, str):
                raise InvalidValuesError(
                    f"sentence {position} is a single string; a sentence is a "
                    f"sequence of words"
                )
            for word in sentence:
                if not isinstance(word, str) or not word:
                    raise InvalidValuesError(
                        f"every word must be a non-empty string, got {word!r} in "
                        f"sentence {position}"
                    )
            checked.append(tuple(sentence))
            counter.update(sentence)

        if not counter:
            raise EmptyValuesError("no sentence holds any word")

        self._sentences = tuple(checked)
        self._word_counts = WordCounts(
            [WordCount(word, count) for word, count in counter.items()]
        )

    @classmethod
    def from_texts(
        cls, corpus: Corpus | Sequence[str], pre_tokenizer: PreTokenizer
    ) -> TokenisedCorpus:
        """Split every text of ``corpus`` with ``pre_tokenizer``.

        Raises
        ------
        InvalidValuesError
            If the corpus is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty or blank, or the pre-tokenizer finds no
            words in any text.
        """
        return cls([pre_tokenizer.split(text).texts for text in Corpus.of(corpus)])

    @property
    def sentences(self) -> tuple[tuple[str, ...], ...]:
        """Every text as its words, in corpus order."""
        return self._sentences

    @property
    def word_counts(self) -> WordCounts:
        """How often each distinct word appears across the corpus."""
        return self._word_counts

    @property
    def n_sentences(self) -> int:
        """How many texts there are."""
        return len(self._sentences)

    @property
    def n_words(self) -> int:
        """How many word occurrences there are, distinct or not."""
        return self._word_counts.total

    def vocabulary(self, minimum_count: int = 1) -> Vocabulary:
        """The words seen at least ``minimum_count`` times, commonest first.

        Ties in count break alphabetically, so the order depends on the corpus
        and on nothing else. No unknown token: a word below the minimum is
        simply not in the vocabulary, and asking for its vector raises.

        Raises
        ------
        TooFewValuesError
            If no word reaches the minimum.
        """
        surviving = [
            word_count
            for word_count in self._word_counts
            if word_count.count >= minimum_count
        ]
        if not surviving:
            raise TooFewValuesError(
                f"no word appears {minimum_count} times or more; the commonest "
                f"appears {max(word_count.count for word_count in self._word_counts)}"
            )

        surviving.sort(key=lambda word_count: (-word_count.count, word_count.word))
        return Vocabulary([word_count.word for word_count in surviving])

    def id_sequences(self, vocabulary: Vocabulary) -> tuple[tuple[int, ...], ...]:
        """Every sentence as ids, with words outside ``vocabulary`` dropped.

        Dropped rather than replaced, so the words on either side of a rare
        word become neighbours, as in the reference implementations.
        """
        return tuple(
            tuple(vocabulary.id_of(word) for word in sentence if word in vocabulary)
            for sentence in self._sentences
        )

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        return iter(self._sentences)

    def __len__(self) -> int:
        return len(self._sentences)

    def __repr__(self) -> str:
        return (
            f"TokenisedCorpus(n_sentences={self.n_sentences}, n_words={self.n_words})"
        )


class CorpusEmbedder(Fittable, ABC):
    """What every embedder learns from: texts, a rule for the words, a floor.

    The shared frame under :class:`WordEmbedder` and the document embedders in
    :mod:`~oop_ml.core.natural_language_processing.embeddings.documents`. The
    two answer different things -- a vector per word, a vector per text -- so
    neither is a special case of the other, but both are configured the same
    way and both begin by tokenising the corpus once, and a frame written
    twice is a frame that drifts.

    Parameters
    ----------
    pre_tokenizer:
        Decides where the words are in each text.
    minimum_count:
        A word seen fewer times than this across the corpus is not given a
        place in the vocabulary. A subclass handed its vocabulary from
        elsewhere may refuse any value but the default.
    """

    pre_tokenizer: PreTokenizer = Field(default_factory=WhitespacePreTokenizer)
    minimum_count: int = Field(default=1, ge=1)

    @abstractmethod
    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn from ``corpus`` and return ``self``.

        Implementations should tokenise through :meth:`_tokenised`, compute
        into locals, assign the private attributes at the end, then call
        ``self._mark_fitted()``.
        """

    def _tokenised(self, corpus: Sequence[str]) -> TokenisedCorpus:
        """The corpus split by this embedder's pre-tokenizer, validated once."""
        return TokenisedCorpus.from_texts(corpus, self.pre_tokenizer)


class WordEmbedder(CorpusEmbedder):
    """Learns one vector per word from a corpus of texts.

    Construction configures the rule for where the words are and how rare a
    word may be; ``fit`` learns the table; every question about a word is
    answered by the
    :class:`~oop_ml.core.natural_language_processing.embeddings.vectors.WordEmbeddings`
    the fit produced, reached through :attr:`embeddings` or the conveniences
    below. A word below ``minimum_count`` gets no vector and is dropped from
    every sentence before any window is formed.
    """

    @abstractmethod
    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn a vector for every word of ``corpus`` and return ``self``.

        Implementations should tokenise through :meth:`_tokenised`, compute
        into locals, assign the private attributes at the end, then call
        ``self._mark_fitted()``.
        """

    @property
    @abstractmethod
    def embeddings(self) -> WordEmbeddings:
        """The vocabulary and its table, once fitted.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """

    @property
    def vocabulary(self) -> Vocabulary:
        """The words that have a vector, in id order. See :attr:`embeddings`."""
        return self.embeddings.vocabulary

    def vector_of(self, word: str) -> WordVector:
        """The learned vector for ``word``. See :meth:`WordEmbeddings.vector_of`."""
        return self.embeddings.vector_of(word)

    def similarity(self, first_word: str, second_word: str) -> float:
        """Cosine similarity between two words.

        See :meth:`WordEmbeddings.similarity`.
        """
        return self.embeddings.similarity(first_word, second_word)

    def most_similar(self, word: str, n_results: int = 10) -> SimilarWords:
        """The nearest words to ``word``. See :meth:`WordEmbeddings.most_similar`."""
        return self.embeddings.most_similar(word, n_results)
